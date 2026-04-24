"""
Object Detection Quantization Benchmark
========================================
Tests quantization (via LoFloat / sensitivity_search) on multiple SOTA
edge-oriented detection networks, evaluated on COCO val2017 and/or Pascal VOC 2012 val.

Supported models:
  - yolov8n          (Ultralytics)
  - yolo26n          (Ultralytics)
  - rtdetr-n         (Ultralytics, RT-DETR small)
  - mobilenet_ssd_v2 (torchvision SSDLite320 + MobileNetV3-Large)
  - fasterrcnn_mobilenet (torchvision Faster R-CNN + MobileNetV3-Large-FPN, optional)
  - deimv2_dinov3_s  (DEIMv2-S with DINOv3 ViT-S backbone, ~9.7M params, 50.9 AP on COCO)

Supported datasets:
  - coco   (val2017, ~5 k images, 80 classes)
  - pascal  (VOC2012 val, ~5.8 k images, 20 classes)

Usage:
  python obj_det_quant_bench.py --models yolov8n yolo26n mobilenet_ssd_v2 rtdetr-n deimv2_dinov3_s \
                                --datasets coco pascal \
                                --device cuda --batch-size 16
"""

import argparse
import json
import os
import sys
import time
import urllib.request
import zipfile
import tarfile
from collections import defaultdict
from pathlib import Path
import io, contextlib

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import yaml

# ---------------------------------------------------------------------------
# We import the user's custom quantization libs; adjust paths as needed.
# ---------------------------------------------------------------------------
import logging
logging.getLogger("ultralytics").setLevel(logging.WARNING)
sys.path.append("..")
try:
    import LoFloat as lof
    HAS_LOFLOAT = True
except ImportError:
    HAS_LOFLOAT = False
    print("[WARN] LoFloat / sensitivity_search not found — "
          "quantization steps will be skipped, only FP baselines will run.")

import sensitivity_search as ss

# ===================================================================
#  GLOBAL CONFIG (overridden by CLI)
# ===================================================================
IMG_SIZE   = 640
BATCH_SIZE = 16
WORKERS    = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

COCO_ROOT   = "./coco"
PASCAL_ROOT = "."

SEARCH_SUBSET_SIZE = 100  # Number of images for eval_fn during greedy search

ACCURACY_TARGETS = [0.1, 0.05]  # Run quantization for each of these targets


def _update_config(batch_size, img_size, workers, device, search_subset_size, accuracy_targets):
    global BATCH_SIZE, IMG_SIZE, WORKERS, DEVICE, SEARCH_SUBSET_SIZE, ACCURACY_TARGETS
    BATCH_SIZE = batch_size
    IMG_SIZE   = img_size
    WORKERS    = workers
    DEVICE     = device
    SEARCH_SUBSET_SIZE = search_subset_size
    ACCURACY_TARGETS = accuracy_targets

# ===================================================================
#  DOWNLOAD / DATASET HELPERS
# ===================================================================
def _download(url: str, dest_dir: str):
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.join(dest_dir, url.split("/")[-1])
    if os.path.exists(filename):
        print(f"  Already downloaded: {filename}")
        return filename
    print(f"  Downloading {url} ...")
    def _hook(count, block, total):
        pct = count * block * 100 // total if total > 0 else 0
        print(f"\r  Progress: {pct}%", end="", flush=True)
    urllib.request.urlretrieve(url, filename, reporthook=_hook)
    print()
    return filename


def _extract_zip(path, dest):
    print(f"  Extracting {path} ...")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(dest)


def _extract_tar(path, dest):
    print(f"  Extracting {path} ...")
    with tarfile.open(path, "r:*") as tf:
        tf.extractall(dest)


# ----------------------------- COCO -----------------------------------
COCO_URLS = {
    "val2017":     "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}


def convert_coco_to_yolo(annot_json, output_label_dir, images_dir):
    """COCO JSON  ->  per-image YOLO .txt files."""
    os.makedirs(output_label_dir, exist_ok=True)
    with open(annot_json) as f:
        coco = json.load(f)

    image_info = {img["id"]: img for img in coco["images"]}
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)
    cat_id_to_idx = {cat["id"]: idx for idx, cat in enumerate(coco["categories"])}

    for image_id, img in image_info.items():
        w, h = img["width"], img["height"]
        stem = os.path.splitext(img["file_name"])[0]
        label_path = os.path.join(output_label_dir, stem + ".txt")
        lines = []
        for ann in anns_by_image.get(image_id, []):
            if ann.get("iscrowd", 0):
                continue
            cls = cat_id_to_idx[ann["category_id"]]
            x, y, bw, bh = ann["bbox"]
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f}")
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
    print(f"  COCO labels written to: {output_label_dir}")


COCO_CLASS_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck",
    "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
    "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
    "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
    "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed","dining table","toilet",
    "tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven",
    "toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush",
]


def prepare_coco():
    images_dir = os.path.join(COCO_ROOT, "images")
    annot_dir  = os.path.join(COCO_ROOT, "annotations", "annotations", "annotations")

    for name, url in COCO_URLS.items():
        target_dir = annot_dir if name == "annotations" else images_dir
        marker = os.path.join(target_dir, name if name != "annotations"
                              else "instances_val2017.json")
        if os.path.exists(marker):
            print(f"  [{name}] already present, skipping download.")
        else:
            f = _download(url, target_dir)
            _extract_zip(f, target_dir)

    convert_coco_to_yolo(
        annot_json=os.path.join(annot_dir, "instances_val2017.json"),
        output_label_dir=os.path.join(COCO_ROOT, "labels", "val2017"),
        images_dir=os.path.join(images_dir, "val2017"),
    )

    yaml_path = os.path.join(COCO_ROOT, "coco_val.yaml")
    cfg = dict(
        path=os.path.abspath(COCO_ROOT),
        val="images/val2017",
        train="images/val2017",
        nc=80,
        names=COCO_CLASS_NAMES,
    )
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  Written COCO yaml -> {yaml_path}")
    return yaml_path


# ----------------------------- PASCAL VOC ----------------------------
PASCAL_URLS = [
    "https://pjreddie.com/media/files/VOCtrainval_11-May-2012.tar",
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar",
]

PASCAL_CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
]


def _convert_voc_to_yolo(voc_root):
    import xml.etree.ElementTree as ET
    img_dir   = os.path.join(voc_root, "JPEGImages")
    annot_dir = os.path.join(voc_root, "Annotations/annotations")
    label_dir = os.path.join(voc_root, "labels", "val")
    os.makedirs(label_dir, exist_ok=True)
    cls_to_idx = {c: i for i, c in enumerate(PASCAL_CLASS_NAMES)}
    val_txt = os.path.join(voc_root, "ImageSets", "Main", "val.txt")
    with open(val_txt) as f:
        ids = [line.strip() for line in f if line.strip()]
    for img_id in ids:
        xml_path = os.path.join(annot_dir, img_id + ".xml")
        if not os.path.exists(xml_path):
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        w = int(size.find("width").text)
        h = int(size.find("height").text)
        lines = []
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in cls_to_idx:
                continue
            difficult = obj.find("difficult")
            if difficult is not None and int(difficult.text) == 1:
                continue
            cls = cls_to_idx[cls_name]
            bbox = obj.find("bndbox")
            x1 = float(bbox.find("xmin").text)
            y1 = float(bbox.find("ymin").text)
            x2 = float(bbox.find("xmax").text)
            y2 = float(bbox.find("ymax").text)
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        with open(os.path.join(label_dir, img_id + ".txt"), "w") as f:
            f.write("\n".join(lines))
    print(f"  VOC labels written to: {label_dir}")
    img_val_dir = os.path.join(voc_root, "images", "val")
    os.makedirs(img_val_dir, exist_ok=True)
    for img_id in ids:
        src = os.path.join(img_dir, img_id + ".jpg")
        dst = os.path.join(img_val_dir, img_id + ".jpg")
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(os.path.abspath(src), dst)
    print(f"  VOC image symlinks created in: {img_val_dir}")


def prepare_pascal():
    voc_root = os.path.join(PASCAL_ROOT, "VOCdevkit", "VOC2012")
    if not os.path.exists(voc_root):
        downloaded = False
        for url in PASCAL_URLS:
            try:
                print(f"  Trying: {url}")
                f = _download(url, ".")
                _extract_tar(f, ".")
                downloaded = True
                break
            except Exception as e:
                print(f"  Failed ({e}), trying next mirror...")
        if not downloaded:
            raise RuntimeError("Could not download Pascal VOC 2012 from any mirror.")
    else:
        print("  [pascal] VOC2012 already present.")
    _convert_voc_to_yolo(voc_root)
    val_txt = os.path.join(voc_root, "ImageSets", "Main", "val.txt")
    img_list = os.path.join(voc_root, "val_images.txt")
    with open(val_txt) as f:
        ids = [l.strip() for l in f if l.strip()]
    with open(img_list, "w") as f:
        for i in ids:
            f.write(os.path.abspath(os.path.join(voc_root, "JPEGImages", i + ".jpg")) + "\n")
    yaml_path = os.path.join(PASCAL_ROOT, "voc2012_val.yaml")
    cfg = dict(path=os.path.abspath(voc_root), val="images/val", train="images/val",
               nc=20, names=PASCAL_CLASS_NAMES)
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  Written Pascal yaml -> {yaml_path}")
    return yaml_path


# ===================================================================
#  CALIBRATION DATASET
# ===================================================================
from ultralytics.data.dataset import YOLODataset

class CalibDataset(Dataset):
    def __init__(self, img_path, data_yaml, imgsz=640):
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
        self.dataset = YOLODataset(img_path=img_path, data=data, imgsz=imgsz,
                                   augment=False, rect=False, cache=False)
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        img = sample["img"]
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img)
        img = img.float() / 255.0
        if img.ndim == 3 and img.shape[-1] in (1, 3):
            img = img.permute(2, 0, 1)
        cls = sample.get("cls", torch.empty(0))
        if not isinstance(cls, torch.Tensor):
            cls = torch.tensor(cls)
        cls = cls.view(-1).long()
        max_boxes = 100
        label = torch.zeros(max_boxes, dtype=torch.long)
        n = min(len(cls), max_boxes)
        label[:n] = cls[:n]
        return img, label


def cosine_distance_chunked(a, b, chunk=1 << 20, eps=1e-8):
    """1 - cos_sim(a, b) without allocating full-size temporaries.

    Autograd-friendly: gradients flow through `a` and `b` if they require grad,
    which is required when this is used as a loss_fn inside sensitivity search
    (greedy_sensitivity + hessian measure needs ∇loss w.r.t. model params).
    For grad-free usage (e.g. `.item()` prints), wrap the call in torch.no_grad().

    Float64 accumulators avoid precision loss when summing millions of products.
    """
    a = a.reshape(-1)
    b = b.reshape(-1)
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]

    dot = a.new_zeros((), dtype=torch.float64)
    aa  = a.new_zeros((), dtype=torch.float64)
    bb  = a.new_zeros((), dtype=torch.float64)

    for i in range(0, n, chunk):
        ac = a[i:i+chunk].double()
        bc = b[i:i+chunk].double()
        dot = dot + (ac * bc).sum()
        aa  = aa  + (ac * ac).sum()
        bb  = bb  + (bc * bc).sum()

    denom = (aa.sqrt() * bb.sqrt()).clamp_min(eps)
    out_dtype = a.dtype if a.is_floating_point() else torch.float32
    return (1.0 - dot / denom).to(out_dtype)


# ===================================================================
#  TORCHVISION EVAL HELPERS
# ===================================================================
def _get_coco_gt(annot_json):
    from pycocotools.coco import COCO as COCOGT
    return COCOGT(annot_json)

def _build_torchvision_dataloader(dataset_name, batch_size):
    import torchvision.transforms as T
    from torchvision.datasets import CocoDetection, VOCDetection
    class RGBToTensor:
        def __call__(self, img):
            img = img.convert("RGB")
            return T.functional.to_tensor(img)
    transform = RGBToTensor()
    if dataset_name == "coco":
        img_dir = os.path.join(COCO_ROOT, "images", "val2017")
        ann_file = os.path.join(COCO_ROOT, "annotations", "annotations", "annotations", "instances_val2017.json")
        ds = CocoDetection(img_dir, ann_file, transform=transform)
        coco_gt = _get_coco_gt(ann_file)
        info = dict(name="coco", nc=80)
    else:
        ds = VOCDetection(root=PASCAL_ROOT, year="2012", image_set="val",
                          download=False, transform=transform)
        coco_gt = None
        info = dict(name="pascal", nc=20)
    def collate_fn(batch):
        return list(zip(*batch))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=WORKERS, collate_fn=collate_fn)
    return loader, coco_gt, info

def eval_torchvision_coco(model, dataloader, coco_gt, device):
    from pycocotools.cocoeval import COCOeval
    model.eval()
    model.to(device)
    dataset = dataloader.dataset
    if isinstance(dataset, Subset):
        all_img_ids = [dataset.dataset.ids[i] for i in dataset.indices]
    else:
        all_img_ids = dataset.ids
    results = []
    n_images = len(dataset)
    print(f"  Evaluating torchvision model on {n_images} COCO val images ...")
    t0 = time.time()
    sample_idx = 0
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        with torch.no_grad():
            outputs = model(images)
        for i, out in enumerate(outputs):
            image_id = all_img_ids[sample_idx + i]
            boxes = out["boxes"].cpu()
            scores = out["scores"].cpu()
            labels = out["labels"].cpu()
            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b.tolist()
                results.append({"image_id": image_id, "category_id": l.item(),
                                "bbox": [x1, y1, x2-x1, y2-y1], "score": s.item()})
        sample_idx += len(images)
    elapsed = time.time() - t0
    if len(results) == 0:
        print("  [WARN] No detections produced!")
        return {"map50": 0.0, "map": 0.0}
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = list(set(r["image_id"] for r in results))
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    return {"map": coco_eval.stats[0], "map50": coco_eval.stats[1]}


# ===================================================================
#  PASCAL VOC -> COCO-FORMAT EVALUATION HELPERS
# ===================================================================
VOC_CLASS_TO_CATID = {c: i + 1 for i, c in enumerate(PASCAL_CLASS_NAMES)}

COCO_CONTIGUOUS_TO_VOC = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorbike", 4: "aeroplane",
    5: "bus", 6: "train", 8: "boat", 14: "bird", 15: "cat",
    16: "dog", 17: "horse", 18: "sheep", 19: "cow", 39: "bottle",
    56: "chair", 57: "sofa", 58: "pottedplant",
    60: "diningtable", 62: "tvmonitor",
}

COCO_CATID_TO_VOC_CATID = {
    1: VOC_CLASS_TO_CATID["person"], 2: VOC_CLASS_TO_CATID["bicycle"],
    3: VOC_CLASS_TO_CATID["car"], 4: VOC_CLASS_TO_CATID["motorbike"],
    5: VOC_CLASS_TO_CATID["aeroplane"], 6: VOC_CLASS_TO_CATID["bus"],
    7: VOC_CLASS_TO_CATID["train"], 9: VOC_CLASS_TO_CATID["boat"],
    16: VOC_CLASS_TO_CATID["bird"], 17: VOC_CLASS_TO_CATID["cat"],
    18: VOC_CLASS_TO_CATID["dog"], 19: VOC_CLASS_TO_CATID["horse"],
    20: VOC_CLASS_TO_CATID["sheep"], 21: VOC_CLASS_TO_CATID["cow"],
    44: VOC_CLASS_TO_CATID["bottle"], 62: VOC_CLASS_TO_CATID["chair"],
    63: VOC_CLASS_TO_CATID["sofa"], 64: VOC_CLASS_TO_CATID["pottedplant"],
    67: VOC_CLASS_TO_CATID["diningtable"], 70: VOC_CLASS_TO_CATID["tvmonitor"],
}

COCO_CONTIGUOUS_TO_VOC_CATID = {
    k: VOC_CLASS_TO_CATID[v] for k, v in COCO_CONTIGUOUS_TO_VOC.items()
}


def _build_pascal_coco_gt(eval_indices=None):
    import xml.etree.ElementTree as ET
    from pycocotools.coco import COCO as COCOGT
    voc_root = os.path.join(PASCAL_ROOT, "VOCdevkit", "VOC2012")
    val_txt = os.path.join(voc_root, "ImageSets", "Main", "val.txt")
    annot_dir = os.path.join(voc_root, "Annotations")
    with open(val_txt) as f:
        all_ids = [line.strip() for line in f if line.strip()]
    selected_ids = [all_ids[i] for i in eval_indices] if eval_indices is not None else all_ids
    images, annotations = [], []
    ann_id = 1
    for img_idx, img_id in enumerate(selected_ids):
        image_id = img_idx + 1
        xml_path = os.path.join(annot_dir, img_id + ".xml")
        if not os.path.exists(xml_path):
            continue
        tree = ET.parse(xml_path)
        root = tree.getroot()
        size = root.find("size")
        w, h = int(size.find("width").text), int(size.find("height").text)
        images.append({"id": image_id, "file_name": img_id + ".jpg", "width": w, "height": h})
        for obj in root.findall("object"):
            cls_name = obj.find("name").text
            if cls_name not in VOC_CLASS_TO_CATID:
                continue
            difficult = obj.find("difficult")
            is_difficult = (difficult is not None and int(difficult.text) == 1)
            bb = obj.find("bndbox")
            x1, y1 = float(bb.find("xmin").text), float(bb.find("ymin").text)
            x2, y2 = float(bb.find("xmax").text), float(bb.find("ymax").text)
            bw, bh = x2 - x1, y2 - y1
            annotations.append({"id": ann_id, "image_id": image_id,
                                "category_id": VOC_CLASS_TO_CATID[cls_name],
                                "bbox": [x1, y1, bw, bh], "area": bw * bh,
                                "iscrowd": 0, "ignore": int(is_difficult)})
            ann_id += 1
    categories = [{"id": i+1, "name": c} for i, c in enumerate(PASCAL_CLASS_NAMES)]
    coco_dict = {"images": images, "annotations": annotations, "categories": categories}
    json_path = os.path.join(PASCAL_ROOT, "_pascal_coco_gt_temp.json")
    with open(json_path, "w") as f:
        json.dump(coco_dict, f)
    coco_gt = COCOGT(json_path)
    return coco_gt, [img["id"] for img in images]

def _run_cocoeval_pascal(results, coco_gt, image_ids=None):
    from pycocotools.cocoeval import COCOeval
    if len(results) == 0:
        print("  [WARN] No detections produced!")
        return {"map50": 0.0, "map": 0.0}
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    if image_ids is not None:
        coco_eval.params.imgIds = image_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    return {"map": coco_eval.stats[0], "map50": coco_eval.stats[1]}

def eval_torchvision_pascal(model, device, max_images=None, seed=42):
    import torchvision.transforms as T
    from torchvision.datasets import VOCDetection
    import random as _random
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB")), T.ToTensor()])
    voc_ds = VOCDetection(root=PASCAL_ROOT, year="2012", image_set="val",
                          download=False, transform=transform)
    if max_images is not None and max_images < len(voc_ds):
        _random.seed(seed)
        eval_indices = _random.sample(range(len(voc_ds)), max_images)
    else:
        eval_indices = list(range(len(voc_ds)))
    coco_gt, image_ids = _build_pascal_coco_gt(eval_indices)
    model.eval(); model.to(device)
    results = []
    print(f"  Evaluating torchvision model on {len(eval_indices)} Pascal VOC images ...")
    t0 = time.time()
    for loop_i, img_idx in enumerate(eval_indices):
        img_tensor, target = voc_ds[img_idx]
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            out = model([img_tensor])[0]
        image_id = loop_i + 1
        for b, s, l in zip(out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()):
            coco_id = l.item()
            if coco_id not in COCO_CATID_TO_VOC_CATID:
                continue
            x1, y1, x2, y2 = b.tolist()
            results.append({"image_id": image_id, "category_id": COCO_CATID_TO_VOC_CATID[coco_id],
                            "bbox": [x1, y1, x2-x1, y2-y1], "score": s.item()})
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s  ({len(results)} detections)")
    metrics = _run_cocoeval_pascal(results, coco_gt, image_ids)
    print(f"  mAP@0.50 = {metrics['map50']:.4f}  |  mAP@0.50:0.95 = {metrics['map']:.4f}")
    return metrics


# ===================================================================
#  DEIMv2 + DINOv3 HELPERS
# ===================================================================
COCO_CONTIGUOUS_TO_CATID = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42,
    43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
    62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84,
    85, 86, 87, 88, 89, 90,
]
COCO_CATID_TO_CONTIGUOUS = {v: i for i, v in enumerate(COCO_CONTIGUOUS_TO_CATID)}


def load_deimv2_model(device):
    print("  Loading DEIMv2-S (DINOv3 ViT-S backbone) from HuggingFace Hub ...")
    try:
        from huggingface_hub import hf_hub_download
        import subprocess
        try:
            from engine.backbone import DINOv3STAs
            _deimv2_available = True
        except ImportError:
            _deimv2_available = False
        if not _deimv2_available:
            deimv2_dir = os.path.join(os.getcwd(), "_deimv2_repo")
            if not os.path.exists(deimv2_dir):
                print("  Cloning DEIMv2 repository for model definitions ...")
                subprocess.run(["git", "clone", "--depth", "1",
                    "https://github.com/Intellindust-AI-Lab/DEIMv2.git", deimv2_dir],
                    check=True, capture_output=True)
            sys.path.insert(0, deimv2_dir)
        from engine.backbone import DINOv3STAs
        from engine.deim import HybridEncoder, DEIMTransformer
        from engine.deim.postprocessor import PostProcessor
        import torch.nn as nn
        from huggingface_hub import PyTorchModelHubMixin

        class DEIMv2(nn.Module, PyTorchModelHubMixin):
            def __init__(self, config):
                super().__init__()
                self.backbone = DINOv3STAs(**config["DINOv3STAs"])
                self.encoder = HybridEncoder(**config["HybridEncoder"])
                self.decoder = DEIMTransformer(**config["DEIMTransformer"])
                self.postprocessor = PostProcessor(**config["PostProcessor"])
            def forward(self, x, orig_target_sizes):
                x = self.backbone(x)
                x = self.encoder(x)
                x = self.decoder(x)
                x = self.postprocessor(x, orig_target_sizes)
                return x

        model = DEIMv2.from_pretrained("Intellindust/DEIMv2_DINOv3_S_COCO")
        model.eval(); model.to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  DEIMv2-S loaded: {n_params / 1e6:.2f}M parameters")
        return model
    except Exception as e:
        print(f"  [ERROR] Failed to load DEIMv2: {e}")
        raise

def _deimv2_preprocess_image(img_pil, target_size=640):
    import torchvision.transforms as T
    transform = T.Compose([T.Lambda(lambda img: img.convert("RGB")),
                           T.Resize((target_size, target_size)), T.ToTensor()])
    return transform(img_pil)


# ===================================================================
#  DEIMv2 WRAPPER for single-argument forward (quantization compat)
# ===================================================================
class DEIMv2SingleArgWrapper(torch.nn.Module):
    """
    Wraps a DEIMv2 model so that forward(x) only takes the image tensor.
    Internally supplies orig_target_sizes assuming all images are
    (target_size x target_size) -- which is true for calibration batches.

    This is needed because sensitivity_search / sensitivities.find_range
    calls model(calib_data) with a single tensor argument.
    """
    def __init__(self, model, target_size):
        super().__init__()
        self.model = model
        self.target_size = target_size

    def forward(self, x):
        orig_sizes = torch.tensor(
            [[self.target_size, self.target_size]] * x.shape[0],
            dtype=torch.long, device=x.device,
        )
        return self.model(x, orig_sizes)


def _unwrap_deimv2(model):
    """Get the raw DEIMv2 model from a possible wrapper."""
    return model.model if isinstance(model, DEIMv2SingleArgWrapper) else model


def eval_deimv2_coco(model, device, batch_size=16, target_size=640):
    from pycocotools.cocoeval import COCOeval
    from torchvision.datasets import CocoDetection
    img_dir = os.path.join(COCO_ROOT, "images", "val2017")
    ann_file = os.path.join(COCO_ROOT, "annotations", "annotations","annotations" ,"instances_val2017.json")
    coco_gt = _get_coco_gt(ann_file)
    ds = CocoDetection(img_dir, ann_file)
    all_img_ids = ds.ids
    raw_model = _unwrap_deimv2(model)
    raw_model.eval(); raw_model.to(device)
    results = []
    n_images = len(ds)
    print(f"  Evaluating DEIMv2-S on {n_images} COCO val images ...")
    t0 = time.time()
    for start_idx in range(0, n_images, batch_size):
        end_idx = min(start_idx + batch_size, n_images)
        batch_imgs, batch_ids = [], []
        for i in range(start_idx, end_idx):
            img_pil, _ = ds[i]
            orig_w, orig_h = img_pil.size
            batch_imgs.append(_deimv2_preprocess_image(img_pil, target_size))
            batch_ids.append((all_img_ids[i], orig_w, orig_h))
        images = torch.stack(batch_imgs).to(device)
        orig_sizes = torch.tensor([[h, w] for (_, w, h) in batch_ids],
                                  dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = raw_model(images, orig_sizes)
        for bi in range(len(batch_ids)):
            image_id = batch_ids[bi][0]
            labels_i = outputs[bi]['labels'].cpu()
            boxes_i = outputs[bi]['boxes'].cpu()
            scores_i = outputs[bi]['scores'].cpu()
            for j in range(len(scores_i)):
                score = scores_i[j].item()
                if score < 0.01: continue
                cls_idx = labels_i[j].item()
                if cls_idx < 0 or cls_idx >= len(COCO_CONTIGUOUS_TO_CATID): continue
                cat_id = COCO_CONTIGUOUS_TO_CATID[cls_idx]
                x1, y1, x2, y2 = boxes_i[j].tolist()
                results.append({"image_id": image_id, "category_id": cat_id,
                                "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
        if (start_idx // batch_size) % 50 == 0:
            print(f"    processed {end_idx}/{n_images} images ...")
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s  ({len(results)} detections)")
    if len(results) == 0:
        return {"map50": 0.0, "map": 0.0}
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate(); coco_eval.accumulate()
    with contextlib.redirect_stdout(io.StringIO()):
        coco_eval.summarize()
    return {"map": coco_eval.stats[0], "map50": coco_eval.stats[1]}


def eval_deimv2_pascal(model, device, target_size=640, max_images=None, seed=42):
    from torchvision.datasets import VOCDetection
    import random as _random
    voc_ds = VOCDetection(root=PASCAL_ROOT, year="2012", image_set="val", download=False)
    if max_images is not None and max_images < len(voc_ds):
        _random.seed(seed)
        eval_indices = _random.sample(range(len(voc_ds)), max_images)
    else:
        eval_indices = list(range(len(voc_ds)))
    coco_gt, image_ids = _build_pascal_coco_gt(eval_indices)
    raw_model = _unwrap_deimv2(model)
    raw_model.eval(); raw_model.to(device)
    results = []
    print(f"  Evaluating DEIMv2-S on {len(eval_indices)} Pascal VOC images ...")
    t0 = time.time()
    for loop_i, img_idx in enumerate(eval_indices):
        img_pil, target = voc_ds[img_idx]
        orig_w, orig_h = img_pil.size
        img_tensor = _deimv2_preprocess_image(img_pil, target_size).unsqueeze(0).to(device)
        orig_sizes = torch.tensor([[orig_h, orig_w]], dtype=torch.long, device=device)
        with torch.no_grad():
            outputs = raw_model(img_tensor, orig_sizes)
        image_id = loop_i + 1
        labels_i = outputs[0]["labels"].cpu()
        boxes_i = outputs[0]["boxes"].cpu()
        scores_i = outputs[0]["scores"].cpu()
        for j in range(len(scores_i)):
            score = scores_i[j].item()
            if score < 0.01: continue
            cls_idx = labels_i[j].item()
            if cls_idx not in COCO_CONTIGUOUS_TO_VOC_CATID: continue
            x1, y1, x2, y2 = boxes_i[j].tolist()
            results.append({"image_id": image_id, "category_id": COCO_CONTIGUOUS_TO_VOC_CATID[cls_idx],
                            "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s  ({len(results)} detections)")
    metrics = _run_cocoeval_pascal(results, coco_gt, image_ids)
    print(f"  mAP@0.50 = {metrics['map50']:.4f}  |  mAP@0.50:0.95 = {metrics['map']:.4f}")
    return metrics


# ===================================================================
#  MODEL LOADERS
# ===================================================================
def load_ultralytics_model(model_key):
    from ultralytics import YOLO
    name_map = {"yolov8n": "yolov8n.pt", "yolo26n": "yolo26n.pt", "rtdetr-n": "rtdetr-l.pt"}
    pt_name = name_map.get(model_key, model_key + ".pt")
    print(f"  Loading Ultralytics model: {pt_name}")
    model = YOLO(pt_name)
    print(model)
    return model

def load_torchvision_model(model_key, device):
    import torchvision
    if model_key == "mobilenet_ssd_v2":
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights="DEFAULT")
    elif model_key == "fasterrcnn_mobilenet":
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT")
    else:
        raise ValueError(f"Unknown torchvision model key: {model_key}")
    model.eval(); model.to(device)
    return model

ULTRALYTICS_MODELS = {"yolov8n", "yolo26n", "rtdetr-n"}
TORCHVISION_MODELS = {"mobilenet_ssd_v2", "fasterrcnn_mobilenet"}
DEIMV2_MODELS      = {"deimv2_dinov3_s"}
ALL_MODELS = ULTRALYTICS_MODELS | TORCHVISION_MODELS | DEIMV2_MODELS


# ===================================================================
#  ULTRALYTICS EVAL
# ===================================================================
def _print_results(results, tag, model_name, device, elapsed):
    print("\n" + "=" * 60)
    print(f"  {tag}  |  {model_name}  |  {device}  |  {elapsed:.1f}s")
    print(f"  mAP@0.50: {results.box.map50:.4f}  mAP@0.50:0.95: {results.box.map:.4f}")
    print("=" * 60)

def eval_ultralytics_pascal(model, device, max_images=None, seed=42):
    from torchvision.datasets import VOCDetection
    import random as _random
    voc_ds = VOCDetection(root=PASCAL_ROOT, year="2012", image_set="val", download=False)
    if max_images is not None and max_images < len(voc_ds):
        _random.seed(seed)
        eval_indices = _random.sample(range(len(voc_ds)), max_images)
    else:
        eval_indices = list(range(len(voc_ds)))
    coco_gt, image_ids = _build_pascal_coco_gt(eval_indices)
    results = []
    print(f"  Evaluating Ultralytics model on {len(eval_indices)} Pascal VOC images ...")
    t0 = time.time()
    for loop_i, img_idx in enumerate(eval_indices):
        img_pil, target = voc_ds[img_idx]
        img_pil = img_pil.convert("RGB")
        preds = model.predict(img_pil, imgsz=IMG_SIZE, device=device, verbose=False)
        pred = preds[0]
        image_id = loop_i + 1
        for b, s, l in zip(pred.boxes.xyxy.cpu(), pred.boxes.conf.cpu(), pred.boxes.cls.cpu().int()):
            cls_idx = l.item()
            if cls_idx not in COCO_CONTIGUOUS_TO_VOC_CATID: continue
            x1, y1, x2, y2 = b.tolist()
            results.append({"image_id": image_id, "category_id": COCO_CONTIGUOUS_TO_VOC_CATID[cls_idx],
                            "bbox": [x1, y1, x2-x1, y2-y1], "score": s.item()})
        if (loop_i + 1) % 500 == 0:
            print(f"    processed {loop_i + 1}/{len(eval_indices)} images ...")
    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s  ({len(results)} detections)")
    metrics = _run_cocoeval_pascal(results, coco_gt, image_ids)
    print(f"  mAP@0.50 = {metrics['map50']:.4f}  |  mAP@0.50:0.95 = {metrics['map']:.4f}")
    return metrics

def run_ultralytics_baseline(model_key, data_yaml, dataset_name, device):
    model = load_ultralytics_model(model_key)
    model.to(device)
    t0 = time.time()
    if dataset_name == "pascal":
        metrics = eval_ultralytics_pascal(model, device)
        print(f"\n  [{model_key} / pascal] FP32 baseline: {metrics}  ({time.time()-t0:.1f}s)")
        return model, metrics
    else:
        results = model.val(data=data_yaml, imgsz=IMG_SIZE, batch=BATCH_SIZE,
                            device=device, workers=WORKERS, split="val", verbose=True)
        _print_results(results, "FP32 baseline", model_key, device, time.time()-t0)
        return model, {"map50": results.box.map50, "map": results.box.map}

def run_ultralytics_quantized(model_key, data_yaml, calib_img_path, dataset_name, device,
                              accuracy_target=0.01):
    if not HAS_LOFLOAT:
        print("  [SKIP] LoFloat not available."); return None
    print(f"\n  >>> Quantizing {model_key} with accuracy_target={accuracy_target} <<<")
    model = load_ultralytics_model(model_key)
    inner = model.model.to(device); inner.eval()
    n_samples = 64
    dataset = CalibDataset(calib_img_path, data_yaml, imgsz=IMG_SIZE)
    indices = torch.randperm(len(dataset))[:n_samples]
    subset = Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=64, shuffle=False)
    all_imgs = []
    for imgs, _ in loader: all_imgs.append(imgs)
    val_data = torch.cat(all_imgs, dim=0).to(device)
    with torch.no_grad(): baseline_out = inner(val_data)
    def _detach(x):
        if isinstance(x, torch.Tensor): return x.detach().to(device)
        elif isinstance(x, dict): return {k: _detach(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)): return type(x)(_detach(v) for v in x)
        return x
    baseline = _detach(baseline_out)
    def _flatten(x):
        if isinstance(x, torch.Tensor): return [x.float().flatten()]
        elif isinstance(x, dict): return [t for v in x.values() for t in _flatten(v)]
        elif isinstance(x, (list, tuple)): return [t for v in x for t in _flatten(v)]
        return []
    baseline_flat = torch.cat(_flatten(baseline))
    def loss_fn(preds):
        p_flat = torch.cat(_flatten(preds))
        return cosine_distance_chunked(p_flat, baseline_flat)

    import random as _random
    os.environ["YOLO_VERBOSE"] = "false"
    def _make_val_subset(data_yaml, n_images, seed=42):
        with open(data_yaml) as f: cfg = yaml.safe_load(f)
        base_path = cfg['path']; val_rel = cfg['val']
        val_path = os.path.join(base_path, val_rel)
        all_imgs = sorted([fn for fn in os.listdir(val_path) if fn.lower().endswith(('.jpg','.jpeg','.png'))])
        _random.seed(seed)
        chosen = _random.sample(all_imgs, min(n_images, len(all_imgs)))
        subset_img_dir = os.path.join(base_path, 'images', 'val_subset')
        os.makedirs(subset_img_dir, exist_ok=True)
        for img in chosen:
            dst = os.path.join(subset_img_dir, img)
            if not os.path.exists(dst): os.symlink(os.path.abspath(os.path.join(val_path, img)), dst)
        src_label_dir = os.path.join(base_path, 'labels', os.path.basename(val_rel))
        subset_lbl_dir = os.path.join(base_path, 'labels', 'val_subset')
        os.makedirs(subset_lbl_dir, exist_ok=True)
        for img in chosen:
            lbl = os.path.splitext(img)[0] + '.txt'
            src = os.path.join(src_label_dir, lbl); dst = os.path.join(subset_lbl_dir, lbl)
            if os.path.exists(src) and not os.path.exists(dst): os.symlink(os.path.abspath(src), dst)
        subset_yaml = data_yaml.replace('.yaml', '_subset.yaml')
        subset_cfg = dict(cfg); subset_cfg['val'] = 'images/val_subset'
        with open(subset_yaml, 'w') as f: yaml.dump(subset_cfg, f, default_flow_style=False)
        return subset_yaml

    fast_data_yaml = _make_val_subset(data_yaml, n_images=SEARCH_SUBSET_SIZE)
    logging.getLogger("ultralytics").setLevel(logging.CRITICAL)
    if dataset_name == "pascal":
        fp32_subset_metrics = eval_ultralytics_pascal(model, device, max_images=SEARCH_SUBSET_SIZE, seed=42)
        fp32_map = fp32_subset_metrics.get("map", 0.0)
    else:
        fp32_results = model.val(data=fast_data_yaml, imgsz=IMG_SIZE, batch=BATCH_SIZE,
                                 device=device, workers=WORKERS, split="val", verbose=False, plots=False)
        fp32_map = fp32_results.box.map
    print(f"  FP32 mAP on {SEARCH_SUBSET_SIZE}-image subset: {fp32_map:.4f} (cached)")

    def eval_fn(m, data):
        model.model = m.eval().to(device)
        if dataset_name == "pascal":
            metrics = eval_ultralytics_pascal(model, device, max_images=SEARCH_SUBSET_SIZE, seed=42)
            cur_map = metrics.get("map", 0.0)
        else:
            results = model.val(data=fast_data_yaml, imgsz=IMG_SIZE, batch=BATCH_SIZE,
                                device=device, workers=WORKERS, split="val", verbose=False, plots=False)
            cur_map = results.box.map
        return (fp32_map - cur_map) / fp32_map

    baseline_score = eval_fn(inner, None)
    print(f"  Baseline score: {baseline_score:.4f}")
    lof_model = lof.lofloatify(inner); lof_model.to(device).eval()
    with torch.no_grad():
        lof_out = lof_model(val_data)
        cos_dist = loss_fn(lof_out).item()
    print("  Cosine distance (fully quantized vs FP32):", cos_dist)
    del lof_out
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"  Running greedy_sensitivity quantization search (accuracy_target={accuracy_target}) ...")
    quantized_inner = ss.greedy_sensitivity(model=inner, sensitivity_measure="hessian",
        data=dataset, loss_fn=loss_fn, eval_fn=eval_fn, accuracy_target=accuracy_target,
        bs=[4,3,2], es=[4,3,2], accum_bw=[14,12,10], n_samples=n_samples, device=device, baseline=baseline_score)
    print(lof.record_formats(quantized_inner))
    quantized_inner.to(device)
    with torch.no_grad():
        preds = quantized_inner(val_data)
        final_cos = loss_fn(preds).item()
    print(f"  Final cosine distance: {final_cos:.6f}")
    del preds
    for name,module in quantized_inner.named_modules():
        if isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            print(f"  Layer: {name}  |  Format: {lof.record_formats(module)}")
            print(f"    Weights: {module.weight.shape}  |  Bias: {module.bias.shape if module.bias is not None else None}")
            print(f"    Weight stats: mean={module.weight.mean().item():.4f}  std={module.weight.std().item():.4f}")
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.model = quantized_inner
    t0 = time.time()
    if dataset_name == "pascal":
        metrics = eval_ultralytics_pascal(model, device)
        print(f"\n  [{model_key}/pascal] Quantized (target={accuracy_target}): {metrics}  ({time.time()-t0:.1f}s)")
        return metrics
    else:
        results = model.val(data=data_yaml, imgsz=IMG_SIZE, batch=BATCH_SIZE,
                            device=device, workers=WORKERS, split="val", verbose=True)
        _print_results(results, f"Quantized (LoFloat, target={accuracy_target})", model_key, device, time.time()-t0)
        return {"map50": results.box.map50, "map": results.box.map}


# ===================================================================
#  TORCHVISION QUANTIZED
# ===================================================================
def run_torchvision_baseline(model_key, dataset_name, device):
    model = load_torchvision_model(model_key, device)
    if dataset_name == "coco":
        loader, coco_gt, info = _build_torchvision_dataloader("coco", BATCH_SIZE)
        metrics = eval_torchvision_coco(model, loader, coco_gt, device)
    else:
        metrics = eval_torchvision_pascal(model, device)
    print(f"\n  [{model_key} / {dataset_name}] FP32 baseline: {metrics}")
    return model, metrics

def run_torchvision_quantized(model_key, dataset_name, device, accuracy_target=0.01):
    if not HAS_LOFLOAT:
        print("  [SKIP] LoFloat not available."); return None
    print(f"\n  >>> Quantizing {model_key} with accuracy_target={accuracy_target} <<<")
    model = load_torchvision_model(model_key, device)
    import torchvision.transforms as T
    class RGBToTensor:
        def __call__(self, img):
            img = img.convert("RGB"); return T.functional.to_tensor(img)
    transform = RGBToTensor()
    if dataset_name == "coco":
        from torchvision.datasets import CocoDetection
        img_dir = os.path.join(COCO_ROOT, "images", "val2017")
        ann_file = os.path.join(COCO_ROOT, "annotations", "annotations", "annotations", "instances_val2017.json")
        ds = CocoDetection(img_dir, ann_file, transform=transform)
        coco_gt = _get_coco_gt(ann_file)
    else:
        from torchvision.datasets import VOCDetection
        ds = VOCDetection(root=PASCAL_ROOT, year="2012", image_set="val", download=False, transform=transform)
        coco_gt = None
    target_size = 320 if "ssd" in model_key else 640
    n_calib = 128
    indices = torch.randperm(len(ds))[:n_calib]
    calib_images = []
    for i in indices:
        img, _ = ds[int(i)]
        img = F.interpolate(img.unsqueeze(0), size=(target_size, target_size),
                            mode="bilinear", align_corners=False).squeeze(0)
        calib_images.append(img)
    calib_batch = torch.stack(calib_images).to(device)
    model.eval()
    with torch.no_grad(): baseline_out = model([img for img in calib_batch])
    def _flatten(outputs):
        parts = []
        if isinstance(outputs, (list, tuple)):
            for o in outputs:
                if isinstance(o, dict):
                    if "boxes" in o and len(o["boxes"]) > 0: parts.append(o["boxes"].float().reshape(-1))
                    if "scores" in o and len(o["scores"]) > 0: parts.append(o["scores"].float().reshape(-1))
                elif isinstance(o, torch.Tensor): parts.append(o.float().flatten())
        elif isinstance(outputs, dict):
            for v in outputs.values():
                if isinstance(v, torch.Tensor) and v.numel() > 0: parts.append(v.float().reshape(-1))
        elif isinstance(outputs, torch.Tensor): parts.append(outputs.float().flatten())
        return torch.cat(parts) if parts else torch.zeros(1, device=device)
    baseline_flat = _flatten(baseline_out).detach()
    def loss_fn(preds):
        p_flat = _flatten(preds)
        return cosine_distance_chunked(p_flat, baseline_flat)

    import random as _random
    _random.seed(42)
    subset_indices = _random.sample(range(len(ds)), min(SEARCH_SUBSET_SIZE, len(ds)))
    eval_subset = Subset(ds, subset_indices)
    def _collate_no_resize(batch): return list(zip(*batch))
    subset_loader = DataLoader(eval_subset, batch_size=BATCH_SIZE, shuffle=False,
                               num_workers=WORKERS, collate_fn=_collate_no_resize)
    if dataset_name == "coco":
        fp32_subset_metrics = eval_torchvision_coco(model, subset_loader, coco_gt, device)
    else:
        fp32_subset_metrics = eval_torchvision_pascal(model, device, max_images=SEARCH_SUBSET_SIZE, seed=42)
    fp32_map = fp32_subset_metrics.get("map", 0.0)
    print(f"  FP32 mAP on {SEARCH_SUBSET_SIZE}-image subset: {fp32_map:.4f} (cached)")

    def eval_fn(m, data):
        m.eval().to(device)
        if dataset_name == "coco":
            metrics = eval_torchvision_coco(m, subset_loader, coco_gt, device)
        else:
            metrics = eval_torchvision_pascal(m, device, max_images=SEARCH_SUBSET_SIZE, seed=42)
        cur_map = metrics.get("map", 0.0)
        return 0.0 if fp32_map == 0.0 else (fp32_map - cur_map) / fp32_map

    baseline_score = eval_fn(model, None)
    print(f"  Baseline score (subset): {baseline_score:.4f}")
    lof_model = lof.lofloatify(model); lof_model.to(device).eval()
    with torch.no_grad():
        lof_out = lof_model([img for img in calib_batch])
        cos_dist = loss_fn(lof_out).item()
    print("  Cosine distance (fully quantized vs FP32):", cos_dist)
    del lof_out
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    def tv_collate_fn(batch):
        images, labels = zip(*batch)
        images = torch.stack([F.interpolate(img.unsqueeze(0), size=(target_size, target_size),
                              mode="bilinear", align_corners=False).squeeze(0) for img in images])
        return images, list(labels)
    print(f"  Running greedy_sensitivity quantization search (accuracy_target={accuracy_target}) ...")
    quantized_model = ss.greedy_sensitivity(model=model, sensitivity_measure="hessian",
        data=ds, loss_fn=loss_fn, eval_fn=eval_fn, accuracy_target=accuracy_target,
        bs=[4,3,2], es=[4,3,2,1], accum_bw=[14,12,10], n_samples=n_calib, device=device,
        collate_fn=tv_collate_fn, baseline=baseline_score)
    print(lof.record_formats(quantized_model))
    quantized_model.to(device).eval()
    with torch.no_grad():
        preds = quantized_model([img for img in calib_batch])
        final_cos = loss_fn(preds).item()
    print(f"  Final cosine distance: {final_cos:.6f}")
    del preds
   
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    if dataset_name == "coco":
        loader, coco_gt_full, _ = _build_torchvision_dataloader("coco", BATCH_SIZE)
        metrics = eval_torchvision_coco(quantized_model, loader, coco_gt_full, device)
    else:
        metrics = eval_torchvision_pascal(quantized_model, device)
    print(f"\n  [{model_key} / {dataset_name}] Quantized (LoFloat, target={accuracy_target}): {metrics}")
    return metrics


# ===================================================================
#  DEIMv2 BASELINE & QUANTIZED
# ===================================================================
def run_deimv2_baseline(dataset_name, device):
    model = load_deimv2_model(device)
    if dataset_name == "coco":
        metrics = eval_deimv2_coco(model, device, batch_size=BATCH_SIZE, target_size=IMG_SIZE)
    else:
        metrics = eval_deimv2_pascal(model, device, target_size=IMG_SIZE)
    print(f"\n  [deimv2_dinov3_s / {dataset_name}] FP32 baseline: {metrics}")
    return model, metrics

def run_deimv2_quantized(dataset_name, device, accuracy_target=0.01):
    """
    DEIMv2's forward() requires (images, orig_target_sizes).
    sensitivity_search calls model(calib_data) with one arg.
    Solution: wrap in DEIMv2SingleArgWrapper for the search,
    unwrap for eval functions that need real image dimensions.
    """
    if not HAS_LOFLOAT:
        print("  [SKIP] LoFloat not available."); return None
    print(f"\n  >>> Quantizing deimv2_dinov3_s with accuracy_target={accuracy_target} <<<")
    model = load_deimv2_model(device)
    target_size = IMG_SIZE
    wrapped_model = DEIMv2SingleArgWrapper(model, target_size)

    if dataset_name == "coco":
        from torchvision.datasets import CocoDetection
        img_dir = os.path.join(COCO_ROOT, "images", "val2017")
        ann_file = os.path.join(COCO_ROOT, "annotations", "annotations", "annotations", "instances_val2017.json")
        ds = CocoDetection(img_dir, ann_file)
    else:
        from torchvision.datasets import VOCDetection
        ds = VOCDetection(root=PASCAL_ROOT, year="2012", image_set="val", download=False)

    n_calib = 128
    indices = torch.randperm(len(ds))[:n_calib]
    calib_images = []
    for i in indices:
        img_pil = ds[int(i)][0]
        calib_images.append(_deimv2_preprocess_image(img_pil, target_size))
    calib_batch = torch.stack(calib_images).to(device)

    # FP32 baseline via wrapper (single-arg forward)
    wrapped_model.eval(); wrapped_model.to(device)
    with torch.no_grad(): baseline_out = wrapped_model(calib_batch)

    def _flatten(outputs):
        parts = []
        if isinstance(outputs, (list, tuple)):
            for o in outputs:
                if isinstance(o, dict):
                    for v in o.values():
                        if isinstance(v, torch.Tensor) and v.numel() > 0: parts.append(v.float().reshape(-1))
                elif isinstance(o, torch.Tensor) and o.numel() > 0: parts.append(o.float().reshape(-1))
        elif isinstance(outputs, dict):
            for v in outputs.values():
                if isinstance(v, torch.Tensor) and v.numel() > 0: parts.append(v.float().reshape(-1))
        elif isinstance(outputs, torch.Tensor) and outputs.numel() > 0: parts.append(outputs.float().reshape(-1))
        return torch.cat(parts) if parts else torch.zeros(1, device=device)

    baseline_flat = _flatten(baseline_out).detach()
    def loss_fn(preds):
        p_flat = _flatten(preds)
        return cosine_distance_chunked(p_flat, baseline_flat)

    # Subset eval helpers (unwrap automatically)
    import random as _random
    _random.seed(42)
    subset_indices = _random.sample(range(len(ds)), min(SEARCH_SUBSET_SIZE, len(ds)))
    eval_subset = Subset(ds, subset_indices)

    def _eval_deimv2_on_subset(m, subset, dataset_name):
        from pycocotools.cocoeval import COCOeval
        raw_m = _unwrap_deimv2(m); raw_m.eval().to(device)
        results_list = []
        if dataset_name == "coco":
            coco_gt = _get_coco_gt(os.path.join(COCO_ROOT, "annotations", "annotations", "annotations","instances_val2017.json"))
            base_ds = subset.dataset if isinstance(subset, Subset) else subset
            all_ids = base_ds.ids
        n = len(subset)
        for idx in range(0, n, BATCH_SIZE):
            end = min(idx + BATCH_SIZE, n)
            batch_imgs, batch_meta = [], []
            for j in range(idx, end):
                img_pil = subset[j][0]; orig_w, orig_h = img_pil.size
                batch_imgs.append(_deimv2_preprocess_image(img_pil, target_size))
                if dataset_name == "coco":
                    orig_idx = subset.indices[j] if isinstance(subset, Subset) else j
                    batch_meta.append((all_ids[orig_idx], orig_w, orig_h))
                else:
                    batch_meta.append((j, orig_w, orig_h))
            images = torch.stack(batch_imgs).to(device)
            orig_sizes = torch.tensor([[h, w] for (_, w, h) in batch_meta], dtype=torch.long, device=device)
            with torch.no_grad(): outputs = raw_m(images, orig_sizes)
            for bi in range(len(batch_meta)):
                labels_i = outputs[bi]['labels'].cpu()
                boxes_i = outputs[bi]['boxes'].cpu()
                scores_i = outputs[bi]['scores'].cpu()
                for k in range(len(scores_i)):
                    score = scores_i[k].item()
                    if score < 0.01: continue
                    cls_idx = labels_i[k].item()
                    if dataset_name == "coco":
                        if cls_idx < 0 or cls_idx >= len(COCO_CONTIGUOUS_TO_CATID): continue
                        x1, y1, x2, y2 = boxes_i[k].tolist()
                        results_list.append({"image_id": batch_meta[bi][0],
                            "category_id": COCO_CONTIGUOUS_TO_CATID[cls_idx],
                            "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
        if dataset_name == "coco":
            if not results_list: return {"map": 0.0, "map50": 0.0}
            coco_dt = coco_gt.loadRes(results_list)
            coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
            coco_eval.params.imgIds = [all_ids[subset.indices[j]] for j in range(len(subset))]
            coco_eval.evaluate(); coco_eval.accumulate()
            with contextlib.redirect_stdout(io.StringIO()): coco_eval.summarize()
            return {"map": coco_eval.stats[0], "map50": coco_eval.stats[1]}
        else:
            return _eval_deimv2_pascal_subset(m, subset, device, target_size)

    def _eval_deimv2_pascal_subset(m, subset, device, target_size):
        raw_m = _unwrap_deimv2(m); raw_m.eval().to(device)
        eval_indices = list(subset.indices) if isinstance(subset, Subset) else list(range(len(subset)))
        coco_gt, image_ids = _build_pascal_coco_gt(eval_indices)
        results = []
        for loop_i in range(len(subset)):
            img_pil, target = subset[loop_i]; orig_w, orig_h = img_pil.size
            img_tensor = _deimv2_preprocess_image(img_pil, target_size).unsqueeze(0).to(device)
            orig_sizes = torch.tensor([[orig_h, orig_w]], dtype=torch.long, device=device)
            with torch.no_grad(): outputs = raw_m(img_tensor, orig_sizes)
            image_id = loop_i + 1
            for j in range(len(outputs[0]["scores"])):
                score = outputs[0]["scores"][j].item()
                if score < 0.01: continue
                cls_idx = outputs[0]["labels"][j].item()
                if cls_idx not in COCO_CONTIGUOUS_TO_VOC_CATID: continue
                x1, y1, x2, y2 = outputs[0]["boxes"][j].tolist()
                results.append({"image_id": image_id, "category_id": COCO_CONTIGUOUS_TO_VOC_CATID[cls_idx],
                                "bbox": [x1, y1, x2-x1, y2-y1], "score": score})
        metrics = _run_cocoeval_pascal(results, coco_gt, image_ids)
        print(f"  Subset mAP@0.50={metrics['map50']:.4f} | mAP@0.50:0.95={metrics['map']:.4f}")
        return metrics

    # Precompute FP32 mAP on subset
    fp32_subset_metrics = _eval_deimv2_on_subset(wrapped_model, eval_subset, dataset_name)
    fp32_map = fp32_subset_metrics.get("map", 0.0)
    print(f"  FP32 mAP on {SEARCH_SUBSET_SIZE}-image subset: {fp32_map:.4f} (cached)")

    def eval_fn(m, data):
        m.eval().to(device)
        metrics = _eval_deimv2_on_subset(m, eval_subset, dataset_name)
        cur_map = metrics.get("map", 0.0)
        return 0.0 if fp32_map == 0.0 else (fp32_map - cur_map) / fp32_map

    baseline_score = eval_fn(wrapped_model, None)
    print(f"  Baseline score (subset): {baseline_score:.4f}")

    lof_model = lof.lofloatify(wrapped_model); lof_model.to(device).eval()
    with torch.no_grad():
        lof_out = lof_model(calib_batch)
        cos_dist = loss_fn(lof_out).item()
    print("  Cosine distance (fully quantized vs FP32):", cos_dist)
    del lof_out
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Calib dataset — just images (wrapper handles sizes)
    class DEIMv2CalibDataset(Dataset):
        def __init__(self, images):
            self.images = images
        def __len__(self): return len(self.images)
        def __getitem__(self, idx): return self.images[idx], torch.zeros(1)

    calib_ds = DEIMv2CalibDataset(calib_batch.cpu())

    print(f"  Running greedy_sensitivity quantization search (accuracy_target={accuracy_target}) ...")
    quantized_wrapped = ss.greedy_sensitivity(
        model=wrapped_model, sensitivity_measure="hessian", data=calib_ds,
        loss_fn=loss_fn, eval_fn=eval_fn, accuracy_target=accuracy_target,
        bs=[4,3,2], es=[4,3,2,1], accum_bw=[14,12,10], n_samples=n_calib, device=device, baseline=baseline_score)

    print(lof.record_formats(quantized_wrapped))
    quantized_wrapped.to(device).eval()
    with torch.no_grad():
        preds = quantized_wrapped(calib_batch)
        final_cos = loss_fn(preds).item()
    print(f"  Final cosine distance: {final_cos:.6f}")
    del preds
    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Final eval on full dataset (eval functions auto-unwrap)
    if dataset_name == "coco":
        metrics = eval_deimv2_coco(quantized_wrapped, device, batch_size=BATCH_SIZE, target_size=target_size)
    else:
        metrics = eval_deimv2_pascal(quantized_wrapped, device, target_size=target_size)
    print(f"\n  [deimv2_dinov3_s / {dataset_name}] Quantized (LoFloat, target={accuracy_target}): {metrics}")
    return metrics


# ===================================================================
#  MAIN ORCHESTRATOR
# ===================================================================
def print_summary(all_results):
    print("\n\n" + "=" * 90)
    print("  FINAL SUMMARY")
    print("=" * 90)
    header = f"  {'Model':<25s} {'Dataset':<10s} {'Variant':<12s} {'Acc Target':<12s} {'mAP@0.50':<12s} {'mAP@0.50:95':<12s}"
    print(header); print("-" * 90)
    for r in all_results:
        map50 = r.get("map50", "-"); map95 = r.get("map", "-")
        if isinstance(map50, float): map50 = f"{map50:.4f}"
        if isinstance(map95, float): map95 = f"{map95:.4f}"
        acc_target = r.get("accuracy_target", "-")
        if isinstance(acc_target, float): acc_target = f"{acc_target}"
        print(f"  {r['model']:<25s} {r['dataset']:<10s} {r['variant']:<12s} {str(acc_target):<12s} {map50:<12s} {map95:<12s}")
    print("=" * 90)

def main():
    parser = argparse.ArgumentParser(description="Object Detection Quantization Benchmark")
    parser.add_argument("--models", nargs="+",
                        default=["yolov8n", "yolo26n", "mobilenet_ssd_v2", "rtdetr-n", "deimv2_dinov3_s"],
                        choices=sorted(ALL_MODELS))
    parser.add_argument("--datasets", nargs="+", default=["coco"], choices=["coco", "pascal"])
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--skip-quantization", action="store_true")
    parser.add_argument("--accuracy-targets", nargs="+", type=float, default=ACCURACY_TARGETS,
                        help="List of accuracy targets to run quantization for (default: 0.01 0.05)")
    parser.add_argument("--search-subset-size", type=int, default=SEARCH_SUBSET_SIZE)
    args = parser.parse_args()
    _update_config(args.batch_size, args.img_size, args.workers, args.device,
                   args.search_subset_size, args.accuracy_targets)

    data_yamls, calib_img_paths = {}, {}
    if "coco" in args.datasets:
        print("\n=== Preparing COCO val2017 ===")
        data_yamls["coco"] = prepare_coco()
        calib_img_paths["coco"] = os.path.join(COCO_ROOT, "images", "val2017")
    if "pascal" in args.datasets:
        print("\n=== Preparing Pascal VOC 2012 val ===")
        data_yamls["pascal"] = prepare_pascal()
        calib_img_paths["pascal"] = os.path.join(PASCAL_ROOT, "VOCdevkit", "VOC2012", "images", "val")

    all_results = []
    for model_key in args.models:
        for ds_name in args.datasets:
            print(f"\n{'#'*60}\n  MODEL: {model_key}  |  DATASET: {ds_name}\n{'#'*60}")
            is_ultra = model_key in ULTRALYTICS_MODELS
            is_tv = model_key in TORCHVISION_MODELS
            is_deim = model_key in DEIMV2_MODELS

            if is_ultra:
                _, mfp32 = run_ultralytics_baseline(model_key, data_yamls[ds_name], ds_name, DEVICE)
            elif is_deim:
                _, mfp32 = run_deimv2_baseline(ds_name, DEVICE)
            else:
                _, mfp32 = run_torchvision_baseline(model_key, ds_name, DEVICE)
            all_results.append(dict(model=model_key, dataset=ds_name, variant="FP32",
                                    accuracy_target="-", **mfp32))

            if not args.skip_quantization:
                for acc_target in ACCURACY_TARGETS:
                    print(f"\n  --- Quantization pass: accuracy_target={acc_target} ---")
                    if is_ultra:
                        mq = run_ultralytics_quantized(model_key, data_yamls[ds_name],
                                                       calib_img_paths[ds_name], ds_name, DEVICE,
                                                       accuracy_target=acc_target)
                    elif is_deim:
                        mq = run_deimv2_quantized(ds_name, DEVICE, accuracy_target=acc_target)
                    else:
                        mq = run_torchvision_quantized(model_key, ds_name, DEVICE,
                                                       accuracy_target=acc_target)
                    if mq is not None:
                        all_results.append(dict(model=model_key, dataset=ds_name, variant="LoFloat",
                                                accuracy_target=acc_target, **mq))

    print_summary(all_results)

if __name__ == "__main__":
    main()