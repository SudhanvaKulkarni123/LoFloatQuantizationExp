"""
Object Detection Quantization Benchmark
========================================
Tests quantization (via LoFloat / sensitivity_search) on multiple SOTA
edge-oriented detection networks, evaluated on COCO val2017 and/or Pascal VOC 2007 test.

Supported models:
  - yolov8n          (Ultralytics)
  - yolo26n          (Ultralytics)
  - rtdetr-s         (Ultralytics, RT-DETR small)
  - mobilenet_ssd_v2 (torchvision SSDLite320 + MobileNetV3-Large)
  - fasterrcnn_mobilenet (torchvision Faster R-CNN + MobileNetV3-Large-FPN, optional)

Supported datasets:
  - coco   (val2017, ~5 k images, 80 classes)
  - pascal  (VOC2007 test, ~4.9 k images, 20 classes)

Usage:
  python obj_det_quant_bench.py --models yolov8n yolo26n mobilenet_ssd_v2 rtdetr-s \
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

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset

import yaml

# ---------------------------------------------------------------------------
# We import the user's custom quantization libs; adjust paths as needed.
# ---------------------------------------------------------------------------
sys.path.append("..")
try:
    import LoFloat as lof
    import sensitivity_search as ss
    HAS_LOFLOAT = True
except ImportError:
    HAS_LOFLOAT = False
    print("[WARN] LoFloat / sensitivity_search not found — "
          "quantization steps will be skipped, only FP baselines will run.")

# ===================================================================
#  GLOBAL CONFIG (overridden by CLI)
# ===================================================================
IMG_SIZE   = 640
BATCH_SIZE = 16
WORKERS    = 4
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

COCO_ROOT   = "./coco"
PASCAL_ROOT = "./VOCdevkit"


def _update_config(batch_size, img_size, workers, device):
    global BATCH_SIZE, IMG_SIZE, WORKERS, DEVICE
    BATCH_SIZE = batch_size
    IMG_SIZE   = img_size
    WORKERS    = workers
    DEVICE     = device

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
    """COCO JSON  →  per-image YOLO .txt files."""
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
    annot_dir  = os.path.join(COCO_ROOT, "annotations", "annotations")

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
    print(f"  Written COCO yaml → {yaml_path}")
    return yaml_path


# ----------------------------- PASCAL VOC ----------------------------
PASCAL_URLS = [
    # pjreddie mirror (most reliable)
    "https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar",
    # Original Oxford host (frequently down)
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar",
]

PASCAL_CLASS_NAMES = [
    "aeroplane","bicycle","bird","boat","bottle","bus","car","cat",
    "chair","cow","diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
]


def _convert_voc_to_yolo(voc_root):
    """Convert Pascal VOC XML annotations → YOLO txt labels."""
    import xml.etree.ElementTree as ET

    img_dir   = os.path.join(voc_root, "JPEGImages")
    annot_dir = os.path.join(voc_root, "Annotations")
    label_dir = os.path.join(voc_root, "labels")
    os.makedirs(label_dir, exist_ok=True)

    cls_to_idx = {c: i for i, c in enumerate(PASCAL_CLASS_NAMES)}

    test_txt = os.path.join(voc_root, "ImageSets", "Main", "test.txt")
    with open(test_txt) as f:
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


def prepare_pascal():
    if not os.path.exists(os.path.join(PASCAL_ROOT, "VOC2007")):
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
            raise RuntimeError(
                "Could not download Pascal VOC 2007 from any mirror.\n"
                "Please download VOCtest_06-Nov-2007.tar manually and extract "
                "it so that ./VOCdevkit/VOC2007/ exists.\n"
                "Mirrors to try:\n"
                "  https://pjreddie.com/media/files/VOCtest_06-Nov-2007.tar\n"
                "  https://www.kaggle.com/datasets/zaraks/pascal-voc-2007\n"
            )
    else:
        print("  [pascal] VOC2007 already present.")

    voc_root = os.path.join(PASCAL_ROOT, "VOC2007")
    _convert_voc_to_yolo(voc_root)

    # Build image list file for the test split
    test_txt = os.path.join(voc_root, "ImageSets", "Main", "test.txt")
    img_list = os.path.join(voc_root, "test_images.txt")
    with open(test_txt) as f:
        ids = [l.strip() for l in f if l.strip()]
    with open(img_list, "w") as f:
        for i in ids:
            f.write(os.path.abspath(os.path.join(voc_root, "JPEGImages", i + ".jpg")) + "\n")

    yaml_path = os.path.join(PASCAL_ROOT, "voc2007_test.yaml")
    cfg = dict(
        path=os.path.abspath(voc_root),
        val="JPEGImages",       # ultralytics will look for labels/ alongside
        train="JPEGImages",
        nc=20,
        names=PASCAL_CLASS_NAMES,
    )
    with open(yaml_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    print(f"  Written Pascal yaml → {yaml_path}")
    return yaml_path


# ===================================================================
#  CALIBRATION DATASET (for quantization search)
# ===================================================================
from ultralytics.data.dataset import YOLODataset


class CalibDataset(Dataset):
    """Wraps a YOLO-format image directory into (image_tensor, label) pairs."""

    def __init__(self, img_path, data_yaml, imgsz=640):
        with open(data_yaml) as f:
            data = yaml.safe_load(f)
        self.dataset = YOLODataset(
            img_path=img_path, data=data, imgsz=imgsz,
            augment=False, rect=False, cache=False,
        )

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


# ===================================================================
#  TORCHVISION EVAL HELPERS (pycocotools-based)
# ===================================================================
def _get_coco_gt(annot_json):
    """Return a pycocotools COCO ground-truth object."""
    from pycocotools.coco import COCO as COCOGT
    return COCOGT(annot_json)


def _build_torchvision_dataloader(dataset_name, batch_size):
    """
    Return (dataloader, coco_gt_or_None, dataset_info_dict) for torchvision
    models. These models expect un-normalised [0,1] tensors and produce
    list[dict] outputs.
    """
    import torchvision.transforms as T
    from torchvision.datasets import CocoDetection, VOCDetection
    from torch.utils.data import DataLoader
    from PIL import Image

    # Ensure RGB conversion (some COCO images are grayscale/RGBA)
    class RGBToTensor:
        def __call__(self, img):
            img = img.convert("RGB")
            return T.functional.to_tensor(img)

    transform = RGBToTensor()

    if dataset_name == "coco":
        img_dir = os.path.join(COCO_ROOT, "images", "val2017")
        ann_file = os.path.join(COCO_ROOT, "annotations", "annotations",
                                "instances_val2017.json")
        ds = CocoDetection(img_dir, ann_file, transform=transform)
        coco_gt = _get_coco_gt(ann_file)
        info = dict(name="coco", nc=80)
    else:  # pascal
        voc_root = os.path.join(PASCAL_ROOT, "VOC2007")
        ds = VOCDetection(root=".", year="2007", image_set="test",
                          download=False, transform=transform)
        coco_gt = None
        info = dict(name="pascal", nc=20)

    # Custom collate — images have different sizes
    def collate_fn(batch):
        return list(zip(*batch))

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=WORKERS, collate_fn=collate_fn)
    return loader, coco_gt, info


def eval_torchvision_coco(model, dataloader, coco_gt, device):
    """
    Run torchvision detection model on COCO val, return mAP via pycocotools.

    IMPORTANT: torchvision pretrained detection models output labels that are
    the *original* COCO category IDs (sparse, 1-90), NOT contiguous indices.
    So label=1 → person, label=44 → bottle, etc. We pass them straight
    through to pycocotools without any remapping.
    """
    from pycocotools.cocoeval import COCOeval
    model.eval()
    model.to(device)

    # CocoDetection stores the ordered list of image IDs as dataset.ids
    dataset = dataloader.dataset
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

            boxes  = out["boxes"].cpu()
            scores = out["scores"].cpu()
            labels = out["labels"].cpu()

            for b, s, l in zip(boxes, scores, labels):
                x1, y1, x2, y2 = b.tolist()
                results.append({
                    "image_id":    image_id,
                    "category_id": l.item(),  # already a COCO category ID
                    "bbox":        [x1, y1, x2 - x1, y2 - y1],
                    "score":       s.item(),
                })
        sample_idx += len(images)

    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s  ({len(results)} detections)")

    if len(results) == 0:
        print("  [WARN] No detections produced!")
        return {"map50": 0.0, "map": 0.0}

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return {
        "map":   coco_eval.stats[0],   # mAP@0.50:0.95
        "map50": coco_eval.stats[1],   # mAP@0.50
    }


def eval_torchvision_pascal(model, device):
    """
    Evaluate a torchvision model on Pascal VOC 2007 test via a simple
    per-class AP computation (IoU=0.5), following the VOC protocol.
    """
    import torchvision.transforms as T
    from torchvision.datasets import VOCDetection
    import numpy as np

    transform = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.ToTensor(),
    ])
    voc_ds = VOCDetection(root=".", year="2007", image_set="test",
                          download=False, transform=transform)

    model.eval()
    model.to(device)

    # COCO-pretrained torchvision models use COCO label ids.
    # We need to map COCO labels → VOC class names.
    COCO_TO_VOC = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorbike", 5: "aeroplane",
        6: "bus", 7: "train", 8: "truck", 9: "boat", 16: "bird", 17: "cat",
        18: "dog", 19: "horse", 20: "sheep", 21: "cow", 44: "bottle",
        62: "chair", 63: "sofa", 64: "pottedplant", 65: "bed",
        67: "diningtable", 70: "tvmonitor",
    }
    voc_idx = {c: i for i, c in enumerate(PASCAL_CLASS_NAMES)}

    all_dets = {c: [] for c in PASCAL_CLASS_NAMES}   # class -> list of (img_idx, score, box)
    all_gts  = {c: [] for c in PASCAL_CLASS_NAMES}   # class -> list of (img_idx, box)

    print(f"  Evaluating torchvision model on {len(voc_ds)} Pascal VOC images ...")
    t0 = time.time()
    for img_idx in range(len(voc_ds)):
        img_tensor, target = voc_ds[img_idx]
        img_tensor = img_tensor.to(device)
        with torch.no_grad():
            out = model([img_tensor])[0]

        # Parse GT
        annot = target["annotation"]
        objects = annot.get("object", [])
        if not isinstance(objects, list):
            objects = [objects]
        for obj in objects:
            cls_name = obj["name"]
            if cls_name not in voc_idx:
                continue
            diff = int(obj.get("difficult", 0))
            if diff:
                continue
            bb = obj["bndbox"]
            box = [float(bb["xmin"]), float(bb["ymin"]),
                   float(bb["xmax"]), float(bb["ymax"])]
            all_gts[cls_name].append((img_idx, box))

        # Parse predictions
        for b, s, l in zip(out["boxes"].cpu(), out["scores"].cpu(), out["labels"].cpu()):
            coco_id = l.item()
            if coco_id not in COCO_TO_VOC:
                continue
            cls_name = COCO_TO_VOC[coco_id]
            all_dets[cls_name].append((img_idx, s.item(), b.tolist()))

    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s")

    # VOC-style AP@0.5 per class
    def voc_ap(rec, prec):
        """VOC 2010+ style (area under interpolated PR curve)."""
        mrec = np.concatenate(([0.0], rec, [1.0]))
        mpre = np.concatenate(([0.0], prec, [0.0]))
        for i in range(len(mpre) - 1, 0, -1):
            mpre[i - 1] = max(mpre[i - 1], mpre[i])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        return np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])

    def iou(a, b):
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        aa = (a[2]-a[0])*(a[3]-a[1])
        ab = (b[2]-b[0])*(b[3]-b[1])
        return inter / (aa + ab - inter + 1e-9)

    aps = []
    for cls_name in PASCAL_CLASS_NAMES:
        dets = sorted(all_dets[cls_name], key=lambda x: -x[1])
        gts  = all_gts[cls_name]
        n_pos = len(gts)
        if n_pos == 0:
            continue

        # group gt by image
        gt_by_img = defaultdict(list)
        for (img_i, box) in gts:
            gt_by_img[img_i].append({"box": box, "used": False})

        tp = np.zeros(len(dets))
        fp = np.zeros(len(dets))
        for di, (img_i, score, box) in enumerate(dets):
            best_iou = 0.0
            best_gt  = -1
            for gi, g in enumerate(gt_by_img.get(img_i, [])):
                ov = iou(box, g["box"])
                if ov > best_iou:
                    best_iou = ov
                    best_gt = gi
            if best_iou >= 0.5 and best_gt >= 0 and not gt_by_img[img_i][best_gt]["used"]:
                tp[di] = 1
                gt_by_img[img_i][best_gt]["used"] = True
            else:
                fp[di] = 1

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        rec  = tp_cum / n_pos
        prec = tp_cum / (tp_cum + fp_cum)
        ap = voc_ap(rec, prec)
        aps.append(ap)
        print(f"    {cls_name:<20s}  AP@0.5 = {ap:.4f}")

    mAP = np.mean(aps) if aps else 0.0
    print(f"  mAP@0.5 = {mAP:.4f}")
    return {"map50": mAP}


# ===================================================================
#  MODEL LOADERS
# ===================================================================
def load_ultralytics_model(model_key):
    """Return an Ultralytics YOLO() wrapper."""
    from ultralytics import YOLO
    name_map = {
        "yolov8n":  "yolov8n.pt",
        "yolo26n":  "yolo26n.pt",
        "rtdetr-s":  "rtdetr-l.pt",   # ultralytics ships rtdetr-l; closest small
    }
    # Try exact match, fallback to key + .pt
    pt_name = name_map.get(model_key, model_key + ".pt")
    print(f"  Loading Ultralytics model: {pt_name}")
    model = YOLO(pt_name)
    return model


def load_torchvision_model(model_key, device):
    """Return a torchvision detection model (already eval-mode)."""
    import torchvision

    if model_key == "mobilenet_ssd_v2":
        print("  Loading torchvision SSDLite320 + MobileNetV3-Large ...")
        model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(
            weights="DEFAULT"
        )
    elif model_key == "fasterrcnn_mobilenet":
        print("  Loading torchvision Faster R-CNN + MobileNetV3-Large-FPN ...")
        model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
            weights="DEFAULT"
        )
    else:
        raise ValueError(f"Unknown torchvision model key: {model_key}")

    model.eval()
    model.to(device)
    return model


ULTRALYTICS_MODELS = {"yolov8n", "yolo26n", "rtdetr-s"}
TORCHVISION_MODELS = {"mobilenet_ssd_v2", "fasterrcnn_mobilenet"}
ALL_MODELS = ULTRALYTICS_MODELS | TORCHVISION_MODELS


# ===================================================================
#  ULTRALYTICS EVAL (with optional quantization)
# ===================================================================
def _print_results(results, tag, model_name, device, elapsed):
    print("\n" + "=" * 60)
    print(f"  {tag}")
    print(f"  Model            : {model_name}")
    print(f"  Device           : {device}")
    print(f"  Total time       : {elapsed:.1f}s")
    print("-" * 60)
    print(f"  mAP@0.50         : {results.box.map50:.4f}")
    print(f"  mAP@0.50:0.95    : {results.box.map:.4f}")
    print(f"  Precision        : {results.box.mp:.4f}")
    print(f"  Recall           : {results.box.mr:.4f}")
    print("=" * 60)


def run_ultralytics_baseline(model_key, data_yaml, device):
    model = load_ultralytics_model(model_key)
    model.to(device)
    t0 = time.time()
    results = model.val(
        data=data_yaml, imgsz=IMG_SIZE, batch=BATCH_SIZE,
        device=device, workers=WORKERS, split="val", verbose=True,
    )
    _print_results(results, f"FP32 baseline", model_key, device, time.time() - t0)
    return model, results


def run_ultralytics_quantized(model_key, data_yaml, calib_img_path, device):
    """
    Mirrors the user's evaluate_bisected() flow:
      1. Load model, get inner nn.Module
      2. Build calibration data, compute FP32 baseline output
      3. Run greedy_sensitivity from sensitivity_search
      4. Swap quantised inner model back, run model.val()
    """
    if not HAS_LOFLOAT:
        print("  [SKIP] LoFloat not available — cannot run quantization.")
        return None

    model = load_ultralytics_model(model_key)
    inner = model.model.to(device)
    inner.eval()

    # Calibration subset
    dataset = CalibDataset(calib_img_path, data_yaml, imgsz=IMG_SIZE)
    indices = torch.randperm(len(dataset))[:256]
    subset  = Subset(dataset, indices)
    loader  = DataLoader(subset, batch_size=64, shuffle=False)
    val_data, _ = next(iter(loader))
    val_data = val_data.to(device)

    # FP32 baseline outputs
    with torch.no_grad():
        baseline_out = inner(val_data)

    def _detach(x):
        if isinstance(x, torch.Tensor):
            return x.detach().to(device)
        elif isinstance(x, dict):
            return {k: _detach(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(_detach(v) for v in x)
        return x
    baseline = _detach(baseline_out)

    def loss_fn(preds):
        def _mse(p, b):
            if isinstance(b, torch.Tensor):
                return F.mse_loss(p.float(), b.float())
            elif isinstance(b, dict):
                return sum(_mse(p[k], b[k]) for k in b)
            elif isinstance(b, (list, tuple)):
                return sum(_mse(pi, bi) for pi, bi in zip(p, b))
            return torch.tensor(0.0, device=device)
        return _mse(preds, baseline)

    def eval_fn(m, data):
        m.eval().to(device)
        with torch.no_grad():
            preds = m(val_data)
        return -loss_fn(preds)

    baseline_score = eval_fn(inner, None)
    accuracy_target = 0.01
    print(f"  Baseline score: {baseline_score:.4f}, accuracy target: {accuracy_target}")

    print("  Running greedy_sensitivity quantization search ...")
    quantized_inner = ss.greedy_sensitivity(
        model=inner,
        sensitivity_measure="hessian",
        data=dataset,
        loss_fn=loss_fn,
        eval_fn=eval_fn,
        accuracy_target=accuracy_target,
        bs=[6, 5, 4, 3, 2],
        n_samples=128,
        device=device,
    )

    # Swap back
    model.model = quantized_inner

    t0 = time.time()
    results = model.val(
        data=data_yaml, imgsz=IMG_SIZE, batch=BATCH_SIZE,
        device=device, workers=WORKERS, split="val", verbose=True,
    )
    _print_results(results, f"Quantized (LoFloat)", model_key, device, time.time() - t0)
    return results


# ===================================================================
#  TORCHVISION EVAL (with optional quantization)
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


def run_torchvision_quantized(model_key, dataset_name, device):
    """
    Apply the same LoFloat quantization flow to a torchvision model.
    Uses a small calibration set and MSE-to-baseline as the loss signal.
    """
    if not HAS_LOFLOAT:
        print("  [SKIP] LoFloat not available — cannot run quantization.")
        return None

    model = load_torchvision_model(model_key, device)

    # Build a small calibration batch from the dataset
    import torchvision.transforms as T

    if dataset_name == "coco":
        from torchvision.datasets import CocoDetection
        img_dir  = os.path.join(COCO_ROOT, "images", "val2017")
        ann_file = os.path.join(COCO_ROOT, "annotations", "annotations",
                                "instances_val2017.json")
        ds = CocoDetection(img_dir, ann_file, transform=T.ToTensor())
    else:
        from torchvision.datasets import VOCDetection
        ds = VOCDetection(root=".", year="2007", image_set="test",
                          download=False, transform=T.ToTensor())

    # Grab N calibration images (use smallest dimension crop to unify sizes)
    n_calib = 64
    indices = torch.randperm(len(ds))[:n_calib]
    calib_images = []
    for i in indices:
        img, _ = ds[int(i)]
        # Resize to 320×320 for SSDLite or 640 for others
        target_size = 320 if "ssd" in model_key else 640
        img = F.interpolate(img.unsqueeze(0), size=(target_size, target_size),
                            mode="bilinear", align_corners=False).squeeze(0)
        calib_images.append(img)
    calib_batch = torch.stack(calib_images).to(device)

    # FP32 baseline — torchvision models return list[dict] in eval mode.
    # Switch to train mode for consistent tensor outputs if possible,
    # otherwise wrap the loss around eval-mode outputs.
    model.eval()
    with torch.no_grad():
        # Convert batch to list of tensors (torchvision API)
        baseline_out = model([img for img in calib_batch])

    # Flatten all detection tensors for MSE comparison
    def _flatten_detections(outputs):
        """Concatenate all boxes+scores into a single tensor for MSE."""
        parts = []
        for o in outputs:
            if len(o["boxes"]) > 0:
                parts.append(o["boxes"].reshape(-1))
                parts.append(o["scores"].reshape(-1))
        if parts:
            return torch.cat(parts)
        return torch.zeros(1, device=device)

    baseline_flat = _flatten_detections(baseline_out).detach()

    def loss_fn(preds):
        pred_flat = _flatten_detections(preds)
        # Pad/truncate to match
        min_len = min(len(pred_flat), len(baseline_flat))
        return F.mse_loss(pred_flat[:min_len].float(), baseline_flat[:min_len].float())

    def eval_fn(m, data):
        m.eval().to(device)
        with torch.no_grad():
            preds = m([img for img in calib_batch])
        return -loss_fn(preds)

    baseline_score = eval_fn(model, None)
    accuracy_target = 0.01
    print(f"  Baseline score: {baseline_score:.4f}, accuracy target: {accuracy_target}")

    print("  Running greedy_sensitivity quantization search ...")
    quantized_model = ss.greedy_sensitivity(
        model=model,
        sensitivity_measure="hessian",
        data=ds,
        loss_fn=loss_fn,
        eval_fn=eval_fn,
        accuracy_target=accuracy_target,
        bs=[6, 5, 4, 3, 2],
        n_samples=n_calib,
        device=device,
    )

    print(lof.record_formats(quantized_model))

    # Full eval on quantized model
    if dataset_name == "coco":
        loader, coco_gt, _ = _build_torchvision_dataloader("coco", BATCH_SIZE)
        metrics = eval_torchvision_coco(quantized_model, loader, coco_gt, device)
    else:
        metrics = eval_torchvision_pascal(quantized_model, device)

    print(f"\n  [{model_key} / {dataset_name}] Quantized (LoFloat): {metrics}")
    return metrics


# ===================================================================
#  MAIN ORCHESTRATOR
# ===================================================================
def print_summary(all_results):
    """Print a final comparison table."""
    print("\n\n" + "=" * 80)
    print("  FINAL SUMMARY")
    print("=" * 80)
    header = f"  {'Model':<25s} {'Dataset':<10s} {'Variant':<12s} {'mAP@0.50':<12s} {'mAP@0.50:95':<12s}"
    print(header)
    print("-" * 80)
    for r in all_results:
        map50 = r.get("map50", "—")
        map95 = r.get("map", "—")
        if isinstance(map50, float):
            map50 = f"{map50:.4f}"
        if isinstance(map95, float):
            map95 = f"{map95:.4f}"
        print(f"  {r['model']:<25s} {r['dataset']:<10s} {r['variant']:<12s} {map50:<12s} {map95:<12s}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="Object Detection Quantization Benchmark")
    parser.add_argument("--models", nargs="+",
                        default=["yolov8n", "yolo26n", "mobilenet_ssd_v2", "rtdetr-s"],
                        choices=sorted(ALL_MODELS),
                        help="Models to benchmark")
    parser.add_argument("--datasets", nargs="+", default=["coco"],
                        choices=["coco", "pascal"],
                        help="Datasets to evaluate on")
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--img-size", type=int, default=IMG_SIZE)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--skip-quantization", action="store_true",
                        help="Only run FP32 baselines, skip LoFloat quantization")
    parser.add_argument("--accuracy-target", type=float, default=0.01,
                        help="Accuracy target for greedy_sensitivity (default 0.01)")
    args = parser.parse_args()

    # Update module-level config from CLI args
    _update_config(args.batch_size, args.img_size, args.workers, args.device)

    # ---- Prepare datasets ----
    data_yamls = {}
    calib_img_paths = {}

    if "coco" in args.datasets:
        print("\n=== Preparing COCO val2017 ===")
        data_yamls["coco"] = prepare_coco()
        calib_img_paths["coco"] = os.path.join(COCO_ROOT, "images", "val2017")

    if "pascal" in args.datasets:
        print("\n=== Preparing Pascal VOC 2007 test ===")
        data_yamls["pascal"] = prepare_pascal()
        calib_img_paths["pascal"] = os.path.join(PASCAL_ROOT, "VOC2007", "JPEGImages")

    all_results = []

    # ---- Run benchmarks ----
    for model_key in args.models:
        for ds_name in args.datasets:
            print(f"\n{'#' * 60}")
            print(f"  MODEL: {model_key}  |  DATASET: {ds_name}")
            print(f"{'#' * 60}")

            is_ultralytics = model_key in ULTRALYTICS_MODELS
            is_torchvision = model_key in TORCHVISION_MODELS

            # --- FP32 baseline ---
            if is_ultralytics:
                _, res_fp32 = run_ultralytics_baseline(
                    model_key, data_yamls[ds_name], DEVICE)
                all_results.append(dict(
                    model=model_key, dataset=ds_name, variant="FP32",
                    map50=res_fp32.box.map50, map=res_fp32.box.map,
                ))
            else:
                _, metrics_fp32 = run_torchvision_baseline(
                    model_key, ds_name, DEVICE)
                all_results.append(dict(
                    model=model_key, dataset=ds_name, variant="FP32",
                    **metrics_fp32,
                ))

            # --- Quantized ---
            if not args.skip_quantization:
                if is_ultralytics:
                    res_q = run_ultralytics_quantized(
                        model_key, data_yamls[ds_name],
                        calib_img_paths[ds_name], DEVICE)
                    if res_q is not None:
                        all_results.append(dict(
                            model=model_key, dataset=ds_name, variant="LoFloat",
                            map50=res_q.box.map50, map=res_q.box.map,
                        ))
                else:
                    metrics_q = run_torchvision_quantized(
                        model_key, ds_name, DEVICE)
                    if metrics_q is not None:
                        all_results.append(dict(
                            model=model_key, dataset=ds_name, variant="LoFloat",
                            **metrics_q,
                        ))

    # ---- Summary ----
    print_summary(all_results)


if __name__ == "__main__":
    main()