"""
QuIP# (fixed E8-lattice codebook) — full detection-accuracy benchmark & bit sweep
=================================================================================

This is the "run at the end that measures full accuracy on pascal or coco"
requested for the QuIP# experiment.  It reuses ``yolo_test.py``'s dataset
preparation and mAP evaluation, but instead of the LoFloat scalar-float greedy
search it quantizes the model's Conv2d/Linear weights with the fixed-codebook
E8-lattice vector quantizer in ``quip_sharp_vq.py`` and then runs the *full*
COCO val2017 / Pascal VOC 2012-val mAP evaluation.

Two quantizers are compared:
  * **plain**  — RHT incoherence + (residual) E8P nearest-rounding, no Hessian.
  * **ldlq**   — QuIP#'s BlockLDLQ: same codebook but with calibration-Hessian
                 error-feedback rounding (needs a calibration pass).

``--sweep`` runs both at 2 / 4 / 6 / 8 bits-per-weight so you can read off the
bit budget at which mAP returns to the FP32 baseline (the "lossless crossover").

Because QuIP# quantization is applied as fake-quant (weights are quantized then
dequantized back to FP so the model runs on the stock ultralytics/torch path),
no custom kernel is needed to measure end-to-end accuracy.

Examples
--------
    cd YOLO_test
    # full bit sweep (plain + LDLQ, 2/4/6/8 bpw) on YOLOv8n, Pascal VOC val:
    python quip_sharp_yolo.py --model yolov8n --dataset pascal --sweep

    # quick smoke test on 200 images:
    python quip_sharp_yolo.py --dataset coco --max-images 200 --configs e8p2 ldlq2

    # pick individual configs:
    python quip_sharp_yolo.py --dataset pascal --configs fp32 rvq4 ldlq4
"""

import argparse
import os
import sys
import time

# make both the repo root (for quip_sharp_vq) and this dir (for yolo_test) importable
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
sys.path.insert(0, _ROOT)
os.chdir(_HERE)          # yolo_test.py uses paths relative to YOLO_test/

import torch
from torch.utils.data import DataLoader, Subset

import quip_sharp_vq as qsvq
import yolo_test as yt


# ---- named QuIP# configs (label -> QuipConfig kwargs + description) --------
# bpw = stages * 2 (each E8P stage is 2 bits/weight over an 8-dim block).
def _spec(stages, ldlq, desc):
    return dict(method=("rvq" if stages > 1 else "e8p"), stages=stages,
                use_rht=True, use_ldlq=ldlq, desc=desc)

CONFIG_ZOO = {
    # plain (nearest-rounding) --------------------------------------------
    "e8p2":      _spec(1, False, "plain E8P 2-bit"),
    "rvq4":      _spec(2, False, "plain resid-E8P 4-bit"),
    "rvq6":      _spec(3, False, "plain resid-E8P 6-bit"),
    "rvq8":      _spec(4, False, "plain resid-E8P 8-bit"),
    "e8p2_norht":dict(method="e8p", stages=1, use_rht=False, use_ldlq=False,
                      desc="plain E8P 2-bit (no RHT)"),
    "e8lattice": dict(method="e8lattice", stages=1, use_rht=True, use_ldlq=False,
                      lattice_rms=1.0, desc="raw E8 lattice (RHT)"),
    # BlockLDLQ (Hessian error-feedback) ----------------------------------
    "ldlq2":     _spec(1, True,  "BlockLDLQ E8P 2-bit"),
    "ldlq4":     _spec(2, True,  "BlockLDLQ resid-E8P 4-bit"),
    "ldlq6":     _spec(3, True,  "BlockLDLQ resid-E8P 6-bit"),
    "ldlq8":     _spec(4, True,  "BlockLDLQ resid-E8P 8-bit"),
}

SWEEP = ["fp32", "e8p2", "rvq4", "rvq6", "rvq8", "ldlq2", "ldlq4", "ldlq6", "ldlq8"]


def _eval_map(model, dataset, data_yaml, device, max_images):
    model.to(device)
    if dataset == "pascal":
        m = yt.eval_ultralytics_pascal(model, device, max_images=max_images)
        return m["map"], m["map50"]
    else:
        res = model.val(data=data_yaml, imgsz=yt.IMG_SIZE, batch=yt.BATCH_SIZE,
                        device=device, workers=yt.WORKERS, split="val",
                        verbose=False, plots=False)
        return res.box.map, res.box.map50


def _build_calib(dataset, data_yaml, device, n_samples=128, batch_size=16):
    """A list of image-tensor batches for the LDLQ Hessian calibration pass."""
    if dataset == "pascal":
        calib_dir = os.path.join(yt.PASCAL_ROOT, "VOCdevkit", "VOC2012", "images", "val")
    else:
        calib_dir = os.path.join(yt.COCO_ROOT, "images", "val2017")
    ds = yt.CalibDataset(calib_dir, data_yaml, imgsz=yt.IMG_SIZE)
    idx = torch.randperm(len(ds))[:n_samples]
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False)
    return [imgs.to(device) for imgs, _ in loader]


def main():
    ap = argparse.ArgumentParser(description="QuIP# fixed-codebook VQ full mAP benchmark")
    ap.add_argument("--model", default="yolov8n", choices=sorted(yt.ULTRALYTICS_MODELS))
    ap.add_argument("--dataset", default="pascal", choices=["coco", "pascal"])
    ap.add_argument("--configs", nargs="+", default=["fp32", "e8p2", "rvq4"],
                    help="fp32 + any of: " + ", ".join(CONFIG_ZOO))
    ap.add_argument("--sweep", action="store_true",
                    help="2/4/6/8-bpw sweep for both plain and LDLQ (overrides --configs)")
    ap.add_argument("--device", default=yt.DEVICE)
    ap.add_argument("--img-size", type=int, default=640)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-images", type=int, default=None,
                    help="subsample eval set for speed (default: full val set)")
    ap.add_argument("--calib-samples", type=int, default=128,
                    help="calibration images for the LDLQ Hessian")
    ap.add_argument("--min-numel", type=int, default=8192,
                    help="leave layers with fewer weights than this in FP")
    args = ap.parse_args()

    yt._update_config(args.batch_size, args.img_size, args.workers, args.device,
                      search_subset_size=100, accuracy_targets=[])

    configs = SWEEP if args.sweep else args.configs
    need_calib = any(CONFIG_ZOO.get(c, {}).get("use_ldlq") for c in configs if c != "fp32")

    if args.dataset == "pascal":
        print("\n=== Preparing Pascal VOC 2012 val ===")
        data_yaml = yt.prepare_pascal()
    else:
        print("\n=== Preparing COCO val2017 ===")
        data_yaml = yt.prepare_coco()

    device = args.device
    calib_inputs = None
    if need_calib:
        print(f"\n=== Building {args.calib_samples}-image calibration set (LDLQ) ===")
        calib_inputs = _build_calib(args.dataset, data_yaml, device, args.calib_samples,
                                    args.batch_size)

    results = []   # (label, desc, map, map50, bpw, relerr, rel_drop)

    # ---- FP32 baseline (always) ----
    print(f"\n{'#'*64}\n  FP32 baseline: {args.model} / {args.dataset}\n{'#'*64}")
    model = yt.load_ultralytics_model(args.model); model.to(device)
    t0 = time.time()
    fp32_map, fp32_map50 = _eval_map(model, args.dataset, data_yaml, device, args.max_images)
    print(f"  FP32 mAP@0.5:0.95 = {fp32_map:.4f}  mAP@0.5 = {fp32_map50:.4f}  ({time.time()-t0:.1f}s)")
    results.append(("fp32", "FP32 baseline", fp32_map, fp32_map50, 32.0, 0.0, 0.0))
    del model

    for label in configs:
        if label == "fp32":
            continue
        if label not in CONFIG_ZOO:
            print(f"  [skip] unknown config '{label}'"); continue
        spec = dict(CONFIG_ZOO[label])
        desc = spec.pop("desc")
        print(f"\n{'#'*64}\n  QuIP# config '{label}': {desc}\n{'#'*64}")

        # fresh model — quantization is destructive/in-place
        model = yt.load_ultralytics_model(args.model); model.to(device)
        inner = model.model
        cfg = qsvq.QuipConfig(min_numel=args.min_numel, **spec)
        t0 = time.time()
        report = qsvq.quantize_model_quip(inner, cfg, calib_inputs=calib_inputs, verbose=True)
        print(f"  quantized in {time.time()-t0:.1f}s")

        t0 = time.time()
        q_map, q_map50 = _eval_map(model, args.dataset, data_yaml, device, args.max_images)
        rel_drop = 0.0 if fp32_map == 0 else (fp32_map - q_map) / fp32_map
        print(f"  [{label}] mAP@0.5:0.95 = {q_map:.4f}  mAP@0.5 = {q_map50:.4f}  "
              f"(rel drop {rel_drop:+.3f}, {time.time()-t0:.1f}s)")
        results.append((label, desc, q_map, q_map50, report.index_bpw,
                        report.w_relerr, rel_drop))
        del model, inner

    # ---- summary ----
    print("\n" + "=" * 92)
    print(f"  QuIP# fixed-codebook VQ  |  {args.model}  |  {args.dataset}  |  {device}")
    if args.max_images:
        print(f"  (evaluated on {args.max_images} images)")
    print("=" * 92)
    print(f"  {'config':<12s} {'description':<28s} {'idx bpw':>8s} {'w_relerr':>9s} "
          f"{'mAP':>7s} {'mAP50':>7s} {'rel drop':>9s}")
    print("  " + "-" * 90)
    for label, desc, mp, mp50, bpw, relerr, drop in results:
        print(f"  {label:<12s} {desc:<28s} {bpw:8.3f} {relerr:9.4f} "
              f"{mp:7.4f} {mp50:7.4f} {drop:+9.3f}")
    print("=" * 92)

    # crossover: lowest bpw within 1% relative mAP of FP32, per family
    def _crossover(fam):
        rows = [(bpw, drop) for (lbl, _d, _m, _m5, bpw, _r, drop) in results
                if lbl != "fp32" and lbl in CONFIG_ZOO
                and CONFIG_ZOO[lbl].get("use_ldlq", False) == fam]
        ok = sorted([b for b, d in rows if abs(d) <= 0.01])
        return ok[0] if ok else None
    xo_plain, xo_ldlq = _crossover(False), _crossover(True)
    print(f"  lossless crossover (|rel mAP drop| <= 1%):  "
          f"plain = {xo_plain if xo_plain else '>8'} bpw   "
          f"BlockLDLQ = {xo_ldlq if xo_ldlq else '>8'} bpw")

    # ---- markdown dump ----
    out = os.path.join(_ROOT, f"quip_sharp_{args.model}_{args.dataset}_results.md")
    with open(out, "w") as f:
        f.write(f"# QuIP# fixed-codebook (E8 lattice) VQ — {args.model} / {args.dataset}\n\n")
        f.write(f"- Model: `{args.model}` (Ultralytics)\n")
        f.write(f"- Dataset: {args.dataset} val "
                f"({'full' if not args.max_images else str(args.max_images)+' images'})\n")
        f.write(f"- Device: {device}\n")
        f.write("- Quantizer: `quip_sharp_vq.py` — RHT incoherence + E8/E8P lattice VQ. "
                "`plain` = nearest-rounding; `ldlq` = QuIP# BlockLDLQ (calibration-Hessian "
                f"error feedback, {args.calib_samples} calib images). First/last layer and "
                f"layers < {args.min_numel} weights left in FP. No fine-tuning.\n\n")
        f.write("| config | description | idx bpw | w_relerr | mAP@.5:.95 | mAP@.5 | rel mAP drop |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for label, desc, mp, mp50, bpw, relerr, drop in results:
            f.write(f"| `{label}` | {desc} | {bpw:.3f} | {relerr:.4f} | "
                    f"{mp:.4f} | {mp50:.4f} | {drop:+.3f} |\n")
        f.write(f"\n**Lossless crossover** (|rel mAP drop| ≤ 1%): "
                f"plain ≈ {xo_plain if xo_plain else '>8'} bpw, "
                f"BlockLDLQ ≈ {xo_ldlq if xo_ldlq else '>8'} bpw.\n")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
