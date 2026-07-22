# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Project Is

This repo is the **experiments / benchmarking harness** for the [LoFloat](../LoFloat) library — it does *not* contain the float-format simulator itself. LoFloat (installed as the `LoFloat` Python package, source at `/home/spuser/LoFloat/`) provides the C++/CUDA quantized float kernels and the `LoF_Linear` / `LoF_Conv2d` / batch-norm / SiLU layers. Code here drives those layers through:

1. **Sensitivity analysis** of model layers under different float formats (Hessian-trace via `pyhessian`, or noise-injection).
2. **Mantissa / exponent / accumulator search** to find per-layer mixed-precision configs that hit an accuracy budget.
3. **End-to-end object-detection benchmarks** (COCO val2017, Pascal VOC 2012) on YOLOv8/26, RT-DETR, MobileNet-SSD, Faster R-CNN, and DEIMv2-S+DINOv3.
4. **GPTQ** weight quantization (modified to use `lof.mantissa_quantize`) and **Hadamard rotation** for activation outlier suppression.
5. **Strassen-viability grading**: per-layer noise injection that determines whether a layer tolerates Strassen-style (K²) accumulation noise, dot-product (K) noise, or needs exact accumulation.

## Repo Layout

- `sensitivity_search.py` — top-level entry points: `bisection_sensitivity()` (older), `greedy_sensitivity()` (current). The greedy search order is **exponent → accumulator → slack → mantissa → accumulator (2nd pass) → BN+SiLU replacement → Strassen grading**. First/last `nn.Linear`/`nn.Conv2d` are kept in FP. Also contains `replace_batchnorm2d()` (swap to L1/Linf/FISR/PWL variants), `replace_silu()`, and `apply_hadamard_to_weights()` (block-diagonal FWHT pre-rotation of contraction axes).
- `sensitivities.py` — calibration utilities: `find_range()`, `find_exp_bits_and_bias()`, `find_batchnorm_scales()`, `hess_sensitivity()` (pyhessian), `noise_sensitivity_full()`, and `quantize_weights_with_gptq()`.
- `gptq.py` — modified GPTQ (originally from the GPTQ repo) that calls `lof.mantissa_quantize` instead of integer quantization.
- `lo_gemm_test.py` — pytest suite for the `LoF_Linear`/`LoF_Conv2d`/`L1BatchNorm` layers and `lof_gemm` correctness (includes layout-convention diagnostics for the GEMM transpose convention).
- `sil_exp.py` — small sanity script around SiLU sensitivity (`x·f'/f`).
- `YOLO_test/yolo_test.py` — the main object-detection benchmark CLI; downloads COCO/VOC, runs FP32 baselines, then `greedy_sensitivity` per accuracy target.

## Common Commands

**Run the layer correctness suite:**
```bash
pytest lo_gemm_test.py -v
# or a single test
pytest lo_gemm_test.py::TestLoFSwapCorrectness::test_lof_gemm_layout_convention -v
```

**Run the detection benchmark** (from `YOLO_test/`):
```bash
cd YOLO_test
python yolo_test.py --models yolov8n yolo26n mobilenet_ssd_v2 rtdetr-n deimv2_dinov3_s \
                    --datasets coco pascal --device cuda --batch-size 16
# Skip quantization (FP32 baselines only):
python yolo_test.py --models yolov8n --datasets coco --skip-quantization
# Pick accuracy targets the search must hit (relative mAP drop):
python yolo_test.py --models yolov8n --accuracy-targets 0.01 0.05
```
The bench downloads `val2017.zip`, COCO annotations, and `VOCtrainval_11-May-2012.tar` on first run into `YOLO_test/coco/` and `YOLO_test/VOCdevkit/`.

## Architecture Notes

- **Everything routes through `lof.lofloatify(model)`**, which walks an `nn.Module` and swaps `Linear`/`Conv2d` for their LoF equivalents. `skip_layer_names` is used to keep the first and last layers in FP — every search routine here computes that set from the *original* model before lofloatifying.
- **Per-layer precision is set via dicts of layer name → bits**, then applied with `lof.set_mantissa_fields(...)`, `lof.set_exponent_fields(...)`, `lof.set_accumulation_precisions(...)`. The search routines maintain `w_weights`, `w_activ`, `w_bias`, `weights_exp`, `activ_exp`, `bias_exp`, and `accum_precs` dicts in lockstep.
- **`eval_fn(model, data)` returns a relative accuracy drop** (e.g. `(fp32_map - cur_map) / fp32_map`). The search compares `abs(eval_fn(...)) <= accuracy_target` to decide whether a more aggressive quantization is acceptable.
- **The greedy search has hardcoded sensitivity = 0** in `greedy_sensitivity` (lines around 650–660 in `sensitivity_search.py`) — the Hessian/noise paths are commented out and all layers are treated as equally sensitive. Re-enable by uncommenting the `hess_sensitivity`/`noise_sensitivity_full` block if you want true sensitivity-ordered search.
- **Hadamard rotation auto-picks block size** as the largest power-of-2 dividing the contraction dim, clamped by `max_block_size`. Layers whose largest pow2 divisor is below `min_block_size` (default 2) are skipped. Both weights *and* activations rotate at runtime via `module.hadamard_transform = True`.
- **DEIMv2 wrapping**: `DEIMv2SingleArgWrapper` adapts the two-arg DEIMv2 forward `(x, orig_target_sizes)` to the single-arg `forward(x)` that calibration/sensitivity hooks assume. Use `_unwrap_deimv2()` before any direct DEIMv2 call.
- **`lof_gemm` layout convention**: it expects `(M,K) @ (K,N)` (standard). The diagnostic tests in `lo_gemm_test.py::TestLoFSwapCorrectness` exist because of a prior transpose bug in the swap path — keep them green when touching `_lof_linear` or call sites that pass `weight.t()`.

## External Dependencies

- The `LoFloat` package lives at `/home/spuser/LoFloat/` and must be `pip install -e .`'d there first. See its own `CLAUDE.md` for build details (CMake + CUDA + xsimd + CUTLASS).
- `pyhessian` (Hessian-trace sensitivity), `ultralytics` (YOLOv8/26, RT-DETR), `torchvision` (SSD, Faster R-CNN, VOC/COCO datasets), `pycocotools` (mAP eval), `huggingface_hub` (DEIMv2 weights). The DEIMv2 path also clones `Intellindust-AI-Lab/DEIMv2` into `_deimv2_repo/` on demand.
- `VOCtrainval_11-May-2012.tar` is checked into the repo root as a fallback when the upstream Pascal mirrors are flaky.
