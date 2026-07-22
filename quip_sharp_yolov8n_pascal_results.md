# QuIP# fixed-codebook (E8 lattice) VQ — yolov8n / pascal

- Model: `yolov8n` (Ultralytics)
- Dataset: pascal val (120 images)
- Device: cpu
- Quantizer: `quip_sharp_vq.py` — RHT incoherence + E8/E8P lattice VQ. `plain` = nearest-rounding; `ldlq` = QuIP# BlockLDLQ (calibration-Hessian error feedback, 32 calib images). First/last layer and layers < 8192 weights left in FP. No fine-tuning.

| config | description | idx bpw | w_relerr | mAP@.5:.95 | mAP@.5 | rel mAP drop |
|---|---|---|---|---|---|---|
| `fp32` | FP32 baseline | 32.000 | 0.0000 | 0.5157 | 0.6722 | +0.000 |
| `e8p2` | plain E8P 2-bit | 2.000 | 0.4054 | 0.0000 | 0.0000 | +1.000 |
| `rvq4` | plain resid-E8P 4-bit | 4.000 | 0.2759 | 0.0000 | 0.0000 | +1.000 |
| `rvq6` | plain resid-E8P 6-bit | 6.000 | 0.2452 | 0.0000 | 0.0000 | +1.000 |
| `rvq8` | plain resid-E8P 8-bit | 8.000 | 0.2344 | 0.0000 | 0.0000 | +1.000 |
| `ldlq2` | BlockLDLQ E8P 2-bit | 2.000 | 0.5297 | 0.0000 | 0.0000 | +1.000 |
| `ldlq4` | BlockLDLQ resid-E8P 4-bit | 4.000 | 0.3899 | 0.0000 | 0.0000 | +1.000 |
| `ldlq6` | BlockLDLQ resid-E8P 6-bit | 6.000 | 0.3545 | 0.0075 | 0.0134 | +0.985 |
| `ldlq8` | BlockLDLQ resid-E8P 8-bit | 8.000 | 0.3420 | 0.0320 | 0.0425 | +0.938 |

**Lossless crossover** (|rel mAP drop| ≤ 1%): plain ≈ >8 bpw, BlockLDLQ ≈ >8 bpw.
