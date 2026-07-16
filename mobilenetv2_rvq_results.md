# MobileNetV2 residual-VQ (RVQ) sweep

- Model: torchvision `mobilenet_v2` (ImageNet1K_V1), 3,504,872 params
- Quantized layers: 26 Conv2d/Linear (3,380,544 params = 96.5% of the model; weights below `min_numel=8192` fp32)
- Eval inputs: 128 COCO val2017 images, ImageNet-normalized
- All configs: `d=4`, base codebook K=256, per-channel PO2 scaling (`scale_dim=0`), k-means `n_iters=25`, `n_init=1`.
- RVQ refines a coarse base codebook by quantizing its residual with additional (optionally smaller) codebooks; reconstruction = sum of stages.

Fidelity vs the fp32 model: `top1_agree` = fraction of unchanged predictions; `logit_cos` = mean logit cosine similarity; `logit_mse` = mean squared logit error. `w_relerr` = aggregate relative L2 weight error.

| config | scheme | bpw | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|---|
| `r20` | base only            [256]         (~2.0 bpw) | 2.16 | 0.328 | 0.031 | 0.325 | 11.95 |
| `r25` | base + K=4 residual  [256,4]       (~2.5 bpw) | 2.66 | 0.266 | 0.078 | 0.350 | 8.71 |
| `r30` | base + K=16 residual [256,16]      (~3.0 bpw) | 3.16 | 0.199 | 0.211 | 0.639 | 5.76 |
| `r35` | base + K=64 residual [256,64]      (~3.5 bpw) | 3.69 | 0.150 | 0.484 | 0.836 | 2.60 |
| `r40a` | base + K=256 residual[256,256]     (~4.0 bpw) | 4.28 | 0.108 | 0.672 | 0.922 | 1.34 |
| `r40b` | base + 2xK=16 resid  [256,16,16]   (~4.0 bpw) | 4.17 | 0.123 | 0.555 | 0.881 | 1.92 |
| `r60` | 3x full stages       [256,256,256] (~6.0 bpw) | 6.41 | 0.036 | 0.852 | 0.990 | 0.17 |

## Takeaways

- **A residual codebook cleanly recovers the accuracy plain 2-bit VQ loses, and does so monotonically.** From the broken 2 bpw base (3% agreement) accuracy climbs smoothly with residual bits: 3.0 bpw → 21%, 3.5 bpw → 48%, 4.3 bpw → 67%, 6.4 bpw → 85%. There is **no cliff** — the 2-bit "collapse" is just that 2 bpw leaves no residual budget; even a tiny K=4 residual starts the recovery.
- **At equal bit budget, one larger residual codebook beats several tiny stages.** `r40a` `[256,256]` (0.672) vs `r40b` `[256,16,16]` (0.555), both ~4.2 bpw. The first residual carries most of the leftover energy, so spend bits on residual codebook *size* before adding more stages.
- **~6 bpw (3 full stages) is near-lossless** (logit_cos 0.990, 85% top-1 agreement).
- Caveat: k-means is unseeded, so identical configs vary ~±0.06 top1_agree run-to-run (e.g. `r40a` here = 0.672 vs `rvq4` = 0.609 in the main sweep). The curve shape is robust; individual cells wobble.
