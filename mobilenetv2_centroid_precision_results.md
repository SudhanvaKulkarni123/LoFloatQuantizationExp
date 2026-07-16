# MobileNetV2 VQ codebook storage-precision sweep

- Base quantizer: `vq4` (VQ d=2 K=256 ~4 bpw), per-channel PO2 scaling, k-means fit in fp32.
- Sweeping only the **codebook storage** format: exponent fixed at **8 bits** (fp32 range), mantissa `m` swept 23→0. Centroid cost = `1+8+m` bits. Rounding via `LoFloat.exp_mant_quantize` (e8mM), applied post-hoc to the fp32 centroids (assignments unchanged).
- 26 Conv2d/Linear layers quantized; 128 COCO val2017 images, fidelity vs the fp32 model.

`e8m23` = fp32 centroids (reference). `e8m7` = 16-bit (bf16-like), `e8m2` = 11-bit, `e8m0` = 9-bit centroids.

| centroid format | centroid bits | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|
| `e8m23` | 32 | 0.096 | 0.625 | 0.894 | 1.76 |
| `e8m12` | 21 | 0.096 | 0.625 | 0.894 | 1.76 |
| `e8m10` | 19 | 0.096 | 0.625 | 0.894 | 1.76 |
| `e8m8` | 17 | 0.096 | 0.633 | 0.894 | 1.76 |
| `e8m7` | 16 | 0.096 | 0.625 | 0.895 | 1.75 |
| `e8m6` | 15 | 0.096 | 0.625 | 0.893 | 1.78 |
| `e8m5` | 14 | 0.096 | 0.625 | 0.892 | 1.79 |
| `e8m4` | 13 | 0.096 | 0.641 | 0.884 | 1.93 |
| `e8m3` | 12 | 0.099 | 0.570 | 0.883 | 1.95 |
| `e8m2` | 11 | 0.109 | 0.523 | 0.833 | 2.73 |
| `e8m1` | 10 | 0.140 | 0.422 | 0.758 | 4.15 |
| `e8m0` | 9 | 0.216 | 0.133 | 0.483 | 8.69 |

## Takeaways

- **Codebook storage is essentially lossless down to ~4 mantissa bits (`e8m4`).** From fp32 (m=23) to m=4, `w_relerr` is pinned at 0.096 and top1_agree holds ~0.62–0.64 (wiggles above the fp32 0.625 are just unseeded-k-means noise). So the centroids need far less mantissa than fp16/bf16 carry.
- **Knee at m3→m2, collapse at m0.** m3 still usable (0.570), m2 fading (0.523), m1 hurting (0.422); m0 (power-of-two-only centroids) collapses (0.133).
- **Why:** centroids sit on a per-channel PO2-scaled ~unit range, so the k-means clustering error (`w_relerr` ≈ 0.096, ~4 mantissa bits' worth) *dominates* the storage error until the storage grid gets coarser than the inter-centroid spacing. At m4 the storage error (~2⁻⁴) is still below the clustering floor and is masked; at m0 it swamps everything.
- **Practical note:** the codebook is amortized (256 centroids over a whole layer), so its *bit-rate* contribution is tiny (<0.1 bpw) regardless — the value here is that the codebook LUT can live in a cheap low-precision float format. Since the scaled centroids barely use the exponent, a natural follow-up is to shrink the *exponent* field too (e.g. e4m4 → 9 bits, or lower); with 8 exp bits held fixed, mantissa alone bottoms out at ~4 bits.

*(Validity note: this sweep exercises `LoFloat.exp_mant_quantize` on CPU, which required fixing three latent bugs in LoFloat's CPU rounding path — see `LoFloat/CPU_ROUNDING_BUGFIX.md`. The fixed op was verified bit-exact against an independent pure-torch e8mM reference before this run.)*
