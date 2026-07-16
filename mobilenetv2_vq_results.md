# MobileNetV2 vector-quantization results

- Model: torchvision `mobilenet_v2` (ImageNet1K_V1), 3,504,872 params
- Quantized layers: 26 Conv2d/Linear (3,380,544 params = 96.5% of the model; weights below `min_numel=8192` left fp32)
- Eval inputs: 128 COCO val2017 images, ImageNet-normalized
- k-means: `n_iters=25`, `n_init=1`, K=256 (8-bit indices)
- Per-channel **power-of-two** scaling (`scale_dim=0`): each output channel is divided by the nearest power of two to its peak magnitude before clustering and the scale folded back losslessly on decode.

Fidelity is measured **against the fp32 model** (no ImageNet labels available): `top1_agree` = fraction of inputs with unchanged predicted class; `logit_cos` = mean logit cosine similarity; `logit_mse` = mean squared logit error. `w_relerr` = aggregate relative L2 weight error.

| config | scheme | bpw | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|---|
| `vq2` | VQ  d=4 K=256  (~2 bpw) | 2.16 | 0.327 | 0.016 | 0.372 | 9.00 |
| `vq4` | VQ  d=2 K=256  (~4 bpw) | 4.09 | 0.096 | 0.625 | 0.912 | 1.44 |
| `pq2` | PQ  d=8 M=2 K=256  (~2 bpw) | 2.28 | 0.323 | 0.000 | 0.312 | 9.61 |
| `pq4` | PQ  d=8 M=4 K=256  (~4 bpw) | 4.28 | 0.093 | 0.688 | 0.937 | 1.06 |
| `rvq2` | RVQ d=4 S=1 K=256  (~2 bpw) | 2.16 | 0.328 | 0.016 | 0.317 | 9.31 |
| `rvq4` | RVQ d=4 S=2 K=256  (~4 bpw) | 4.28 | 0.108 | 0.609 | 0.916 | 1.41 |
| `vq2_lof` | VQ  d=4 K=256 + LoFloat e4m3 centroids  (~2 bpw) | 2.09 | 0.331 | 0.008 | 0.315 | 10.90 |

## Takeaways

Pre-scaling baseline (no `scale_dim`): `vq2` top1_agree 0.008 / logit_cos 0.244; `vq4` 0.648 / 0.928.

- **2-bit stays broken.** Per-channel PO2 scaling lifts `vq2` logit_cos 0.24→0.37 (outputs less decorrelated from fp32) but top1_agree is still ~random (chance ≈ 0.001). All ~2 bpw schemes collapse; `pq2` is the worst (0.000).
- **4-bit is the practical floor**, and PQ is the best scheme there: `pq4` = 0.688 top1_agree, logit_cos 0.937.
- Scaling *raises* aggregate `w_relerr` (0.302→0.327 at 2-bit) because global k-means spends codewords on the high-norm channels that dominate the L2 norm; scaling instead equalizes *relative* error across channels. `w_relerr` is therefore a poor accuracy proxy here.
- Why PO2 scaling underdelivers on MobileNetV2: (1) VQ's global k-means already self-adapts codeword allocation by inertia; (2) PO2 is coarse (equalizes only to within √2); (3) 2-bit ≈ 4 levels/weight is below this model's capacity floor. Loss-aware (Hessian) importance weighting or QAT would be the next levers below 4 bpw.
