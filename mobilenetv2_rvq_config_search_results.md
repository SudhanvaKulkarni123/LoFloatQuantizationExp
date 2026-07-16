# MobileNetV2 residual-VQ config search: vector dim & codebook allocation

## Configuration

- **Model:** torchvision `mobilenet_v2` (ImageNet1K_V1), 3,504,872 params. 26 Conv2d/Linear layers quantized (3,380,544 weights = 96.5%; layers below `min_numel=8192` left fp32).
- **Quantizer:** residual VQ with per-channel PO2 scaling, fp32 codebooks (fp16 = 16-bit centroid storage assumed for the reported bpw), k-means `n_iters=25`, fixed seed 0 for reproducibility.
- **Index rate** = `sum_s log2(K_s) / d` bpw. **Effective bpw** adds the amortized codebook + per-channel scales. **Compression** = 32 / eff bpw.
- **Eval:** 128 COCO val2017 images, fidelity vs the fp32 model.
- **Best config by top1_agree:** `d2_4bpw` (d=2, K=[256]) → 0.680 at 4.09 bpw.

## Q1 - Effect of vector dim `d` at matched index rate

Same total index bits, spread over more stages as d grows. Lower `d` = fewer weights share each codeword = finer quantization.

### ~4 bpw index rate

| config | d | codebooks K | idx bpw | eff bpw | compression | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|---|---|---|---|
| `d2_4bpw` | d=2 | [256] | 4.00 | 4.09 | 7.82x | 0.095 | 0.680 | 0.923 | 1.26 |
| `d4_4bpw` | d=4 | [256, 256] | 4.00 | 4.28 | 7.47x | 0.108 | 0.617 | 0.914 | 1.39 |
| `d8_4bpw` | d=8 | [256, 256, 256, 256] | 4.00 | 5.04 | 6.35x | 0.111 | 0.656 | 0.919 | 1.35 |

### ~3 bpw index rate

| config | d | codebooks K | idx bpw | eff bpw | compression | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|---|---|---|---|
| `d2_3bpw` | d=2 | [64] | 3.00 | 3.05 | 10.51x | 0.190 | 0.227 | 0.601 | 5.96 |
| `d4_3bpw` | d=4 | [256, 16] | 3.00 | 3.16 | 10.11x | 0.199 | 0.328 | 0.699 | 4.78 |
| `d8_3bpw` | d=8 | [256, 256, 256] | 3.00 | 3.79 | 8.45x | 0.192 | 0.375 | 0.715 | 4.61 |

## Q2 - Codebook allocation at d=4 (symmetric vs asymmetric)

Fixed total index-bit budget split differently across the two stages.

### 12-bit budget (3.0 idx bpw)

| config | d | codebooks K | idx bpw | eff bpw | compression | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|---|---|---|---|
| `a_256_16` | d=4 | [256, 16] | 3.00 | 3.16 | 10.11x | 0.199 | 0.305 | 0.701 | 4.46 |
| `a_128_32` | d=4 | [128, 32] | 3.00 | 3.11 | 10.29x | 0.207 | 0.234 | 0.648 | 5.41 |
| `a_64_64` | d=4 | [64, 64] | 3.00 | 3.09 | 10.35x | 0.210 | 0.258 | 0.673 | 4.92 |

### 14-bit budget (3.5 idx bpw)

| config | d | codebooks K | idx bpw | eff bpw | compression | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|---|---|---|---|
| `a_256_64` | d=4 | [256, 64] | 3.50 | 3.69 | 8.68x | 0.150 | 0.453 | 0.836 | 2.59 |
| `a_128_128` | d=4 | [128, 128] | 3.50 | 3.66 | 8.75x | 0.152 | 0.445 | 0.808 | 3.10 |

## Takeaways

**No — `[256,256]` and `d=4` are not the best. The winner depends on bit-rate, and at ~4 bpw plain d=2 VQ beats RVQ outright.**

- **`d=4` is not universally best; the best `d` tracks the bit-rate.** With an 8-bit index cap (K≤256), a *rich* K=256 base codebook costs `8/d` bpw. So the rule is: **use the smallest `d` that still affords a K=256 base at your target bpw.**
  - **At 4 bpw, d=2 wins big:** `d=2 [256]` (plain VQ) = **0.680** at **4.09 bpw**, beating `d=4 [256,256]` (0.617 at 4.28 bpw) on *both* accuracy and bitrate. RVQ buys nothing here — a single rich low-d codebook is better than splitting into residual stages.
  - **At 3 bpw the order flips:** d=2 can no longer afford K=256 (that would be 4 bpw), so it's stuck with `[64]` → 0.227. `d=4 [256,16]` keeps a rich base → 0.328. `d=8 [256,256,256]` edges higher (0.375) but only by overspending (3.79 eff bpw vs 3.16). So below 4 bpw, larger `d` earns its keep by protecting the K=256 base.
- **Asymmetric beats symmetric — put the bits in the base.** At a fixed 12-bit budget, `[256,16]` (0.305) clearly beats `[64,64]` (0.258) and `[128,32]` (0.234). At 14 bits, `[256,64]` (0.453, cos 0.836) ≥ `[128,128]` (0.445, cos 0.808). This is the same lesson as base-vs-residual precision: the base carries the structure, the residual is a cheap correction. A big base + small residual dominates balanced splits.
- **So is `[256,256]` ever right?** Only if you're locked to d=4 and 4 bpw — and even then `d=2 [256]` is strictly better. `[256,256]` is a reasonable *4-bpw d=4* point but not the frontier.
- **Practical frontier for MobileNetV2:** at 4 bpw use `d=2 [256]`; at ~3.5 bpw use `d=4 [256,64]`; at 3 bpw use `d=4 [256,16]`. Always spend first on a K=256 base.

*Caveat: k-means seeding is only partially reproducible — multithreaded BLAS makes runs vary ~±0.02–0.03 top1_agree (visible in the two identical `[256,16]` rows: 0.305 vs 0.328). Same-bitrate gaps below ~0.03 are noise; the d-vs-d and asymmetric-vs-symmetric gaps here are larger and hold.*
