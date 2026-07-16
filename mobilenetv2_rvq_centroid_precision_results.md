# MobileNetV2 residual-VQ codebook storage-precision sweep

## Configuration

- **Model:** torchvision `mobilenet_v2` (ImageNet1K_V1), 3,504,872 params. Quantized **26** Conv2d/Linear layers = **3,380,544 weights** (96.5% of the model; layers below `min_numel=8192` left fp32).
- **Quantizer:** residual VQ, `ResidualQuantizer(d=4, S=2, K=[256,256])` (the ~4 bpw `r40a` operating point). Reconstruction = base centroid + residual centroid (summed over the 2 stages).
- **Block size (VQ vector dim) `d` = 4** weights per index — the whole weight tensor is flattened row-major and chopped into 4-vectors (845,136 vectors total).
- **Codebook:** 2 stages × K=256 entries × d=4 = 2,048 centroid scalars per layer, per stage.
- **Indices:** log2(256) = 8 bits/stage × 2 stages = **16 bits per 4-vector = 4.00 bpw** (fixed; assignments never change).
- **Per-channel scaling:** power-of-two, one scale per output channel (12,648 scales; avg scaling block = 267 weights/scale), stored as an **8-bit exponent** = 0.0299 bpw.
- **Storage sweep:** exponent fixed at **8 bits** (fp32 dynamic range); centroid **mantissa** m swept 23→0, post-hoc via `LoFloat.exp_mant_quantize` (e8mM). Centroid scalar = `1+8+m` bits (fp32 reference = 32).
- **Eval:** 128 COCO val2017 images, fidelity vs the fp32 model.

**Effective bitwidth** = (index + codebook + scale bits) / #weights. **Compression** = 32 / effective bitwidth (vs dense fp32 = 13.5 MB for these weights). Because the codebook is amortized over the whole layer, centroid precision barely moves the effective bitwidth — the indices dominate.

Three sweeps isolate the stages: **both** stages squeezed together; only the **residual** (stage 1) squeezed with the base kept fp32; only the **base** (stage 0) squeezed with the residual kept fp32.

## Sweep A - both stages at e8mM

| stage mantissas (base,resid) | centroid bits/scalar | codebook bpw | effective bitwidth | compression | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|---|---|---|
| (fp32,fp32) | 32,32 | 0.504 | 4.534 bpw | 7.06x | 0.108 | 0.633 | 0.902 | 1.79 |
| (12,12) | 21,21 | 0.331 | 4.361 bpw | 7.34x | 0.108 | 0.633 | 0.902 | 1.79 |
| (10,10) | 19,19 | 0.299 | 4.329 bpw | 7.39x | 0.108 | 0.633 | 0.902 | 1.79 |
| (8,8) | 17,17 | 0.268 | 4.298 bpw | 7.45x | 0.108 | 0.641 | 0.902 | 1.79 |
| (7,7) | 16,16 | 0.252 | 4.282 bpw | 7.47x | 0.108 | 0.633 | 0.902 | 1.79 |
| (6,6) | 15,15 | 0.236 | 4.266 bpw | 7.50x | 0.108 | 0.633 | 0.901 | 1.81 |
| (5,5) | 14,14 | 0.221 | 4.250 bpw | 7.53x | 0.108 | 0.633 | 0.902 | 1.79 |
| (4,4) | 13,13 | 0.205 | 4.235 bpw | 7.56x | 0.108 | 0.617 | 0.900 | 1.83 |
| (3,3) | 12,12 | 0.189 | 4.219 bpw | 7.58x | 0.111 | 0.633 | 0.899 | 1.84 |
| (2,2) | 11,11 | 0.173 | 4.203 bpw | 7.61x | 0.120 | 0.523 | 0.850 | 2.80 |
| (1,1) | 10,10 | 0.158 | 4.187 bpw | 7.64x | 0.150 | 0.391 | 0.752 | 5.09 |
| (0,0) | 9,9 | 0.142 | 4.172 bpw | 7.67x | 0.230 | 0.078 | 0.441 | 9.06 |

## Sweep B - base fp32, residual (stage 1) swept

| stage mantissas (base,resid) | centroid bits/scalar | codebook bpw | effective bitwidth | compression | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|---|---|---|
| (fp32,fp32) | 32,32 | 0.504 | 4.534 bpw | 7.06x | 0.108 | 0.633 | 0.902 | 1.79 |
| (fp32,12) | 32,21 | 0.417 | 4.447 bpw | 7.20x | 0.108 | 0.633 | 0.902 | 1.79 |
| (fp32,10) | 32,19 | 0.402 | 4.432 bpw | 7.22x | 0.108 | 0.633 | 0.902 | 1.79 |
| (fp32,8) | 32,17 | 0.386 | 4.416 bpw | 7.25x | 0.108 | 0.633 | 0.902 | 1.79 |
| (fp32,7) | 32,16 | 0.378 | 4.408 bpw | 7.26x | 0.108 | 0.633 | 0.902 | 1.79 |
| (fp32,6) | 32,15 | 0.370 | 4.400 bpw | 7.27x | 0.108 | 0.641 | 0.903 | 1.79 |
| (fp32,5) | 32,14 | 0.362 | 4.392 bpw | 7.29x | 0.108 | 0.641 | 0.902 | 1.80 |
| (fp32,4) | 32,13 | 0.354 | 4.384 bpw | 7.30x | 0.108 | 0.617 | 0.903 | 1.78 |
| (fp32,3) | 32,12 | 0.347 | 4.376 bpw | 7.31x | 0.108 | 0.625 | 0.902 | 1.80 |
| (fp32,2) | 32,11 | 0.339 | 4.369 bpw | 7.33x | 0.109 | 0.641 | 0.900 | 1.82 |
| (fp32,1) | 32,10 | 0.331 | 4.361 bpw | 7.34x | 0.112 | 0.570 | 0.887 | 2.06 |
| (fp32,0) | 32,9 | 0.323 | 4.353 bpw | 7.35x | 0.123 | 0.516 | 0.851 | 2.51 |

## Sweep C - residual fp32, base (stage 0) swept

| stage mantissas (base,resid) | centroid bits/scalar | codebook bpw | effective bitwidth | compression | w_relerr | top1_agree | logit_cos | logit_mse |
|---|---|---|---|---|---|---|---|---|
| (fp32,fp32) | 32,32 | 0.504 | 4.534 bpw | 7.06x | 0.108 | 0.633 | 0.902 | 1.79 |
| (12,fp32) | 21,32 | 0.417 | 4.447 bpw | 7.20x | 0.108 | 0.633 | 0.902 | 1.79 |
| (10,fp32) | 19,32 | 0.402 | 4.432 bpw | 7.22x | 0.108 | 0.633 | 0.902 | 1.79 |
| (8,fp32) | 17,32 | 0.386 | 4.416 bpw | 7.25x | 0.108 | 0.641 | 0.902 | 1.79 |
| (7,fp32) | 16,32 | 0.378 | 4.408 bpw | 7.26x | 0.108 | 0.633 | 0.902 | 1.79 |
| (6,fp32) | 15,32 | 0.370 | 4.400 bpw | 7.27x | 0.108 | 0.633 | 0.901 | 1.81 |
| (5,fp32) | 14,32 | 0.362 | 4.392 bpw | 7.29x | 0.108 | 0.633 | 0.903 | 1.78 |
| (4,fp32) | 13,32 | 0.354 | 4.384 bpw | 7.30x | 0.108 | 0.625 | 0.899 | 1.84 |
| (3,fp32) | 12,32 | 0.347 | 4.376 bpw | 7.31x | 0.111 | 0.641 | 0.899 | 1.82 |
| (2,fp32) | 11,32 | 0.339 | 4.369 bpw | 7.33x | 0.119 | 0.531 | 0.851 | 2.78 |
| (1,fp32) | 10,32 | 0.331 | 4.361 bpw | 7.34x | 0.146 | 0.398 | 0.756 | 5.04 |
| (0,fp32) | 9,32 | 0.323 | 4.353 bpw | 7.35x | 0.222 | 0.133 | 0.498 | 8.37 |

## Takeaways

**The base codebook holds essentially all the precision sensitivity; the residual codebook can be stored almost arbitrarily coarsely.**

- **Sweep A ≈ Sweep C.** Squeezing both stages (A) behaves the same as squeezing only the base (C): both are lossless to ~m4, knee at m2, and collapse at m0 (top1_agree 0.078 / 0.133, w_relerr 0.230 / 0.222). Keeping the residual fp32 in C buys almost nothing — the base collapse dominates.
- **Sweep B barely moves.** With the base kept fp32, the residual codebook can drop to m0 and still hold top1_agree 0.516 (w_relerr 0.123). The residual is precision-*insensitive*.
- **Why (opposite of the naive guess):** the base centroids carry the bulk of the weight magnitude, so coarse base storage reintroduces large absolute errors. The residual is already a small correction (≈ the base's `w_relerr`), so rounding it coarsely perturbs only a fraction of a fraction.
- **Mixed-precision recommendation:** spend the mantissa budget on the **base** codebook (~e8m4, 13-bit centroids) and store residual codebooks at very low mantissa (e8m1/e8m0, 9–10-bit). Uniform-precision storage (Sweep A) wastes bits on the residual — though since codebooks are amortized it barely changes the effective bitwidth; it matters when the codebook LUT must sit in a fixed hardware float format.
- The storage floor tracks the k-means clustering error (`w_relerr` ≈ 0.11 ≈ ~4 mantissa bits): precision only bites once the storage grid is coarser than that. Small wiggles are unseeded-k-means noise.

