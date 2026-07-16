# Vector Quantization & Codebook Methods for Post-Training Quantization — Literature Survey

*Compiled 2026-07-07 for the LoFloatQuantizationExp codebase (which already has GPTQ-style Hessians + low-precision float scalar quant). Goal: ground the design of a VQ/codebook PTQ framework.*

---

## 0. Background & Taxonomy

**Scalar quantization (SQ)** maps each weight independently to a grid (uniform int or non-uniform float). It cannot exploit correlations between weights, so its rate-distortion is bounded by the per-coordinate entropy. **Vector quantization (VQ)** jointly quantizes a group of `d` weights to a single codebook entry (centroid). By Shannon rate-distortion theory, quantizing in higher dimension gives strictly better distortion at a fixed bit budget — the "blessing of dimensionality" that GPTVQ names. The cost is a codebook of `k = 2^(b·d)` entries and a nearest-centroid search per group.

Three classic codebook structures (from the ANN-search literature, all applicable to weights):

| Structure | Approximation | Codebook cost | Notes |
|---|---|---|---|
| **Product Quantization (PQ)** | `x ≈ [c¹_{i1}; c²_{i2}; …; c^M_{iM}]` (concatenation of sub-space centroids) | M codebooks, disjoint subspaces | Jégou, Douze, Schmid, *"Product Quantization for Nearest Neighbor Search"*, IEEE TPAMI 2011. Cheapest; each sub-vector quantized independently. |
| **Residual VQ (RVQ)** | `x ≈ Σ_{m=1}^M c^m_{i_m}` where stage `m` quantizes the residual of stage `m−1` | M codebooks, full-D, sequential | Chen, Guan, Wang 2010; Martinez et al. *"Improved RVQ"* arXiv:1509.05195. Greedy/coarse-to-fine. |
| **Additive Quantization (AQ)** | `x ≈ Σ_{m=1}^M c^m_{i_m}`, codebooks jointly optimized (not residual/greedy) | M full-D codebooks | Babenko & Lempitsky, *"Additive Quantization for Extreme Vector Compression"*, CVPR 2014. Most expressive, hardest assignment (beam search). AQLM is the LLM incarnation. |

All modern weight-VQ PTQ methods reuse the **GPTQ proxy objective**: minimize the per-layer output MSE
```
L(Ŵ) = E_x || (Ŵ − W) x ||²  =  tr[ (Ŵ − W) H (Ŵ − W)ᵀ ],   H = E_x[ x xᵀ ]
```
where `H` is the (calibration-data) second moment of the layer input — exactly the "GPTQ Hessian" the repo already computes. The central research theme below is **how to fold `H` into codebook construction, code assignment, and error feedback**, rather than treating VQ as plain Euclidean k-means.

---

## 1. GPTVQ — *The Blessing of Dimensionality for LLM Quantization*

- **Cite:** van Baalen, Kuzmin, Nagel, et al. (Qualcomm AI Research), arXiv:**2402.15319**, Feb 2024. Code: `github.com/Qualcomm-AI-research/gptvq`.
- **One-liner:** GPTQ generalized from scalar to vector codebooks, with Hessian-weighted assignment and error feedback.

**Algorithm.** Extends GPTQ to quantize `d` columns at a time (d ∈ {1, 2, 4}). Uses the same running-error-feedback loop as GPTQ, but each `d`-dim group is snapped to a codebook centroid instead of a scalar grid point.

- **Hessian-weighted assignment** (their Eq. 4) — assign group `x` to centroid `c^(m)` minimizing the Mahalanobis (H-weighted) distance, not Euclidean:
  ```
  j = argmin_m (x − c^(m))ᵀ H^(i) (x − c^(m))
  ```
- **Codebook init via data-aware EM / weighted k-means:**
  - E-step: assign with Eq. 4.
  - M-step (closed form, their Eq. 6): `c^(m) = ( Σ_{i∈I_m} H^(i) )⁺ ( Σ_{i∈I_m} H^(i) x^(i) )` — an H-weighted centroid (reduces to the mean when `H = I`).
  - Faster **"Mahalanobis initialization"** as a k-means++ substitute: sort points by Mahalanobis distance and sample evenly.
- **GPTQ column update / error feedback:** after quantizing a group, the remaining unquantized columns are updated using rows of `H⁻¹` (Cholesky), absorbing the group's quantization error exactly as in GPTQ.
- **Codebook storage compression:** (1) blockwise normalization of weights by max-abs with 4-bit log-scale scales; (2) 8-bit symmetric quantization of the codebook itself; (3) for 1D, SVD/low-rank compression of the codebook tensor. This is what keeps the effective bits-per-value (BPV) near the nominal budget:
  ```
  BPV = log₂(k)      +  k·d·b_c / l   +  b_s / N_s
        (index bits)    (codebook amort.)  (scale amort.)
  ```

**Numbers (Llama-2-70B, Wikitext ppl):** GPTVQ-2D at 2.125 BPV → **4.64** vs GPTQ 6.74; at 3.125 BPV → 3.55 vs 3.85. Runtime 30 min–1 h (7B) to 3–11 h (70B) on one H100.

**Why it matters here:** GPTVQ is the *most direct upgrade path* from an existing GPTQ implementation — it is literally GPTQ with (a) a codebook lookup replacing the scalar round, (b) an H-weighted assignment, and (c) the same error-feedback loop. If the repo has GPTQ Hessians, ~80% of the machinery already exists.

---

## 2. AQLM — Additive Quantization for Language Models

- **Cite:** Egiazarian, Panferov, Kuznedelev, Frantar, Babenko, Alistarh, *"Extreme Compression of LLMs via Additive Quantization"*, arXiv:**2401.06118**, ICML 2024. Code: `github.com/Vahe1994/AQLM`.
- **One-liner:** classic Additive/Multi-Codebook Quantization (MCQ) applied to weight rows, with calibration-aware codebook fitting and block fine-tuning.

**Formulation.** Split each weight row into groups of dimension `g`. Approximate each group as a **sum of M codewords**, one from each of M learned codebooks, each codebook having `2^B` entries:
```
w_group ≈ Σ_{m=1}^{M} C_m b_m ,   b_m ∈ {one-hot over 2^B entries}
```
Effective rate ≈ `B·M / g` bits/weight (codes dominate; FP16 codebooks + scales amortize to near-zero). Typical config `1×16` = M=1, B=16 (one 65536-entry codebook), or `2×8`, etc.

**Optimization (alternating, per layer, minimizing the GPTQ proxy `‖(Ŵ−W)X‖²`):**
1. **Code assignment (discrete):** with codebooks fixed, find the code tuple `(b_1,…,b_M)` per group. Because AQ assignment is NP-hard, AQLM uses **beam search** over combinations of codewords (this is the AQ-specific step that distinguishes it from RVQ's greedy residual assignment).
2. **Codebook + scale update (continuous):** with codes frozen, solve a **least-squares problem weighted by `XXᵀ`** (the calibration Hessian) for the codebook entries — i.e. the update is Hessian/calibration-aware, not plain MSE.
3. **Block-wise fine-tuning:** after per-layer fitting, jointly fine-tune codebooks, scales, and non-quantized params (norms) over a transformer block via backprop (codes frozen), to absorb cross-layer error interactions.

**Numbers (2 bits/param, Wikitext ppl):** Llama-2-7B **6.93**, 13B **5.70**, 70B **3.94**. First method Pareto-optimal below 3 bits; ~2.5 bits identified as the sweet spot for Llama-2. Inference: custom kernels give 1.2–3× speedup.

**Cost/caveat:** codebook fitting + beam search + fine-tuning is *expensive* (much slower to quantize than GPTVQ), and needs backprop through blocks. Highest accuracy at ≤2 bits, but heaviest engineering.

---

## 3. QuIP and QuIP# — Incoherence Processing + Lattice Codebooks

### 3a. QuIP (the original)
- **Cite:** Chee, Cai, Kuleshov, De Sa, *"QuIP: 2-Bit Quantization of LLMs With Guarantees"*, arXiv:**2307.13304**, NeurIPS 2023.
- **Key idea — incoherence:** quantization is easier when the weight matrix `W` and Hessian `H` are *incoherent* — weights uniform in magnitude, and the important rounding directions not axis-aligned. QuIP multiplies `W` and `H` by random orthogonal matrices (`W ← U W Vᵀ`, `H ← V H Vᵀ`) so outliers are spread out.
- **LDLQ adaptive rounding:** a linear-feedback rounding scheme that is *provably optimal* among such schemes for the proxy loss. Uses the LDL decomposition `H = LᵀDL`; the feedback matrix is `U = Lᵀ − I`. This generalizes/subsumes GPTQ's rounding.

### 3b. QuIP# (the strong version)
- **Cite:** Tseng, Chee, Sun, Kuleshov, De Sa, *"QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks"*, arXiv:**2402.04396**, ICML 2024.
- **Randomized Hadamard Transform (RHT):** replaces QuIP's Kronecker random orthogonals with `x → V S x`, V a Hadamard matrix, S a random ±1 sign diagonal. Runtime drops from Θ(n√n) to **Θ(n log n)**, with better (√log vs log²) incoherence bounds. Incoherent weights become approximately i.i.d. sub-Gaussian → **ball-shaped** distribution.
- **E8P lattice codebook:** exploits the ball shape by quantizing **8-dim vectors** to the **E₈ lattice**, which gives the optimal 8-D unit-ball packing. The `E8P` codebook has `2^16` effective entries but stores only a `2^8`-entry *source codebook* (227 vectors of norm ≤√10 + padding); the other bits encode sign flips (7 bits, 8th inferred by parity) and a ±1/4 shift. Net: the 8-D codebook is ~1 KiB instead of ~1 MiB (≈1000× smaller) — hardware/cache friendly.
- **BlockLDLQ:** the vector-valued generalization of LDLQ that snaps 8-D blocks with Hessian-driven feedback:
  ```
  Ŵ_k = Q( W_k + (W_{k−1} − Ŵ_{k−1}) A_k ) ,   A_k from block-LDL of H = LᵀDL
  ```
- **Bitrate via residual VQ:** 2-bit = one E8P; 3-bit = E8P + a 1-bit E₈ residual codebook; 4-bit = two stacked E8P. (This is exactly RVQ stacking on top of the lattice quantizer.)
- **Fine-tuning:** after each layer, relax sign vectors to FP16 and fine-tune norms/sign-vectors/LM-head to match full-model activations (<0.01 bits/wt overhead).

**Numbers (Wikitext ppl):** Llama-2-70B 2-bit **4.16** / 3-bit **3.56** / 4-bit **3.38**; 13B 2-bit **5.74**. Famous result: 3-bit QuIP# scales *better* than lossless-4-bit.

### 3c. QTIP (successor to QuIP#)
- **Cite:** Tseng, Sun, Hou, De Sa, *"QTIP: Quantization with Trellises and Incoherence Processing"*, arXiv:**2406.11235**, NeurIPS 2024.
- **Idea:** VQ codebook size is exponential in dimension, capping practical VQ at d≤8. QTIP replaces the lattice codebook with **trellis-coded quantization (TCQ)** using a hardware-efficient **"bitshift trellis"** and compute-based random Gaussian codes — achieving *ultra-high effective dimension* without storing a codebook, plus parallel decoding. State-of-the-art quality + speed, but the most complex to implement. Same RHT incoherence front-end as QuIP#.

**Why QuIP-family matters here:** the incoherence (RHT) front-end is *cheap and orthogonal* — it can be bolted onto any of the other methods to make their codebooks work better, and the LDLQ/BlockLDLQ rounding is the theoretically-clean generalization of the GPTQ update the repo already has.

---

## 4. VPTQ — Vector Post-Training Quantization

- **Cite:** Liu, Wang, et al. (Microsoft), *"VPTQ: Extreme Low-bit Vector Post-Training Quantization for LLMs"*, arXiv:**2409.17066**, EMNLP 2024. Code: `github.com/microsoft/VPTQ`.
- **One-liner:** GPTVQ-style H-weighted VQ, but reformulated **channel-independently** to cut error accumulation, plus residual + outlier codebooks; targets 1–2 bits.

**Algorithm.**
- **Channel-Independent Second-Order Optimization:** quantizes each *column* of `W` independently under the GPTQ proxy. Because the relevant Hessian term is (approximately) diagonal per channel, the objective decouples and the leading term reduces to a **weighted k-means** with Hessian-diagonal weights — giving both a cheap centroid-init and reduced cross-column error accumulation vs GPTVQ.
- **Hessian-weighted k-means codebook init:** centroids initialized by weighting each vector's contribution by its Hessian-derived importance (parallels GPTVQ Eq. 6 / SqueezeLLM's Fisher weighting), then a short refinement — much lower centroid-training overhead.
- **Residual VQ:** a second codebook quantizes the residual of the first (coarse-to-fine), buying accuracy at small extra bits.
- **Outlier quantization:** a separate codebook/handling for outlier channels.
- **Inference:** pure **lookup** — store index tables + codebooks; dequant = gather centroids. Reports **1.6–1.8× throughput** vs FP16 baseline and only 10–19% of the quantization *time* of comparable SOTA.

**Numbers (2-bit):** beats prior SOTA by 0.01–0.34 ppl on Llama-2, 0.38–0.68 on Mistral-7B, large gains on Llama-3; +0.79–1.5% QA accuracy. Can push 70B/405B to 1–2 bits.

**Why it matters here:** VPTQ is arguably the best-engineered "GPTVQ done right" — same Hessian machinery, simpler per-channel decomposition, faster to run, with residual+outlier extensions that are individually easy to add.

---

## 5. SqueezeLLM — Sensitivity (Fisher/Hessian)-weighted k-means + Dense-and-Sparse

- **Cite:** Kim, Hooper, Gholami, et al. (UC Berkeley), *"SqueezeLLM: Dense-and-Sparse Quantization"*, arXiv:**2306.07629**, ICML 2024.
- **One-liner:** *scalar* (1-D) non-uniform quantization, but with a Hessian-weighted k-means codebook + a full-precision sparse outlier residual. The cleanest "gateway drug" to codebook methods.

**Sensitivity-weighted k-means** (their objective): choose the `2^b` centroids `Q` minimizing the **diagonal-Fisher-weighted** reconstruction error, so that high-sensitivity weights sit near a centroid:
```
Q* ≈ argmin_Q  Σ_i  F_ii ( w_i − Q(w_i) )²
```
where the Fisher information approximates the Hessian, `H ≃ F = (1/|D|) Σ_{d∈D} g_d g_dᵀ`, taken diagonal. (This is a 1-D k-means — a codebook of scalars — but the weighting principle is identical to the vector methods above.)

**Dense-and-Sparse decomposition:** `W = D + S`. `S` holds ~0.05% *sensitive* weights (largest Fisher) + ~0.4% magnitude *outliers*, kept in **FP16 CSR sparse** form; `D` (the ~99.5% dense remainder) is aggressively quantized. Removing outliers before k-means dramatically tightens the centroid dynamic range.

**Numbers:** Llama-7B 3-bit + 0.45% sparsity → C4 ppl **7.56** vs FP16 7.08; 2.1–2.3× speedup on A6000.

**Why it matters here:** it's the smallest possible step from scalar float quant — a 1-D codebook via weighted k-means — and the dense-and-sparse outlier trick is a reusable component for *every* VQ method (VQ hates outliers; pulling ~0.1–0.5% into a sparse FP16 side-channel helps all of them).

---

## 6. PV-Tuning — better fine-tuning for extreme-VQ models

- **Cite:** Malinovskii, Mazur, Ilin, Kuznedelev, Burlachenko, Yi, Alistarh, Richtárik, *"PV-Tuning: Beyond Straight-Through Estimation for Extreme LLM Compression"*, arXiv:**2405.14852**, NeurIPS 2024. (Same lab as AQLM; ships in the AQLM repo.)
- **Problem it fixes:** QuIP#/AQLM fine-tune codebooks with **straight-through estimators (STE)**, which PV-Tuning shows is sub-optimal at 1–2 bits.
- **Method:** a representation-agnostic fine-tuning framework that alternates updating the **continuous P**art (codebooks, scales, non-quantized params) and the **discrete V**alue part (the code assignments) with a principled subspace/coordinate strategy, with convergence guarantees in restricted cases. It is a *post-processing wrapper* over any codebook representation (AQLM, VQ, etc.).
- **Numbers:** first **Pareto-optimal 2-bit** Llama-2, beating QuIP#/AQLM at matched size.

**Why it matters here:** not a quantizer itself — it's the fine-tuning stage that squeezes the last accuracy out of whatever codebook you produce. Only worth building after a base VQ quantizer exists and you're targeting ≤2 bits.

---

## 7. Practical kernels, libraries & fast assignment

**Codebook construction (offline):**
- **faiss** (`github.com/facebookresearch/faiss`) — the workhorse. `faiss.Kmeans(d, k, gpu=True)` does GPU balanced k-means for centroid learning; `IndexPQ` / `ProductQuantizer` build PQ codebooks; `IndexIVFPQ` gives the coarse+residual (two-level) structure that maps directly onto RVQ. Assignment uses SIMD/GPU batched L2. Note: faiss does **plain Euclidean** k-means — for H-weighted objectives you must either (a) pre-whiten vectors by `H^{1/2}` so Euclidean-in-whitened-space = Mahalanobis-in-original (works when a shared per-group `H` factor exists), or (b) roll your own weighted Lloyd iteration.
- **torch-native k-means:** for H-weighted / per-group-Hessian objectives, a custom Lloyd loop is usually simplest — assignment is a batched `torch.cdist`/`einsum` `argmin` over `(x−c)ᵀH(x−c)`, and the M-step is the closed-form solve in §1. `torch.cdist` + `topk` handles Euclidean nearest-centroid at scale; `scikit-learn-intelex` or `kmeans_pytorch` are drop-ins for the unweighted case.
- **Whitening trick:** because `(x−c)ᵀH(x−c) = ‖H^{1/2}(x−c)‖²`, apply `H^{1/2}` (or the RHT of QuIP#) once up front, then all assignment is ordinary Euclidean → lets you reuse faiss/standard kernels for the H-weighted problem.

**Inference / dequant kernels:**
- Dequant is a **gather**: for each group, read the b-bit index, look up the `d`-dim centroid, write out. The bottleneck is memory bandwidth on the index stream + codebook cache residency, so small codebooks (QuIP#'s 1 KiB E8P; PQ's per-subspace tables) are decisive.
- **AQLM** and **QuIP#/QTIP** ship custom **CUDA/Triton kernels** that fuse dequant into the GEMM (dequantize on-chip into shared memory / registers, never materialize FP16 weights in DRAM). AQLM reports 1.2–3×, QuIP#/QTIP target matched-or-better-than-FP16 throughput.
- **VPTQ** ships lookup-table dequant kernels (index → centroid gather) reporting 1.6–1.8× over FP16.
- **CodeGEMM** (arXiv:2512.17970) is a codebook-centric GEMM formulation worth watching for fused codebook×activation matmul.
- **LUT residency:** keep the codebook in shared memory / L2; for PQ, the per-subspace tables are tiny and enable the classic "ADC" (asymmetric distance computation) table-lookup trick if you ever need distance estimates.

---

## 8. Recommended implementation priorities

Given the repo **already has (a) GPTQ-style per-layer Hessians and (b) low-precision float scalar quant**, here is the value-ordered build path. Each tier reuses the previous tier's code.

**Tier 1 — Highest value, lowest marginal effort (build first):**
1. **SqueezeLLM-style Hessian/Fisher-weighted 1-D k-means codebook** + **dense-and-sparse outlier extraction.** This is the smallest step from the existing scalar float quant: swap the float grid for a learned scalar codebook via weighted k-means (reuse the diagonal of the existing Hessian as weights). The dense-and-sparse FP16 side-channel is a standalone, reusable component that benefits every later method. Ships fast, validates the codebook/index storage plumbing and dequant-gather kernel.
2. **GPTVQ (2-D, then 4-D).** This is *GPTQ + codebook lookup + H-weighted assignment + the same error-feedback loop* — the repo's GPTQ Hessian and Cholesky/`H⁻¹` machinery transfer almost verbatim. The only new pieces are the EM/weighted-k-means init (§1 Eq. 6, which you already wrote in Tier 1) and the `(x−c)ᵀH(x−c)` assignment. Best accuracy-per-engineering-hour for a Hessian-equipped codebase. Add the 8-bit + SVD codebook compression once the core works.

**Tier 2 — High value, moderate effort:**
3. **VPTQ-style channel-independent reformulation + residual VQ + outlier codebook.** Once GPTVQ works, VPTQ is a refinement: per-channel decoupling (cheaper, less error accumulation) + a second residual codebook (trivial RVQ stacking) + reuse the Tier-1 outlier handling. This is the practical accuracy/speed sweet spot at 2 bits.
4. **QuIP# incoherence front-end (RHT).** Add `x → V S x` randomized Hadamard pre/post-processing as an *optional preprocessing pass* usable by all quantizers above (it's cheap, Θ(n log n), and the whitening it provides lets you reuse standard Euclidean k-means/faiss kernels). Optionally add the E8P lattice codebook as an alternative to learned codebooks for the ball-shaped post-RHT weights.

**Tier 3 — Highest accuracy at ≤2 bits, highest effort (only if you must hit 2-bit SOTA):**
5. **AQLM (additive/multi-codebook + beam search + block fine-tuning).** Best ≤2-bit accuracy but expensive to run and to implement (beam-search assignment, backprop block fine-tuning, custom kernels).
6. **PV-Tuning** as the fine-tuning stage on top of AQLM/VPTQ codebooks — only meaningful once a base VQ quantizer exists and you're chasing the 1–2 bit Pareto frontier.
7. **QTIP (trellis-coded quantization).** SOTA quality+speed but by far the most complex (trellis decoding, custom kernels); defer unless the codebook-size ceiling of d≤8 VQ becomes the binding constraint.

**Rule of thumb:** for 3–4 bit targets, Tier-1 GPTVQ/SqueezeLLM already recovers near-FP16 quality cheaply. Only descend to Tier-3 (AQLM/PV-Tuning/QTIP) if the product target is a genuine 2-bit-or-below regime.

---

## Sources
- GPTVQ — arXiv:2402.15319 · https://arxiv.org/abs/2402.15319 · code https://github.com/Qualcomm-AI-research/gptvq
- AQLM — arXiv:2401.06118 · https://arxiv.org/abs/2401.06118 · code https://github.com/Vahe1994/AQLM · explainer https://towardsdatascience.com/the-aqlm-quantization-algorithm-explained-8cf33e4a783e/
- QuIP — arXiv:2307.13304 · https://arxiv.org/abs/2307.13304
- QuIP# — arXiv:2402.04396 · https://arxiv.org/abs/2402.04396
- QTIP — arXiv:2406.11235 · https://arxiv.org/abs/2406.11235
- VPTQ — arXiv:2409.17066 · https://arxiv.org/abs/2409.17066 · code https://github.com/microsoft/VPTQ
- SqueezeLLM — arXiv:2306.07629 · https://arxiv.org/abs/2306.07629
- PV-Tuning — arXiv:2405.14852 · https://arxiv.org/abs/2405.14852
- Product Quantization — Jégou, Douze, Schmid, IEEE TPAMI 2011 · https://pubmed.ncbi.nlm.nih.gov/21088323/
- Improved Residual VQ — arXiv:1509.05195 · https://arxiv.org/abs/1509.05195
- Additive Quantization — Babenko & Lempitsky, CVPR 2014
- faiss — https://github.com/facebookresearch/faiss/wiki/Faiss-building-blocks:-clustering,-PCA,-quantization
- CodeGEMM — arXiv:2512.17970
