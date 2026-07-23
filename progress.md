# Vector Quantization Framework — Progress

_Last updated: 2026-07-07_

## Goal
Build a vector quantization (VQ) framework for post-training quantization (PTQ)
in this repo, composing with the existing LoFloat low-precision float scalar
quantization. Standalone tensor-in/tensor-out toolkit first; model-coupled
GPTVQ-style integration later.

## Design decisions (confirmed with user)
- **LoFloat centroids from the start** — quantization-aware k-means snaps
  centroids onto the LoFloat grid each iteration (via `lof.exp_mant_quantize`).
- **Standalone toolkit first** — no model/layer coupling in `vector_quant.py`.
- **Hessian/importance weighting built in** — weighted distortion + weighted
  centroid updates (the closed-form `c = (Σ w·x)/(Σ w)`).

## Environment facts
- torch 2.10.0+cu129, CUDA available. sklearn 1.9 present. **faiss NOT installed.**
- LoFloat API used: `lof.exp_mant_quantize(tensor, exp_bits, mantissa_bits)`,
  `lof.mantissa_quantize`, `lof.LoF_Linear`, `lof.LoF_Conv2d`.
- Real GPTQ lives in `sensitivities.py::quantize_weights_with_gptq`
  (lines ~465-677), **NOT** `gptq.py` (that file is effectively a stub).
  - Hessian `H` is the per-input-column Gram matrix `(columns, columns)`.
  - `diag(H)` is the natural per-column importance weight (this is what GPTVQ uses).
  - Scalar quantizer used there: `lof.exp_mant_quantize(q, exp_bits, mant_bits)`
    — exactly what our `CentroidQuantizer` wraps, so codebooks compose with no
    conversion layer.
  - Column-by-column error feedback via Cholesky of `H⁻¹` (lines ~637-666).

## DONE — `vector_quant.py` (standalone toolkit, tested)
All tensor-in/tensor-out, arbitrary vector length (zero-padded), GPU-friendly.

- **Distances** — `pairwise_sqdist`, `weighted_pairwise_sqdist`. Both use the
  `||x-c||² = ||x||² − 2x·c + ||c||²` matmul expansion; never form `(N,K,d)`.
- **`weighted_kmeans`** — Lloyd + k-means++ init, empty-cluster reseed (split
  worst-fit points), relative-inertia convergence, `n_init` restarts. Weight
  shapes accepted: `None` / `(d,)` per-dim / `(N,)` per-vector / `(N,d)` full.
  Optional `centroid_quantizer` applied every iteration (quant-aware).
- **`CentroidQuantizer`** — wraps `lof.exp_mant_quantize(exp_bits, mantissa_bits)`.
- **`VectorQuantizer`** — plain single-codebook VQ (fit/encode/decode/quantize/
  bits_per_weight).
- **`ProductQuantizer`** — M disjoint subspaces, one codebook each.
- **`ResidualQuantizer`** — greedy S-stage additive RVQ.

### Smoke test results (passed)
- Uniform k-means on clustered synthetic data: all 16/16 centroids used.
- Weighted (per-element) k-means: runs, converges.
- VQ on `(130, 257)` matrix (exercises padding): correct shape, bpw 2.07.
- LoFloat-centroid VQ: `codebook == cq(codebook)` verified (on-grid).
- PQ bpw 2.06, RVQ bpw 1.68, all correct output shapes.

## DONE — Literature surveys
- **`vq_literature_kmeans.md`** — k-means variants, weighted/Hessian k-means,
  PQ vs RVQ, bit-budget math, fast assignment, grouping. Confirmed our design
  (weighted centroid = ratio of Hessian-weighted sums; GEMM distance trick;
  k-means++ D² bound). Note: sklearn `sample_weight` is per-vector only, so
  per-element Hessian weighting must be our own torch path (which we built).
- **`vq_literature_methods.md`** — GPTVQ, AQLM, QuIP/QuIP#/QTIP, VPTQ,
  SqueezeLLM, PV-Tuning + practical kernels. All arXiv IDs verified.
  - **Recommended build path** (value-ordered, each tier reuses the last):
    - **Tier 1:** SqueezeLLM-style Hessian-weighted 1-D k-means + dense-and-sparse
      outliers; then **GPTVQ** (GPTQ + codebook lookup + H-weighted assignment +
      same error-feedback loop — ~80% of machinery already exists here).
    - **Tier 2:** VPTQ (channel-independent + residual + outlier codebook);
      QuIP# randomized-Hadamard (RHT) front-end.
    - **Tier 3 (≤2-bit SOTA only):** AQLM, PV-Tuning, QTIP.

## Current status
The "simple vector quant methods" milestone is **complete and validated**. The
implemented `weighted_kmeans` update IS GPTVQ's Eq. 6 M-step and SqueezeLLM's
Fisher-weighted centroid — i.e. the Tier-1 foundation is in place.

## NEXT STEP (not yet started) — GPTVQ integration
Two pieces separate the current standalone toolkit from GPTVQ; both need the
calibration Hessian and belong in a model-coupled function, not the toolkit:

1. **Mahalanobis (full-block-`H`) assignment** `argmin (x−c)ᵀ H (x−c)` instead
   of the current diagonal/per-element weighted distance.
2. **Error feedback** — propagate each group's quantization error to the
   unquantized columns via `H⁻¹` (reuse the Cholesky loop in `sensitivities.py`).

Plan: add a function alongside `quantize_weights_with_gptq` that swaps the
scalar `exp_mant_quantize` round for a `VectorQuantizer` codebook lookup,
feeding `diag(H)` (or full block `H`) as the importance weight, reusing the
existing Cholesky/error-feedback machinery. Add 8-bit + SVD codebook
compression once the core works.

## Open questions / TODO
- Decide grouping axis for GPTVQ (columns-at-a-time, d ∈ {1,2,4} per the paper).
- Codebook storage/compression accounting for realistic bits-per-weight.
- An example harness running the quantizers on a real `LoF_Linear` weight to
  plot reconstruction RMSE vs bits-per-weight (offered, not yet built).
