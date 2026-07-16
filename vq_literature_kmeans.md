# Codebook Construction & K-Means Clustering for Vector Quantization of NN Weights

A practical, implementation-level survey for building a robust, fast, GPU-friendly
Vector Quantization (VQ) framework in PyTorch supporting **arbitrary vector/group
length** and **optional per-element importance (Hessian) weights**. Written for a
PTQ / weight-compression research codebase.

> Scope note: this is about *codebook construction* (offline clustering of weight
> sub-vectors into a lookup table) and the clustering objective, not about the
> transformer/GPTQ error-feedback loop itself — though §2 shows how the two connect.

---

## 0. Notation and the core objective

We quantize a weight matrix `W ∈ R^{out × in}`. Reshape/slice it into a set of
`N` vectors `{x_i}`, each of length `d` (the *group* or *vector* length). We learn
a codebook `C = {c_1, …, c_K}`, `c_k ∈ R^d`, and an assignment `a_i ∈ {1..K}`, and
replace each `x_i` by `c_{a_i}`. Storage per vector is `⌈log2 K⌉` bits for the index
plus the amortized codebook cost.

Plain (unweighted) VQ minimizes the k-means distortion

$$
J = \sum_{i=1}^{N} \lVert x_i - c_{a_i}\rVert_2^2 .
$$

The whole game in *weight* quantization is that this Euclidean distortion is a
**proxy** for the loss increase. The better proxy (§2) weights each coordinate by
its curvature (diagonal Hessian), which turns k-means into *weighted* k-means.

---

## 1. K-means variants

### 1.1 Lloyd's algorithm (batch k-means)

The workhorse. Alternate two steps until convergence:

1. **Assignment (E-step):** `a_i = argmin_k ||x_i − c_k||²`.
2. **Update (M-step):** `c_k = mean{ x_i : a_i = k }`.

Each step is non-increasing in `J`, so it converges to a *local* optimum. It is
sensitive to initialization (hence §1.2). Cost per iteration is `O(N·K·d)`.

**Convergence criteria** (use several, stop on any):
- No assignments changed (`a` stable) — exact convergence, but can oscillate on ties.
- Relative distortion change `(J_{t-1} − J_t)/J_{t-1} < tol` (e.g. `tol = 1e-4`).
- Centroid shift `max_k ||c_k^{t} − c_k^{t-1}|| < tol`.
- A hard iteration cap (e.g. 50–100). For weight VQ, distortion usually plateaus in
  10–30 iters.

**Numerical stability:**
- Do distance math and the mean in **fp32** even if weights are fp16/bf16. Accumulating
  a mean of thousands of bf16 values in bf16 loses bits badly; cast up, reduce, cast back.
- Guard the mean divisor against 0 (empty clusters, §1.5).
- If features have wildly different scales, k-means is dominated by large-scale dims.
  For weights this is usually fine (per-channel/per-group scaling normalizes it), but
  if you cluster raw pre-scale weights, consider standardizing first.

### 1.2 k-means++ initialization (D² seeding)

Arthur & Vassilvitskii (SODA 2007). Instead of random centers, spread seeds out:

1. Pick `c_1` uniformly at random from the data.
2. For each remaining seed, pick `x` as the next center with probability
   `∝ D(x)²`, where `D(x)` is the distance from `x` to the *nearest already-chosen*
   center.
3. Repeat until `K` seeds are chosen, then run Lloyd's.

This "D² weighting" gives an **`O(log K)` approximation guarantee in expectation**
and dramatically improves both final distortion and convergence speed vs. random
seeding. It is the default in scikit-learn (`init="k-means++"`), which also runs
`n_local_trials ≈ 2 + log(K)` candidates per step and keeps the best. This is the
recommended default initializer for weight VQ.

```python
def kmeanspp(X, K, w=None):        # X: (N,d) fp32 on device; w: optional (N,) weights
    N = X.shape[0]
    idx0 = torch.randint(N, (1,), device=X.device)
    centers = [X[idx0[0]]]
    d2 = torch.cdist(X, centers[-1][None]).squeeze(1).pow(2)   # (N,)
    for _ in range(K - 1):
        p = d2 * (w if w is not None else 1.0)
        p = p / p.sum().clamp_min(1e-30)
        j = torch.multinomial(p, 1)
        centers.append(X[j[0]])
        d2 = torch.minimum(d2, torch.cdist(X, centers[-1][None]).squeeze(1).pow(2))
    return torch.stack(centers)
```

(For weighted k-means, seed with `w·D²` so important vectors are more likely to become
centers — this matches the weighted objective.)

Related: **k-means‖ (k-means-parallel)**, Bahmani et al. VLDB 2012, oversamples
`O(K)` points per round over `O(log N)` rounds then reclusters the seeds — designed for
distributed/very-large `N`. Rarely needed at weight-VQ scale.

### 1.3 Mini-batch k-means (Sculley, WWW 2010)

For large `N` (a big weight matrix reshaped into millions of `d`-vectors), full-batch
Lloyd's is expensive. Mini-batch draws a random subset `b` each iteration, assigns them,
and does a **per-center streaming update** with a per-center learning rate:

- Keep a running count `n_k` per center.
- For each point `x` in the batch assigned to `k`: `n_k ← n_k + 1`,
  learning rate `η = 1/n_k`, then `c_k ← (1−η) c_k + η x`.

Because `η_k = 1/n_k → 0`, early points move centers a lot and later points refine —
a Robbins–Monro stochastic-approximation schedule. scikit-learn's `MiniBatchKMeans`
implements exactly this (plus reassignment of low-count centers). Recent theory (mini-batch
k-means terminates in `O(d/ε)` iterations) shows a *constant* learning rate can also work,
but the `1/n_k` schedule is the standard, robust choice.

Use mini-batch when `N·K·d` per full iteration is too large for memory/time; otherwise
full-batch on GPU is usually faster and gives lower distortion.

### 1.4 GPU / torch-native implementation

Everything above is a few batched tensor ops. A full-batch torch k-means:

```python
def kmeans(X, K, iters=50, tol=1e-4, w=None):     # X:(N,d) fp32 CUDA
    C = kmeanspp(X, K, w)                          # (K,d)
    prev = None
    for _ in range(iters):
        a = assign(X, C)                           # (N,) argmin, see §5
        C = weighted_update(X, a, K, w)            # §2 / §1.5
        J = distortion(X, C, a, w)
        if prev is not None and abs(prev - J) < tol * prev:
            break
        prev = J
    return C, a
```

Key GPU points:
- Keep `X`, `C` resident on the device; never round-trip to CPU inside the loop.
- The M-step is a **scatter-add**: `torch.zeros(K,d).index_add_(0, a, X)` then divide by
  per-cluster counts (`torch.bincount(a, minlength=K)`). This is exact and fast.
- Chunk the assignment step over `N` (and/or `K`) to bound the `N×K` distance matrix
  memory (§5). A `50k × 4096 × d` distance matrix will OOM; a chunk of ~a few thousand
  rows will not.
- For many independent small problems (e.g. one codebook per output channel/group),
  batch them along a leading dim and use `torch.bmm`/broadcasting so the GPU stays busy.

### 1.5 Empty clusters

An empty cluster has no assigned points → undefined mean → NaN if you divide by 0. This
is common at low bit-rates / large `K` relative to data spread. Standard fixes, applied
each M-step:

- **Re-seed from the largest cluster:** take the point *farthest* from the centroid of the
  most populous cluster and make it the new center of the empty one. (This is what serious
  implementations, and partitioning-guided variants, do — it targets under-modeled regions.)
- **Re-seed from the worst-quantized point** overall (largest `w·||x−c_a||²`) — very natural
  for weight VQ since it directly attacks the biggest error.
- **Random data point** re-seed (cheapest, what sklearn's mini-batch does for low-count centers).

Always clamp the divisor: `counts.clamp_min(1)` and then overwrite empty rows explicitly, so
you never propagate NaNs even for one iteration.

---

## 2. Weighted / Hessian-weighted k-means

### 2.1 Why

For network quantization the relevant quantity is the increase in task loss, not the
weight MSE. A second-order Taylor expansion around the trained weights gives
`ΔL ≈ ½ Δw^T H Δw` (the first-order term vanishes at a minimum). Using only the
**diagonal** of the Hessian, `h = diag(H)` (per-weight curvature, cheaply estimated as
the Fisher/GPTQ diagonal from calibration activations, `h_j ≈ E[(∂L/∂w_j)²]` or
`diag(X_calib^T X_calib)` for the layer-wise output MSE), the distortion becomes the
**Hessian-weighted k-means** objective (Choi et al., "Towards the Limit of Network
Quantization", ICLR 2017; reused by VPTQ, GPTVQ):

$$
J_H = \sum_i \sum_{t=1}^{d} h_{i,t}\,\bigl(x_{i,t} - c_{a_i,t}\bigr)^2 .
$$

`h_{i,t}` is the importance of coordinate `t` of vector `i`. Per-**element** weights are
the general case; per-**vector** weights (`h_i` scalar) are the special case
`h_{i,t}=h_i ∀t`.

### 2.2 Exact weighted centroid update

Minimizing `J_H` w.r.t. `c_k` (set gradient to 0) gives a **coordinate-wise
Hessian-weighted mean** of the members of cluster `k`:

$$
\boxed{\,c_{k,t} \;=\; \frac{\displaystyle\sum_{i:\,a_i=k} h_{i,t}\,x_{i,t}}
{\displaystyle\sum_{i:\,a_i=k} h_{i,t}}\,}
$$

i.e. per output coordinate `t`, the weighted average of the members' `t`-th coordinate,
weights = their importance at `t`. For scalar per-vector weights this collapses to
`c_k = (Σ h_i x_i)/(Σ h_i)`.

```python
def weighted_update(X, a, K, w):          # X:(N,d), a:(N,), w:(N,d) or (N,) or None
    d = X.shape[1]
    if w is None:
        num = torch.zeros(K, d, device=X.device).index_add_(0, a, X)
        den = torch.bincount(a, minlength=K).clamp_min(1).unsqueeze(1).to(X.dtype)
        return num / den
    if w.dim() == 1:                       # per-vector -> broadcast to per-element
        w = w.unsqueeze(1).expand_as(X)
    num = torch.zeros(K, d, device=X.device).index_add_(0, a, w * X)
    den = torch.zeros(K, d, device=X.device).index_add_(0, a, w)
    return num / den.clamp_min(1e-12)      # clamp guards empty/zero-weight coords
```

### 2.3 Weighted assignment

The E-step must use the *same* weighted metric, else you optimize a different objective:

$$
a_i = \arg\min_k \sum_t h_{i,t}\,(x_{i,t}-c_{k,t})^2 .
$$

Note `h` here is per-*point* — it does **not** factor out of the argmin, so the fast
`||x||²−2x·c+||c||²` trick (§5) must be applied to the *whitened* residual. Two options:
- **Per-element weights:** expand the square:
  `Σ_t h_{i,t} x_{i,t}² − 2 Σ_t (h_{i,t} x_{i,t}) c_{k,t} + Σ_t h_{i,t} c_{k,t}²`.
  The middle term is `(h_i ⊙ x_i) · c_k` (a matmul), the last is per-(i,k) since `h`
  depends on `i` — so it costs an extra `(N×K)` reduction. Still fully batched.
- **Per-vector weights:** `h_i` is a positive scalar multiplying every term, so it does
  **not** change the argmin — assignment reduces to plain nearest-centroid, and only the
  *update* is weighted. Much cheaper; prefer per-vector weights when the accuracy loss vs.
  per-element is acceptable.

The distortion for convergence checks must likewise be the weighted `J_H`.

### 2.4 Practical notes on the weights
- Clamp `h ≥ ε > 0` (a small floor). Zero-curvature coordinates otherwise make the update
  denominator 0 and let those coords drift arbitrarily.
- Normalize `h` (e.g. divide by its mean) for numerical range; scaling `h` by a constant
  does not change argmin or the centroid formula but keeps sums well-conditioned.
- VPTQ and GPTVQ additionally *initialize* centroids using the Hessian-weighted metric,
  which reduces final quantization error — worth doing in the k-means++ seeding (§1.2,
  pass `w`).

---

## 3. Product Quantization (PQ) and Residual VQ (RVQ)

Both attack the exponential codebook-storage problem: a single joint codebook over `R^d`
needs `K = 2^{b·d}` entries for `b` bits/weight — infeasible. PQ and RVQ compose several
small codebooks.

### 3.1 Product Quantization (Jégou, Douze, Schmid, PAMI 2011)

Split the `d`-vector into `M` **disjoint sub-vectors** of length `d/M`:

`x = [x^(1), …, x^(M)]`, `x^(m) ∈ R^{d/M}`.

Train an **independent** k-means codebook `C^(m)` of size `K'` on each subspace
(just run §1–2 on the sub-vectors). Encode each sub-vector to its nearest sub-centroid;
the code is the tuple `(a^1, …, a^M)`.

- **Effective codebook** = Cartesian product, size `K'^M`, but storage is only
  `M·K'` centroids and the index costs `M·⌈log2 K'⌉` bits.
- **Reconstruction:** concatenate the chosen sub-centroids,
  `x̂ = [c^{(1)}_{a^1}, …, c^{(M)}_{a^M}]`.
- Distortion decomposes as a sum over subspaces, so subspaces train fully in parallel.
- **Optimized PQ (OPQ)** learns a rotation `R` (`x → Rx`) before splitting to balance
  variance/decorrelate across subspaces — worth it for weights whose coordinates have
  uneven variance.

With Hessian weights, PQ is clean: the diagonal metric is separable across coordinates,
so slice `h` into the same `M` blocks and run weighted k-means (§2) per subspace.

### 3.2 Residual Vector Quantization (RVQ / stacked/multi-stage VQ)

Chen et al. 2010. Coarse-to-fine over the **full `d`-vector** with `L` stages:

1. Stage 1: codebook `C^(1)`, quantize `x → c^{(1)}_{a^1}`; residual `r^(1) = x − c^{(1)}_{a^1}`.
2. Stage 2: train `C^(2)` on the residuals `r^(1)`, quantize them; `r^(2) = r^(1) − c^{(2)}_{a^2}`.
3. … repeat for `L` stages.

- **Reconstruction:** `x̂ = Σ_{ℓ=1}^{L} c^{(ℓ)}_{a^ℓ}` (a *sum*, not a concat — this is the
  key difference from PQ).
- Index cost `Σ_ℓ ⌈log2 K_ℓ⌉` bits, storage `Σ_ℓ K_ℓ` centroids of length `d`.
- Effective codebook size `∏_ℓ K_ℓ`. Stages exploit correlations PQ's hard split ignores,
  but stages are **sequential** (each trains on the previous residual).
- VPTQ uses exactly this: a main VQ codebook plus **residual** codebooks (and separate
  outlier codebooks) to push to ~2 bits/weight.

**PQ vs RVQ rule of thumb:** PQ is embarrassingly parallel and great when subspaces are
roughly independent; RVQ captures cross-coordinate structure and reaches lower distortion
at equal bits but must be trained stage-by-stage. They compose (PQ of residuals, "RVPQ").

---

## 4. Codebook size / bit-budget math

For a single codebook of size `K` over vectors of length `d`, the **index cost** is
`log2 K` bits per vector, i.e.

$$
\text{bits per weight} \;=\; \frac{\log_2 K}{d} \quad(\text{index only}).
$$

Add the amortized codebook storage: `K·d` values at `p` bits each, shared over `N`
vectors → `+ (K·d·p)/(N·d) = Kp/N` bits/weight. Keep `K ≪ N` so this stays small
(e.g. `K=256`, `N=10^6`, `p=16` → +0.004 bpw — negligible). At extreme low bit / small
`N` (per-channel codebooks) the codebook overhead is *not* negligible — account for it.

For **composed** codebooks:
- PQ: `bpw = M·⌈log2 K'⌉ / d`.
- RVQ: `bpw = Σ_ℓ ⌈log2 K_ℓ⌉ / d`.

Useful operating points (index-only bits/weight = `log2 K / d`):

| d (group) | K    | log2 K | **bits/weight** | effective states |
|-----------|------|--------|-----------------|------------------|
| 1         | 16   | 4      | 4.00            | 16               |
| 1         | 256  | 8      | 8.00            | 256              |
| 2         | 256  | 8      | 4.00            | 256              |
| 4         | 256  | 8      | 2.00            | 256              |
| 8         | 256  | 8      | 1.00            | 256              |
| 4         | 16   | 4      | 1.00            | 16               |
| 8         | 4096 | 12     | 1.50            | 4096             |
| 6         | 4096 | 12     | 2.00            | 4096             |
| 8         | 65536| 16     | 2.00            | 65536            |
| 16        | 256  | 8      | 0.50            | 256              |

Reading the table: at a fixed `K=256` codebook, doubling the group length `d` halves the
bit-rate but forces each 256-entry codebook to cover a higher-dim space (higher distortion).
The "blessing of dimensionality" (GPTVQ) is that larger `d` at matched *bits* (i.e. larger
`K` too) generally wins, because a `d`-dim lattice packs better than `d` independent scalars —
but `K` grows as `2^{bpw·d}`, so you switch from a single codebook to PQ/RVQ once
`log2 K` exceeds ~12–16 bits to keep clustering and lookup tractable.

---

## 5. Fast nearest-centroid assignment

### 5.1 The expansion trick

Computing `||x_i − c_k||²` for all `(i,k)` naively materializes differences. Expand:

$$
\lVert x_i - c_k\rVert^2 = \lVert x_i\rVert^2 - 2\,x_i\!\cdot\!c_k + \lVert c_k\rVert^2 .
$$

The `||x_i||²` term is constant across `k`, so it **drops out of the argmin** — you only
need `-2 X Cᵀ + ||c||²` (a matmul plus a row-broadcast add), which is a single GEMM and
runs at full tensor-core throughput.

```python
def assign(X, C, chunk=8192):                 # X:(N,d), C:(K,d) fp32
    c_norm = (C * C).sum(1)                    # (K,)  add ||c||^2 only
    out = torch.empty(X.shape[0], dtype=torch.long, device=X.device)
    for s in range(0, X.shape[0], chunk):      # chunk to bound the N x K matrix
        xb = X[s:s+chunk]
        d2 = c_norm[None] - 2.0 * (xb @ C.t()) # (b,K); drop ||x||^2 (const in argmin)
        out[s:s+chunk] = d2.argmin(1)
    return out
```

Numerical caveat: the expansion can produce small **negative** "distances" from
catastrophic cancellation when `x ≈ c`. That is harmless for `argmin` (you never take a
sqrt), but if you *report* distortion, clamp to `≥ 0` or compute the true residual for the
selected centers. `torch.cdist(X, C)` is the numerically safe (but slower, more memory)
alternative; use it for small problems or final distortion accounting.

Weighted assignment (§2.3): for per-vector weights, this exact GEMM path is unchanged
(scalar factor doesn't move the argmin). For per-element weights, precompute
`Xw = h ⊙ X` and use `c_norm_i = (h_i ⊙ C·C summed)` per point — costs one extra
`(N×K)` term via `Xh @ (C·C)ᵀ` where `Xh = h`.

### 5.2 Libraries

- **faiss** (`Clustering`, `IndexFlat`, `IndexIVFPQ`, GPU) is the gold standard for large-scale
  PQ/IVF and fast assignment — but it is **NOT installed here**, and pulling it in for
  weight-VQ (where `N` and `K` are modest and everything is already a torch tensor on CUDA)
  buys little over the GEMM path above while adding a host↔device copy and a dependency.
  Recommendation: **stay torch-native.**
- **scikit-learn 1.9 IS installed.** `sklearn.cluster.KMeans` (Lloyd's + k-means++, Elkan/
  Lloyd, multiple `n_init`) and `MiniBatchKMeans` are excellent CPU references — use them to
  *validate* your torch implementation on small slices, or as a fallback for tiny per-channel
  codebooks. But sklearn is CPU-only and does **not** support per-element (Hessian) weights
  (`sample_weight` is per-sample scalar only), so it can't be your production path for §2.
- `torch.cdist` / `torch.cluster` (PyG) exist but are unnecessary given §5.1.

### 5.3 sklearn `sample_weight` = per-vector k-means for free

`KMeans(...).fit(X, sample_weight=h)` implements exactly the per-*vector* weighted objective
(§2.2 scalar case): weighted centroids `Σ h_i x_i / Σ h_i`. Handy for a quick baseline when
per-vector Hessian weights suffice; it cannot do the per-element case.

---

## 6. Initialization from the data distribution & weight-specific grouping

**Seeding from the data (not from a Gaussian):** always seed centroids from *actual*
weight sub-vectors (k-means++ picks real points). Random-Gaussian init wastes codewords in
empty regions of weight space. For heavy-tailed weight distributions, k-means++ D² seeding
naturally captures outliers as their own centers.

**Grouping choices — the most impactful design decision for weight VQ:**
- **Per-output-channel codebooks:** cluster the vectors formed within each output row
  (or a small group of rows). Aligns with per-output-channel scales already used in PTQ;
  each channel gets a codebook tuned to its dynamic range, but `N` per codebook is small,
  so codebook overhead (§4) and empty clusters (§1.5) bite harder. Good when channels have
  very different statistics.
- **Per-input-group (along the contraction dim):** split each row into contiguous groups of
  `d` input weights and quantize those `d`-vectors. This is the PQ-style layout and matches
  how GPTQ/VPTQ process columns; contiguous input weights that multiply the same activations
  make the Hessian-weighted metric meaningful (the calibration Hessian is over input dims).
- **Shared codebook across the whole layer/tensor:** maximizes `N` (best statistics, lowest
  codebook overhead, fewest empty clusters), at the cost of not adapting to per-channel scale.
  Usually combined with per-channel **scales** so one codebook is reused after normalization.

Tricks that consistently help:
- **Normalize/scale before clustering** (per-channel or per-group scale, optionally an OPQ
  rotation) so one codebook covers a whitened distribution — big distortion win.
- **Hessian-weighted init** (feed `w` into k-means++, §1.2) — VPTQ/GPTVQ report measurable
  gains.
- **Reshape the whole tensor to `(N, d)` once**, cluster all groups jointly when sharing a
  codebook, or batch many independent per-channel problems with a leading dim + `bmm`.
- Deduplicate identical sub-vectors (common after scaling/rounding) before clustering to
  shrink `N` and stabilize counts.

---

## 7. Concrete recommendations (robust, GPU-friendly, arbitrary `d`, optional weights)

1. **Core loop:** full-batch Lloyd's on CUDA in **fp32**, with **k-means++** seeding.
   Fall back to **mini-batch** (`1/n_k` learning rate) only when `N·K·d` per iteration is
   too big for memory/time.
2. **Assignment:** the `−2XCᵀ + ||C||²` GEMM (§5.1), **chunked over `N`** (e.g. 8k rows) to
   bound the `N×K` matrix; `argmin` in fp32. Clamp reported distances to `≥0`.
3. **Update:** `index_add_` scatter for both numerator and denominator; **fp32 accumulation**.
   Support three weight modes with one code path: `None` (plain), per-vector `(N,)`, and
   per-element `(N,d)` Hessian weights, using the boxed formula in §2.2.
4. **Weighted metric consistency:** if weights are per-element, use the weighted metric in
   *both* E- and M-steps and in the convergence `J`; if per-vector, weight only the update
   (assignment is unchanged) for a large speedup.
5. **Empty clusters:** every M-step, `clamp_min` the denominator, then re-seed empty centers
   from the **highest-error point** (largest `w·||x−c_a||²`) — never emit NaNs.
6. **Convergence:** stop on relative `ΔJ < 1e-4` OR centroid shift `< tol` OR iter cap (≈50);
   run `n_init` = 3–10 seeds for small/critical codebooks and keep the lowest `J`.
7. **Numerical floors:** `h ← clamp(h, min=ε)`; normalize `h` by its mean; keep centroids and
   distances fp32 regardless of weight dtype; cast back to the storage dtype only at the end.
8. **Scaling / codebook composition:** apply per-channel (or per-group) scales — optionally an
   OPQ rotation — before clustering. For bit-rates needing `log2 K > ~12`, switch from a single
   codebook to **PQ** (independent subspaces, embarrassingly parallel, slice `h` per subspace)
   or **RVQ** (sequential residual stages, `x̂ = Σ_ℓ c^{(ℓ)}`) rather than growing `K`.
9. **Grouping:** default to a **shared per-layer codebook over per-input-group `d`-vectors**
   after per-channel scaling (best statistics + Hessian metric aligns with the input-dim
   calibration Hessian); go per-output-channel only when channel statistics differ enough to
   justify the codebook overhead.
10. **Tooling:** stay **torch-native** (no faiss needed/installed); use **sklearn 1.9**
    `KMeans`/`MiniBatchKMeans` as a CPU validation oracle and a fallback for tiny per-vector-
    weighted codebooks (its `sample_weight` = per-vector case only).

---

## Sources

- Arthur & Vassilvitskii, *k-means++: The Advantages of Careful Seeding* (D² seeding, O(log K) bound) — summary via [Grokipedia: k-means++](https://grokipedia.com/page/K-means++)
- Bahmani et al., *Scalable K-Means++ (k-means‖)*, VLDB 2012 — [PDF](https://theory.stanford.edu/~sergei/papers/vldb12-kmpar.pdf)
- Sculley, *Web-Scale K-Means Clustering* (mini-batch, 1/n_k learning rate) — via [scikit-learn MiniBatchKMeans docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html) and [openreview: mini-batch k-means O(d/ε)](https://openreview.net/pdf?id=jREF4bkfi_S)
- Choi, El-Khamy, Lee, *Towards the Limit of Network Quantization* (Hessian-weighted k-means, ICLR 2017) — [OpenReview PDF](https://openreview.net/pdf?id=rJ8uNptgl)
- *VPTQ: Extreme Low-bit Vector Post-Training Quantization for LLMs*, EMNLP 2024 (Hessian-weighted centroid init, residual + outlier codebooks) — [arXiv 2409.17066](https://arxiv.org/abs/2409.17066), [ACL PDF](https://aclanthology.org/2024.emnlp-main.467.pdf)
- van Baalen et al., *GPTVQ: The Blessing of Dimensionality for LLM Quantization* — [arXiv HTML](https://arxiv.org/html/2402.15319v1)
- Jégou, Douze, Schmid, *Product Quantization for Nearest Neighbor Search*, PAMI 2011 — [ResearchGate PDF](https://www.researchgate.net/publication/47815472_Product_Quantization_for_Nearest_Neighbor_Search); overview: [Product Quantization (arpitbhayani.me)](https://arpitbhayani.me/blogs/product-quantization/)
- Chen et al., residual/multi-stage VQ — via [Residual Quantization with Implicit Neural Codebooks](https://arxiv.org/pdf/2401.14732)
- *Partitioning-Guided K-Means: Extreme Empty Cluster Resolution* — [arXiv PDF](https://arxiv.org/pdf/2306.14031)
- [scikit-learn 1.9 KMeans / MiniBatchKMeans documentation](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html)
