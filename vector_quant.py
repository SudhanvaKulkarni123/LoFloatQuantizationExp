"""Vector quantization (VQ) framework for post-training quantization (PTQ).

This is a *standalone*, tensor-in / tensor-out toolkit — it has no dependency on
any particular model or layer type.  It composes with the project's LoFloat
low-precision float formats: codebook centroids can be snapped onto a LoFloat
grid *during* k-means (quantization-aware clustering), so the resulting codebook
is natively low-precision rather than fp32 rounded after the fact.

Design goals
------------
* **Arbitrary vector length** ``d``.  A weight matrix is reshaped into a set of
  ``d``-dimensional vectors; a codebook of ``K`` centroids then represents each
  vector by a ``ceil(log2 K)``-bit index, i.e. ``log2(K) / d`` bits per weight.
* **Hessian / importance weighting.**  The distortion minimized by k-means can
  be weighted per element (e.g. by a diagonal Hessian from calibration data,
  GPTQ-style), so that error is spent where it hurts the loss most.
* **GPU friendly.**  All hot loops are batched torch ops; nearest-centroid
  assignment goes through :func:`torch.cdist` (unweighted) or the
  ``||x-c||^2 = ||x||^2 - 2 x·c + ||c||^2`` expansion (per-element weighted), so
  it is a couple of matmuls rather than an explicit ``(N, K, d)`` tensor.

Library reuse: k-means++ seeding delegates to
:func:`sklearn.cluster.kmeans_plusplus` and the plain / per-vector-scalar-weight
clustering path to :class:`sklearn.cluster.KMeans`.  The bespoke Lloyd loop is
kept only for the two capabilities no library k-means offers — general
per-element (Hessian) importance weights and quantization-aware clustering.

The building blocks:
    weighted_kmeans        - Lloyd's algorithm w/ k-means++ init, importance
                             weights, empty-cluster reseeding, optional
                             quantization-aware centroid projection; delegates to
                             scikit-learn's KMeans where feature-compatible.
    CentroidQuantizer      - snaps centroids onto a LoFloat exp/mantissa grid.
    VectorQuantizer        - plain VQ of a weight matrix (grouping, encode,
                             decode, bit accounting).
    ProductQuantizer       - PQ: split each vector into M subspaces, one
                             codebook per subspace.
    ResidualQuantizer      - RVQ / additive VQ: a stack of codebooks whose
                             decoded vectors sum to the reconstruction.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import torch
from sklearn.cluster import KMeans, kmeans_plusplus

try:  # LoFloat is the project's low-precision float library; optional at import.
    import LoFloat as lof
except Exception:  # pragma: no cover - allows using the VQ core without LoFloat
    lof = None


# ---------------------------------------------------------------------------
# Distances
# ---------------------------------------------------------------------------

def pairwise_sqdist(X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between every row of ``X`` and every row of ``C``.

    Thin wrapper over :func:`torch.cdist` (which internally uses the
    ``||x-c||^2 = ||x||^2 - 2 x·c + ||c||^2`` expansion, so it stays a couple of
    matmuls rather than a materialized ``(N, K, d)`` difference tensor); the
    Euclidean distance is squared to give the k-means distortion metric.

    Args:
        X: ``(N, d)`` data vectors.
        C: ``(K, d)`` centroids.
    Returns:
        ``(N, K)`` non-negative squared distances.
    """
    return torch.cdist(X, C).square()


def weighted_pairwise_sqdist(
    X: torch.Tensor, C: torch.Tensor, W: torch.Tensor
) -> torch.Tensor:
    """Per-element weighted squared distance ``sum_j W[n,j] (X[n,j]-C[k,j])^2``.

    Expanded the same way as :func:`pairwise_sqdist` so it is three matmul-shaped
    reductions and never forms an ``(N, K, d)`` tensor::

        D[n,k] = sum_j W[n,j] X[n,j]^2  -  2 (W*X)[n,:] · C[k,:]  +  W[n,:] · (C*C)[k,:]

    Args:
        X: ``(N, d)`` data vectors.
        C: ``(K, d)`` centroids.
        W: ``(N, d)`` non-negative per-element importance weights.
    Returns:
        ``(N, K)`` weighted squared distances (clamped non-negative).
    """
    WX = W * X                                     # (N, d)
    term1 = (WX * X).sum(dim=1, keepdim=True)      # (N, 1)
    term2 = WX @ C.t()                             # (N, K)
    term3 = W @ (C * C).t()                        # (N, K)
    d2 = term1 - 2.0 * term2 + term3
    return d2.clamp_min_(0.0)


# ---------------------------------------------------------------------------
# LoFloat centroid quantizer
# ---------------------------------------------------------------------------

@dataclass
class CentroidQuantizer:
    """Snaps centroids onto a LoFloat exponent/mantissa grid.

    Applied inside :func:`weighted_kmeans` after every centroid update, so the
    codebook stays representable in the target low-precision float format at
    every iteration (quantization-aware clustering).  ``None`` anywhere a
    ``CentroidQuantizer`` is accepted means "keep centroids in full precision".

    Args:
        exp_bits: exponent field width of the target LoFloat format.
        mantissa_bits: mantissa (fraction) field width.
    """

    exp_bits: int = 4
    mantissa_bits: int = 3

    def __post_init__(self):
        if lof is None:
            raise ImportError(
                "CentroidQuantizer requires the LoFloat library (import failed)."
            )

    def __call__(self, C: torch.Tensor) -> torch.Tensor:
        return lof.exp_mant_quantize(C, self.exp_bits, self.mantissa_bits)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def kmeans_plus_plus_init(
    X: torch.Tensor,
    k: int,
    sample_weight: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """k-means++ seeding (Arthur & Vassilvitskii, 2007).

    Delegates to :func:`sklearn.cluster.kmeans_plusplus`, which picks centers one
    at a time, each sampled with probability proportional to its squared distance
    from the closest center already chosen (optionally scaled by
    ``sample_weight``) — an ``O(log k)``-competitive starting distortion.  The
    data is moved to CPU/NumPy for the call and the seeds are returned on ``X``'s
    original device/dtype.

    Args:
        X: ``(N, d)`` data.
        k: number of centers.
        sample_weight: optional ``(N,)`` per-vector weight biasing the sampling.
        generator: optional torch RNG; its seed is forwarded as scikit-learn's
            ``random_state`` for reproducibility.
    Returns:
        ``(k, d)`` initial centroids (copies, safe to mutate).
    """
    N, _ = X.shape
    k = min(k, N)
    X_np = X.detach().cpu().numpy()
    sw_np = None if sample_weight is None else sample_weight.detach().cpu().numpy()
    random_state = None if generator is None else int(generator.initial_seed() % (2 ** 32))
    centers, _ = kmeans_plusplus(
        X_np, n_clusters=k, sample_weight=sw_np, random_state=random_state
    )
    return torch.as_tensor(centers, dtype=X.dtype, device=X.device).clone()


# ---------------------------------------------------------------------------
# Weighted k-means (Lloyd's algorithm)
# ---------------------------------------------------------------------------

@dataclass
class KMeansResult:
    """Output of :func:`weighted_kmeans`."""

    centroids: torch.Tensor          # (K, d)
    assignments: torch.Tensor        # (N,) int64 index into centroids
    inertia: float                   # final (weighted) distortion
    n_iter: int                      # iterations actually run
    history: list = field(default_factory=list)  # inertia per iteration


def weighted_kmeans(
    X: torch.Tensor,
    k: int,
    weights: Optional[torch.Tensor] = None,
    n_iters: int = 100,
    tol: float = 1e-4,
    init: str = "k-means++",
    n_init: int = 1,
    centroid_quantizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    generator: Optional[torch.Generator] = None,
    verbose: bool = False,
) -> KMeansResult:
    """Lloyd's algorithm minimizing (optionally weighted) squared distortion.

    Objective::

        J = sum_n  W[n,:] · (X[n,:] - C[a(n),:])^2          (elementwise, then dot)

    with assignment ``a(n) = argmin_k`` and the weighted centroid update
    ``C[k,j] = (sum_{n: a(n)=k} W[n,j] X[n,j]) / (sum_{n: a(n)=k} W[n,j])`` — i.e.
    a per-dimension importance-weighted mean, the closed-form minimizer of ``J``.

    Args:
        X: ``(N, d)`` data vectors (float).
        k: number of centroids.
        weights: importance weights. Accepted shapes:
            * ``None``               -> uniform (plain k-means);
            * ``(d,)``               -> per-dimension, shared across all vectors;
            * ``(N,)``               -> per-vector scalar importance;
            * ``(N, d)``             -> fully general per-element weights.
        n_iters: max Lloyd iterations.
        tol: stop when relative inertia improvement drops below this.
        init: ``"k-means++"`` or ``"random"``.
        n_init: number of independent restarts; the lowest-inertia one is kept.
        centroid_quantizer: optional callable applied to the centroids after
            every update (e.g. a :class:`CentroidQuantizer`) for
            quantization-aware clustering.
        generator: optional torch RNG for reproducibility.
        verbose: print per-iteration inertia.
    Returns:
        :class:`KMeansResult`.
    """
    if X.dim() != 2:
        raise ValueError(f"X must be (N, d); got shape {tuple(X.shape)}")
    N, d = X.shape
    X = X.float()

    # Hybrid dispatch: scikit-learn's tuned KMeans handles the plain and
    # per-vector-scalar-weight cases; the custom Lloyd loop below is used only
    # for the two things no library k-means supports — per-dimension / general
    # per-element importance weights, and quantization-aware clustering (a
    # ``centroid_quantizer`` applied after every centroid update).
    kind = _weight_kind(weights, N, d)
    if centroid_quantizer is None and kind in ("none", "sample"):
        return _sklearn_kmeans(
            X, k, weights, kind, n_iters, tol, init, n_init, generator, verbose
        )

    W = _broadcast_weights(weights, N, d, X.device, X.dtype)  # (N, d) or None

    # Per-vector sampling weight for k-means++ seeding.
    sample_w = None
    if W is not None:
        sample_w = W.sum(dim=1)
        sample_w = sample_w / sample_w.sum().clamp_min(1e-12)

    best: Optional[KMeansResult] = None
    for _ in range(max(1, n_init)):
        res = _kmeans_single(
            X, k, W, sample_w, n_iters, tol, init,
            centroid_quantizer, generator, verbose,
        )
        if best is None or res.inertia < best.inertia:
            best = res
    return best


def _weight_kind(weights, N, d):
    """Classify the importance-weight argument so :func:`weighted_kmeans` can
    dispatch.  Precedence matches :func:`_broadcast_weights` (per-dimension is
    tried before per-vector when ``N == d``).

    Returns one of ``"none"``, ``"sample"`` (per-vector ``(N,)`` scalar — the
    only weighting scikit-learn's KMeans can express), ``"dim"`` (per-dimension
    ``(d,)``) or ``"elem"`` (general ``(N, d)``).
    """
    if weights is None:
        return "none"
    if weights.dim() == 1 and weights.numel() == d:
        return "dim"
    if weights.dim() == 1 and weights.numel() == N:
        return "sample"
    if tuple(weights.shape) == (N, d):
        return "elem"
    raise ValueError(
        f"weights shape {tuple(weights.shape)} incompatible with X ({N}, {d}); "
        "expected (d,), (N,), or (N, d)."
    )


def _sklearn_kmeans(X, k, weights, kind, n_iters, tol, init, n_init, generator, verbose):
    """Plain / per-vector-scalar-weight k-means via :class:`sklearn.cluster.KMeans`.

    ``X`` is moved to CPU/NumPy for the fit; centroids and assignments come back
    on ``X``'s device/dtype.  scikit-learn does not expose a per-iteration inertia
    trace, so :attr:`KMeansResult.history` holds only the final value.
    """
    k = min(k, X.shape[0])
    sample_weight = None
    if kind == "sample":
        sw = weights.to(device=X.device, dtype=X.dtype)
        if (sw < 0).any():
            raise ValueError("importance weights must be non-negative.")
        sample_weight = sw.detach().cpu().numpy()
    random_state = None if generator is None else int(generator.initial_seed() % (2 ** 32))
    km = KMeans(
        n_clusters=k, init=init, n_init=n_init, max_iter=n_iters,
        tol=tol, random_state=random_state, verbose=int(verbose),
    )
    km.fit(X.detach().cpu().numpy(), sample_weight=sample_weight)
    centroids = torch.as_tensor(km.cluster_centers_, dtype=X.dtype, device=X.device)
    assign = torch.as_tensor(km.labels_, dtype=torch.long, device=X.device)
    inertia = float(km.inertia_)
    return KMeansResult(centroids, assign, inertia, int(km.n_iter_), [inertia])


def _broadcast_weights(weights, N, d, device, dtype):
    """Normalize the several accepted weight shapes to ``(N, d)`` (or ``None``)."""
    if weights is None:
        return None
    w = weights.to(device=device, dtype=dtype)
    if w.dim() == 1 and w.numel() == d:          # (d,) per-dimension
        w = w.unsqueeze(0).expand(N, d)
    elif w.dim() == 1 and w.numel() == N:        # (N,) per-vector scalar
        w = w.unsqueeze(1).expand(N, d)
    elif w.shape == (N, d):                       # (N, d) general
        pass
    else:
        raise ValueError(
            f"weights shape {tuple(w.shape)} incompatible with X ({N}, {d}); "
            "expected (d,), (N,), or (N, d)."
        )
    if (w < 0).any():
        raise ValueError("importance weights must be non-negative.")
    return w.contiguous()


def _assign(X, C, W):
    """Nearest-centroid assignment + inertia. ``W`` may be None (uniform)."""
    d2 = pairwise_sqdist(X, C) if W is None else weighted_pairwise_sqdist(X, C, W)
    min_d2, assign = d2.min(dim=1)
    return assign, min_d2.sum().item()


def _kmeans_single(
    X, k, W, sample_w, n_iters, tol, init,
    centroid_quantizer, generator, verbose,
):
    N, d = X.shape
    k = min(k, N)
    if init == "k-means++":
        C = kmeans_plus_plus_init(X, k, sample_w, generator)
    elif init == "random":
        idx = torch.randperm(N, device=X.device, generator=generator)[:k]
        C = X[idx].clone()
    else:
        raise ValueError(f"unknown init '{init}'")
    if centroid_quantizer is not None:
        C = centroid_quantizer(C)

    prev_inertia = math.inf
    history = []
    assign = torch.zeros(N, dtype=torch.long, device=X.device)
    it = 0
    for it in range(n_iters):
        assign, inertia = _assign(X, C, W)
        history.append(inertia)
        if verbose:
            print(f"  [kmeans] iter {it:3d}  inertia {inertia:.6g}")

        # --- weighted centroid update via scatter-add ---
        if W is None:
            num = torch.zeros_like(C).index_add_(0, assign, X)
            cnt = torch.zeros(k, device=X.device, dtype=X.dtype).index_add_(
                0, assign, torch.ones(N, device=X.device, dtype=X.dtype)
            )
            denom = cnt.unsqueeze(1).clamp_min(1e-12)
            newC = num / denom
            empty = cnt == 0
        else:
            num = torch.zeros_like(C).index_add_(0, assign, W * X)
            den = torch.zeros_like(C).index_add_(0, assign, W)
            newC = num / den.clamp_min(1e-12)
            empty = den.sum(dim=1) == 0

        # --- reseed empty clusters at the worst-fit points ---
        if empty.any():
            newC = _reseed_empty(X, C, W, assign, newC, empty)

        if centroid_quantizer is not None:
            newC = centroid_quantizer(newC)
        C = newC

        if prev_inertia < math.inf:
            rel = (prev_inertia - inertia) / max(prev_inertia, 1e-12)
            if 0 <= rel < tol:
                break
        prev_inertia = inertia

    assign, inertia = _assign(X, C, W)
    return KMeansResult(C, assign, inertia, it + 1, history)


def _reseed_empty(X, C, W, assign, newC, empty):
    """Move empty centroids onto the currently worst-represented data points.

    Splitting the highest-distortion clusters is the standard fix for the empty
    cluster / dead codeword problem and keeps ``K`` codewords all in use.
    """
    d2 = pairwise_sqdist(X, C) if W is None else weighted_pairwise_sqdist(X, C, W)
    per_point = d2.gather(1, assign.unsqueeze(1)).squeeze(1)  # dist to own center
    empty_idx = torch.nonzero(empty, as_tuple=False).squeeze(1)
    worst = torch.topk(per_point, empty_idx.numel()).indices
    newC[empty_idx] = X[worst]
    return newC


# ---------------------------------------------------------------------------
# Reshaping a weight matrix into vectors
# ---------------------------------------------------------------------------

def _to_vectors(W: torch.Tensor, d: int) -> Tuple[torch.Tensor, torch.Size, int]:
    """Flatten ``W`` and chop it into ``d``-length vectors.

    ``W`` is flattened in row-major (C) order; the flat length is padded with
    zeros up to a multiple of ``d`` so arbitrary ``d`` is supported.  Returns the
    ``(N, d)`` vectors, the original shape, and the pad amount so
    :func:`_from_vectors` can invert it exactly.
    """
    orig_shape = W.shape
    flat = W.reshape(-1)
    pad = (-flat.numel()) % d
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    return flat.reshape(-1, d), orig_shape, pad


def _from_vectors(vecs: torch.Tensor, orig_shape: torch.Size, pad: int) -> torch.Tensor:
    """Inverse of :func:`_to_vectors`."""
    flat = vecs.reshape(-1)
    if pad:
        flat = flat[:-pad]
    return flat.reshape(orig_shape)


# ---------------------------------------------------------------------------
# Per-channel power-of-two scaling
# ---------------------------------------------------------------------------

def round_fp32_mantissa(x: torch.Tensor, mantissa_bits: int) -> torch.Tensor:
    """Round a float32 tensor to ``mantissa_bits`` of mantissa, keeping fp32's
    8-bit exponent — i.e. the ``e8m{mantissa_bits}`` low-precision float format.

    Because an ``e8`` float shares fp32's exponent field (8 bits, bias 127),
    quantizing to ``e8mM`` is exactly truncating the fp32 mantissa from 23 bits
    to ``M`` with round-to-nearest-even; carries propagate naturally into the
    exponent through the contiguous exp+mantissa bit field.  This is a correct,
    device-agnostic pure-torch reference for centroid *storage* precision (it
    does not model P3109 inf/NaN exponent reservations, irrelevant for in-range
    weight centroids).  ``mantissa_bits >= 23`` is a no-op (already fp32).

    Args:
        x: float32 tensor (computed in full precision).
        mantissa_bits: retained mantissa width ``M`` (0..23).
    Returns:
        ``x`` rounded to ``e8mM``, same shape/dtype/device.
    """
    if mantissa_bits >= 23:
        return x.clone()
    if mantissa_bits < 0:
        raise ValueError("mantissa_bits must be >= 0")
    xi = x.contiguous().view(torch.int32)
    sign = xi & (-2147483648)                      # top bit (0x80000000)
    mag = xi & 0x7FFFFFFF                          # exponent + mantissa
    shift = 23 - mantissa_bits
    lsb = (mag >> shift) & 1                        # kept least-significant bit
    bias = (1 << (shift - 1)) - 1 + lsb            # round-to-nearest-even bias
    mag = (mag + bias) & (0x7FFFFFFF ^ ((1 << shift) - 1))  # add, clear dropped bits
    return (sign | mag).view(torch.float32)


def _po2_channel_scale(W: torch.Tensor, dim: int) -> torch.Tensor:
    """Per-channel power-of-two scale for weight tensor ``W``.

    Returns a tensor broadcastable to ``W`` (length ``W.shape[dim]`` along
    ``dim``, 1 elsewhere) whose entries are ``2**round(log2(max|W_channel|))`` —
    the nearest power of two to each channel's peak magnitude.  Dividing ``W`` by
    this scale before clustering aligns every channel to roughly unit range so a
    single shared codebook fits them all; because the scale is an *exact* power
    of two, folding it back on decode is lossless (a mantissa-preserving exponent
    shift, no extra rounding).

    Args:
        W: weight tensor.
        dim: channel axis to scale along (0 = per output channel for both
            ``nn.Conv2d`` ``(out, in, kH, kW)`` and ``nn.Linear`` ``(out, in)``).
    """
    other = tuple(i for i in range(W.dim()) if i != dim)
    amax = W.abs().amax(dim=other, keepdim=True).clamp_min(1e-12)
    return torch.exp2(torch.round(torch.log2(amax)))


# ---------------------------------------------------------------------------
# Plain vector quantizer
# ---------------------------------------------------------------------------

class VectorQuantizer:
    """Plain VQ of a weight tensor: one shared codebook of ``K`` ``d``-vectors.

    Reshapes the weight into ``d``-dimensional vectors, clusters them with
    (optionally Hessian-weighted) k-means, and stores each vector as an index
    into the codebook.  Bit rate is ``log2(K) / d`` bits per weight plus the
    (amortized) codebook itself.

    Args:
        vector_dim: length ``d`` of each quantized vector.
        codebook_size: number of centroids ``K``.
        centroid_quantizer: optional LoFloat :class:`CentroidQuantizer` so the
            stored codebook is itself low precision.
        scale_dim: if not ``None``, per-channel power-of-two scaling along this
            axis (0 = per output channel).  Each channel is divided by the
            nearest power of two to its peak magnitude before clustering and the
            scale is folded back losslessly on decode, so one shared codebook can
            cover channels of very different magnitude — the standard fix for
            low-bit PTQ on CNNs.  ``None`` disables it (single global codebook).
        kmeans_kwargs: extra keyword args forwarded to :func:`weighted_kmeans`.
    """

    def __init__(
        self,
        vector_dim: int,
        codebook_size: int,
        centroid_quantizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_dim: Optional[int] = None,
        **kmeans_kwargs,
    ):
        self.d = int(vector_dim)
        self.K = int(codebook_size)
        self.centroid_quantizer = centroid_quantizer
        self.scale_dim = scale_dim
        self.kmeans_kwargs = kmeans_kwargs
        self.codebook: Optional[torch.Tensor] = None      # (K, d)
        self.indices: Optional[torch.Tensor] = None        # (N,)
        self._orig_shape: Optional[torch.Size] = None
        self._pad: int = 0
        self._scale: Optional[torch.Tensor] = None          # per-channel PO2 scale
        self._inertia: float = float("nan")

    def fit(self, W: torch.Tensor, weights: Optional[torch.Tensor] = None) -> "VectorQuantizer":
        """Build the codebook from weight tensor ``W``.

        Args:
            W: weight tensor of any shape.
            weights: optional importance weights, *same shape as* ``W`` (e.g. a
                diagonal Hessian). Reshaped into vectors the same way as ``W``.
        """
        if self.scale_dim is not None:
            self._scale = _po2_channel_scale(W, self.scale_dim)
            W = W / self._scale
        vecs, self._orig_shape, self._pad = _to_vectors(W, self.d)
        vw = None
        if weights is not None:
            vw, _, _ = _to_vectors(weights, self.d)
        res = weighted_kmeans(
            vecs, self.K, weights=vw,
            centroid_quantizer=self.centroid_quantizer,
            **self.kmeans_kwargs,
        )
        self.codebook = res.centroids
        self.indices = res.assignments
        self._inertia = res.inertia
        return self

    def encode(self, W: torch.Tensor) -> torch.Tensor:
        """Return the ``(N,)`` codebook indices for a (new) weight tensor."""
        if self.codebook is None:
            raise RuntimeError("call fit() before encode().")
        if self._scale is not None:
            W = W / self._scale         # codebook lives in normalized space
        vecs, _, _ = _to_vectors(W, self.d)
        assign, _ = _assign(vecs, self.codebook, None)
        return assign

    def decode(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reconstruct the weight tensor from indices (defaults to fitted ones)."""
        if self.codebook is None:
            raise RuntimeError("call fit() before decode().")
        idx = self.indices if indices is None else indices
        vecs = self.codebook[idx]
        out = _from_vectors(vecs, self._orig_shape, self._pad)
        if self._scale is not None:
            out = out * self._scale
        return out

    def quantize(self, W: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convenience: ``fit`` then ``decode`` — returns the reconstruction."""
        return self.fit(W, weights).decode()

    def bits_per_weight(
        self, include_codebook: bool = True, centroid_bits: int = 16,
        scale_bits: int = 8,
    ) -> float:
        """Effective bits per weight.

        Args:
            include_codebook: amortize the codebook storage over all weights.
            centroid_bits: bits to store each centroid scalar (16 for fp16; set
                to the LoFloat element width when a ``centroid_quantizer`` is used).
            scale_bits: bits to store each per-channel power-of-two scale (only
                counted when ``scale_dim`` was set; the scale is just an exponent).
        """
        n_vectors = self.indices.numel()
        index_bits = math.ceil(math.log2(self.K)) if self.K > 1 else 0
        total = index_bits * n_vectors
        if include_codebook:
            total += self.K * self.d * centroid_bits
        if self._scale is not None:
            total += self._scale.numel() * scale_bits
        return total / (n_vectors * self.d)


# ---------------------------------------------------------------------------
# Product quantization
# ---------------------------------------------------------------------------

class ProductQuantizer:
    """Product Quantization (Jégou et al., 2011): split each vector into ``M``
    disjoint subvectors and quantize each subspace with its own codebook.

    A ``d``-vector is split into ``M`` subvectors of length ``d/M``; subspace
    ``m`` gets a ``K``-entry codebook. A vector is then ``M`` indices, giving an
    effective codebook of ``K**M`` at ``M*log2(K)`` bits, i.e.
    ``M*log2(K) / d`` bits per weight.

    Args:
        vector_dim: full vector length ``d`` (must be divisible by ``n_subspaces``).
        n_subspaces: number of subspaces ``M``.
        codebook_size: entries per subspace codebook ``K``.
        centroid_quantizer: optional LoFloat centroid quantizer (shared).
        scale_dim: if not ``None``, per-channel power-of-two scaling along this
            axis before quantization (see :class:`VectorQuantizer`).
        kmeans_kwargs: forwarded to :func:`weighted_kmeans`.
    """

    def __init__(
        self,
        vector_dim: int,
        n_subspaces: int,
        codebook_size: int,
        centroid_quantizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_dim: Optional[int] = None,
        **kmeans_kwargs,
    ):
        if vector_dim % n_subspaces != 0:
            raise ValueError(
                f"vector_dim ({vector_dim}) must be divisible by n_subspaces "
                f"({n_subspaces})."
            )
        self.d = int(vector_dim)
        self.M = int(n_subspaces)
        self.dsub = self.d // self.M
        self.K = int(codebook_size)
        self.centroid_quantizer = centroid_quantizer
        self.scale_dim = scale_dim
        self.kmeans_kwargs = kmeans_kwargs
        self.codebooks: list = []               # M x (K, dsub)
        self.indices: Optional[torch.Tensor] = None   # (N, M)
        self._orig_shape = None
        self._pad = 0
        self._scale: Optional[torch.Tensor] = None

    def fit(self, W: torch.Tensor, weights: Optional[torch.Tensor] = None) -> "ProductQuantizer":
        if self.scale_dim is not None:
            self._scale = _po2_channel_scale(W, self.scale_dim)
            W = W / self._scale
        vecs, self._orig_shape, self._pad = _to_vectors(W, self.d)
        vw = None
        if weights is not None:
            vw, _, _ = _to_vectors(weights, self.d)
        N = vecs.shape[0]
        self.codebooks = []
        self.indices = torch.empty((N, self.M), dtype=torch.long, device=vecs.device)
        for m in range(self.M):
            sl = slice(m * self.dsub, (m + 1) * self.dsub)
            sub = vecs[:, sl]
            subw = vw[:, sl] if vw is not None else None
            res = weighted_kmeans(
                sub, self.K, weights=subw,
                centroid_quantizer=self.centroid_quantizer,
                **self.kmeans_kwargs,
            )
            self.codebooks.append(res.centroids)
            self.indices[:, m] = res.assignments
        return self

    def decode(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.codebooks:
            raise RuntimeError("call fit() before decode().")
        idx = self.indices if indices is None else indices
        subs = [self.codebooks[m][idx[:, m]] for m in range(self.M)]
        vecs = torch.cat(subs, dim=1)
        out = _from_vectors(vecs, self._orig_shape, self._pad)
        if self._scale is not None:
            out = out * self._scale
        return out

    def quantize(self, W: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fit(W, weights).decode()

    def bits_per_weight(
        self, include_codebook: bool = True, centroid_bits: int = 16,
        scale_bits: int = 8,
    ) -> float:
        n_vectors = self.indices.shape[0]
        index_bits = math.ceil(math.log2(self.K)) if self.K > 1 else 0
        total = index_bits * self.M * n_vectors
        if include_codebook:
            total += self.M * self.K * self.dsub * centroid_bits
        if self._scale is not None:
            total += self._scale.numel() * scale_bits
        return total / (n_vectors * self.d)


# ---------------------------------------------------------------------------
# Residual (additive) vector quantization
# ---------------------------------------------------------------------------

class ResidualQuantizer:
    """Residual VQ / additive quantization: a stack of ``n_stages`` codebooks
    whose decoded vectors *sum* to the reconstruction.

    Stage 0 quantizes the vectors; stage 1 quantizes the residual left by stage
    0; and so on. With ``S`` stages of ``K_s`` entries each the rate is
    ``sum_s log2(K_s) / d`` bits per weight, and the reconstruction is
    ``sum_s codebook_s[idx_s]``. This is the greedy (sequential) RVQ variant;
    full additive-quantization beam search over stages is a future extension.

    Args:
        vector_dim: vector length ``d``.
        n_stages: number of residual codebooks ``S``.
        codebook_size: entries per stage. Either an ``int`` (shared by every
            stage) or a length-``S`` sequence of per-stage sizes ``[K_0, ...,
            K_{S-1}]`` — a *cheaper* residual codebook (e.g. ``[256, 16]``) lets
            you refine a coarse base at a fine bit-rate granularity instead of
            paying a full extra ``log2(K)`` bits per stage.
        centroid_quantizer: optional LoFloat centroid quantizer (shared).
        scale_dim: if not ``None``, per-channel power-of-two scaling along this
            axis before quantization (see :class:`VectorQuantizer`).
        kmeans_kwargs: forwarded to :func:`weighted_kmeans`.
    """

    def __init__(
        self,
        vector_dim: int,
        n_stages: int,
        codebook_size,
        centroid_quantizer: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        scale_dim: Optional[int] = None,
        **kmeans_kwargs,
    ):
        self.d = int(vector_dim)
        self.S = int(n_stages)
        if isinstance(codebook_size, int):
            self.Ks = [int(codebook_size)] * self.S
        else:
            self.Ks = [int(k) for k in codebook_size]
            if len(self.Ks) != self.S:
                raise ValueError(
                    f"codebook_size list has {len(self.Ks)} entries but "
                    f"n_stages={self.S}."
                )
        self.K = self.Ks[0]                     # back-compat convenience
        self.centroid_quantizer = centroid_quantizer
        self.scale_dim = scale_dim
        self.kmeans_kwargs = kmeans_kwargs
        self.codebooks: list = []               # S x (K_s, d)
        self.indices: Optional[torch.Tensor] = None   # (N, S)
        self._orig_shape = None
        self._pad = 0
        self._scale: Optional[torch.Tensor] = None

    def fit(self, W: torch.Tensor, weights: Optional[torch.Tensor] = None) -> "ResidualQuantizer":
        if self.scale_dim is not None:
            self._scale = _po2_channel_scale(W, self.scale_dim)
            W = W / self._scale
        vecs, self._orig_shape, self._pad = _to_vectors(W, self.d)
        vw = None
        if weights is not None:
            vw, _, _ = _to_vectors(weights, self.d)
        N = vecs.shape[0]
        residual = vecs.clone()
        self.codebooks = []
        self.indices = torch.empty((N, self.S), dtype=torch.long, device=vecs.device)
        for s in range(self.S):
            res = weighted_kmeans(
                residual, self.Ks[s], weights=vw,
                centroid_quantizer=self.centroid_quantizer,
                **self.kmeans_kwargs,
            )
            self.codebooks.append(res.centroids)
            self.indices[:, s] = res.assignments
            residual = residual - res.centroids[res.assignments]
        return self

    def decode(self, indices: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.codebooks:
            raise RuntimeError("call fit() before decode().")
        idx = self.indices if indices is None else indices
        vecs = sum(self.codebooks[s][idx[:, s]] for s in range(self.S))
        out = _from_vectors(vecs, self._orig_shape, self._pad)
        if self._scale is not None:
            out = out * self._scale
        return out

    def quantize(self, W: torch.Tensor, weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.fit(W, weights).decode()

    def bits_per_weight(
        self, include_codebook: bool = True, centroid_bits: int = 16,
        scale_bits: int = 8,
    ) -> float:
        n_vectors = self.indices.shape[0]
        index_bits = sum(math.ceil(math.log2(k)) if k > 1 else 0 for k in self.Ks)
        total = index_bits * n_vectors
        if include_codebook:
            total += sum(self.Ks) * self.d * centroid_bits
        if self._scale is not None:
            total += self._scale.numel() * scale_bits
        return total / (n_vectors * self.d)


__all__ = [
    "pairwise_sqdist",
    "weighted_pairwise_sqdist",
    "CentroidQuantizer",
    "kmeans_plus_plus_init",
    "weighted_kmeans",
    "KMeansResult",
    "VectorQuantizer",
    "ProductQuantizer",
    "ResidualQuantizer",
]
