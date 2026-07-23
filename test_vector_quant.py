"""Pytest suite for vector_quant.py.

Small CPU tensors + fixed seeds so it runs fast and deterministically.  The
optional LoFloat library is never imported; wherever a ``centroid_quantizer`` is
needed we pass a plain ``lambda C: C.round()`` rounding callable, which still
exercises the quantization-aware custom Lloyd loop.
"""

from __future__ import annotations

import math

import pytest
import torch

import vector_quant as vq
from vector_quant import (
    KMeansResult,
    ProductQuantizer,
    ResidualQuantizer,
    VectorQuantizer,
    pairwise_sqdist,
    weighted_kmeans,
    weighted_pairwise_sqdist,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _brute_sqdist(X: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Reference (N, K) squared distance via an explicit (N, K, d) tensor."""
    diff = X.unsqueeze(1) - C.unsqueeze(0)          # (N, K, d)
    return (diff * diff).sum(dim=2)


def _brute_weighted_sqdist(X, C, W) -> torch.Tensor:
    """Reference per-element weighted (N, K) squared distance."""
    diff = X.unsqueeze(1) - C.unsqueeze(0)          # (N, K, d)
    return (W.unsqueeze(1) * diff * diff).sum(dim=2)


def _gen(seed: int = 0) -> torch.Generator:
    g = torch.Generator()
    g.manual_seed(seed)
    return g


# ---------------------------------------------------------------------------
# Distance functions
# ---------------------------------------------------------------------------

def test_pairwise_sqdist_matches_bruteforce():
    torch.manual_seed(0)
    X = torch.randn(11, 4)
    C = torch.randn(5, 4)
    got = pairwise_sqdist(X, C)
    assert got.shape == (11, 5)
    assert torch.all(got >= -1e-5)
    torch.testing.assert_close(got, _brute_sqdist(X, C), rtol=1e-4, atol=1e-4)


def test_weighted_pairwise_sqdist_matches_bruteforce():
    torch.manual_seed(1)
    X = torch.randn(9, 3)
    C = torch.randn(4, 3)
    W = torch.rand(9, 3)                              # non-negative weights
    got = weighted_pairwise_sqdist(X, C, W)
    assert got.shape == (9, 4)
    assert torch.all(got >= 0.0)                      # clamped non-negative
    torch.testing.assert_close(
        got, _brute_weighted_sqdist(X, C, W), rtol=1e-4, atol=1e-4
    )


def test_weighted_reduces_to_unweighted_when_weights_one():
    torch.manual_seed(2)
    X = torch.randn(7, 5)
    C = torch.randn(3, 5)
    W = torch.ones(7, 5)
    torch.testing.assert_close(
        weighted_pairwise_sqdist(X, C, W), pairwise_sqdist(X, C),
        rtol=1e-4, atol=1e-4,
    )


# ---------------------------------------------------------------------------
# _weight_kind classification
# ---------------------------------------------------------------------------

def test_weight_kind_none():
    assert vq._weight_kind(None, 10, 4) == "none"


def test_weight_kind_dim():
    w = torch.ones(4)
    assert vq._weight_kind(w, 10, 4) == "dim"


def test_weight_kind_sample():
    w = torch.ones(10)
    assert vq._weight_kind(w, 10, 4) == "sample"


def test_weight_kind_elem():
    w = torch.ones(10, 4)
    assert vq._weight_kind(w, 10, 4) == "elem"


def test_weight_kind_ambiguous_N_equals_d_is_dim():
    # A (5,) weight when N == d == 5 must classify as per-dimension.
    w = torch.ones(5)
    assert vq._weight_kind(w, 5, 5) == "dim"


def test_weight_kind_invalid_raises():
    w = torch.ones(3, 7)
    with pytest.raises(ValueError):
        vq._weight_kind(w, 10, 4)


# ---------------------------------------------------------------------------
# weighted_kmeans - all four weight kinds
# ---------------------------------------------------------------------------

def _make_data(N=40, d=4, seed=0):
    torch.manual_seed(seed)
    return torch.randn(N, d)


@pytest.mark.parametrize(
    "kind",
    ["none", "sample", "dim", "elem"],
)
def test_weighted_kmeans_basic(kind):
    N, d, K = 40, 4, 5
    X = _make_data(N, d, seed=3)
    if kind == "none":
        weights = None
    elif kind == "sample":
        weights = torch.rand(N) + 0.1
    elif kind == "dim":
        weights = torch.rand(d) + 0.1
    else:  # elem
        weights = torch.rand(N, d) + 0.1

    res = weighted_kmeans(X, K, weights=weights, generator=_gen(7))

    assert isinstance(res, KMeansResult)
    assert res.centroids.shape == (K, d)
    assert res.assignments.shape == (N,)
    # assignments are valid indices in [0, K)
    assert int(res.assignments.min()) >= 0
    assert int(res.assignments.max()) < K
    # inertia non-negative and finite
    assert math.isfinite(res.inertia)
    assert res.inertia >= 0.0
    assert len(res.history) >= 1
    assert all(math.isfinite(h) and h >= 0.0 for h in res.history)


@pytest.mark.parametrize("kind", ["dim", "elem"])
def test_weighted_kmeans_custom_loop_monotonic(kind):
    """Custom Lloyd loop: inertia must not increase across iterations."""
    N, d, K = 60, 4, 6
    X = _make_data(N, d, seed=11)
    if kind == "dim":
        weights = torch.rand(d) + 0.1
    else:
        weights = torch.rand(N, d) + 0.1

    res = weighted_kmeans(X, K, weights=weights, n_iters=50, generator=_gen(5))
    hist = res.history
    assert len(hist) >= 1
    for a, b in zip(hist, hist[1:]):
        # allow tiny numerical slack
        assert b <= a + 1e-4, f"inertia increased: {a} -> {b}"


def test_weighted_kmeans_recovers_blobs():
    """Three well-separated Gaussian blobs -> 3 distinct low-inertia centroids."""
    torch.manual_seed(21)
    centers = torch.tensor([[0.0, 0.0], [20.0, 20.0], [-20.0, 20.0]])
    pts = []
    for c in centers:
        pts.append(c + 0.2 * torch.randn(50, 2))
    X = torch.cat(pts, dim=0)

    res = weighted_kmeans(X, 3, generator=_gen(0), n_init=3)
    assert res.centroids.shape == (3, 2)

    # each recovered centroid should be near a distinct true center
    D = pairwise_sqdist(res.centroids, centers)       # (3, 3)
    nearest = D.argmin(dim=1)
    assert len(set(nearest.tolist())) == 3            # all three matched
    # matched centroid close to its true center
    assert float(D.min(dim=1).values.max()) < 1.0
    # inertia is small relative to a naive single-centroid distortion
    assert res.inertia < 200.0


def test_weighted_kmeans_centroid_quantizer_rounds_centroids():
    """Passing a rounding centroid_quantizer runs the quant-aware custom loop
    and the returned centroids must equal their rounded values."""
    X = _make_data(50, 3, seed=13) * 5.0              # spread so rounding is nontrivial
    res = weighted_kmeans(
        X, 4, weights=None,
        centroid_quantizer=lambda C: C.round(),
        generator=_gen(1),
    )
    assert res.centroids.shape == (4, 3)
    torch.testing.assert_close(res.centroids, res.centroids.round())


def test_weighted_kmeans_empty_cluster_reseed():
    """k larger than the number of distinct points forces empty clusters, which
    must be reseeded by _reseed_empty without crashing; K centroids returned."""
    # Only 2 distinct points, duplicated; request more clusters than distinct.
    base = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    X = base.repeat(4, 1)                              # (8, 3), 2 distinct values
    K = 4
    # per-dimension weights force the custom Lloyd loop (which does reseeding).
    weights = torch.ones(3)
    res = weighted_kmeans(X, K, weights=weights, n_iters=20, generator=_gen(2))
    assert res.centroids.shape == (K, 3)
    assert math.isfinite(res.inertia)
    assert int(res.assignments.max()) < K


def test_weighted_kmeans_rejects_bad_shape():
    X = _make_data(10, 4)
    with pytest.raises(ValueError):
        weighted_kmeans(X.reshape(2, 5, 4), 3)         # not 2-D


# ---------------------------------------------------------------------------
# Determinism (sklearn path)
# ---------------------------------------------------------------------------

def test_sklearn_path_deterministic():
    X = _make_data(50, 4, seed=99)
    r1 = weighted_kmeans(X, 5, weights=None, generator=_gen(123))
    r2 = weighted_kmeans(X, 5, weights=None, generator=_gen(123))
    torch.testing.assert_close(r1.centroids, r2.centroids)
    assert torch.equal(r1.assignments, r2.assignments)
    assert r1.inertia == pytest.approx(r2.inertia)


def test_sample_weight_path_deterministic():
    X = _make_data(50, 4, seed=98)
    w = torch.rand(50, generator=_gen(3)) + 0.1
    r1 = weighted_kmeans(X, 5, weights=w, generator=_gen(55))
    r2 = weighted_kmeans(X, 5, weights=w, generator=_gen(55))
    torch.testing.assert_close(r1.centroids, r2.centroids)
    assert torch.equal(r1.assignments, r2.assignments)


# ---------------------------------------------------------------------------
# VectorQuantizer
# ---------------------------------------------------------------------------

def test_vector_quantizer_roundtrip():
    torch.manual_seed(31)
    W = torch.randn(8, 6)
    vqz = VectorQuantizer(vector_dim=3, codebook_size=8, generator=_gen(1))
    recon = vqz.quantize(W)
    assert recon.shape == W.shape

    # encode/decode round-trip consistency
    idx = vqz.encode(W)
    assert idx.shape == vqz.indices.shape
    recon2 = vqz.decode(idx)
    assert recon2.shape == W.shape

    bpw = vqz.bits_per_weight()
    assert isinstance(bpw, float)
    assert bpw > 0.0


def test_vector_quantizer_requires_fit():
    vqz = VectorQuantizer(vector_dim=3, codebook_size=4)
    with pytest.raises(RuntimeError):
        vqz.encode(torch.randn(4, 3))
    with pytest.raises(RuntimeError):
        vqz.decode()


def test_vector_quantizer_padding_roundtrip_shape():
    torch.manual_seed(32)
    W = torch.randn(5, 7)                              # 35 elements, d=4 -> padding
    vqz = VectorQuantizer(vector_dim=4, codebook_size=6, generator=_gen(2))
    recon = vqz.quantize(W)
    assert recon.shape == W.shape


# ---------------------------------------------------------------------------
# ProductQuantizer
# ---------------------------------------------------------------------------

def test_product_quantizer_roundtrip():
    torch.manual_seed(41)
    W = torch.randn(10, 8)
    pq = ProductQuantizer(
        vector_dim=8, n_subspaces=2, codebook_size=8, generator=_gen(1)
    )
    recon = pq.quantize(W)
    assert recon.shape == W.shape
    assert pq.indices.shape[1] == 2                    # M subspaces
    recon2 = pq.decode()
    assert recon2.shape == W.shape
    bpw = pq.bits_per_weight()
    assert isinstance(bpw, float) and bpw > 0.0


def test_product_quantizer_bad_subspaces():
    with pytest.raises(ValueError):
        ProductQuantizer(vector_dim=7, n_subspaces=2, codebook_size=4)


# ---------------------------------------------------------------------------
# ResidualQuantizer
# ---------------------------------------------------------------------------

def test_residual_quantizer_roundtrip():
    torch.manual_seed(51)
    W = torch.randn(12, 6)
    rq = ResidualQuantizer(
        vector_dim=3, n_stages=2, codebook_size=8, generator=_gen(1)
    )
    recon = rq.quantize(W)
    assert recon.shape == W.shape
    assert rq.indices.shape[1] == 2
    bpw = rq.bits_per_weight()
    assert isinstance(bpw, float) and bpw > 0.0


def test_residual_quantizer_mse_decreases_with_stages():
    torch.manual_seed(52)
    W = torch.randn(60, 4)

    def mse_for_stages(S):
        rq = ResidualQuantizer(
            vector_dim=4, n_stages=S, codebook_size=8, generator=_gen(9)
        )
        recon = rq.quantize(W)
        return float(((recon - W) ** 2).mean())

    m1 = mse_for_stages(1)
    m2 = mse_for_stages(2)
    m3 = mse_for_stages(3)
    tol = 1e-6
    assert m2 <= m1 + tol
    assert m3 <= m2 + tol
