"""Acceptance tests for the UNIFORM per-precision / per-operand MX tile selector.

Contract under test:
  uniform_mx_tile(matrices, element_format, scale_format, scale_target, frob_tau,
                  start=16, max_bits=2048) -> (rows, cols)
    - returns the LARGEST-AREA tile (non-unit axes multiples of 8; may exceed any
      matrix's dims; area*element_format.total_bits <= max_bits) such that
      max over matrices of  ||X - Q_tile(X)||_F / ||X||_F  <= frob_tau.
    - deterministic (no forward pass); returns (start,1) as a floor if even that
      violates tau.
  greedy_sensitivity gains `tile_mode` ('per_layer'|'uniform') and `frob_tau`.

Owned by the monitor. The implementer must NOT edit this file.
"""
import inspect
import pytest
import torch

lof = pytest.importorskip("LoFloat")
ss = pytest.importorskip("sensitivity_search")

RNE = lof.RoundingMode.RoundToNearestEven


def _elem():
    return lof.create_p3109_params(8, 4)


def _scale():
    return lof.create_e8m0_params()


def _elem_bits(elem):
    return getattr(elem, "total_bits", None) or getattr(elem, "bitwidth", None) or 8


def _rel_frob(X, tile, elem, scale, target):
    Q = lof.virtual_mx_round(X.clone(), tile, elem, scale,
                             scale_target=float(target), round_mode=RNE)
    return (X - Q).norm().item() / (X.norm().item() + 1e-12)


def _gentle(shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(shape, generator=g) * 0.02        # narrow range -> low Frob


def _wide(shape, seed):
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=g) * 0.02
    flat = w.reshape(-1)
    idx = torch.randperm(flat.numel(), generator=g)
    flat[idx[:flat.numel() // 40]] = 3.0                 # outliers -> big dynamic range
    return flat.reshape(shape)


def _require():
    assert hasattr(ss, "uniform_mx_tile"), "uniform_mx_tile not implemented"


# --------------------------------------------------------------------------- #
# 1. signature wiring
# --------------------------------------------------------------------------- #
def test_greedy_has_tile_mode_and_frob_tau():
    p = inspect.signature(ss.greedy_sensitivity).parameters
    assert "tile_mode" in p and "frob_tau" in p


def test_greedy_has_tile_criterion():
    assert "tile_criterion" in inspect.signature(ss.greedy_sensitivity).parameters


# --------------------------------------------------------------------------- #
# 1b. shared search core: uniform_tile_search with a monotone mock predicate
# --------------------------------------------------------------------------- #
def test_uniform_tile_search_core_maximizes_under_monotone_accept():
    assert hasattr(ss, "uniform_tile_search"), "uniform_tile_search not implemented"
    elem = _elem()                       # 8-bit -> bit-budget area cap = 256
    K = 64                               # accept iff area <= 64 (monotone in area)
    accept = lambda t: t[0] * t[1] <= K
    tile = ss.uniform_tile_search(accept, elem, start=16, max_bits=2048)
    r, c = tile
    if r > 1:
        assert r % 8 == 0, tile
    if c > 1:
        assert c % 8 == 0, tile
    assert accept(tile), tile                     # respects the predicate
    assert r * c * _elem_bits(elem) <= 2048       # respects the bit budget
    assert r * c == 64, tile                       # maximal: largest x8 area <= 64


def test_uniform_tile_search_floor_when_nothing_accepts():
    assert hasattr(ss, "uniform_tile_search"), "uniform_tile_search not implemented"
    elem = _elem()
    tile = ss.uniform_tile_search(lambda t: False, elem, start=16, max_bits=2048)
    assert tile == (16, 1), tile


# --------------------------------------------------------------------------- #
# 2. returned tile actually keeps max relative Frob <= tau, and is well-formed
# --------------------------------------------------------------------------- #
def test_uniform_tile_respects_tau_and_invariants():
    _require()
    elem, scale = _elem(), _scale()
    mats = [_gentle((128, 128), i) for i in range(3)]
    tau = 0.15
    tile = ss.uniform_mx_tile(mats, elem, scale, 1.0, tau)
    assert isinstance(tile, tuple) and len(tile) == 2
    r, c = tile
    if r > 1:
        assert r % 8 == 0, tile
    if c > 1:
        assert c % 8 == 0, tile
    assert r * c * _elem_bits(elem) <= 2048, (tile, _elem_bits(elem))
    worst = max(_rel_frob(X, tile, elem, scale, 1.0) for X in mats)
    assert worst <= tau + 1e-6, (tile, worst, tau)


# --------------------------------------------------------------------------- #
# 3. it is the WORST matrix that binds (max-over-class), not the average
# --------------------------------------------------------------------------- #
def test_uniform_tile_bound_by_worst_matrix():
    _require()
    elem, scale = _elem(), _scale()
    # one wide matrix mixed with gentle ones; the wide one must stay under tau too
    mats = [_gentle((128, 128), 1), _gentle((128, 128), 2), _wide((128, 128), 9)]
    tau = 0.15
    tile = ss.uniform_mx_tile(mats, elem, scale, 1.0, tau)
    for X in mats:
        assert _rel_frob(X, tile, elem, scale, 1.0) <= tau + 1e-6, (tile, "a matrix exceeds tau")


# --------------------------------------------------------------------------- #
# 4. tile may exceed matrix dims (not clamped to the smallest matrix)
# --------------------------------------------------------------------------- #
def test_uniform_tile_can_exceed_matrix():
    _require()
    elem, scale = _elem(), _scale()
    mats = [_gentle((16, 16), i) for i in range(2)]      # tiny matrices, low Frob
    tile = ss.uniform_mx_tile(mats, elem, scale, 1.0, frob_tau=0.9)  # generous tau
    r, c = tile
    assert max(r, c) > 16, f"tile {tile} was clamped to the 16-wide matrix"
    assert r * c * _elem_bits(elem) <= 2048


# --------------------------------------------------------------------------- #
# 5. impossible tau -> graceful (start,1) floor, no crash
# --------------------------------------------------------------------------- #
def test_uniform_tile_floor_on_impossible_tau():
    _require()
    elem, scale = _elem(), _scale()
    mats = [_wide((128, 128), i) for i in range(2)]
    tile = ss.uniform_mx_tile(mats, elem, scale, 1.0, frob_tau=1e-6, start=16)
    assert tile == (16, 1), tile
