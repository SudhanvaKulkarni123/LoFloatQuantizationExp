"""Acceptance tests for the MX tile-shape search (Implementer Agent 2).

Contract under test (per new_MX_formulation.md + the plan):
  * enable_microscaling / greedy_sensitivity gain a REQUIRED `scale_target`.
  * enable_microscaling sets `module.mx_scale_target`.
  * mx_search(weight, activation, element_format, scale_format, scale_target,
              accuracy_fn=None, max_tile=2048, start=32) -> (rows, cols)
      - accuracy_fn=None  => deterministic, hard no-underflow search only.
      - the returned tile must have ZERO underflow (the hard constraint).
  * lof.set_mx_tile_fields applies a returned tile onto the module.

Owned by the monitor. Implementer Agent 2 must NOT edit this file.
"""
import inspect
import pytest
import torch
import torch.nn as nn

lof = pytest.importorskip("LoFloat")
ss = pytest.importorskip("sensitivity_search")

RNE = lof.RoundingMode.RoundToNearestEven


def _elem():
    return lof.create_p3109_params(8, 4)


def _scale_fmt():
    return lof.create_e8m0_params()


def _heavy_tailed(shape, seed=0):
    """Wide-dynamic-range tensor: mostly small, a few large outliers and a few
    tiny values, so growing an MX tile eventually induces underflow."""
    g = torch.Generator().manual_seed(seed)
    w = torch.randn(shape, generator=g) * 0.05
    n = w.numel()
    flat = w.reshape(-1)
    idx = torch.randperm(n, generator=g)
    flat[idx[:n // 50]] = 1.0            # sparse large outliers
    flat[idx[n // 50: n // 25]] = 1e-4   # sparse tiny values
    return flat.reshape(shape)


def _row_ramp(shape, seed=0):
    """Magnitude decays geometrically DOWN the rows (1.0 -> 1e-4). A small
    row-window has a narrow dynamic range (no underflow); a tall row-tile spans
    the full range and underflows. Respects the doc's precondition that the
    (start,1) tile is clean, so the hard no-underflow search has room to grow."""
    R, C = shape
    g = torch.Generator().manual_seed(seed)
    mag = torch.logspace(0.0, -4.0, R).unsqueeze(1)          # (R,1)
    jitter = 1.0 + 0.05 * torch.randn(shape, generator=g)
    return mag * jitter


# --------------------------------------------------------------------------- #
# 1. required scale_target on the public entry points
# --------------------------------------------------------------------------- #
def test_enable_microscaling_requires_scale_target():
    sig = inspect.signature(ss.enable_microscaling)
    assert "scale_target" in sig.parameters
    assert sig.parameters["scale_target"].default is inspect._empty, \
        "scale_target must be required (no default)"


def test_greedy_sensitivity_has_scale_target():
    assert "scale_target" in inspect.signature(ss.greedy_sensitivity).parameters


# --------------------------------------------------------------------------- #
# 2. enable_microscaling sets module.mx_scale_target
# --------------------------------------------------------------------------- #
def test_enable_microscaling_sets_field():
    model = nn.Sequential(nn.Conv2d(8, 8, 3, padding=1), nn.ReLU(),
                          nn.Conv2d(8, 8, 3, padding=1))
    lof_model = lof.lofloatify(model)
    ss.enable_microscaling(lof_model, block_size=32, scale_target=1.0)
    tgts = [getattr(m, "mx_scale_target", None)
            for m in lof_model.modules()
            if getattr(m, "scaling", None) == "mx"]
    assert tgts and all(t == 1.0 for t in tgts), tgts


# --------------------------------------------------------------------------- #
# 3. count_mx_underflow premise: underflow is non-decreasing as the tile grows
#    (this is what mx_search relies on to find an onset).
# --------------------------------------------------------------------------- #
def test_underflow_monotone_in_tile_size():
    w = _heavy_tailed((256, 256))
    elem, scale = _elem(), _scale_fmt()
    counts = [lof.count_mx_underflow(w, elem, scale_format=scale,
                                     block_size=(r, 1), rounding_mode=RNE,
                                     scale_target=1.0)
              for r in (32, 64, 128, 256)]
    assert counts == sorted(counts), counts
    assert counts[-1] > 0, "expected some underflow at the largest tile"


# --------------------------------------------------------------------------- #
# 4. mx_search (underflow-only mode) returns a valid, zero-underflow tile
# --------------------------------------------------------------------------- #
def test_mx_search_underflow_only_is_zero_underflow():
    assert hasattr(ss, "mx_search"), "mx_search not implemented"
    w = _row_ramp((256, 256), seed=3)   # start tile clean, tall tiles underflow
    elem, scale = _elem(), _scale_fmt()
    # sanity: the doc's precondition holds — the (start,1) tile has no underflow
    assert lof.count_mx_underflow(w, elem, scale_format=scale, block_size=(32, 1),
                                  rounding_mode=RNE, scale_target=1.0) == 0
    tile = ss.mx_search(w, None, elem, scale, 1.0, accuracy_fn=None)
    assert isinstance(tile, tuple) and len(tile) == 2
    r, c = tile
    assert isinstance(r, int) and isinstance(c, int) and r >= 1 and c >= 1
    # non-unit axes must be multiples of 8; a unit axis (1-D case) is exempt
    if r > 1:
        assert r % 8 == 0, f"rows not a multiple of 8: {tile}"
    if c > 1:
        assert c % 8 == 0, f"cols not a multiple of 8: {tile}"
    # total-bitwidth budget: area * elem_bits <= 2048
    elem_bits = getattr(elem, "total_bits", None) or getattr(elem, "bitwidth", None) or 8
    assert r * c * elem_bits <= 2048, (tile, elem_bits)
    assert r * c >= 16, tile   # >= start (mx_search starts at (16,1))
    uf = lof.count_mx_underflow(w, elem, scale_format=scale, block_size=(r, c),
                                rounding_mode=RNE, scale_target=1.0)
    assert uf == 0, f"returned tile {tile} still underflows ({uf})"


# --------------------------------------------------------------------------- #
# 5. mx_search output is a valid arg to lof.set_mx_tile_fields
# --------------------------------------------------------------------------- #
def test_mx_search_tile_applies_via_set_mx_tile_fields():
    model = nn.Sequential(nn.Conv2d(16, 16, 1), nn.ReLU(), nn.Conv2d(16, 16, 1))
    lof_model = lof.lofloatify(model)
    ss.enable_microscaling(lof_model, block_size=32, scale_target=1.0)
    name = next(n for n, m in lof_model.named_modules()
                if getattr(m, "scaling", None) == "mx")
    lof.set_mx_tile_fields(lof_model, {name: (4, 8)})
    m = dict(lof_model.named_modules())[name]
    got = getattr(m, "weight_mx_block_size", None) or getattr(m, "mx_block_size", None)
    assert tuple(got) == (4, 8), got
