# Plan: Naive MX tile search integrated into `greedy_sensitivity`'s exp search

## Context

`new_MX_formulation.md` asks for a **naive `mx_search`** that, per layer, finds the
*largest* MX tile shape `(rows, cols)` that either (a) prevents **all** underflow, or
(b) still meets the accuracy budget — and to wire it into `greedy_sensitivity`'s
**exponent** search so that whenever lowering exponent bits introduces underflow, each
affected layer gets a repaired tile instead of silently losing small values.

Larger MX tiles are cheaper (fewer shared scale factors) but underflow more, because
each tile shares one E8M0 scale pinned to the tile's `amax`; the whole error budget
falls on the *small* elements. So the search grows the tile as large as possible
subject to the no-underflow (hard) / accuracy (soft) constraint. **Overflow is not a
concern** under per-tile amax scaling: the largest element is mapped to the top of the
element format's range by construction, so reducing exponent bits produces underflow,
not overflow. Trigger is **underflow-only** via the existing `count_mx_underflow`. Tiles
are **shared** across weight+activation per layer (one tile per layer;
`underflow = weight_underflow + activation_underflow` must be 0).

## Confirmed LoFloat API (installed at `/home/spuser/LoFloat-pvt/`, exposed as `lof.*`)

The public tree `/home/spuser/LoFloat/` is stale — everything below was verified against
the *installed* package by direct probing.

- `lof.count_mx_underflow(tensor, element_format, scale_format=None, block_size=32|(rows,cols), rounding_mode=RNE) -> int`
  (`LoFloat-pvt/LoFloat/utils.py:493`). Runs the **same `STEMXRound`** the layers use, so
  it is bit-faithful when fed the runtime-shaped tensor. `element_format` = the layer's
  live `module.weight_params` / `module.act_params` (`FloatFormatDescriptor`); these
  reflect the current exp+mantissa bits. `block_size` accepts a `(tile_rows, tile_cols)`
  2-tuple applied to the **last two dims**. Oversize tiles are tolerated (treated as
  whole-axis). Already imported at `sensitivity_search.py:6`.
- `lof.set_mx_tile_fields(model, tile_dict, scaling_dict=None)` (`utils.py:222`). `tile_dict`
  maps `name -> int | (a,b) tuple | {"act":..,"weight":..}`; sets the layer's block config
  and flips it to `scaling="mx"`. A single tuple fans out to both operands. Layers absent
  from the dict are left untouched (stay `per_tensor`). This is the correct applier —
  **`set_mx_tile_shapes` does NOT exist** in the installed package.
- Runtime MX shapes (so the proxy matches deployment exactly):
  - Linear (`layers.py:674-681`): activation `x` as-is; weight `module.weight` `(N,K)`.
  - Conv2d (`layers.py:1034-1052`): activation `F.unfold(x, kernel_size, dilation, padding,
    stride).transpose(1,2).contiguous()` → `(B,L,K)`; weight `module.weight.view(C_out,-1)`
    → `(C_out,K)`. K = (C_in//groups)·kH·kW.

**Pre-existing bug this fixes:** `greedy_sensitivity` defaults `microscaling=True`, and
`_push_exp` (`sensitivity_search.py:789-793`) calls the non-existent
`lof.set_mx_tile_shapes(...)` — so the current default path would `AttributeError`. This
work replaces that call and the stale `*_mx_shape`/`is_valid_mx`/`next_mx_step` scaffolding.

## Design — all in `sensitivity_search.py`

### 1. Activation cache helper (`capture_mx_activations`)
Hook every `LoF_Linear`/`LoF_Conv2d` (pattern in `sensitivities.find_range`,
`sensitivities.py:70-94`), run **one small calib batch** (reuse
`sensitivities.make_calib_data`; keep n small, e.g. 4–8, to bound memory — unfolded conv
activations `(B,L,K)` are large), store per layer the **runtime-shaped pre-round
activation** on CPU:
- Linear: raw input `x`.
- Conv2d: `F.unfold(x, ...).transpose(1,2).contiguous()` (matches `layers.py:1044-1049`).

Weights are read live from `module.weight` at search time (no cache): Linear → as-is;
Conv2d → `module.weight.view(C_out, -1)`.

### 2. Underflow + acceptance predicate
```
def _tile_underflow(module, name, tile, activ_cache) -> int:
    w = module.weight if Linear else module.weight.view(C_out, -1)
    uw = count_mx_underflow(w.float(), module.weight_params, block_size=tile)
    ua = count_mx_underflow(activ_cache[name].to(dev).float(), module.act_params, block_size=tile)
    return uw + ua
```
```
def _accept(module, name, tile, activ_cache, model, data, eval_fn, accuracy_target) -> bool:
    if _tile_underflow(...) == 0:            # cheap, no forward pass
        return True
    lof.set_mx_tile_fields(model, {name: tile})   # apply only this layer
    return abs(eval_fn(model, data)) <= accuracy_target   # expensive fallback
```
The cheap underflow path drives most growth; `eval_fn` (same one greedy uses) only runs
once the tile can no longer be kept underflow-free.

### 3. `mx_search(module, name, activ_cache, model, data, eval_fn, accuracy_target, start=32, cap=2048)`
Directed search per pseudocode, returning the max-area accepted tile:
- **`_grow(fixed_axis, grow_axis)`**: start tile `(32,1)` (row branch) / `(1,32)` (col
  branch), clamped to `cap`. Double the growth axis while `_accept` holds; the first
  rejected step stops it. If growth reached `cap` purely via the underflow==0 path, return
  `(cap,1)`/`(1,cap)` **without** the second-axis expansion (honors "if we reach 2048
  return"). Otherwise take the last accepted `(r,1)` and expand the **other** axis
  (`(r,2),(r,4),…`) while `_accept` holds.
- Run `_grow` **row-first** and **col-first**; return whichever tile has larger area
  (`rows*cols`). Ties → row branch.
- Edge case: if even the start tile `(32,1)` fails both underflow and accuracy, return the
  start tile as best-effort — the exp rung that caused it will be reverted by greedy's
  normal keep/revert (Section 4), so MX never widens a layer it can't rescue.
- Leaves `model` with `mx_tiles[name]` applied to `best` before returning.

### 4. Integration into the exp axis of `greedy_sensitivity`
- Keep the existing `mx_tiles = {}` dict (`sensitivity_search.py:734`); it is already what
  `set_mx_tile_fields` consumes at final-apply (`:946-947`). Populate it via `mx_search`.
- **Fix `_push_exp`** (`:784-793`): replace the `lof.set_mx_tile_shapes(...)` call with
  `if microscaling and mx_tiles: lof.set_mx_tile_fields(lof_model, mx_tiles)`.
- **Repair hook in `run_rung`** (`:854-883`): after `axis["push"]()` and before the batch
  `eval_fn`, when `axis["name"] == "exp" and microscaling`, for each layer in the batch (its
  exp bits just dropped) compute default-tile underflow; for layers with `underflow > 0`,
  call `mx_search`, record the result in `mx_tiles`, and `set_mx_tile_fields`. The
  subsequent batch `eval_fn` then gates keep/revert as today — and on revert, the exp bits
  and the layer's `mx_tiles` entry for that batch are rolled back together. Layers that
  don't underflow stay `per_tensor` (never added to `mx_tiles`).
- Capture the activation cache once, right after `find_range`/`find_exp_bits_and_bias`
  (`:655-660`), and thread it into `run_rung`.

### 5. Cleanup
- Remove the stale `bias_mx_shape`/`weights_mx_shape`/`activ_mx_shape` dicts (`:727-729`) —
  the new `mx_tiles` (name→`(rows,cols)`) is the single source of truth; absent layers stay
  per_tensor automatically.
- Remove the dead `is_valid_mx`/`next_mx_step`/`exp_mx_search` (`:960-1019`): they use an
  incompatible `(rows,cols,scale_factors)` + register-size model and a non-existent applier.

## Caveats to encode as comments
- For **Linear activations** the tile's row axis is the token/batch axis (runtime tiles the
  last two dims of `x`), so tile `rows` interact with inference batch size; the search
  measures on the calib batch. Weight row axis is the output-channel axis. Documented, not
  fixed — this is the library's MX convention.
- `count_mx_underflow` requires finite inputs (Inf makes the block scale Inf → NaN); calib
  activations are finite in practice.

## Verification
**A full `yolo_test.py` run is very expensive (1–2 h) — NOT the dev-loop check.** Gate on
the cheap tests, and only ever run e2e with the exp+MX rungs alone.

1. **Unit-level (fast):** tiny `nn.Sequential` of Linear+Conv2d lofloatified with small
   `p3109` params — assert `mx_search` returns a valid `(rows,cols)`, that the returned
   tile's `_tile_underflow == 0` whenever any zero-underflow tile ≥ start exists, and that
   `set_mx_tile_fields` flips `scaling=="mx"` with matching
   `act_mx_block_size`/`weight_mx_block_size`.
2. **Proxy fidelity (fast):** `count_mx_underflow` on the cached runtime-shaped tensor
   equals nnz-loss from an actual `STEMXRound.apply` at the same tile.
3. **Exp+MX-only e2e (only after 1–2 pass):** call `greedy_sensitivity` with
   **`search_order=("exp",)`** so only the exponent axis and its MX sub-step run (no
   `accum`, no `mant`). Use a small `n_samples` and `YOLO_test/voc2012_val_subset.yaml`.
   Confirm: (a) no `AttributeError`, (b) exp search logs MX repairs, (c) `mx_tiles`
   populated and re-applied, (d) accuracy drop ≤ target.
4. Optionally repeat step 3 with `microscaling=False` to confirm the MX path is additive.
