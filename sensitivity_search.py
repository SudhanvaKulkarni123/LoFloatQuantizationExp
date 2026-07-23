import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import LoFloat as lof
from LoFloat import LoF_Linear, LoF_Conv2d, L1BatchNorm, LinfBatchNorm, FISRBatchNorm, PWLBatchNorm, PWLSiLU, _fwht
import copy
import math
import sensitivities
import gptq
import warnings



def replace_batchnorm2d(model, p, L1_scales=None, Linf_scales=None,
                        lut_bits=4, lut_method="minimax"):
    """
    p == 2 or 2.0      -> no-op.
    p == 1.0           -> L1BatchNorm   (calibrated per-channel scale)
    p == math.inf      -> LinfBatchNorm (calibrated per-channel scale)
    p == 'fisr'        -> FISRBatchNorm (Quake III rsqrt + 1 NR step)
    p == 'pwl'         -> PWLBatchNorm  (LUT + linear interp for rsqrt)

    L1/Linf: running_{mad,maxdev} = scale * sqrt(running_var + eps), so
        pre-affine output matches BN2d's (x-mean)/std exactly.
    FISR/PWL: running_var copied directly; layer approximates 1/sqrt at runtime.
    """
    print(f"p = {p}")
    if p == 2 or p == 2.0:
        return model
    replaced = False
    # Resolve target class and how to populate it from the source BN.
    if p == 1.0:
        if not replaced :
            print("Replacing BatchNorm2d with L1BatchNorm...")
            replaced = True
        target_cls, scales_dict, stat_name = L1BatchNorm, L1_scales, "running_mad"
        mode = "calibrated"
    elif p == math.inf:
        if not replaced :
            print("Replacing BatchNorm2d with LinfBatchNorm...")
            replaced = True
        target_cls, scales_dict, stat_name = LinfBatchNorm, Linf_scales, "running_maxdev"
        mode = "calibrated"
    elif p == "fisr":
        if not replaced:
            print("Replacing with fisr....")
            replaced = True
        target_cls, mode = FISRBatchNorm, "rsqrt_approx"
    elif p == "pwl":
        if not replaced:
            print(f"Replacing with PWL approximation of rsqrt with lut_bits={lut_bits}, lut_method={lut_method}...")
            replaced = True
        target_cls, mode = PWLBatchNorm, "rsqrt_approx"
    else:
        raise ValueError(f"p must be 1.0, 2, math.inf, 'fisr', or 'pwl'; got {p!r}")

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.BatchNorm2d):
                continue
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            device = child.running_mean.device
            dtype = child.running_mean.dtype
            print("actually identiftied BN correctly")

            if mode == "calibrated":
                print("calibrated, using L-p norm")
                if full_name not in scales_dict:
                    print(f"  [SKIP] {full_name} — no calibration scale")
                    continue
                scale_c = scales_dict[full_name].to(device=device, dtype=dtype)
                new = target_cls(
                    num_features=child.num_features,
                    eps=child.eps, momentum=child.momentum,
                    affine=child.affine, scale=scale_c,
                ).to(device=device, dtype=dtype)
                running_std = (child.running_var.detach() + child.eps).sqrt()
                getattr(new, stat_name).copy_(scale_c * running_std)
                new.running_mean.copy_(child.running_mean.detach())

            else:  # mode == "rsqrt_approx"
                kwargs = dict(
                    num_features=child.num_features,
                    eps=child.eps, momentum=child.momentum,
                    affine=child.affine,
                )
                if target_cls is PWLBatchNorm:
                    kwargs["lut_bits"] = lut_bits
                    kwargs["lut_method"] = lut_method
                new = target_cls(**kwargs).to(device=device, dtype=dtype)
                new.running_mean.copy_(child.running_mean.detach())
                new.running_var.copy_(child.running_var.detach())
                if hasattr(new, "num_batches_tracked"):
                    new.num_batches_tracked.copy_(child.num_batches_tracked)

            if child.affine:
                new.weight.data.copy_(child.weight.detach())
                new.bias.data.copy_(child.bias.detach())
            setattr(parent, child_name, new)

    return model

def replace_silu(model, R=8.0, lut_bits=4, lut_method="minimax"):
    """Replace all nn.SiLU instances in `model` with PWLSiLU layers."""
    for parent in model.modules():
        for child_name, child in list(parent.named_children()):
            if isinstance(child, nn.SiLU):
                new = PWLSiLU(R=R, lut_bits=lut_bits, lut_method=lut_method)
                # Match dtype/device of an adjacent parameter, if any.
                for p in parent.parameters(recurse=False):
                    new = new.to(device=p.device, dtype=p.dtype)
                    break
                setattr(parent, child_name, new)
    return model



def _largest_pow2_divisor(n: int) -> int:
    """Largest power of 2 that divides n. 0 for n<=0."""
    if n <= 0:
        return 0
    return n & (-n)


@torch.no_grad()
def apply_hadamard_to_weights(model, skip_names=(), skip_first=True, skip_last=True,
                              min_block_size=2, max_block_size=None):
    """Pre-rotate the contraction axis of LoF_Linear / LoF_Conv2d weights by a
    block-diagonal orthonormal Hadamard, and flip on `hadamard_transform` so
    the layer rotates activations to match at runtime. Net GEMM unchanged.

    For each layer, the block size is the largest power of 2 that divides the
    contraction dim (in_features for Linear, in_channels//groups for Conv2d),
    optionally clamped by `max_block_size`. Layers whose largest pow2 divisor
    is below `min_block_size` (default 2) are skipped — no useful mixing.

    Args:
        model: nn.Module to rotate in place.
        skip_names: extra module names to leave untouched.
        skip_first / skip_last: skip the first/last LoF layer in module order.
        min_block_size: skip layers whose auto-picked block_size is < this.
        max_block_size: optional cap on auto-picked block_size (must be pow2).
    """
    if max_block_size is not None and (max_block_size <= 0 or
                                       (max_block_size & (max_block_size - 1)) != 0):
        raise ValueError(f"max_block_size must be a power of 2, got {max_block_size}")

    lof_layers = [(name, m) for name, m in model.named_modules()
                  if isinstance(m, (LoF_Linear, LoF_Conv2d))]

    skip = set(skip_names)
    if skip_first and lof_layers:
        skip.add(lof_layers[0][0])
    if skip_last and lof_layers:
        skip.add(lof_layers[-1][0])

    n_lin = n_conv = n_already = 0
    skipped_endpoints = []
    skipped_too_small = []
    block_size_log = []

    for name, module in lof_layers:
        if name in skip:
            skipped_endpoints.append(name)
            continue
        if module.hadamard_transform:
            warnings.warn(f"{name}: hadamard_transform already True, skipping.")
            n_already += 1
            continue

        K = (module.in_features if isinstance(module, LoF_Linear)
             else module.in_channels // module.groups)

        bsz = _largest_pow2_divisor(K)
        if max_block_size is not None:
            bsz = min(bsz, max_block_size)

        if bsz < min_block_size:
            skipped_too_small.append((name, K, bsz))
            continue

        if isinstance(module, LoF_Linear):
            module.weight.data.copy_(_fwht(module.weight.data, block_size=bsz))
            module.hadamard_transform = True
            module.hadamard_block_size = bsz
            n_lin += 1
        else:
            w = module.weight.data
            w = w.movedim(1, -1).contiguous()
            w = _fwht(w, block_size=bsz)
            w = w.movedim(-1, 1).contiguous()
            module.weight.data.copy_(w)
            module.hadamard_transform = True
            module.hadamard_block_size = bsz
            n_conv += 1

        block_size_log.append((name, K, bsz))

    print(f"[hadamard] rotated {n_lin} LoF_Linear, {n_conv} LoF_Conv2d "
          f"(endpoints skipped: {len(skipped_endpoints)}, "
          f"too-small skipped: {len(skipped_too_small)}, "
          f"already-rotated: {n_already}).")
    if block_size_log:
        from collections import Counter
        sizes = Counter(b for _, _, b in block_size_log)
        print(f"[hadamard] block size distribution: {dict(sorted(sizes.items()))}")
    if skipped_too_small:
        print(f"[hadamard] skipped (no useful pow2 divisor): {skipped_too_small}")
    return model

@torch.no_grad()
def enable_microscaling(model, scale_target, block_size=32, scale_format=None,
                        skip_names=()):
    """Switch eligible LoF_Linear / LoF_Conv2d layers to MX (microscaling)
    quantization: a per-block shared scale along the reduction (K) axis, stored
    in E8M0 by default (see LoF_Quantize / LoF_Linear in LoFloat/layers.py).

    `scale_target` (required, absolute float) is the MX scale target: each
    block's shared scale maps that block's amax to `scale_target`. It is stored
    on every switched module as `module.mx_scale_target` so the runtime round
    and the underflow counting (`lof.count_mx_underflow(..., scale_target=...)`)
    agree with the tile search here.

    The searched act/weight/bias formats become the *element* format of each
    block; only the shared scale is added on top, so this composes with the
    mantissa/exponent/accum search unchanged.

    K need NOT be divisible by block_size — virtual_mx_round handles a partial
    trailing block. The only real runtime constraints, both Conv2d-only, are:
      - groups == 1  (the MX conv path reshapes the weight to (C_out, K) and
                       runs a single GEMM; grouped/depthwise convs can't work).
      - backend != "cutlass"  (no MX kernel on that path).

    Args:
        model: model with LoF layers, mutated in place.
        scale_target: required absolute float; the MX scale maps each block's
            amax to this value. Stored as `module.mx_scale_target`.
        block_size: MX block size along K (default 32).
        scale_format: shared-scale format; defaults to E8M0.
        skip_names: extra module names to leave in per_tensor scaling.
    """
    if scale_format is None:
        scale_format = lof.create_e8m0_params()
    skip = set(skip_names)

    n_lin = n_conv = 0
    skipped = []
    for name, module in model.named_modules():
        if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
            continue
        if name in skip:
            continue

        if isinstance(module, LoF_Conv2d):
            if getattr(module, "backend", "im2col") == "cutlass":
                skipped.append((name, "cutlass backend has no MX support"))
                continue
            if module.groups != 1:
                skipped.append((name, f"groups={module.groups} (depthwise) unsupported"))
                continue

        module.scaling = "mx"
        module.mx_block_size = block_size
        module.scale_format = scale_format
        module.mx_scale_target = scale_target
        if isinstance(module, LoF_Linear):
            n_lin += 1
        else:
            n_conv += 1

    print(f"[microscaling] enabled MX (block_size={block_size}, E8M0 scale) on "
          f"{n_lin} LoF_Linear, {n_conv} LoF_Conv2d "
          f"({len(skipped)} ineligible left per_tensor).")
    if skipped:
        print(f"[microscaling] skipped: {skipped}")
    return model


# =====================================================================
# MX tile-shape search (per new_MX_formulation.md)
# =====================================================================
@torch.no_grad()
def _capture_layer_inputs(model, dataset, n_samples, device, collate_fn=None,
                          chunk_size=32, max_rows=8192):
    """Capture one representative 2-D input tensor per LoF layer over a small
    calibration batch, so `mx_search` can count activation underflow.

    Mirrors the forward-hook / calibration pattern in `sensitivities.find_range`.
    Each captured tensor is reshaped to `(rows, K)` where K is the reduction dim
    the MX blocks tile along: for `LoF_Linear` this is `(-1, in_features)`; for
    `LoF_Conv2d` the input is im2col-unfolded to `(N*L, K)` — the exact operand
    layout the MX conv path GEMMs on. Rows are capped at `max_rows` for memory.
    """
    model.eval()
    n = min(n_samples, chunk_size)
    calib = sensitivities.make_calib_data(
        dataset=dataset, n_samples=n, collate_fn=collate_fn
    )
    cache = {}

    def make_hook(name):
        def hook(mod, inp, out):
            if name in cache:
                return
            x = inp[0].detach()
            if isinstance(mod, LoF_Conv2d):
                unf = F.unfold(x, kernel_size=mod.kernel_size, stride=mod.stride,
                               padding=mod.padding, dilation=mod.dilation)  # (N,K,L)
                x2d = unf.transpose(1, 2).reshape(-1, unf.shape[1]).contiguous()
            else:
                x2d = x.reshape(-1, mod.in_features).contiguous()
            if x2d.shape[0] > max_rows:
                x2d = x2d[:max_rows]
            cache[name] = x2d.float()
        return hook

    hooks = [
        module.register_forward_hook(make_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, (LoF_Linear, LoF_Conv2d))
    ]
    try:
        batch = calib.to(device, non_blocking=True)
        model(batch)
        del batch
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()
    finally:
        for h in hooks:
            h.remove()
    return cache


@torch.no_grad()
def mx_search(weight, activation, element_format, scale_format, scale_target,
              accuracy_fn=None, max_tile=2048, start=16, accuracy_target=0.0,
              label="layer"):
    """Find the largest MX tile `(rows, cols)` for one layer.

    `rows` indexes the out_features axis and `cols` the K/reduction axis. The
    search is run over BOTH operands: `weight` and `activation` are 2-D tensors
    (`activation` may be None -> count underflow on the weight only), and the
    total underflow at a tile is `weight_underflow + activation_underflow`.

    Two variants are run and the larger-area result (rows*cols) is returned:
      * rows-first: grow rows from `(start, 1)`, then (accuracy mode) columns.
      * cols-first: the mirror image — grow cols from `(1, start)`, then rows.

    Bounds / stepping:
      * The cap is a TOTAL BIT BUDGET, `MAX_TILE_BITS = 2048`. With
        `elem_bits = element_format` bit width (recomputed per call), every tile
        must satisfy `rows * cols * elem_bits <= 2048`, i.e. element-area
        <= `2048 // elem_bits` (256 for 8-bit, 227 for 9-bit).
      * Any NON-unit axis length is a multiple of 8. 1-D tiles (other axis == 1)
        constrain only the non-unit axis; 2-D tiles (both > 1) constrain BOTH.
        Growing 1-D -> 2-D the secondary axis jumps 1 -> 8 -> 16 -> ... `start`
        (16) is already a multiple of 8.
      * Every monotone boundary is found by BINARY SEARCH over the grid of
        multiples of 8, logging each tile it evaluates.

    Modes:
      * `accuracy_fn is None` -> deterministic hard no-underflow search (counts
        only, no forward pass): binary-search the largest primary L (multiple of
        8, in `[start, floor8(2048//elem_bits)]`) with `total_underflow((L,1))
        == 0`, and return `(L, 1)`. The returned tile has ZERO underflow. If the
        start tile already underflows the variant is dropped; if both variants
        drop, `(start, 1)` is returned as a non-crashing fallback.
      * `accuracy_fn` provided -> `accuracy_fn(tile)` applies `tile` and returns
        a bool (acceptable?) or a relative accuracy drop (compared against
        `accuracy_target`). Binary-search the largest acceptable primary L up to
        the budget max, then from that P binary-search the largest acceptable
        secondary M. Both boundary searches add a neighbor-check (L+/-8) to catch
        a single-step non-monotonic wobble in the subset metric.

    Budget edge: if `floor8(2048 // elem_bits) < start`, returns `(start, 1)`.

    Returns a plain `(int, int)` tuple, both >= 1.
    """
    def total_underflow(r, c):
        u = int(lof.count_mx_underflow(
            weight, element_format, scale_format=scale_format,
            block_size=(r, c), scale_target=scale_target))
        if activation is not None:
            u += int(lof.count_mx_underflow(
                activation, element_format, scale_format=scale_format,
                block_size=(r, c), scale_target=scale_target))
        return u

    def evaluate(tile):
        """Return (acceptable_bool, acc_drop_or_None) for a forward pass."""
        res = accuracy_fn(tile)
        if isinstance(res, bool):
            return res, None
        d = abs(float(res))
        return d <= accuracy_target, d

    def _log_uf(tag, tile, uf):
        verdict = "ACCEPT" if uf == 0 else "REJECT:underflow"
        print(f"[mx_search] {label} {tag}: tile {tile} underflow={uf} -> {verdict}")

    def _log_acc(tag, tile, ok, drop):
        verdict = "ACCEPT" if ok else "REJECT:accuracy"
        if drop is None:
            print(f"[mx_search] {label} {tag}: tile {tile} "
                  f"acc={'ok' if ok else 'bad'} -> {verdict}")
        else:
            print(f"[mx_search] {label} {tag}: tile {tile} "
                  f"acc_drop={drop:.4f} tgt={accuracy_target:.4f} -> {verdict}")

    # ── Bit-budget bounds (recomputed per call from the element format) ──
    MAX_TILE_BITS = 2048
    elem_bits = getattr(element_format, "bitwidth", None)
    if not isinstance(elem_bits, int) or elem_bits <= 0:
        elem_bits = getattr(element_format, "total_bits", None)
    if not isinstance(elem_bits, int) or elem_bits <= 0:
        elem_bits = 8  # sane fallback for mocks / partial formats

    def floor8(x):
        return (int(x) // 8) * 8

    area_max = MAX_TILE_BITS // elem_bits          # max element-area
    Lmax = floor8(area_max)                        # largest multiple-of-8 axis

    def largest_true_mult8(lo, hi, pred):
        """Largest v (multiple of 8) in [lo, hi] with pred(v) True, assuming
        pred is monotone (True then False). Returns None if pred(lo) is False.
        Binary search in units of 8; every evaluated v is logged inside pred."""
        lo, hi = floor8(lo), floor8(hi)
        if hi < lo or not pred(lo):
            return None
        a, b = lo // 8, hi // 8
        while a < b:
            mid = (a + b + 1) // 2
            if pred(mid * 8):
                a = mid
            else:
                b = mid - 1
        return a * 8

    def neighbor_expand(v, hi, pred):
        """Guard against a single-step wobble: after converging on v, also try
        v+8 (and v-8 for logging); keep the largest that passes."""
        best = v
        for cand in (v + 8, v - 8):
            if cand < 8 or cand > hi:
                continue
            if pred(cand) and cand > best:
                best = cand
        return best

    def grow(primary_is_rows):
        tag = "rows" if primary_is_rows else "cols"

        # mk(n, m): tile with n on the primary axis and m on the other axis.
        def mk(n, m=1):
            return (int(n), int(m)) if primary_is_rows else (int(m), int(n))

        # ── Deterministic (hard) mode: primary underflow-onset boundary ──
        if accuracy_fn is None:
            def uf_pred(L):
                tile = mk(L, 1)
                uf = total_underflow(*tile)
                _log_uf(tag, tile, uf)
                return uf == 0

            L = largest_true_mult8(start, Lmax, uf_pred)
            if L is None:
                return None  # start tile underflows -> drop this orientation
            return mk(L, 1)

        # ── Accuracy mode: primary accuracy boundary (1-D, secondary == 1) ──
        def acc_pred_primary(L):
            tile = mk(L, 1)
            ok, drop = evaluate(tile)
            _log_acc(tag, tile, ok, drop)
            return ok

        P = largest_true_mult8(start, Lmax, acc_pred_primary)
        if P is None:
            # Even the smallest tile is unacceptable — keep it as least-bad.
            return mk(start, 1)
        P = neighbor_expand(P, Lmax, acc_pred_primary)

        # ── Perpendicular axis: largest acceptable secondary M (multiple of 8,
        #    >= 8) subject to the bit budget P*M*elem_bits <= 2048. ──
        Mmax = floor8(area_max // P)
        best = mk(P, 1)
        if Mmax >= 8:
            def acc_pred_sec(M):
                tile = mk(P, M)
                ok, drop = evaluate(tile)
                _log_acc(tag, tile, ok, drop)
                return ok

            M = largest_true_mult8(8, Mmax, acc_pred_sec)
            if M is not None:
                M = neighbor_expand(M, Mmax, acc_pred_sec)
                best = mk(P, M)
        return best

    # ── Budget edge: no valid multiple-of-8 axis fits the bit budget ──
    if Lmax < start:
        chosen = (int(start), 1)
        print(f"[mx_search] {label}: budget too small "
              f"(elem_bits={elem_bits}, area_max={area_max}, floor8={Lmax} "
              f"< start={start}) -> CHOSE tile {chosen}")
        return chosen

    rf = grow(True)
    cf = grow(False)
    cands = [t for t in (rf, cf) if t is not None]
    if not cands:
        # Both start tiles underflow in deterministic (hard-only) mode. Return
        # the minimal-area start tile so the result is still a valid tile.
        chosen = (int(start), 1)
    else:
        chosen = max(cands, key=lambda t: t[0] * t[1])
    print(f"[mx_search] {label}: CHOSE tile {chosen} area={chosen[0] * chosen[1]} "
          f"elem_bits={elem_bits} bits={chosen[0] * chosen[1] * elem_bits} "
          f"(rows-first vs cols-first winner)")
    return chosen


# =====================================================================
# Uniform per-precision / per-operand MX tile selection
# =====================================================================
def mx_format_key(fmt):
    """Canonical HASHABLE key for a FloatFormatDescriptor -> its precision
    class. The descriptor object itself is not hashable, so we key on
    `(total_bits, mantissa_bits, bias, signedness)`."""
    return (
        getattr(fmt, "total_bits", None),
        getattr(fmt, "mantissa_bits", None),
        getattr(fmt, "bias", None),
        str(getattr(fmt, "signedness", None)),
    )


def _mx_format_bits(fmt):
    b = getattr(fmt, "bitwidth", None)
    if not isinstance(b, int) or b <= 0:
        b = getattr(fmt, "total_bits", None)
    if not isinstance(b, int) or b <= 0:
        b = 8
    return b


@torch.no_grad()
def mx_rel_frob_max(matrices, element_format, scale_format, scale_target, tile):
    """Max over `matrices` of the relative Frobenius error
    `||X - Q_tile(X)||_F / ||X||_F`, where `Q_tile` is MX fake-quant at `tile`.
    Tiles may exceed a matrix's dims — `virtual_mx_round` edge-pads, we never
    clamp. Zero-norm matrices are skipped."""
    RNE = lof.RoundingMode.RoundToNearestEven
    worst = 0.0
    for X in matrices:
        if X is None or X.numel() == 0:
            continue
        Xf = X.float()
        nrm = Xf.norm().item()
        if nrm == 0.0:
            continue
        Q = lof.virtual_mx_round(Xf, (int(tile[0]), int(tile[1])),
                                 element_format, scale_format, scale_target, RNE)
        err = (Xf - Q).norm().item() / nrm
        if err > worst:
            worst = err
    return worst


@torch.no_grad()
def uniform_mx_tile(matrices, element_format, scale_format, scale_target, frob_tau,
                    start=16, max_bits=2048):
    """Largest-area MX tile `(rows, cols)` shared uniformly across every matrix
    in `matrices` (one precision/operand class).

    Constraints: dims are multiples of 8 (a unit axis is exempt); the tile fits
    the bit budget `rows * cols * element_format.total_bits <= max_bits`; a tile
    MAY exceed a matrix's dims (edge-padded by the kernel — never clamped, so the
    smallest matrix can't bottleneck the class).

    Accept criterion: `max over matrices of ||X - Q_tile(X)||_F/||X||_F
    <= frob_tau`. Frobenius error is monotone increasing in tile area, so each
    axis boundary is found by binary search over the multiples-of-8 grid, using
    the same rows-first / cols-first + perpendicular structure as `mx_search`.

    Fully deterministic (no forward pass). If even `(start, 1)` violates tau for
    some matrix (or there are no usable matrices / the budget is too small),
    returns `(start, 1)` as a non-crashing floor.
    """
    mats = [X for X in matrices if X is not None and X.numel() > 0]
    if not mats:
        return (int(start), 1)

    def accept(tile):
        return mx_rel_frob_max(mats, element_format, scale_format,
                               scale_target, tile) <= frob_tau

    return uniform_tile_search(accept, element_format, start=start,
                               max_bits=max_bits)


def uniform_tile_search(accept, element_format, start=16, max_bits=2048):
    """Shared tile-shape search core: return the LARGEST-AREA multiple-of-8 tile
    `(rows, cols)` (a unit axis is exempt from the ×8 rule) within the bit budget
    `rows * cols * element_format.total_bits <= max_bits` for which
    `accept(tile) == True`. Returns `(start, 1)` if even that fails.

    `accept(tile) -> bool` is an arbitrary predicate ASSUMED MONOTONE: if
    `accept(t)` is False then every larger-area tile is also False. Tiles may
    exceed a matrix's dims (the kernel edge-pads; we never clamp). Each axis
    boundary is found by binary search over the ×8 grid, using the same
    rows-first / cols-first + perpendicular structure as `mx_search`.
    """
    elem_bits = _mx_format_bits(element_format)

    def floor8(x):
        return (int(x) // 8) * 8

    area_max = max_bits // elem_bits
    Lmax = floor8(area_max)
    floor_tile = (int(start), 1)

    if Lmax < start:
        return floor_tile

    def largest_true_mult8(lo, hi, pred):
        lo, hi = floor8(lo), floor8(hi)
        if hi < lo or not pred(lo):
            return None
        a, b = lo // 8, hi // 8
        while a < b:
            mid = (a + b + 1) // 2
            if pred(mid * 8):
                a = mid
            else:
                b = mid - 1
        return a * 8

    def variant(primary_is_rows):
        def mk(n, m=1):
            return (int(n), int(m)) if primary_is_rows else (int(m), int(n))

        P = largest_true_mult8(start, Lmax, lambda L: accept(mk(L, 1)))
        if P is None:
            return None  # even the start tile fails in this orientation
        best = mk(P, 1)
        Mmax = floor8(area_max // P)
        if Mmax >= 8:
            M = largest_true_mult8(8, Mmax, lambda m: accept(mk(P, m)))
            if M is not None:
                best = mk(P, M)
        return best

    rf = variant(True)
    cf = variant(False)
    cands = [t for t in (rf, cf) if t is not None]
    if not cands:
        return floor_tile
    return max(cands, key=lambda t: t[0] * t[1])


def bisection_sensitivity(model, sensitivity_measure, data, loss_fn, eval_fn,
                          accuracy_target, bs=[8, 6, 4], es=[8, 6, 4], n_samples=256, device='cuda', collate_fn=None, baseline=0.0):

    lof_model = lof.lofloatify(model)
    lof_model.to(device)

    with torch.no_grad():
        weights_minmax, activ_minmax, bias_minmax = sensitivities.find_range(lof_model, data, n_samples, device, collate_fn=collate_fn)
        weights_exp, weights_bias, activ_exp, activ_bias, bias_exp, bias_bias = sensitivities.find_exp_bits_and_bias(weights_minmax, activ_minmax, bias_minmax)


    # Sensitivity analysis
    if sensitivity_measure == "hessian":
        weight_sens, activ_sens, bias_sens = sensitivities.hess_sensitivity(lof_model, data, n_samples, device, collate_fn=collate_fn)
    else:
        weight_sens, activ_sens, bias_sens = sensitivities.noise_sensitivity_full(lof_model, data, loss_fn, n_samples, device, collate_fn=collate_fn)

    # Sort layers by weight sensitivity ascending (least sensitive = quantize first)
    if sensitivity_measure == "hessian":
        sort_key = lambda item: next(iter(item[1].values()))
    else:
        sort_key = lambda item: next(iter(item.values()))

    ll = [k for k, v in sorted(weight_sens.items(), key=sort_key)]

    # Identify first and last LoF layers to skip
    all_lof_layers = [n for n, m in lof_model.named_modules()
                      if isinstance(m, (LoF_Linear, LoF_Conv2d))]
    first_layer = all_lof_layers[0]
    last_layer = all_lof_layers[-1]
    skip_layers = {first_layer, last_layer}

    # Initialize working config with max(bs) mantissa bits
    max_b = max(bs)
    w_weights = {layer: max_b for layer in list(weight_sens.keys())}
    w_activ   = {layer: max_b for layer in list(weight_sens.keys())}
    w_bias    = {layer: max_b for layer in list(weight_sens.keys())}

    # Populate missing LoFloat layers
    for name, module in lof_model.named_modules():
        if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
            continue
        if name not in w_weights:
            w_weights[name] = max_b
        if name not in w_activ:
            w_activ[name] = max_b
        if name not in w_bias:
            w_bias[name] = max_b
        if name not in ll and name not in skip_layers:
            ll.append(name)

    # Sort bit widths descending
    bs = sorted(bs, reverse=True)

    # ── Mantissa bisection search ──
    for b in bs:
        # Filter out skip layers from candidate list
        candidates = [layer for layer in ll if layer not in skip_layers]
        thr  = len(candidates) // 2
        upl  = len(candidates)
        lowl = 0

        prev_thr = None
        while thr != prev_thr:
            prev_thr = thr

            # Local working config: copy current, then assign b to first thr candidates
            lw_weights = dict(w_weights)
            lw_activ   = dict(w_activ)
            lw_bias    = dict(w_bias)

            for layer in candidates[:thr]:
                lw_weights[layer] = b
                lw_activ[layer]   = b
                lw_bias[layer]    = b

            # Apply local config and evaluate
            lof.set_mantissa_fields(model=lof_model,
                                    activation_mantissa_bits=lw_activ,
                                    weight_mantissa_bits=lw_weights,
                                    bias_mantissa_bits=lw_bias)
            a = abs(eval_fn(lof_model, data))

            if a <= accuracy_target:
                lowl = thr
                thr  = thr + (upl - thr) // 2   # push threshold up
            else:
                upl  = thr
                thr  = thr - (thr - lowl) // 2  # pull threshold down

        # Commit the converged threshold to the working config
        for layer in candidates[:thr]:
            w_weights[layer] = b
            w_activ[layer]   = b
            w_bias[layer]    = b

        # Shrink ll: only layers already quantized are candidates for next round
        ll = candidates[:thr]

    # ── Exponent bisection search ──
    # Re-sort full layer list for exponent search
    ll = [k for k, v in sorted(weight_sens.items(), key=sort_key)]

    for name in bias_exp:
        bias_exp[name] = max(bias_exp[name], 1)
        weights_exp[name] = max(weights_exp[name], 1)
        activ_exp[name] = max(activ_exp[name], 1)

    max_e = max(es)
    for name, module in lof_model.named_modules():
        if not isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            continue
        if name not in weights_exp:
            weights_exp[name] = max_e
        if name not in activ_exp:
            activ_exp[name] = max_e
        if name not in bias_exp:
            bias_exp[name] = max_e
        if name not in ll and name not in skip_layers:
            ll.append(name)

    es = sorted(es, reverse=True)

    for b in es:
        candidates = [layer for layer in ll if layer not in skip_layers]
        thr  = len(candidates) // 2
        upl  = len(candidates)
        lowl = 0

        prev_thr = None
        while thr != prev_thr:
            prev_thr = thr

            # Local working config: copy current, then assign b to first thr candidates
            lw_wexp = dict(weights_exp)
            lw_aexp = dict(activ_exp)
            lw_bexp = dict(bias_exp)

            for layer in candidates[:thr]:
                lw_wexp[layer] = min(b, lw_wexp[layer])
                lw_aexp[layer] = min(b, lw_aexp[layer])
                lw_bexp[layer] = min(b, lw_bexp[layer])

            # Apply local config and evaluate
            lof.set_exponent_fields(model=lof_model,
                                    activation_exp_bits=lw_aexp,
                                    weight_exp_bits=lw_wexp,
                                    bias_exp_bits=lw_bexp)
            a = abs(eval_fn(lof_model, data))

            if a <= accuracy_target:
                lowl = thr
                thr  = thr + (upl - thr) // 2
            else:
                upl  = thr
                thr  = thr - (thr - lowl) // 2

        # Commit converged exponent config
        for layer in candidates[:thr]:
            weights_exp[layer] = min(b, weights_exp[layer])
            activ_exp[layer]   = min(b, activ_exp[layer])
            bias_exp[layer]    = min(b, bias_exp[layer])

        # Shrink candidate list
        ll = candidates[:thr]

    # ── Apply final config with GPTQ ──
    lof_model = lof.lofloatify(model)

    print("running gptq with exp_bits = %d and mantissa_bits = %d" % (list(weights_exp.values())[9], list(w_weights.values())[9]))
    print(f"First/last layers kept at {max_b} mantissa bits")
    # lof_model = sensitivities.quantize_weights_with_gptq(
    #     model=lof_model, dataset=data, mantissa_bits=w_weights,
    #     exponent_bits=weights_exp, n_samples=128, device=device, collate_fn=collate_fn
    # )
    lof.set_mantissa_fields(
        model=lof_model,
        activation_mantissa_bits=w_activ,
        weight_mantissa_bits=w_weights,
        bias_mantissa_bits=w_bias,
    )
    lof.set_exponent_fields(
        model=lof_model,
        activation_exp_bits=activ_exp,
        weight_exp_bits=weights_exp,
        bias_exp_bits=bias_exp,
    )
    lof_model.to(device)

    return lof_model

import torch
import torch.nn as nn
import math
import copy




# =====================================================================
# Strassen / fast-multiply viability check
# =====================================================================
def strassen_viability_check(
    model,
    data,
    eval_fn,
    accuracy_target,
    device="cuda",
    target_layer_types=None,
    max_active_hooks=12,
):
    """
    For each interior LoF layer, inject multiplication-rounding noise into
    the GEMM output and assign a grade:

      C  –  Strassen-viable (loosest bound, K² with global max-norms)
             noise[i,j] = U(-1,1) * ||A||_max * ||B||_max * K² * 0.5 * 2^{-p}

      B  –  Standard dot-product bound (tighter, per-row/col inf-norms)
             noise[i,j] = U(-1,1) * K * ||row_i(A)||_∞ * ||col_j(B)||_∞ * 0.5 * 2^{-p}

      A  –  Needs exact accumulation (neither bound was tolerated)

    For each layer we try C first; if that fails we try B; if that also
    fails we assign A.  Perturbations are cumulative.

    max_active_hooks: caps concurrent forward hooks for VRAM safety.
    """

    if target_layer_types is None:
        target_layer_types = (LoF_Linear, LoF_Conv2d)

    all_targets = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, target_layer_types)
    ]
    if len(all_targets) <= 2:
        print("[strassen] ≤2 target layers found, nothing to test.")
        return {name: "A" for name, _ in all_targets}

    interior = all_targets[1:-1]
    skipped = {all_targets[0][0], all_targets[-1][0]}
    print(f"[strassen] skipping first/last: {sorted(skipped)}")
    print(f"[strassen] testing {len(interior)} interior layers")
    print(f"[strassen] max concurrent hooks = {max_active_hooks}")

    grades = {name: "A" for name, _ in all_targets}
    active_hooks = {}

    def _make_hook_grade_c(module):
        """Grade C: Strassen bound – global max-norms, K²."""
        def hook_fn(mod, inputs, output):
            x = inputs[0]
            p = getattr(mod, "accum_mant_bits", 10)
            eps = 0.5 * (2.0 ** (-p))
            if isinstance(mod, LoF_Conv2d):
                K = mod.weight.shape[1] * mod.weight.shape[2] * mod.weight.shape[3]
            else:
                K = mod.weight.shape[1]
            max_A = x.abs().max().item()
            max_B = mod.weight.abs().max().item()
            scale = max_A * max_B * (K ** 2) * eps
            noise = (2.0 * torch.rand_like(output) - 1.0) * scale
            return output + noise
        return hook_fn

    def _make_hook_grade_b(module):
        """Grade B: dot-product bound – per-row/col inf-norms, K."""
        def hook_fn(mod, inputs, output):
            x = inputs[0]
            p = getattr(mod, "accum_mant_bits", 10)
            eps = 0.5 * (2.0 ** (-p))

            if isinstance(mod, LoF_Conv2d):
                batch = x.shape[0]
                groups = mod.groups
                C_out = mod.weight.shape[0]
                C_out_g = C_out // groups
                K_g = mod.weight.shape[1] * mod.weight.shape[2] * mod.weight.shape[3]
                H_out, W_out = output.shape[2], output.shape[3]

                unfolded = F.unfold(
                    x,
                    kernel_size=mod.kernel_size,
                    stride=mod.stride,
                    padding=mod.padding,
                    dilation=mod.dilation,
                )
                L = unfolded.shape[2]
                unfolded = unfolded.view(batch, groups, K_g, L)
                row_inf = unfolded.abs().max(dim=2).values
                del unfolded

                W_g = mod.weight.view(groups, C_out_g, K_g)
                col_inf = W_g.abs().max(dim=2).values

                row_inf = row_inf.view(batch, groups, 1, H_out, W_out)
                col_inf = col_inf.view(1, groups, C_out_g, 1, 1)

                scale = K_g * eps
                out_5d = output.view(batch, groups, C_out_g, H_out, W_out)
                noise = (2.0 * torch.rand_like(out_5d) - 1.0) * scale * row_inf * col_inf
                return output + noise.view_as(output)

            else:
                K = mod.weight.shape[1]
                N = mod.weight.shape[0]
                x_2d = x.reshape(-1, K)
                row_inf = x_2d.abs().max(dim=1).values
                col_inf = mod.weight.abs().max(dim=1).values
                row_inf = row_inf.view(*output.shape[:-1], 1)
                ones = [1] * (len(output.shape) - 1)
                col_inf = col_inf.view(*ones, N)
                scale = K * eps
                noise = (2.0 * torch.rand_like(output) - 1.0) * scale * row_inf * col_inf
                return output + noise

        return hook_fn

    for name, module in interior:
        if len(active_hooks) >= max_active_hooks:
            print(f"  {name:50s}  A   (hook cap reached, skipped)")
            continue

        # ── try grade C ──
        hook_c = module.register_forward_hook(_make_hook_grade_c(module))
        with torch.no_grad():
            torch.cuda.empty_cache()
            acc_c = abs(eval_fn(model, data))

        if acc_c <= accuracy_target:
            grades[name] = "C"
            active_hooks[name] = hook_c
            print(f"  {name:50s}  C   (acc={acc_c:.4f})")
            continue

        hook_c.remove()

        # ── try grade B ──
        hook_b = module.register_forward_hook(_make_hook_grade_b(module))
        with torch.no_grad():
            torch.cuda.empty_cache()
            acc_b = abs(eval_fn(model, data))

        if acc_b <= accuracy_target:
            grades[name] = "B"
            active_hooks[name] = hook_b
            print(f"  {name:50s}  B   (acc={acc_b:.4f})")
        else:
            hook_b.remove()
            grades[name] = "A"
            print(f"  {name:50s}  A   (C acc={acc_c:.4f}, B acc={acc_b:.4f}, reverted)")

    for h in active_hooks.values():
        h.remove()
    torch.cuda.empty_cache()

    n_c = sum(1 for g in grades.values() if g == "C")
    n_b = sum(1 for g in grades.values() if g == "B")
    n_a = sum(1 for g in grades.values() if g == "A")
    print(f"\n[strassen] results across {len(interior)} interior layers:")
    print(f"  C (Strassen-viable):        {n_c}")
    print(f"  B (dot-product bound OK):   {n_b}")
    print(f"  A (exact accumulation):     {n_a}")

    return grades


# =====================================================================
# greedy_sensitivity
#
# Search order:
#   1. Exponent
#   2. Accumulation (first pass)
#   3. Slack – bump each layer's accum up by 1 or 2 steps (coin flip)
#   4. Mantissa
#   5. Accumulation (second pass, starting from slackened base)
#   6. BN + SiLU replacement
#   7. Strassen viability check
# =====================================================================


def greedy_sensitivity(model, sensitivity_measure, data, loss_fn, eval_fn,
                       accuracy_target, bs=[8, 6, 4], es=[8, 6, 4], accum_bw=[10, 8, 6, 4],
                       n_samples=128, device='cuda', collate_fn=None, baseline=0.0, batch_size=5,
                       hadamard=True, strassen_max_hooks=12,
                       microscaling=True, mx_block_size=32, scale_target=1.0,
                       use_mx_search=True, tile_mode='per_layer', frob_tau=0.1,
                       tile_criterion='frob',
                       search_axes=('accum', 'mant', 'exp', 'bn_silu')):
    """
    Search order: exp → accum → slack → mant → accum(2nd).

    `search_axes` selects which search phases actually run: any subset of
    {'accum', 'mant', 'exp', 'bn_silu'}. Skipped axes keep their initial
    (max-precision) settings. The MX tile search is part of the 'exp' axis.

    The slack step gives the accumulator breathing room before the mantissa
    search reduces operand precision.  Each layer's accumulation width is
    bumped up by one or two steps in `accum_bw` (chosen by coin flip).
    The second accumulation pass then tightens back down from that
    slackened baseline.

    First and last nn.Linear/nn.Conv2d (in the original model) are kept in
    full precision.
    """
    # ── Identify first/last layers to skip ──
    orig_target_layers = [
        name for name, m in model.named_modules()
        if isinstance(m, (nn.Linear, nn.Conv2d))
    ]
    skip_layer_names = set()
    if orig_target_layers:
        skip_layer_names.add(orig_target_layers[0])
        skip_layer_names.add(orig_target_layers[-1])
    print(f"[greedy] keeping in full precision (not lofloatified): "
          f"{sorted(skip_layer_names)}")

    lof_model = lof.lofloatify(model, skip_layer_names=skip_layer_names)
    lof_model.to(device)

    # ── Apply Hadamard rotation to weights BEFORE calibration / sensitivity.
    # Every remaining LoF layer is rotated (first/last are already excluded
    # by lofloatify above, so skip_first/skip_last are False here). All ranges
    # and sensitivities below are then measured in the rotated basis, and the
    # eval_fn calls during the search run with `hadamard_transform=True` on
    # each layer, so activations get rotated to match the rotated weights.
    if hadamard:
        apply_hadamard_to_weights(lof_model, skip_first=False, skip_last=False)

    # ── Enable MX (E8M0) microscaling BEFORE calibration / sensitivity, so all
    # ranges and sensitivities below are measured in MX mode and the search
    # finds precisions valid under microscaling. The searched act/weight/bias
    # formats become each block's element format; the E8M0 shared scale rides
    # on top. First/last are already non-LoF (excluded by lofloatify).
    if microscaling:
        enable_microscaling(lof_model, scale_target=scale_target,
                            block_size=mx_block_size)

    with torch.no_grad():
        weights_minmax, activ_minmax, bias_minmax = sensitivities.find_range(
            lof_model, data, n_samples, device, collate_fn=collate_fn
        )
        weights_exp, weights_bias, activ_exp, activ_bias, bias_exp, bias_bias = \
            sensitivities.find_exp_bits_and_bias(weights_minmax, activ_minmax, bias_minmax)

    weights_sat_dict = {}
    activ_sat_dict = {}
    bias_sat_dict = {}
    #if either of the minmax values are inf for the layer, set that entry to be true in the sat dict

    for name, values in weights_minmax.items():
        min_val = values["min"]
        max_val = values["max"]

        if math.isinf(min_val) or math.isinf(max_val):
            weights_sat_dict[name] = True
        else:
            weights_sat_dict[name] = False

    for name, values in activ_minmax.items():
        min_val = values["min"]
        max_val = values["max"]

        if math.isinf(min_val) or math.isinf(max_val):
            activ_sat_dict[name] = True
        else :
            activ_sat_dict[name] = False

    for name, values in bias_minmax.items():
        min_val = values["min"]
        max_val = values["max"]

        if math.isinf(min_val) or math.isinf(max_val):
            bias_sat_dict[name] = True
        else :
            bias_sat_dict[name] = False

    if sensitivity_measure == "hessian":
        weight_sens, activ_sens, bias_sens, accum_sens = sensitivities.hess_sensitivity(
            lof_model, data, n_samples, device, collate_fn=collate_fn
        )
    else:
        weight_sens, activ_sens, bias_sens, accum_sens = sensitivities.noise_sensitivity_full(
            lof_model, data, loss_fn, n_samples, device, collate_fn=collate_fn
        )

    # Initialize all sensitivities to 0 (skips actual hessian/noise computation)

    # weight_sens = {}
    # activ_sens = {}
    # bias_sens = {}
    # accum_sens = {}
    # for name, module in lof_model.named_modules():
    #     if isinstance(module, (LoF_Conv2d, LoF_Linear)):
    #         weight_sens[name] = {'weight': 0.0}
    #         activ_sens[name] = 0.0
    #         bias_sens[name] = {'bias': 0.0}
    #         accum_sens[name] = 0.0

    print("weight sensitivities:")
    print(weight_sens)
    print("activation sensitivities:")
    print(activ_sens)
    print("bias sensitivities:")
    print(bias_sens)
    print("accum sensitivities:")
    print(accum_sens)

    def get_sens_val(sens_dict, key):
        v = sens_dict.get(key, 0)
        if isinstance(v, dict):
            return next(iter(v.values()), 0)
        return v if isinstance(v, (int, float)) else 0

    # Mantissa/exponent search: sort by the most sensitive of weight/activ/bias
    def max_sort_key(item):
        k, _ = item
        return max(
            get_sens_val(weight_sens, k),
            get_sens_val(activ_sens, k),
            get_sens_val(bias_sens, k),
        )

    # Accumulation search: sort by accumulator sensitivity
    def accum_sort_key(item):
        k, _ = item
        return get_sens_val(accum_sens, k)

    ll = [k for k, v in sorted(weight_sens.items(), key=max_sort_key)]

    # Defense-in-depth: skipped layers are no longer LoF, so they shouldn't
    # appear in sensitivity dicts. Kept as a guard in case a downstream
    # sensitivity routine ever returns them.
    skip_layers = set(skip_layer_names)

    # ── initialise mantissa dicts (not searched yet, but needed for set_mantissa_fields) ──
    max_b = max(bs)
    w_weights = {layer: max_b for layer in list(weight_sens.keys())}
    w_activ   = {layer: max_b for layer in list(weight_sens.keys())}
    w_bias    = {layer: max_b for layer in list(weight_sens.keys())}

    # ── initialise accum dicts ──
    max_accum_b = max(accum_bw)
    accum_precs = {layer: max_accum_b for layer in list(weight_sens.keys())}
    lof.set_accumulation_precisions(lof_model, accum_precs)



    for name, module in lof_model.named_modules():
        if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
            continue
        if name in skip_layers:
            continue
        if name not in w_weights:
            w_weights[name] = max_b
        if name not in w_activ:
            w_activ[name] = max_b
        if name not in w_bias:
            w_bias[name] = max_b
        if name not in accum_precs:
            accum_precs[name] = max_accum_b
        if name not in ll and name not in skip_layers:
            ll.append(name)

    bs = sorted(bs, reverse=True)

    # --- batching helper ---
    def make_batches(layer_list, bsz):
        if bsz <= 1:
            return [[layer] for layer in layer_list]
        return [layer_list[i:i + bsz] for i in range(0, len(layer_list), bsz)]

    lof.set_saturation_modes(lof_model, activ_sat_dict, weights_sat_dict, bias_sat_dict)

    # ==================== ACCUMULATION SEARCH ====================
    print("accum sensitivities:")
    print(accum_sens)
    ll = [k for k, v in sorted(accum_sens.items(), key=accum_sort_key)]
    for name, module in lof_model.named_modules():
        if not isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            continue
        if name in skip_layers:
            continue
        if name not in ll:
            ll.append(name)

    accum_bw = sorted(accum_bw, reverse=True)
    for b in accum_bw:
        ql = []

        for batch in make_batches(ll, batch_size):
            active_layers = [l for l in batch if l not in skip_layers]
            if not active_layers:
                continue

            prev = {l: accum_precs[l] for l in active_layers}

            for l in active_layers:
                accum_precs[l] = b

            lof.set_accumulation_precisions(lof_model, accum_precs)
            a = abs(eval_fn(lof_model, data))

            if a <= accuracy_target:
                ql.extend(active_layers)
            else:
                for l in active_layers:
                    accum_precs[l] = prev[l]

        ll = ql


    # ==================== MANTISSA SEARCH ====================
    for b in bs:
        ql = []

        for batch in make_batches(ll, batch_size):
            active_layers = [l for l in batch if l not in skip_layers]
            if not active_layers:
                continue

            prev = {l: (w_weights[l], w_activ[l], w_bias[l]) for l in active_layers}

            for l in active_layers:
                w_weights[l] = b
                w_activ[l]   = b
                w_bias[l]    = b

            lof.set_mantissa_fields(model=lof_model,
                                    activation_mantissa_bits=w_activ,
                                    weight_mantissa_bits=w_weights,
                                    bias_mantissa_bits=w_bias)

            a = abs(eval_fn(lof_model, data))

            if a <= accuracy_target:
                ql.extend(active_layers)
            else:
                for l in active_layers:
                    w_weights[l], w_activ[l], w_bias[l] = prev[l]

        ll = ql

    # ==================== EXPONENT SEARCH ====================
    ll = [k for k, v in sorted(weight_sens.items(), key=max_sort_key)]
    print("weight sensitivities:")
    print(weight_sens)

    for name in bias_exp:
        bias_exp[name] = max(bias_exp[name], 1)
        weights_exp[name] = max(weights_exp[name], 1)
        activ_exp[name] = max(activ_exp[name], 1)

    max_e = max(es)
    for name, module in lof_model.named_modules():
        if not isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            continue
        if name in skip_layers:
            continue
        if name not in weights_exp:
            weights_exp[name] = max_e
        if name not in activ_exp:
            activ_exp[name] = max_e
        if name not in bias_exp:
            bias_exp[name] = max_e
        if name not in ll:
            ll.append(name)

    # ── MX tile-shape search state (per new_MX_formulation.md) ──
    # When lowering exp bits introduces MX underflow for a layer, grow that
    # layer's MX tile as large as possible subject to the no-underflow (hard)
    # / accuracy (soft) constraint. Tiles chosen here are recorded in
    # `mx_tiles` and re-applied to the final rebuilt model below.
    mx_tiles = {}
    mx_act_cache = None  # captured lazily on first need (representative inputs)
    modules_by_name = dict(lof_model.named_modules())

    # ── Uniform-tile state (tile_mode='uniform'): one tile per precision class,
    # per operand. Keyed by mx_format_key(element_format). ──
    weight_tile_by_fmt = {}
    activ_tile_by_fmt = {}

    def _run_mx_search_for_layers(layers):
        nonlocal mx_act_cache
        if not microscaling or not use_mx_search:
            return
        # Make the committed exp config live on the modules so module.*_params
        # reflect the just-lowered exponent widths for the underflow counts.
        lof.set_exponent_fields(model=lof_model,
                                activation_exp_bits=activ_exp,
                                weight_exp_bits=weights_exp,
                                bias_exp_bits=bias_exp)
        for layer in layers:
            module = modules_by_name.get(layer)
            if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
                continue

            elem_fmt = module.weight_params
            scl_fmt = getattr(module, "scale_format", None)
            if scl_fmt is None:
                scl_fmt = lof.create_e8m0_params()
            cur_bs = getattr(module, "mx_block_size", mx_block_size)
            cur_tile = ((int(cur_bs[0]), int(cur_bs[1]))
                        if isinstance(cur_bs, (tuple, list))
                        else (1, int(cur_bs)))

            w2d = module.weight.detach().reshape(module.weight.shape[0], -1).float()

            if mx_act_cache is None:
                mx_act_cache = _capture_layer_inputs(
                    lof_model, data, n_samples, device, collate_fn=collate_fn
                )
            act = mx_act_cache.get(layer)

            # Did lowering exp bits introduce underflow at the current tile?
            uf = int(lof.count_mx_underflow(
                w2d, elem_fmt, scale_format=scl_fmt,
                block_size=cur_tile, scale_target=scale_target))
            if act is not None:
                uf += int(lof.count_mx_underflow(
                    act, elem_fmt, scale_format=scl_fmt,
                    block_size=cur_tile, scale_target=scale_target))
            if uf <= 0:
                continue  # no underflow -> leave the layer's tile unchanged.

            def _acc_pred(tile, _layer=layer):
                # Return the relative accuracy drop (float); mx_search compares
                # it against accuracy_target and logs acc_drop per candidate.
                lof.set_mx_tile_fields(lof_model, {_layer: tile})
                return abs(eval_fn(lof_model, data))

            print(f"[mx_search] {layer}: START (underflow {uf} at current tile "
                  f"{cur_tile}); searching...")
            tile = mx_search(
                w2d, act, elem_fmt, scl_fmt, scale_target,
                accuracy_fn=_acc_pred, accuracy_target=accuracy_target,
                label=layer,
            )
            mx_tiles[layer] = tile
            lof.set_mx_tile_fields(lof_model, {layer: tile})

    def _run_uniform_tile_pass():
        """Uniform per-(operand, format) MX tile selection. For every distinct
        element format currently present across the lofloatified layers — per
        operand — pick ONE largest-area tile shared by the whole class, update
        the {fmt_key -> tile} dicts, then apply per-operand to every layer.

        Two criteria (tile_criterion): 'frob' picks each class independently by
        the class-max relative Frobenius error <= frob_tau (deterministic, no
        forward pass). 'accuracy' picks by NETWORK mAP: because mAP couples all
        classes, it selects GREEDILY in a deterministic class order, holding all
        other classes at their current dict tiles."""
        nonlocal mx_act_cache
        if not microscaling:
            return
        # Make the committed exp config live so module.*_params are current.
        lof.set_exponent_fields(model=lof_model,
                                activation_exp_bits=activ_exp,
                                weight_exp_bits=weights_exp,
                                bias_exp_bits=bias_exp)
        if mx_act_cache is None:
            mx_act_cache = _capture_layer_inputs(
                lof_model, data, n_samples, device, collate_fn=collate_fn
            )
        scale_fmt = lof.create_e8m0_params()

        # Group each operand's matrices + member layers by element-format class.
        weight_classes = {}   # fmt_key -> (element_format, [matrices])
        activ_classes = {}
        weight_members, activ_members = {}, {}   # fmt_key -> [layer names]
        layer_wk, layer_ak = {}, {}
        for name, module in modules_by_name.items():
            if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
                continue
            if name in skip_layers:
                continue
            wfmt, afmt = module.weight_params, module.act_params
            wk, ak = mx_format_key(wfmt), mx_format_key(afmt)
            layer_wk[name], layer_ak[name] = wk, ak
            w2d = module.weight.detach().reshape(
                module.weight.shape[0], -1).float()
            weight_classes.setdefault(wk, (wfmt, []))[1].append(w2d)
            weight_members.setdefault(wk, []).append(name)
            act = mx_act_cache.get(name)
            if act is not None:
                activ_classes.setdefault(ak, (afmt, []))[1].append(act.float())
                activ_members.setdefault(ak, []).append(name)

        def _apply_all_dicts():
            """Apply the current {fmt->tile} dicts per-operand to every layer,
            record into mx_tiles for the final rebuild."""
            tile_dict = {}
            for name, wk in layer_wk.items():
                ak = layer_ak.get(name)
                entry = {}
                if wk in weight_tile_by_fmt:
                    entry["weight"] = weight_tile_by_fmt[wk]
                if ak is not None and ak in activ_tile_by_fmt:
                    entry["act"] = activ_tile_by_fmt[ak]
                if entry:
                    tile_dict[name] = entry
                    mx_tiles[name] = entry
            if tile_dict:
                lof.set_mx_tile_fields(lof_model, tile_dict)

        if tile_criterion == 'accuracy':
            # Greedy over classes in a deterministic order, coupled through mAP.
            # Start from the current dict state applied to all layers.
            _apply_all_dicts()
            order = ([('weight', k) for k in sorted(weight_classes, key=str)] +
                     [('act', k) for k in sorted(activ_classes, key=str)])
            for operand, key in order:
                if operand == 'weight':
                    fmt = weight_classes[key][0]
                    members = weight_members[key]
                    table = weight_tile_by_fmt
                else:
                    fmt = activ_classes[key][0]
                    members = activ_members[key]
                    table = activ_tile_by_fmt

                def accept(t, _members=members, _operand=operand):
                    lof.set_mx_tile_fields(
                        lof_model, {n: {_operand: t} for n in _members})
                    return abs(eval_fn(lof_model, data)) <= accuracy_target

                tile = uniform_tile_search(accept, fmt, start=16)
                table[key] = tile
                # Re-apply the winner (last accept() may have set a rejected tile)
                # and report the resulting mAP drop.
                lof.set_mx_tile_fields(
                    lof_model, {n: {operand: tile} for n in members})
                drop = abs(eval_fn(lof_model, data))
                print(f"[uniform-tile] {operand} fmt={key} -> {tile} "
                      f"mAP_drop={drop:.4f} (n_matrices={len(members)})")
        else:
            # Frob criterion: each class independent, deterministic.
            for wk, (wfmt, mats) in weight_classes.items():
                tile = uniform_mx_tile(mats, wfmt, scale_fmt, scale_target,
                                       frob_tau)
                weight_tile_by_fmt[wk] = tile
                v = mx_rel_frob_max(mats, wfmt, scale_fmt, scale_target, tile)
                print(f"[uniform-tile] weight fmt={wk} -> {tile} "
                      f"maxFrob={v:.4f} (n_matrices={len(mats)})")
            for ak, (afmt, mats) in activ_classes.items():
                tile = uniform_mx_tile(mats, afmt, scale_fmt, scale_target,
                                       frob_tau)
                activ_tile_by_fmt[ak] = tile
                v = mx_rel_frob_max(mats, afmt, scale_fmt, scale_target, tile)
                print(f"[uniform-tile] activation fmt={ak} -> {tile} "
                      f"maxFrob={v:.4f} (n_matrices={len(mats)})")

        # Apply the resulting per-operand tiles to every layer.
        _apply_all_dicts()

    es = sorted(es, reverse=True)
    for b in es:
        ql = []
        candidates_this_level = [l for l in ll if l not in skip_layers]

        for batch in make_batches(ll, batch_size):
            active_layers = [l for l in batch if l not in skip_layers]
            if not active_layers:
                continue

            prev = {}
            for layer in active_layers:
                try:
                    prev_w = weights_exp[layer]
                except KeyError:
                    prev_w = b
                    weights_exp[layer] = b
                try:
                    prev_a = activ_exp[layer]
                except KeyError:
                    prev_a = b
                    activ_exp[layer] = b
                try:
                    prev_b_val = bias_exp[layer]
                except KeyError:
                    prev_b_val = b
                    bias_exp[layer] = b
                prev[layer] = (prev_w, prev_a, prev_b_val)

            for layer in active_layers:
                prev_w, prev_a, prev_b_val = prev[layer]
                weights_exp[layer] = min(b, prev_w)
                activ_exp[layer]   = min(b, prev_a)
                bias_exp[layer]    = min(b, prev_b_val)

            lof.set_exponent_fields(model=lof_model,
                                    activation_exp_bits=activ_exp,
                                    weight_exp_bits=weights_exp,
                                    bias_exp_bits=bias_exp)
            a = abs(eval_fn(lof_model, data))

            if a <= accuracy_target:
                ql.extend(active_layers)
            else:
                for layer in active_layers:
                    prev_w, prev_a, prev_b_val = prev[layer]
                    weights_exp[layer] = prev_w
                    activ_exp[layer]   = prev_a
                    bias_exp[layer]    = prev_b_val

        ll = ql

        print(f"[exp-search] level b={b}: accepted exp={b} on "
              f"{len(ql)}/{len(candidates_this_level)} candidate layers")

        # After committing this exp level, pick MX tiles. In 'per_layer' mode,
        # grow a per-layer tile only where lowering exp actually introduced
        # underflow. In 'uniform' mode, re-pick one tile per (operand, format)
        # class across all layers. Everything else stays identical.
        if tile_mode == 'uniform':
            _run_uniform_tile_pass()
        else:
            _run_mx_search_for_layers(ql)

    # ── Final dump of the committed per-layer config (greppable) ──
    def _exp_hist(d):
        from collections import Counter
        return dict(sorted(
            Counter(v for k, v in d.items() if k not in skip_layers).items(),
            reverse=True))

    print(f"[exp-search] final exponent distribution (weights): "
          f"{_exp_hist(weights_exp)}")
    print(f"[exp-search] final exponent distribution (activations): "
          f"{_exp_hist(activ_exp)}")
    print(f"[exp-search] final weights_exp: {weights_exp}")
    print(f"[exp-search] final activ_exp: {activ_exp}")
    print(f"[exp-search] chosen MX tiles ({len(mx_tiles)} layers): {mx_tiles}")
    if tile_mode == 'uniform':
        print(f"[uniform-tile] final weight_tile_by_fmt: {weight_tile_by_fmt}")
        print(f"[uniform-tile] final activ_tile_by_fmt: {activ_tile_by_fmt}")

    # ======================GPTQ pruning======================
    # lof_model = sensitivities.quantize_weights_with_gptq(
    #     model=lof_model, dataset=data, mantissa_bits=w_weights,
    #     exponent_bits=weights_exp, n_samples=128, device=device, micro_batch_size=4, collate_fn=collate_fn
    # )

    
    # ==================== APPLY FINAL CONFIG ====================
    lof_model = lof.lofloatify(model, skip_layer_names=skip_layer_names)
    lof_model = lof_model.to(device)

    if hadamard:
        apply_hadamard_to_weights(lof_model, skip_first=False, skip_last=False)

    # Re-enable MX on the freshly rebuilt model so the final config matches what
    # the search evaluated.
    if microscaling:
        enable_microscaling(lof_model, scale_target=scale_target,
                            block_size=mx_block_size)
        # Re-apply the per-layer MX tiles chosen during the exponent search
        # (the rebuild above reset every layer to the default MX block).
        if mx_tiles:
            lof.set_mx_tile_fields(lof_model, mx_tiles)
            print(f"[mx_search] re-applied {len(mx_tiles)} searched MX tiles "
                  f"on the final model.")

    print("running gptq with exp_bits = %d and mantissa_bits = %d" %
          (list(weights_exp.values())[9], list(w_weights.values())[9]))
    print(f"First/last layers ({sorted(skip_layer_names)}) kept as full-precision "
          f"nn.Linear/nn.Conv2d (not lofloatified).")

    lof.set_mantissa_fields(
        model=lof_model,
        activation_mantissa_bits=w_activ,
        weight_mantissa_bits=w_weights,
        bias_mantissa_bits=w_bias,
    )
    for name, module in lof_model.named_modules():
        if isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            print(f"{name}: mantissa={module.weight_params.mantissa_bits}, "
                  f"in dict={w_weights.get(name, 'MISSING')}")
            break
    lof.set_exponent_fields(
        model=lof_model,
        activation_exp_bits=activ_exp,
        weight_exp_bits=weights_exp,
        bias_exp_bits=bias_exp,
    )

    print(accum_precs)
    lof.set_accumulation_precisions(lof_model, accum_precs)
    lof_model = lof_model.to(device)

    # ==================== BN + SILU REPLACEMENT ====================
    L1_s, Linf_s = sensitivities.find_batchnorm_scales(
        lof_model, data, n_samples, device, collate_fn=collate_fn
    )

    bn_candidates = [
        (1.0,      "L1",   {"L1_scales": L1_s, "Linf_scales": Linf_s}),
        (math.inf, "Linf", {"L1_scales": L1_s, "Linf_scales": Linf_s}),
        ("fisr",   "FISR", {}),
        ("pwl",    "PWL",  {"lut_bits": 8, "lut_method": "minimax"}),
    ]

    chosen = None
    best_acc = float('inf')
    for p, tag, kwargs in bn_candidates:
        cand = replace_batchnorm2d(copy.deepcopy(lof_model), p=p, **kwargs)
        cand = cand.to(device)
        acc = abs(eval_fn(cand, data))
        ok = acc <= accuracy_target
        print(f"{tag} BN replacement: "
              f"{f'OK acc={acc:.4f}' if ok else f'FAILED acc={acc:.4f}'}")
        if ok and acc <= best_acc:
            best_acc = acc
            chosen = cand
            chosen_tag = tag

    if chosen is not None:
        print(f"  -> using {chosen_tag} BN replacement (acc={best_acc:.4f})")

    if chosen is None:
        print("Trying again with higher-precision PWL BN LUT...")
        cand = replace_batchnorm2d(copy.deepcopy(lof_model), p="pwl", lut_bits=12, lut_method="minimax")
        cand = cand.to(device)
        acc = abs(eval_fn(cand, data))
        if acc <= accuracy_target:
            print(f"  -> using higher-precision PWL BN replacement (acc={acc:.4f})")
            chosen = cand
        else:
            cand = replace_batchnorm2d(copy.deepcopy(lof_model), p="pwl", lut_bits=16, lut_method="minimax")
            cand = cand.to(device)
            acc = abs(eval_fn(cand, data))
            if acc <= accuracy_target:
                print(f"  -> using even higher-precision PWL BN replacement (acc={acc:.4f})")
                chosen = cand
            else:
                print(f"  -> higher-precision PWL BN replacement also failed (acc={acc:.4f}), keeping original BN")

    base = chosen if chosen is not None else lof_model
    base = base.to(device)

    silu_cand = replace_silu(copy.deepcopy(base),
                             R=8.0, lut_bits=6, lut_method="minimax").to(device)
    acc = abs(eval_fn(silu_cand, data))
    if acc <= accuracy_target:
        print(f"SiLU LUT replacement: OK acc={acc:.4f}")
        final_model = silu_cand.to(device)
    else:
        print(f"SiLU LUT replacement: FAILED acc={acc:.4f}, trying larger LUT")
        silu_cand = replace_silu(copy.deepcopy(base),
                                 R=8.0, lut_bits=10, lut_method="minimax").to(device)
        acc = abs(eval_fn(silu_cand, data))
        if acc <= accuracy_target:
            print(f"SiLU LUT replacement with larger LUT: OK acc={acc:.4f}")
            final_model = silu_cand.to(device)
        else:
            print(f"SiLU LUT replacement with larger LUT: FAILED acc={acc:.4f}, increasing LUT size once more")
            silu_cand = replace_silu(copy.deepcopy(base),
                                     R=8.0, lut_bits=12, lut_method="minimax").to(device)
            acc = abs(eval_fn(silu_cand, data))
            if acc <= accuracy_target:
                print(f"SiLU LUT replacement with even larger LUT: OK acc={acc:.4f}")
                final_model = silu_cand.to(device)
            else:
                print(f"SiLU LUT replacement with even larger LUT: FAILED acc={acc:.4f}, keeping original SiLU")
                final_model = base.to(device)


    return final_model

