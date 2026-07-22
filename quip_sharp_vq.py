"""
QuIP#-style fixed-codebook (E8 lattice) vector quantization
===========================================================

This module adds a *fixed-codebook* vector-quantization experiment to the
LoFloatQuantizationExp harness, in the spirit of **QuIP#**
(Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence
and Lattice Codebooks", arXiv:2402.04396).

Unlike the k-means / RVQ / PQ experiments already in this repo (which *learn* a
data-dependent codebook per layer), QuIP# uses a **data-independent codebook
built from the E8 lattice** — the optimal 8-dimensional sphere packing.  The two
ingredients are:

  1. **Incoherence processing** via a Randomized Hadamard Transform (RHT).
     Multiplying the weight matrix on the left and right by ``(1/sqrt n) H diag(s)``
     (H a Hadamard matrix, ``s`` a random +/-1 sign vector) makes the weights
     approximately i.i.d. sub-Gaussian, i.e. "ball shaped".  The rotation is
     orthogonal, so it is undone exactly after quantization (fake-quant: the
     layer's forward pass is unchanged apart from the quantization error).

  2. **E8 lattice quantization.**  The ball-shaped, incoherent weights are
     quantized in groups of 8 to the nearest point of the E8 lattice.  Two
     flavours are provided:
        * ``e8lattice`` -- exact nearest-E8-point via the Conway&Sloane decoder
          (a true, unbounded lattice quantizer; rate is controlled by the scale).
        * ``e8p`` -- QuIP#'s finite **E8P** codebook (2^16 entries -> exactly
          2 bits/weight for an 8-dim block).  See :func:`build_e8p_codebook`.
     Higher rates (e.g. 4-bit) are obtained by **residual VQ**: quantize the
     residual of the first stage with a second (scaled) codebook.

The module is deliberately self-contained (plain PyTorch, no LoFloat kernel
dependency) so it can quantize an arbitrary ``nn.Module`` and be dropped into any
eval harness.  ``quip_sharp_yolo.py`` (under ``YOLO_test/``) uses it to run a
*full* COCO / Pascal mAP evaluation, reusing ``yolo_test.py``'s eval code.

Reference notes distilled from the paper live in ``QUIP_SHARP_NOTES.md``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn


# ===================================================================
#  Randomized Hadamard Transform (incoherence processing)
# ===================================================================
def _hadamard_pow2(n: int, device, dtype) -> torch.Tensor:
    """Un-normalized Sylvester Hadamard matrix of size n (n a power of two)."""
    assert n > 0 and (n & (n - 1)) == 0, f"{n} is not a power of two"
    H = torch.ones((1, 1), device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], dim=1),
                       torch.cat([H, -H], dim=1)], dim=0)
    return H


def _largest_pow2_divisor(n: int) -> int:
    return n & (-n) if n > 0 else 0


def _random_orthogonal(n: int, generator: torch.Generator, device, dtype) -> torch.Tensor:
    """Haar-random orthogonal n x n matrix via QR of a Gaussian."""
    if n == 1:
        return torch.ones((1, 1), device=device, dtype=dtype)
    a = torch.randn((n, n), generator=generator, device=device, dtype=dtype)
    q, r = torch.linalg.qr(a)
    # fix signs so the map is a proper Haar sample
    d = torch.sign(torch.diagonal(r))
    d[d == 0] = 1.0
    return q * d.unsqueeze(0)


_RHT_CACHE: dict = {}


def build_rht_matrix(n: int, seed: int, device="cpu", dtype=torch.float64) -> torch.Tensor:
    """Return an orthogonal n x n randomized-Hadamard rotation matrix.

    Following QuIP#, when ``n`` is not a power of two we Kronecker a Hadamard
    matrix over the largest power-of-two factor with a (seeded) Haar-random
    orthogonal matrix over the remaining factor, then apply a random +/-1 sign
    diagonal.  The result is exactly orthogonal for any ``n``.
    """
    key = (n, seed, str(device), str(dtype))
    cached = _RHT_CACHE.get(key)
    if cached is not None:
        return cached
    gen = torch.Generator(device="cpu").manual_seed(seed * 2654435761 % (2 ** 31) + n)
    a = _largest_pow2_divisor(n)
    b = n // a
    H = _hadamard_pow2(a, device="cpu", dtype=dtype) / math.sqrt(a)
    if b == 1:
        R0 = H
    else:
        Q = _random_orthogonal(b, gen, device="cpu", dtype=dtype)
        R0 = torch.kron(H.contiguous(), Q.contiguous())
    signs = torch.randint(0, 2, (n,), generator=gen, dtype=torch.int64) * 2 - 1
    R = R0 * signs.to(dtype).unsqueeze(0)   # right-multiply by diag(signs)
    R = R.to(device=device, dtype=dtype)
    _RHT_CACHE[key] = R
    return R


# ===================================================================
#  E8 lattice: exact nearest-point decoder (Conway & Sloane, SPLAG)
# ===================================================================
def _nearest_Dn(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of the checkerboard lattice D_n (integers with even sum).

    x: (..., n).  Returns the nearest D_n point with the same shape.
    """
    f = torch.round(x)
    parity = f.sum(dim=-1)
    odd = (parity.long() % 2 != 0)
    if odd.any():
        diff = x - f
        # coordinate whose rounding error is largest -> round it the other way
        idx = torch.argmax(torch.abs(diff), dim=-1, keepdim=True)
        chosen_diff = torch.gather(diff, -1, idx)
        chosen_f = torch.gather(f, -1, idx)
        flipped = torch.where(chosen_diff >= 0, chosen_f + 1.0, chosen_f - 1.0)
        g = f.scatter(-1, idx, flipped)
        f = torch.where(odd.unsqueeze(-1), g, f)
    return f


def nearest_e8(x: torch.Tensor) -> torch.Tensor:
    """Nearest point of the E8 lattice = D8 union (D8 + 1/2).

    x: (..., 8).  Returns the nearest E8 lattice point.
    """
    c1 = _nearest_Dn(x)
    c2 = _nearest_Dn(x - 0.5) + 0.5
    d1 = ((x - c1) ** 2).sum(dim=-1)
    d2 = ((x - c2) ** 2).sum(dim=-1)
    return torch.where((d1 <= d2).unsqueeze(-1), c1, c2)


# ===================================================================
#  E8P finite codebook (QuIP#)
# ===================================================================
# The E8P codebook (QUIP_SHARP_NOTES.md sec. "E8P Codebook Construction") is a
# 2^16-entry codebook built from the half-integer lattice
#     D̂8 = { x in Z^8 + 1/2 : 1^T x is even }
# It stores a 2^8 = 256 "source" codebook of magnitude patterns (227 with
# ‖·‖^2 <= 10 plus 29 padding at ‖·‖^2 = 12) and expands each by 2^7 even-sign-
# flip patterns (the 8th sign fixed by the even-sum parity) and a 1-bit +1/4
# shift:  256 * 128 * 2 = 65536 = 2^16, i.e. exactly 2 bits / weight over an
# 8-dim block.
#
# For quantization *quality* only the materialized set of codebook vectors
# matters (the source-codebook + sign-bit factorization is a storage/kernel
# trick that does not change which point a block snaps to).  Since every D̂8
# point has all-nonzero (half-integer) coordinates, each of the 256 magnitude
# reps has exactly 2^7 sign patterns, so the 256*128 = 2^15 lowest-norm D̂8
# points ARE the sign-expanded source codebook.  We therefore materialize E8P
# directly as {2^15 closest D̂8 points} ∪ {same + 1/4} and do exact
# nearest-neighbour assignment.  The 2-bits/weight budget is accounted for
# analytically as stages * log2(K) / 8.
def _enum_lattice_ball(values: torch.Tensor, max_norm2: float) -> torch.Tensor:
    """All 8-tuples of ``values`` with squared norm <= ``max_norm2``.

    Built incrementally with per-coordinate norm pruning so the frontier stays
    bounded (only points inside the ball survive), avoiding a full |values|^8
    cartesian product.
    """
    partial = values.reshape(-1, 1)
    for _ in range(7):
        m = values.numel()
        left = partial.repeat_interleave(m, dim=0)
        right = values.reshape(m, 1).repeat(partial.shape[0], 1)
        cand = torch.cat([left, right], dim=1)
        keep = (cand ** 2).sum(dim=1) <= max_norm2 + 1e-6
        partial = cand[keep]
    return partial


def _enumerate_dhat8(max_abs_half: float = 2.5, device="cpu") -> torch.Tensor:
    """D̂8 points (all-half-integer, even coordinate sum) with |coord| bounded."""
    k = int(max_abs_half + 0.5)
    vals = (torch.arange(-k, k, dtype=torch.float64) + 0.5)
    pts = _enum_lattice_ball(vals, max_norm2=float(8 * (k - 0.5) ** 2))
    even = (torch.round(pts.sum(dim=1)).long() % 2 == 0)
    return pts[even].to(device)


def build_e8p_codebook(n_bits: int = 16, device="cpu", normalize: bool = True,
                       kind: str = "e8") -> torch.Tensor:
    """Build a fixed E8-lattice codebook with ``2**n_bits`` entries.

    kind="e8"  (default): the ``2**n_bits`` lowest-norm points of the *full* E8
        lattice (D8 integer points ∪ half-integer coset).  This **includes the
        origin** and small integer points, which is essential for residual VQ
        (a small residual can snap to 0 instead of a min-norm point) and for a
        clean bits->accuracy trade-off.
    kind="dhat8": the paper-exact QuIP# E8P — 2^15 lowest-norm D̂8 (half-integer)
        points + their +1/4-shifted copies.  Faithful to the paper but has no
        origin, so residual stages saturate without fine-tuning.

    ``normalize`` rescales to unit per-coordinate RMS so a global/per-channel
    weight scale can be estimated cleanly.
    """
    K = 1 << n_bits
    if kind == "dhat8":
        half = K // 2
        pts = _enumerate_dhat8(2.5, device=device)
        if pts.shape[0] < half:
            pts = _enumerate_dhat8(3.5, device=device)
        order = torch.argsort((pts ** 2).sum(dim=1))
        base = pts[order[:half]].contiguous()
        cb = torch.cat([base, base + 0.25], dim=0)
    else:  # full E8 lattice, closest K points (includes origin)
        R = 12.0
        for _ in range(4):
            iv = torch.arange(-4, 5, dtype=torch.float64)          # integer D8
            ipts = _enum_lattice_ball(iv, R)
            ipts = ipts[torch.round(ipts.sum(dim=1)).long() % 2 == 0]
            hv = torch.arange(-4, 4, dtype=torch.float64) + 0.5    # half-integer coset
            hpts = _enum_lattice_ball(hv, R)
            hpts = hpts[torch.round(hpts.sum(dim=1)).long() % 2 == 0]
            pts = torch.cat([ipts, hpts], dim=0)
            if pts.shape[0] >= K:
                break
            R += 4.0
        order = torch.argsort((pts ** 2).sum(dim=1))
        cb = pts[order[:K]].contiguous().to(device)
    if normalize:
        cb = cb / cb.pow(2).mean().sqrt()
    return cb.float()


# ===================================================================
#  Vector-quantization primitives
# ===================================================================
def _assign_nearest(x: torch.Tensor, codebook: torch.Tensor,
                    chunk: int = 4096) -> torch.Tensor:
    """Nearest-codebook-entry index for each row of x (chunked cdist argmin).

    x: (N, 8), codebook: (K, 8).  Returns (N,) long indices.
    """
    N = x.shape[0]
    idx = torch.empty(N, dtype=torch.long, device=x.device)
    cb2 = (codebook ** 2).sum(dim=1)                    # (K,)
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        xb = x[s:e]                                     # (c, 8)
        # ||x||^2 - 2 x.c^T + ||c||^2  ; drop ||x||^2 (constant per row)
        d = -2.0 * (xb @ codebook.t()) + cb2.unsqueeze(0)
        idx[s:e] = torch.argmin(d, dim=1)
    return idx


def _quantize_one_stage(blocks: torch.Tensor, method: str,
                        codebook: Optional[torch.Tensor],
                        scale_refine_iters: int = 4) -> torch.Tensor:
    """Snap (N,8) blocks with a single codebook/lattice stage.

    A single least-squares-refined global scale is fitted for this stage.
    Returns the decoded (N,8) approximation.
    """
    if method == "e8lattice":
        return nearest_e8(blocks)
    assert codebook is not None
    scale = blocks.pow(2).mean().sqrt().clamp_min(1e-8)
    dec = None
    for _ in range(max(1, scale_refine_iters)):
        idx = _assign_nearest(blocks / scale, codebook)
        dec = codebook[idx] * scale
        num = (blocks * dec).sum()
        den = (dec * dec).sum().clamp_min(1e-12)
        g = (num / den).item()
        if g > 0:
            scale = scale / g
    idx = _assign_nearest(blocks / scale, codebook)
    return codebook[idx] * scale


# ===================================================================
#  Per-tensor weight quantization
# ===================================================================
@dataclass
class LayerStat:
    name: str
    shape: tuple
    numel: int
    bpw: float
    w_relerr: float


def quantize_weight_tensor(W: torch.Tensor, method: str = "e8p",
                           codebook: Optional[torch.Tensor] = None,
                           use_rht: bool = True, stages: int = 1,
                           scale_refine_iters: int = 4, lattice_rms: float = 1.0,
                           per_channel: bool = True,
                           rht_seed: int = 0, compute_dtype=torch.float32):
    """Fake-quantize a 2-D weight matrix (out, in_flat) with QuIP#-style VQ.

    Pipeline: incoherence-rotate -> per-output-channel normalize -> group into
    8-dim blocks -> (residual) E8/E8P quantize -> unnormalize -> undo rotation.
    Returns (W_hat, bits_per_weight, w_relerr).
    """
    orig_dtype = W.dtype
    dev = W.device
    Wc = W.detach().to(compute_dtype)
    out_f, in_f = Wc.shape

    # ---- incoherence processing: W_incoh = R_out^T W R_in ----
    if use_rht:
        R_in = build_rht_matrix(in_f, rht_seed + 1, device=dev, dtype=compute_dtype)
        R_out = build_rht_matrix(out_f, rht_seed + 2, device=dev, dtype=compute_dtype)
        W_incoh = R_out.t() @ Wc @ R_in
    else:
        R_in = R_out = None
        W_incoh = Wc

    # ---- per-output-channel scale (nearly free: one fp16 per row) ----
    if per_channel and method != "e8lattice":
        row_scale = W_incoh.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        W_incoh = W_incoh / row_scale
    else:
        row_scale = None

    flat = W_incoh.reshape(-1)
    pad = (-flat.numel()) % 8
    if pad:
        flat = torch.cat([flat, flat.new_zeros(pad)])
    blocks = flat.reshape(-1, 8)

    # ---- (residual) vector quantization ----
    if method == "e8lattice":
        # fixed scale sets the lattice rate (rate grows with lattice_rms)
        s = blocks.pow(2).mean().sqrt().clamp_min(1e-8) / max(lattice_rms, 1e-6)
        dec_total = nearest_e8(blocks / s) * s
        n_stages = 1
    else:
        n_stages = max(1, stages)
        dec_total = torch.zeros_like(blocks)
        residual = blocks
        for _ in range(n_stages):
            dec = _quantize_one_stage(residual, method, codebook, scale_refine_iters)
            dec_total = dec_total + dec
            residual = residual - dec

    W_incoh_hat = dec_total.reshape(-1)[:W_incoh.numel()].reshape(out_f, in_f)
    if row_scale is not None:
        W_incoh_hat = W_incoh_hat * row_scale

    # ---- undo incoherence rotation ----
    if use_rht:
        W_hat = R_out @ W_incoh_hat @ R_in.t()
    else:
        W_hat = W_incoh_hat

    w_relerr = (W_hat - Wc).norm().item() / Wc.norm().clamp_min(1e-12).item()

    # ---- bits/weight ----
    if method == "e8lattice":
        # unbounded lattice: report log2(#distinct points actually used)/8
        bpw = _e8_empirical_bpw(blocks / s)
    else:
        bpw = n_stages * math.log2(codebook.shape[0]) / 8.0

    return W_hat.to(orig_dtype), bpw, w_relerr


# ===================================================================
#  BlockLDLQ: Hessian error-feedback rounding (QuIP#'s adaptive rounding)
# ===================================================================
# This is the vector generalization of GPTQ/LDLQ (cf. gptq.py in this repo).
# We reuse the exact GPTQ column-by-column error-feedback loop, but (a) apply
# the RHT incoherence rotation to W and H first, (b) replace the per-column
# scalar quantizer with a residual-E8P vector quantizer over 8-output-blocks
# (the RHT row rotation makes 8 consecutive output channels ~i.i.d., so an 8-dim
# lattice quantizer is well matched), and (c) rotate the result back.
def compute_layer_hessians(model: nn.Module, calib_inputs, device="cpu",
                           max_batches: int = 8, min_numel: int = 8192,
                           skip_first_last: bool = True, verbose: bool = True):
    """Accumulate the GPTQ Hessian H = E[x x^T] for each Conv2d/Linear layer.

    ``calib_inputs`` is an iterable of input batches (tensors) fed to ``model``.
    Returns {layer_name: H (cols x cols)} for the layers that would be quantized.
    """
    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, _QUANTIZABLE)]
    skip = set()
    if skip_first_last and layers:
        skip.add(layers[0][0]); skip.add(layers[-1][0])
    targets = {n: m for n, m in layers
               if n not in skip and m.weight.numel() >= min_numel}

    H = {n: None for n in targets}
    nsamp = {n: 0 for n in targets}

    def _make_hook(name, mod):
        def hook(_m, inp, _out):
            x = inp[0].detach()
            if isinstance(mod, nn.Conv2d):
                unfold = nn.Unfold(mod.kernel_size, dilation=mod.dilation,
                                   padding=mod.padding, stride=mod.stride)
                x = unfold(x)                       # (B, cols, L)
                x = x.permute(1, 0, 2).reshape(x.shape[1], -1)   # (cols, B*L)
            else:
                if x.dim() > 2:
                    x = x.reshape(-1, x.shape[-1])
                x = x.t()                           # (cols, N)
            x = x.float()
            b = x.shape[1]
            if H[name] is None:
                H[name] = torch.zeros((x.shape[0], x.shape[0]), device=device)
            H[name] *= nsamp[name] / (nsamp[name] + b)
            nsamp[name] += b
            H[name] += (math.sqrt(2.0 / nsamp[name]) * x) @ \
                       (math.sqrt(2.0 / nsamp[name]) * x).t()
        return hook

    handles = [m.register_forward_hook(_make_hook(n, m)) for n, m in targets.items()]
    model.eval()
    with torch.no_grad():
        for bi, batch in enumerate(calib_inputs):
            if bi >= max_batches:
                break
            model(batch.to(device))
    for h in handles:
        h.remove()
    if verbose:
        print(f"  [ldlq] computed Hessians for {len(H)} layers over "
              f"{min(max_batches, bi + 1)} calib batches")
    return H


def _fit_stage_scales(blocks: torch.Tensor, codebook: torch.Tensor,
                      stages: int, refine_iters: int = 4):
    """Greedy per-stage global scales for residual-E8P (fixed up front)."""
    scales = []
    residual = blocks
    for _ in range(stages):
        scale = residual.pow(2).mean().sqrt().clamp_min(1e-8)
        for _ in range(refine_iters):
            idx = _assign_nearest(residual / scale, codebook)
            dec = codebook[idx] * scale
            g = (residual * dec).sum() / (dec * dec).sum().clamp_min(1e-12)
            if g.item() > 0:
                scale = scale / g.item()
        idx = _assign_nearest(residual / scale, codebook)
        residual = residual - codebook[idx] * scale
        scales.append(float(scale))
    return scales


def _quantize_column_e8p(col: torch.Tensor, codebook: torch.Tensor,
                         scales) -> torch.Tensor:
    """Residual-E8P quantize a length-m column grouped into 8-output-blocks."""
    m = col.numel()
    pad = (-m) % 8
    x = col if pad == 0 else torch.cat([col, col.new_zeros(pad)])
    g = x.reshape(-1, 8)
    dec = torch.zeros_like(g)
    residual = g
    for sc in scales:
        idx = _assign_nearest(residual / sc, codebook)
        q = codebook[idx] * sc
        dec = dec + q
        residual = residual - q
    return dec.reshape(-1)[:m]


def blockldlq_quantize(W2d: torch.Tensor, H: torch.Tensor,
                       codebook: torch.Tensor, use_rht: bool = True,
                       stages: int = 1, rht_seed: int = 0, percdamp: float = 0.01,
                       per_channel: bool = True, compute_dtype=torch.float32):
    """QuIP# BlockLDLQ: RHT + E8P vector quantization with Hessian feedback.

    W2d: (out, in_flat).  H: (in_flat, in_flat) calibration Hessian.
    Returns (W_hat, bits_per_weight, w_relerr).
    """
    dev = W2d.device
    Wc = W2d.detach().to(compute_dtype)
    out_f, cols = Wc.shape

    if use_rht:
        R_in = build_rht_matrix(cols, rht_seed + 1, device=dev, dtype=compute_dtype)
        R_out = build_rht_matrix(out_f, rht_seed + 2, device=dev, dtype=compute_dtype)
        W = R_out.t() @ Wc @ R_in
        Hc = (R_in.t() @ H.to(compute_dtype) @ R_in)
    else:
        R_in = R_out = None
        W = Wc.clone()
        Hc = H.to(compute_dtype).clone()

    # per-output-channel normalization (rows); reduces dynamic range for the VQ
    if per_channel:
        row_scale = W.pow(2).mean(dim=1, keepdim=True).sqrt().clamp_min(1e-8)
        W = W / row_scale
    else:
        row_scale = None

    # global per-stage scales, fit on the (rotated, normalized) weights up front
    flat = W.reshape(-1)
    scales = _fit_stage_scales(flat[:flat.numel() - flat.numel() % 8].reshape(-1, 8)
                               if flat.numel() % 8 else flat.reshape(-1, 8),
                               codebook, max(1, stages))

    # damped inverse-Hessian Cholesky factor (upper), exactly as GPTQ
    dead = torch.diag(Hc) == 0
    Hc[dead, dead] = 1.0
    W[:, dead] = 0.0
    damp = percdamp * torch.mean(torch.diag(Hc)).clamp_min(1e-12)
    idx = torch.arange(cols, device=dev)
    Hc[idx, idx] += damp
    L = torch.linalg.cholesky(Hc)
    Hinv = torch.cholesky_inverse(L)
    Hinv = torch.linalg.cholesky(Hinv, upper=True)

    Q = torch.zeros_like(W)
    for i in range(cols):
        w = W[:, i]
        d = Hinv[i, i]
        q = _quantize_column_e8p(w, codebook, scales)
        Q[:, i] = q
        err = (w - q) / d
        W[:, i:] -= err.unsqueeze(1) @ Hinv[i, i:].unsqueeze(0)

    if row_scale is not None:
        Q = Q * row_scale
    W_hat = (R_out @ Q @ R_in.t()) if use_rht else Q
    w_relerr = (W_hat - Wc).norm().item() / Wc.norm().clamp_min(1e-12).item()
    bpw = max(1, stages) * math.log2(codebook.shape[0]) / 8.0
    return W_hat.to(W2d.dtype), bpw, w_relerr


def _e8_empirical_bpw(blocks: torch.Tensor) -> float:
    q = nearest_e8(blocks)
    # hash rows to count distinct
    key = q.mul(2).round().to(torch.int64)             # E8 points are on 1/2 grid
    base = key.new_tensor([1, 7, 49, 343, 2401, 16807, 117649, 823543])
    h = (key * base).sum(dim=1)
    n_distinct = torch.unique(h).numel()
    return math.log2(max(n_distinct, 1)) / 8.0


# ===================================================================
#  Model-level quantization
# ===================================================================
_QUANTIZABLE = (nn.Linear, nn.Conv2d)


@dataclass
class QuipConfig:
    method: str = "e8p"           # 'e8p' | 'e8lattice' | 'rvq'
    n_bits: int = 16              # codebook size = 2**n_bits (16 -> 2 bpw)
    stages: int = 1              # residual stages (bpw = stages * 2 for e8p)
    use_rht: bool = True
    use_ldlq: bool = False        # QuIP# BlockLDLQ (Hessian error feedback)
    percdamp: float = 0.01        # Hessian damping for LDLQ
    lattice_rms: float = 1.0      # e8lattice rate knob (higher -> more bits)
    min_numel: int = 8192         # leave tiny layers in fp
    skip_first_last: bool = True
    rht_seed: int = 0


@dataclass
class QuipReport:
    cfg: QuipConfig
    layers: list = field(default_factory=list)
    n_quantized: int = 0
    n_skipped: int = 0
    total_quantized_params: int = 0
    total_params: int = 0

    @property
    def index_bpw(self) -> float:
        if not self.layers:
            return float("nan")
        tot = sum(l.numel for l in self.layers)
        return sum(l.bpw * l.numel for l in self.layers) / max(tot, 1)

    @property
    def w_relerr(self) -> float:
        if not self.layers:
            return float("nan")
        # aggregate relative L2 error, weighted by number of weights
        num = sum((l.w_relerr ** 2) * l.numel for l in self.layers)
        den = sum(l.numel for l in self.layers)
        return math.sqrt(num / max(den, 1))


@torch.no_grad()
def quantize_model_quip(model: nn.Module, cfg: QuipConfig,
                        calib_inputs=None, verbose: bool = True) -> QuipReport:
    """In-place fake-quantize every Conv2d/Linear weight of ``model``.

    If ``cfg.use_ldlq`` is set, ``calib_inputs`` (an iterable of input batches)
    must be provided so per-layer Hessians can be estimated for BlockLDLQ.
    """
    codebook = None
    if cfg.method in ("e8p", "rvq"):
        codebook = build_e8p_codebook(cfg.n_bits)

    layers = [(n, m) for n, m in model.named_modules() if isinstance(m, _QUANTIZABLE)]
    skip = set()
    if cfg.skip_first_last and layers:
        skip.add(layers[0][0]); skip.add(layers[-1][0])

    hessians = {}
    if cfg.use_ldlq:
        if calib_inputs is None:
            raise ValueError("use_ldlq=True requires calib_inputs for the Hessian")
        if cfg.method == "e8lattice":
            raise ValueError("LDLQ requires a finite codebook (method e8p/rvq)")
        dev = next(model.parameters()).device
        hessians = compute_layer_hessians(
            model, calib_inputs, device=dev, min_numel=cfg.min_numel,
            skip_first_last=cfg.skip_first_last, verbose=verbose)

    report = QuipReport(cfg=cfg)
    report.total_params = sum(m.weight.numel() for _, m in layers)
    for name, m in layers:
        W = m.weight.data
        if name in skip or W.numel() < cfg.min_numel:
            report.n_skipped += 1
            continue
        codebook_dev = codebook.to(W.device) if codebook is not None else None
        W2d = W.reshape(W.shape[0], -1)
        if cfg.use_ldlq and hessians.get(name) is not None:
            W_hat, bpw, relerr = blockldlq_quantize(
                W2d, hessians[name].to(W.device), codebook_dev,
                use_rht=cfg.use_rht, stages=cfg.stages, rht_seed=cfg.rht_seed,
                percdamp=cfg.percdamp)
        else:
            W_hat, bpw, relerr = quantize_weight_tensor(
                W2d, method=cfg.method, codebook=codebook_dev,
                use_rht=cfg.use_rht, stages=cfg.stages, lattice_rms=cfg.lattice_rms,
                rht_seed=cfg.rht_seed)
        m.weight.data.copy_(W_hat.reshape(W.shape))
        report.layers.append(LayerStat(name, tuple(W.shape), W.numel(), bpw, relerr))
        report.n_quantized += 1
        report.total_quantized_params += W.numel()
        if verbose:
            print(f"    [quip] {name:<40s} {str(tuple(W.shape)):<20s} "
                  f"bpw={bpw:5.3f}  relerr={relerr:6.4f}")
    if verbose:
        print(f"  [quip] quantized {report.n_quantized} layers "
              f"({report.total_quantized_params:,} weights, "
              f"{100*report.total_quantized_params/max(report.total_params,1):.1f}%), "
              f"skipped {report.n_skipped}.  "
              f"index bpw={report.index_bpw:.3f}  agg w_relerr={report.w_relerr:.4f}")
    return report


# ===================================================================
#  Self-test / demo (no external data needed)
# ===================================================================
def _self_test():
    torch.manual_seed(0)
    print("== E8 decoder sanity ==")
    x = torch.randn(1000, 8)
    q = nearest_e8(x)
    # every E8 point: all-integer or all-half-integer, even sum
    twice = (q * 2).round()
    assert torch.all(twice == q * 2), "E8 points must lie on the 1/2 grid"
    is_int = torch.all((q - q.round()).abs() < 1e-6, dim=1)
    is_half = torch.all(((q - 0.5) - (q - 0.5).round()).abs() < 1e-6, dim=1)
    assert torch.all(is_int | is_half), "each E8 point is all-int or all-half"
    even = (q.sum(dim=1).round().long() % 2 == 0)
    assert torch.all(even), "E8 points have even coordinate sum"
    print("  ok: decoded points are valid E8 lattice points")

    print("== E8P codebook ==")
    cb = build_e8p_codebook(16)
    print(f"  codebook: {tuple(cb.shape)}  rms={cb.pow(2).mean().sqrt():.4f}")

    print("== weight quantization ==")
    W = torch.randn(256, 512)
    for method, kw in [("e8lattice", {}), ("e8p", {}), ("rvq", dict(stages=2))]:
        c = None if method == "e8lattice" else cb
        Wh, bpw, err = quantize_weight_tensor(W, method=method, codebook=c,
                                              stages=kw.get("stages", 1))
        print(f"  {method:<10s} bpw={bpw:5.3f}  relerr={err:.4f}")

    print("== BlockLDLQ (Hessian feedback) vs plain, on one layer ==")
    torch.manual_seed(1)
    Wt = torch.randn(64, 128)
    X = torch.randn(128, 2048)                          # fake calibration inputs
    Hh = (X @ X.t()) / X.shape[1]
    _, _, e_plain = quantize_weight_tensor(Wt, method="e8p", codebook=cb, use_rht=True)
    _, _, e_ldlq = blockldlq_quantize(Wt, Hh, cb, use_rht=True, stages=1)
    # error measured in the H-metric (what LDLQ optimizes) should improve
    def herr(Wh):
        D = Wh - Wt
        return torch.sqrt(torch.trace(D @ Hh @ D.t())).item()
    Wp, _, _ = quantize_weight_tensor(Wt, method="e8p", codebook=cb, use_rht=True)
    Wl, _, _ = blockldlq_quantize(Wt, Hh, cb, use_rht=True, stages=1)
    print(f"  plain relerr={e_plain:.4f} H-err={herr(Wp):.3f} | "
          f"ldlq relerr={e_ldlq:.4f} H-err={herr(Wl):.3f}")

    print("== model quantization (toy) ==")
    net = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
                        nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.Linear(128, 10))
    quantize_model_quip(net, QuipConfig(method="e8p", min_numel=1024))


if __name__ == "__main__":
    _self_test()
