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
                       hadamard=True, strassen_max_hooks=12):
    """
    Search order: exp → accum → slack → mant → accum(2nd).

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

    with torch.no_grad():
        weights_minmax, activ_minmax, bias_minmax = sensitivities.find_range(
            lof_model, data, n_samples, device, collate_fn=collate_fn
        )
        weights_exp, weights_bias, activ_exp, activ_bias, bias_exp, bias_bias = \
            sensitivities.find_exp_bits_and_bias(weights_minmax, activ_minmax, bias_minmax)

    # if sensitivity_measure == "hessian":
    #     weight_sens, activ_sens, bias_sens, accum_sens = sensitivities.hess_sensitivity(
    #         lof_model, data, n_samples, device, collate_fn=collate_fn
    #     )
    # else:
    #     weight_sens, activ_sens, bias_sens, accum_sens = sensitivities.noise_sensitivity_full(
    #         lof_model, data, loss_fn, n_samples, device, collate_fn=collate_fn
    #     )

    # Initialize all sensitivities to 0 (skips actual hessian/noise computation)
    weight_sens = {}
    activ_sens = {}
    bias_sens = {}
    accum_sens = {}
    for name, module in lof_model.named_modules():
        if isinstance(module, (LoF_Conv2d, LoF_Linear)):
            weight_sens[name] = {'weight': 0.0}
            activ_sens[name] = 0.0
            bias_sens[name] = {'bias': 0.0}
            accum_sens[name] = 0.0

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

    es = sorted(es, reverse=True)
    for b in es:
        ql = []

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

    # ======================GPTQ pruning======================
    # lof_model = sensitivities.quantize_weights_with_gptq(
    #     model=lof_model, dataset=data, mantissa_bits=w_weights,
    #     exponent_bits=weights_exp, n_samples=128, device=device, micro_batch_size=4, collate_fn=collate_fn
    # )

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

    # ==================== APPLY FINAL CONFIG ====================
    lof_model = lof.lofloatify(model, skip_layer_names=skip_layer_names)
    lof_model = lof_model.to(device)

    if hadamard:
        apply_hadamard_to_weights(lof_model, skip_first=False, skip_last=False)

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

