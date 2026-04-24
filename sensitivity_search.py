import torch
import torch.nn as nn
import LoFloat as lof
from LoFloat import LoF_Linear, LoF_Conv2d, L1BatchNorm, LinfBatchNorm
import copy
import math
import sensitivities
import gptq




def replace_batchnorm2d(model, p, L1_scales, Linf_scales):
    """
    p == 1.0      -> L1BatchNorm   (scale = per-channel L1_scale)
    p == math.inf -> LinfBatchNorm (scale = per-channel Linf_scale)
    p == 2        -> no-op.

    For both: running_{mad,maxdev} = scale * sqrt(running_var + eps) from the
    original BN, so pre-affine output matches BN2d's (x-mean)/std exactly and
    weight/bias transfer without correction.
    """
    if p == 2 or p == 2.0:
        return model

    if p == 1.0:
        scales_dict, NewCls, stat_name = L1_scales, L1BatchNorm, "running_mad"
    elif p == math.inf:
        scales_dict, NewCls, stat_name = Linf_scales, LinfBatchNorm, "running_maxdev"
    else:
        raise ValueError(f"p must be 1.0, 2, or math.inf; got {p!r}")

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if not isinstance(child, nn.BatchNorm2d):
                continue

            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if full_name not in scales_dict:
                print(f"  [SKIP] {full_name} — no calibration scale")
                continue

            device  = child.running_mean.device
            dtype   = child.running_mean.dtype
            scale_c = scales_dict[full_name].to(device=device, dtype=dtype)

            new = NewCls(
                num_features=child.num_features,
                eps=child.eps,
                momentum=child.momentum,
                affine=child.affine,
                scale=scale_c,
            ).to(device=device, dtype=dtype)

            running_std = (child.running_var.detach() + child.eps).sqrt()
            getattr(new, stat_name).copy_(scale_c * running_std)
            new.running_mean.copy_(child.running_mean.detach())
            if child.affine:
                new.weight.data.copy_(child.weight.detach())
                new.bias.data.copy_(child.bias.detach())

            setattr(parent, child_name, new)

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



def greedy_sensitivity(model, sensitivity_measure, data, loss_fn, eval_fn,
                       accuracy_target, bs=[8, 6, 4], es=[8, 6, 4], accum_bw=[10, 8, 6, 4],
                       n_samples=128, device='cuda', collate_fn=None, baseline=0.0, batch_size=5):
    """
    batch_size: number of layers to try flipping together per eval call.
        batch_size=1 -> one-layer-at-a-time. Accumulation pass uses batch_size*4
        for its least-sensitive tier.
    accum_bw: candidate accumulation mantissa bit widths. First and last LoF
        layers are kept at the initial (max) value.

    Search strategy (all three passes):
      - Sensitivity-sorted layer list is split into sensitivity tiers; the
        most sensitive tier uses small batches, least sensitive uses large.
      - Bit-width candidates are searched via bisection per batch: O(log k)
        evals instead of O(k).
      - Widths are walked ascending (most aggressive first), so a layer that
        tolerates the narrowest width is decided in a single eval.
    """
    lof_model = lof.lofloatify(model)
    lof_model.to(device)

    with torch.no_grad():
        weights_minmax, activ_minmax, bias_minmax = sensitivities.find_range(
            lof_model, data, n_samples, device, collate_fn=collate_fn
        )
        weights_exp, weights_bias, activ_exp, activ_bias, bias_exp, bias_bias = \
            sensitivities.find_exp_bits_and_bias(weights_minmax, activ_minmax, bias_minmax)

    if sensitivity_measure == "hessian":
        weight_sens, activ_sens, bias_sens, accum_sens = sensitivities.hess_sensitivity(
            lof_model, data, n_samples, device, collate_fn=collate_fn
        )
    else:
        weight_sens, activ_sens, bias_sens, accum_sens = sensitivities.noise_sensitivity_full(
            lof_model, data, loss_fn, n_samples, device, collate_fn=collate_fn
        )

    def get_sens_val(sens_dict, key):
        v = sens_dict.get(key, 0)
        if isinstance(v, dict):
            return next(iter(v.values()), 0)
        return v if isinstance(v, (int, float)) else 0

    def max_sens(k):
        return max(
            get_sens_val(weight_sens, k),
            get_sens_val(activ_sens, k),
            get_sens_val(bias_sens, k),
        )

    all_lof_layers = [n for n, m in lof_model.named_modules()
                      if isinstance(m, (LoF_Linear, LoF_Conv2d))]
    first_layer = all_lof_layers[0]
    last_layer = all_lof_layers[-1]
    skip_layers = {first_layer, last_layer}

    max_b = max(bs)
    w_weights = {layer: max_b for layer in list(weight_sens.keys())}
    w_activ   = {layer: max_b for layer in list(weight_sens.keys())}
    w_bias    = {layer: max_b for layer in list(weight_sens.keys())}

    max_accum_b = max(accum_bw)
    accum_precs = {layer: max_accum_b for layer in list(weight_sens.keys())}
    # lof.set_accumulation_precisions(lof_model, accum_precs)

    for name, module in lof_model.named_modules():
        if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
            continue
        if name not in w_weights:   w_weights[name]   = max_b
        if name not in w_activ:     w_activ[name]     = max_b
        if name not in w_bias:      w_bias[name]      = max_b
        if name not in accum_precs: accum_precs[name] = max_accum_b

    def make_batches(layer_list, bsz):
        if bsz <= 1:
            return [[layer] for layer in layer_list]
        return [layer_list[i:i + bsz] for i in range(0, len(layer_list), bsz)]

    def tiered_batches(layer_list, sens_fn, base_bsz):
        """Split layers into three sensitivity tiers and batch each tier with
        a different size: most sensitive -> small batches, least -> large."""
        ranked = sorted(layer_list, key=sens_fn, reverse=True)
        n = len(ranked)
        if n == 0:
            return []
        top = ranked[:n // 3]
        mid = ranked[n // 3 : 2 * n // 3]
        bot = ranked[2 * n // 3:]
        return (make_batches(top, max(1, base_bsz))
              + make_batches(mid, base_bsz * 2)
              + make_batches(bot, base_bsz * 4))

    def bisect_batch(active_layers, widths_asc, apply_fn, current_getter):
        """Binary-search the narrowest width in widths_asc that keeps accuracy.
        apply_fn(layers, b): mutate working dicts + call the corresponding
            lof.set_* function for that pass.
        current_getter(layer): snapshot the layer's current config for revert.
        Returns the accepted width, or None if even the widest failed."""
        snapshot = {l: current_getter(l) for l in active_layers}
        lo, hi = 0, len(widths_asc) - 1
        best = None
        while lo <= hi:
            mid = (lo + hi) // 2
            b = widths_asc[mid]
            apply_fn(active_layers, b)
            if abs(eval_fn(lof_model, data)) <= accuracy_target:
                best = b
                hi = mid - 1
            else:
                lo = mid + 1
        if best is None:
            apply_fn(active_layers, None, revert=snapshot)
        else:
            apply_fn(active_layers, best)
        return best

    # ==================== MANTISSA SEARCH ====================
    bs_asc = sorted(bs)

    def apply_mantissa(layers, b, revert=None):
        if revert is not None:
            for l in layers:
                w_weights[l], w_activ[l], w_bias[l] = revert[l]
        else:
            for l in layers:
                w_weights[l] = b
                w_activ[l]   = b
                w_bias[l]    = b
        lof.set_mantissa_fields(model=lof_model,
                                activation_mantissa_bits=w_activ,
                                weight_mantissa_bits=w_weights,
                                bias_mantissa_bits=w_bias)

    def mantissa_snapshot(l):
        return (w_weights[l], w_activ[l], w_bias[l])

    ll = [k for k in weight_sens.keys()]
    for name, module in lof_model.named_modules():
        if isinstance(module, (LoF_Linear, LoF_Conv2d)) and name not in ll and name not in skip_layers:
            ll.append(name)

    for batch in tiered_batches(ll, max_sens, batch_size):
        active = [l for l in batch if l not in skip_layers]
        if not active:
            continue
        bisect_batch(active, bs_asc, apply_mantissa, mantissa_snapshot)

    # ==================== EXPONENT SEARCH ====================
    for name in bias_exp:
        bias_exp[name]    = max(bias_exp[name], 1)
        weights_exp[name] = max(weights_exp[name], 1)
        activ_exp[name]   = max(activ_exp[name], 1)

    max_e = max(es)
    es_asc = sorted(es)

    for name, module in lof_model.named_modules():
        if not isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            continue
        if name not in weights_exp: weights_exp[name] = max_e
        if name not in activ_exp:   activ_exp[name]   = max_e
        if name not in bias_exp:    bias_exp[name]    = max_e

    def apply_exponent(layers, b, revert=None):
        if revert is not None:
            for l in layers:
                weights_exp[l], activ_exp[l], bias_exp[l] = revert[l]
        else:
            for l in layers:
                # Respect any existing narrower setting from prior iteration.
                weights_exp[l] = min(b, weights_exp[l])
                activ_exp[l]   = min(b, activ_exp[l])
                bias_exp[l]    = min(b, bias_exp[l])
        lof.set_exponent_fields(model=lof_model,
                                activation_exp_bits=activ_exp,
                                weight_exp_bits=weights_exp,
                                bias_exp_bits=bias_exp)

    def exponent_snapshot(l):
        return (weights_exp[l], activ_exp[l], bias_exp[l])

    ll = [k for k in weight_sens.keys()]
    for name, module in lof_model.named_modules():
        if isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)) and name not in ll and name not in skip_layers:
            ll.append(name)

    for batch in tiered_batches(ll, max_sens, batch_size):
        active = [l for l in batch if l not in skip_layers]
        if not active:
            continue
        bisect_batch(active, es_asc, apply_exponent, exponent_snapshot)

    # ==================== ACCUMULATION SEARCH ====================
    accum_bw_asc = sorted(accum_bw)

    def apply_accum(layers, b, revert=None):
        if revert is not None:
            for l in layers:
                accum_precs[l] = revert[l]
        else:
            for l in layers:
                accum_precs[l] = b
        lof.set_accumulation_precisions(lof_model, accum_precs)

    def accum_snapshot(l):
        return accum_precs[l]

    def accum_sens_fn(l):
        return get_sens_val(accum_sens, l)

    ll = list(accum_sens.keys())
    for name, module in lof_model.named_modules():
        if isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)) and name not in ll and name not in skip_layers:
            ll.append(name)

    for batch in tiered_batches(ll, accum_sens_fn, batch_size):
        active = [l for l in batch if l not in skip_layers]
        if not active:
            continue
        bisect_batch(active, accum_bw_asc, apply_accum, accum_snapshot)

    # ==================== APPLY FINAL CONFIG ====================
    lof_model = lof.lofloatify(model)

    print("running gptq with exp_bits = %d and mantissa_bits = %d" %
          (list(weights_exp.values())[9], list(w_weights.values())[9]))
    print(f"First/last layers kept at {max_b} mantissa bits, {max_accum_b} accum bits")

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
    print("accum precs dict")
    print(accum_precs)
    lof.set_accumulation_precisions(lof_model, accum_precs)
    lof_model.to(device)

    #check if its okay to replace 2 norm with 1 norm or inf norm
    L1_s, Linf_s = sensitivities.find_batchnorm_scales(lof_model, data, n_samples, device, collate_fn=collate_fn)
    for p, tag in [(1.0, "L1"), (math.inf, "Linf")]:
        cand = replace_batchnorm2d(copy.deepcopy(lof_model), p=p, L1_scales=L1_s, Linf_scales=Linf_s)
        ok = abs(eval_fn(cand, data)) <= accuracy_target
        print(f"{tag} BN replacement: {'OK' if ok else 'FAILED'}")
        if ok: return cand
    return lof_model 


    print("\n=== Per-Layer RMS Weight Difference ===")
    return lof_model