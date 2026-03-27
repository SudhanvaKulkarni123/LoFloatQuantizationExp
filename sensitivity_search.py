import torch
import torch.nn as nn
import LoFloat as lof
from LoFloat import LoF_Linear, LoF_Conv2d
import copy
import math
import sensitivities
import gptq


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
    lof_model = sensitivities.quantize_weights_with_gptq(
        model=lof_model, dataset=data, mantissa_bits=w_weights,
        exponent_bits=weights_exp, n_samples=128, device=device, collate_fn=collate_fn
    )
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
                       accuracy_target, bs=[8, 6, 4], es=[8,6,4], n_samples=128, device='cuda', collate_fn=None, baseline=0.0):
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
    first_layer_module = dict(lof_model.named_modules())[first_layer]
    w = first_layer_module.weight.data 

    # Initialize working config w with max(bs) mantissa bits
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



    for b in bs:
        ql = []  # layers that survived this round

        for layer in ll:
            # Skip first/last — they stay at max_b
            if layer in skip_layers:
                continue

            # Try setting this layer to bit width b
            prev_w = w_weights[layer]
            prev_a = w_activ[layer]
            prev_b_val = w_bias[layer]

            w_weights[layer] = b
            w_activ[layer]   = b
            w_bias[layer]    = b
            lof.set_mantissa_fields(model=lof_model,
                                    activation_mantissa_bits=w_activ,
                                    weight_mantissa_bits=w_weights,
                                    bias_mantissa_bits=w_bias)

            a = abs(eval_fn(lof_model, data))


            if a <= accuracy_target:
                ql.append(layer)
            else:
                # Revert
                w_weights[layer] = prev_w
                w_activ[layer]   = prev_a
                w_bias[layer]    = prev_b_val

        # Only layers that accepted this bit width are candidates for the next
        ll = ql
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
        ql = []  # layers that survived this round
        for layer in ll:
            # Skip first/last — they stay at max_b
            if layer in skip_layers:
                continue

            # Try setting this layer to bit width b
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

            weights_exp[layer] = min(b, prev_w)
            activ_exp[layer]   = min(b, prev_a)
            bias_exp[layer]    = min(b, prev_b_val)

            lof.set_exponent_fields(model=lof_model,
                                    activation_exp_bits=activ_exp,
                                    weight_exp_bits=weights_exp,
                                    bias_exp_bits=bias_exp)
            a = abs(eval_fn(lof_model, data))

            if a <= accuracy_target:
                ql.append(layer)
            else:
                # Revert
                weights_exp[layer] = prev_w
                activ_exp[layer]   = prev_a
                bias_exp[layer]    = prev_b_val

        # Only layers that accepted this bit width are candidates for the next
        ll = ql


    # Apply final config
    lof_model = lof.lofloatify(model)
  
    print("running gptq with exp_bits = %d and mantissa_bits = %d" % (list(weights_exp.values())[9], list(w_weights.values())[9]))
    print(f"First/last layers kept at {max_b} mantissa bits")
    lof_model = sensitivities.quantize_weights_with_gptq(
        model=lof_model, dataset=data, mantissa_bits=w_weights,
        exponent_bits=weights_exp, n_samples=128, device=device, collate_fn=collate_fn
    )
    lof.set_mantissa_fields(
        model=lof_model,
        activation_mantissa_bits=w_activ,
        weight_mantissa_bits=w_weights,
        bias_mantissa_bits=w_bias,
    )
    for name, module in lof_model.named_modules():
        if isinstance(module, (lof.LoF_Linear, lof.LoF_Conv2d)):
            print(f"{name}: mantissa={module.weight_params.mantissa_bits}, in dict={w_weights.get(name, 'MISSING')}")
            break
    lof.set_exponent_fields(
        model=lof_model,
        activation_exp_bits=activ_exp,
        weight_exp_bits=weights_exp,
        bias_exp_bits=bias_exp,
    )
    lof_model.to(device)

    # --- RMS difference between original and quantized weight matrices ---
    print("\n=== Per-Layer RMS Weight Difference ===")
    total_rms_sum = 0.0
    total_params = 0

    return lof_model



    




   