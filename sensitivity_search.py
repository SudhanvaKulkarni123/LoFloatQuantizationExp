import torch
import torch.nn as nn
import LoFloat as lof
from LoFloat import LoF_Linear, LoF_Conv2d
import copy
import math
import sensitivities
import gptq


def bisection_sensitivity(model, sensitivity_measure, data, loss_fn, eval_fn,
                          accuracy_target, bs=[8, 6, 4], n_samples=256, use_gptq=True, device='cuda'):

    lof_model = lof.lofloatify(model)
    lof_model.to(device)
    weights_minmax, activ_minmax, bias_minmax = sensitivities.find_range(lof_model, data, n_samples, device)
    weights_exp, weights_bias, activ_exp, activ_bias, bias_exp, bias_bias = sensitivities.find_exp_bits_and_bias(weights_minmax, activ_minmax, bias_minmax)
    lof.set_exponent_fields(model=lof_model, activation_exp_bits=activ_exp, weight_exp_bits=weights_exp, bias_exp_bits=bias_exp)
    #lof.set_exponentbias_fields(model=lof_model, activation_expbias=activ_bias, weight_expbias=weights_bias, bias_expbias=bias_bias)

    # exponent search done, now search over mantissa
    if sensitivity_measure == "hessian":
        weight_sens, activ_sens, bias_sens = sensitivities.hess_sensitivity(lof_model, data, n_samples, device)
    else:
        weight_sens, activ_sens, bias_sens = sensitivities.noise_sensitivity_full(lof_model, data, loss_fn, n_samples, device)

    # Sort layers by weight sensitivity ascending (least sensitive first)
    ll = [k for k, v in sorted(weight_sens.items(), key=lambda item: next(iter(item[1].values())))]

    # Initialize working config w with mantissa=11 for all layers
    w_weights = {layer: 11 for layer in ll}
    w_activ   = {layer: 11 for layer in ll}
    w_bias    = {layer: 11 for layer in ll}

    for b in bs:
        thr  = len(ll) // 2
        upl  = len(ll)
        lowl = 0

        count = 0
        prev_thr = None
        while thr != prev_thr:
            count = count + 1
            prev_thr = thr

            # Local working config: copy w, then assign b to first thr layers
            lw_weights = dict(w_weights)
            lw_activ   = dict(w_activ)
            lw_bias    = dict(w_bias)

            for layer in ll[:thr]:
                lw_weights[layer] = b
                lw_activ[layer]   = b
                lw_bias[layer]    = b

            # Apply local config and evaluate accuracy
            # lof.set_mantissa_fields(model=lof_model,
            #                         activation_mantissa_bits=lw_activ,
            #                         weight_mantissa_bits=lw_weights,
            #                         bias_mantissa_bits=lw_bias)
            lof.set_mantissa_fields(model=lof_model,
                                    activation_mantissa_bits=lw_activ,
                                    weight_mantissa_bits=lw_weights,
                                    bias_mantissa_bits=lw_bias)
            a = eval_fn(lof_model, data)

            if abs(a) <= accuracy_target:
                lowl = thr
                thr  = thr + (upl - thr) // 2   # push threshold up
            else:
                upl  = thr
                thr  = thr - (thr - lowl) // 2  # pull threshold down

        # Commit the converged threshold to the working config
        for layer in ll[:thr]:
            w_weights[layer] = b
            w_activ[layer]   = b
            w_bias[layer]    = b

        # Shrink ll: only layers already quantized are candidates for next round
        ll = ll[:thr]

    # Apply the final optimal configuration to the model
    sensitivities.quantize_weights_with_gptq(model=lof_model, dataset=data, mantissa_bits=w_weights, exponent_bits=weights_exp, n_samples=n_samples, device=device)
    lof.set_mantissa_fields(model=lof_model,
                            activation_mantissa_bits=w_activ,
                            weight_mantissa_bits=w_weights,
                            bias_mantissa_bits=w_bias)
    lof.set_exponent_fields(model=lof_model, activation_exp_bits=activ_exp, weight_exp_bits=weights_exp, bias_exp_bits=bias_exp)

    return lof_model

def greedy_sensitivity(model, sensitivity_measure, data, loss_fn, eval_fn,
                       accuracy_target, bs=[8, 6, 4], n_samples=128, device='cuda'):
    lof_model = lof.lofloatify(model)
    lof_model.to(device)

    weights_minmax, activ_minmax, bias_minmax = sensitivities.find_range(lof_model, data, n_samples, device)
    weights_exp, weights_bias, activ_exp, activ_bias, bias_exp, bias_bias = sensitivities.find_exp_bits_and_bias(weights_minmax, activ_minmax, bias_minmax)
    lof.set_exponent_fields(model=lof_model, activation_exp_bits=activ_exp, weight_exp_bits=weights_exp, bias_exp_bits=bias_exp)

    # Sensitivity analysis
    if sensitivity_measure == "hessian":
        weight_sens, activ_sens, bias_sens = sensitivities.hess_sensitivity(lof_model, data, n_samples, device)
    else:
        weight_sens, activ_sens, bias_sens = sensitivities.noise_sensitivity_full(lof_model, data, loss_fn, n_samples, device)

    # Sort layers by weight sensitivity ascending (least sensitive = quantize first)
    ll = [k for k, v in sorted(weight_sens.items(), key=lambda item: next(iter(item[1].values())))]

    # Initialize working config w with max(bs) mantissa bits
    max_b = max(bs)
    w_weights = {layer: max_b for layer in ll}
    w_activ   = {layer: max_b for layer in ll}
    w_bias    = {layer: max_b for layer in ll}

    # Sort bit widths descending (try most aggressive quantization first)
    bs = sorted(bs, reverse=True)

    for b in bs:
        ql = []  # layers that survived this round

        for layer in ll:
            # Try setting this layer to bit width b
            prev_w = w_weights[layer]
            prev_a = w_activ[layer]
            prev_b = w_bias[layer]

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
                w_bias[layer]    = prev_b

        # Only layers that accepted this bit width are candidates for the next
        ll = ql

    # Apply final config
    sensitivities.quantize_weights_with_gptq(model=lof_model, dataset=data, mantissa_bits=w_weights, exponent_bits=weights_exp, n_samples=128, device=device)
    lof.set_mantissa_fields(model=lof_model,
                            activation_mantissa_bits=w_activ,
                            weight_mantissa_bits=w_weights,
                            bias_mantissa_bits=w_bias)
    lof.set_exponent_fields(model=lof_model, activation_exp_bits=activ_exp, weight_exp_bits=weights_exp, bias_exp_bits=bias_exp)

    print("Greedy search complete. Final configuration applied.")

    return lof_model