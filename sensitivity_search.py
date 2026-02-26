import torch
import torch.nn as nn
import LoFloat as lof
from LoFloat import LoF_Linear, LoF_Conv2d
import copy
import math
import sensitivities


def bisetion_sensitivity(model, sensitivity_measure, data, n_samples=256, device='cpu'):

    lof_model = lof.lofloatify(model)
    weights_minmax, activ_minmax, bias_minmax = sensitivities.find_range(lof_model, data, n_samples, device)
    weights_exp, weights_bias, activ_exp, activ_bias, bias_exp, bias_bias = sensitivities.find_exp_bits_and_bias(weights_minmax, activ_minmax)
    lof.set_exponent_fields(model=lof_model, activation_exp_bits=activ_exp, weight_exp_bits=weights_exp, bias_exp_bits=bias_exp)
    lof.set_bias_fields(model=lof_model, activation_bias=activ_bias, weight_bias=weights_bias, bias_bias=bias_bias)

    #exponent search done, now search over mantissa
    if sensitivity_measure == "hessian":
        

    sorted_data_asc = dict(sorted(data.items(), key=lambda item: item[1]))




    return


def greedy_sensitivity():
    return