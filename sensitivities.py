import torch
import torch.nn as nn
import torch.nn.functional as F
import LoFloat as lof
from LoFloat import LoF_Linear, LoF_Conv2d
import copy
import math



def make_calib_data(dataset, n_samples=256, collate_fn=None):
    indices = torch.randperm(len(dataset))[:n_samples].tolist()
    calib_subset = torch.utils.data.Subset(dataset, indices)
    calib_loader = torch.utils.data.DataLoader(
        calib_subset, batch_size=n_samples, shuffle=False,
        collate_fn=collate_fn
    )
    calib_data, _ = next(iter(calib_loader))
    
    return calib_data

def batch_hutchinson_approx(model, loss, param, n_samples, retain_graph=True, batch_size=1):
    param_shape = param.shape
    param_numel = param.numel()

    # First pass for Jacobian
    grad, = torch.autograd.grad(
        loss, param,
        create_graph=True,
        retain_graph=True
    )

    trace_accumulator = 0.0

    for i in range(n_samples):
        # Generate ONE random vector at a time instead of all n_samples at once
        v = (torch.randint(0, 2, (param_numel,),
                           device=param.device).float() * 2 - 1)

        is_last = (i == n_samples - 1)
        keep = not is_last or retain_graph

        Hv, = torch.autograd.grad(
            grad, param,
            grad_outputs=v.reshape(param_shape),
            retain_graph=keep
        )

        trace_accumulator += (v * Hv.reshape(-1).detach()).sum().item()

        del v, Hv
        torch.cuda.empty_cache()

    del grad
    torch.cuda.empty_cache()

    return trace_accumulator / n_samples

def find_range(model, dataset, n_samples=256, device='cuda',
               collate_fn=None, chunk_size=32):
    model.eval()

    calib_data_cpu = make_calib_data(
        dataset=dataset, n_samples=n_samples, collate_fn=collate_fn
    )

    act_max = {}
    act_min = {}  # smallest nonzero |x|

    def make_hook(layer_name):
        def hook(module, inp, out):
            x = inp[0].detach().abs()
            cur_max = x.max().item()
            # cheap nonzero-min without materializing a filtered copy:
            # replace zeros with +inf, take min
            x_nz = torch.where(x > 0, x, torch.full_like(x, float('inf')))
            cur_min = x_nz.min().item()
            if cur_min == float('inf'):
                cur_min = 0.0

            if layer_name not in act_max:
                act_max[layer_name] = cur_max
                act_min[layer_name] = cur_min if cur_min > 0 else float('inf')
            else:
                act_max[layer_name] = max(act_max[layer_name], cur_max)
                if cur_min > 0:
                    act_min[layer_name] = min(act_min[layer_name], cur_min)
        return hook

    hooks = [
        module.register_forward_hook(make_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, (LoF_Linear, LoF_Conv2d))
    ]

    with torch.no_grad():
        for i in range(0, n_samples, chunk_size):
            chunk = calib_data_cpu[i:i+chunk_size].to(device, non_blocking=True)
            model(chunk)
            del chunk
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    # Weights + biases are tiny relative to activations — do them plainly.
    def abs_range(tensor):
        t = tensor.detach().abs()
        mx = t.max().item()
        t_nz = torch.where(t > 0, t, torch.full_like(t, float('inf')))
        mn = t_nz.min().item()
        return {"min": (mn if mn != float('inf') else 0.0), "max": mx}

    weights_minmax, activations_minmax, bias_minmax = {}, {}, {}
    for name, module in model.named_modules():
        if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
            continue
        if name not in act_max:
            print(f"  [SKIP] {name} — forward() never called")
            continue
        activations_minmax[name] = {
            "min": act_min[name] if act_min[name] != float('inf') else 0.0,
            "max": act_max[name],
        }
        weights_minmax[name] = abs_range(module.weight)
        bias_minmax[name] = (abs_range(module.bias) if module.bias is not None
                             else {"min": 0.0, "max": 0.0})

    return weights_minmax, activations_minmax, bias_minmax

def find_exp_bits_and_bias(weights_minmax, activations_minmax, bias_minmax):

    weights_exp_bits = {}
    weights_bias = {}
    activations_exp_bits = {}
    activations_bias = {}
    bias_exp_bits = {}
    bias_bias = {}

    for name, val_range in weights_minmax.items():
        _, min_exp = math.frexp(val_range["min"])
        _, max_exp = math.frexp(val_range["max"])
        exp_range = (abs(max_exp - min_exp))  #need abs if min is 0 since frexp(0) = 0
        weights_exp_bits[name] = max(math.ceil(math.log2(exp_range)), 0) if exp_range > 0 else 0
        weights_bias[name] = 0 #dont set bias for now 

    for name, val_range in activations_minmax.items():
        _, min_exp = math.frexp(val_range["min"])
        _, max_exp = math.frexp(val_range["max"])
        exp_range = (abs(max_exp - min_exp))
        activations_exp_bits[name] = max(math.ceil(math.log2(exp_range)), 0) if exp_range > 0 else 0
        activations_bias[name] = 0

    for name, val_range in bias_minmax.items():
        _, min_exp = math.frexp(val_range["min"])
        _, max_exp = math.frexp(val_range["max"])
        exp_range = ((max_exp - min_exp))
        bias_exp_bits[name] = max(math.ceil(math.log2(exp_range)), 0) if exp_range > 0 else 0
        bias_bias[name] = 0

    return weights_exp_bits, weights_bias, activations_exp_bits, activations_bias, bias_exp_bits, bias_bias


def find_batchnorm_scales(model, dataset, n_samples=256, device='cuda',
                          collate_fn=None, chunk_size=32, eps=1e-5):
    """
    Scan model for BatchNorm2d layers; compute per-channel empirical
    L1 / L2 / Linf dispersion via forward hooks. Returns:
        L1_scales[name]   = (MAD    + eps) / sqrt(var + eps)   # shape [C]
        Linf_scales[name] = (maxdev + eps) / sqrt(var + eps)   # shape [C]

    Reduction matches BN2d (over N, H, W). Single forward pass. Per-batch
    stats combined as N-weighted mean (MAD, var) and running max (maxdev).
    All accumulators are [C] and live on CPU.
    """
    model.eval()

    calib_data_cpu = make_calib_data(
        dataset=dataset, n_samples=n_samples, collate_fn=collate_fn
    )

    mad_sum = {}   # weighted-sum of per-batch MAD, CPU, [C]
    var_sum = {}   # weighted-sum of per-batch var, CPU, [C]
    maxdev  = {}   # running per-channel max of |x - mean|, CPU, [C]
    n_total = {}   # total batch samples seen per layer

    def make_hook(layer_name):
        def hook(module, inp, out):
            x = inp[0].detach()
            reduce_dims = [0] + list(range(2, x.dim()))    # [0, 2, 3] for BN2d
            view_shape  = [1, -1] + [1] * (x.dim() - 2)

            mean = x.mean(dim=reduce_dims)
            x_c  = x - mean.view(view_shape)               # 1 big tensor

            # var BEFORE abs_ so we only ever hold one x-sized tensor after this
            var_b = x_c.pow(2).mean(dim=reduce_dims).cpu()
            x_c.abs_()                                     # in-place: |x - mean|
            mad_b = x_c.mean(dim=reduce_dims).cpu()
            mxd_b = x_c.amax(dim=reduce_dims).cpu()
            del x_c

            bn = x.shape[0]
            if layer_name not in mad_sum:
                mad_sum[layer_name] = mad_b.mul_(bn)
                var_sum[layer_name] = var_b.mul_(bn)
                maxdev[layer_name]  = mxd_b
                n_total[layer_name] = bn
            else:
                mad_sum[layer_name].add_(mad_b, alpha=bn)
                var_sum[layer_name].add_(var_b, alpha=bn)
                torch.maximum(maxdev[layer_name], mxd_b, out=maxdev[layer_name])
                n_total[layer_name] += bn
        return hook

    hooks = [
        m.register_forward_hook(make_hook(name))
        for name, m in model.named_modules()
        if isinstance(m, nn.BatchNorm2d)
    ]

    with torch.no_grad():
        for i in range(0, n_samples, chunk_size):
            chunk = calib_data_cpu[i:i+chunk_size].to(device, non_blocking=True)
            model(chunk)
            del chunk
            if device != 'cpu' and torch.cuda.is_available():
                torch.cuda.empty_cache()

    for h in hooks:
        h.remove()

    L1_scales, Linf_scales = {}, {}
    for name in mad_sum:
        w         = n_total[name]
        mad_mean  = mad_sum[name] / w
        var_mean  = var_sum[name] / w
        std       = (var_mean + eps).sqrt()
        L1_scales[name]   = (mad_mean       + eps) / std
        Linf_scales[name] = (maxdev[name]   + eps) / std

    return L1_scales, Linf_scales

import contextlib

@contextlib.contextmanager
def _bypass_quantization(module):
    original = module._quantize
    module._quantize = lambda tensor, params: tensor
    try:
        yield
    finally:
        module._quantize = original

from pyhessian import hessian as pyhessian
import torch
import torch.nn.functional as F
from contextlib import nullcontext

def hess_sensitivity(
    model, dataset, n_samples=8, device='cpu',
    collate_fn=None, chunk_size=4, target_layer_types=None,
):
    model.eval()
    model.to(device)

    calib_data_cpu = make_calib_data(
        dataset=dataset, n_samples=n_samples, collate_fn=collate_fn
    )
    n = calib_data_cpu.size(0)

    if target_layer_types is None:
        target_layer_types = (LoF_Linear, LoF_Conv2d)

    target_layers = {
        name: module for name, module in model.named_modules()
        if isinstance(module, target_layer_types)
    }

    # Map param pointer -> full name
    target_param_ptrs = {}
    for layer_name, module in target_layers.items():
        for pname, p in module.named_parameters(recurse=False):
            target_param_ptrs[p.data_ptr()] = f"{layer_name}.{pname}"

    # SCALAR accumulators — not full-shape tensors
    param_grad_sq = {full: 0.0 for full in target_param_ptrs.values()}
    activation_grads = {name: 0.0 for name in target_layers}

    # Tensor-level grad hooks: no activation retention, no module backward hook
    def make_fwd_hook(layer_name):
        def hook(module, inp, out):
            act = inp[0]
            if not isinstance(act, torch.Tensor):
                return
            if not act.requires_grad:
                act.requires_grad_(True)
            def grad_hook(grad):
                activation_grads[layer_name] += grad.detach().float().pow(2).sum().item()
            act.register_hook(grad_hook)
        return hook

    fwd_hooks = [
        m.register_forward_hook(make_fwd_hook(name))
        for name, m in target_layers.items()
    ]

    def cosine_loss(p, b):
        if isinstance(p, torch.Tensor):
            pf = p.reshape(-1).float()
            bf = b.reshape(-1).float()
            m = min(pf.shape[0], bf.shape[0])
            return 1.0 - F.cosine_similarity(
                pf[:m].unsqueeze(0), bf[:m].unsqueeze(0)
            ).squeeze()
        if isinstance(p, dict):
            return sum(cosine_loss(p[k], b[k]) for k in b)
        if isinstance(p, (list, tuple)):
            return sum(cosine_loss(pi, bi) for pi, bi in zip(p, b))
        return torch.tensor(0.0, device=device)

    def slice_out(x, i):
        if isinstance(x, torch.Tensor): return x[i:i+1]
        if isinstance(x, dict): return {k: slice_out(v, i) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return type(x)(slice_out(v, i) for v in x)
        return x

    # Process one chunk at a time: compute baseline for the chunk, then
    # per-sample backward inside it. Baseline chunk is released before next chunk.
    for c0 in range(0, n, chunk_size):
        c1 = min(c0 + chunk_size, n)
        chunk = calib_data_cpu[c0:c1].to(device, non_blocking=True)

        with torch.no_grad():
            baseline_chunk = model(chunk)
            if isinstance(baseline_chunk, torch.Tensor):
                baseline_chunk = baseline_chunk.detach()

        for j in range(c1 - c0):
            model.zero_grad(set_to_none=True)

            sample = chunk[j:j+1]
            sample_baseline = slice_out(baseline_chunk, j)

            output = model(sample)
            loss = cosine_loss(output, sample_baseline)
            loss.backward()

            # scalar accumulate — no per-param tensor allocation
            for p in model.parameters():
                ptr = p.data_ptr()
                if ptr in target_param_ptrs and p.grad is not None:
                    param_grad_sq[target_param_ptrs[ptr]] += \
                        p.grad.detach().float().pow(2).sum().item()

            del output, loss, sample_baseline, sample

        del chunk, baseline_chunk
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()

    for h in fwd_hooks:
        h.remove()
    weight_sensitivity, bias_sensitivity, activation_sensitivity, accum_sensitivity = {}, {}, {}, {}
    for layer_name, module in target_layers.items():
        w_traces, b_traces = {}, {}
        for pname, param in module.named_parameters(recurse=False):
            full = f"{layer_name}.{pname}"
            trace_val = param_grad_sq.get(full, 0.0) / n
            (b_traces if 'bias' in pname else w_traces)[pname] = trace_val

            if 'weight' in pname:
                w_frob = param.detach().float().norm().item()  # Frobenius norm
                if isinstance(module, LoF_Linear):
                    k = module.weight.shape[1]                 # [out, in] → k = in
                elif isinstance(module, LoF_Conv2d):
                    k = param.shape[1] * param.shape[2] * param.shape[3]  # in_C * kH * kW
                else:
                    k = 1
                accum_sensitivity[layer_name] = w_frob * math.sqrt(k) * trace_val

        weight_sensitivity[layer_name] = w_traces
        bias_sensitivity[layer_name] = b_traces
    for k in activation_grads:
        activation_sensitivity[k] = activation_grads[k] / n

    if device != 'cpu' and torch.cuda.is_available():
        torch.cuda.empty_cache()

    return weight_sensitivity, activation_sensitivity, bias_sensitivity, accum_sensitivity
    
def noise_sensitivity_full(model, dataset, loss_fn, n_samples=256, device='cpu', collate_fn=None):

    weight_sensitivity = {}
    activation_sensitivity = {}
    bias_sensitivity = {}          # <-- new

    model.eval()
    model.to(device)

    calib_data = make_calib_data(dataset=dataset, n_samples=n_samples, collate_fn=collate_fn)
    calib_data = calib_data.to(device)

    with torch.no_grad():
        baseline_loss = loss_fn(model(calib_data)).item()

    captured_activation = {}
    def make_capture_hook(layer_name):
        def hook(module, input, output):
            captured_activation[layer_name] = input[0].detach()
        return hook

    capture_hooks = [
        module.register_forward_hook(make_capture_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, (LoF_Linear, LoF_Conv2d))
    ]
    with torch.no_grad():
        model(calib_data)
    for h in capture_hooks:
        h.remove()

    for name, module in model.named_modules():
        if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
            continue

        with torch.no_grad():
            # --- Weight sensitivity ---
            W = module.weight
            w_sigma = W.abs().max().item()
            noise = torch.randn_like(W) * w_sigma
            module.weight.add_(noise)
            weight_sensitivity[name] = baseline_loss - loss_fn(model(calib_data)).item()
            module.weight.sub_(noise)

            # --- Bias sensitivity ---             
            if module.bias is not None:
                B = module.bias
                b_sigma = B.abs().max().item()
                b_noise = torch.randn_like(B) * b_sigma
                module.bias.add_(b_noise)
                bias_sensitivity[name] = baseline_loss - loss_fn(model(calib_data)).item()
                module.bias.sub_(b_noise)

            # --- Activation sensitivity ---
            A = captured_activation[name]
            a_sigma = A.abs().max().item()

            def make_noise_hook(sigma):
                def hook(module, input, output):
                    noisy_input = input[0] + torch.randn_like(input[0]) * sigma
                    if isinstance(module, LoF_Linear):
                        return torch.matmul(noisy_input, module.weight.t()) + (module.bias if module.bias is not None else 0)
                    elif isinstance(module, LoF_Conv2d):
                        return F.conv2d(noisy_input, module.weight, module.bias,
                                        stride=module.stride, padding=module.padding,
                                        dilation=module.dilation, groups=module.groups)
                return hook

            noise_hook = module.register_forward_hook(make_noise_hook(a_sigma))
            activation_sensitivity[name] = baseline_loss - loss_fn(model(calib_data)).item()
            noise_hook.remove()

    return weight_sensitivity, activation_sensitivity, bias_sensitivity  



def replace_batchnorm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, nn.Identity())
    return model

#works by using forward hook to collect data at the layer, for XX^t and perform Cholesky decomp. Then we use GPTQ to update the weight and continue with the forward pass
def quantize_weights_with_gptq(model, dataset, exponent_bits, mantissa_bits, n_samples=256, device='cpu', perturb_ratio=0.01, collate_fn=None):
    model.eval()
    model.to(device)
    calib_data = make_calib_data(dataset=dataset, n_samples=n_samples, collate_fn=collate_fn)
    calib_data = calib_data.to(device)
    captured_activations = {}
    captured_pre_choleskys = {}
    blocksize = 64

    def make_capture_hook(layer_name):
        def hook(module, input):
            exp_bits = exponent_bits[layer_name]
            mant_bits = mantissa_bits[layer_name]
            smallest_subnormal = 2 ** (1 - (1 << exp_bits) - mant_bits)

            X = input[0].detach()

            # --- Build Hessian (X^T X / n) ---
            if isinstance(module, LoF_Conv2d) and module.groups > 1:
                C = module.weight.shape[0]
                kH, kW = module.kernel_size
                k2 = kH * kW
                X_unf = torch.nn.functional.unfold(
                    X,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                )
                X_unf = X_unf.reshape(X.shape[0], C, k2, -1)
                X_unf = X_unf.permute(1, 2, 0, 3).reshape(C, k2, -1)
                H = torch.zeros(k2, k2, device=X.device)
                for c in range(C):
                    H += X_unf[c] @ X_unf[c].t()
                H /= (C * X_unf.shape[2])
            elif X.ndim == 3:
                X = X.reshape(-1, X.shape[-1])
                nsamples = X.shape[0]
                H = (X.t() @ X) / nsamples
            elif X.ndim == 4:
                X = torch.nn.functional.unfold(
                    X,
                    kernel_size=module.kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                )
                X = X.permute(0, 2, 1).reshape(-1, X.shape[1])
                nsamples = X.shape[0]
                H = (X.t() @ X) / nsamples
            else:
                nsamples = X.shape[0]
                H = (X.t() @ X) / nsamples

            # --- Shared GPTQ from here ---
            damp = perturb_ratio * torch.diag(H).mean()
            H += damp * torch.eye(H.shape[0], device=H.device)
            perm = torch.argsort(torch.diag(H), descending=True)
            invperm = torch.argsort(perm)

            W = module.weight.data.clone()

            # --- Underflow mask: weights that will be forced to zero (SparseGPT) ---
            underflow_mask = W.abs() < smallest_subnormal

            orig_shape = W.shape

            # Reshape for conv layers (flatten spatial dims into columns)
            if W.ndim == 4:
                W = W.reshape(W.shape[0], -1)
                underflow_mask = underflow_mask.reshape(underflow_mask.shape[0], -1)

            columns = W.shape[1]

            # Apply the same column permutation to W, H, and the underflow mask
            W = W[:, perm]
            H = H[perm][:, perm]
            underflow_mask = underflow_mask[:, perm]

            # --- Cholesky decomposition of Hessian inverse ---
            def has_inf_or_nan(tensor):
                return not torch.isfinite(tensor).all()

            try:
                H_d = H.double()
                cho = torch.linalg.cholesky(H_d)
                H_inv = torch.cholesky_inverse(cho)
                Hinv = torch.linalg.cholesky(H_inv, upper=True).float()
            except torch.linalg.LinAlgError:
                print(f"  [WARN] Cholesky failed for layer {layer_name}, skipping GPTQ (keeping weights as-is)")
                return
           

            Q = torch.zeros_like(W)

            # --- Joint SparseGPT + GPTQ block loop ---
            for i1 in range(0, columns, blocksize):
                i2 = min(i1 + blocksize, columns)
                count = i2 - i1

                W_block = W[:, i1:i2].clone()
                Q_block = torch.zeros_like(W_block)
                Err_block = torch.zeros_like(W_block)
                Hinv_block = Hinv[i1:i2, i1:i2]
                mask_block = underflow_mask[:, i1:i2]

                for j in range(count):
                    w = W_block[:, j]
                    d = Hinv_block[j, j]
                    q = w.clone()

                    # SparseGPT step: force underflowing weights to zero
                    q[mask_block[:, j]] = 0.0

                    # GPTQ step: quantize the surviving (non-underflow) weights
                    survive = ~mask_block[:, j]
                    if survive.any():
                        q[survive] = lof.exp_mant_quantize(
                            q[survive], exp_bits, mant_bits
                        )

                    Q_block[:, j] = q

                    # Error compensation (shared by both sparse and quant updates)
                    err = (w - q) / d
                    Err_block[:, j] = err

                    # Propagate error to remaining columns in this block
                    if j + 1 < count:
                        W_block[:, j + 1:] -= (
                            err.unsqueeze(1) * Hinv_block[j, j + 1:].unsqueeze(0)
                        )

                Q[:, i1:i2] = Q_block

                # Propagate error to all remaining columns beyond this block
                if i2 < columns:
                    W[:, i2:] -= Err_block @ Hinv[i1:i2, i2:]

            # --- Undo permutation and write back ---
            Q = Q[:, invperm]
            if len(orig_shape) == 4:
                Q = Q.reshape(orig_shape)
            module.weight.data.copy_(Q)

        return hook

    def delete_activation_hook(layer_name):
        def hook(module, input, output):
            captured_activations.pop(layer_name, None)
            captured_pre_choleskys.pop(layer_name, None)
        return hook

    # --- Register hooks on all LoF layers ---
    capture_pre_hooks = [
        module.register_forward_pre_hook(make_capture_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, (LoF_Linear, LoF_Conv2d))
    ]
    capture_post_hooks = [
        module.register_forward_hook(delete_activation_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, (LoF_Linear, LoF_Conv2d))
    ]

    # --- Run calibration data through model (triggers all hooks) ---
    with torch.no_grad():
        model(calib_data)

    # --- Clean up hooks ---
    for h in capture_pre_hooks:
        h.remove()
    for h in capture_post_hooks:
        h.remove()

    return model

def sanity_test():

    def test_hutchinson_quadratic():
        """
        For loss = 0.5 * x^T A x, the Hessian is exactly A.
        So trace(H) = trace(A), which we can compute exactly.
        We test this with a simple diagonal A so trace(A) = sum(diag).
        """
        torch.manual_seed(42)

        # Diagonal Hessian with known trace
        # loss = 0.5 * sum(a_i * x_i^2)  =>  d²loss/dx_i² = a_i
        diag_values = torch.tensor([1.0, 2.0, 3.0, 4.0])  # trace = 10.0
        expected_trace = diag_values.sum().item()           # 10.0

        x = torch.zeros(4, requires_grad=True)

        # Build a fake "model" and "loss" — we just need a scalar with a graph
        loss = 0.5 * (diag_values * x * x).sum()

        # Use a large n_samples since it's a stochastic estimator
        estimated_trace = batch_hutchinson_approx(
            model=None,      # not used inside the function
            loss=loss,
            param=x,
            n_samples=2000,
            retain_graph=False
        )

        # Hutchinson is unbiased so with enough samples we should be close
        assert abs(estimated_trace - expected_trace) < 0.5, (
            f"Expected trace ~{expected_trace}, got {estimated_trace:.4f}"
        )


    def test_hutchinson_identity_hessian():
        """
        loss = 0.5 * ||x||^2  =>  H = I  =>  trace = n
        """
        torch.manual_seed(0)
        n = 16
        x = torch.zeros(n, requires_grad=True)
        loss = 0.5 * (x * x).sum()

        estimated_trace = batch_hutchinson_approx(
            model=None,
            loss=loss,
            param=x,
            n_samples=2000,
            retain_graph=False
        )

        assert abs(estimated_trace - n) < 1.0, (
            f"Expected trace ~{n}, got {estimated_trace:.4f}"
        )


    def test_hutchinson_zero_hessian():
        """
        loss = w^T x (linear in x) => H = 0 => trace = 0
        Hutchinson should return exactly 0 here (not just close),
        since Hv = 0 for all v.
        """
        torch.manual_seed(0)
        x = torch.randn(8, requires_grad=True)
        w = torch.randn(8)
        loss = (w * x).sum()

        estimated_trace = batch_hutchinson_approx(
            model=None,
            loss=loss,
            param=x,
            n_samples=100,
            retain_graph=False
        )

        assert abs(estimated_trace) < 1e-6, (
            f"Expected trace = 0.0, got {estimated_trace:.6f}"
        )

    test_hutchinson_identity_hessian()
    test_hutchinson_quadratic()
    test_hutchinson_zero_hessian

# sanity_test()

def test_hess_sensitivity_quantized_mlp():
    torch.manual_seed(42)

    n = 64
    X = torch.randn(n, 128)
    y = torch.randint(0, 10, (n,))
    dataset = torch.utils.data.TensorDataset(X, y)

    model = nn.Sequential(
        lof.LoF_Linear(128, 64, act_mant=4, act_exp=3, weight_mant=4, weight_exp=3, bias_mant=4, bias_exp=3),
        nn.ReLU(),
        lof.LoF_Linear(64, 32, act_mant=4, act_exp=3, bias_mant=4, bias_exp=3, weight_mant=4, weight_exp=3),
        nn.ReLU(),
        lof.LoF_Linear(32, 10, act_mant=4, act_exp=3,weight_mant=4, weight_exp=3, bias_mant=4, bias_exp=3)
    )
    loss_fn = nn.CrossEntropyLoss()

    weight_sensitivity, activation_sensitivity = hess_sensitivity(
        model=model,
        dataset=dataset,
        loss_fn=loss_fn,
        n_samples=32,
        device='cpu'
    )

    # With named_modules, expect nested keys like 'fc1', 'fc2' etc.
    # LoF_Quantize and ReLU should be absent
    for key in weight_sensitivity:
        assert key in activation_sensitivity
        for param_name, trace in weight_sensitivity[key].items():
            assert torch.isfinite(torch.tensor(trace)), \
                f"{key}.{param_name} weight trace not finite: {trace}"
        act_trace = activation_sensitivity[key]
        assert torch.isfinite(torch.tensor(act_trace)), \
            f"{key} activation trace not finite: {act_trace}"

    assert len(weight_sensitivity) == 3, \
        f"Expected 3 LoF_Linear layers, got {len(weight_sensitivity)}"

    print("PASSED: test_hess_sensitivity_quantized_mlp")
    for key in weight_sensitivity:
        for param_name, trace in weight_sensitivity[key].items():
            print(f"  weight [{key}][{param_name}]: {trace:.4f}")
        print(f"  activation [{key}]: {activation_sensitivity[key]:.4f}")

