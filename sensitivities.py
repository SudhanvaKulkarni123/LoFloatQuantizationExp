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
                          collate_fn=None, chunk_size=32,
                          linf_quantile=0.999):
    model.eval()
    calib_data_cpu = make_calib_data(
        dataset=dataset, n_samples=n_samples, collate_fn=collate_fn
    )

    # Per-layer accumulators
    mad_sum, maxdev_sum, n_total = {}, {}, {}
    bn_modules = {}  # name -> module, so we can read running_var directly

    def make_hook(name, module):
        bn_modules[name] = module
        # Use BN's own running stats — these are what the model actually uses
        mean = module.running_mean.detach()  # [C], on device
        view_shape = None  # set on first call

        def hook(_, inp, __):
            nonlocal view_shape
            x = inp[0].detach()
            if view_shape is None:
                view_shape = [1, -1] + [1] * (x.dim() - 2)
            reduce_dims = [0] + list(range(2, x.dim()))

            x_c = (x - mean.view(view_shape)).abs_()  # |x - μ_running|
            mad_b = x_c.mean(dim=reduce_dims).cpu()

            # Quantile-based "soft max" — flatten non-channel dims
            C = x_c.shape[1]
            x_c_flat = x_c.permute(1, 0, *range(2, x_c.dim())).reshape(C, -1)
            mxd_b = torch.quantile(x_c_flat, linf_quantile, dim=1).cpu()

            del x_c, x_c_flat
            bn = x.shape[0]

            if name not in mad_sum:
                mad_sum[name] = mad_b * bn
                maxdev_sum[name] = mxd_b * bn
                n_total[name] = bn
            else:
                mad_sum[name].add_(mad_b, alpha=bn)
                maxdev_sum[name].add_(mxd_b, alpha=bn)
                n_total[name] += bn

        return hook

    hooks = [
        m.register_forward_hook(make_hook(name, m))
        for name, m in model.named_modules()
        if isinstance(m, nn.BatchNorm2d)
    ]

    with torch.no_grad():
        for i in range(0, n_samples, chunk_size):
            chunk = calib_data_cpu[i:i+chunk_size].to(device, non_blocking=True)
            model(chunk)

    for h in hooks:
        h.remove()

    L1_scales, Linf_scales = {}, {}
    for name, module in bn_modules.items():
        w = n_total[name]
        mad_mean = mad_sum[name] / w
        maxdev_mean = maxdev_sum[name] / w

        # Use BN's actual normalization constant
        std = (module.running_var + module.eps).sqrt().cpu()

        L1_scales[name] = mad_mean / std
        Linf_scales[name] = maxdev_mean / std

    return L1_scales, Linf_scales


def find_gemm_grades(model, dataset, accum_sensitivities, device='cuda'):
    model.eval()
    model.to(device)


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

    def output_l2(x):
        if isinstance(x, torch.Tensor):
            if x.is_floating_point() and x.requires_grad:
                return 0.5 * x.float().pow(2).sum()
            return None
        if isinstance(x, dict):
            terms = [output_l2(v) for v in x.values()]
        elif isinstance(x, (list, tuple)):
            terms = [output_l2(v) for v in x]
        else:
            return None
        terms = [t for t in terms if t is not None]
        return sum(terms) if terms else None

    # Process one chunk at a time, per-sample backward.
    for c0 in range(0, n, chunk_size):
        c1 = min(c0 + chunk_size, n)
        chunk = calib_data_cpu[c0:c1].to(device, non_blocking=True)

        for j in range(c1 - c0):
            model.zero_grad(set_to_none=True)

            sample = chunk[j:j+1]

            output = model(sample)
            loss = output_l2(output)
            if loss is None:
                continue
            loss.backward()

            # scalar accumulate — no per-param tensor allocation
            for p in model.parameters():
                ptr = p.data_ptr()
                if ptr in target_param_ptrs and p.grad is not None:
                    param_grad_sq[target_param_ptrs[ptr]] += \
                        p.grad.detach().float().pow(2).sum().item()

            del output, loss, sample

        del chunk
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



def quantize_weights_with_gptq(
    model,
    dataset,
    exponent_bits,
    mantissa_bits,
    n_samples=256,
    device='cpu',
    perturb_ratio=0.01,
    collate_fn=None,
    micro_batch_size=4,
):
    """
    Memory-efficient sequential GPTQ for LoF layers.

    Processes one layer at a time:
      For each LoF layer L (in forward order):
        1. Register a Hessian-accumulation hook on L.
        2. Run the calibration set through the model in micro-batches.
           Earlier layers have *already* been quantized in previous
           iterations, so L's Hessian is built from the same post-
           quantization activations the original single-pass code saw.
        3. Apply the SparseGPT + GPTQ block update to L.
        4. Remove the hook and move on.

    This restores the error-feedback property of the original code (each
    layer compensates for upstream quantization error) while never holding
    more than one layer's Hessian or activations in memory.

    Cost: roughly N_layers forward passes through the model. For YOLOv8n /
    YOLO26n this is a minute or two on a 3070 with micro_batch_size=1.
    """
    model.eval()
    model.to(device)

    calib_data = make_calib_data(dataset=dataset, n_samples=n_samples, collate_fn=collate_fn)
    calib_data = calib_data.cpu()  # move per micro-batch

    blocksize = 64
    use_cuda = str(device).startswith('cuda')

    # LoF layers in forward (registration) order
    lof_layers = [
        (name, m) for name, m in model.named_modules()
        if isinstance(m, (LoF_Linear, LoF_Conv2d))
    ]

    for layer_idx, (layer_name, module) in enumerate(lof_layers):
        # ---------- Phase 1: accumulate H for this layer only ----------
        hess_state = {'H_sum': None, 'n': 0}

        def _accumulate(H_local, n_local):
            if hess_state['H_sum'] is None:
                hess_state['H_sum'] = H_local
            else:
                hess_state['H_sum'].add_(H_local)
            hess_state['n'] += n_local

        def hook(mod, input):
            X = input[0].detach()

            # Depthwise conv: H is (k^2, k^2)
            if isinstance(mod, LoF_Conv2d) and mod.groups > 1:
                C = mod.weight.shape[0]
                kH, kW = mod.kernel_size
                k2 = kH * kW
                H_local = torch.zeros(k2, k2, device=X.device)
                n_local = 0
                for b in range(X.shape[0]):
                    X_unf = torch.nn.functional.unfold(
                        X[b:b + 1],
                        kernel_size=mod.kernel_size,
                        stride=mod.stride,
                        padding=mod.padding,
                        dilation=mod.dilation,
                    )                                              # (1, C*k2, L)
                    X_unf = X_unf.reshape(1, C, k2, -1).squeeze(0)  # (C, k2, L)
                    L = X_unf.shape[-1]
                    for c in range(C):
                        Xc = X_unf[c]
                        H_local.add_(Xc @ Xc.t())
                    n_local += C * L
                    del X_unf
                _accumulate(H_local, n_local)
                return

            # 3D linear input
            if X.ndim == 3:
                X2 = X.reshape(-1, X.shape[-1])
                _accumulate(X2.t() @ X2, X2.shape[0])
                return

            # Standard conv
            if X.ndim == 4:
                kH, kW = mod.kernel_size
                col = X.shape[1] * kH * kW
                H_local = torch.zeros(col, col, device=X.device)
                n_local = 0
                for b in range(X.shape[0]):
                    Xb = torch.nn.functional.unfold(
                        X[b:b + 1],
                        kernel_size=mod.kernel_size,
                        stride=mod.stride,
                        padding=mod.padding,
                        dilation=mod.dilation,
                    )                                              # (1, C*k2, L)
                    Xb = Xb.squeeze(0).t().contiguous()             # (L, C*k2)
                    H_local.add_(Xb.t() @ Xb)
                    n_local += Xb.shape[0]
                    del Xb
                _accumulate(H_local, n_local)
                return

            # Plain Linear
            _accumulate(X.t() @ X, X.shape[0])

        h = module.register_forward_pre_hook(hook)

        with torch.no_grad():
            for i in range(0, calib_data.shape[0], micro_batch_size):
                batch = calib_data[i:i + micro_batch_size].to(device, non_blocking=True)
                model(batch)
                del batch
                if use_cuda:
                    torch.cuda.empty_cache()

        h.remove()

        if hess_state['H_sum'] is None:
            # Layer was never reached during forward (e.g. dead branch); skip
            continue

        H = hess_state['H_sum'] / hess_state['n']
        del hess_state

        # ---------- Phase 2: GPTQ update for this layer ----------
        exp_bits = exponent_bits[layer_name]
        mant_bits = mantissa_bits[layer_name]
        smallest_subnormal = 2 ** (1 - (1 << exp_bits) - mant_bits)

        damp = perturb_ratio * torch.diag(H).mean()
        H += damp * torch.eye(H.shape[0], device=H.device)
        perm = torch.argsort(torch.diag(H), descending=True)
        invperm = torch.argsort(perm)

        W = module.weight.data.clone()
        underflow_mask = W.abs() < smallest_subnormal
        orig_shape = W.shape

        if W.ndim == 4:
            W = W.reshape(W.shape[0], -1)
            underflow_mask = underflow_mask.reshape(underflow_mask.shape[0], -1)

        columns = W.shape[1]
        W = W[:, perm]
        H = H[perm][:, perm]
        underflow_mask = underflow_mask[:, perm]

        try:
            H_d = H.double()
            cho = torch.linalg.cholesky(H_d)
            H_inv = torch.cholesky_inverse(cho)
            Hinv = torch.linalg.cholesky(H_inv, upper=True).float()
            del H_d, cho, H_inv
        except torch.linalg.LinAlgError:
            print(f"  [WARN] Cholesky failed for layer {layer_name}, skipping GPTQ (keeping weights as-is)")
            del H, W, underflow_mask
            if use_cuda:
                torch.cuda.empty_cache()
            continue

        Q = torch.zeros_like(W)

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

                q[mask_block[:, j]] = 0.0
                survive = ~mask_block[:, j]
                if survive.any():
                    q[survive] = lof.exp_mant_quantize(q[survive], exp_bits, mant_bits)

                Q_block[:, j] = q
                err = (w - q) / d
                Err_block[:, j] = err

                if j + 1 < count:
                    W_block[:, j + 1:] -= err.unsqueeze(1) * Hinv_block[j, j + 1:].unsqueeze(0)

            Q[:, i1:i2] = Q_block
            if i2 < columns:
                W[:, i2:] -= Err_block @ Hinv[i1:i2, i2:]

        Q = Q[:, invperm]
        if len(orig_shape) == 4:
            Q = Q.reshape(orig_shape)
        module.weight.data.copy_(Q)

        del W, H, Hinv, Q, underflow_mask
        if use_cuda:
            torch.cuda.empty_cache()

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

