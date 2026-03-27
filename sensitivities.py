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

def find_range(model, dataset, n_samples=256, device='cuda', collate_fn=None):
    weights_minmax = {}
    activations_minmax = {}
    bias_minmax = {}
    calib_data = make_calib_data(dataset=dataset, n_samples=n_samples, collate_fn=collate_fn)
    calib_data = calib_data.to(device)

    # Capture activations
    captured_activations = {}
    def make_capture_hook(layer_name):
        def hook(module, input, output):
            captured_activations[layer_name] = input[0].detach()
        return hook
    model.eval()
    capture_hooks = [
        module.register_forward_hook(make_capture_hook(name))
        for name, module in model.named_modules()
        if isinstance(module, (LoF_Linear, LoF_Conv2d))
    ]


    with torch.no_grad():
        model(calib_data)
    for h in capture_hooks:
        h.remove()

    def abs_range(tensor):
        flat = tensor.abs().flatten()
        nonzero = flat[flat != 0]
        min_val = nonzero.min().item() if len(nonzero) > 0 else 0.0
        max_val = flat.max().item()
        return {"min": min_val, "max": max_val}

    for name, module in model.named_modules():
        if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
            continue
        if name not in captured_activations:
            print(f"  [SKIP] {name} — forward() never called")
            continue
        weights_minmax[name]     = abs_range(module.weight.detach())
        activations_minmax[name] = abs_range(captured_activations[name])
        bias_minmax[name]        = abs_range(module.bias.detach()) if module.bias is not None else {"min": 0.0, "max": 0.0}

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
    model,
    dataset,
    n_samples=8,
    device='cpu',
    collate_fn=None,
    chunk_size=4,
    target_layer_types=None
):
    """
    Compute diagonal Fisher Information (Hessian approx) per layer.
    """
    weight_sensitivity = {}
    activation_sensitivity = {}
    bias_sensitivity = {}

    model.eval()
    model.to(device)

    calib_data = make_calib_data(dataset=dataset, n_samples=n_samples, collate_fn=collate_fn)
    calib_data = calib_data.to(device)

    n = calib_data.size(0)

    # ── Compute baseline output (detached, full batch, no grad) ──
    def detach_output(x):
        if isinstance(x, torch.Tensor):
            return x.detach()
        elif isinstance(x, dict):
            return {k: detach_output(v) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(detach_output(v) for v in x)
        return x

    with torch.no_grad():
        baseline = detach_output(model(calib_data))

    # ── Loss: cosine distance, flatten to avoid broadcasting ──
    def cosine_loss_recursive(p, b):
        if isinstance(b, torch.Tensor):
            p_flat = p.reshape(-1).float()
            b_flat = b.reshape(-1).float()
            min_len = min(p_flat.shape[0], b_flat.shape[0])
            return 1.0 - F.cosine_similarity(
                p_flat[:min_len].unsqueeze(0),
                b_flat[:min_len].unsqueeze(0)
            ).squeeze()
        elif isinstance(b, dict):
            return sum(cosine_loss_recursive(p[k], b[k]) for k in b)
        elif isinstance(b, (list, tuple)):
            return sum(cosine_loss_recursive(pi, bi) for pi, bi in zip(p, b))
        return torch.tensor(0.0, device=device)

    def slice_output(x, i):
        if isinstance(x, torch.Tensor):
            return x[i:i+1]
        elif isinstance(x, dict):
            return {k: slice_output(v, i) for k, v in x.items()}
        elif isinstance(x, (list, tuple)):
            return type(x)(slice_output(v, i) for v in x)
        return x

    # ── Identify target layers ──
    if target_layer_types is None:
        target_layer_types = (LoF_Linear, LoF_Conv2d)

    target_layers = {
        name: module for name, module in model.named_modules()
        if isinstance(module, target_layer_types)
    }

    # ── Build a fast param-name lookup ──
    target_param_ptrs = {}
    for layer_name, module in target_layers.items():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{layer_name}.{param_name}"
            target_param_ptrs[param.data_ptr()] = (full_name, 'bias' in param_name)

    # ── Pre-allocate accumulators ──
    param_grad_sq = {}
    for pname, param in model.named_parameters():
        if param.data_ptr() in target_param_ptrs:
            param_grad_sq[target_param_ptrs[param.data_ptr()][0]] = \
                torch.zeros_like(param, dtype=torch.float32)

    activation_grads = {name: 0.0 for name in target_layers}

    # ── Activation hooks ──
    captured_inputs = {}

    def make_activation_hook(layer_name):
        def hook(module, inp, out):
            act = inp[0]
            if not act.requires_grad:
                act.requires_grad_(True)
                act.retain_grad()
            captured_inputs[layer_name] = act
        return hook

    fwd_hooks = []
    for name, module in target_layers.items():
        fwd_hooks.append(module.register_forward_hook(make_activation_hook(name)))

    # ── Main loop ──
    for chunk_start in range(0, n, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n)

        for i in range(chunk_start, chunk_end):
            model.zero_grad(set_to_none=True)
            captured_inputs.clear()

            sample = calib_data[i:i+1]
            sample_baseline = slice_output(baseline, i)

            output = model(sample)
            loss = cosine_loss_recursive(output, sample_baseline)
            loss.backward()

            # ── Accumulate squared grads (only target params) ──
            for param in model.parameters():
                ptr = param.data_ptr()
                if ptr in target_param_ptrs and param.grad is not None:
                    full_name = target_param_ptrs[ptr][0]
                    g = param.grad.detach().float()
                    param_grad_sq[full_name].addcmul_(g, g, value=1.0)

            # ── Activation grads ──
            for layer_name, act in captured_inputs.items():
                if act.grad is not None:
                    g = act.grad.detach().float()
                    activation_grads[layer_name] += g.pow(2).sum().item()

        # Free intermediate graph memory between chunks
        if device != 'cpu' and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Cleanup hooks ──
    for h in fwd_hooks:
        h.remove()

    # ── Average and organize results ──
    for layer_name, module in target_layers.items():
        w_traces = {}
        b_traces = {}

        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{layer_name}.{param_name}"
            fisher_diag = param_grad_sq.get(full_name)
            trace_val = (fisher_diag.sum().item() / n) if fisher_diag is not None else 0.0

            if 'bias' in param_name:
                b_traces[param_name] = trace_val
            else:
                w_traces[param_name] = trace_val

        weight_sensitivity[layer_name] = w_traces
        bias_sensitivity[layer_name] = b_traces

    for k in activation_grads:
        activation_sensitivity[k] = activation_grads[k] / n

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return weight_sensitivity, activation_sensitivity, bias_sensitivity

    
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

