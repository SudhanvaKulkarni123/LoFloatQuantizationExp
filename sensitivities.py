import torch
import torch.nn as nn
import torch.nn.functional as F
import LoFloat as lof
from LoFloat import LoF_Linear, LoF_Conv2d
import copy
import math



def make_calib_data(dataset, n_samples=256):

    indices = torch.randperm(len(dataset))[:n_samples]
    calib_subset = torch.utils.data.Subset(dataset, indices)
    calib_loader = torch.utils.data.DataLoader(calib_subset, batch_size=n_samples, shuffle=False)
    calib_data, calib_labels = next(iter(calib_loader))
    return calib_data, calib_labels

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

def find_range(model, dataset, n_samples=256, device='cuda'):
    weights_minmax = {}
    activations_minmax = {}
    bias_minmax = {}
    calib_data, calib_labels = make_calib_data(dataset=dataset, n_samples=n_samples)
    calib_data, calib_labels = calib_data.to(device), calib_labels.to(device)

    # Capture activations
    captured_activations = {}
    def make_capture_hook(layer_name):
        def hook(module, input, output):
            captured_activations[layer_name] = input[0].detach()
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

    def abs_range(tensor):
        flat = tensor.abs().flatten()
        nonzero = flat[flat != 0]
        min_val = nonzero.min().item() if len(nonzero) > 0 else 0.0
        max_val = flat.max().item()
        return {"min": min_val, "max": max_val}

    for name, module in model.named_modules():
        if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
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
        exp_range = abs(max_exp - min_exp)  #need abs if min is 0 since frexp(0) = 0
        print("exp range for {}: {}".format(name, exp_range))
        weights_exp_bits[name] = math.ceil(math.log2(exp_range)) if exp_range > 0 else 0
        print("exp bits for {}: {}".format(name, weights_exp_bits[name]))
        weights_bias[name] = -min_exp 

    for name, val_range in activations_minmax.items():
        _, min_exp = math.frexp(val_range["min"])
        _, max_exp = math.frexp(val_range["max"])
        exp_range = abs(max_exp - min_exp)
        activations_exp_bits[name] = math.ceil(math.log2(exp_range)) if exp_range > 0 else 0
        activations_bias[name] = -min_exp 

    for name, val_range in bias_minmax.items():
        _, min_exp = math.frexp(val_range["min"])
        _, max_exp = math.frexp(val_range["max"])
        exp_range = abs(max_exp - min_exp)
        bias_exp_bits[name] = math.ceil(math.log2(exp_range)) if exp_range > 0 else 0
        bias_bias[name] = -min_exp 

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
def hess_sensitivity(model, dataset, n_samples=8, device='cpu'):
    weight_sensitivity = {}
    activation_sensitivity = {}
    bias_sensitivity = {}

    model.eval()
    model.to(device)

    calib_data, _ = make_calib_data(dataset=dataset, n_samples=n_samples)
    calib_data = calib_data.to(device)

    # ── Compute baseline output (detached) ──
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

    def internal_loss_fn(preds):
        def mse_recursive(p, b):
            if isinstance(b, torch.Tensor):
                return F.mse_loss(p.float(), b.float())
            elif isinstance(b, dict):
                return sum(mse_recursive(p[k], b[k]) for k in b)
            elif isinstance(b, (list, tuple)):
                return sum(mse_recursive(pi, bi) for pi, bi in zip(p, b))
            return torch.tensor(0.0, device=device)
        return mse_recursive(preds, baseline)

    # ── Step 1: Capture activations and register gradient hooks ──
    # We need activations to require grad so we can get gradients w.r.t. them

    activation_grads = {}   # layer_name -> list of squared grads per sample
    captured_inputs = {}

    def make_activation_hook(layer_name):
        """Forward hook: capture input activation and enable its gradient."""
        def hook(module, inp, out):
            act = inp[0]
            if not act.requires_grad:
                act.requires_grad_(True)
                act.retain_grad()
            captured_inputs[layer_name] = act
        return hook

    # Register forward hooks on target layers
    target_layers = {
        name: module for name, module in model.named_modules()
        if isinstance(module, (LoF_Linear, LoF_Conv2d))
    }

    fwd_hooks = []
    for name, module in target_layers.items():
        fwd_hooks.append(module.register_forward_hook(make_activation_hook(name)))

    # ── Step 2: Run forward + backward per sample (Fisher = E[grad^2]) ──
    # Accumulate squared gradients across samples

    param_grad_sq = {}  # full_param_name -> accumulated (grad)^2

    for i in range(calib_data.size(0)):
        model.zero_grad()
        captured_inputs.clear()

        sample = calib_data[i:i+1]
        output = model(sample)
        loss = internal_loss_fn(output)
        loss.backward()

        # --- Parameter (weight/bias) Fisher diagonal ---
        for pname, param in model.named_parameters():
            if param.grad is not None:
                g2 = param.grad.detach().pow(2)
                if pname in param_grad_sq:
                    param_grad_sq[pname] += g2
                else:
                    param_grad_sq[pname] = g2.clone()

        # --- Activation Fisher diagonal ---
        for layer_name, act in captured_inputs.items():
            if act.grad is not None:
                g2 = act.grad.detach().pow(2).sum()  # scalar sensitivity
                if layer_name in activation_grads:
                    activation_grads[layer_name] += g2.item()
                else:
                    activation_grads[layer_name] = g2.item()

    # Clean up hooks
    for h in fwd_hooks:
        h.remove()

    n = calib_data.size(0)

    # ── Step 3: Average and map to layer names ──

    # Average the squared gradients
    for k in param_grad_sq:
        param_grad_sq[k] /= n

    for layer_name, module in target_layers.items():
        w_traces = {}
        b_traces = {}

        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{layer_name}.{param_name}"
            # Sum the diagonal Fisher for this parameter -> scalar sensitivity
            fisher_trace = param_grad_sq.get(full_name)
            if fisher_trace is not None:
                trace_val = fisher_trace.sum().item()
            else:
                trace_val = 0.0

            if 'bias' in param_name:
                b_traces[param_name] = trace_val
            else:
                w_traces[param_name] = trace_val

        weight_sensitivity[layer_name] = w_traces
        bias_sensitivity[layer_name] = b_traces

    # Activation sensitivity (averaged)
    for k in activation_grads:
        activation_sensitivity[k] = activation_grads[k] / n

    torch.cuda.empty_cache()
    return weight_sensitivity, activation_sensitivity, bias_sensitivity


def noise_sensitivity_full(model, dataset, loss_fn, n_samples=256, device='cpu'):

    weight_sensitivity = {}
    activation_sensitivity = {}
    bias_sensitivity = {}          # <-- new

    model.eval()
    model.to(device)

    calib_data, calib_labels = make_calib_data(dataset=dataset, n_samples=n_samples)
    calib_data, calib_labels = calib_data.to(device), calib_labels.to(device)

    with torch.no_grad():
        baseline_loss = loss_fn(model(calib_data)).item()

    captured_activations = {}
    def make_capture_hook(layer_name):
        def hook(module, input, output):
            captured_activations[layer_name] = input[0].detach()
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
            A = captured_activations[name]
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

