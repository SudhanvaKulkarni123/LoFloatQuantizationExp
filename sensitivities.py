import torch
import torch.nn as nn
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



def batch_hutchinson_approx(model, loss, param, n_samples, retain_graph=True):

    param_shape = param.shape
    param_numel = param.numel()

    V = (torch.randint(0, 2, (n_samples, param_numel),
                       device=param.device).float() * 2 - 1)

    # first pass for Jacobian
    grad, = torch.autograd.grad(
        loss, param,
        create_graph=True,
        retain_graph=True
    )

    #second for Hessian, eed to do this in a loop since PyTprch autograd needs scalar
    Hv_list = []
    for i in range(n_samples):
        v_i = V[i].reshape(param_shape)
        keep = True if i < n_samples - 1 else retain_graph

        Hv_i, = torch.autograd.grad(
            grad, param,
            grad_outputs=v_i,
            retain_graph=keep
        )
        Hv_list.append(Hv_i.reshape(-1).detach())

    Hv_matrix = torch.stack(Hv_list, dim=0)
    trace_estimate = (V * Hv_matrix).sum(dim=1).mean().item()

    return trace_estimate
    


def find_range(model, dataset, n_samples=256, device='cpu', low_percentile=0.0, high_percentile=1.0):

    weights_minmax = {}
    activations_minmax = {}
    bias_minmax = {}

    calib_data, calib_labels = make_calib_data(dataset=dataset, n_samples=n_samples)
    calib_data, calib_labels = calib_data.to(device), calib_labels.to(device)

    #define and apply hooks to record activation data
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

    #run model to capture data
    with torch.no_grad():
        model(calib_data)
    for h in capture_hooks:
        h.remove()

    def percentile_range(tensor):
        flat = tensor.abs().flatten()
        low  = torch.quantile(flat, low_percentile).item()
        high = torch.quantile(flat, high_percentile).item()
        return {"min": low, "max": high}

           
    for name, module in model.named_modules():
        if not isinstance(module, (LoF_Linear, LoF_Conv2d)):
            continue

         # Weight min/max
        bias_minmax[name]  = percentile_range(module.bias.detach())
        weights_minmax[name]     = percentile_range(module.weight.detach())
        activations_minmax[name] = percentile_range(captured_activations[name])

    return weights_minmax, activations_minmax, bias_minmax


def find_exp_bits_and_bias(weights_minmax, activations_minmax, bias_minmax):

    weights_exp_bits = {}
    weights_bias = {}
    activations_exp_bits = {}
    activations_bias = {}
    bias_exp_bits = {}
    bias_bias = {}

    for name, val_range in weights_minmax:
        _, min_exp = math.frexp(val_range["min"])
        _, max_exp = math.frexp(val_range["max"])
        exp_range = max_exp - min_exp
        weights_exp_bits[name] = math.ceil(math.log2(exp_range))
        weights_bias[name] = -min_exp 

    for name, val_range in activations_minmax:
        _, min_exp = math.frexp(val_range["min"])
        _, max_exp = math.frexp(val_range["max"])
        exp_range = max_exp - min_exp
        activations_exp_bits[name] = math.ceil(math.log2(exp_range))
        activations_bias[name] = -min_exp 

    for name, val_range in bias_minmax:
        _, min_exp = math.frexp(val_range["min"])
        _, max_exp = math.frexp(val_range["max"])
        exp_range = max_exp - min_exp
        bias_exp_bits[name] = math.ceil(math.log2(exp_range))
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

def hess_sensitivity(model, dataset, loss_fn, n_samples=256, device='cpu'):

    weight_sensitivity = {}
    activation_sensitivity = {}
    bias_sensitivity = {}          # <-- new

    model.eval()
    model.to(device)

    calib_data, calib_labels = make_calib_data(dataset=dataset, n_samples=n_samples)
    calib_data, calib_labels = calib_data.to(device), calib_labels.to(device)

    with torch.no_grad():
        baseline_loss = loss_fn(model(calib_data), calib_labels).item()

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

        weight_traces = {}
        bias_traces = {}           
        for param_name, param in module.named_parameters(recurse=False):
            if param is None or not param.requires_grad:
                continue
            with _bypass_quantization(module):
                loss = loss_fn(model(calib_data), calib_labels)
            trace = batch_hutchinson_approx(
                model, loss, param,
                n_samples=n_samples,
                retain_graph=False
            )
            if 'bias' in param_name:           
                bias_traces[param_name] = trace 
            else:
                weight_traces[param_name] = trace
        weight_sensitivity[name] = weight_traces
        bias_sensitivity[name] = bias_traces   

        act = captured_activations.get(name)
        if act is not None:
            act_leaf = act.detach().requires_grad_(True)

            def make_inject_hook(a):
                def hook(mod, input, output):
                    if isinstance(mod, LoF_Linear):
                        return torch.nn.functional.linear(a, mod.weight, mod.bias_)
                    elif isinstance(mod, LoF_Conv2d):
                        return torch.nn.functional.conv2d(
                            a, mod.weight, mod.bias_,
                            mod.stride, mod.padding, mod.dilation, mod.groups
                        )
                return hook

            inject_hook = module.register_forward_hook(make_inject_hook(act_leaf))
            act_loss = loss_fn(model(calib_data), calib_labels)
            inject_hook.remove()

            act_trace = batch_hutchinson_approx(
                model, act_loss, act_leaf,
                n_samples=n_samples,
                retain_graph=False
            )
            activation_sensitivity[name] = act_trace

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
        baseline_loss = loss_fn(model(calib_data), calib_labels).item()

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
            weight_sensitivity[name] = baseline_loss - loss_fn(model(calib_data), calib_labels).item()
            module.weight.sub_(noise)

            # --- Bias sensitivity ---             
            if module.bias_ is not None:
                B = module.bias_
                b_sigma = B.abs().max().item()
                b_noise = torch.randn_like(B) * b_sigma
                module.bias_.add_(b_noise)
                bias_sensitivity[name] = baseline_loss - loss_fn(model(calib_data), calib_labels).item()
                module.bias_.sub_(b_noise)

            # --- Activation sensitivity ---
            A = captured_activations[name]
            a_sigma = A.abs().max().item()

            def make_noise_hook(sigma):
                def hook(module, input, output):
                    noisy_input = input[0] + torch.randn_like(input[0]) * sigma
                    if isinstance(module, LoF_Linear):
                        return torch.matmul(noisy_input, module.weight.t()) + (module.bias_ if module.bias_ is not None else 0)
                    elif isinstance(module, LoF_Conv2d):
                        return F.conv2d(noisy_input, module.weight, module.bias_,
                                        stride=module.stride, padding=module.padding,
                                        dilation=module.dilation, groups=module.groups)
                return hook

            noise_hook = module.register_forward_hook(make_noise_hook(a_sigma))
            activation_sensitivity[name] = baseline_loss - loss_fn(model(calib_data), calib_labels).item()
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

test_hess_sensitivity_quantized_mlp()