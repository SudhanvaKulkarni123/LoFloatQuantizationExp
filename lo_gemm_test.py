import copy
import math
import torch
import torch.nn as nn
import pytest
import LoFloat as lof
from LoFloat import LoF_Linear, LoF_Conv2d, L1BatchNorm



@pytest.fixture
def lin_kwargs():
    return dict(act_exp=4, act_mant=3,
                weight_exp=4, weight_mant=3,
                bias_exp=4, bias_mant=3)

@pytest.fixture
def conv_kwargs():
    return dict(act_exp=4, act_mant=3,
                weight_exp=4, weight_mant=3,
                bias_exp=4, bias_mant=3)


# ------------------------------------------------------------------
# LoF_Linear
# ------------------------------------------------------------------
class TestLoFLinear:
    def test_forward_shape(self, lin_kwargs):
        layer = LoF_Linear(16, 8, **lin_kwargs)
        x = torch.randn(4, 16)
        x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
        layer = layer.to(x.device)
        y = layer(x)
        assert y.shape == (4, 8)
        assert torch.isfinite(y).all()

    def test_backward_runs_without_error(self, lin_kwargs):
        # Note: custom lof_gemm doesn't propagate grad to input x.
        # This just checks .backward() doesn't raise.
        layer = LoF_Linear(16, 8, **lin_kwargs).to('cuda' if torch.cuda.is_available() else 'cpu')
        x = torch.randn(4, 16, requires_grad=True).to('cuda' if torch.cuda.is_available() else 'cpu')
        y = layer(x)
        y.sum().backward()  # should not raise

    def test_from_linear_preserves_weights(self, lin_kwargs):
        ref = nn.Linear(16, 8)
        params = lof.create_p3109_params(8, 4, True, True)
        layer = LoF_Linear.from_linear(ref, params, params, params)
        assert torch.equal(layer.weight, ref.weight)
        assert torch.equal(layer.bias, ref.bias)

    def test_deepcopy_is_independent(self, lin_kwargs):
        layer = LoF_Linear(16, 8, **lin_kwargs)
        clone = copy.deepcopy(layer)
        assert clone is not layer
        assert torch.equal(clone.weight, layer.weight)
        with torch.no_grad():
            clone.weight.add_(1.0)
        assert not torch.equal(clone.weight, layer.weight)

    def test_setters_update_params(self, lin_kwargs):
        layer = LoF_Linear(16, 8, **lin_kwargs)
        layer.set_mantissa(2, 2, 2)
        # create_p3109_params stores mantissa_bits = (arg - 1) internally,
        # and set_mantissa passes mant+1, so the stored value equals the input.
        assert layer.act_params.mantissa_bits == 2
        assert layer.weight_params.mantissa_bits == 2
        assert layer.bias_params.mantissa_bits == 2

        layer.set_accumulation_precision(10)
        assert layer.accum_mant_bits == 10


# ------------------------------------------------------------------
# LoF_Conv2d
# ------------------------------------------------------------------
class TestLoFConv2d:
    def test_forward_shape(self, conv_kwargs):
        layer = LoF_Conv2d(3, 8, kernel_size=3, padding=1, **conv_kwargs)
        x = torch.randn(2, 3, 16, 16)
        x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
        layer = layer.to(x.device)
        y = layer(x)
        assert y.shape == (2, 8, 16, 16)
        assert torch.isfinite(y).all()

    def test_backward_runs_without_error(self, conv_kwargs):
        layer = LoF_Conv2d(3, 8, kernel_size=3, padding=1, **conv_kwargs)
        x = torch.randn(2, 3, 16, 16, requires_grad=True).to('cuda' if torch.cuda.is_available() else 'cpu')
        layer = layer.to(x.device)
        layer(x).sum().backward()  # should not raise

    def test_strided_output_shape(self, conv_kwargs):
        layer = LoF_Conv2d(3, 4, kernel_size=3, stride=2, padding=1, **conv_kwargs)
        x = torch.randn(1, 3, 8, 8)
        x =x.to('cuda' if torch.cuda.is_available() else 'cpu')
        layer = layer.to(x.device)
        assert layer(x).shape == (1, 4, 4, 4)

    def test_grouped_conv(self, conv_kwargs):
        layer = LoF_Conv2d(8, 8, kernel_size=3, padding=1, groups=4, **conv_kwargs)
        x = torch.randn(2, 8, 10, 10)
        x = x.to('cuda' if torch.cuda.is_available() else 'cpu')
        layer = layer.to(x.device)
        assert layer(x).shape == (2, 8, 10, 10)

    def test_from_conv2d_preserves_weights(self):
        ref = nn.Conv2d(3, 8, kernel_size=3, padding=1)
        params = lof.create_p3109_params(8, 4, True, True)
        layer = LoF_Conv2d.from_conv2d(ref, params, params, params)
        assert torch.equal(layer.weight, ref.weight)
        assert torch.equal(layer.bias, ref.bias)

    def test_deepcopy(self, conv_kwargs):
        layer = LoF_Conv2d(3, 8, kernel_size=3, padding=1, **conv_kwargs)
        clone = copy.deepcopy(layer)
        assert clone is not layer
        assert torch.equal(clone.weight, layer.weight)


# ------------------------------------------------------------------
# L1BatchNorm
# ------------------------------------------------------------------
class TestL1BatchNorm:
    def test_train_normalizes_to_zero_mean(self):
        bn = L1BatchNorm(16)
        bn.train()
        x = torch.randn(32, 16, 8, 8) * 3.0 + 5.0
        y = bn(x)
        assert y.shape == x.shape
        per_ch_mean = y.mean(dim=[0, 2, 3])
        assert torch.allclose(per_ch_mean, torch.zeros(16), atol=1e-4)

    def test_running_stats_update(self):
        bn = L1BatchNorm(4, momentum=0.5)
        bn.train()
        rm_before = bn.running_mean.clone()
        mad_before = bn.running_mad.clone()
        _ = bn(torch.randn(8, 4, 5, 5) + 2.0)
        assert not torch.equal(bn.running_mean, rm_before)
        assert not torch.equal(bn.running_mad, mad_before)

    def test_eval_uses_running_stats(self):
        bn = L1BatchNorm(4)
        bn.train()
        for _ in range(5):
            bn(torch.randn(8, 4, 5, 5))
        bn.eval()
        x = torch.randn(2, 4, 5, 5)
        assert torch.equal(bn(x), bn(x))

    def test_2d_input(self):
        bn = L1BatchNorm(10)
        assert bn(torch.randn(32, 10)).shape == (32, 10)

    def test_grad_flows(self):
        bn = L1BatchNorm(4)
        x = torch.randn(8, 4, 5, 5, requires_grad=True)
        bn(x).sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()
        assert bn.weight.grad is not None
        assert bn.bias.grad is not None

# ------------------------------------------------------------------
# Debug / diagnostic tests for the nn -> LoF swap
# ------------------------------------------------------------------
class TestLoFSwapCorrectness:
    """
    These tests isolate whether a zero-accuracy failure after swapping
    nn.Linear -> LoF_Linear is caused by:
      (a) a layout/transpose/bias bug in the swap itself (test #1), or
      (b) a bug in lof_gemm or how we're calling it (test #3).

    Runs on CUDA if available, else CPU.
    """

    @staticmethod
    def _device():
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Debug #1 -------------------------------------------------
    def test_lof_linear_matches_nn_linear_at_high_precision(self):
        """
        Build a LoF_Linear from an nn.Linear using a near-fp32 format,
        push the same input through both, and require agreement.

        If this FAILS at high precision, the problem is NOT the low-bit
        format — it's correctness: transpose, reshape, bias placement,
        or a bad lof_gemm argument layout inside _lof_linear.
        """
        torch.manual_seed(0)
        device = self._device()

        ref = nn.Linear(32, 32).to(device)

        # Wide format: effectively a no-op quantization.
        hp = lof.create_p3109_params(16, 10, True, True)
        layer = LoF_Linear.from_linear(
            ref, hp, hp, hp,
            accum_mant_bits=23,
        ).to(device)

        x = torch.randn(32, 32, device=device)
        x = x.to(device)

        y_ref = ref(x)
        y_lof = layer(x)

        assert torch.isfinite(y_lof).all(), (
            "LoF_Linear produced NaN/Inf even at high precision -> "
            "saturation or bad format flags."
        )

        max_abs = (y_ref - y_lof).abs().max().item()
        rel = max_abs / (y_ref.abs().max().item() + 1e-12)
        assert max_abs < 1e-2, (
            f"LoF_Linear disagrees with nn.Linear at near-fp32 precision: "
            f"max |diff| = {max_abs:.4e} (rel {rel:.2e}).\n"
            f"This is a correctness bug in the swap, not a precision issue. "
            f"Prime suspects: _lof_linear's weight.t() layout, the reshape "
            f"back to leading dims, or bias add placement."
        )

    def test_lof_linear_bias_is_applied_correctly(self):
        """
        Subtest of #1: isolate the bias path. With zero weight and non-zero
        bias, output should equal the (unquantized) bias broadcast.
        """
        device = self._device()
        ref = nn.Linear(8, 4).to(device)
        with torch.no_grad():
            ref.weight.zero_()
            ref.bias.copy_(torch.arange(4, dtype=torch.float32, device=device))

        hp = lof.create_p3109_params(16, 10, True, True)
        layer = LoF_Linear.from_linear(ref, hp, hp, hp, accum_mant_bits=23).to(device)

        x = torch.randn(5, 8, device=device)
        y = layer(x)

        expected = ref.bias.expand(5, 4)
        assert torch.allclose(y, expected, atol=1e-3), (
            f"Bias not applied correctly. Got {y[0].tolist()}, "
            f"expected {expected[0].tolist()}."
        )

    # --- Debug #3 -------------------------------------------------
    def test_lof_gemm_matches_torch_matmul(self):
        """
        Direct check: lof_gemm(A, B) with 23-bit accumulation on fp32 inputs
        should reproduce A @ B to within a small tolerance.

        If this fails, the problem is in lof_gemm itself or in how its
        arguments are being laid out (K in the wrong dim, etc.).
        """
        torch.manual_seed(0)
        device = self._device()

        A = torch.randn(80, 160, device=device)
        B = torch.randn(160, 320, device=device)

        ref = A @ B
        got = lof.lof_gemm(
            A.contiguous(), B.contiguous(),
            23, lof.RoundingMode.RoundToNearestEven, 0,
        )

        assert got.shape == ref.shape, (
            f"lof_gemm output shape {tuple(got.shape)} != expected "
            f"{tuple(ref.shape)} -> lof_gemm likely uses a different "
            f"argument convention than (M,K) @ (K,N)."
        )
        assert torch.isfinite(got).all(), "lof_gemm produced NaN/Inf."

        max_abs = (ref - got).abs().max().item()
        assert max_abs < 1e-2, (
            f"lof_gemm(A, B) does not match A @ B: max |diff| = {max_abs:.4e}. "
            f"Try the next test to check if lof_gemm wants B pre-transposed."
        )

    def test_lof_gemm_layout_convention(self):
        """
        Diagnostic: checks whether lof_gemm expects B as (K, N) (standard)
        or as (N, K) (pre-transposed). Exactly one form should match A @ B_kn.

        If the (N, K) form matches and the (K, N) form doesn't, then every
        call site that does `weight.t().contiguous()` before lof_gemm is
        double-transposing. That would explain zero accuracy cleanly.
        """
        torch.manual_seed(0)
        device = self._device()

        M, K, N = 16, 32, 24
        A   = torch.randn(M, K, device=device)
        Bkn = torch.randn(K, N, device=device)          # standard layout
        Bnk = Bkn.t().contiguous()                      # pre-transposed layout

        ref = A @ Bkn   # (M, N)

        got_std = lof.lof_gemm(
            A.contiguous(), Bkn.contiguous(),
            23, lof.RoundingMode.RoundToNearestEven, 0,
        )
        try:
            got_pre = lof.lof_gemm(
                A.contiguous(), Bnk,
                23, lof.RoundingMode.RoundToNearestEven, 0,
            )
        except Exception as e:
            got_pre = e

        std_ok = (
            isinstance(got_std, torch.Tensor)
            and got_std.shape == ref.shape
            and torch.isfinite(got_std).all()
            and (ref - got_std).abs().max().item() < 1e-2
        )
        pre_ok = (
            isinstance(got_pre, torch.Tensor)
            and got_pre.shape == ref.shape
            and torch.isfinite(got_pre).all()
            and (ref - got_pre).abs().max().item() < 1e-2
        )

        msg = (
            f"lof_gemm layout diagnosis:\n"
            f"  (M,K)@(K,N) matches A@B : {std_ok}\n"
            f"  (M,K)@(N,K) matches A@B : {pre_ok}\n"
            f"If only the second is True, drop the .t() in _lof_linear and "
            f"pass `weight` directly (shape (N, K)) instead of `weight.t()`."
        )
        assert std_ok, msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    A = torch.randn(80, 160, device='cuda')        # non-square M, K
    B = torch.randn(160, 320, device='cuda')       # K, N with K != N
    ref = A @ B
    got = lof.lof_gemm(A, B.contiguous(), 23, lof.RoundingMode.RoundToNearestEven, 0)
    print("max abs diff:", (ref - got).abs().max().item())
    # If this is huge, the transpose bug is real.
# If you do `B.t().contiguous()` and it matches, bug confirmed.