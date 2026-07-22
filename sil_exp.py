import numpy as np

def silu(x):
    return x / (1 + np.exp(-x))

def rel_sens(x):
    s = 1 / (1 + np.exp(-x))
    return abs(x * s * (1 + x * (1 - s)) / (x * s))

np.random.seed(42)
n = 100
SCALE = 1e7

x = np.full(n, 1.0 / n, dtype=np.float32)

eps = 1e-7
W_noise = (np.random.randn(n, n) * eps).astype(np.float32)

for c, label in [
    (-12, "c = -1.278  (sensitivity min, |x·f′/f| ≈ 0.00)"),
    (+1.278, "c = +1.278  (sensitivity > 1,  |x·f′/f| ≈ 1.28)"),
]:
    W_base      = np.full((n, n), c, dtype=np.float32)
    W_perturbed = W_base + W_noise

    out_base = silu(W_base      @ x)
    out_pert = silu(W_perturbed @ x)

    diff_norm = np.linalg.norm(out_pert - out_base)/np.linalg.norm(out_base)

    s = 1 / (1 + np.exp(-c))
    fp = s * (1 + c * (1 - s))
    f  = c * s
    xffpf = c * fp / f

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  x·f′(x)/f(x):                 {xffpf:.5f}")
    print(f"  |x·f′(x)/f(x)|:               {abs(xffpf):.5f}")
    print(f"  ||out_perturbed - out_base||:  {diff_norm:.6e}")
    print(f"  scaled by 1e7:                 {diff_norm * SCALE:.4f}")