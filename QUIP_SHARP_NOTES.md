# QuIP#: Implementation Reference

**Paper:** "QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks"  
**Authors:** Albert Tseng, Jerry Chee, Qingyao Sun, Volodymyr Kuleshov, Christopher De Sa  
**Venue:** ICML 2024  
**arXiv:** 2402.04396

---

## Overview

QuIP# is a post-training quantization (PTQ) method for compressing LLM weights to ≤4 bits per weight. The pipeline consists of:

1. **Incoherence processing**: Randomized Hadamard Transform (RHT) applied to weight matrices via random sign vectors and Hadamard matrices
2. **Vector quantization**: E8 lattice-based codebooks (E8P) that achieve optimal 8-dimensional unit ball packing
3. **Adaptive rounding**: BlockLDLQ algorithm that minimizes quantization error via block-wise LDL decomposition of the Hessian
4. **Fine-tuning**: Block-level and model-level optimization on small calibration data
5. **Residual VQ**: Enables 3-bit and 4-bit compression via sequential quantization of residuals

---

## The E8 Lattice and E8P Codebook

### E8 Lattice Definition

The E8 lattice is formally defined as all-integer or all-half-integer vectors in ℝ⁸ whose sum is even:

$$E_8 = (\mathbb{Z}^8 \cup (\mathbb{Z}^8 + 1/2)) \cap \{x \mid \mathbf{1}^T x \text{ is even}\}$$

The construction leverages the **D̂₈ half-integer lattice** (dense subset of E8):

$$\hat{D}_8 = \{x \in \mathbb{Z}^8 + 1/2 \mid \mathbf{1}^T x \text{ is even}\}$$

Key property: The D̂₈ lattice has optimal packing density in 8 dimensions and exhibits symmetries useful for codebook compression.

### E8P Codebook Construction

The E8P codebook is a 16-bit codeword that encodes 2^16 = 65,536 lattice points, structured as:

- **8 bits**: Index into the *source codebook* S
- **7 bits**: Sign-flip pattern (indicating which coordinates to flip signs)
- **1 bit**: ±1/4 residual shift

**Source Codebook (256 entries):**
- 227 elements of D̂₈ with norm ≤ √10
- 29 padding elements with norm √12
- Total: 256 base codebook vectors

**Sign Symmetry Trick:**
Flipping any (nonzero) even number of signs of an element in D̂₈ yields another distinct element in D̂₈. This allows compression of a much larger effective codebook down to 256 base entries via 2^7 sign-flip patterns.

**Encoding Process:**

Given an 8-dimensional weight vector **w** ∈ ℝ⁸:

1. Normalize: **w̃** = **w** / σ (where σ is a per-block scale)
2. Find nearest D̂₈ lattice point **ŵ** to **w̃** via sphere decoding or nearest-neighbor search
3. Extract sign pattern from **ŵ** → encode as 7-bit sign-flip index
4. Find base codeword **s** ∈ S matching the magnitude structure of **ŵ** → encode as 8-bit index
5. Encode residual shift (±1/4) as 1 bit if used
6. Concatenate: **code** = [8-bit index, 7-bit signs, 1-bit shift] = 16 bits total

**Quantization Error Bound:**
For an 8-dimensional block, the expected quantization error satisfies:

$$\mathbb{E}[\|\hat{\mathbf{w}} - \mathbf{w}\|^2] \leq C \cdot \sigma^2$$

where C is a small constant determined by the lattice packing radius (≈ 0.25 for E8).

### Bit Budget and Compression Ratio

- **2-bit per weight**: One 8-dimensional vector quantized to 16-bit codeword
  - Bits per dimension: 16/8 = 2 bits
  - Compression: 32 → 16 bits per 8 weights (2×)

- **4-bit per weight**: Two 16-bit codewords per 8-dimensional block (via Residual VQ)
  - First codebook (primary): 16 bits → 2-bit quantization
  - Second codebook (residual): 16 bits → 2-bit residual quantization
  - Total: 32 bits per 8 weights (4×)

---

## Incoherence Processing via Randomized Hadamard Transform (RHT)

### Motivation

Lattice quantizers perform best on sub-Gaussian vectors with bounded incoherence (maximum norm bounded by log factor of dimension). Raw LLM weight matrices have high incoherence due to outliers and non-uniform structure. RHT preprocesses weights to achieve approximate incoherence.

### RHT Formulation

**Transform Application:**

For a weight matrix **W** ∈ ℝᵐˣⁿ and Hessian **H** ∈ ℝⁿˣⁿ:

1. Sample random sign vectors: **S_U** ∈ {±1}ᵐ, **S_V** ∈ {±1}ⁿ (Rademacher distributions)
2. Apply diagonal sign scaling: **Ŵ** ← diag(**S_U**) **W** diag(**S_V**)
3. Apply Hadamard transform on both sides:
   $$\hat{\mathbf{W}} \leftarrow \text{Had}(\text{diag}(\mathbf{S}_U) \text{Had}(\text{diag}(\mathbf{S}_V) \mathbf{W}^T)^T)$$

   where Had() denotes the Fast Walsh-Hadamard Transform (FWHT).

**Hessian Transform:**

The Hessian is similarly transformed:
$$\hat{\mathbf{H}} \leftarrow \text{Had}(\text{diag}(\mathbf{S}_V) \text{Had}(\mathbf{H})^T)$$

This ensures the proxy loss computed on transformed weights matches the original space loss.

### Computational Complexity

- Standard Kronecker factorization (QuIP): O(n²) or O(n² log n)
- RHT via FWHT: O(n log n) per side
- Total RHT cost: O((m + n) log n) for m×n matrix

### Theoretical Guarantees

The RHT achieves incoherence parameter:

$$\mu_n = 2\log(2n^2/\delta)$$

with high probability 1-δ, where μ_n bounds the maximum entry norm relative to the Frobenius norm. This is better than Kronecker's log²(n) dependence.

### Runtime Storage

Sign vectors **S_U**, **S_V** are stored as FP16:
- Storage: 2 × n floats (FP16) ≈ <0.01 bits per weight for typical LLM dimensions
- Applied at inference time via element-wise multiplication before/after decoding

---

## BlockLDLQ: Adaptive Rounding with Hessian Feedback

### Quantization Loss Function

The per-layer loss minimized is:

$$\ell(\hat{\mathbf{W}}) = \mathbb{E}_{\mathbf{x}}[(\hat{\mathbf{W}} - \mathbf{W})\mathbf{x}]^2 = \text{tr}((\hat{\mathbf{W}} - \mathbf{W})\mathbf{H}(\hat{\mathbf{W}} - \mathbf{W})^T)$$

where **H** = E[**x****x**^T] is the calibration Hessian (empirical second moment of activations).

### Block LDL Decomposition

Instead of standard LDL decomposition, use **g-block LDL** where:

$$\mathbf{H} = \mathbf{L}^T \mathbf{D} \mathbf{L}$$

where:
- **L** is a unit block lower triangular matrix (block size g)
- **D** is block diagonal

This allows processing the weight matrix **W** in blocks of g rows/columns, with linear feedback from previously quantized blocks.

### BlockLDLQ Algorithm

**Pseudocode:**

```
Input: Weight matrix W ∈ ℝ^(m×n), Hessian H
Output: Quantized weights Ŵ, per-block scales σ

// Precompute block LDL decomposition
L, D = block_ldl_decompose(H, block_size=g)

// Initialize
Ŵ ← zeros_like(W)
for k in 1 to (n / g):
    block_start = (k-1) * g
    block_end = k * g
    
    // Extract block and feedback from previous blocks
    W_k = W[:, block_start:block_end]
    if k > 1:
        feedback = (W[:, :block_start] - Ŵ[:, :block_start]) @ A_k
    else:
        feedback = 0
    
    // Compute target for quantization
    W_k_target = W_k + feedback
    
    // Per-block scaling (optional: per-vector or global)
    σ_k = estimate_scale(W_k_target, D[block_start:block_end])
    
    // Vector quantize block to E8 lattice
    for i in 0 to (m / 8):
        w_8d = W_k_target[i*8:(i+1)*8, :].T  // 8-dimensional vector
        ŵ_8d, code_i = vq_e8p(w_8d / σ_k)
        Ŵ[i*8:(i+1)*8, block_start:block_end] = ŵ_8d * σ_k
        store_codebook(code_i)
```

### Error Bound

The expected quantization error per block is bounded by:

$$\mathbb{E}[\text{tr}((\hat{\mathbf{W}} - \mathbf{W})\mathbf{H}(\hat{\mathbf{W}} - \mathbf{W})^T)] \leq \frac{g \cdot m \cdot \mu^2 \cdot \sigma^2}{n} \text{tr}(\mathbf{H}^{1/2})^2$$

where μ is the incoherence parameter and σ² is the per-block variance.

---

## Vector Quantization: E8P Encoding/Decoding

### Encoding (Forward Pass - Quantization)

**Input:** 8-dimensional vector **w** ∈ ℝ⁸  
**Output:** 16-bit codeword c, scale σ

```
def vq_e8p_encode(w, sigma):
    // Normalize by scale
    w_normalized = w / sigma
    
    // Sphere decode to find nearest D̂₈ lattice point
    w_lattice = sphere_decode(w_normalized, D_hat_8)
    
    // Extract sign pattern (7 bits)
    signs = extract_signs(w_lattice)
    sign_index = encode_sign_pattern(signs)  // 7 bits
    
    // Find base codeword index (8 bits)
    magnitude_structure = abs(w_lattice)
    base_index = nearest_in_codebook_S(magnitude_structure)
    
    // Optional: encode residual shift (1 bit)
    residual = w_normalized - w_lattice
    shift_bit = 1 if |residual| > 0.125 else 0
    
    // Concatenate into 16-bit codeword
    code = (base_index << 8) | (sign_index << 1) | shift_bit
    
    return code, sigma
```

**Sphere Decoding (Simplified):**
Use nearest-neighbor search or Schnorr-Euchner sphere decoder:
1. Compute all lattice points within Euclidean ball of radius R around **w̃**
2. Evaluate distance to each candidate
3. Return minimum-distance point

For E8, precomputed tables can make this O(1) with small lookup overhead.

### Decoding (Inference - Decompression)

**Input:** 16-bit codeword c, scale σ, input activation vector **x** ∈ ℝⁿ  
**Output:** Quantized result **y** ∈ ℝⁿ

```
def vq_e8p_decode(code, sigma, x, S_V, S_U):
    // Decode codeword
    base_index = (code >> 8) & 0xFF
    sign_index = (code >> 1) & 0x7F
    shift_bit = code & 1
    
    // Reconstruct lattice point
    s = S[base_index]                    // base codebook lookup
    signs = decode_sign_pattern(sign_index)
    w_lattice = signs * s
    if shift_bit:
        w_lattice += 0.25 * ones_like(w_lattice)
    
    // Scale back
    w_quantized = w_lattice * sigma
    
    return w_quantized
```

**Full Forward Pass (Inference Procedure):**

```
def forward_quantized(x, W_codes, scales, S_V, S_U):
    // Apply RHT to input
    y = hadamard_fwht(S_V * x)
    
    // Apply quantized weight matrix via table lookups
    y = decompress_multiply(W_codes, scales, y)
    
    // Reverse RHT on output
    y = hadamard_fwht(S_U * y)
    
    return y
```

where `decompress_multiply` performs the lattice codebook lookup and accumulated matrix-vector product.

---

## Scaling Strategy

### Per-Block Scaling

Each 8-dimensional block is assigned a scalar scale σ:

$$\sigma = \text{median}(\mathbf{w}) \text{ or } \text{std}(\mathbf{w}) \text{ or argmin}_\sigma \mathbb{E}[\|\hat{\mathbf{w}} - \mathbf{w}\|^2]$$

**Storage:** 1 FP16 value per 8-dimensional block
- For a 4096-dimensional layer: 512 scale values ≈ 1KB
- Negligible overhead relative to weights

### Quantization-Aware Scaling

Scale parameters can be optimized jointly with sign vectors during fine-tuning (treated as real-valued, not constrained to fixed values).

---

## Fine-Tuning Strategy

### Two-Stage Approach

**Stage 1: Block-Level Fine-Tuning**

For each layer k in sequence:
1. Quantize layer k using BlockLDLQ
2. Fix Ŵ_k (codebook indices)
3. Optimize remaining unquantized layers {W_{k+1}, ..., W_L} and all layernorms to minimize proxy loss on validation calibration data
4. Update {**S_V**, **S_U**, scales} of quantized layer k to reduce error
5. Move to next layer

Loss function per block:
$$\mathcal{L}_k = \mathbb{E}[\|(\hat{\mathbf{W}}_k - \mathbf{W}_k)\mathbf{H}_k(\hat{\mathbf{W}}_k - \mathbf{W}_k)^T\| + \text{downstream error}]$$

**Stage 2: Model-Level Fine-Tuning**

After all layers quantized:
1. Jointly optimize all sign vectors **S_U**, **S_V** (as real-valued, not constrained to ±1)
2. Optimize layernorm parameters and scales across all layers
3. Minimize full-model loss (e.g., next-token prediction on small validation set)

**Typical Hyperparameters:**
- Learning rate: 1e-3 to 1e-2 (adaptive via Adam)
- Calibration data size: 128 sequences of length 2048
- Fine-tune steps: 100-500 per layer, 1000-5000 for model-level
- Regularization: Optional L2 on scale parameters to prevent instability

---

## Residual Vector Quantization (RVQ) for Higher Bitrates

### 3-Bit Variant

Decompose into two sequential quantization passes:

$$\mathbf{w} = \mathbf{w}_1 + \mathbf{r}_1$$

where:
- **w₁**: Quantized to 2-bit E8P codebook (16-bit code)
- **r₁** = **w** - **w₁**: Residual, quantized to 1-bit via thresholding
  - 1-bit: sign of each dimension (8 bits total)
  
**Total: 16 + 8 = 24 bits per 8-d block → 3 bits per weight**

### 4-Bit Variant

Two full 2-bit quantizations:

$$\mathbf{w} = \mathbf{w}_1 + \mathbf{r}_1 = \mathbf{w}_1 + \mathbf{w}_2 + \mathbf{r}_2$$

where:
- **w₁**: 2-bit E8P (16-bit code)
- **w₂**: 2-bit E8P on residual **r₁** (16-bit code)
- **r₂**: Discarded (acceptable error for 4-bit)

**Total: 16 + 16 = 32 bits per 8-d block → 4 bits per weight**

### RVQ Encoding

```
def vq_e8p_rvq_encode(w, depth=2):
    codes = []
    w_accum = 0
    
    for d in range(depth):
        residual = w - w_accum
        code_d, sigma_d = vq_e8p_encode(residual, estimate_scale(residual))
        codes.append(code_d)
        
        w_d, _ = vq_e8p_decode(code_d, sigma_d)
        w_accum += w_d
    
    return codes  // List of codes, concatenated for storage
```

---

## Quantization Procedure Summary

### Full Pipeline

1. **Incoherence Processing:**
   - Sample **S_U**, **S_V** ∈ {±1}^{m,n} (Rademacher)
   - Transform **W̃** ← Had(diag(**S_U**) Had(diag(**S_V**) **W**^T)^T)
   - Transform **H̃** ← Had(diag(**S_V**) Had(**H**)^T)

2. **Compute Hessian:**
   - Calibration pass on 128+ examples
   - **H** = E[**x****x**^T] (or generalized Gauss-Newton)

3. **Block LDL Decomposition:**
   - Decompose **H̃** = **L**^T **D** **L** (g-block structure, g=128 typical)

4. **BlockLDLQ Quantization:**
   - For each block of g dimensions:
     - Estimate per-block scale σ
     - Vector quantize 8-d sub-blocks to E8P lattice
     - Store codebook indices and scales

5. **Fine-Tuning:**
   - Block-level: optimize downstream layers and scales
   - Model-level: optimize sign vectors and layernorms jointly

6. **RVQ (if 3+ bits):**
   - For 3-bit: add 1-bit sign quantization on residuals
   - For 4-bit: apply second 2-bit E8P to residuals

---

## Key Implementation Details

### Hadamard Transform

Use **Fast Walsh-Hadamard Transform (FWHT)** for O(n log n) complexity:

```
def hadamard_fwht(x):
    """
    In-place or out-of-place FWHT on vector x ∈ ℝ^n (n = 2^k).
    Scales by 1/sqrt(n) for orthonormality.
    """
    n = len(x)
    result = x.copy()
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            for j in range(i, i + h):
                temp = result[j]
                result[j] += result[j + h]
                result[j + h] = temp - result[j + h]
        h *= 2
    return result / sqrt(n)
```

### Sphere Decoding for E8

Precompute all lattice points in D̂₈ within norm ≤ √12 (the effective codebook radius). For each input vector **w̃**:

1. Compute nearest-neighbor search via KD-tree or exhaustive lookup (256 base entries × 128 sign patterns ≈ 32k candidates, manageable)
2. Or use standard Schnorr-Euchner decoder with lattice basis vectors

### Scale Estimation

Robust per-block scale:

```
def estimate_scale(w, diagonal_H):
    """
    Estimate optimal scale σ minimizing:
    E[||w̃ - closest_E8_point||^2] where w̃ = w / σ
    """
    // Heuristic: use median-absolute-deviation or robust std
    scale = median(abs(w)) / quantile_normal(0.68)  // robust MAD
    // Or: jointly optimize σ during fine-tuning
    return scale
```

### Storage Layout

**Weight storage per layer:**
- Codebook indices: uint16 array, shape (m/8 blocks, n/g block-rows)
- Per-vector scales: float16 array, shape (m/8,)
- Sign vectors: **S_U**, **S_V** as float16, shape (m,) and (n,)

**Total per-layer overhead:**
- m×n weights: 16 bits/weight → 2m×n bytes
- m/8 scales: 2 bytes each → m/4 bytes
- m+n sign vectors: 2 bytes each → 2(m+n) bytes
- Sign vector overhead: <0.01 bits/weight for 4096+ dimensions

---

## Experimental Results Summary

### Performance (Llama 2 7B, WikiText2, context 2048)

| Bitrate | QuIP# | OmniQuant | AQLM |
|---------|--------|-----------|------|
| 2-bit | **4.16** | 7.81 | - |
| 3-bit | **3.56** | 3.92 | - |
| 4-bit | **3.38** | - | 3.36 |

**Inference Speed (RTX 4090):**
- 2-7B model: 170.5 tokens/sec
- 2-70B model: 32.74 tokens/sec
- Memory bandwidth utilization: >50% of peak (1008 GB/s)

### Ablation Analysis

- **RHT vs. Kronecker:** ~0.5 PPL improvement at 2-bit
- **E8P vs. scalar VQ:** ~0.1-0.4 PPL improvement
- **Fine-tuning:** 0.2-1.0 PPL improvement (2-bit to 4-bit)
- **BlockLDLQ vs. LDLQ:** Modest improvement in stability (fewer outliers)

---

## References & Related Work

- **E8 lattice references:** Conway & Sloane, *Sphere Packings, Lattices, and Groups*
- **Sphere decoding:** Schnorr & Euchner; Guo & Nilsson
- **LDLQ foundation:** Carbin et al. (original LDLQ for quantization)
- **QuIP (predecessor):** Tseng et al., prior work using Kronecker factorization
- **GPTQ:** Frantar et al., weight quantization baseline
- **AQLM:** Adelkamp et al., vector quantization baseline

---

## Notes for Implementation

1. **Dimension alignment:** Ensure model dimensions are divisible by 8 for E8 codebook. Pad if necessary.
2. **Hessian computation:** Use activation statistics from calibration set (128 examples, ~2K tokens each typical).
3. **Sign vector initialization:** Uniform random ±1 is sufficient; no special warm-start needed.
4. **Codebook storage:** Consider memory layout optimization for GPU (coalesced access on quantized weights).
5. **Fine-tuning convergence:** Monitor validation perplexity; typical plateau within 500 block-level updates per layer.
6. **4-bit coverage:** Test both single 4-bit E8P and 2+2-bit RVQ to find optimal quality vs. inference speed tradeoff.
