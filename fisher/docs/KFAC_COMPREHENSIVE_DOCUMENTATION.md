# KFAC Comprehensive Documentation

**Kronecker-Factored Approximate Curvature (KFAC) Implementation**

## Table of Contents

1. [Theory and Background](#theory-and-background)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Implementation Details](#implementation-details)
4. [Woodbury Matrix Identity](#woodbury-matrix-identity)
5. [Numerical Precision and Stability](#numerical-precision-and-stability)
6. [Memory Efficiency](#memory-efficiency)
7. [Distributed Operations](#distributed-operations)
8. [API Reference](#api-reference)
9. [Configuration Parameters](#configuration-parameters)
10. [Best Practices](#best-practices)
11. [Troubleshooting](#troubleshooting)
12. [References](#references)

---

## Theory and Background

### What is KFAC?

KFAC (Kronecker-Factored Approximate Curvature) is a second-order optimization method that approximates the Fisher Information Matrix (FIM) using a Kronecker product structure. This allows for efficient computation of natural gradients without the prohibitive memory and computational costs of full second-order methods.

### Key Concepts

- **Natural Gradient**: ∇_nat = F^(-1) * ∇ (where F is the Fisher Information Matrix)
- **KFAC Approximation**: F ≈ A ⊗ G (Kronecker product of activation and gradient covariances)
- **Block-Diagonal Structure**: Captures more interactions than diagonal Fisher while remaining tractable

### Why KFAC?

1. **Computational Efficiency**: O(d²) instead of O(d⁴) for full Fisher
2. **Memory Efficiency**: Store factors instead of full matrices
3. **Numerical Stability**: Better conditioning than naive approximations
4. **Theoretical Foundation**: Well-grounded in information geometry

---

## Mathematical Foundation

### Fisher Information Matrix

For a neural network with parameters θ, the Fisher Information Matrix is:

```
F = E[∇_θ log p(y|x,θ) ∇_θ log p(y|x,θ)^T]
```

### KFAC Approximation

KFAC approximates the Fisher as a block-diagonal matrix where each block is a Kronecker product:

```
F ≈ diag(A₁ ⊗ G₁, A₂ ⊗ G₂, ..., Aₙ ⊗ Gₙ)
```

Where:
- **Aᵢ**: Activation covariance matrix for layer i
- **Gᵢ**: Gradient covariance matrix for layer i
- **⊗**: Kronecker product

### Natural Gradient Computation

The natural gradient is computed as:

```
∇_nat = (A ⊗ G + λI)^(-1) * ∇
```

This can be efficiently computed using the eigendecomposition:

```
∇_nat = V_G * diag(1/(λ_G + λ)) * V_G^T * ∇ * V_A * diag(1/(λ_A + λ)) * V_A^T
```

Where:
- **V_A, λ_A**: Eigenvectors and eigenvalues of A
- **V_G, λ_G**: Eigenvectors and eigenvalues of G
- **λ**: Damping parameter

---

## Implementation Details

### Core Architecture

The `KFACNaturalGradient` class provides a unified interface for KFAC computation with the following key components:

1. **Factor Collection**: `collect_kfac_factors()` - Gathers activation and gradient statistics
2. **Natural Gradient**: `compute_natural_gradient()` - Applies KFAC preconditioning
3. **Fisher-Vector Product**: `compute_fisher_vector_product()` - For trust region methods
4. **Powered Scaling**: `_compute_powered_natural_gradient()` - For fractional powers

### Factor Storage Schema

```python
kfac_factors = {
    layer_name: {
        # A-side (always eigendecomp)
        'A_eigvecs': torch.Tensor,    # [in_dim, in_dim], float32, CPU
        'A_eigvals': torch.Tensor,    # [in_dim], float32, CPU
        'A_bias_augmented': bool,     # Whether bias was included
        
        # G-side (either eigendecomp or Woodbury)
        'G_type': str,                # 'eig' | 'woodbury_empirical' | 'woodbury_true'
        
        # If G_type == 'eig':
        'G_eigvecs': torch.Tensor,    # [out_dim, out_dim], float32, CPU
        'G_eigvals': torch.Tensor,    # [out_dim], float32, CPU
        
        # If G_type == 'woodbury_empirical':
        'U': torch.Tensor,            # [out_dim, T], fp16/bf16, device per policy
        'S_inv': torch.Tensor,        # [T, T], float32, device per policy
        'lambda_G': float,            # Damping parameter
        'T_effective': int            # Number of effective tokens
    }
}
```

### Hook-Based Data Collection

The implementation uses PyTorch hooks to collect activations and gradients:

```python
# Forward hook: Collect activations
def save_input_hook(name):
    def hook(module, input, output):
        # Extract input, flatten batch dims, add bias term
        # Apply attention masking, offload to CPU
        activations[name] = processed_activation

# Backward hook: Collect gradients  
def save_grad_hook(name):
    def hook(module, grad_input, grad_output):
        # Extract gradient, flatten batch dims
        # Apply attention masking, offload to CPU
        gradients[name] = processed_gradient
```

---

## Woodbury Matrix Identity

### Theory

For empirical Fisher, the gradient covariance G has low rank structure:

```
G = (1/T) * Σᵢ gᵢ gᵢ^T = U U^T
```

Where U = [g₁, g₂, ..., gₜ] / √T is the stacked gradients.

The Woodbury identity allows efficient inversion:

```
(G + λI)^(-1) = (1/λ)I - (1/λ)U S^(-1) U^T (1/λ)
```

Where S = I + (1/λ)U^T U is a T×T matrix.

### Implementation

```python
# Build U matrix: [out_dim, T]
U = (G_tokens.t().contiguous() / sqrt_T).to(dtype=fp16)

# Build S matrix: [T, T]  
S = (U.t() @ U) / lambda_G + I_T

# Invert S via Cholesky with jitter backoff
for attempt in range(3):
    try:
        L = torch.linalg.cholesky(S)
        S_inv = torch.cholesky_inverse(L)
        break
    except RuntimeError:
        S.diagonal().add_(eps)
        eps *= 10.0

# Apply Woodbury formula
lambda_inv = 1.0 / lambda_G
Y0 = lambda_inv * Y
Z = U.t() @ Y0
W = S_inv @ Z  
Y_nat = Y0 - lambda_inv * (U @ W)
```

### Memory Efficiency

Woodbury provides significant memory savings:

- **Traditional**: O(out_dim²) for G matrix
- **Woodbury**: O(out_dim × T + T²) for U and S_inv
- **Savings**: Up to 1000x for large output dimensions

### Shape Validation

Critical shape requirements:

- **U**: `[out_dim, T]` - Each column is a scaled gradient
- **S_inv**: `[T, T]` - Inverted S matrix
- **G_tokens**: `[T, out_dim]` - Original gradient matrix

---

## Numerical Precision and Stability

### Damping Strategy

KFAC uses Tikhonov regularization to ensure positive definiteness:

```python
# A-side damping
A_damped = A + λ_A * I

# G-side damping  
G_damped = G + λ_G * I
```

### Eigenvalue Clipping

To prevent numerical issues, eigenvalues are clipped:

```python
max_eig = eigvals.max()
min_allowed = max_eig / max_condition_number
eigvals_clipped = torch.clamp(eigvals, min=min_allowed)
```

### Condition Number Limits

- **Default**: κ_max = 1e6 (6 orders of magnitude)
- **Rationale**: FP32 has ~7 decimal digits, leaving safety margin
- **Effect**: Prevents inversion instability while preserving natural gradient quality

### Jitter Backoff

For Cholesky decomposition failures:

```python
eps = 1e-6
for attempt in range(3):
    try:
        L = torch.linalg.cholesky(S)
        S_inv = torch.cholesky_inverse(L)
        break
    except RuntimeError:
        S.diagonal().add_(eps)
        eps *= 10.0
```

### Precision Considerations

- **A-side**: Always computed in FP32 for stability
- **G-side (Woodbury)**: U in FP16/BF16, S_inv in FP32
- **Natural gradient**: Computed in FP32, cast back to original dtype

---

## Memory Efficiency

### Storage Policies

```python
# Auto policy: GPU if < 500MB, CPU otherwise
if total_mb < 500:
    store_device = torch.device('cuda')
else:
    store_device = torch.device('cpu')

# Manual policies
woodbury_store_device = "cuda" | "cpu" | "auto"
```

### Memory Usage Analysis

For a layer with out_dim=4096, T=1024:

- **Traditional G**: 4096² × 4B = 67MB
- **Woodbury U**: 4096 × 1024 × 2B = 8MB  
- **Woodbury S_inv**: 1024² × 4B = 4MB
- **Total Woodbury**: 12MB (5.6x reduction)

### CPU Offloading

Activations and gradients are immediately offloaded to CPU:

```python
# Offload activations to CPU immediately
if act.is_cuda:
    act = act.to(device='cpu')
activations[name] = act.to(dtype=torch.float32, copy=False)
```

### Cache Management

```python
# Periodic CUDA cache cleanup
if torch.cuda.is_available():
    if idx % 4 == 0:
        torch.cuda.empty_cache()
```

---

## Distributed Operations

### DDP/FSDP Support

KFAC supports distributed training with multiple reduction strategies:

#### Gather Mode (Recommended)
```python
# All-gather U columns with padding
T_max = dist.all_reduce(T_local, op=dist.ReduceOp.MAX)
U_pad = torch.zeros(out_dim, T_max, device=U.device, dtype=U.dtype)
U_pad[:, :T_local] = U
dist.all_gather(U_list, U_pad)

# Concatenate unpadded columns
U_global = torch.cat([U_list[i][:, :lens[i]] for i in range(world_size)], dim=1)
```

#### Gram Mode (Deprecated)
```python
# WARNING: Mathematically incorrect for different token counts
U_T_U = U.t() @ U  # [T, T]
dist.all_reduce(U_T_U, op=dist.ReduceOp.SUM)
```

### Communication Volume

- **Gather**: O(out_dim × T_max × world_size) bytes
- **Gram**: O(T²) bytes (but mathematically incorrect)

### Padding Strategy

Variable token counts across ranks require padding:

```python
# Find max T across ranks
T_max = dist.all_reduce(T_local, op=dist.ReduceOp.MAX)

# Pad to T_max
U_pad = torch.zeros(out_dim, T_max, device=U.device, dtype=U.dtype)
U_pad[:, :T_local] = U
```

---

## API Reference

### Core Methods

#### `collect_kfac_factors(model, batch, loss=None, fisher_type="empirical")`

Collects KFAC factors from a forward-backward pass.

**Parameters:**
- `model`: Neural network model
- `batch`: Input batch dictionary
- `loss`: Optional precomputed loss tensor
- `fisher_type`: "empirical" | "true" | "mc"

**Returns:**
- Dictionary of KFAC factors per layer

**Example:**
```python
kfac = KFACNaturalGradient(damping=1e-4)
factors = kfac.collect_kfac_factors(model, batch)
```

#### `compute_natural_gradient(gradients, model, scale=1.0)`

Transforms gradients to natural gradients using KFAC.

**Parameters:**
- `gradients`: Dictionary of parameter gradients
- `model`: Model for layer structure
- `scale`: Global scaling factor

**Returns:**
- Dictionary of natural gradients

**Example:**
```python
# Collect gradients
gradients = {name: param.grad.clone() for name, param in model.named_parameters()}

# Compute natural gradients
nat_grads = kfac.compute_natural_gradient(gradients, model)

# Apply to parameters
for name, param in model.named_parameters():
    if name in nat_grads:
        param.grad = nat_grads[name]
```

#### `compute_fisher_vector_product(vector, scale=1.0, model=None)`

Computes Fisher-vector product: (G + λI) * v * (A + λI).

**Parameters:**
- `vector`: Vector to multiply with Fisher
- `scale`: Scaling factor
- `model`: Optional model for layer structure

**Returns:**
- Fisher-vector product

**Example:**
```python
# For trust region methods
fvp = kfac.compute_fisher_vector_product(gradients, model=model)
```

#### `get_fisher_scaled_gradient(model, batch, power=-1.0, fisher_type="empirical")`

Main interface for Fisher scaling with arbitrary powers.

**Parameters:**
- `model`: Model with computed gradients
- `batch`: Input batch
- `power`: Power to raise Fisher to (-1.0 for natural gradient)
- `fisher_type`: Type of Fisher to compute

**Returns:**
- Fisher-scaled gradients

**Example:**
```python
# Natural gradient (power=-1.0)
nat_grads = kfac.get_fisher_scaled_gradient(model, batch)

# Fisher normalization (power=-0.5)
norm_grads = kfac.get_fisher_scaled_gradient(model, batch, power=-0.5)
```

### Utility Methods

#### `_stabilize_matrix(M, name="", damping=None)`

Stabilizes a covariance matrix by enforcing bounded condition number.

**Parameters:**
- `M`: Symmetric positive semidefinite matrix
- `name`: Layer name for logging
- `damping`: Minimum eigenvalue threshold

**Returns:**
- Dictionary with 'eigvecs' and 'eigvals'

#### `update_diagonal_fisher(param_name, fisher_values, ema=True)`

Updates diagonal Fisher approximation for fallback.

**Parameters:**
- `param_name`: Parameter name
- `fisher_values`: Diagonal Fisher values
- `ema`: Use exponential moving average

#### `clear_cache()`

Clears all cached inverses to free memory.

#### `reset()`

Resets all stored factors and cache.

---

## Configuration Parameters

### Core Parameters

#### `damping: float = 1e-4`
Base damping factor λ for Tikhonov regularization.

**Rationale**: Standard in K-FAC literature (Martens & Grosse 2015)
**Effect**: Ensures Fisher approximation F + λI remains positive definite
**Tuning**: Should be tuned via validation performance

#### `damping_A: Optional[float] = None`
Separate damping for activation covariance A.

**Use case**: When activations and gradients have vastly different scales
**Default**: Uses `damping` if None

#### `damping_G: Optional[float] = None`
Separate damping for gradient covariance G.

**Use case**: Models with gradient scale issues
**Default**: Uses `damping` if None

### Update Parameters

#### `ema_decay: float = 0.99`
Exponential moving average decay α for running estimates.

**Formula**: F_t = α·F_{t-1} + (1-α)·F_batch
**Reference**: Ba et al. 2017, "Distributed Second-Order Optimization"
**Effect**: Does not affect final results, only convergence

#### `update_freq: int = 10`
Update K-FAC factors every N steps.

**Rationale**: Computational efficiency
**Effect**: Does not affect converged results

#### `min_layer_size: int = 32`
Minimum layer dimension to use K-FAC.

**Rationale**: K-FAC overhead outweighs benefits for very small layers
**Effect**: Performance optimization, not a hyperparameter

### Numerical Stability

#### `use_eigenvalue_correction: bool = True`
Whether to apply eigenvalue clipping.

**Critical**: Setting to False violates K-FAC assumptions
**Effect**: May lead to numerical instability or divergence
**Recommendation**: Always keep True for publication

#### `max_condition_number: float = 1e6`
Maximum condition number κ for Fisher factors.

**Formula**: λ_min' = max(λ_min, λ_max / κ_max)
**Rationale**: FP32 has ~7 decimal digits, 1e6 leaves safety margin
**Effect**: Prevents numerical issues in matrix inversion
**Sensitivity**: Results should be reported with this value

#### `kfac_eps: float = 1e-6`
Numerical stabilization epsilon for Woodbury solve.

**Usage**: Initial jitter for Cholesky with exponential backoff
**Backoff**: ε, 10ε, 100ε if needed

### Woodbury Configuration

#### `kfac_use_woodbury: bool = True`
Whether to use Woodbury identity for G-side computation.

**Theory**: For empirical Fisher, G = U U^T (rank-T)
**Benefit**: Exact (G + λI)^{-1} without forming o×o matrix
**Memory**: Uses only U and T×T algebra

#### `kfac_policy: str = "all"`
When to apply Woodbury.

**Options**:
- `"all"`: Use Woodbury for all layers (cleanest, exact for empirical Fisher)
- `"auto"`: Choose based on compute cost (Woodbury if T ≤ ρ·o and T ≤ T_max)
- `"hybrid"`: Woodbury for large output layers (out_features ≥ threshold or lm_head)
- `"small_only"`: Never use Woodbury (legacy eigendecomp path)

**Recommendation**: Use "all" for paper results, "auto" for production

#### `kfac_auto_rho: float = 1.0`
Cost ratio threshold for "auto" policy.

**Theory**: Woodbury O(oT²+T³) vs eigendecomp O(o³); break-even at T≈ρ·o
**Usage**: Use Woodbury if T ≤ ρ·out_dim

#### `kfac_auto_t_max: int = 8192`
Maximum T for Woodbury in "auto" policy.

**Rationale**: Avoid large T×T matrices even if cheaper than eigendecomp
**Memory**: T=8192 → S is 8k×8k×4B = 256MB (manageable)

#### `woodbury_store_device: str = "auto"`
Device for storing Woodbury factors.

**Options**: "auto" | "cuda" | "cpu"
**Auto logic**: GPU if < 500MB, CPU otherwise
**Memory**: U+S_inv moved to target device on use

#### `woodbury_dtype: str = "fp16"`
Dtype for U matrix.

**Options**: "fp16" | "bf16"
**Note**: S_inv always computed in fp32 for stability
**Rationale**: Sufficient precision for gradient statistics

### Distributed Configuration

#### `kfac_distributed_reduce: str = "gram"`
DDP/FSDP reduction mode.

**Options**:
- `"gram"`: Reduce Gram matrix U^T@U (cheap, exact for empirical Fisher)
- `"gather"`: All-gather U columns (more memory, supports arbitrary ops)
- `"none"`: No distributed reduction (single-GPU or debug)

**Warning**: "gram" mode is mathematically incorrect for different token counts across ranks

### Performance Parameters

#### `use_gpu_eigh: bool = True`
Whether to prefer GPU for eigendecomposition.

**Effect**: Faster when memory allows
**Note**: Does not affect numerical results (CPU and GPU are deterministic)

#### `show_progress: bool = False`
Whether to display progress bars.

**Effect**: UI only, no effect on results

---

## Best Practices

### Initialization

```python
# Recommended configuration for research
kfac = KFACNaturalGradient(
    damping=1e-4,                    # Standard damping
    kfac_use_woodbury=True,         # Use Woodbury for efficiency
    kfac_policy="all",              # Woodbury everywhere
    use_eigenvalue_correction=True,  # Required for stability
    max_condition_number=1e6,       # Standard condition limit
    update_freq=10,                 # Update every 10 steps
    min_layer_size=32,              # Skip very small layers
    show_progress=False             # Disable for production
)
```

### Usage Pattern

```python
# Training loop
for batch in dataloader:
    # Forward pass
    outputs = model(**batch)
    loss = outputs.loss
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Collect KFAC factors (periodically)
    if step % kfac.update_freq == 0:
        kfac.collect_kfac_factors(model, batch, loss=None)
    
    # Compute natural gradients
    gradients = {name: param.grad.clone() for name, param in model.named_parameters()}
    nat_grads = kfac.compute_natural_gradient(gradients, model)
    
    # Apply natural gradients
    for name, param in model.named_parameters():
        if name in nat_grads:
            param.grad = nat_grads[name]
    
    # Optimizer step
    optimizer.step()
```

### Memory Management

```python
# Clear cache periodically
if step % 100 == 0:
    kfac.clear_cache()
    torch.cuda.empty_cache()

# Reset factors if needed
if step % 1000 == 0:
    kfac.reset()
```

### Distributed Training

```python
# Use gather mode for correctness
kfac = KFACNaturalGradient(
    kfac_distributed_reduce="gather",  # Mathematically correct
    # ... other parameters
)

# Ensure proper synchronization
if dist.is_initialized():
    dist.barrier()  # Before KFAC computation
```

### Hyperparameter Tuning

1. **Start with defaults**: `damping=1e-4`, `max_condition_number=1e6`
2. **Tune damping**: Try `{1e-5, 1e-4, 1e-3}` based on validation performance
3. **Monitor condition numbers**: Log κ_A and κ_G values
4. **Adjust update frequency**: Balance accuracy vs. computational cost

---

## Troubleshooting

### Common Issues

#### "No factors collected"
**Cause**: Layers too small or update_freq too high
**Solution**: 
```python
kfac = KFACNaturalGradient(
    min_layer_size=16,  # Lower threshold
    update_freq=1       # Update every step
)
```

#### "Cholesky decomposition failed"
**Cause**: Ill-conditioned S matrix
**Solution**: Increase damping or check data quality
```python
kfac = KFACNaturalGradient(
    damping=1e-3,           # Higher damping
    kfac_eps=1e-5           # Larger initial jitter
)
```

#### "Out of memory"
**Cause**: Large batch size or model
**Solution**: 
```python
kfac = KFACNaturalGradient(
    woodbury_store_device="cpu",  # Store on CPU
    update_freq=20                # Update less frequently
)
```

#### "Double backward pass error"
**Cause**: Calling backward() twice on same graph
**Solution**: Use `loss=None` in collect_kfac_factors
```python
# Correct usage
model.zero_grad()
loss.backward()
factors = kfac.collect_kfac_factors(model, batch, loss=None)
```

### Debugging Tips

1. **Enable logging**: Set `logging.basicConfig(level=logging.DEBUG)`
2. **Check shapes**: Verify U and S_inv have correct dimensions
3. **Monitor condition numbers**: Log κ values for numerical stability
4. **Validate factors**: Check for NaN/Inf values in stored factors

### Performance Optimization

1. **Use Woodbury**: Set `kfac_use_woodbury=True` for large layers
2. **Adjust update frequency**: Balance accuracy vs. speed
3. **CPU offloading**: Use `woodbury_store_device="cpu"` for large models
4. **Batch size**: Larger batches improve factor quality

---

## References

### Primary References

1. **Martens & Grosse (2015)**: "Optimizing Neural Networks with Kronecker-factored Approximate Curvature" (ICML)
   - Original KFAC paper
   - Theoretical foundation and algorithm

2. **Ba et al. (2017)**: "Distributed Second-Order Optimization using Kronecker-Factored Approximations" (ICLR)
   - Distributed KFAC implementation
   - EMA decay strategy

3. **Nocedal & Wright (2006)**: "Numerical Optimization" (2nd ed.), Chapter 3
   - Condition number theory
   - Numerical stability in optimization

### Implementation References

4. **Woodbury Matrix Identity**: Wikipedia and numerical analysis textbooks
   - Efficient matrix inversion for low-rank updates

5. **PyTorch Documentation**: Hook system and distributed training
   - Implementation details for gradient collection

### Additional Reading

6. **Amari (1998)**: "Natural Gradient Works Efficiently in Learning"
   - Information geometry foundation

7. **Ollivier et al. (2017)**: "Online Natural Gradient as a Kalman Filter"
   - Online natural gradient methods

8. **Grosse & Martens (2016)**: "A Kronecker-factored Approximate Fisher Matrix for Convolution Layers"
   - Extension to convolutional layers

---

## Appendix

### Mathematical Notation

- **θ**: Model parameters
- **F**: Fisher Information Matrix
- **A**: Activation covariance matrix
- **G**: Gradient covariance matrix
- **λ**: Damping parameter
- **⊗**: Kronecker product
- **∇**: Gradient operator
- **∇_nat**: Natural gradient

### Implementation Notes

- All eigendecompositions use `torch.linalg.eigh()` for numerical stability
- Woodbury factors stored in mixed precision (FP16/BF16 for U, FP32 for S_inv)
- CPU offloading used to manage GPU memory
- Distributed operations support DDP and FSDP

### Version History

- **v1.0**: Initial implementation with eigendecomposition
- **v2.0**: Added Woodbury matrix identity support
- **v3.0**: Distributed training support
- **v4.0**: Memory optimizations and numerical stability improvements

---

*This documentation covers the complete KFAC implementation as of the current version. For updates and additional information, refer to the source code and test suite.*
