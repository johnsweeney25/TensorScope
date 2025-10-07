# Fisher Accumulation Methods: EMA vs Welford

## Quick Reference Table

| Method | Primary Use Case | Accumulation Method | Storage | Notes |
|--------|-----------------|---------------------|---------|-------|
| **Group Fisher** | Pruning, merging, task arithmetic | **BOTH** (Welford + EMA) | `fisher_accumulated` (Welford)<br>`fisher_ema` (legacy) | ⭐ Welford is primary |
| **KFAC** | Natural gradient, second-order optimization | **Neither** (periodic refresh) | `kfac_factors` (eigendecomp or Woodbury) | Refreshes every N steps |
| **Lanczos** | Eigenspectrum analysis | **None** (one-shot) | Transient (not stored) | Computes on-demand |
| **FisherSpectral** | Block-diagonal capacity | **None** (one-shot or streaming) | Per-block eigenvalues | Optional gradient accumulator |

---

## Detailed Breakdown

### 1. Group Fisher (FisherCollector)

**Location:** `fisher/core/fisher_collector.py`

**Accumulation:** **BOTH Welford AND EMA** (dual-mode for backward compatibility)

#### Welford Accumulation (Primary, Recommended)
```python
# FisherCollector.update_fisher_welford (fisher_collector.py)
# Welford's Algorithm for Numerically Stable Accumulation
# This is MORE ACCURATE than EMA for getting true Fisher expectation
# EMA downweights old data, Welford gives equal weight to ALL data

# NOTE: Variable named 'n_samples_seen' but semantically stores token_total
n = self.n_samples_seen[task]  # Running sum of active tokens (not batch count!)
weight = float(total_active_tokens)  # Weight for this batch  
old_mean = self.fisher_accumulated[task][welford_key]
new_total_weight = n + weight

# Float64 for numerical stability (accumulate in float64, persist to float32/fp16)
delta = group_fisher_f64 - old_mean_f64
new_mean_f64 = old_mean_f64 + (delta * weight / new_total_weight)

# Update M2 for variance (in float64)
delta2 = group_fisher_f64 - new_mean_f64
m2_f64 += weight * delta * delta2

# Store results (keep running mean/M2 in float64 during training)
self.fisher_accumulated[task][welford_key] = new_mean_f64
self.fisher_m2[task][welford_key] = m2_f64
self.n_samples_seen[task] += weight  # Increment token counter (line 773)

# Variance computed ON-DEMAND (lazy evaluation, not during accumulation)
# When needed: variance = M2 / (W - 1) where W = n_samples_seen[task]
# For general weights w_i: variance = m2 / (W - W_2/W) where W = Σw_i, W_2 = Σw_i²
# For integer frequency weights (tokens): variance = m2 / (W - 1)
# See get_fisher_confidence_interval() for usage (computed once when requested)
```

**Storage:**
- `self.fisher_accumulated[task]` - Welford mean (unbiased)
- `self.fisher_m2[task]` - Welford M2 for variance (float64)
- `self.fisher_variance[task]` - Computed on-demand (not during accumulation)
- `self.n_samples_seen[task]` - Total weight (sum of active tokens; misnomer, not batch count!)

**Advantages:**
- ✅ **Unbiased estimator** - equal weight to all samples
- ✅ **Numerically stable** - accumulate in float64, persist to float32/fp16 if needed
- ✅ **Provides variance** - enables statistical tests with Bessel correction
- ✅ **Token-weighted** - uses weight = #active tokens per batch, matching empirical expectation over tokens

#### EMA Accumulation (Legacy, Backward Compatible)
```python
# FisherCollector.update_fisher_ema (optional, can be disabled)
# EMA Fisher (maintained for backward compatibility)
fisher_ema[task_key] = (
    self.ema_decay * old_fisher_ema + 
    (1 - self.ema_decay) * group_fisher
)
```

**Storage:**
- `self.fisher_ema[task_key]` - EMA estimate (biased toward recent)

**Advantages:**
- ✅ **Fast convergence** - adapts quickly to distribution shifts
- ⚠️ **Biased** - downweights old data (ema_decay=0.99)
- 📝 **Legacy support** - for existing code

**Configuration:**
```python
collector = FisherCollector(
    ema_decay=0.99,  # Only affects fisher_ema
    # Welford is always active (no config needed)
)

# For publication work, use Welford:
fisher_values = collector.fisher_accumulated[task]

# For quick adaptation, use EMA:
fisher_values = collector.fisher_ema
```

---

### 2. KFAC (KFACNaturalGradient)

**Location:** `fisher/kfac_utils.py`

**Accumulation:** **Periodic Refresh** (not continuous accumulation)

#### How It Works
```python
# KFACNaturalGradient.collect_kfac_factors
self.update_count += 1

# Only update periodically
if self.update_count % self.update_freq != 0:
    return self.kfac_factors  # Return cached factors
    
# Otherwise, recompute from scratch
self.kfac_factors = self._build_fresh_factors(model, batch)
```

**Why Not EMA/Welford:**
- KFAC stores **eigendecompositions** or **Woodbury factors**, not raw Fisher values
- Eigendecompositions can't be meaningfully averaged (eigenvectors change)
- Instead: compute fresh every `update_freq` steps

**Distributed Training (DDP/FSDP):**
Under DDP/FSDP, we aggregate Woodbury statistics across ranks by all-reducing `U^T @ U` (and `U^T @ Y` at apply-time) or all-gathering token columns of `U`. We default to Gram all-reduce (`kfac_distributed_reduce="gram"`) for stability and bandwidth efficiency. This yields the same result as concatenating tokens across ranks without allocating a global `U`.

**Storage:**
```python
self.kfac_factors[layer] = {
    # A-side (activation covariance)
    'A_eigvecs': ...,  # Eigenvectors
    'A_eigvals': ...,  # Eigenvalues (includes damping)
    
    # G-side (gradient covariance)
    'G_type': 'woodbury_empirical',  # or 'eig'
    'U': ...,          # Woodbury factors
    'S_inv': ...,      # Precomputed inverse
    # OR
    'G_eigvecs': ...,  # Eigendecomp
    'G_eigvals': ...,
}
```

**Update Strategy:**
```python
KFACNaturalGradient(
    update_freq=10,    # Refresh every 10 steps
    ema_decay=0.99,    # NOT USED for K-FAC factors
                       # (only for diagonal_fisher fallback)
)
```

**Note on EMA Parameter:**
The `ema_decay` parameter in K-FAC **does not affect K-FAC factors** (eigenspaces/Woodbury stats are never EMA-averaged). It's only used for:
1. Diagonal Fisher fallback (legacy path)
2. Backward compatibility with old code that expects this parameter

**Rationale:**
- K-FAC factors are **expensive to compute** (O(o³) for eigendecomp)
- Periodic refresh (every 10-100 steps) is standard practice (Martens & Grosse 2015)
- Fresh computation is more stable than trying to average eigenspaces

---

### 3. Lanczos Spectrum

**Location:** `fisher/core/fisher_lanczos_unified.py`

**Accumulation:** **None** (one-shot computation)

#### How It Works
```python
# FisherLanczos.lanczos_algorithm (fisher_lanczos_unified.py)
def lanczos_algorithm(op: LinOp, config: LanczosConfig):
    """
    Compute top-k eigenvalues via Lanczos iteration.
    
    This is a ONE-SHOT algorithm:
    - Runs for max_iters iterations
    - Returns eigenvalues immediately
    - Does NOT accumulate over multiple batches
    """
    # Initialize random vector
    v_curr = random_init()
    
    # Lanczos iterations
    for i in range(max_iters):
        w = op.apply(v_curr)  # Matrix-vector product
        alpha = dot(v_curr, w)
        # ... build tridiagonal matrix T ...
    
    # Solve eigenvalue problem
    eigenvalues = eigh(T)
    return eigenvalues  # One-shot result
```

**Why No Accumulation:**
- Lanczos computes **extreme eigenvalues** (largest/smallest)
- These are properties of the **current curvature matrix**
- Averaging eigenvalues across batches would be meaningless
  - Different batches → different loss landscapes
  - Eigenvalues describe local geometry, not global average

**Usage Pattern:**
```python
# Compute spectrum for current batch
spectrum = collector.lanczos_spectrum(
    model, batch,
    operator='ggn',
    k=10,  # Top-10 eigenvalues
)

# Result is for THIS batch only, not accumulated
eigenvalues = spectrum['eigenvalues']  # [λ₁, λ₂, ..., λ₁₀]
```

**When to Recompute:**
- Every few hundred steps (track loss landscape evolution)
- At checkpoints (for reproducibility)
- After significant training progress

---

### 4. FisherSpectral (Block-Diagonal)

**Location:** `fisher/core/fisher_spectral.py`

**Accumulation:** **Optional Streaming** (default: one-shot)

#### Default: One-Shot
```python
# FisherSpectral.compute_fisher_spectrum (fisher_spectral.py)
def compute_fisher_spectrum(self, model, batch, ...):
    """
    Compute Fisher spectrum for current batch.
    
    Default mode: ONE-SHOT
    - Collects per-sample gradients
    - Computes block-diagonal Fisher
    - Returns eigenvalues immediately
    """
    # Collect gradients for this batch
    gradients = self._collect_per_sample_gradients(model, batch)
    
    # Compute per-block spectrum
    for block in blocks:
        F_block = compute_gram_matrix(gradients[block])
        eigenvalues[block] = eigh(F_block)
    
    return eigenvalues  # One-shot result
```

#### Optional: Streaming Mode
```python
# FisherSpectral.gradient_accumulator (optional)
self.gradient_accumulator = {}  # Optional accumulator

# Can accumulate gradients across batches
def accumulate_gradients(self, gradients):
    """Optional: accumulate gradients for larger effective batch."""
    for key, grad in gradients.items():
        if key not in self.gradient_accumulator:
            self.gradient_accumulator[key] = []
        self.gradient_accumulator[key].append(grad)
```

**Why Usually One-Shot:**
- Block-diagonal Fisher is **cheap to compute** (per-layer O(n³) where n = layer size)
- More accurate to compute on **fresh data** than accumulate stale gradients
- Gradient storage is expensive (memory intensive)

**Usage Pattern:**
```python
# One-shot (typical)
spectral = FisherSpectral()
spectrum = spectral.compute_fisher_spectrum(model, batch)

# Streaming (rare, for very small batches)
spectral = FisherSpectral(config=SpectralConfig(storage_mode='streaming'))
for batch in batches:
    spectral.accumulate_gradients(batch)
spectrum = spectral.finalize()  # Compute spectrum from accumulated
```

---

## Comparison: When to Use Each

### Welford (Group Fisher)
✅ **Use when:**
- Computing **long-term Fisher estimates** (pruning, merging)
- Need **unbiased statistics** (publication quality)
- Want **variance estimates** (confidence intervals)
- Accumulating over **many batches** (100s-1000s)

❌ **Don't use when:**
- Need **fast adaptation** to distribution shifts
- Memory constrained (stores mean + M2 + variance)

### EMA (Group Fisher - Legacy)
✅ **Use when:**
- Need **quick adaptation** (online learning)
- Distribution is **non-stationary**
- Backward compatibility with **old code**

❌ **Don't use when:**
- Need **unbiased estimates** (publication)
- Want statistical significance testing

### Periodic Refresh (KFAC)
✅ **Use when:**
- Computing **natural gradients** (optimization)
- Need **second-order curvature** (per-step)
- Can afford **periodic recomputation** (10-100 steps)

❌ **Don't use when:**
- Need **continuous tracking** (every step)
- Extremely memory constrained

### One-Shot (Lanczos, FisherSpectral)
✅ **Use when:**
- Analyzing **current loss landscape** (checkpoints)
- Need **extreme eigenvalues** (optimization diagnostics)
- Memory constrained (no storage)

❌ **Don't use when:**
- Need **accumulated statistics** (long-term trends)
- Want **variance estimates**

---

## Implementation Notes

### Numerical Stability

**Welford (Group Fisher):**
```python
# Accumulate Welford state in float64; persist to storage as float32 (or fp16) if needed,
# but keep the running mean/M2 in float64 during training
group_fisher_f64 = group_fisher.double()
old_mean_f64 = old_mean.double()

# Prevents catastrophic cancellation
# Float32: ~7 digits → breaks around 10⁷ samples
# Float64: ~16 digits → safe for 10¹⁵ samples
```

**KFAC:**
```python
# Eigenvalues include Tikhonov damping (KFACNaturalGradient._stabilize_matrix)
eigvals = eigvals + float(damping)  # Regularization

# Woodbury uses Cholesky (float32 S_inv, KFACNaturalGradient.collect_kfac_factors)
# S is always computed in FP32 for numerical stability
S_inv = torch.cholesky_inverse(torch.linalg.cholesky(S))
```

**Lanczos:**
```python
# Tridiagonal matrix in float64 (FisherLanczos.lanczos_algorithm)
T = np.zeros((n, n), dtype=np.float64)

# Selective reorthogonalization (FisherLanczos._reorthogonalize)
# Maintains numerical stability in Lanczos vectors
```

### Memory Usage

| Method | Per-Parameter Memory | Total Memory (1.5B params) | Notes |
|--------|---------------------|---------------------------|-------|
| **Welford** | 3× (mean + M2 + var) | ~18 GB (float32) | Accumulation in float64 |
| **EMA** | 1× (running avg) | ~6 GB (float32) | Legacy mode |
| **KFAC** | ~2× (eigvecs + vals) | ~12 GB (CPU offloaded) | Per-layer eigendecomp |
| **KFAC (Woodbury)** | O(o·T + T²) per layer | ~50 MB (e.g., vocab=50k, T=512) | Memory scales with o×T + T², not o² |
| **Lanczos** | k vectors × n_params | ~30 GB (k=5, 1.5B params, FP32) | Memory ≈ k × n_params × dtype_size |
| **FisherSpectral** | Per-block eigenvalues | ~100 KB (top-k only) | Stores eigenvalues, not full matrices |

---

## Recommendations for ICLR 2026

### For Publication Results

**Parameter Importance (Pruning/Merging):**
```python
collector = FisherCollector()
collector.update_fisher_welford(model, batch, task='math')

# Use Welford accumulation (unbiased)
fisher = collector.fisher_accumulated['math']
variance = collector.fisher_variance['math']

# Report with confidence intervals
importance = fisher / (variance.sqrt() + eps)
```

**Natural Gradient Optimization:**
```python
kfac = KFACNaturalGradient(
    damping=1e-4,
    update_freq=10,  # Standard practice
    kfac_policy="all"  # Woodbury everywhere
)

# Refresh periodically
kfac.collect_kfac_factors(model, batch)
nat_grad = kfac.compute_natural_gradient(gradients, model)
```

**Loss Landscape Analysis:**
```python
# At checkpoints or every N epochs
spectrum = collector.lanczos_spectrum(
    model, batch,
    operator='ggn',
    k=20
)

# Report: λ_max, condition number, spectral gap
metrics = {
    'max_eigenvalue': spectrum['eigenvalues'][0],
    'condition_number': spectrum['condition_number'],
    'spectral_gap': spectrum['spectral_gap']
}
```

---

## Summary

| Question | Answer |
|----------|---------|
| **Which methods use EMA?** | Only **Group Fisher** (optional, legacy) |
| **Which methods use Welford?** | Only **Group Fisher** (primary) |
| **Which methods use neither?** | **KFAC** (periodic refresh), **Lanczos** (one-shot), **FisherSpectral** (one-shot) |
| **Which is most accurate?** | **Welford** (unbiased, all data weighted equally) |
| **Which is fastest?** | **EMA** (no variance computation) |
| **Which for publication?** | **Welford** (unbiased statistics, variance estimates) |

**Bottom Line:**
- 📊 **Pruning/Merging:** Use Welford (`fisher_accumulated`)
- 🚀 **Optimization:** Use KFAC (periodic refresh)
- 🔍 **Analysis:** Use Lanczos (one-shot at checkpoints)
- 📈 **Capacity:** Use FisherSpectral (one-shot per-block)
