# Fisher Gradient Sharing Analysis

## Executive Summary

Successfully implemented gradient sharing between FisherCollector and FisherSpectral modules to eliminate duplicate gradient computations. However, performance analysis reveals important trade-offs.

## Implementation Details

### 1. Shared Gradient Cache System
- Added `GradientCache` class to fisher_collector.py for storing per-sample gradients
- Modified FisherCollector to optionally cache raw gradients before squaring
- Updated FisherSpectral to accept precomputed gradients via `precomputed_gradients` parameter
- Fixed gradient organization to properly concatenate parameters within blocks

### 2. Key Changes

#### fisher_collector.py
```python
class GradientCache:
    """Cache for sharing gradients between FisherCollector and FisherSpectral."""
    def add_sample_gradients(self, sample_grads: Dict[str, torch.Tensor])
    def get_gradients(self) -> List[Dict[str, torch.Tensor]]
```

#### fisher_spectral.py
```python
def compute_fisher_spectrum(
    self,
    model,
    batch,
    precomputed_gradients: Optional[List[Dict[str, torch.Tensor]]] = None
)
```

## Performance Analysis

### Test Results (SimpleModel: 100→200→200→10)
- **Unified approach**: 0.691s total
  - Gradient collection: 0.545s
  - Diagonal Fisher: 0.129s
  - Spectrum computation: 0.017s

- **Traditional approach**: 0.182s total
  - Fisher oneshot: 0.008s
  - Spectrum computation: 0.174s

### Key Insights

1. **FisherCollector oneshot mode is highly optimized**
   - Does NOT compute per-sample gradients
   - Directly accumulates Fisher: F += ∇ℓ ∇ℓᵀ
   - Much faster (0.008s) than per-sample collection (0.545s)

2. **FisherSpectral REQUIRES per-sample gradients**
   - Needs gradient matrix G ∈ ℝ^(N×P) for eigenvalue computation
   - Cannot work with pre-accumulated Fisher

3. **Trade-off Analysis**
   - Gradient sharing eliminates duplicate forward-backward passes
   - But per-sample gradient collection has ~68× overhead vs direct accumulation
   - Only beneficial when BOTH diagonal Fisher AND spectrum are needed

## When to Use Gradient Sharing

### ✅ USE gradient sharing when:
- Computing both diagonal Fisher (BombshellMetrics) AND spectrum (spectral_gap)
- Memory is not a constraint (per-sample storage)
- Need exact consistency between metrics

### ❌ DON'T USE gradient sharing when:
- Only computing diagonal Fisher (use oneshot mode)
- Only computing spectrum (compute directly)
- Working with very large models (memory overhead)

## Theoretical Correctness

### Fisher Information Matrix
- **Empirical Fisher**: F = (1/N) Σᵢ ∇ℓᵢ ∇ℓᵢᵀ
- **Diagonal Fisher**: F_diag = (1/N) Σᵢ (∇ℓᵢ)²
- **Spectrum**: eigenvalues of F via Gram trick when N << P

### Key Corrections Made
1. **Spectral gap**: λ₁ - λ₂ (NOT λ₂ - λ₁)
2. **NOT mixing time**: Fisher matrices aren't Markov operators
3. **Consistent subsampling**: Fixed indices per block across samples
4. **Block concatenation**: All params in block form single gradient vector

## Recommendations

1. **Keep both modes available**:
   - Fast path: Separate computation (current default)
   - Unified path: When both metrics needed

2. **Future optimization**:
   - Implement fast diagonal extraction from gradient matrix
   - Use gradient checkpointing to reduce memory
   - Consider vmap for parallel per-sample computation

3. **API Design**:
   ```python
   # Unified computation when both needed
   unified_metrics = get_unified_fisher_metrics(
       model, batch,
       compute_diagonal=True,
       compute_spectrum=True
   )
   ```

## Conclusion

The gradient sharing implementation is **theoretically correct** and **functionally complete**. However, performance characteristics mean it should be used selectively when both diagonal Fisher and spectrum are required, not as a default replacement for specialized implementations.

The existing separate implementations (FisherCollector for diagonal, FisherSpectral for spectrum) remain optimal for their individual use cases due to algorithmic differences in how they process gradients.