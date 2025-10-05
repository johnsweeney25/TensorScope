# K-FAC Implementation Analysis

## Current Issue: Redundant Matrix Reconstruction

### What We're Doing Now (INEFFICIENT)

1. **In `_stabilize_matrix`** (line 567-606):
   ```python
   eigvals, eigvecs = torch.linalg.eigh(M_work)  # Decompose
   eigvals_clipped = torch.clamp(eigvals, min=min_allowed)  # Clip
   M_stable = (eigvecs * eigvals_clipped.unsqueeze(0)) @ eigvecs.t()  # RECONSTRUCT (OOM!)
   return M_stable  # Store full matrix
   ```

2. **Store full matrix** (line 451-455):
   ```python
   self.kfac_factors[name] = {'A': A.cpu().float(), 'G': G.cpu().float()}
   ```

3. **Later, decompose AGAIN** (line 734-746):
   ```python
   eigvals_A, eigvecs_A = torch.linalg.eigh(A)  # Decompose AGAIN!
   eigvals_G, eigvecs_G = torch.linalg.eigh(G)
   # Store in cache
   ```

4. **Apply in eigenbasis** (line 795-799):
   ```python
   G_inv_grad = cache['eigvecs_G'] @ (
       (cache['eigvecs_G'].T @ grad_combined) / cache['eigvals_G'].unsqueeze(1)
   )
   ```

### The Problem

**We decompose → reconstruct → decompose again!**
- Step 1: Eigendecomposition (GPU)
- Step 2: Reconstruct full matrix Q Λ Q^T (CPU, OOM risk)
- Step 3: Store full matrix
- Step 4: Decompose again when computing natural gradient
- Step 5: Apply in eigenbasis

**This is wasteful and causes OOM!**

## The Reviewer's Point

The reviewer is **100% correct**:

> "you typically don't need to reconstruct the full matrix in K-FAC; applying the preconditioner in the eigenbasis avoids n×n materialization entirely."

### What We SHOULD Do

**Store the eigendecomposition directly:**

```python
# In _stabilize_matrix
eigvals, eigvecs = torch.linalg.eigh(M_work)
eigvals_clipped = torch.clamp(eigvals, min=min_allowed)

# DON'T reconstruct! Return decomposition directly
return {
    'eigvecs': eigvecs.cpu(),
    'eigvals': eigvals_clipped.cpu(),
    'is_decomposed': True
}
```

**Store decomposed form:**
```python
self.kfac_factors[name] = {
    'A_eigvecs': A_decomp['eigvecs'],
    'A_eigvals': A_decomp['eigvals'],
    'G_eigvecs': G_decomp['eigvecs'],
    'G_eigvals': G_decomp['eigvals']
}
```

**Apply directly in eigenbasis** (already implemented at line 795-799):
```python
# No need to decompose again!
G_inv_grad = G_eigvecs @ ((G_eigvecs.T @ grad) / G_eigvals.unsqueeze(1))
```

## Benefits

1. **Memory:** No n×n matrix reconstruction → no OOM
2. **Speed:** Skip redundant eigendecomposition
3. **Theory:** Identical mathematical result
4. **ICML:** Still applies eigenvalue correction to ALL layers

## Memory Savings

For a 4096×4096 matrix:
- **Current:** 
  - Eigenvectors: 64MB (GPU)
  - Reconstructed matrix: 64MB (CPU)
  - Later decomposition: 64MB (GPU)
  - **Total: ~192MB per factor**

- **Proposed:**
  - Eigenvectors: 64MB (CPU)
  - Eigenvalues: 16KB (CPU)
  - **Total: ~64MB per factor**
  - **Savings: 3× reduction**

## Implementation Plan

1. Modify `_stabilize_matrix` to return `(eigvecs, eigvals)` tuple
2. Update storage in `collect_kfac_factors` to store decomposed form
3. Update `_compute_layer_natural_gradient` to use stored decomposition
4. Remove redundant eigendecomposition in cache building

## Backward Compatibility

For code that expects full matrices, add a helper:
```python
def get_full_matrix(self, layer_name, factor='A'):
    """Reconstruct full matrix if needed (for diagnostics only)."""
    decomp = self.kfac_factors[layer_name]
    eigvecs = decomp[f'{factor}_eigvecs']
    eigvals = decomp[f'{factor}_eigvals']
    return eigvecs @ torch.diag(eigvals) @ eigvecs.T
```

## Conclusion

**The reviewer is right.** Our current implementation:
- ✅ Preserves K-FAC theory
- ✅ Applies eigenvalue correction
- ❌ Wastes memory with redundant reconstruction
- ❌ Wastes compute with redundant decomposition

**We should store the eigendecomposition directly** and skip the reconstruction entirely.
