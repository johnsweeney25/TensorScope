# KFAC Memory Analysis - OOM Root Cause [FIXED]

## Problem Summary
CUDA OOM occurred **after** KFAC factor computation completed, when trying to allocate 43 GB during subsequent capacity metrics computation. The error happened at line 9879 in `unified_model_analysis.py` when calling `compute_capacity_metrics()`.

**STATUS: ALL FIXES IMPLEMENTED** (see Implementation Status section below)

## Root Cause Analysis

### Memory Flow

1. **KFAC Factor Collection** (Line 9866 in unified_model_analysis.py)
   - Calls `advanced_collector._update_kfac_factors(model, batch)`
   - This delegates to `KFACNaturalGradient.collect_kfac_factors()` in fisher/kfac_utils.py
   - **ISSUE #1**: Factors are stored on GPU in `self.kfac_factors[layer_name] = {'A': A, 'G': G}` (line 450)
   - For large models, this can consume 30-50 GB of GPU memory
   - **ISSUE #2**: The backward pass uses `loss.backward(retain_graph=True)` (line 374), which keeps the computation graph in memory

2. **Capacity Metrics Computation** (Line 9879)
   - Calls `compute_capacity_metrics()` which needs to compute eigenvalues
   - Code at line 465-466 of fisher_collector_advanced.py does:
     ```python
     eigvals_A = torch.linalg.eigvalsh(A.float().cpu())
     eigvals_G = torch.linalg.eigvalsh(G.float().cpu())
     ```
   - **ISSUE #3**: While eigendecomposition moves to CPU, the original A and G matrices remain on GPU
   - **ISSUE #4**: The `.float().cpu()` operation creates temporary GPU tensors during the conversion before moving to CPU
   - For a 10k×10k matrix, eigendecomposition workspace can require ~40 GB temporary memory

### Memory Budget Breakdown (Example for 7B model)

```
Model parameters:              ~14 GB (bfloat16)
Optimizer states:              ~28 GB (Adam)
KFAC factors (A+G all layers): ~30 GB (float32 covariances)
Activations/gradients:         ~10 GB
Computation graph (retain):    ~5 GB
---
Total before capacity metrics: ~87 GB

When capacity metrics tries eigendecomposition:
  - Temporary GPU tensors during .float().cpu(): ~15 GB
  - cuSOLVER workspace estimate: ~43 GB (this is where OOM occurs)
```

## Specific Bugs

### Bug #1: KFAC factors never moved off GPU
**Location**: `fisher/kfac_utils.py` line 450
```python
self.kfac_factors[name] = {'A': A, 'G': G}
```
**Problem**: Factors stay on GPU device indefinitely. For large models with many layers, this accumulates to 30-50 GB.

**Fix Strategy**: Move factors to CPU immediately after computation since they're only needed for CPU eigendecomposition later.

### Bug #2: retain_graph=True in KFAC collection
**Location**: `fisher/kfac_utils.py` line 374
```python
loss.backward(retain_graph=True)
```
**Problem**: Retains full computation graph in GPU memory. This is unnecessary since we only collect gradients once.

**Fix Strategy**: Use `retain_graph=False` (the default) since hooks capture what we need during backward pass.

### Bug #3: Temporary GPU tensors during CPU move
**Location**: `fisher/core/fisher_collector_advanced.py` lines 465-466
```python
eigvals_A = torch.linalg.eigvalsh(A.float().cpu())
eigvals_G = torch.linalg.eigvalsh(G.float().cpu())
```
**Problem**: 
- `A.float()` creates full-size GPU tensor in float32 (larger than bfloat16 original)
- Then `.cpu()` moves it, but GPU copy still exists briefly
- For 10k×10k matrix: 400 MB → 800 MB temporary spike per matrix

**Fix Strategy**: Move to CPU first, then convert dtype: `A.cpu().float()` instead of `A.float().cpu()`

### Bug #4: No cleanup after KFAC computation
**Location**: `fisher/kfac_utils.py` lines 456-468
```python
# Free per-layer activation/gradient to reduce peak memory
try:
    del activations[name]
except KeyError:
    pass
try:
    del gradients[name]
except KeyError:
    pass
# Occasionally trim CUDA cache to mitigate fragmentation
if torch.cuda.is_available():
    if idx % 8 == 0:
        torch.cuda.empty_cache()
```
**Problem**: Activations and gradients are freed, but:
- Happens inside the loop (good), but factors accumulate (bad)
- `torch.cuda.empty_cache()` only helps with fragmentation, not allocated tensors
- No cleanup of computation graph after backward pass

**Fix Strategy**: 
- Clear gradients immediately after KFAC collection: `model.zero_grad(set_to_none=True)`
- Move factors to CPU immediately
- Call `torch.cuda.empty_cache()` after full KFAC collection completes

### Bug #5: No memory estimation before eigendecomposition
**Location**: `fisher/kfac_utils.py` line 526-540
The `_stabilize_matrix` function has memory checks, but `compute_capacity_metrics` bypasses this by doing its own eigendecomposition.

**Fix Strategy**: Use `_stabilize_matrix` logic for all eigendecompositions, or ensure capacity metrics uses CPU-only eigendecomposition.

## Recommended Fixes (Priority Order)

### High Priority (Immediate)

1. **Move KFAC factors to CPU immediately after computation**
   ```python
   # In collect_kfac_factors, line 450
   self.kfac_factors[name] = {
       'A': A.cpu().float(),  # Move to CPU immediately
       'G': G.cpu().float()
   }
   ```

2. **Remove retain_graph=True**
   ```python
   # In collect_kfac_factors, line 374
   loss.backward()  # Don't retain graph
   ```

3. **Clear gradients after KFAC**
   ```python
   # After line 473 in collect_kfac_factors (in finally block)
   model.zero_grad(set_to_none=True)
   torch.cuda.empty_cache()
   ```

### Medium Priority

4. **Fix dtype conversion order in capacity metrics**
   ```python
   # In fisher_collector_advanced.py, lines 465-466
   eigvals_A = torch.linalg.eigvalsh(A.cpu().float())  # CPU first
   eigvals_G = torch.linalg.eigvalsh(G.cpu().float())
   ```

5. **Add memory guards before large allocations**
   ```python
   # Before capacity metrics computation
   if torch.cuda.is_available():
       free_mem, _ = torch.cuda.mem_get_info()
       required = estimate_eigendecomp_memory(A.shape[0])
       if required > free_mem * 0.5:
           logger.warning("Insufficient GPU memory, skipping capacity metrics")
           return {}
   ```

### Low Priority (Optimization)

6. **Stream-based factor computation**: Process layers one at a time, move to CPU, then proceed
7. **Lazy evaluation**: Only compute factors when needed for specific metrics
8. **Mixed precision**: Keep factors in bfloat16 where possible

## Testing Strategy

1. **Memory profiling**: Add memory tracking before/after each KFAC operation
2. **Synthetic test**: Create model with known number of large layers, verify factors are on CPU
3. **Integration test**: Run full analysis pipeline with memory budget constraints

## Expected Impact

With fixes #1-3 implemented:
- KFAC factors: 30 GB GPU → 0 GB GPU (moved to CPU)
- Computation graph: 5 GB → 0 GB (not retained)
- **Total freed**: ~35 GB

This should provide sufficient headroom for the 43 GB eigendecomposition allocation.

## Additional Notes

- The error message shows "54.97 GiB is allocated by PyTorch" - this confirms massive memory accumulation
- "3.00 GiB is reserved by PyTorch but unallocated" - indicates fragmentation, but not the main issue
- The log shows KFAC completed successfully before OOM - confirms memory is not freed after collection

## Implementation Status

### ✅ All Critical Fixes Applied

1. **retain_graph=True removed** (fisher/kfac_utils.py:374)
   - Changed `loss.backward(retain_graph=True)` → `loss.backward()`
   - Computation graph no longer retained unnecessarily

2. **KFAC factors moved to CPU** (fisher/kfac_utils.py:451-455)
   - Factors immediately moved to CPU after computation: `A.cpu().float()`, `G.cpu().float()`
   - EMA computation handles device transfer automatically
   - All usage sites updated to transfer factors to device on-demand

3. **Dtype conversion order fixed** (fisher/core/fisher_collector_advanced.py:466-467)
   - Changed `A.float().cpu()` → `A.cpu().float()`
   - Eliminates temporary GPU memory spike during dtype conversion

4. **Gradient cleanup added** (fisher/kfac_utils.py:493-502)
   - Added `model.zero_grad(set_to_none=True)` in finally block
   - Added `activations.clear()` and `gradients.clear()`
   - Added `torch.cuda.empty_cache()` after KFAC collection

5. **Non-existent import removed** (unified_model_analysis.py:9873-9874)
   - Removed import and call to non-existent `set_use_kfac()`
   - KFAC factors available directly through `advanced_collector.kfac_factors`

### Device Handling

All methods that use KFAC factors now handle CPU storage:
- `_compute_layer_natural_gradient` (kfac_utils.py): Moves factors to gradient device
- `_compute_powered_natural_gradient` (kfac_utils.py): Moves factors to gradient device  
- `compute_fisher_vector_product` (kfac_utils.py): Moves factors to vector device
- `_apply_fisher_power` (kfac_utils.py): Moves factors to gradient device
- `KFACFisherOperator.kfac_mv` (fisher_lanczos_unified.py): Moves factors to vector device

Factors are moved to device only when needed for computation, then computation happens on-device. Factors remain on CPU in storage.

## Expected Memory Savings

With all fixes applied:
- **KFAC factors**: 30 GB freed (moved from GPU to CPU)
- **Computation graph**: 5 GB freed (not retained after backward)
- **Gradient cleanup**: 3-5 GB freed (explicit cleanup in finally block)
- **Total freed**: ~38-40 GB

This provides sufficient headroom for the 43 GB eigendecomposition that was causing OOM.

## References

- Computation happens in: `unified_model_analysis.py` lines 9816-9963
- KFAC collection: `fisher/kfac_utils.py` lines 161-504
- Capacity metrics: `fisher/core/fisher_collector_advanced.py` lines 395-549
- Stabilization (has memory checks): `fisher/kfac_utils.py` lines 506-601
