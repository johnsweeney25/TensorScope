# Phase 6 Fisher Shape Verification

## Direct Code Analysis

### What Phase 1 Stores

From `fisher/core/fisher_collector.py`, lines 656-757:

```python
# Line 656: After computing grad_sq, reduce to groups
group_fisher, group_type, num_groups = self._reduce_to_groups(
    param_name, grad_sq, param.shape, model, task
)

# Line 757: Store in fisher_ema
self.fisher_ema[key] = group_fisher
```

**For attention parameters** (lines 1393-1404):
```python
if weight_shape[0] % num_heads_actual == 0:
    grad_reshaped = grad_sq.view(num_heads_actual, head_dim, weight_shape[-1])
    # Sum over head_dim and hidden_size
    group_fisher = grad_reshaped.sum(dim=[1, 2])  # Result: [num_heads]
    return group_fisher, 'head', num_heads_actual
```

**Result**: `fisher_ema[param_name]` has shape `[num_heads]`, e.g., `[16]` for 16 heads.

**Critical**: Q, K, V are ALL treated identically - same reduction logic (lines 1287-1295).

---

### What Phase 6 Expects

From `fisher/qkov/qkov_interference.py`, lines 607-610:

```python
score, diagnostics = self.compute_block_head_score(
    contrib=task_a_contribs[param_name],  # From contribution_cache
    grad=task_b_grads[param_name],        # From gradient_manager
    fisher=fisher_ema[param_name],        # From fisher_ema ⚠️
    layer=layer, head=head, block=block, param_name=param_name
)
```

Then line 489-491:
```python
C_i_bh = self.indexer.slice_tensor(C_full, layer, head, block, param_name)
g_j_bh = self.indexer.slice_tensor(G_full, layer, head, block, param_name)
I_n_bh = self.indexer.slice_tensor(F_full, layer, head, block, param_name)
```

**QKOVIndexer.slice_tensor** expects to slice by Q/K/V/O blocks:
- From `fisher/qkov/qkov_interference.py`, lines 254-298:

For **Q projection**:
```python
if block == 'Q':
    start = head * d_k  # e.g., head=4, d_k=128 → start=512
    end = start + d_k   # end=640
    return BlockHeadSlice(row_slice=(start, end), col_slice=None)
    # Expects to apply: tensor[512:640, :] on full [4096, 4096] tensor
```

**Problem**: If `tensor` is `[16]` (group-reduced), then `tensor[512:640, :]` will fail!

---

## The Shape Mismatch

| Data | Phase 1 Storage | Phase 6 Expectation | Match? |
|------|----------------|-------------------|--------|
| **contribution_cache** | `[4096, 4096]` ✓ | `[4096, 4096]` | ✅ YES |
| **gradient_manager** | `[4096, 4096]` ✓ | `[4096, 4096]` | ✅ YES |
| **fisher_ema** | `[16]` ❌ | `[4096, 4096]` | ❌ NO |

**Verdict**: Phase 6 cannot slice `fisher_ema` because it's group-reduced.

---

## What Actually Happens

When Phase 6 calls `slice_tensor(fisher_ema[param_name], ...)`:

```python
# BlockHeadSlice.apply() tries:
tensor = tensor[512:640, :]  # Expects [4096, 4096]
# But tensor is actually [16]
```

**Possible outcomes**:
1. **IndexError**: `[16][512:640]` is out of bounds → empty tensor
2. **RuntimeError**: Shape mismatch in subsequent operations
3. **Silent failure**: Caught by `try/except` in unified_model_analysis.py:10100

---

## Verification from Production Code

From `unified_model_analysis.py`, lines 10098-10103:

```python
except Exception as e:
    logger.warning(f"  QK-OV interference analysis failed: {e}")
    logger.debug(f"  Full traceback: ", exc_info=True)
    results['qkov_interference'] = {'error': str(e)}
```

**This means Phase 6 errors are silently caught and logged!**

Check your logs for:
```
QK-OV interference analysis failed: ...
```

---

## The Theoretical Problem

Phase 6 claims:
> "Block-wise resolution (Q, K, V, O)"

But Phase 1 doesn't distinguish Q/K/V:
- All three use identical reduction logic (sum over head_dim and hidden)
- Only O is different (sums over different dimensions)
- The group reduction collapses Q/K/V information

**Even if we fixed the shape issue**, Phase 6 still couldn't distinguish Q from K from V contributions in the Fisher normalization, because Phase 1 didn't store them separately.

---

## Solution Options

### Option 1: Use Contribution-Based Fisher (Recommended)

Phase 6 already has full contributions in `contribution_cache`. Compute Fisher from them:

```python
# In qkov_interference.py, compute_block_head_score:
if self.normalization_mode == 'contribution_based':
    # Approximate Fisher from stored contributions
    I_n_full = self._compute_fisher_from_contributions(task_a, param_name)
    I_n_bh = self.indexer.slice_tensor(I_n_full, layer, head, block, param_name)
```

**Pros**:
- Uses existing data (no extra storage)
- Properly distinguishes Q/K/V/O blocks
- Statistical valid: I ≈ E[C_i] = E[(∇log p)²]

**Cons**:
- Requires enough samples per parameter
- Not the same as EMA Fisher (but theoretically equivalent)

### Option 2: Store Full Fisher in Phase 1 (Memory-Intensive)

Add parallel storage:
```python
if self.enable_cross_task_analysis:
    self.fisher_ema[key] = group_fisher  # Existing: [16]
    self.fisher_full[key] = grad_sq      # New: [4096, 4096]
```

**Pros**:
- Uses exact EMA Fisher
- No statistical approximation

**Cons**:
- ~2GB per parameter for LLaMA-2-7B
- Defeats the purpose of group reduction

### Option 3: Accept Limitation & Revise Claims

Phase 6 uses head-level Fisher (no Q/K/V distinction):
- Normalize by head, not by block
- Revise documentation to clarify

**Pros**:
- No code changes needed
- Honest about capabilities

**Cons**:
- Loses the "block-wise resolution" novelty
- Less granular analysis

---

## Recommended Action

**Immediate**:
1. Check logs for Phase 6 errors
2. Add explicit shape validation in `QKOVInterferenceMetric.__init__`:
   ```python
   # Verify fisher_ema has expected shapes
   sample_param = next(iter(self.fisher_collector.fisher_ema.values()))
   if sample_param.ndim == 1:
       logger.warning(
           "fisher_ema contains group-reduced tensors. "
           "Phase 6 requires full parameter tensors for QK-OV slicing. "
           "Falling back to contribution-based Fisher."
       )
       self.normalization_mode = 'contribution_based'
   ```

**Short-term**: Implement Option 1 (contribution-based Fisher)

**Long-term**: Document the architectural decision and theoretical implications
