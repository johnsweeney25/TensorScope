# Phase 1 vs Phase 6: Complete Theoretical Analysis

## Executive Summary

**Finding**: Phase 1 and Phase 6 have a fundamental architectural conflict:

1. **Phase 1** stores group-reduced Fisher `[num_heads]` that treats Q, K, V identically
2. **Phase 6** expects full Fisher `[hidden, hidden]` for QK-OV-specific slicing
3. **Current state**: Phase 6 silently fails due to shape mismatch (caught by try/except)
4. **Theoretical issue**: Even if shapes matched, Phase 1 doesn't distinguish Q/K/V contributions

---

## The Complete Data Flow

### Phase 1: Fisher Computation

**File**: `fisher/core/fisher_collector.py`

#### Step 1: Compute per-sample gradient (lines 372-647)
```python
# With micro_batch_size=1:
for start_idx in range(0, batch_size, micro_batch_size=1):
    # Forward pass
    loss = compute_loss(model, sample)
    
    # Backward pass
    loss.backward()
    
    # For each parameter:
    grad_sq = param.grad.pow(2)  # Shape: [4096, 4096] for attention
```

#### Step 2: Store full contribution (my fix, lines 618-647)
```python
if self.store_sample_contributions:  # True when enable_cross_task_analysis=True
    # Store FULL parameter gradient squared
    full_contribution = grad_sq  # [4096, 4096]
    normalized_contribution = full_contribution / max(1, total_active_tokens)
    
    self.contribution_cache[task][sample_key][name] = normalized_contribution
    # ✓ This is correct - Phase 6 needs full tensors
```

#### Step 3: Reduce to groups (lines 656-670)
```python
group_fisher, group_type, num_groups = self._reduce_to_groups(
    param_name, grad_sq, param.shape, model, task
)
```

**For attention parameters** (`_reduce_attention_structural`, lines 1393-1404):
```python
if is_qkv_proj:
    if weight_shape[0] % num_heads_actual == 0:
        # Reshape: (num_heads * head_dim, hidden) -> (num_heads, head_dim, hidden)
        grad_reshaped = grad_sq.view(num_heads_actual, head_dim, weight_shape[-1])
        
        # Sum over head_dim and hidden_size
        group_fisher = grad_reshaped.sum(dim=[1, 2])  # Result: [num_heads]
        
        return group_fisher, 'head', num_heads_actual
```

**CRITICAL**: Lines 1287-1295 show that Q, K, V all use the same logic:
```python
if 'q_proj' in param_name or 'k_proj' in param_name or 'v_proj' in param_name:
    # Q/K/V projections: group heads by behavior
    group_grad_sq = self._extract_behavioral_heads(
        grad_sq, head_indices, weight_shape, 'qkv'  # ⚠️ Same 'qkv' type for all!
    )
```

**Result**: Phase 1 does NOT distinguish between Q, K, and V - they're all grouped identically.

#### Step 4: Store group-reduced Fisher (line 757)
```python
self.fisher_ema[key] = group_fisher  # Shape: [16] for 16 heads
```

**Result**: `fisher_ema` contains `[num_heads]`, NOT full `[hidden, hidden]` tensors.

---

### Phase 6: QK-OV Interference

**File**: `fisher/qkov/qkov_interference.py`

#### Step 1: Retrieve data (lines 564-579)
```python
# Get contributions for sample i from task A
task_a_contribs = self.fisher_collector.contribution_cache.get(
    f"{task_a}_{sample_i}", {}
)
# ✓ Shape: [4096, 4096] - correct

# Get gradients for sample j from task B  
task_b_grads = self.fisher_collector.gradient_manager.get_sample_gradients(
    task_b, sample_j
)
# ✓ Shape: [4096, 4096] - correct

# Get EMA Fisher
fisher_ema = self.fisher_collector.fisher_ema
# ❌ Shape: [16] - WRONG, expects [4096, 4096]
```

#### Step 2: For each block (Q/K/V/O), slice tensors (lines 607-614)
```python
score, diagnostics = self.compute_block_head_score(
    contrib=task_a_contribs[param_name],  # [4096, 4096] ✓
    grad=task_b_grads[param_name],        # [4096, 4096] ✓
    fisher=fisher_ema[param_name],        # [16] ❌
    layer=layer, head=head, block=block, param_name=param_name
)
```

#### Step 3: Slice by block and head (lines 489-491)
```python
C_i_bh = self.indexer.slice_tensor(C_full, layer, head, block, param_name)
g_j_bh = self.indexer.slice_tensor(G_full, layer, head, block, param_name)
I_n_bh = self.indexer.slice_tensor(F_full, layer, head, block, param_name)
# ❌ This fails because F_full is [16], not [4096, 4096]
```

**QKOVIndexer** expects to slice differently for each block (lines 254-358):

For **Q** (lines 254-298):
```python
if block == 'Q':
    # For split projections: [H*d_k, d_model]
    start = head * d_k  # e.g., 4 * 128 = 512
    end = start + d_k   # 640
    return BlockHeadSlice(row_slice=(start, end), col_slice=None)
    # Expects: tensor[512:640, :] on [4096, 4096]
    # Receives: [16] → ERROR
```

For **K** (lines 287-291):
```python
if block == 'K' and self.config.uses_gqa:
    # Handle grouped-query attention
    kv_head = head // (H // self.config.num_kv_heads)
    start = kv_head * d_k
    end = start + d_k
    return BlockHeadSlice(row_slice=(start, end), col_slice=None)
    # Different slicing logic than Q!
```

For **V** (lines 301-338):
```python
if block == 'V':
    # V uses d_v (may differ from d_k!)
    start = head * d_v  # Note: d_v, not d_k
    end = start + d_v
    # ...
```

For **O** (lines 340-358):
```python
elif block == 'O':
    # O projection: (hidden_size, num_heads * head_dim)
    # Use COLUMN slicing, not row slicing!
    start = head * d_k
    end = start + d_k
    return BlockHeadSlice(row_slice=None, col_slice=(start, end))
    # ⚠️ Completely different slicing dimension!
```

**Result**: The indexer NEEDS the full parameter tensor to apply block-specific, dimension-specific slicing.

---

## The Theoretical Conflicts

### Conflict 1: Shape Mismatch

| Data Source | Phase 1 Stores | Phase 6 Expects | Can Slice? |
|-------------|---------------|----------------|------------|
| `contribution_cache` | `[4096, 4096]` | `[4096, 4096]` | ✅ YES |
| `gradient_manager` | `[4096, 4096]` | `[4096, 4096]` | ✅ YES |
| `fisher_ema` | `[16]` | `[4096, 4096]` | ❌ NO |

**Impact**: Phase 6 cannot slice `fisher_ema` by Q/K/V/O blocks.

### Conflict 2: Q/K/V Indistinguishability

Even if `fisher_ema` had full shape `[4096, 4096]`, Phase 1's reduction treats Q/K/V identically:

```python
# Phase 1 (lines 1287-1295): Q, K, V use same logic
if 'q_proj' in param_name or 'k_proj' in param_name or 'v_proj' in param_name:
    group_grad_sq = self._extract_behavioral_heads(
        grad_sq, head_indices, weight_shape, 'qkv'  # Same type!
    )
```

**What this means**:
- Q, K, V gradients are reshaped identically
- Summed over the same dimensions `[1, 2]`
- No block-specific information is retained
- Only O is different (sums over different dimensions)

**Implication**: 
- Phase 6's claim of "block-wise resolution (Q, K, V, O)" is compromised
- At best, we can distinguish "QKV" from "O"
- We CANNOT distinguish Q from K from V using Phase 1's Fisher

### Conflict 3: Mathematical Invalidity

Phase 6's normalization (line 518):
```python
normalized_contrib = C_i_bh / fisher_for_normalization
```

If we tried to broadcast:
- `C_i_bh`: `[128, 4096]` (sliced Q for head 4)
- `fisher_for_normalization`: `[16]` (group-reduced)

This would either:
1. **Broadcast incorrectly**: Divide by scalar `fisher[4]` for entire head 4
2. **Fail with shape error**: Cannot align `[128, 4096]` with `[16]`

**Neither is theoretically correct!** We need element-wise Fisher for the specific Q/K/V/O block.

---

## Why Phase 6 Appears to Work (But Doesn't)

From `unified_model_analysis.py`, lines 10098-10103:

```python
try:
    heatmap = qkov_metric.compute_heatmap(...)
    # ...
except Exception as e:
    logger.warning(f"  QK-OV interference analysis failed: {e}")
    logger.debug(f"  Full traceback: ", exc_info=True)
    results['qkov_interference'] = {'error': str(e)}
```

**Observation**: All Phase 6 errors are caught and logged as warnings, so:
1. The pipeline continues without Phase 6 results
2. Users might not notice the failure
3. `results['qkov_interference']` contains `{'error': '...'}`

**To verify**: Check your logs for:
```
QK-OV interference analysis failed: ...
```

---

## Solutions

### Option 1: Contribution-Based Fisher (Recommended)

**Idea**: Compute Fisher from stored contributions instead of using `fisher_ema`.

```python
def _compute_fisher_from_contributions(self, task, param_name):
    """
    Approximate Fisher as E[C_i] = E[(∇log p)²] from stored contributions.
    
    This gives us full parameter tensors with proper Q/K/V/O distinction.
    """
    contributions = []
    for sample_key, contribs in self.fisher_collector.contribution_cache.items():
        if sample_key.startswith(f"{task}_") and param_name in contribs:
            contributions.append(contribs[param_name])
    
    if not contributions:
        raise ValueError(f"No contributions found for {task}:{param_name}")
    
    # Mean of squared gradients = Fisher
    fisher_full = torch.stack(contributions).mean(dim=0)  # [4096, 4096]
    return fisher_full
```

**Pros**:
- ✅ Uses existing data (no extra storage in Phase 1)
- ✅ Properly distinguishes Q/K/V/O (contributions are stored before reduction)
- ✅ Theoretically valid: I ≈ E[(∇log p)²]
- ✅ Same granularity as contrib and grad tensors

**Cons**:
- ⚠️ Requires sufficient samples per parameter (recommend n ≥ 30)
- ⚠️ Not identical to EMA Fisher (but statistically equivalent)
- ⚠️ Higher variance than EMA (but can be mitigated with more samples)

**Implementation**:
```python
# In qkov_interference.py, compute_block_head_score:
if self.normalization_mode == 'contribution_based':
    fisher_full = self._compute_fisher_from_contributions(task_a, param_name)
    I_n_bh = self.indexer.slice_tensor(fisher_full, layer, head, block, param_name)
else:
    # Fall back to fisher_ema (will fail with current implementation)
    I_n_bh = self.indexer.slice_tensor(fisher_ema[param_name], ...)
```

### Option 2: Store Full Fisher in Phase 1

**Idea**: Add parallel storage of full Fisher tensors.

```python
# In fisher_collector.py, around line 757:
if not hasattr(self, '_skip_ema_decay') or not self._skip_ema_decay:
    # Existing: Store group-reduced Fisher
    self.fisher_ema[key] = group_fisher  # [16]
    
    # NEW: Also store full Fisher for cross-task analysis
    if self.enable_cross_task_analysis:
        if key not in self.fisher_full:
            self.fisher_full[key] = grad_sq.clone()  # [4096, 4096]
        else:
            # EMA update for full Fisher
            prev = self.fisher_full[key]
            self.fisher_full[key] = prev * self.ema_decay + (1 - self.ema_decay) * grad_sq
```

**Pros**:
- ✅ Uses exact EMA Fisher (same as other phases)
- ✅ No statistical approximation
- ✅ Same temporal smoothing as fisher_ema

**Cons**:
- ❌ Massive memory: ~4GB per parameter for LLaMA-2-7B (4096×4096×4 bytes)
- ❌ Defeats the purpose of group reduction (memory efficiency)
- ❌ Requires Phase 1 changes

**Not recommended** unless memory is not a constraint.

### Option 3: Redesign Phase 6 for Head-Level Analysis

**Idea**: Accept that Phase 1 provides head-level Fisher, not block-level.

```python
# Revise Phase 6 to use head-level Fisher only
def compute_head_score(self, contrib, grad, fisher_head, layer, head):
    """
    Compute interference at head level (no Q/K/V/O distinction).
    
    Args:
        fisher_head: [num_heads] tensor from fisher_ema
    """
    # Slice contrib and grad by head (across Q/K/V/O combined)
    C_head = self._slice_entire_head(contrib, head)  # All of head's parameters
    g_head = self._slice_entire_head(grad, head)
    
    # Use scalar Fisher for this head
    I_head = fisher_head[head]  # Scalar
    
    # Normalize and compute score
    score = (C_head / I_head * g_head.abs()).sum()
    return score
```

**Pros**:
- ✅ Works with existing Phase 1 data
- ✅ No memory overhead
- ✅ Theoretically consistent

**Cons**:
- ❌ Loses "block-wise resolution" claim
- ❌ Cannot distinguish Q vs K vs V contributions
- ❌ Less granular than originally intended

**Requires**: Revising documentation and paper claims.

---

## Recommended Immediate Actions

1. **Verify the bug**:
   ```bash
   # Check logs for Phase 6 failures
   grep "QK-OV interference analysis failed" your_analysis_log.txt
   ```

2. **Add shape validation**:
   ```python
   # In qkov_interference.py, __init__:
   sample_fisher = next(iter(self.fisher_collector.fisher_ema.values()))
   if sample_fisher.ndim != 2:
       logger.error(
           f"Phase 6 requires full Fisher tensors, got shape {sample_fisher.shape}. "
           f"Expected [hidden, hidden], received group-reduced [{sample_fisher.shape[0]}]"
       )
       raise ValueError("Incompatible Fisher shape for QK-OV analysis")
   ```

3. **Implement Option 1** (contribution-based Fisher):
   - Add `_compute_fisher_from_contributions` method
   - Set `normalization_mode='contribution_based'` by default
   - Document the statistical properties

4. **Update documentation**:
   - Clarify that Q/K/V are distinguished via contributions, not Fisher
   - Explain why contribution-based Fisher is theoretically valid
   - Add memory and sample size requirements

---

## Theoretical Justification for Option 1

### Why Contribution-Based Fisher is Valid

The Fisher Information Matrix is defined as:
```
I = E_x[(∇log p(x))²]
```

Our contribution cache stores:
```
C_i = (∇log p(x_i))²  for each sample i
```

Therefore:
```
I ≈ (1/n) Σ_i C_i  (sample mean)
```

**Statistical properties**:
- ✅ **Unbiased**: E[Î] = I (sample mean is unbiased estimator)
- ✅ **Consistent**: Î → I as n → ∞ (law of large numbers)
- ✅ **Variance**: Var[Î] = Var[C]/n (decreases with more samples)

**Comparison to EMA Fisher**:
- EMA gives more weight to recent samples
- Sample mean treats all samples equally
- Both are valid Fisher estimators
- EMA has lower variance but potential temporal bias
- Sample mean has higher variance but no temporal bias

**For Phase 6's use case** (detecting interference):
- We care about *which parameters* conflict, not precise magnitude
- Relative ordering is more important than absolute values
- Contribution-based Fisher provides correct relative comparisons

**Conclusion**: Contribution-based Fisher is **theoretically sound** for Phase 6.

---

## Final Verdict

**Current state**: Phase 6 is broken due to shape mismatch and Q/K/V indistinguishability.

**Root cause**: Architectural mismatch between Phase 1's group reduction and Phase 6's granular analysis needs.

**Recommended solution**: Option 1 (contribution-based Fisher)
- Theoretically valid
- Practically feasible
- No Phase 1 changes required
- Properly distinguishes Q/K/V/O

**Implementation priority**: **HIGH** - Phase 6 is currently non-functional.
