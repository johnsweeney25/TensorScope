# Phase 1 vs Phase 6: Critical Theoretical Conflict

## Summary
**CRITICAL BUG**: Phase 1 stores group-reduced Fisher tensors, but Phase 6 expects full parameter tensors for QK-OV slicing.

## The Data Flow

### Phase 1: Fisher Computation (`fisher_collector.py`)

```python
# Line 656-670: After computing gradient, reduce to groups
group_fisher, group_type, num_groups = self._reduce_to_groups(
    param_name, grad_sq, param.shape, model, task
)

# Line 757: Store GROUP-REDUCED tensor in fisher_ema
self.fisher_ema[key] = group_fisher  # Shape: [16] for 16 heads
```

**What's stored in `fisher_ema`**:
- For attention parameters: `[num_heads]` e.g., `[16]`
- For MLP parameters: `[num_channels]` e.g., `[4096]` 
- NOT the full parameter tensor `[4096, 4096]`

**Group reduction logic** (`_reduce_attention_structural`):
- Lines 1287-1295: For Q/K/V: Treats them ALL THE SAME
  - Reshape `(num_heads * head_dim, hidden)` → `(num_heads, head_dim, hidden)`
  - Sum over `[head_dim, hidden]` → result: `[num_heads]`
- Lines 1297-1305: For O: Different sum dimensions but still groups
  - Result: `[num_heads]`
- **Q, K, V are NOT distinguished** - they're all grouped by head

### Phase 6: QK-OV Interference (`qkov_interference.py`)

```python
# Line 579: Get Fisher EMA (group-reduced!)
fisher_ema = self.fisher_collector.fisher_ema

# Line 610: Pass group-reduced Fisher to compute_block_head_score
score, diagnostics = self.compute_block_head_score(
    contrib=task_a_contribs[param_name],  # Full: [4096, 4096]
    grad=task_b_grads[param_name],        # Full: [4096, 4096]
    fisher=fisher_ema[param_name],        # GROUP-REDUCED: [16] ⚠️
    layer=layer, head=head, block=block, param_name=param_name
)

# Line 491: Try to slice the Fisher tensor
I_n_bh = self.indexer.slice_tensor(F_full, layer, head, block, param_name)
```

**What Phase 6 expects**:
- Full parameter tensor `[4096, 4096]` for each Q/K/V/O parameter
- QKOVIndexer slices by:
  - **Q**: rows `[head * d_k : (head+1) * d_k, :]` → `[128, 4096]`
  - **K**: rows `[head * d_k : (head+1) * d_k, :]` → `[128, 4096]`
  - **V**: rows `[head * d_v : (head+1) * d_v, :]` → `[128, 4096]`
  - **O**: columns `[:, head * d_k : (head+1) * d_k]` → `[4096, 128]`

## The Conflict

### What Actually Happens

When `slice_tensor` receives a `[16]` tensor and tries to apply `row_slice=(512, 640)`:

```python
# BlockHeadSlice.apply (line ~150)
if self.row_slice:
    tensor = tensor[self.row_slice[0]:self.row_slice[1], :]
    # Tries: [16][512:640, :] → SHAPE MISMATCH!
```

**Possible outcomes**:
1. **IndexError**: If tensor is `[16]` and we try `[512:640]`
2. **Silent wrong slicing**: If PyTorch broadcasts/reshapes unexpectedly
3. **Returns empty tensor**: `[16][512:640]` might be out of bounds

### Why This Is Critical

Phase 6 claims to distinguish Q, K, V, O contributions, but:

1. **Phase 1 doesn't store QK-OV-specific data**
   - It treats Q/K/V identically (same reduction logic)
   - Only distinguishes Q/K/V from O (different sum dimensions)
   - Stores aggregated head-level Fisher, not block-specific

2. **Phase 6 can't reconstruct QK-OV from group data**
   - You can't slice `[16]` to get Q-specific head 4 data
   - The group reduction has already mixed Q/K/V information
   - The indexer NEEDS the full tensor to do QK-OV-specific slicing

## The Theoretical Implications

### What Phase 6 Documentation Claims:
> "Circuit-level interference analysis: Block-wise resolution (Q, K, V, O), Head-level attribution"

### What Phase 6 Actually Does:
- Uses head-grouped Fisher that doesn't distinguish Q from K from V
- Attempts to slice group-reduced tensors as if they were full tensors
- The "block-wise resolution" is compromised because Phase 1's grouping has already collapsed the Q/K/V distinction

## The Solution Options

### Option 1: Store Full Fisher in Phase 1 (Correct but Memory-Intensive)
```python
# In fisher_collector.py, add parallel storage:
if self.enable_cross_task_analysis:
    # Store BOTH group-reduced AND full tensors
    self.fisher_ema[key] = group_fisher  # Existing: [16]
    self.fisher_full[key] = grad_sq      # New: [4096, 4096]
```

**Pros**:
- Phase 6 gets what it actually needs
- Theoretically correct QK-OV separation

**Cons**:
- Massive memory: ~2GB per parameter for LLaMA-2-7B
- Storage scales as O(parameters²) not O(groups)

### Option 2: Phase 6 Stores Its Own Full Fisher (Current Implementation)
```python
# contribution_cache stores full grad_squared (my fix)
self.contribution_cache[task][sample_key][name] = normalized_contribution  # [4096, 4096]
```

**Pros**:
- Memory only for sampled data (not full EMA)
- Phase 6 gets the granularity it needs for contributions

**Cons**:
- Still uses group-reduced `fisher_ema` for normalization
- Inconsistent: normalizes with head-level Fisher but analyzes with block-level contributions
- The formula `C_i / I_n` mixes granularities

### Option 3: Redesign Phase 6 to Use Group Data (Breaking Change)
- Phase 6 accepts that Phase 1 doesn't distinguish Q/K/V
- Redefine metric to use head-level data only
- Drop the block-wise (Q/K/V/O) resolution claim

### Option 4: Reimpute Full Fisher from Contributions (Statistical)
```python
# In Phase 6: Approximate full Fisher from stored contributions
I_n_full ≈ E[C_i] = (1/N) Σ (g_i)²
```

**Pros**:
- Uses existing per-sample data
- No additional Phase 1 storage

**Cons**:
- Statistical approximation, not the true EMA Fisher
- Requires enough samples per parameter

## Current State Assessment

**My fix enabled Phase 6 to run**, but there's a **deeper architectural conflict**:

1. ✅ **contribution_cache**: Now correctly stores full tensors
2. ✅ **gradient_manager**: Stores full gradients
3. ❌ **fisher_ema**: Still stores group-reduced tensors
4. ⚠️ **Phase 6 compute_block_head_score**: Slices full contrib/grad but group-reduced Fisher

**The normalization step is theoretically invalid**:
```python
# Line 518: Normalize contrib by Fisher
normalized_contrib = C_i_bh / fisher_for_normalization

# But:
# - C_i_bh: Sliced from full tensor → shape [128, 4096] for Q head 4
# - fisher_for_normalization: From group-reduced → shape [16] or similar
```

This will fail or broadcast incorrectly.

## Recommended Action

**Immediate**: Test Phase 6 with actual data to see what error occurs
```python
# Create minimal test
model = ...
fisher_collector = FisherCollector(...)
# ... run Phase 1 ...
qkov = QK_OVInterferenceMetric(config, fisher_collector)
scores = qkov.compute_sample_pair(...)  # Will this crash?
```

**Short-term**: Implement Option 4 (reimpute Fisher from contributions)
**Long-term**: Consider Option 1 with memory optimizations (sparse storage, on-demand computation)
