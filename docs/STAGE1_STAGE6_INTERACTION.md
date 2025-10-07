# Stage 1 ↔ Stage 6 Interaction: Critical Analysis

## The Question

Does Stage 1 consider QK-OV circuits, and how does this affect Stage 6?

## Current Implementation

### Stage 1 (Fisher Computation) - Lines 615-651

**What it stores in `contribution_cache`**:
```python
# Line 616: Get accumulated gradient
grad = gradient_accumulator[name]  # Sum or weighted mean of micro-batch gradients

# Line 618: Apply group reduction
group_fisher, group_type, num_groups = self._reduce_to_groups(name, grad, param.shape, model)

# Line 638: Square the GROUP-REDUCED gradient
grad_squared = grad_f32.pow(2)  # ← This is grad AFTER group reduction!

# Line 651: Store in contribution_cache
self.contribution_cache[task][sample_key][name] = normalized_contribution
```

**Key insight**: The contribution stored is:
- From GROUP-REDUCED gradient (e.g., aggregated to head level)
- NOT the full parameter gradient
- Shape determined by `group_type` ('head', 'channel', 'param', etc.)

### Stage 6 (QK-OV Interference) - Lines 489-491

**What it expects**:
```python
# Line 484-486: Get contributions
C_full = contrib.detach()  # Expects FULL parameter tensor
G_full = grad.detach()     # Expects FULL parameter tensor
F_full = fisher.detach()   # Expects FULL parameter tensor

# Line 489-491: Apply QK-OV slicing
C_i_bh = self.indexer.slice_tensor(C_full, layer, head, block, param_name)
g_j_bh = self.indexer.slice_tensor(G_full, layer, head, block, param_name)
I_n_bh = self.indexer.slice_tensor(F_full, layer, head, block, param_name)
```

**What indexer.slice_tensor does** (lines 237-358):
- Expects weight matrix shape: `[out_features, in_features]`
- For Q/K/V: Slices rows by head: `[head*head_dim:(head+1)*head_dim, :]`
- For O: Slices columns by head: `[:, head*head_dim:(head+1)*head_dim]`

## THE CRITICAL PROBLEM

**Stage 1 stores**: Group-reduced tensors (e.g., shape `[num_heads]` for attention)
**Stage 6 expects**: Full parameter tensors (e.g., shape `[num_heads*head_dim, hidden_dim]`)

**When Stage 6 tries to slice a `[num_heads]` tensor expecting `[num_heads*head_dim, hidden_dim]`**:
- ❌ `slice_tensor` will fail or give wrong results
- ❌ The tensor shapes are incompatible

## Let Me Check What `_reduce_to_groups` Actually Returns

From fisher_collector.py:1075:
```python
def _reduce_to_groups(self, param_name, grad, param_shape, model):
    grad_fp32 = grad.to(torch.float32)
    grad_sq = grad_fp32.pow(2)  # Square the gradient
    
    # If reduction == 'param':
    return grad_sq, 'param', grad_sq.numel()
    
    # If reduction == 'group' (default):
    # Apply head/channel reduction
    if 'attention' in param_name:
        return self._reduce_attention(...)  # Returns aggregated by head
    elif 'mlp' in param_name:
        return self._reduce_linear(...)     # Returns aggregated by channel
```

**For attention parameters**:
- Input: `grad` with shape `[num_heads*head_dim, hidden_dim]`
- Output: Aggregated tensor with shape `[num_heads]`

## The Fix I Made Is WRONG!

My recent edit at line 616-651 stores:
```python
group_fisher = (accumulated_grad)²  # After group reduction
self.contribution_cache[task][sample_key][name] = group_fisher
```

This stores `[num_heads]` shaped tensor, but Stage 6 expects `[num_heads*head_dim, hidden_dim]`!

## What Needs to Happen Instead

**Two options**:

### Option A: Store BEFORE Group Reduction
```python
# Line 615: BEFORE calling _reduce_to_groups
if hasattr(self, 'store_sample_contributions') and self.store_sample_contributions:
    if end_idx - start_idx == 1:  # Single sample
        # Store FULL parameter gradient squared
        grad_f32 = gradient_accumulator[name].float()
        full_contribution = grad_f32.pow(2)
        
        sample_key = f"{task}_{start_idx}"
        if sample_key not in self.contribution_cache[task]:
            self.contribution_cache[task][sample_key] = {}
        
        # Store FULL parameter tensor (not group-reduced)
        self.contribution_cache[task][sample_key][name] = full_contribution.detach().cpu()

# Line 618: THEN do group reduction for Fisher accumulation
group_fisher, group_type, num_groups = self._reduce_to_groups(...)
```

### Option B: Store with Reduction Flag
```python
# Check if FisherCollector is set to 'param' mode (no reduction)
if self.reduction == 'param':
    # Safe to store and slice
    self.contribution_cache[task][sample_key][name] = grad_f32.pow(2)
else:
    # Need full parameter tensor for QK-OV slicing
    logger.warning("QK-OV requires reduction='param' mode")
```

## Theoretical Implications

**Does Stage 1 "consider" QK-OV circuits?**

Current answer: **NO**
- Stage 1 uses generic group reduction (heads, channels, etc.)
- Group reduction is INDEPENDENT of QK-OV structure
- Stage 6 is supposed to apply QK-OV-specific slicing

**Should Stage 1 "consider" QK-OV circuits?**

Answer: **NO - but it should store data in QK-OV-compatible format**
- Stage 1: Store full parameter contributions (no reduction)
- Stage 6: Apply QK-OV slicing on demand
- This separation of concerns is correct design

**The bug**: Stage 1 currently applies group reduction before storing, breaking Stage 6

## The Correct Fix

```python
# Line 625-651 (CORRECTED VERSION)
if hasattr(self, 'store_sample_contributions') and self.store_sample_contributions:
    if task not in self.contribution_cache:
        self.contribution_cache[task] = {}

    # For single-sample micro-batches, store the contribution
    if end_idx - start_idx == 1:
        # CRITICAL: Store BEFORE group reduction
        # Stage 6 needs full parameter tensors for QK-OV slicing
        grad_f32 = gradient_accumulator[name].float()
        
        # Square the FULL gradient (not group-reduced)
        full_contribution = grad_f32.pow(2)
        
        # Create unique key for this sample and parameter
        sample_key = f"{task}_{start_idx}"
        if sample_key not in self.contribution_cache[task]:
            self.contribution_cache[task][sample_key] = {}
        
        # Store FULL parameter tensor for QK-OV slicing
        self.contribution_cache[task][sample_key][name] = full_contribution.detach().cpu()

# THEN (line 618) do group reduction for Fisher Welford accumulation
group_fisher, group_type, num_groups = self._reduce_to_groups(
    name, gradient_accumulator[name], param.shape, model
)
```

## Memory Impact

**Before** (with group reduction):
- Attention parameter: `[16]` heads (64 bytes fp16)
- MLP parameter: `[4096]` channels (8KB fp16)

**After** (without group reduction):
- Attention parameter: `[4096, 4096]` (32MB fp16)
- MLP parameter: `[4096, 11008]` (88MB fp16)

**Per task with 768 samples storing ~50 parameters**:
- Before: ~50 × 768 × 8KB = 307MB
- After: ~50 × 768 × 60MB = **2.3TB** ❌

**This is why they DO group reduction** - to save memory!

## The Real Tradeoff

**Option 1**: Store full tensors (QK-OV works, uses 2.3TB)
**Option 2**: Store group-reduced tensors (saves memory, QK-OV broken)
**Option 3**: Use `reduction='param'` mode (no reduction, but breaks other analyses)

**Current state**: Option 2 (my recent fix made this worse)

## Recommended Solution

**Conditional storage based on needs**:
```python
# In FisherCollector.__init__
self.store_full_parameters_for_qkov = False  # User-controlled flag

# In update_fisher_welford
if self.store_sample_contributions:
    if self.store_full_parameters_for_qkov:
        # Store full tensors for QK-OV (memory expensive)
        contribution = gradient_accumulator[name].pow(2)
    else:
        # Store group-reduced tensors (memory efficient, no QK-OV)
        contribution = group_fisher  # After reduction
    
    self.contribution_cache[task][sample_key][name] = contribution.detach().cpu()
```

**User decides the tradeoff**:
```python
config = UnifiedConfig(
    enable_cross_task_analysis=True,  # Enable Phase 5
    enable_qkov_analysis=True,        # Enable Phase 6 (uses 2.3TB!)
)
```

## Current Status

**My recent fix is BROKEN** because:
1. It stores group-reduced tensors
2. Stage 6 expects full parameter tensors
3. QK-OV slicing will fail on wrong-shaped tensors

**Need to**:
1. Revert to storing full parameter tensors
2. Add memory warning
3. Make it optional via flag
4. Document the memory cost
