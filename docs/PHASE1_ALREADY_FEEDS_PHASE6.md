# Phase 1 Already Feeds Phase 6 - No Recalculation Needed!

## TL;DR

**You do NOT need to recalculate Phase 1.** Phase 1 already produces the correct data for Phase 6 in `contribution_cache`. The problem is that **Phase 6 is using the wrong data source** (`fisher_ema` instead of `contribution_cache`).

---

## The Evidence

### Phase 1 Produces the Right Data

**File**: `unified_model_analysis.py`, line 9555

```python
fisher_computed = bombshell.compute_fisher_welford_batches(
    model=model,
    batches=batch_list,
    task=task_name,
    cache_gradients=self.config.enable_cross_task_analysis,  # ← Key parameter
    show_progress=getattr(self.config, 'verbose', False),
    max_batches=max_batches
)
```

**What this does**:
- Passes `cache_gradients=True` when `enable_cross_task_analysis=True`
- This triggers special per-sample processing in Phase 1

---

### Phase 1 Sets micro_batch_size=1

**File**: `BombshellMetrics.py`, line 426

```python
def compute_fisher_welford_batches(self, model, batches, task, cache_gradients=False, ...):
    # CRITICAL: Use micro_batch_size=1 for per-sample gradient storage
    micro_batch_size = 1 if (cache_gradients or self.enable_cross_task_analysis) else 10
```

**Result**: When `cache_gradients=True` or `enable_cross_task_analysis=True`, Phase 1 processes samples one at a time.

---

### Phase 1 Stores Full Tensors in contribution_cache

**File**: `fisher/core/fisher_collector.py`, lines 629-654 (my fix)

```python
# === NOVEL: Store per-sample contributions BEFORE group reduction ===
# CRITICAL: Stage 6 (QK-OV) needs FULL parameter tensors for head slicing
# Must store BEFORE _reduce_to_groups() which changes tensor shape
if hasattr(self, 'store_sample_contributions') and self.store_sample_contributions:
    if task not in self.contribution_cache:
        self.contribution_cache[task] = {}

    # For single-sample micro-batches, store the contribution
    if end_idx - start_idx == 1:
        # Convert to float32 for compatibility
        grad_f32 = grad.float() if grad.dtype == torch.bfloat16 else grad
        
        # Store FULL parameter gradient squared (not group-reduced!)
        # Stage 6 (QK-OV) will apply its own slicing later
        full_contribution = grad_f32.pow(2)  # Shape: [4096, 4096]
        
        # Normalize by tokens
        normalized_contribution = full_contribution / max(1, total_active_tokens)
        
        # Store with sample key
        sample_key = f"{task}_{start_idx}"
        if sample_key not in self.contribution_cache[task]:
            self.contribution_cache[task][sample_key] = {}
        
        # Store FULL tensor (not group-reduced!)
        self.contribution_cache[task][sample_key][name] = normalized_contribution.detach().cpu()

# Line 656: THEN do group reduction for Fisher EMA
group_fisher, group_type, num_groups = self._reduce_to_groups(...)
```

**What this stores**:
- ✅ Full parameter tensors: `[4096, 4096]` for attention
- ✅ Before group reduction (no information loss)
- ✅ Per-sample (keyed by `f"{task}_{sample_idx}"`)
- ✅ CPU-offloaded (memory efficient)

---

### Phase 1 Also Stores Group-Reduced Fisher

**File**: `fisher/core/fisher_collector.py`, line 757

```python
# Update EMA (only if not skipped for Welford-only mode)
if not hasattr(self, '_skip_ema_decay') or not self._skip_ema_decay:
    if key in self.fisher_ema:
        self.fisher_ema[key] = prev + (1 - self.ema_decay) * group_fisher  # [16] heads
    else:
        self.fisher_ema[key] = group_fisher  # [16] heads
```

**What this stores**:
- ❌ Group-reduced tensors: `[16]` for 16 heads
- ❌ Cannot be sliced by Q/K/V/O blocks
- ✅ Good for Phases 2-4 (mask generation, overlap)

---

## What Phase 6 Actually Does (THE BUG)

**File**: `fisher/qkov/qkov_interference.py`, lines 579-610

```python
# Get contributions for sample i from task A
task_a_contribs = self.fisher_collector.contribution_cache.get(
    f"{task_a}_{sample_i}", {}
)
# ✅ Uses contribution_cache - CORRECT (full tensors)

# Get gradients for sample j from task B  
task_b_grads = self.fisher_collector.gradient_manager.get_sample_gradients(
    task_b, sample_j
)
# ✅ Uses gradient_manager - CORRECT (full tensors)

# Get EMA Fisher
fisher_ema = self.fisher_collector.fisher_ema
# ❌ Uses fisher_ema - WRONG (group-reduced tensors)

# ...

# Compute score
score, diagnostics = self.compute_block_head_score(
    contrib=task_a_contribs[param_name],  # [4096, 4096] ✓
    grad=task_b_grads[param_name],        # [4096, 4096] ✓
    fisher=fisher_ema[param_name],        # [16] ❌ SHAPE MISMATCH!
    layer=layer, head=head, block=block, param_name=param_name
)
```

**The problem**: Phase 6 uses `fisher_ema` (group-reduced) when it should compute Fisher from `contribution_cache` (full tensors).

---

## The Warning Code (As You Suspected)

**File**: `unified_model_analysis.py`, lines 10098-10103

```python
try:
    # Create QK-OV metric
    qkov_metric = QKOVInterferenceMetric(...)
    
    # Compute heatmap
    heatmap = qkov_metric.compute_heatmap(...)
    
    # Extract results
    results['qkov_interference'] = {...}
    
except ImportError as e:
    logger.warning(f"  QK-OV analysis skipped: fisher.qkov module not available ({e})")
except Exception as e:
    logger.warning(f"  QK-OV interference analysis failed: {e}")  # ← THIS IS THE WARNING
    logger.debug(f"  Full traceback: ", exc_info=True)
    results['qkov_interference'] = {'error': str(e)}
```

**To check your logs**:
```bash
grep "QK-OV interference analysis failed" your_analysis_log.txt
```

You should see errors like:
- "Shape mismatch: expected [4096, 4096], got [16]"
- "Index out of bounds: tensor has size 16, but trying to access [512:640]"

---

## The Data Flow Summary

| Phase | Operation | Produces | Used By |
|-------|-----------|---------|---------|
| **Phase 1 Step 1** | `compute_fisher_welford_batches()` | Calls `update_fisher_welford()` | - |
| **Phase 1 Step 2** | `update_fisher_welford()` (lines 629-654) | `contribution_cache` (full tensors) | ✅ Phase 6 |
| **Phase 1 Step 3** | `update_fisher_welford()` (lines 656-757) | `fisher_ema` (group-reduced) | ✅ Phases 2-4 |
| **Phase 2-4** | Mask generation, comparison, overlap | Masks, statistics | - |
| **Phase 5** | Cross-task conflict detection | Conflict scores | - |
| **Phase 6** | QK-OV interference | Should use `contribution_cache` | ❌ Uses `fisher_ema` |

---

## Why Phase 6 Feeds From Phase 1 (By Design)

From the architecture, all downstream phases are SUPPOSED to feed from Phase 1:

1. **Phase 1**: Computes Fisher + stores per-sample data
   - Output: `fisher_ema`, `contribution_cache`, `gradient_manager`

2. **Phase 2-4**: Use `fisher_ema` for masks and overlap
   - Input: `fisher_ema` (group-reduced is fine for masks)

3. **Phase 5**: Uses `contribution_cache` + `gradient_manager` for conflicts
   - Input: `contribution_cache`, `gradient_manager` (needs full tensors)

4. **Phase 6**: Should use `contribution_cache` for QK-OV interference
   - Input: `contribution_cache`, `gradient_manager` (needs full tensors)
   - **BUG**: Currently uses `fisher_ema` instead

---

## The Fix: Phase 6 Should Compute Fisher From Contributions

**Option 1: Contribution-Based Fisher** (RECOMMENDED)

Phase 6 should compute its own Fisher from the contributions that Phase 1 already stored:

```python
# In qkov_interference.py, add this method:
def _compute_fisher_from_contributions(self, task, param_name):
    """
    Approximate Fisher as E[(∇log p)²] from stored contributions.
    
    Phase 1 already stored full parameter contributions in contribution_cache.
    We just need to average them.
    """
    contributions = []
    
    # Iterate over all samples for this task
    for sample_key in self.fisher_collector.contribution_cache.keys():
        if sample_key.startswith(f"{task}_"):
            contribs = self.fisher_collector.contribution_cache[sample_key]
            if param_name in contribs:
                contributions.append(contribs[param_name])
    
    if not contributions:
        raise ValueError(f"No contributions found for {task}:{param_name}")
    
    # Fisher = E[(∇log p)²] ≈ mean of squared gradients
    fisher_full = torch.stack(contributions).mean(dim=0)  # [4096, 4096]
    return fisher_full

# In compute_block_head_score, replace:
# fisher_ema[param_name]  # ❌ Wrong
# with:
fisher_full = self._compute_fisher_from_contributions(task_a, param_name)  # ✅ Correct
```

**Why this works**:
- ✅ Phase 1 already computed and stored the contributions
- ✅ Full tensors available (`[4096, 4096]`)
- ✅ Theoretically valid: Fisher = E[(∇log p)²]
- ✅ No Phase 1 recalculation needed
- ✅ Q/K/V are distinct (stored before group reduction)

---

## Verification Steps

1. **Check that Phase 1 populated contribution_cache**:
   ```python
   # After Phase 1 completes
   print(f"contribution_cache keys: {list(bombshell.contribution_cache.keys())}")
   # Should show: ['math', 'general'] or similar
   
   for task, samples in bombshell.contribution_cache.items():
       print(f"Task '{task}': {len(samples)} samples")
       sample_key = list(samples.keys())[0]
       for param, contrib in list(samples[sample_key].items())[:3]:
           print(f"  {param}: shape {contrib.shape}")
   # Should show: [4096, 4096] or similar FULL parameter shapes
   ```

2. **Check Phase 6 warnings**:
   ```bash
   grep "QK-OV interference analysis failed" your_log.txt
   ```

3. **Verify Phase 1 used micro_batch_size=1**:
   ```bash
   grep "micro_batch_size=1" your_log.txt
   # or
   grep "Using TRUE Welford accumulation" your_log.txt
   ```

---

## Answer to Your Question

> "Do we have to recalculate phase 1? phase 1 is supposed to feed all of the down stream ones."

**NO, you do NOT need to recalculate Phase 1!**

Phase 1 already:
- ✅ Computes Fisher with proper Welford accumulation
- ✅ Stores full parameter tensors in `contribution_cache` (when `enable_cross_task_analysis=True`)
- ✅ Stores group-reduced tensors in `fisher_ema` for Phases 2-4
- ✅ Stores per-sample gradients in `gradient_manager` for Phase 5

**Phase 6 just needs to use the right data source** (contribution_cache, not fisher_ema).

> "In unified_model_analysis.py there is code to print the warnings right?"

**YES, line 10101**:
```python
logger.warning(f"  QK-OV interference analysis failed: {e}")
```

Check your logs for this warning. The error will likely be a shape mismatch when trying to slice `fisher_ema[param]` (shape `[16]`) with row indices like `[512:640]`.

---

## Implementation Priority

**IMMEDIATE FIX** (1-2 hours):

1. Add `_compute_fisher_from_contributions()` method to `QKOVInterferenceMetric`
2. Modify `compute_block_head_score()` to use contribution-based Fisher
3. Add validation: warn if n_samples < 30

**NO PHASE 1 CHANGES NEEDED** - it already works correctly!
