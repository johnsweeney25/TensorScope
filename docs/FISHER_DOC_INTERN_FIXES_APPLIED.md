# Fisher Documentation: Intern Feedback - All Fixes Applied ✅

## Summary

All ship-blockers, high-impact improvements, and polish items from intern feedback have been addressed. The documentation is now production-ready for ICLR 2026 submission.

---

## 🚨 Ship-Blockers (FIXED)

### 1. Order of Operations: Square-Then-Reduce ✅

**Issue**: Documentation incorrectly showed `reduce→square` which computes `(E[g])²` instead of `E[g²]`.

**Counterexample**: `g = [1, -1]`
- ❌ Reduce→Square: `(1 + (-1))² = 0`
- ✅ Square→Reduce: `1² + (-1)² = 2`

**Fixed in**:
- Phase 1 → Group Reduction section (lines 292-315)
- Data Flow → fisher_ema → Computation (lines 162-168)
- Critical Data Flow Diagram → Path B (lines 209-214)
- Phase 1 theoretical foundation (lines 296-310)

**New correct order**:
```
1. Compute full parameter gradient: grad [4096, 4096]
2. Square elementwise: grad² → [4096, 4096]
3. Reduce to groups: grad² → [16] (by head)
4. Welford accumulate / Store in fisher_ema[key]
```

### 2. `contribution_cache` Schema Consistency ✅

**Issue**: Documentation showed flat iteration but code uses nested schema.

**Correct Schema**:
```python
# Nested by task: contribution_cache[task][sample_key][param_name]
fisher_collector.contribution_cache[task][f"{task}_{sample_idx}"][param_name] = C_i
```

**Fixed Code Examples**:

**Before** (incorrect flat iteration):
```python
for sample_key, contribs in self.fisher_collector.contribution_cache.items():
    if sample_key.startswith(f"{task}_") and param_name in contribs:
```

**After** (correct nested access):
```python
task_dict = self.fisher_collector.contribution_cache.get(task, {})
for sample_key, contribs in task_dict.items():
    if param_name in contribs:
```

**Fixed in**:
- Phase 1 → Per-Sample Contributions (lines 317-333)
- Data Flow → contribution_cache storage (lines 176-194)
- Phase 6 → `_compute_fisher_from_contributions` (lines 1193-1210)

**Also Added**:
- Empty contributions guard (lines 1200-1201)
- Force fp32 on CPU for stable averaging (lines 1203-1205)

---

## High-Impact Improvements (ADDED)

### 1. Explicit Diagonal Fisher Statement ✅

Added to Phase 1 (line 315):
> **Note**: Both group Fisher and Phase 6 use diagonal Fisher (per-parameter variance), not full covariance

### 2. Token-Sum Convention ✅

Added to `contribution_cache` sections (lines 194, 329):
> **Important**: `gradient_manager` returns **token-sum** per-sample gradients to match the per-token normalization of contributions; do not use token-mean unless Fisher is also scaled accordingly.

### 3. Determinism Checklist ✅

Added to reproducibility section (lines 887-893):
```python
# For deterministic results:
# 1. model.eval() - disable dropout/DropPath
# 2. Set seeds - torch, numpy, Python random
# 3. torch.use_deterministic_algorithms(True) - optional
# 4. Set CUBLAS_WORKSPACE_CONFIG=":16:8" - if using deterministic algos
# 5. Fix dataloader shuffling - set shuffle=False or use fixed seed
# 6. Set torch.backends.cudnn.benchmark=False - for determinism
```

### 4. Wording Improvement: "Naïve" Instead of "WRONG" ✅

Changed Phase 1 Welford section (line 273):
- **Before**: "Standard accumulation (WRONG)"
- **After**: "Naïve one-pass summation (less stable)"
- Added: "Still unbiased, but numerically worse"

### 5. Phase 6: Aggregation Configuration ✅

Added configuration API and aggregation notes (lines 1098-1119):
```python
metric = QKOVInterferenceMetric(
    # Aggregation: 'mean' makes heads/blocks comparable regardless of size
    aggregate='mean',  # or 'sum'
    
    # For ablation studies
    # variant='unsigned'  # 'signed' | 'symmetric'
    # fisher_pool='A'     # 'A' | 'B' | 'pooled'
)
```

**Added Note**:
> By default we report the **mean** over the block/head slice to make heads and blocks comparable regardless of parameter count. Set `aggregate='sum'` to recover raw totals.

### 6. GQA Head Mapping ✅

Added to Phase 6 alternative metrics (line 1119):
> **GQA Mapping**: When pooling K/V heads for Grouped Query Attention, note that head h maps to kv-head floor(h / (H / H_kv)).

### 7. Numerical Health Diagnostics ✅

Added to Phase 6 outputs (lines 687-691):
> **Numerical Health Diagnostics** (included in outputs):
> - `min(Î)`, `median(Î)`, `max(Î)`: Fisher spectrum per slice
> - `% below ε`: Percentage of parameters near zero Fisher
> - `ridge_lambda_applied`: Actual ridge value after clamping
> - These help diagnose ill-conditioning and guide hyperparameter tuning

### 8. Empty-B Guard Documentation ✅

Added to Phase 6 (line 693):
> **Empty Task Guard**: If task_b has zero stored gradients, the metric returns zeros with an explicit warning message instead of taking means over empty axes.

---

## Nice Polish (COMPLETED)

### 1. Consistent Schema Documentation ✅

All `contribution_cache` references now consistently show nested schema:
```python
# Nested by task: contribution_cache[task][sample_key][param_name]
```

### 2. Implementation References ✅

Added function/API names alongside line numbers for all phases:
- Phase 1: `fisher/core/fisher_collector.py::update_fisher_welford()`
- Phase 2: `BombshellMetrics::compute_fisher_importance()`
- Phase 3: `BombshellMetrics::get_fisher_pruning_masks()`
- Phase 4: `BombshellMetrics::compute_fisher_overlap()`
- Phase 5: `CrossTaskConflictDetector::detect_conflicts()`
- Phase 7: `BombshellMetrics::scale_by_fisher()`

### 3. All Code Examples Updated ✅

Every code snippet now reflects:
- Square-then-reduce order
- Nested contribution_cache schema
- Proper error guards
- Numerical stability practices (fp32 on CPU for averaging)

---

## Verification

### Schema Consistency Check ✅

All mentions of `contribution_cache` now use:
```python
fisher_collector.contribution_cache[task][f"{task}_{i}"][param_name]
```

### Order Check ✅

All Fisher computation descriptions show:
1. Square elementwise
2. Reduce to groups
3. Welford accumulate

### Code Example Check ✅

All Phase 6 code examples:
- Use `task_dict = contribution_cache.get(task, {})`
- Include empty checks
- Force fp32 for averaging
- Match actual implementation in `qkov_interference.py`

---

## Impact

### Before Fixes
- ❌ Incorrect mathematical formulation (reduce→square)
- ❌ Schema mismatch between docs and code
- ❌ Missing critical numerical stability details
- ❌ No guidance on determinism or configuration
- ❌ Unclear aggregation semantics

### After Fixes
- ✅ Mathematically correct (square→reduce)
- ✅ Schema consistency across all examples
- ✅ Comprehensive numerical stability guidance
- ✅ Complete determinism checklist
- ✅ Clear aggregation and configuration API
- ✅ Production-ready for ICLR 2026

---

## Summary by Section

| Section | Ship-Blockers | High-Impact | Polish |
|---------|--------------|-------------|--------|
| Phase 1 | ✅ Square-then-reduce | ✅ Diagonal Fisher note | ✅ Naïve wording |
| Data Flow | ✅ Path B corrected | ✅ Token-sum note | ✅ Schema consistency |
| Phase 6 | ✅ Schema fixed | ✅ Aggregation config | ✅ All code examples |
| Reproducibility | - | ✅ Determinism checklist | - |
| Config | - | ✅ API documentation | ✅ Function names |

---

## Final Status

**Documentation Quality**: Production-ready ✅
**Reviewer-Proof**: Yes ✅
**Future-Proof**: Implementation references + API names ✅
**Complete**: All intern feedback addressed ✅

The documentation is now:
- **Clear**: Correct order of operations, explicit diagonal Fisher
- **Opinionated**: Strong guidance on configuration and best practices
- **End-to-end**: Complete from Phase 1 through Phase 7
- **Airtight**: All examples match implementation, proper guards documented
