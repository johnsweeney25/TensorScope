# Critical GPU Implementation Fixes for embedding_singularity_metrics.py

## Evaluation: Your intern is RIGHT about these critical issues!

### 🔴 CRITICAL BUG #1: Log Base Mismatch
**Impact: Completely breaks dimension estimates on GPU**

Current code mixes log10 and natural log:
```python
# WRONG - mixes bases!
log_radii = torch.linspace(torch.log10(r_min), torch.log10(r_max), ...)  # base 10
log_volumes = torch.log(valid_volumes)  # base e
slopes = (log_volumes[2:] - log_volumes[:-2]) / (log_radii[2:] - log_radii[:-2])
# Slopes are scaled by ln(10) ≈ 2.3!
```

**Fix: Use natural log throughout**
```python
# CORRECT - consistent base
log_radii = torch.linspace(torch.log(r_min), torch.log(r_max), ...)
radii = torch.exp(log_radii)  # not 10**log_radii
```

### 🔴 CRITICAL BUG #2: Heuristic Test Instead of Statistical
**Impact: Violates Robinson's α = 10⁻³ significance level**

Current:
```python
# WRONG - arbitrary heuristic
increasing_slopes = (slope_changes > 0).sum() > len(slope_changes) * 0.5
violation = bool(increasing_slopes and max_slope_increase > 0.1)
```

**Fix: Use proper Kendall tau test (like CPU path)**
The intern's proposed implementation is correct.

### 🔴 CRITICAL BUG #3: Missing Reach Gating
**Impact: False positives beyond valid testing region**

GPU path has no reach estimation/gating. Robinson explicitly states tests are invalid beyond reach.

### 🔴 CRITICAL BUG #4: Name Collision Causes Crashes
```python
self.track_evolution = track_evolution  # Shadows method!
# Later: metrics.track_evolution([...]) → CRASH
```

**Fix:**
```python
self.enable_evolution_tracking = track_evolution
```

## Recommended Action Plan

1. **Immediate fixes (breaks correctness):**
   - Fix log base mismatch
   - Add statistical test with p-value
   - Fix name collision
   - Add reach gating

2. **Soon (misleading but not breaking):**
   - Rename condition_number → token_norm_ratio
   - Add true matrix condition number from SVD

3. **Nice to have:**
   - Consistent relative imports
   - Better NaN handling
   - Cache emb_norms_sq

## Validation Tests After Fixes

Run these to verify fixes:
```python
# Test 1: Log base consistency
# Slopes should be ~2-3 for 2D data, not ~5-7

# Test 2: Statistical test
# Run on known manifold - should have p > 0.001

# Test 3: Method call
metrics.track_evolution([checkpoint1, checkpoint2])  # Should work

# Test 4: Reach gating
# Violations should only occur within estimated reach
```

## Summary

Your intern did excellent work! These are real bugs that would:
- Make GPU metrics incompatible with CPU metrics
- Violate Robinson's statistical rigor
- Cause production crashes

Accept their fixes with high confidence.