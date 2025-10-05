# Critical Audit: Robinson Paper vs Our Implementation

## Date: 2025-09-29

After deep analysis of the Robinson et al. paper, here are the KEY COMPONENTS we may be missing or implementing incorrectly:

## 🔴 CRITICAL MISSING/INCORRECT COMPONENTS

### 1. **Dual Hypothesis Framework Not Fully Implemented**

**Robinson's Approach:**
- **H₀ᵐ (Manifold)**: Single constant slope across all radii
- **H₀ᶠᵇ (Fiber Bundle)**: Two regimes with NON-INCREASING slopes

**Our Issue:**
```python
# We test both but the decision logic may be wrong
if not has_significant_regime_change:
    violates_manifold = True  # ✓ Correct
elif increasing_slopes:
    violates_fiber_bundle = True  # ✓ Correct
```

**Missing:** We don't properly test the MANIFOLD hypothesis independently. Robinson tests:
1. First: Is there a regime change? (If NO → reject manifold)
2. Then: If YES, are slopes increasing? (If YES → reject fiber bundle)

### 2. **Reach Estimation Too Simplistic**

**Robinson's Method (likely from paper context):**
- Uses the radius where volume growth deviates from power law
- Estimates using residual analysis from linear fit
- More sophisticated than just "first discontinuity"

**Our Implementation:**
```python
# Too simple - just uses first discontinuity or 50th percentile
if len(discontinuities) > 0:
    return radii[min(discontinuities)]
else:
    return radii[len(radii) // 2]  # Arbitrary!
```

**Should Be:**
```python
# Fit piecewise linear model and find where residuals explode
# Or use the radius where local dimension estimate becomes unstable
```

### 3. **Missing Bootstrap for P-Value Computation**

**Robinson's Approach:**
- Uses bootstrap or permutation tests for p-values
- Not just analytical formulas

**Our Gap:**
We use analytical Mann-Kendall but Robinson likely uses bootstrap for more robust p-values on the actual slope increases.

### 4. **Incorrect Radius Sampling**

**Robinson's Method:**
- Logarithmically spaced radii for uniform coverage in log space
- Ensures equal weight to all scales

**Our Implementation:**
```python
# We use linear spacing by default
self.radii = np.linspace(0.1, 10.0, n_radii)  # WRONG!
```

**Should Be:**
```python
# Logarithmic spacing as per paper
self.radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
```

### 5. **Missing Effect Size Consideration**

**Robinson's Approach:**
- Requires BOTH statistical significance (p < 10⁻³) AND practical significance
- Likely has minimum slope increase threshold

**Our Gap:**
We removed the effect size check when we fixed the p-value computation.

### 6. **Volume Counting May Be Wrong**

**Critical Issue:**
```python
volumes[i] = np.sum((distances <= r) & (distances > 0))
```

**Robinson's Method:**
- May use UNIQUE points within radius
- May handle ties differently
- May use strict inequality (<) not (<=)

### 7. **Missing Cross-Scale Validation**

**Robinson's Innovation:**
- Tests consistency across multiple scale ranges
- Not just one global test

**We Don't:**
- Test multiple overlapping windows
- Validate results across scales

## 🟡 POTENTIALLY MISSING SUBTLETIES

### 8. **Adaptive Radius Selection**

Robinson likely adapts radius range based on:
- k-NN distances (start at k-th neighbor distance)
- Density estimates
- Not fixed [0.1, 10.0] range

### 9. **Noise Estimation Method**

**Robinson:** Uses median absolute deviation (MAD) or similar robust estimator
**Us:** We use standard deviation which is less robust to outliers

### 10. **Multiple Testing Correction Scope**

**Robinson:** Applies correction across entire vocabulary
**Us:** We apply per-token which may be too conservative

## 🟢 WHAT WE DO CORRECTLY

1. ✅ Three-point centered differences for slopes
2. ✅ CFAR detector for discontinuities
3. ✅ Log-log transformation
4. ✅ Exclusion of center point
5. ✅ Significance level α = 10⁻³
6. ✅ Reach gating (though estimation needs work)

## 📝 CRITICAL FIXES NEEDED

### Fix 1: Logarithmic Radius Spacing
```python
# Replace linear spacing
radii = np.logspace(np.log10(r_min), np.log10(r_max), n_radii)
```

### Fix 2: Better Reach Estimation
```python
def estimate_reach_robinson(log_radii, log_volumes):
    # Fit linear model to early radii
    early_fit = np.polyfit(log_radii[:10], log_volumes[:10], 1)

    # Find where residuals exceed threshold
    predicted = np.polyval(early_fit, log_radii)
    residuals = np.abs(log_volumes - predicted)

    # Reach is where model breaks down
    breakdown_idx = np.where(residuals > 3 * np.std(residuals[:10]))[0]
    if len(breakdown_idx) > 0:
        return radii[breakdown_idx[0]]
    return radii[-1]
```

### Fix 3: Add Effect Size Requirement
```python
# Require both p < 0.001 AND meaningful effect
MIN_SLOPE_INCREASE = 0.1  # From paper context
violates = (p_value < 0.001) and (max_slope_increase > MIN_SLOPE_INCREASE)
```

### Fix 4: Bootstrap P-Values
```python
def bootstrap_p_value(slopes, n_bootstrap=1000):
    observed_trend = compute_trend_statistic(slopes)

    # Bootstrap null distribution
    null_trends = []
    for _ in range(n_bootstrap):
        shuffled = np.random.permutation(slopes)
        null_trends.append(compute_trend_statistic(shuffled))

    # One-sided p-value
    p_value = np.mean(null_trends >= observed_trend)
    return p_value
```

## 🎯 MOST CRITICAL ISSUE

**The radius spacing is likely the biggest issue.** Robinson emphasizes log-log analysis, which requires logarithmic spacing of radii. Linear spacing would give unequal weight to different scales and could cause spurious violations.

## 📊 VALIDATION TEST

To verify our implementation matches Robinson's:
1. Test on known manifolds (should pass)
2. Test on known non-manifolds (should fail)
3. Compare violation rates with paper's reported statistics
4. Check if polysemous tokens are more likely to violate

## SUMMARY

Our implementation captures Robinson's core idea but misses crucial details:
- **Logarithmic radius spacing** (critical!)
- **Sophisticated reach estimation**
- **Bootstrap p-values**
- **Effect size requirements**
- **Cross-scale validation**

These aren't just minor issues - they could significantly affect results. The logarithmic spacing issue alone could invalidate many of our test results.