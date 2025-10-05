# Fiber Bundle Methodology Comparison

## Overview

This document explains the two different approaches to testing the fiber bundle hypothesis in our codebase:

1. **Robinson Method** (`robinson_fiber_bundle_test.py`) - Exact implementation from Robinson et al. (2024)
2. **Geometric Method** (`fiber_bundle_core.py`) - Our complementary geometric approach

Both tests detect violations of manifold/fiber bundle structure but focus on different mathematical properties.

---

## Robinson et al. Method (Paper Implementation)

### What It Tests
- **Volume growth patterns**: How many points fall within radius r as r increases
- **Log-log scaling**: Plots log(volume) vs log(radius) and analyzes slopes
- **Key indicator**: Slopes should be piecewise linear and DECREASING
- **Violation**: Slopes that INCREASE indicate semantic instability

### Mathematical Foundation
```
For a proper fiber bundle:
- Small radius regime: N(r) ~ r^d (d = embedding dimension)
- Large radius regime: N(r) ~ (r-ρ)^d_base (d_base = base space dimension)
- Transition at radius ρ (local "reach")
```

### Test Methodology
1. Count points within increasing radii: r ∈ [0.1, 0.2, ..., 10.0]
2. Compute slopes using three-point centered differences:
   ```
   slope[i] = (log(N[i+1]) - log(N[i-1])) / (log(r[i+1]) - log(r[i-1]))
   ```
3. Apply CFAR (Constant False Alarm Rate) detector to find discontinuities
4. Check if slopes increase through discontinuities
5. Apply Holm-Bonferroni correction for multiple testing

### When Violated
- Token has irregular neighborhood structure
- Semantically equivalent prompts may produce different outputs
- Token is "unstable" and should be avoided in critical applications

### Example Result
```python
result = RobinsonTestResult(
    violates_hypothesis=True,
    p_value=0.0003,
    increasing_slopes=True,
    max_slope_increase=0.45,
    small_radius_slope=2.3,
    large_radius_slope=3.1,  # Higher than small - violation!
    transition_radius=1.2
)
```

---

## Our Geometric Method

### What It Tests
- **Dimension consistency**: Is intrinsic dimension stable across scales?
- **Curvature regularity**: Are angles between neighbors distributed uniformly?
- **Tangent alignment**: Do nearby tangent spaces align smoothly?
- **Regime transition**: Is the transition between scales smooth?

### Mathematical Foundation
```
For a smooth fiber bundle:
- Local dimension should be constant
- Curvature should be regular (low variance)
- Tangent spaces should align (small subspace angles)
- Scale transitions should be predictable
```

### Test Methodology
1. Test four geometric properties:
   - Dimension consistency (30% weight)
   - Curvature regularity (25% weight)
   - Tangent alignment (30% weight)
   - Regime transition (15% weight)
2. Combine scores: `test_stat = Σ(weight_i * normalized_score_i)`
3. Bootstrap p-value estimation
4. Reject if p < 0.05

### When Violated
- Local geometry is irregular
- Manifold assumption is inappropriate
- Should use robust methods instead of manifold-based analysis

### Example Result
```python
result = FiberBundleTestResult(
    p_value=0.02,
    reject_null=True,
    dimension_consistency=0.4,  # High = bad
    curvature_regularity=0.3,
    tangent_alignment=0.6,  # High misalignment
    regime_transition=0.2
)
```

---

## Key Differences

| Aspect | Robinson Method | Our Geometric Method |
|--------|----------------|---------------------|
| **Primary Focus** | Volume growth patterns | Local geometric structure |
| **Key Metric** | Log-log slope changes | Tangent/curvature alignment |
| **Violation Means** | Increasing information with radius | Irregular local geometry |
| **Computational Cost** | O(n × k) for k radii | O(n² × d) for dimension d |
| **Best For** | Token stability analysis | Manifold-based method validation |

---

## When to Use Each Method

### Use Robinson Method When:
- Analyzing token embeddings from LLMs
- Testing for semantic stability
- Identifying problematic tokens for prompt engineering
- Following exact methodology from the paper

### Use Geometric Method When:
- Validating manifold-based algorithms
- Analyzing general high-dimensional data
- Need detailed geometric properties
- Testing prerequisites for manifold methods

---

## Interpreting Combined Results

### Both Pass (Rare)
- Data has smooth fiber bundle structure
- Safe to use manifold-based methods
- Tokens are semantically stable

### Robinson Fails, Geometric Passes
- Local geometry is smooth BUT
- Global volume growth is irregular
- Manifold methods may work locally but fail globally

### Robinson Passes, Geometric Fails
- Volume growth is regular BUT
- Local geometry is irregular
- May have noise or outliers affecting local structure

### Both Fail (Common for LLM embeddings)
- Neither manifold nor fiber bundle hypothesis holds
- Use robust, assumption-free methods
- Expect semantic instability

---

## Practical Example: Testing Token Embeddings

```python
from robinson_fiber_bundle_test import RobinsonFiberBundleTest
from fiber_bundle_core import FiberBundleTest
import numpy as np

# Load token embeddings (e.g., from GPT-2)
embeddings = load_embeddings()  # Shape: (vocab_size, embed_dim)

# Test with Robinson method
robinson_test = RobinsonFiberBundleTest()
robinson_result = robinson_test.test_point(embeddings, token_idx=42)

# Test with geometric method
geometric_test = FiberBundleTest()
geometric_result = geometric_test.test_point(embeddings, point_idx=42)

# Interpret results
if robinson_result.violates_hypothesis:
    print(f"Token violates Robinson test (p={robinson_result.p_value:.4f})")
    print(f"  Increasing slopes: {robinson_result.increasing_slopes}")
    print(f"  Max slope increase: {robinson_result.max_slope_increase:.3f}")
    print("  → Token may cause semantic instability")

if geometric_result.reject_null:
    print(f"Token violates geometric test (p={geometric_result.p_value:.4f})")
    print(f"  Dimension inconsistency: {geometric_result.dimension_consistency:.3f}")
    print(f"  Tangent misalignment: {geometric_result.tangent_alignment:.3f}")
    print("  → Local geometry is irregular")
```

---

## Our Test Result: "84% Fiber Bundle"

When we reported "84% of points satisfy fiber bundle structure", this means:
- 84% of tested points PASSED our geometric tests
- They have consistent local dimension, regular curvature, and aligned tangents
- This does NOT mean they pass Robinson's test

**Important**: Robinson et al. found that most LLM tokens FAIL their test, showing increasing slopes. Our 84% pass rate for geometric properties is testing something different - local smoothness rather than volume growth patterns.

The discrepancy highlights that:
1. Local geometric regularity ≠ proper volume scaling
2. Data can be locally smooth but globally irregular
3. Both perspectives are valuable for understanding embedding structure

---

## References

Robinson, M., Dey, S., & Chiang, T. (2024). Token embeddings violate the manifold hypothesis. arXiv preprint arXiv:2504.01002.

---

## Citation

If using these implementations, please cite:

```bibtex
@article{robinson2024token,
  title={Token embeddings violate the manifold hypothesis},
  author={Robinson, Michael and Dey, Sourya and Chiang, Tony},
  journal={arXiv preprint arXiv:2504.01002},
  year={2024}
}
```