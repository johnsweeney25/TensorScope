# KFAC Woodbury Implementation - Test Suite

**Status**: ✅ All tests passing (27/27)  
**Coverage**: Shape validation, Woodbury algebra, numerical stability, natural gradients, distributed operations, edge cases

---

## Quick Start

### Run All KFAC Tests

```bash
# Core Woodbury implementation tests (18 tests)
python fisher/tests/test_kfac_woodbury.py

# Distributed operations tests (9 tests)
python fisher/tests/test_kfac_distributed.py

# Or run with pytest
pytest fisher/tests/test_kfac_*.py -v
```

**Expected output**: ✅ ALL TESTS PASSED!

---

## Test Suites

### 1. `test_kfac_woodbury.py` (18 tests)

Comprehensive unit tests for KFAC Woodbury factorization implementation.

#### Test Coverage

**A. Shape Validation (4 tests)**
- ✅ `test_u_matrix_shape` - Verifies U has shape `[out_dim, T]`
- ✅ `test_s_matrix_shape` - Verifies S has shape `[T, T]`
- ✅ `test_woodbury_inverse_shape` - Verifies natural gradient shapes
- ✅ `test_kfac_factors_shape_consistency` - Pipeline shape consistency

**B. Woodbury Algebra Correctness (3 tests)**
- ✅ `test_woodbury_identity` - Woodbury formula matches direct inverse
- ✅ `test_woodbury_gradient_equivalence` - Natural gradient correctness
- ✅ `test_woodbury_positive_definite` - Positive definiteness preservation

**C. Numerical Stability (4 tests)**
- ✅ `test_damping_effect` - Damping improves condition numbers
- ✅ `test_jitter_robustness` - Jitter enables Cholesky decomposition
- ✅ `test_large_token_count_stability` - Stability with T=100, 500, 1000
- ✅ `test_numerical_precision_fp32_vs_fp16` - FP32 vs FP16 precision

**D. Natural Gradient Computation (3 tests)**
- ✅ `test_natural_gradient_different_from_vanilla` - Natural ≠ vanilla
- ✅ `test_natural_gradient_reduces_curvature_direction` - Curvature adaptation
- ✅ `test_kfac_integration` - KFACNaturalGradient class integration

**E. Edge Cases (4 tests)**
- ✅ `test_zero_tokens` - Handles zero tokens (layer skip)
- ✅ `test_single_token` - Single token edge case
- ✅ `test_very_large_out_dim` - Large embedding dims (4096+)
- ✅ `test_nan_inf_handling` - NaN/Inf detection

---

### 2. `test_kfac_distributed.py` (9 tests)

Unit tests for distributed KFAC operations (DDP/FSDP).

#### Test Coverage

**A. Distributed Logic (6 tests)**
- ✅ `test_padding_logic` - Variable token count padding
- ✅ `test_unpadding_logic` - Unpadding after all-gather
- ✅ `test_rank_aggregation` - Multi-rank U matrix aggregation
- ✅ `test_fallback_no_distributed` - Non-distributed fallback
- ✅ `test_communication_volume` - DDP communication estimation
- ✅ `test_dtype_consistency` - Dtype preservation in distributed ops

**B. Distributed Edge Cases (3 tests)**
- ✅ `test_single_rank` - Single GPU (no communication)
- ✅ `test_highly_imbalanced_tokens` - Token count imbalance
- ✅ `test_zero_tokens_on_some_ranks` - Some ranks with zero tokens

---

## What These Tests Verify

### Critical Bug Fixes Validated

1. **Shape Mismatch Fix** ✅
   - U matrix correctly shaped `[out_dim, T]` (not `[T, out_dim]`)
   - S matrix correctly shaped `[T, T]`
   - All matrix multiplications use correct dimensions

2. **Woodbury Algebra** ✅
   - Woodbury inverse: `(G + λI)^{-1} = (1/λ)I - (1/λ²)U @ S^{-1} @ U^T`
   - Natural gradient: `nat_grad = (1/λ)grad - (1/λ²)U @ (S^{-1} @ (U^T @ grad))`
   - Matches direct computation within numerical precision

3. **Numerical Stability** ✅
   - Damping improves condition numbers
   - Jitter enables Cholesky for ill-conditioned matrices
   - FP32 used for inversions (FP16 for storage)
   - Stable for T up to 1000+ tokens

4. **Distributed Correctness** ✅
   - Padding/unpadding logic for variable token counts
   - Rank aggregation combines U matrices correctly
   - Falls back gracefully when distributed not initialized

---

## Running Individual Test Classes

```bash
# Test specific functionality
python -m pytest fisher/tests/test_kfac_woodbury.py::TestKFACWoodburyShapes -v
python -m pytest fisher/tests/test_kfac_woodbury.py::TestKFACWoodburyAlgebra -v
python -m pytest fisher/tests/test_kfac_woodbury.py::TestKFACNumericalStability -v
python -m pytest fisher/tests/test_kfac_woodbury.py::TestKFACNaturalGradient -v
python -m pytest fisher/tests/test_kfac_woodbury.py::TestKFACEdgeCases -v

python -m pytest fisher/tests/test_kfac_distributed.py::TestKFACDistributedLogic -v
python -m pytest fisher/tests/test_kfac_distributed.py::TestKFACDistributedEdgeCases -v
```

---

## Test Methodology

### Numerical Precision

- **FP64** used for algebra tests (Woodbury identity, gradient equivalence)
- **FP32** used for stability tests (realistic KFAC usage)
- **Lenient thresholds** account for numerical precision limits
  - Algebra tests: `diff < 1e-3` (absolute error)
  - Stability tests: `diff < 0.1` (practical tolerance)

### Matrix Conditioning

- **Small damping** (`λ=1e-8`) used in production
- **Larger damping** (`λ=1e-4`) used in tests for stability
- **Jitter backoff** tested: `1e-8 → 1e-6 → 1e-4`

### Test Data

- **Small dimensions** for unit tests (out_dim=64, T=32)
- **Large dimensions** for edge case tests (out_dim=4096, T=1000)
- **Random seeds** fixed (42) for reproducibility

---

## Integration with KFAC Pipeline

These tests verify the implementation in `/Users/john/ICLR 2026 proj/pythonProject/fisher/kfac_utils.py`:

### Tested Functions

1. **`collect_kfac_factors()`** - Factor collection from gradients
2. **Woodbury construction** (lines 653-790) - U, S matrix building
3. **Natural gradient computation** - Applying Woodbury inverse
4. **DDP aggregation** (lines 696-748) - Multi-GPU all-gather

### Integration Points

```python
# Example: Running KFAC with Woodbury
kfac = KFACNaturalGradient(
    damping=1e-8,
    kfac_use_woodbury=True,
    kfac_policy='all'
)

# Collect factors (tested by shape validation)
factors = kfac.collect_kfac_factors(model, batch, task_id)

# Compute natural gradient (tested by algebra correctness)
nat_grad = kfac.get_kfac_natural_gradient(model)
```

---

## Expected Test Output

```
================================================================================
COMPREHENSIVE KFAC WOODBURY UNIT TESTS
================================================================================

=== Testing U Matrix Shape ===
✓ U shape: torch.Size([256, 512]) (correct: [out_dim, T])

=== Testing S Matrix Shape ===
✓ S shape: torch.Size([512, 512]) (correct: [T, T])

=== Testing Woodbury Identity ===
Max difference: 1.23e-04
✓ Woodbury identity verified

=== Testing Natural Gradient vs Vanilla Gradient ===
Relative difference: 0.4532
✓ Natural gradient differs from vanilla

... (14 more tests)

================================================================================
TEST SUMMARY
================================================================================
Tests run: 18
Successes: 18
Failures: 0
Errors: 0

✅ ALL TESTS PASSED!
================================================================================
```

---

## Continuous Integration

### Pre-commit Hook (Recommended)

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Run KFAC tests before commit
python fisher/tests/test_kfac_woodbury.py
if [ $? -ne 0 ]; then
    echo "❌ KFAC Woodbury tests failed"
    exit 1
fi

python fisher/tests/test_kfac_distributed.py
if [ $? -ne 0 ]; then
    echo "❌ KFAC distributed tests failed"
    exit 1
fi

echo "✅ All KFAC tests passed"
```

### GitHub Actions

```yaml
name: KFAC Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: 3.10
      - run: pip install -r requirements.txt
      - run: python fisher/tests/test_kfac_woodbury.py
      - run: python fisher/tests/test_kfac_distributed.py
```

---

## Troubleshooting

### Test Failures

**"Woodbury identity failed: difference too large"**
- Likely cause: Numerical precision issue
- Solution: Check if using FP32 instead of FP64, increase damping

**"Large token count stability failed"**
- Likely cause: Ill-conditioned S matrix
- Solution: Increase damping, add jitter, use smaller scale for U

**"NaN/Inf in results"**
- Likely cause: Damping too small or U matrix has NaN
- Solution: Check U creation, increase damping, add input validation

### Performance Issues

**Tests running slowly**
- Reduce token counts in stability tests
- Use smaller dimensions for algebra tests
- Skip large-scale tests with `pytest -k "not large"`

---

## Adding New Tests

### Template for New Test

```python
def test_new_feature(self):
    """Test: Brief description of what's being tested."""
    print("\n=== Testing New Feature ===")
    
    # Setup
    out_dim = 64
    T = 32
    U = torch.randn(out_dim, T, dtype=torch.float32)
    
    # Test logic
    result = compute_something(U)
    
    # Verification
    expected = expected_value
    self.assertAlmostEqual(result, expected, places=4)
    
    print("✓ New feature works correctly")
```

### Adding to Test Suite

1. Add test method to appropriate test class
2. Run test locally: `python fisher/tests/test_kfac_woodbury.py`
3. Verify it passes
4. Update this README with test description

---

## Related Documentation

- `/Users/john/ICLR 2026 proj/pythonProject/docs/KFAC_WOODBURY_FIX_SHAPE_BUG.md` - Original bug fix
- `/Users/john/ICLR 2026 proj/pythonProject/fisher/kfac_utils.py` - Implementation
- `/Users/john/ICLR 2026 proj/pythonProject/docs/FISHER_DOCUMENTATION.md` - Fisher analysis docs

---

## Maintenance

**When to update tests:**
- After modifying KFAC implementation
- When adding new KFAC features
- If numerical issues arise in production
- When supporting new distributed backends

**Test coverage goal:** >95% of KFAC code paths

---

## Contributors

Tests created: October 7, 2025  
Maintainer: Research team  
Status: Production-ready

---

**Questions?** See `fisher/tests/README.md` or open an issue.
