# Manifold Violations Test Suite

Comprehensive test suite for manifold violation detection modules.

## Test Coverage

### Robinson Fiber Bundle Test (`test_robinson_fiber_bundle.py`)
- **11 tests** covering:
  - Proper fiber bundle detection (non-violation case)
  - Fiber bundle violation detection (increasing slopes)
  - Manifold violation detection (no regime change)
  - Bootstrap vs exact computation equivalence
  - Edge cases (single point, two points, identical distances)
  - Numerical methods (centered slopes, CFAR detector, p-values)
  - Reach gating for spurious rejection prevention
  - Local signal dimension computation

### Polysemy Detector (`test_polysemy_detector.py`)
- **10 tests** covering:
  - Monosemous token detection (single meaning)
  - Clear polysemy detection (multiple distinct meanings)
  - Homonym detection (opposite meanings)
  - Clustering methods (DBSCAN vs hierarchical)
  - Metric consistency (cosine vs Euclidean)
  - Edge cases (single token, insufficient neighbors, identical embeddings)
  - Confidence scoring calibration
  - Large vocabulary subsampling
  - Full vocabulary analysis

## Running Tests

### Run All Tests
```bash
# From project root
python manifold_violations/tests/run_all_tests.py

# Or using pytest
pytest manifold_violations/tests/ -v
```

### Run Specific Test Module
```bash
# Robinson tests only
pytest manifold_violations/tests/test_robinson_fiber_bundle.py -v

# Polysemy tests only
pytest manifold_violations/tests/test_polysemy_detector.py -v
```

### Run with Coverage
```bash
pytest manifold_violations/tests/ --cov=manifold_violations --cov-report=html
```

## Test Design Philosophy

1. **Rigorous**: Tests verify both positive and negative cases
2. **Edge-case aware**: Handles degenerate inputs gracefully
3. **Statistically sound**: Tests use proper seeds and statistical methods
4. **Performance conscious**: Includes tests for large-scale data
5. **Realistic**: Test data mimics real embedding structures

## Key Test Insights

### Robinson Test Findings
- Requires at least 10 valid radii for reliable hypothesis testing
- Bootstrap sampling gives comparable results to exact computation
- CFAR detector needs minimum 15 points for slope change detection
- P-value computation uses proper one-sided Mann-Kendall test

### Polysemy Detector Findings
- Hierarchical clustering with elbow method threshold selection
- Confidence scores combine cluster count, separation quality, and distance variance
- Handles both cosine and Euclidean metrics appropriately
- Subsampling effective for vocabularies >10,000 tokens

## Test Results Summary

```
======================================================================
TEST SUMMARY
======================================================================
Tests run: 21
Failures: 0
Errors: 0
Skipped: 0
```

All tests pass with only expected warnings for edge cases.

## Future Test Additions

1. **Singularity Mapper Tests** - Test comprehensive singularity profiling
2. **Prompt Robustness Tests** - Test per-token risk assessment
3. **Integration Tests** - Test full pipeline from embeddings to risk scores
4. **Performance Benchmarks** - Track runtime for different vocabulary sizes
5. **Cross-model Tests** - Verify consistency across different model architectures

## Continuous Integration

Add to `.github/workflows/test.yml`:
```yaml
- name: Run Manifold Violation Tests
  run: |
    pip install -r requirements.txt
    python manifold_violations/tests/run_all_tests.py
```

---
*Test suite created: 2025-09-29*
*All tests passing for ICML submission*