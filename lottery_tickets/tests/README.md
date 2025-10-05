# Lottery Tickets Test Suite

Comprehensive unit tests for the lottery ticket hypothesis implementation, designed for the ICML submission.

## Structure

```
lottery_tickets/tests/
├── __init__.py                   # Test suite initialization
├── run_tests.py                  # Main test runner
├── test_ggn_verification.py      # GGN theoretical correctness tests
├── test_importance_scoring.py    # Fisher/Taylor importance tests
├── test_magnitude_pruning.py     # Magnitude pruning tests
└── README.md                     # This file
```

## Running Tests

### Run All Tests
```bash
python run_tests.py
```

### Run Specific Module Tests
```bash
python run_tests.py --module ggn        # GGN verification tests
python run_tests.py --module importance # Importance scoring tests
python run_tests.py --module pruning    # Magnitude pruning tests
```

### Run with Options
```bash
python run_tests.py --verbose           # Verbose output
python run_tests.py --failfast          # Stop on first failure
python run_tests.py --list              # List available modules
```

### Run Individual Test Cases
```bash
# Using unittest directly
python -m unittest test_ggn_verification.TestGGNTheoretical
python -m unittest test_importance_scoring.TestFisherImportance.test_fisher_computation_basic
```

## Test Coverage

### 1. GGN Verification (`test_ggn_verification.py`)
- **TestGGNTheoretical**: Validates theoretical relationships
  - Fisher-GGN equivalence for cross-entropy loss
  - Empirical vs true Fisher differences
  - Numerical precision with different dtypes

- **TestLotteryTicketIntegration**: Integration tests
  - Fisher importance computation
  - Magnitude pruning with histogram quantiles
  - Pruning robustness metrics

- **TestMultiBatchHessian**: Multi-batch computation
  - HVP averaging across batches
  - Variance reduction verification

- **TestUtilityFunctions**: Utility function tests
  - Histogram quantile approximation
  - Model wrapper compatibility
  - Deterministic pruning setup

### 2. Importance Scoring (`test_importance_scoring.py`)
- **TestFisherImportance**: Fisher information tests
  - Basic computation correctness
  - Mixed precision (FP32 accumulation)
  - Gradient clipping for stability
  - Chunked processing efficiency

- **TestTaylorImportance**: Taylor expansion tests
  - Gradient-weight product computation
  - Comparison with magnitude baseline

- **TestMagnitudeImportance**: Magnitude-based tests
  - Weight magnitude extraction
  - Bias parameter exclusion

- **TestGradientNormImportance**: Gradient norm tests
  - L2 norm computation
  - Multi-sample averaging

- **TestHybridImportance**: Hybrid method tests
  - Weight combination strategies
  - Score normalization

### 3. Magnitude Pruning (`test_magnitude_pruning.py`)
- **TestMaskCreation**: Mask generation tests
  - Binary mask validation
  - Histogram vs direct quantile comparison
  - Different sparsity levels
  - Weight-only parameter selection

- **TestPruningRobustness**: Robustness evaluation
  - Performance retention metrics
  - Sparsity curves generation
  - Winning ticket identification
  - Critical sparsity detection

- **TestLotteryTicketFinding**: Lottery ticket tests
  - Global magnitude ranking
  - Layer-wise pruning
  - Importance-weighted pruning
  - Global vs layer-wise comparison

- **TestMaskOperations**: Mask manipulation tests
  - Mask application to model
  - Weight restoration
  - Noise injection for removed weights

## Key Features Tested

### Theoretical Correctness
- ✅ True Fisher = GGN for cross-entropy (verified)
- ✅ Empirical vs true Fisher difference is expected
- ✅ All matrices are PSD (positive semi-definite)

### Memory Efficiency
- ✅ Histogram-based quantiles (O(bins) memory)
- ✅ Chunked parameter processing
- ✅ FP32 accumulation for BF16/FP16 models

### Numerical Stability
- ✅ Gradient clipping for Fisher computation
- ✅ Mixed precision handling
- ✅ Condition number monitoring

### Reproducibility
- ✅ Deterministic seeding
- ✅ Consistent results across runs
- ✅ Platform-independent behavior

## Expected Test Results

All tests should pass with the following characteristics:

1. **GGN Eigenvalues**: ~3x difference between empirical and true (expected for random models)
2. **Fisher Scores**: Non-negative (PSD property)
3. **Pruning Robustness**: Winning ticket score > 0
4. **Histogram Quantiles**: Within 0.1 of true quantiles
5. **Sparsity Accuracy**: Within 10% of target

## ICML Submission Notes

These tests verify:
1. **Reproducibility**: Fixed seeds ensure deterministic results
2. **Theoretical Soundness**: GGN-Fisher relationships validated
3. **Numerical Precision**: FP32/FP64 stability confirmed
4. **Memory Efficiency**: Histogram methods scale to large models
5. **Statistical Validity**: Multi-batch averaging reduces variance

## Troubleshooting

### Import Errors
```bash
# Ensure parent directory is in path
export PYTHONPATH="${PYTHONPATH}:/Users/john/ICLR 2026 proj/pythonProject"
```

### CUDA Out of Memory
- Tests use small models by default
- Reduce batch sizes if needed
- Use CPU for debugging: `CUDA_VISIBLE_DEVICES="" python run_tests.py`

### Test Failures
- Check random seed is set (should print "Deterministic mode enabled")
- Verify all dependencies are installed
- Run with `--verbose` for detailed output

## Contributing

When adding new tests:
1. Follow unittest conventions
2. Include setUp() for reproducibility
3. Use descriptive test names
4. Add docstrings explaining what's tested
5. Update this README with new test descriptions