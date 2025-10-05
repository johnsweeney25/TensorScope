# Superposition Analysis Package

A comprehensive framework for analyzing feature superposition in neural networks, implementing metrics from recent papers on neural scaling laws and representation capacity.

## Package Structure

```
superposition/
├── __init__.py              # Main package exports
├── core/                    # Core implementations
│   ├── __init__.py
│   ├── base.py             # Original SuperpositionMetrics
│   ├── enhanced.py         # Enhanced version with GPU/numerical fixes
│   └── analyzer.py         # Optimized analyzer with caching
├── metrics/                 # Specialized metrics
│   ├── __init__.py
│   └── paper_metrics.py    # Paper-specific metrics (ϕ₁/₂, ϕ₁, regimes)
├── tests/                   # Unit tests (all using unittest)
│   ├── __init__.py
│   ├── test_base.py        # Tests for base metrics
│   ├── test_enhanced.py    # Tests for enhanced version
│   ├── test_analyzer.py    # Tests for analyzer
│   └── run_all_tests.py    # Test runner script
└── docs/                    # Documentation
    └── SUPERPOSITION_CODE_REVIEW_SUMMARY.md

```

## Key Features

### 1. Comprehensive Superposition Analysis
- **Vector interference**: Measure feature overlap and orthogonality
- **Paper metrics**: ϕ₁/₂ (fraction with ||W|| > 0.5), ϕ₁ (fraction with ||W|| > 1.0)
- **Regime classification**: Automatic detection of weak/strong/no superposition
- **Scaling analysis**: Power law fitting for neural scaling laws
- **Geometric analysis**: Welch bounds, √(1/m) scaling verification

### 2. Optimized Implementation
- **Intelligent caching**: ~4.6x speedup on repeated analyses
- **GPU memory management**: Handles 100K+ features efficiently
- **Numerical stability**: Welford's algorithm, proper epsilon values
- **Batch processing**: Memory-efficient computation for large matrices

### 3. Paper Implementations
From "Superposition Yields Robust Neural Scaling" (Liu et al., 2025):
- ϕ₁/₂ and ϕ₁ metrics for quantifying superposition
- Weak vs strong superposition regime classification
- Geometric overlap analysis with Welch bound comparison

## Usage

### Basic Usage

```python
from superposition import SuperpositionAnalyzer

# Create analyzer
analyzer = SuperpositionAnalyzer()

# Analyze weight matrix
weight_matrix = model.embedding.weight
result = analyzer.compute_comprehensive_superposition_analysis(weight_matrix)

print(f"Regime: {result.regime}")
print(f"ϕ₁/₂: {result.phi_half:.3f}")  # Fraction with ||W|| > 0.5
print(f"ϕ₁: {result.phi_one:.3f}")      # Fraction with ||W|| > 1.0
print(f"Mean overlap: {result.mean_overlap:.4f}")
```

### Analyze Entire Model

```python
from superposition import analyze_model_superposition_comprehensive

results = analyze_model_superposition_comprehensive(model)

for layer_name, analysis in results.items():
    print(f"{layer_name}: {analysis.regime} (ϕ₁/₂={analysis.phi_half:.3f})")
```

### Custom Configuration

```python
from superposition import SuperpositionConfig, SuperpositionAnalyzer

config = SuperpositionConfig(
    eps=1e-8,
    overlap_threshold=0.1,
    max_batch_size=1000,
    cleanup_cuda_cache=True
)

analyzer = SuperpositionAnalyzer(config=config)
```

## Running Tests

```bash
# Run all tests
python superposition/tests/run_all_tests.py

# Run specific test module
python superposition/tests/run_all_tests.py test_analyzer

# Run with different verbosity
python superposition/tests/run_all_tests.py -v 1  # Less verbose
```

## Integration with TensorScope

This package is integrated with TensorScope's `UnifiedModelAnalysis`:

```python
from unified_model_analysis import UnifiedModelAnalysis

uma = UnifiedModelAnalysis()
# SuperpositionAnalyzer is automatically loaded at uma.modules['superposition']
```

## Key Metrics Explained

### ϕ₁/₂ (phi_half)
- Fraction of features with L2 norm > 0.5
- Indicates how many features are "represented" in the model
- ϕ₁/₂ ≈ 1 suggests strong superposition (all features represented)

### ϕ₁ (phi_one)
- Fraction of features with L2 norm > 1.0
- Indicates how many features are "strongly represented"
- High ϕ₁ with high ϕ₁/₂ indicates very strong superposition

### Superposition Regimes
- **No superposition**: ϕ₁/₂ ≈ m/n (only m features in m dimensions)
- **Weak superposition**: ϕ₁/₂ > m/n but ϕ₁ ≈ 0 (some features ignored)
- **Strong superposition**: ϕ₁/₂ ≈ 1 (all features represented with interference)

### Welch Bound
- Theoretical minimum overlap for n vectors in d dimensions
- If actual overlap ≈ Welch bound, vectors are optimally packed
- Strong superposition often approaches the Welch bound

## Performance

- Handles matrices up to 100K+ features
- 4.6x speedup with caching on repeated analyses
- GPU-optimized with automatic memory management
- Numerically stable across scales from 1e-10 to 1e10

## References

1. "Superposition Yields Robust Neural Scaling" (Liu et al., 2025)
2. "Toy Models of Superposition" (Anthropic, 2022)
3. TensorScope documentation

## License

Part of the TensorScope project.