# MDL and Compression Analysis Documentation

## Overview

This document describes two complementary approaches for analyzing model complexity and compressibility:

1. **MDL Complexity (`mdl_complexity_proper.py`)** - Theoretical minimum description length
2. **Practical Compression (`InformationTheoryMetrics.compute_practical_compression_ratio`)** - Empirical compression testing

Both methods provide insights into model complexity but from different perspectives.

---

## 1. MDL Complexity Analysis

### Purpose
Implements proper Minimum Description Length (MDL) theory for neural networks following Rissanen (1978) and Grünwald (2007). Provides a theoretically grounded measure of model complexity for model selection and theoretical analysis.

### Key Concepts

#### Two-Part Code Principle
MDL decomposes total complexity as:
```
L(M, D) = L(M) + L(D|M)
```
where:
- `L(M)` = Model description length (architecture + parameters)
- `L(D|M)` = Data description length given the model

### Usage

```python
from mdl_complexity_proper import MDLComplexity

# Initialize
mdl = MDLComplexity(epsilon=1e-8)

# Compute MDL complexity
results = mdl.compute_mdl_complexity(
    model=model,
    data_loader=data_loader,  # Optional: for L(D|M)
    param_bits_per_layer=8,    # Quantization bits
    architecture_mode="universal",  # or "heuristic"
    max_data_samples=1000
)

# Results include:
print(f"Architecture bits: {results['architecture_bits']:.2f}")
print(f"Parameter bits: {results['parameter_bits']:.2f}")
print(f"Data bits: {results.get('data_bits', 'N/A')}")
print(f"Total MDL: {results['total_mdl']:.2f}")
print(f"Compression ratio: {results['compression_ratio']:.2f}x")
```

### Components

#### 1. Architecture Complexity (`L_architecture`)
Uses universal integer codes (Rissanen) to encode:
- Layer types from fixed vocabulary
- Dimensions with L*(n) = log₂(n) + log₂(log₂(n)) + ...
- Hyperparameters (kernel sizes, strides, etc.)
- Connectivity patterns

**Universal mode** (recommended):
```python
bits = Σ [layer_type_bits + dimension_bits + hyperparameter_bits]
```

**Heuristic mode** (simplified):
```python
bits = 5 * n_layers + Σ log₂(dimensions)
```

#### 2. Parameter Complexity (`L_parameters`)
Uses **quantize-then-entropy** approach (MDL-consistent):

1. Quantize weights: `q = round(w/Δ)` where `Δ = σ√12 · 2^(-B)`
2. Compute discrete entropy: `H(q) = -Σ p(q)log₂(p(q))`
3. Total bits: `N·H(q) + overhead`

Overhead includes:
- Quantization step Δ: 32 bits
- Index range: 2 × ceil(log₂(range)) bits

#### 3. Data Complexity (`L_data`)
Negative log-likelihood in bits using **sum-reduction**:
```python
L(D|M) = Σ -log₂ P(y|x,M) = CrossEntropy(y, ŷ) / ln(2)
```

For transformers: correctly ignores masked tokens (labels=-100)

#### 4. Compression Upper Bound
Tests actual compressibility using real codecs (zlib, bzip2, lzma) on raw tensor bytes without pickle overhead.

### Fisher Information Diagnostic

**Not** a proper MDL term - provided as diagnostic only:
```python
fisher_results = mdl.compute_fisher_diagnostic_bits(
    model, data_loader, max_batches=100
)
```

Returns:
- `fisher_diagnostic_bits`: 0.5 × log|F| (incomplete without k/2·log(n) - log(p(θ)))
- `effective_dimensionality`: Model's effective parameter count

### Weight Entropy Spectrum

Analyzes entropy distribution across layers:
```python
spectrum = mdl.compute_weight_entropy_spectrum(model)
# Returns per-layer entropy after quantization
```

---

## 2. Practical Compression Analysis

### Purpose
Empirically measures actual compressibility using real compression algorithms. Provides practical insights into model redundancy and compressibility.

### Usage

```python
from InformationTheoryMetrics import InformationTheoryMetrics

info = InformationTheoryMetrics()
result = info.compute_practical_compression_ratio(
    model=model,
    mode='full',  # 'sample', 'full', 'stream', or 'full+audit'
    codec_name='lzma',  # 'lzma', 'gzip', 'bzip2', etc.
    sample_percentage=0.1,  # For sample mode
    n_jobs=4,  # Parallel processing
    per_layer=True  # Get per-layer breakdown
)

print(f"Compression ratio: {result['compression_ratio']:.2f}x")
print(f"Bits per weight: {result['bits_per_weight']:.2f}")
```

### Modes

#### Sample Mode
- Randomly samples `sample_percentage` of parameters
- Provides confidence intervals
- Fast approximation

#### Full Mode
- Compresses entire model
- Most accurate but slower
- Can use parallel processing

#### Stream Mode
- Processes model in chunks
- Memory-efficient for large models
- Provides progressive results

#### Full+Audit Mode
- Complete analysis with validation
- Per-layer statistics
- Outlier detection
- Compression evolution tracking

### Features

- **Multiple Codecs**: LZMA (best), GZIP (fast), BZIP2 (balanced)
- **Parallel Processing**: Multi-CPU compression for speed
- **Validation**: Checks for realistic compression ratios
- **Edge Cases**: Handles NaN/Inf values, empty tensors
- **Per-Layer Analysis**: Identifies compressible layers

### Output Format

```python
{
    'compression_ratio': 3.2,
    'original_size_bytes': 100000000,
    'compressed_size_bytes': 31250000,
    'bits_per_weight': 10.5,
    'bits_per_weight_data_only': 9.8,  # Excluding metadata
    'per_layer': {
        'layer.0.weight': {
            'compression_ratio': 3.5,
            'bits_per_weight': 9.1,
            ...
        }
    },
    'validation': {
        'realistic': True,
        'warnings': []
    }
}
```

---

## Comparison: MDL vs Practical Compression

| Aspect | MDL Complexity | Practical Compression |
|--------|---------------|----------------------|
| **Approach** | Theoretical (entropy-based) | Empirical (actual compression) |
| **Speed** | Fast (O(n) entropy calculation) | Slower (compression overhead) |
| **Components** | Architecture + Parameters + Data | Parameters only |
| **Quantization** | Explicit (controlled bits) | Implicit (codec-dependent) |
| **Use Case** | Model selection, theory | Redundancy analysis, deployment |
| **Overhead** | Minimal (Δ, range) | Codec headers, dictionaries |
| **Interpretability** | Bits per component | Compression ratio |

### When to Use Which?

**Use MDL Complexity when:**
- Comparing different architectures
- Theoretical model selection
- Need decomposition (arch vs params vs data)
- Publishing theoretical results
- Fast complexity estimates needed

**Use Practical Compression when:**
- Analyzing model redundancy
- Planning model compression/quantization
- Deployment size estimation
- Empirical compressibility testing
- Per-layer redundancy analysis

### Relationship Between Methods

1. **MDL provides lower bound**: MDL's entropy-based parameter bits is the theoretical minimum achievable by any lossless compressor

2. **Practical compression includes overhead**: Real codecs have dictionary/header overhead that MDL theory doesn't account for

3. **Expected relationship**:
   ```
   MDL_param_bits ≤ Practical_compressed_bits
   ```

4. **Compression ratio interpretation**:
   - MDL: `unquantized_bits / quantized_entropy_bits`
   - Practical: `original_bytes / compressed_bytes`

---

## Example: Complete Analysis

```python
import torch
import torch.nn as nn
from mdl_complexity_proper import MDLComplexity
from InformationTheoryMetrics import InformationTheoryMetrics

# Create model
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# Theoretical MDL analysis
mdl = MDLComplexity()
mdl_results = mdl.compute_mdl_complexity(
    model=model,
    data_loader=train_loader,
    param_bits_per_layer=8
)

# Practical compression analysis
info = InformationTheoryMetrics()
compression_results = info.compute_practical_compression_ratio(
    model=model,
    mode='full',
    codec_name='lzma'
)

# Compare results
print("=== Theoretical MDL ===")
print(f"Architecture: {mdl_results['architecture_bits']:.0f} bits")
print(f"Parameters: {mdl_results['parameter_bits']:.0f} bits")
print(f"Data: {mdl_results.get('data_bits', 0):.0f} bits")
print(f"Total MDL: {mdl_results['total_mdl']:.0f} bits")

print("\n=== Practical Compression ===")
print(f"Compression ratio: {compression_results['compression_ratio']:.2f}x")
print(f"Bits per weight: {compression_results['bits_per_weight']:.2f}")

# Compute efficiency gap
mdl_bits_per_weight = mdl_results['parameter_bits'] / sum(
    p.numel() for p in model.parameters()
)
efficiency = mdl_bits_per_weight / compression_results['bits_per_weight']
print(f"\nCompression efficiency: {efficiency:.1%}")
```

---

## Best Practices

### For MDL Analysis
1. Use universal encoding mode for rigorous theoretical results
2. Include data_loader for complete L(M,D) analysis
3. Choose quantization bits based on deployment target
4. Report Fisher diagnostic separately (not as MDL term)

### For Practical Compression
1. Use LZMA for best compression (Kolmogorov approximation)
2. Enable parallel processing for large models
3. Use sample mode for quick estimates
4. Run full+audit for publication results

### For Comparison Studies
1. Report both MDL and practical compression
2. Analyze efficiency gap to identify codec limitations
3. Use per-layer analysis to find redundant components
4. Consider quantization effects in both methods

---

## References

1. Rissanen, J. (1978). "Modeling by shortest data description"
2. Grünwald, P. (2007). "The Minimum Description Length Principle"
3. Blier, L. & Ollivier, Y. (2018). "The Description Length of Deep Learning Models"
4. Shannon, C. (1948). "A Mathematical Theory of Communication"

---

## Troubleshooting

### MDL Issues
- **Negative Fisher diagnostic**: Model may be undertrained or have numerical issues
- **High architecture bits**: Complex non-sequential architectures need more bits
- **Low compression ratio**: Model has high entropy (well-trained, less redundant)

### Compression Issues
- **Compression < 1x**: Model contains random/encrypted data
- **Compression > 100x**: Model likely has many zeros or repeated values
- **Memory errors**: Use stream mode or reduce batch size
- **Slow compression**: Reduce compression level or use faster codec

---

## Future Enhancements

1. **Adaptive quantization**: Layer-specific bit allocation based on sensitivity
2. **Structured MDL**: Exploit weight matrix structure (low-rank, sparse)
3. **Online MDL**: Streaming computation for very large models
4. **Hybrid methods**: Combine theoretical and empirical approaches
5. **Lossy MDL**: Rate-distortion optimal quantization