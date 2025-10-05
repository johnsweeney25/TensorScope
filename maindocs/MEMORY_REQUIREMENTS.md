# Memory Requirements & Computational Complexity

Detailed memory budgets and computational complexity for TensorScope metrics.

[← Back to README](../README.md)

---

## Overview

TensorScope's memory usage varies significantly by metric. This guide helps you:
1. Understand which metrics scale to your model size
2. Configure memory-efficient alternatives
3. Estimate runtime and memory requirements

---

## Scale Summary

| Model Size | Status | Notes |
|------------|--------|-------|
| ≤ 7B params | ✅ Fully tested | All metrics, all configurations |
| 7B - 70B | ⚠️ Tested, some limits | Most metrics work with configuration |
| > 70B | ⚠️ Experimental | Core metrics only, requires tuning |

---

## Metric-Specific Requirements

### Fisher Information & Curvature

**Per-block Fisher Spectrum**
- **Complexity**: O(block_size³) per block
- **Memory**: ~4GB for 1024×1024 blocks
- **Scalability**: ✅ Excellent (blocks are small)
- **Example**: 7B model with 32 layers × 4096 hidden → 32 blocks of ~4096×4096 → tractable

**Fisher/Hessian Lanczos (Top-k eigenvalues)**
- **Complexity**: O(k × n × iterations) where n = parameters
- **Memory**: ~48GB saved vs full decomposition for 1.5B models
- **Scalability**: ✅ Excellent (k << n)
- **Typical**: k=20, iterations=100

**Grouped Fisher (Coarse approximation)**
- **Complexity**: O(groups × samples)
- **Memory**: ~1.4MB for 7B model (3,700× reduction from 5GB)
- **Scalability**: ✅ Excellent
- **Trade-off**: Coarse importance estimates

**Full Hessian**
- **Complexity**: O(n²) storage, O(n³) computation
- **Memory**: Intractable beyond ~100M parameters
- **Scalability**: ❌ Poor
- **Alternative**: Use Lanczos for top eigenvalues

### Attention Analysis

**Attention Patterns (Entropy, Flow)**
- **Complexity**: O(seq_len² × heads × layers)
- **Memory per layer**: seq_len² × heads × 4 bytes (float32)
- **Example**: 
  - seq_len=512, 32 heads, 32 layers: ~1GB
  - seq_len=2048, 32 heads, 32 layers: ~16GB
  - seq_len=4096, 32 heads, 32 layers: ~64GB ⚠️
- **Scalability**: ⚠️ Moderate (quadratic in sequence length)
- **Mitigation**: Use chunking for long sequences

**QKOV Circuit Analysis**
- **Complexity**: O(seq_len² × heads²)
- **Memory**: Similar to attention patterns
- **Scalability**: ⚠️ Moderate
- **Mitigation**: Analyze subset of heads

### Per-Sample Statistics

**Sample-level Conflict Detection**
- **Complexity**: O(batch_size × parameters) for gradients
- **Memory**: Without compression: ~4 bytes × batch_size × parameters
- **Example**: 
  - 7B model, batch=32: ~896GB ❌
  - 7B model, batch=32, int8+zlib: ~28GB ✅
- **Scalability**: ⚠️ Poor without compression
- **Mitigation**: int8+zlib compression (32× reduction), importance gating

**TracIn (Training Data Influence)**
- **Complexity**: O(checkpoints × train_samples × parameters)
- **Memory**: Requires storing gradients per checkpoint
- **Scalability**: ⚠️ Moderate
- **Mitigation**: Checkpoint subsampling, gradient compression

### Geometry & Manifold Analysis

**Embedding Singularities**
- **Complexity**: O(vocab_size × embedding_dim²)
- **Memory**: ~1GB for 50k vocab, 4096 dim
- **Scalability**: ✅ Good
- **Note**: Results are relative indicators on discrete manifolds

**Robinson Fiber Bundle Test**
- **Complexity**: O(samples × embedding_dim³)
- **Memory**: ~2GB for 1000 samples, 4096 dim
- **Scalability**: ✅ Good (GPU-accelerated)

### Representation Analysis

**Superposition Regime**
- **Complexity**: O(features × samples)
- **Memory**: ~500MB for typical analysis
- **Scalability**: ✅ Good

**Linear CKA**
- **Complexity**: O(samples × hidden_dim²)
- **Memory**: ~1GB for 1000 samples, 4096 dim
- **Scalability**: ✅ Good

---

## Memory Budget Table

### 7B Model on 80GB GPU

| Metric | Batch Size | Seq Len | Memory | Time | Status |
|--------|-----------|---------|--------|------|--------|
| Fisher Spectrum | 32 | 512 | ~8GB | ~5min | ✅ |
| Lanczos (k=20) | 32 | 512 | ~4GB | ~2min | ✅ |
| Attention Analysis | 32 | 512 | ~2GB | ~1min | ✅ |
| Attention Analysis | 32 | 2048 | ~16GB | ~8min | ✅ |
| Attention Analysis | 32 | 4096 | ~64GB | ~30min | ⚠️ |
| Sample Conflicts (compressed) | 32 | 512 | ~28GB | ~10min | ✅ |
| Sample Conflicts (uncompressed) | 32 | 512 | ~896GB | N/A | ❌ |
| TracIn (10 checkpoints) | 32 | 512 | ~40GB | ~20min | ✅ |

### 70B Model on 80GB GPU

| Metric | Batch Size | Seq Len | Memory | Time | Status |
|--------|-----------|---------|--------|------|--------|
| Fisher Spectrum | 8 | 512 | ~60GB | ~30min | ✅ |
| Lanczos (k=20) | 8 | 512 | ~30GB | ~15min | ✅ |
| Attention Analysis | 8 | 512 | ~10GB | ~5min | ✅ |
| Attention Analysis | 8 | 2048 | ~70GB | ~40min | ⚠️ |
| Sample Conflicts (compressed) | 8 | 512 | ~70GB | ~60min | ⚠️ |
| Full Hessian | Any | Any | OOM | N/A | ❌ |

---

## Configuration Recommendations

### For 7B Models

```python
analyzer = UnifiedModelAnalyzer(
    batch_size=32,
    max_seq_len=512,  # Increase to 2048 if you have memory
    use_gradient_compression=True,  # For sample-level stats
    attention_chunking=False,  # Not needed at 512 seq_len
)
```

### For 70B Models

```python
analyzer = UnifiedModelAnalyzer(
    batch_size=8,  # Reduce batch size
    max_seq_len=512,  # Keep sequences short
    use_gradient_compression=True,  # Essential
    attention_chunking=True,  # Chunk attention computation
    skip_expensive_metrics=['full_hessian'],  # Skip intractable metrics
)
```

### For Long Sequences (> 2048)

```python
analyzer = UnifiedModelAnalyzer(
    batch_size=16,
    max_seq_len=4096,
    attention_chunking=True,  # Essential
    attention_chunk_size=512,  # Process 512 tokens at a time
    use_gradient_compression=True,
)
```

---

## Determinism & Reproducibility

### Bit-Exact Reproducibility: **Not Guaranteed**

**Why:**
- Floating-point operations are not associative: `(a + b) + c ≠ a + (b + c)` in float32
- GPU parallelism can change operation order
- Different hardware (A100 vs H100) may use different algorithms

**What we guarantee:**
- **Statistical reproducibility**: Differences < 1e-6 (negligible for research)
- **Deterministic with flags**: Set `torch.use_deterministic_algorithms(True)`
- **Same hardware reproducibility**: Bit-exact on same GPU with same seed

### Achieving Maximum Reproducibility

```python
import torch
import random
import numpy as np

# Set all seeds
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)

# Enable deterministic algorithms
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Note: This may reduce performance by 10-20%
```

**Expected behavior:**
- ✅ Same GPU, same seed: Bit-exact (< 1e-12 difference)
- ✅ Different GPUs, same seed: Statistical match (< 1e-6 difference)
- ✅ Sufficient for publication: Yes (differences negligible)

---

## Troubleshooting OOM Errors

### Error: "CUDA out of memory" during attention analysis

**Solution:**
```python
analyzer = UnifiedModelAnalyzer(
    attention_chunking=True,
    attention_chunk_size=512,  # Reduce if still OOM
)
```

### Error: "CUDA out of memory" during sample conflict detection

**Solution:**
```python
analyzer = UnifiedModelAnalyzer(
    use_gradient_compression=True,
    gradient_compression_ratio=32,  # int8 + zlib
    importance_gating_threshold=0.01,  # Only store top 1% gradients
)
```

### Error: "CUDA out of memory" during Fisher spectrum

**Solution:**
- Fisher spectrum is computed per-block, so OOM is rare
- If it occurs, reduce batch size:
```python
analyzer = UnifiedModelAnalyzer(batch_size=8)
```

---

## Performance Tips

1. **Use Lanczos over full spectrum** for large models (48GB saved)
2. **Enable gradient compression** for sample-level analysis (32× reduction)
3. **Chunk attention** for sequences > 2048 (prevents quadratic blowup)
4. **Skip full Hessian** for models > 1B parameters (use Lanczos instead)
5. **Reduce batch size** if OOM (memory scales linearly with batch size)

---

## Questions?

For specific memory issues or performance optimization, see:
- [Research Recipes](RESEARCH_RECIPES.md) for metric-specific usage
- [API Reference](API_REFERENCE.md) for configuration options
- [GitHub Issues](https://github.com/johnsweeney25/tensorscope/issues) for support
