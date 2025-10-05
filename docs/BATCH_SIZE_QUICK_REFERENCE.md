# Batch Size Quick Reference

## All Batch Sizes Are Set in UnifiedConfig ✅

### Location
`unified_model_analysis.py` → class `UnifiedConfig` (lines 624-706)

### Complete List

```python
# Main dataset batch size
batch_size: int = 256

# Attention analysis
attention_chunk_size: int = 96                      # Normal chunking
attention_high_memory_chunk_size: int = 32          # High-memory scenarios
attention_batch_size: int = 32                      # For attention metrics

# Integrated Gradients (token importance)
ig_chunk_size: int = 128                            # Main chunking
ig_internal_batch_size: int = 8                     # Captum's internal batching
integrated_gradients_batch_size: int = 256          # Overall batch size

# Layer-wise attribution
layer_wise_chunk_size: int = 96                     # Main chunking
min_layer_wise_chunk_size: int = 16                 # Minimum for stability
layer_wise_internal_batch_size: int = 8             # Captum's internal batching

# Jacobian computation
jacobian_batch_size: int = 32                       # VJP method

# Fisher Information
fisher_batch_size: int = 32                         # Fisher metrics
ggn_batch_size: int = 32                           # GGN/Fisher

# Hessian
hessian_batch_size: int = 16                        # Hessian eigenvalues

# Gradient metrics
gradient_batch_size: int = 256                      # General gradients
gradient_trajectory_batch_size: int = 64            # Trajectory tracking
gradient_pathology_batch_size: int = 64             # Pathology detection

# Loss landscape
loss_landscape_batch_size: int = 16                 # 25x25 grid

# Other
modularity_batch_size: int = 16                     # CKA operations
causal_batch_size: int = 32                         # Causal analysis
```

## GPU-Specific Configs

### H100 80GB (Default)
```python
config = UnifiedConfig()  # Use all defaults
```

### A100 40GB
```python
config = UnifiedConfig(
    batch_size=128,
    ig_chunk_size=64,
    layer_wise_chunk_size=48,
    attention_chunk_size=48
)
```

### RTX 4090 24GB
```python
config = UnifiedConfig(
    batch_size=64,
    ig_chunk_size=32,
    layer_wise_chunk_size=32,
    attention_chunk_size=32,
    attention_high_memory_chunk_size=16,
    ig_internal_batch_size=4,
    layer_wise_internal_batch_size=4
)
```

### V100 16GB
```python
config = UnifiedConfig(
    batch_size=32,
    ig_chunk_size=16,
    layer_wise_chunk_size=24,
    attention_chunk_size=24,
    attention_high_memory_chunk_size=8,
    ig_internal_batch_size=2,
    layer_wise_internal_batch_size=2,
    fisher_batch_size=16,
    hessian_batch_size=8
)
```

## Memory Impact Guide

| Parameter | Memory Impact | When to Reduce |
|-----------|---------------|----------------|
| `batch_size` | **Very High** | Main OOM cause |
| `ig_chunk_size` | Medium | IG OOMs |
| `layer_wise_chunk_size` | **High** | Layer-wise OOMs |
| `attention_chunk_size` | **Very High** | Attention OOMs |
| `hessian_batch_size` | **Very High** | Hessian OOMs |
| `fisher_batch_size` | High | Fisher OOMs |
| `*_internal_batch_size` | Medium | Fine-tuning |

## Rule of Thumb

**If you get OOM:**
1. Check which metric failed
2. Find corresponding `*_batch_size` parameter
3. Reduce by half
4. Retry

**Example:**
```
ERROR: OOM in layer_wise_attribution
→ Reduce layer_wise_chunk_size from 96 to 48
→ Reduce layer_wise_internal_batch_size from 8 to 4
```

## Verification

All batch sizes come from config - no hardcoded values remain:
```bash
grep -r "chunk_size.*=[0-9]" established_analysis.py | grep -v "self\."
# Should return empty (no results)
```

## Questions?

- See `BATCH_SIZE_CENTRALIZATION_SUMMARY.md` for detailed explanation
- See `BATCH_SIZE_CENTRALIZATION_COMPLETE.md` for implementation details
- All defaults optimized for H100 80GB with 1.5B parameter models