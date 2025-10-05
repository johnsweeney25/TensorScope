# Fisher Update Summary

## Overview
Both `BombshellMetrics` and `ModularityMetrics` now support **two types of Fisher Information estimation**:
1. **Direct Fisher** (NEW DEFAULT) - Immediate, one-shot computation for specific task protection
2. **EMA Fisher** - Exponentially weighted moving average for accumulated knowledge protection

## Key Changes

### 1. BombshellMetrics.py

#### New Methods Added:
- `_estimate_fisher_diagonal()` - Direct Fisher computation
- `_get_top_coordinates_from_fisher()` - Extract important coordinates from Fisher dict

#### Updated Methods:
- `compute_null_space_projection()` - Now supports `fisher_type` parameter
- `get_top_fisher_directions()` - Now supports `fisher_type` parameter

### 2. ModularityMetrics.py

#### New Attributes:
- `self.fisher_ema` - Storage for EMA Fisher
- `self.fisher_ema_decay` - Decay rate (default 0.99)

#### New Methods Added:
- `update_fisher_ema()` - Update EMA Fisher over time
- `_get_ema_fisher_for_task()` - Retrieve EMA Fisher for specific task

#### Updated Methods:
- `compute_fisher_weighted_damage()` - Now supports `fisher_type` parameter

## Usage Examples

### Direct Fisher (Default - Protect Specific Capability)

```python
# BombshellMetrics - Protect math capability specifically
bombshell = BombshellMetrics()
projection = bombshell.compute_null_space_projection(
    gradients=new_gradients,
    model=model,           # Required for direct
    task_data=math_batch,  # Required for direct
    fisher_type='direct'   # Optional (this is default)
)

# ModularityMetrics - Measure immediate damage
modularity = ExtendedModularityMetrics()
damage = modularity.compute_fisher_weighted_damage(
    model=model,
    task_A_batch=math_batch,
    task_B_batch=code_batch,
    fisher_type='direct'   # Optional (this is default)
)
```

### EMA Fisher (Protect Accumulated Knowledge)

```python
# Build up EMA Fisher over training
for batch in training_data:
    bombshell.update_fisher_ema(model, batch, task='general')
    # or
    modularity.update_fisher_ema(model, batch, task='general')

# Use accumulated Fisher for protection
projection = bombshell.compute_null_space_projection(
    gradients=new_gradients,
    fisher_type='ema',     # Must specify EMA
    task='general'         # Task name for lookup
)

# Use accumulated Fisher for damage assessment
damage = modularity.compute_fisher_weighted_damage(
    model=model,
    task_A_batch=None,     # Not needed for EMA
    task_B_batch=code_batch,
    fisher_type='ema',     # Must specify EMA
    task_A_name='general'  # Task name for lookup
)
```

## When to Use Each

### Use Direct Fisher When:
- Protecting a **specific capability** (e.g., "protect math while learning code")
- Analyzing **immediate interference** between two tasks
- Working with a **fine-tuned model** for a specific task
- Need **quick analysis** without training history

### Use EMA Fisher When:
- Protecting **general knowledge** accumulated over training
- Implementing **continual learning** scenarios
- Working with **long training runs** where importance evolves
- Need **smooth, stable** Fisher estimates

## API Details

### `compute_null_space_projection` Parameters:
- `gradients`: Gradients to project
- `coordinate_masks`: Optional pre-computed masks
- `fisher_type`: 'direct' (default) or 'ema'
- `model`: Required if fisher_type='direct'
- `task_data`: Required if fisher_type='direct'
- `task`: Task name for EMA lookup (default 'general')
- `n_fisher_samples`: Samples for direct Fisher (default 16)
- `top_k_per_param`: Max important coordinates (default 100)
- `percentile`: Importance threshold (default 95.0)

### `get_top_fisher_directions` Parameters:
- `task`: Task name for EMA lookup
- `fisher_type`: 'direct' (default) or 'ema'
- `model`: Required if fisher_type='direct'
- `task_data`: Required if fisher_type='direct'
- `top_k_per_param`: Max coordinates per parameter
- `percentile`: Importance threshold
- `n_fisher_samples`: Samples for direct Fisher

### `compute_fisher_weighted_damage` Parameters:
- `model`: Model to analyze
- `task_A_batch`: Data for task A (required for direct)
- `task_B_batch`: Data for task B
- `target_layers`: Layers to analyze
- `normalize`: Normalization method
- `return_by_layer`: Per-layer results
- `n_fisher_samples`: Samples for direct Fisher
- `fisher_type`: 'direct' (default) or 'ema'
- `task_A_name`: Task name for EMA lookup

## Migration Guide

### Old Code (Implicit EMA only):
```python
# BombshellMetrics - Only had EMA
masks = bombshell.get_top_fisher_directions(task='general')

# ModularityMetrics - Only had direct
damage = modularity.compute_fisher_weighted_damage(
    model, math_batch, code_batch
)
```

### New Code (Explicit choice):
```python
# Choose direct (default) for immediate analysis
masks = bombshell.get_top_fisher_directions(
    model=model,
    task_data=math_batch
)

# Or choose EMA for accumulated importance
masks = bombshell.get_top_fisher_directions(
    fisher_type='ema',
    task='general'
)
```

## Benefits

1. **Flexibility**: Choose the right Fisher type for your use case
2. **Better Defaults**: Direct Fisher is now default (more common use case)
3. **Unified API**: Both classes support both methods
4. **Backward Compatible**: Old code still works with proper validation
5. **Clear Semantics**: Explicit about immediate vs accumulated importance

## Performance Considerations

- **Direct Fisher**: Computes fresh each time, O(n_samples * forward_backward)
- **EMA Fisher**: O(1) lookup after building, but requires memory for storage
- **Memory**: EMA stores Fisher for all parameters × tasks
- **Speed**: Direct is slower per call but requires no setup

## Error Handling

The updated code includes proper validation:
- Direct Fisher requires `model` and `task_data`
- EMA Fisher requires prior calls to `update_fisher_ema()`
- Invalid `fisher_type` raises `ValueError`
- Missing EMA data provides helpful error messages

## Testing

Run `test_fisher_updates.py` to verify:
- Direct Fisher computation
- EMA Fisher accumulation
- Parameter validation
- Backward compatibility
- Error handling