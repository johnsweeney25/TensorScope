# Fisher Information Metrics Analysis

## The Problem
11+ Fisher Information functions were completely unused, despite being critical for understanding:
- Task interference in multi-task learning
- Parameter importance for pruning
- Natural gradient geometry
- Model uncertainty
- Optimization landscape

## Fisher Functions Categorization

### 🔴 CRITICAL Metrics (Now Registered)

#### Task Interference Analysis
1. **`fisher_weighted_damage`** (ModularityMetrics)
   - Signature: `(model, task_A_batch, task_B_batch)`
   - Measures: How much task B gradients damage task A's Fisher-important parameters
   - Use case: Quantify catastrophic forgetting risk

2. **`fisher_damage_asymmetry`** (ModularityMetrics)
   - Signature: `(model, math_batch, general_batch)`
   - Measures: Bidirectional damage between tasks
   - Use case: Understand which task is more fragile

#### Parameter Importance
3. **`fisher_importance`** (BombshellMetrics)
   - Signature: `(task='general', normalize=True)`
   - Measures: Parameter importance via Fisher diagonal
   - Use case: Identify critical parameters for preservation

4. **`fisher_pruning_masks`** (BombshellMetrics)
   - Signature: `(threshold=0.1, task='general')`
   - Measures: Creates pruning masks based on Fisher importance
   - Use case: Importance-aware pruning

#### Task Comparison
5. **`compare_task_fisher`** (BombshellMetrics)
   - Signature: `(task1='math', task2='general')`
   - Measures: Compare Fisher matrices between tasks
   - Use case: Understand task similarity at parameter level

6. **`fisher_overlap`** (BombshellMetrics)
   - Signature: `(masks1: Dict, masks2: Dict)`
   - Measures: Overlap between Fisher-based masks
   - Use case: Identify shared important parameters

#### Model Operations
7. **`fisher_weighted_merge`** (BombshellMetrics)
   - Signature: `(model1, model2, fisher_weights)`
   - Measures: Merge models using Fisher weighting
   - Use case: Task arithmetic with importance weighting

8. **`fisher_uncertainty`** (BombshellMetrics)
   - Signature: `(model, batch)`
   - Measures: Uncertainty estimation via Fisher
   - Use case: Confidence estimation

### 🟡 Helper Functions (Not Metrics)

These are internal state management, not exposed as metrics:
- `update_fisher_ema` - Updates exponential moving average
- `reset_fisher_ema` - Resets EMA state
- `cleanup_fisher_ema` - Cleanup function
- `_estimate_fisher_diagonal` - Private helper
- `_get_ema_fisher_for_task` - Private helper
- `_get_top_coordinates_from_fisher` - Private helper

## Why These Matter

### 1. Task Interference Measurement
```python
# Critical for understanding multi-task learning
damage = fisher_weighted_damage(model, math_batch, general_batch)
# If damage is high, math training will hurt general performance!
```

### 2. Smart Pruning
```python
# Prune based on Fisher importance, not just magnitude
masks = get_fisher_pruning_masks(threshold=0.9)
# Preserves critical parameters even if small magnitude
```

### 3. Model Merging
```python
# Merge models intelligently using Fisher weighting
merged = fisher_weighted_merge(model1, model2)
# Preserves important features from both models
```

## Implementation Status

### ✅ Now Registered (13 functions)
- `fisher_information` - Basic Fisher computation
- `fisher_weighted_damage` - Task interference
- `fisher_damage_asymmetry` - Bidirectional damage
- `fisher_importance` - Parameter importance
- `fisher_pruning_masks` - Pruning masks
- `compare_task_fisher` - Task comparison
- `fisher_overlap` - Mask overlap
- `fisher_weighted_merge` - Model merging
- `top_fisher_directions` - Important directions
- `scale_by_fisher` - Scale by importance
- `fisher_uncertainty` - Uncertainty estimation
- `update_fisher_ema` - EMA updates
- `reset_fisher_ema` - EMA reset

### 📊 Signature Types
- **STANDARD**: Basic Fisher computation
- **DUAL_BATCH**: Task interference (needs 2 batches)
- **FISHER_BASED**: Uses pre-computed Fisher
- **PREPROCESSED**: Uses Fisher masks
- **TWO_MODELS**: Model merging operations
- **CUSTOM**: Special handling

## Usage Example

```python
# Create context with math and general batches
context = MetricContext(
    models=[model],
    batches=[math_batch, general_batch]
)

# Measure task interference
damage = registry.compute_with_context('fisher_weighted_damage', context)
asymmetry = registry.compute_with_context('fisher_damage_asymmetry', context)

# If math->general damage >> general->math damage:
# Training on math will catastrophically forget general knowledge!

# Get importance for pruning
importance = registry.compute_with_context('fisher_importance', context)
masks = registry.compute_with_context('fisher_pruning_masks', context)
```

## Why This Was Critical to Fix

Fisher Information provides **unique insights** that no other metrics can:
1. **Natural gradient geometry** - How the loss landscape looks from the model's perspective
2. **Task interference prediction** - Quantify catastrophic forgetting before it happens
3. **Importance-aware operations** - Pruning/merging based on actual importance, not magnitude
4. **Uncertainty quantification** - Principled confidence estimation

Without these metrics, you're missing fundamental understanding of:
- Why models forget
- Which parameters matter
- How tasks interfere
- Where uncertainty lies

## Recommendations

1. **Always compute Fisher damage** when doing multi-task training
2. **Use Fisher importance** for any pruning operations
3. **Check Fisher overlap** when merging models
4. **Monitor Fisher uncertainty** for confidence estimation

These metrics are computationally expensive but provide insights impossible to get otherwise!