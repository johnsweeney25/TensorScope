# Fisher Function Registration Fixes

## Problem
The Fisher Information functions in BombshellMetrics were failing with errors like:
- `compare_task_fisher: takes 3 positional arguments but 4 were given`
- `fisher_overlap: missing 1 required positional argument: 'masks2'`
- `scale_by_fisher: 'NoneType' object is not iterable`
- `fisher_uncertainty: missing 1 required positional argument: 'task'`

## Root Cause
Fisher functions in BombshellMetrics are instance methods with specific signatures that differ from standard metric patterns:
- They take task names (strings) not model/batch inputs
- They require pre-computed masks or gradients
- They use Fisher EMA state stored in the instance

## Solution Applied

### 1. Updated Function Registrations
Changed all Fisher functions from incorrect signature types to `SignatureType.CUSTOM` with appropriate arguments:

```python
# BEFORE (incorrect):
self.register('compare_task_fisher', bomb.compare_task_fisher, 'bombshell',
             signature_type=SignatureType.DUAL_BATCH, min_batches=2)

# AFTER (correct):
self.register('compare_task_fisher', bomb.compare_task_fisher, 'bombshell',
             signature_type=SignatureType.CUSTOM,
             custom_args={'task1': 'math', 'task2': 'general'})
```

### 2. Enhanced Custom Handler
Added specific handlers in `_call_metric_function` for each Fisher function pattern:

- **compare_task_fisher**: Pass task names directly
- **fisher_overlap**: Generate or pass masks
- **scale_by_fisher**: Compute gradients first, then pass with task
- **fisher_uncertainty**: Pass model, sample, and task
- **fisher_weighted_merge**: Pass model list and task list

### 3. Fixed Functions (10 total)
✅ fisher_importance - Parameter importance via Fisher diagonal
✅ fisher_pruning_masks - Pruning masks based on importance  
✅ compare_task_fisher - Compare Fisher between tasks
✅ fisher_overlap - Overlap between Fisher masks
✅ fisher_weighted_merge - Merge models using Fisher weighting
✅ scale_by_fisher - Scale gradients by Fisher info
✅ fisher_uncertainty - Uncertainty estimation
✅ update_fisher_ema - EMA updates (correctly STANDARD)
✅ reset_fisher_ema - Reset EMA state
✅ top_fisher_directions - Important directions (correctly STANDARD)

## Verification
All 10 Fisher functions are now registered and callable through the unified framework with proper signature handling.

## Impact
Fisher Information metrics provide critical insights for:
- Task interference measurement
- Parameter importance for pruning
- Model uncertainty quantification
- Natural gradient geometry
- Smart model merging

These fixes enable proper multi-task analysis and importance-aware operations.
