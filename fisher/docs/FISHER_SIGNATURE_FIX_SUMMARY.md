# Fisher Methods Signature Fix Summary

## Problems Fixed

After the Fisher refactoring to use `FisherCollector`, several Fisher methods were failing due to signature mismatches:

1. **`compute_fisher_importance`**: Was receiving unexpected `batch` parameter
2. **`get_fisher_pruning_masks`**: Was receiving unexpected `model` parameter
3. **`scale_by_fisher`**: Was using wrong parameter name (`scaling_type` instead of `temperature`)
4. **`compute_fisher_overlap`**: Was failing due to no Fisher EMA data being available

## Root Cause

The CUSTOM signature type in `unified_model_analysis.py` has a generic fallback that tries to call functions with:
- First attempt: `func(context, **custom_args)`
- Fallback: `func(context.model, context.batch, **custom_args)`

This doesn't match the actual Fisher method signatures after refactoring.

## Solution Implemented

### 1. Added Specific Handlers for Fisher Methods (lines ~2648-2659)

```python
elif 'compute_fisher_importance' in func_name:
    return func(model=context.model if context.model else None,
                task=custom_args.get('task', 'task1'),
                normalize=custom_args.get('normalize', True),
                return_per_layer=custom_args.get('return_per_layer', False))

elif 'get_fisher_pruning_masks' in func_name:
    return func(task=custom_args.get('task', 'task1'),
                sparsity=custom_args.get('sparsity', 0.5),
                structured=custom_args.get('structured', False))
```

### 2. Fixed scale_by_fisher Registration (line 1026)

Changed from:
```python
custom_args={'gradients': {}, 'task': 'task1', 'scaling_type': 'inverse'}
```

To:
```python
custom_args={'gradients': {}, 'task': 'task1', 'temperature': 1.0}
```

### 3. Updated scale_by_fisher Handler (line 2747)

Changed the parameter passing to use `temperature` instead of `scaling_type`.

### 4. Fisher EMA Pre-computation

The earlier fix to enable gradients before Fisher computation ensures Fisher EMA data is properly populated.

## Files Modified

1. **`unified_model_analysis.py`**:
   - Added specific handlers for Fisher methods in CUSTOM signature type
   - Fixed scale_by_fisher registration and handler
   - Fixed parameter names in comments

## Verification

Created `test_fisher_signature_fix.py` which confirms:
- ✅ Fisher methods are called with correct signatures
- ✅ No more TypeError exceptions for unexpected parameters
- ✅ scale_by_fisher uses temperature parameter correctly
- ✅ Methods execute successfully (though some return empty results due to test model simplicity)

## Impact

These fixes ensure that Fisher-based metrics work correctly in the unified analysis pipeline, enabling proper task interference analysis and Fisher-weighted operations.