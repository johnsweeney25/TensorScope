# top_fisher_directions fisher_type Fix

## Status: ✅ ALREADY FIXED

## Original Error
```
Failed to compute top_fisher_directions: fisher_type must be 'direct' or 'ema', got diagonal
```

## What Was Wrong
The function was being called with `fisher_type='diagonal'` which is not a valid option.

## Valid Options
- **`'ema'`** (default) - Uses accumulated Fisher information from EMA (Exponential Moving Average)
  - Does not require model or data
  - Uses previously computed Fisher information
  - Good for protecting accumulated knowledge

- **`'direct'`** - Computes Fisher information directly
  - Requires model and task_data parameters
  - Computes fresh Fisher information for specific task
  - Good for protecting specific capabilities

## Fix Applied
Changed the registration in `unified_model_analysis.py`:
- **Before**: `custom_args={'task': 'general', 'fisher_type': 'diagonal'}`
- **After**: `custom_args={'task': 'general', 'fisher_type': 'ema'}`

## Why 'ema' is the Right Choice
1. **No additional requirements** - Works without needing model/data
2. **Uses accumulated knowledge** - Leverages previously computed Fisher information
3. **More efficient** - Doesn't need to recompute Fisher each time
4. **Default behavior** - Matches the function's default parameter

## Testing Confirmed
✅ `fisher_type='ema'` works correctly
✅ `fisher_type='direct'` works (but requires model/data)
✅ `fisher_type='diagonal'` correctly raises ValueError
✅ Registry has correct registration with 'ema'

## Note
The error message you're seeing is from an old run before the fix was applied. The current code is correct.