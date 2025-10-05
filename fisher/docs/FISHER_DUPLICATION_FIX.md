# Fisher Importance Duplication Fix

## Problem Identified
The `compute_fisher_importance` metric was finishing in just 0.02 seconds, which seemed suspiciously fast. Investigation revealed that the metric was being computed **twice**:

1. **First computation** (in Fisher analysis suite): Takes normal time, computes from scratch
2. **Second computation** (as individual metric): Takes 0.02s, just reads cached data

## Root Cause
- Fisher importance was being computed in `_compute_fisher_analysis_suite()` as part of comprehensive Fisher analysis
- The same metric was then being computed again during individual metric computation
- The second computation was fast because it just read the already-computed Fisher EMA values from memory
- This created confusing logs showing the metric running twice with vastly different timings

## Solution Implemented
Added logic to detect and skip Fisher metrics that were already computed in the suite:

### 1. Detection Logic (unified_model_analysis.py)
- Added check for Fisher metrics already computed in the suite
- Skip individual computation if results already exist
- Use cached results from the suite instead

### 2. Updated Logging
- Added clear logging when computing Fisher metrics in the suite:
  - `📊 Computing Fisher importance (as part of suite)...`
  - Shows timing for each task computation
- Added skip message when metric is already computed:
  - `↔️ Skipping compute_fisher_importance - already computed in Fisher analysis suite`

### 3. Affected Metrics
The following Fisher metrics are now deduplicated:
- `compute_fisher_importance`
- `get_fisher_pruning_masks`
- `compare_task_fisher`
- `compute_fisher_overlap`

## Code Changes
**File: unified_model_analysis.py**

### Lines 3827-3868 (for 'all' metrics mode)
- Added check for `fisher_analysis_results`
- Skip Fisher metrics if already computed in suite
- Use cached results with 0.0 compute time

### Lines 3890-3931 (for specific metrics mode)
- Same logic as above for when specific metrics are listed

### Lines 5120-5158 (Fisher suite computation)
- Added clearer logging with timing information
- Shows when metrics are computed "as part of suite"

## Benefits
1. **No duplicate computation**: Fisher metrics computed only once
2. **Clearer logs**: Easy to see what's happening and when
3. **Better performance**: Avoids redundant computation
4. **Accurate timing**: Shows where time is actually spent

## Testing
Created test files:
- `test_fisher_duplication_fix.py`: Comprehensive test with log verification
- `test_fisher_simple.py`: Simple test to verify functionality

## Verification
To verify the fix is working:
1. Run any analysis with Fisher metrics enabled
2. Look for log message: `↔️ Skipping compute_fisher_importance - already computed in Fisher analysis suite`
3. Check that Fisher importance compute time is 0.0s in the final results
4. Verify no duplicate "Starting metric 'compute_fisher_importance'" messages

## Impact
This fix ensures:
- Fisher metrics are computed efficiently (once per analysis)
- Log output is clear and meaningful
- Timing information accurately reflects computation cost
- No confusion about why metrics finish quickly