# Memory Leak Tests for Lottery Ticket Implementation

## Overview

These tests verify the critical GPU memory leak fixes applied for ICML 2026 submission. The fixes reduced peak GPU memory usage by ~40% (from 25-27 GB to 16 GB) for Qwen-1.5B models.

## Test File

`test_memory_fixes.py` - Comprehensive unit tests for memory leak fixes

## What Is Tested

### BUG #1: Masks Created on CPU
**Test:** `test_bug1_masks_on_cpu()`
- Verifies `create_magnitude_mask()` returns masks on CPU
- Prevents 1.55 GB GPU leak for Qwen-1.5B
- Checks mask dtype is bool

### BUG #2: Bool vs Float32 Storage
**Test:** `test_bug2_mask_bool_not_float32()`
- Verifies masks stored as bool (not float32)
- Saves 4× memory (1.55 GB vs 6.22 GB)
- Measures actual GPU memory usage

### BUG #3: Restoration Temporary Accumulation
**Test:** `test_bug3_no_restoration_leaks()`
- Verifies weight restoration doesn't leak temporaries
- Checks chunked processing prevents accumulation
- Ensures cleanup after each chunk

### BUG #4: Batch Tensor Cleanup
**Test:** `test_bug4_batch_cleanup()`
- Verifies batch tensors explicitly deleted
- Checks memory returns to baseline
- Tests dict reference cleanup

### Numerical Correctness
**Tests:**
- `test_numerical_correctness_bool_vs_float32()` - Bool = float32 results
- `test_mask_bool_conversion_exact()` - {0,1} conversion exact
- `test_mask_multiplication_exact()` - Multiplication bit-exact
- `test_quality_score_edge_cases()` - Edge cases handled

### Reproducibility
**Test:** `test_mask_creation_reproducibility()`
- Verifies same seed → same masks
- Bit-exact reproduction

### Integration
**Test:** `test_compute_lottery_ticket_quality_memory()`
- End-to-end memory leak test
- Verifies no leaks in full pipeline

## Running Tests

### Run All Memory Tests
```bash
cd lottery_tickets/tests
python3 test_memory_fixes.py
```

### Run via Test Suite
```bash
cd lottery_tickets/tests
python3 run_tests.py --module memory
```

### Run All Lottery Ticket Tests
```bash
cd lottery_tickets/tests
python3 run_tests.py
```

## Requirements

### GPU Tests
- Requires CUDA-enabled GPU
- Tests automatically skipped if no GPU available
- Tested on H100 80GB

### CPU Tests
- Numerical correctness tests run on CPU
- No GPU required

## Expected Results

### With GPU
```
test_bug1_masks_on_cpu ..................... ok
test_bug2_mask_bool_not_float32 ............ ok
test_bug3_no_restoration_leaks ............. ok
test_bug4_batch_cleanup .................... ok
test_numerical_correctness_bool_vs_float32 . ok
test_mask_creation_reproducibility ......... ok
test_compute_lottery_ticket_quality_memory . ok
test_mask_bool_conversion_exact ............ ok
test_mask_multiplication_exact ............. ok
test_quality_score_edge_cases .............. ok

----------------------------------------------------------------------
Ran 10 tests in X.XXs

OK
```

### Without GPU
```
test_mask_bool_conversion_exact ............ ok
test_mask_multiplication_exact ............. ok
test_quality_score_edge_cases .............. ok

----------------------------------------------------------------------
Ran 3 tests in X.XXs

OK (skipped=7)
```

## Test Coverage

- ✅ Mask creation device placement
- ✅ Mask dtype optimization
- ✅ Weight restoration chunking
- ✅ Batch tensor cleanup
- ✅ Numerical precision (bit-exact)
- ✅ Reproducibility (fixed seeds)
- ✅ Integration (full pipeline)
- ✅ Edge cases (div by zero, empty masks)

## Related Documentation

- `LOTTERY_TICKET_COMPLETE_AUDIT.md` - Full analysis of fixes
- `LOTTERY_TICKET_OOM_CRITICAL_FIXES.md` - Detailed fix documentation
- `LOTTERY_TICKET_THEORETICAL_ANALYSIS.md` - Theoretical correctness proof

## ICML 2026 Submission

These tests verify:
1. **Memory efficiency**: Fits on single H100 80GB
2. **Numerical correctness**: Bit-exact results
3. **Reproducibility**: Fixed seeds, deterministic
4. **Theoretical soundness**: Implements algorithm correctly

All tests must pass for ICML submission.