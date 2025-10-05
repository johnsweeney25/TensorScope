# EstablishedAnalysisMethods Memory Management

## Critical: Call `cleanup()` After Each Analysis

When using `EstablishedAnalysisMethods` standalone (outside the unified framework), you **MUST** call `cleanup()` after each analysis to prevent GPU memory accumulation.

## Why This Matters

Without cleanup, GPU memory accumulates across multiple analyses:
- ❌ After 5 batches: ~75 GB used → OOM errors
- ✅ With cleanup: ~15 GB used → stable memory

## Usage Pattern

```python
from established_analysis import EstablishedAnalysisMethods
from transformers import AutoModel, AutoTokenizer

# Initialize once
model = AutoModel.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Math-1.5B")
analyzer = EstablishedAnalysisMethods(model, tokenizer)

# Process multiple batches
for batch in dataloader:
    # Run analysis
    result = analyzer.analyze_token_importance(
        inputs=batch['input_ids'],
        position_of_interest=-1,
        attention_mask=batch['attention_mask']
    )

    # Process results...
    process_results(result)

    # CRITICAL: Clean up after each batch
    analyzer.cleanup()  # Prevents 75GB memory accumulation!
```

## What `cleanup()` Does

1. Clears model gradients with `model.zero_grad(set_to_none=True)`
2. Calls `torch.cuda.empty_cache()` to release GPU memory
3. Safe to call multiple times (idempotent)

## When to Call Cleanup

**Always call after:**
- `analyze_token_importance()`
- `analyze_attention_flow()`
- `compute_position_jacobian()`
- `layer_wise_attribution()`
- `comprehensive_analysis()` (auto-cleaned, but safe to call again)

**Example with comprehensive analysis:**
```python
analyzer = EstablishedAnalysisMethods(model, tokenizer)

for text in texts:
    result = analyzer.comprehensive_analysis(text=text)
    save_results(result)
    # comprehensive_analysis() auto-cleans, but you can still call:
    analyzer.cleanup()  # Extra safety for long loops
```

## Memory Debugging

If you see OOM errors like:
```
CUDA out of memory. Tried to allocate 48.00 MiB. GPU 0 has a total capacity
of 79.19 GiB of which 56.75 MiB is free. Including non-PyTorch memory,
this process has 79.12 GiB memory in use.
```

**Cause:** Missing `cleanup()` calls between analyses.

**Solution:** Add `analyzer.cleanup()` after each analysis in your loop.

## Framework Integration

If using `EstablishedAnalysisMethods` through the unified framework
(`unified_model_analysis.py`), cleanup is automatic. You don't need to call
it manually.