# Fisher Computation: Step-by-Step Mathematical Trace

## Question: Does the code compute E[g²] or (E[g])²?

Let me trace through the actual computation:

### Step 1: Micro-Batch Loop (fisher_collector.py:352-551)

```python
for i in range(num_micro_batches):
    # Line 364: Extract micro-batch
    micro_batch = batch[start_idx:end_idx]
    
    # Line 382: Scale loss by token proportion
    loss = outputs.loss.float() * (micro_active_tokens / total_active_tokens)
    
    # Line 392: Backward pass
    loss.backward()
    
    # Line 547: Accumulate gradients
    gradient_accumulator[name] += param.grad
```

**What this computes**:
- If all micro-batches have equal tokens: `gradient_accumulator = sum(g_i * weight_i)` where weights sum to 1
- This is equivalent to: `gradient_accumulator ≈ mean(g_i)` (weighted average)

### Step 2: Reduce to Groups (fisher_collector.py:618, then _reduce_to_groups:1075)

```python
# Line 616: Get accumulated gradient
grad = gradient_accumulator[name]  # This is mean(g_i)

# Line 618: Call _reduce_to_groups
group_fisher, group_type, num_groups = self._reduce_to_groups(name, grad, ...)

# Inside _reduce_to_groups at line 1075:
grad_fp32 = grad.to(torch.float32)  # Still mean(g_i)
grad_sq = grad_fp32.pow(2)          # (mean(g_i))²
```

**What this computes**:
- `grad_sq = (mean(g_i))²`
- This is **(E[g])²**, NOT **E[g²]**!

### Step 3: Normalize (fisher_collector.py:623)

```python
group_fisher = group_fisher / (total_active_tokens + 1e-8)
```

**Final result**:
- `Fisher = (mean(g_i))² / tokens`

## The Mathematical Problem

**What code computes**:
```
F_computed = (mean(g_i))²
           = (E[g])²
```

**What empirical Fisher should be**:
```
F_empirical = mean(g_i²)
            = E[g²]
```

**The relationship**:
```
E[g²] = (E[g])² + Var[g]
```

**So the code is MISSING the variance term**!

### Implications

1. **At convergence** (when E[g] ≈ 0):
   - Code computes: ≈ 0
   - Should compute: Var[g] (which can be large!)
   - **Result**: Underestimates Fisher dramatically

2. **During training** (when E[g] ≠ 0):
   - Code computes: (E[g])² (gradient magnitude)
   - Should compute: E[g²] = (E[g])² + Var[g]
   - **Result**: Missing variance component

3. **For pruning**:
   - Code ranks by |E[g]|² (how consistently gradient points one way)
   - Should rank by E[g²] (total gradient activity)
   - **Result**: May prune parameters with high variance but low mean

## Wait... But What About Welford?

Let me check if Welford updates fix this...

Looking at fisher_collector.py:666-678:

```python
if welford_key in self.fisher_accumulated[task]:
    # Welford's algorithm
    old_mean = self.fisher_accumulated[task][welford_key]
    delta = group_fisher_f64 - old_mean_f64  # group_fisher is (mean g)²!
    new_mean_f64 = old_mean_f64 + (delta * weight / new_total_weight)
    ...
```

**Welford is averaging over batches**:
- Batch 1 contributes: (mean_batch1(g))²
- Batch 2 contributes: (mean_batch2(g))²
- Final: mean_over_batches((mean_within_batch(g))²)

**This is STILL NOT empirical Fisher**!

True empirical Fisher should be:
```
F = mean_over_all_samples(g²)
```

Not:
```
F = mean_over_batches((mean_within_batch(g))²)
```

## The Correct Computation Would Be

### Option A: Square BEFORE Accumulation

```python
# For each micro-batch:
grad_squared = param.grad.pow(2)  # Square individual gradient
fisher_accumulator[name] += grad_squared  # Accumulate squared gradients

# After all micro-batches:
fisher = fisher_accumulator[name] / total_active_tokens  # Average
```

This gives: `E[g²]` ✅

### Option B: Store Per-Sample Then Average

```python
# For each sample (micro_batch_size=1):
sample_fisher = param.grad.pow(2)
all_sample_fishers.append(sample_fisher)

# After all samples:
fisher = mean(all_sample_fishers)  # Average of squared gradients
```

This also gives: `E[g²]` ✅

## What The Current Code Actually Computes

**Current code**:
```python
# Accumulate gradients
total_grad = sum(g_i for all samples)  # Or weighted mean

# Square the sum
fisher = (total_grad)²
```

This gives: `(E[g])²` ❌

## IS THIS A BUG OR INTENTIONAL?

Let me check the docstrings and comments...

From fisher_collector.py:14-16:
```python
"""
This collector computes the EMPIRICAL Fisher Information Matrix, which uses
ground-truth labels from the training data:
  F = E[∇log p(y_data|x,θ) * ∇log p(y_data|x,θ)^T]
```

**They claim to compute E[g⊗g^T]**, which for diagonal is **E[g²]**.

But the code computes **(E[g])²**.

**CONCLUSION: This appears to be a BUG**.

## Why Hasn't This Broken Things?

Possible reasons the bug hasn't been caught:

1. **Gradient mean vs variance**: In many cases, (E[g])² and Var[g] are similar orders of magnitude
2. **Relative rankings**: For pruning, relative importance matters more than absolute values
3. **Post-convergence**: If tested on converged models, E[g] ≈ 0 so both would be small
4. **Small batches**: With small batches, the difference is less pronounced

## What This Means for Phases 2-7

**All phases use the same (incorrect?) Fisher**!

So whether we use micro_batch_size=1 or 128, we'd still be computing (E[g])² not E[g²].

The micro-batching doesn't fix the fundamental issue.

**HOWEVER**: The per-sample gradient storage in Phase 5 DOES compute g_i² correctly because it squares BEFORE accumulation (line 638).

## ACTION REQUIRED

Need to verify:
1. Is this intended behavior?
2. If not, does fixing it break downstream analyses?
3. Should we compute true empirical Fisher?

