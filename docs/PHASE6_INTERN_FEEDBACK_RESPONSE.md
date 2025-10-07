# Phase 6: Intern Feedback Response & Action Plan

## Summary of Feedback

Your intern provided excellent theoretical and practical feedback on the QK-OV interference metric. Key points:

1. **Theoretical**: Current metric is **diagonal-Fisher, asymmetric, magnitude-weighted**
2. **Numerical**: Need better dtype handling, empty sample guards
3. **API**: Fix sample ID reporting, add shape assertions
4. **Documentation**: Make design choices explicit

---

## Action Plan

### ✅ Priority 1: Critical Fixes (Implement Now)

#### 1.1 Fix `_compute_fisher_from_contributions` dtype handling

**Issue**: Averaging in fp16/bf16 can lose precision

**Fix**:
```python
def _compute_fisher_from_contributions(self, task: str, param_name: str) -> torch.Tensor:
    # ... existing code ...
    
    # OLD: May accumulate in fp16
    # contributions_stacked = torch.stack(contributions)
    
    # NEW: Force fp32 on CPU for stable averaging
    contributions_fp32 = [c.detach().to(dtype=torch.float32, device='cpu') 
                          for c in contributions]
    fisher_full = torch.stack(contributions_fp32, dim=0).mean(dim=0)
```

#### 1.2 Guard empty task_b samples

**Issue**: `n_samples_b=0` causes NaN in heatmap

**Fix**:
```python
def compute_heatmap(self, ...):
    # After getting task_b_samples
    if not task_b_samples or len(task_b_samples) == 0:
        logger.warning(f"No samples for task_b='{task_b}'; returning empty heatmap")
        return {
            block: {
                'scores': np.zeros((len(task_a_samples), 0, len(layers), len(heads))),
                'layer_head_avg': np.zeros((len(layers), len(heads))),
                'top_conflicts': []
            }
            for block in ['Q', 'K', 'V', 'O']
        }
```

#### 1.3 Fix sample IDs in top_conflicts

**Issue**: Reports matrix indices instead of true sample IDs

**Fix**:
```python
# In compute_heatmap, when building top_conflicts:
i_idx, j_idx, l_idx, h_idx = np.unravel_index(idx, scores.shape)

# OLD: Uses array indices
# 'sample_i': i_idx, 'sample_j': j_idx

# NEW: Extract true sample IDs
true_sample_i = int(task_a_samples[i_idx].split('_')[-1])
true_sample_j = int(task_b_samples[j_idx]) if isinstance(task_b_samples[j_idx], int) else int(task_b_samples[j_idx].split('_')[-1])

top_conflicts.append({
    'sample_i': true_sample_i,
    'sample_j': true_sample_j,
    'layer': layers[l_idx],
    'head': heads[h_idx],
    'score': float(scores[i_idx, j_idx, l_idx, h_idx])
})
```

#### 1.4 Add shape assertions

**Fix**:
```python
def compute_block_head_score(self, ...):
    # After slicing
    C_i_bh = self.indexer.slice_tensor(C_full, layer, head, block, param_name)
    g_j_bh = self.indexer.slice_tensor(G_full, layer, head, block, param_name)
    I_n_bh = self.indexer.slice_tensor(F_full, layer, head, block, param_name)
    
    # NEW: Shape validation
    assert C_i_bh.shape == g_j_bh.shape == I_n_bh.shape, (
        f"Shape mismatch for {param_name} L{layer}H{head} {block}: "
        f"C_i={C_i_bh.shape}, g_j={g_j_bh.shape}, I_n={I_n_bh.shape}"
    )
```

---

### 📝 Priority 2: Documentation Improvements (Update Docstrings)

#### 2.1 Update module docstring

**Add explicit design statement**:
```python
"""
QK-OV Block-Wise Interference Metric (Section 4.1)

METRIC DESIGN:
--------------
This implementation uses a **diagonal-Fisher, asymmetric, magnitude-weighted** interference score:

    M^B_{ij,ℓ,h} = Σ_k (g²_{i,k} / Î_k) · |g_{j,k}|

Where:
- Diagonal Fisher: Uses elementwise squares (g²_i), not full covariance g_i g_i^T
- Asymmetric: M_ij ≠ M_ji (uses C_i for normalization, |g_j| for magnitude)
- Magnitude-weighted: Takes |g_j| (unsigned), so doesn't distinguish synergy vs conflict
- Task-A normalized: Î computed from task A contributions only

DESIGN RATIONALE:
-----------------
1. **Diagonal**: Computationally feasible for large models (full Fisher is O(p²))
2. **Directional**: Measures "how sample i from task A stresses parameters when sample j from task B updates them"
3. **Magnitude-only**: Focuses on interference magnitude (|conflict|), not direction (±conflict)
4. **Per-task normalization**: Î_A normalizes by task A's parameter sensitivity

ALTERNATIVE METRICS (for ablation):
------------------------------------
- Signed diagonal: M_ij = Σ (g_{i,k} · g_{j,k}) / Î_k  (captures synergy vs conflict)
- Symmetric: M_ij = M_ji (e.g., use C_i and C_j symmetrically)
- Full Fisher: M_ij = g_i^T F^-1 g_j (requires KFAC or low-rank approximation)
"""
```

#### 2.2 Update `compute_block_head_score` docstring

```python
def compute_block_head_score(self, ...):
    """
    Compute interference score M^B_{ij,ℓ,h} for a single block/head.
    
    METRIC FORMULA:
    ---------------
    M^B_{ij,ℓ,h} = Σ_{k ∈ slice} (C_{i,k} / (Î_{n,k} + ε)) · |g_{j,k}|
                 = Σ_k (g²_{i,k} / (Î_k + ε)) · |g_{j,k}|
    
    Where:
    - C_{i,k} = g²_{i,k}: Per-sample contribution (squared gradient)
    - Î_{n,k} ≈ (1/n) Σ_j C_{j,k}: Diagonal Fisher estimate
    - |g_{j,k}|: Magnitude of task B gradient (unsigned)
    
    PROPERTIES:
    -----------
    - Asymmetric: M_ij ≠ M_ji (directional interference)
    - Unsigned: |g_j| discards synergy (+) vs conflict (-)
    - Diagonal: Uses elementwise Fisher, not full covariance
    - Ridge-regularized: (Î_k + λ) for numerical stability
    
    INTERPRETATION:
    ---------------
    High M_ij means:
    - Sample i has high sensitivity (C_i) in this circuit
    - Sample j has high gradient magnitude (|g_j|) in same circuit
    - After Fisher normalization, these coincide significantly
    → Sample i "interferes" with task B updates driven by sample j
    
    Args:
        contrib: C_i (full parameter tensor, squared gradients)
        grad: g_j (full parameter tensor, task B gradients)
        fisher: Î_n (full parameter tensor, Fisher from contributions)
        ...
    
    Returns:
        (score, diagnostics): M^B_{ij,ℓ,h} and numerical health info
    """
```

---

### 🔬 Priority 3: Add Alternative Metrics (For Ablation Studies)

#### 3.1 Add signed variant

```python
def compute_block_head_score_signed(
    self,
    contrib: torch.Tensor,
    grad: torch.Tensor,
    fisher: torch.Tensor,
    layer: int,
    head: int,
    block: str,
    param_name: str
) -> Tuple[float, Dict[str, Any]]:
    """
    Signed variant: M_ij = Σ (g_{i,k} · g_{j,k}) / Î_k
    
    Captures synergy (positive) vs conflict (negative).
    Requires storing signed gradients, not just squared contributions.
    """
    # Similar to compute_block_head_score but:
    # 1. Use signed g_i (need to store this separately)
    # 2. Compute: (g_i / sqrt(I_n)) · (g_j / sqrt(I_n))
    # 3. Return signed score
    pass
```

#### 3.2 Add symmetric variant

```python
def compute_block_head_score_symmetric(
    self,
    contrib_i: torch.Tensor,
    contrib_j: torch.Tensor,
    fisher: torch.Tensor,
    layer: int,
    head: int,
    block: str,
    param_name: str
) -> float:
    """
    Symmetric variant: M_ij = M_ji = sqrt(C_i) · sqrt(C_j) / I_n
    
    Treats both samples symmetrically.
    """
    # Compute: Σ (sqrt(C_i_k) · sqrt(C_j_k)) / I_k
    pass
```

---

### 🎯 Priority 4: Numerical Improvements

#### 4.1 Better ridge regularization semantics

**Current** (adds λ to all elements):
```python
I_n_regularized = I_n_bh.clamp_min(self.epsilon) + self.ridge_lambda
```

**Clearer semantics** (only rescues tiny modes):
```python
# Option 1: Floor only
I_n_regularized = torch.maximum(I_n_bh, torch.tensor(self.epsilon, device=I_n_bh.device))

# Option 2: Conditional ridge (add λ only when needed)
needs_ridge = I_n_bh < (10 * self.epsilon)
I_n_regularized = I_n_bh + self.ridge_lambda * needs_ridge.float()
```

**Decision**: Keep current for now (it's safe), but add comment clarifying it's elementwise:

```python
# Add ridge regularization (elementwise, not matrix operation)
# This rescues tiny Fisher values while preserving large ones
I_n_regularized = I_n_bh.clamp_min(self.epsilon) + self.ridge_lambda
```

#### 4.2 Add optional device policy

```python
def __init__(
    self,
    config: QKOVConfig,
    fisher_collector,
    epsilon: float = 1e-10,
    ridge_lambda: float = 1e-8,
    min_samples_for_fisher: int = 10,
    computation_device: str = 'cpu'  # NEW: 'cpu', 'cuda', or 'auto'
):
    """
    ...
    Args:
        ...
        computation_device: Device for score computation
            - 'cpu': Force CPU (reproducible, slower)
            - 'cuda': Force GPU (faster, may differ across runs)
            - 'auto': Use GPU if available, else CPU
    """
    self.computation_device = computation_device
    if computation_device == 'auto':
        self.computation_device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

---

### 📊 Priority 5: Add Validation Tests

#### 5.1 Test: Head additivity

```python
def test_head_additivity():
    """
    Sum of head-sliced scores should equal unsliced block score.
    
    Mathematically: Σ_h M_{ij}^{B,ℓ,h} ≈ M_{ij}^{B,ℓ,*} (all heads)
    """
    # For each block B
    for block in ['Q', 'K', 'V', 'O']:
        # Compute per-head scores
        head_scores = []
        for head in range(num_heads):
            score = metric.compute_sample_pair(task_a, i, task_b, j, layer, head)[block]
            head_scores.append(score)
        
        # Sum across heads
        sum_heads = sum(head_scores)
        
        # Compute unsliced block score (would need separate method)
        # block_score = metric.compute_block_score(task_a, i, task_b, j, layer, block)
        
        # Check: sum_heads ≈ block_score (within 1e-5 relative)
        # assert abs(sum_heads - block_score) / block_score < 1e-5
```

#### 5.2 Test: Scale invariance

```python
def test_fisher_normalization():
    """
    Verify Fisher normalization: C_i / I_n should be scale-invariant in i.
    
    If we scale parameter k by c, then:
    - C_{i,k} scales by c²
    - I_{n,k} scales by c²
    - C_i / I_n stays constant (up to numerical floor)
    """
    # Get original score
    orig_score = metric.compute_sample_pair(task_a, i, task_b, j, layer, head)['Q']
    
    # Scale Q projection weights by 2.0
    with torch.no_grad():
        model.layers[layer].self_attn.q_proj.weight *= 2.0
        
        # Recompute contributions and Fisher
        fisher_collector.contribution_cache.clear()
        fisher_collector._fisher_cache.clear()
        fisher_collector.collect_fisher(model, batch, task=task_a)
        
        # Compute new score
        new_score = metric.compute_sample_pair(task_a, i, task_b, j, layer, head)['Q']
        
        # Restore weights
        model.layers[layer].self_attn.q_proj.weight /= 2.0
    
    # Check: new_score should be ~constant (C/I normalization)
    # But |g_j| will scale, so current metric WILL change
    # This is by design (magnitude-weighted)
```

---

## Implementation Priority

### Must-Do (Before Next Run)
1. ✅ Fix dtype handling in `_compute_fisher_from_contributions`
2. ✅ Guard empty task_b samples in `compute_heatmap`
3. ✅ Fix sample IDs in `top_conflicts`
4. ✅ Add shape assertions in `compute_block_head_score`

### Should-Do (For Paper)
5. 📝 Update all docstrings with design choices
6. 📝 Add "METRIC DESIGN" section to module docstring
7. 📝 Clarify diagonal Fisher and asymmetry in paper

### Nice-to-Have (For Ablations)
8. 🔬 Implement signed variant (`compute_block_head_score_signed`)
9. 🔬 Implement symmetric variant
10. 🎯 Add optional computation_device parameter
11. 📊 Add validation tests

---

## Response to Specific Points

### 1. "Diagonal Fisher vs Full Fisher"

**Intern's concern**: "Cross-parameter covariance within a head/block is ignored"

**Our response**: 
- ✅ **By design**: Full Fisher is O(p²) memory and computation
- ✅ **Practical**: Diagonal Fisher is standard (EWC, KFAC diagonal variant)
- ✅ **Ablation**: Can implement KFAC-based variant for comparison

**Action**: Explicitly state in docstring and paper

### 2. "Asymmetry: M_ij ≠ M_ji"

**Intern's concern**: "Using C_i but |g_j| makes it directional"

**Our response**:
- ✅ **Intentional**: Measures "how sample i stresses parameters when j updates them"
- ✅ **Interpretation**: Directional interference (i → j) not symmetric conflict
- ✅ **Alternative**: Can add symmetric variant for comparison

**Action**: Make directionality explicit in paper

### 3. "Sign removal: |g_j|"

**Intern's concern**: "Discards synergy vs conflict"

**Our response**:
- ✅ **Design choice**: We want interference magnitude, not direction
- ✅ **Use case**: For pruning/merging, magnitude matters (both conflict and synergy are problematic)
- ✅ **Alternative**: Signed variant for detecting constructive vs destructive interference

**Action**: Implement signed variant as ablation

### 4. "Task-A-only normalization"

**Intern's concern**: "Biases toward task A's distribution"

**Our response**:
- ⚠️ **Valid point**: Asymmetric by design (i from A, j from B)
- 🤔 **Alternative**: Could pool Fisher from both tasks
- 📊 **Empirical**: Need to test if this matters in practice

**Action**: Add ablation with pooled Fisher: `Î = (Î_A + Î_B) / 2`

---

## Paper Language Suggestions

### For Methods Section

**Current** (vague):
> "We compute Fisher-normalized interference scores between samples..."

**Better** (explicit):
> "We compute a **diagonal-Fisher-normalized, directional interference score** M^B_{ij,ℓ,h} that measures how sample i from task A stresses parameters in circuit block B when sample j from task B updates them. Specifically:
>
> M^B_{ij,ℓ,h} = Σ_{k ∈ B,ℓ,h} (g²_{i,k} / Î_k) · |g_{j,k}|
>
> where g_i are per-sample gradients, Î is the diagonal Fisher estimate from task A contributions, and |·| denotes element-wise absolute value. This formulation:
> 1. Uses **diagonal Fisher** (computationally tractable for large models)
> 2. Is **directional** (M_ij ≠ M_ji, captures asymmetric interference)
> 3. Uses **magnitude-only** gradients (focuses on interference strength, not synergy vs conflict)
>
> We provide ablations with signed (captures synergy/conflict) and symmetric (M_ij = M_ji) variants in Appendix."

### For Ablation Section

Add table:

| Variant | Formula | Measures | Use Case |
|---------|---------|----------|----------|
| **Ours (main)** | Σ (g²_i / Î) · \|g_j\| | Directional magnitude | Pruning, merging |
| Signed | Σ (g_i · g_j) / Î | Synergy (+) vs conflict (-) | Optimization |
| Symmetric | Σ sqrt(C_i · C_j) / Î | Mutual interference | Task similarity |

---

## Verdict

**Intern's assessment is spot-on.** The current implementation is:
- ✅ Theoretically coherent (diagonal Fisher, directional, magnitude-weighted)
- ✅ Practically implementable
- ⚠️ Needs explicit documentation of design choices
- 🔧 Has fixable numerical/API issues

**Our action plan**:
1. **Fix critical issues** (dtype, empty samples, IDs, assertions) ← Do today
2. **Update documentation** (docstrings, paper) ← Do before submission
3. **Add ablations** (signed, symmetric) ← Nice-to-have for reviews

The metric is **defensible as-is** if we clearly state it's diagonal, directional, and magnitude-weighted. The alternatives are good ablation material.
