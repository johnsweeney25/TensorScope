# QK-OV Interference Metric – Engineering Notes

**Purpose**: Implementation guide for Section 4.1 of paper (blockwise, head-resolved interference)

**Status**: ✅ Implemented in `fisher/core/qkov_interference.py`

---

## Formula Recap

For attention layer ℓ, head h, parameter block B ∈ {Q, K, V, O}:

```
M^B_{ij,ℓ,h} = ⟨C_i|_{B,ℓ,h} / (Î_n|_{B,ℓ,h} + ε), |g_j||_{B,ℓ,h}⟩
```

where:
- `C_i`: Per-sample contribution (gradient squared) from task A, sample i
- `g_j`: Per-sample gradient from task B, sample j
- `Î_n`: EMA Fisher (bias-corrected empirical Fisher)
- `ε`: Numerical stability constant (ridge regularization)

---

## Tensor Shape Reference

### Standard Multi-Head Attention

**Fused QKV** (e.g., GPT-2):
```
W_qkv: [3 * num_heads * head_dim, hidden_dim]
├─ Q: rows [0, H*d_k)
├─ K: rows [H*d_k, 2*H*d_k)
└─ V: rows [2*H*d_k, 3*H*d_k)
```

**Split Projections** (e.g., LLaMA):
```
W_q: [num_heads * head_dim, hidden_dim]
W_k: [num_heads * head_dim, hidden_dim]
W_v: [num_heads * head_dim, hidden_dim]
W_o: [hidden_dim, num_heads * head_dim]  ⚠️ COLUMN-sliced!
```

### Grouped-Query Attention (GQA)

**Key/Value sharing**:
```
W_q: [num_heads * head_dim, hidden_dim]          # All query heads
W_k: [num_kv_heads * head_dim, hidden_dim]       # Shared K heads
W_v: [num_kv_heads * head_dim, hidden_dim]       # Shared V heads
W_o: [hidden_dim, num_heads * head_dim]

# Mapping: q_head → kv_head
kv_head = q_head // (num_heads // num_kv_heads)
```

---

## 6 Critical Pitfalls (From Intern's Feedback)

### 1. **Fused vs Split QKV**

❌ **Wrong**: Assume all models have separate `q_proj`, `k_proj`, `v_proj`

✅ **Right**: Detect architecture and handle both:
```python
if has_c_attn:  # GPT-2 style
    # Fused: slice rows by block offset
    block_offset = {'Q': 0, 'K': H*d_k, 'V': 2*H*d_k}[block]
    start = block_offset + head * d_k
    end = start + d_k
    return W_qkv[start:end, :]
else:  # LLaMA style
    # Split: direct head slicing
    start = head * d_k
    end = start + d_k
    return W_q[start:end, :]
```

### 2. **O Projection is Column-Sliced**

❌ **Wrong**: `W_o[head*d_v:(head+1)*d_v, :]` (row slicing like Q/K/V)

✅ **Right**: `W_o[:, head*d_v:(head+1)*d_v]` (column slicing)

**Reason**: W_O maps from concatenated head outputs → hidden_dim
- Input: `[batch, seq, num_heads * head_dim]`
- Output: `[batch, seq, hidden_dim]`
- Weight shape: `[hidden_dim, num_heads * head_dim]`
- Each head's contribution uses columns `[h*d_v, (h+1)*d_v)`

### 3. **GQA/MQA Head Mapping**

❌ **Wrong**: Assume 1:1 Q-to-K/V mapping

✅ **Right**: Use group mapping for K/V:
```python
if uses_gqa and block in ['K', 'V']:
    kv_head = head // (num_heads // num_kv_heads)
    start = kv_head * d_k
    end = start + d_k
```

**Example**: 32 Q heads, 8 KV heads → each KV head serves 4 Q heads

### 4. **Bias Handling**

**Decision point**: Include bias terms or not?

- **Paper convention**: Usually exclude (focus on weight matrices)
- **If including**: Biases are vectors, not matrices
  - Q/K/V bias: `[num_heads * head_dim]` → row slice like weights
  - O bias: `[hidden_dim]` → no head slicing (shared across heads)

**Recommendation**: Start without bias, add as optional flag

### 5. **Numerical Stability**

❌ **Dangerous**: `C_i / I_n` with small I_n → explosion

✅ **Safe**:
```python
# Ridge regularization
I_n_regularized = I_n.clamp_min(epsilon) + ridge_lambda

# Use in normalization
normalized_contrib = C_i / I_n_regularized
```

**Values**:
- `epsilon = 1e-10` (prevent division by zero)
- `ridge_lambda = 1e-8` (dampen small eigenvalues)

### 6. **PyTorch Weight Layout**

`nn.Linear(in_features, out_features)` has weight shape: `[out_features, in_features]`

**Matrix multiply**: `y = x @ W.T` (transpose!)

**Slicing implications**:
- Q/K/V: `out_features = num_heads * head_dim` → **row slicing**
- O: `out_features = hidden_dim`, `in_features = num_heads * head_dim` → **column slicing**

---

## API Usage

### Quick Start

```python
from fisher.core.qkov_interference import QKOVConfig, QKOVInterferenceMetric
from fisher.core.fisher_collector import FisherCollector

# 1. Setup
config = QKOVConfig.from_model(model)
fisher_collector = FisherCollector(
    reduction='param',
    enable_cross_task_analysis=True,
    gradient_memory_mb=100
)

metric = QKOVInterferenceMetric(config, fisher_collector)

# 2. Collect Fisher + contributions for task A
fisher_collector.collect_fisher(model, batch_A, task='math', mode='ema')

# 3. Collect gradients for task B
fisher_collector.collect_fisher(model, batch_B, task='code', mode='ema')

# 4. Compute interference for sample pair
scores = metric.compute_sample_pair(
    task_a='math', sample_i=7,
    task_b='code', sample_j=23,
    layer=3, head=5
)
print(scores)  # {'Q': 0.42, 'K': 0.31, 'V': 0.18, 'O': 0.55}

# 5. Full heatmap across layers/heads
heatmap = metric.compute_heatmap(
    task_a='math',
    task_b='code',
    layers=[3, 4, 5],
    heads=range(12)
)
```

### Statistical Testing

```python
from fisher.core.qkov_statistics import QKOVStatistics

stats = QKOVStatistics(fdr_alpha=0.05, n_permutations=1000)

# Run full testing pipeline
results = stats.test_heatmap(
    heatmap_results=heatmap,
    contributions_by_task={'math': contribs_A},
    gradients_by_task={'code': grads_B},
    fisher=fisher_collector.fisher_ema
)

# Extract significant conflicts
for block in ['Q', 'K', 'V', 'O']:
    sig_conflicts = results['fdr_corrected'][block]
    print(f"{block}: {len(sig_conflicts)} significant layer/head pairs (FDR < 0.05)")
```

---

## Sanity Checks (Section 5 from Intern)

### Check 1: Head Additivity

**Test**: ∑_h M^Q_{ij,ℓ,h} ≈ M^Q_{ij,ℓ} (unsliced block metric)

```python
# Sum over all heads for layer ℓ
head_sum = sum(
    metric.compute_sample_pair('math', 7, 'code', 23, layer=3, head=h)['Q']
    for h in range(config.num_heads)
)

# Compute unsliced block metric (all heads together)
# (Implementation would skip head slicing)
unsliced_score = metric.compute_block_score_no_slicing(...)

assert np.isclose(head_sum, unsliced_score, rtol=1e-3)
```

### Check 2: Scale Invariance

**Test**: Scaling weights by c shouldn't change conflict rankings

```python
# Original scores
original = metric.compute_heatmap('math', 'code')

# Scale all weights by 2.0
for param in model.parameters():
    param.data *= 2.0

# Recompute Fisher and scores
scaled = metric.compute_heatmap('math', 'code')

# Rankings should be identical (Spearman ρ ≈ 1.0)
from scipy.stats import spearmanr
rho, _ = spearmanr(
    original['Q']['layer_head_avg'].flatten(),
    scaled['Q']['layer_head_avg'].flatten()
)
assert rho > 0.99
```

### Check 3: Ablation Validity

**Test**: Zeroing a head collapses its M^B_{ℓ,h}

```python
# Zero out head 5 in layer 3
with torch.no_grad():
    # Find Q projection
    for name, param in model.named_parameters():
        if 'layers.3' in name and 'q_proj' in name:
            d_k = config.head_dim
            param[5*d_k:(5+1)*d_k, :] = 0.0  # Zero head 5

# Recompute
ablated = metric.compute_sample_pair('math', 7, 'code', 23, layer=3, head=5)

# Should be near-zero
assert abs(ablated['Q']) < 1e-6
```

### Check 4: Symmetry

**Test**: Swapping (i,j) and (A,B) flips directionality but preserves head IDs

```python
# Original
score_ij = metric.compute_sample_pair('math', 7, 'code', 23, layer=3, head=5)

# Swapped
score_ji = metric.compute_sample_pair('code', 23, 'math', 7, layer=3, head=5)

# Magnitudes may differ, but head identities (which heads are top) should match
# (Detailed implementation would compare top-k heads)
```

---

## Integration with Unified Analysis

Add to `unified_model_analysis.py`:

```python
# Enable QKOV analysis
if enable_qkov_interference:
    from fisher.core.qkov_interference import QKOVConfig, QKOVInterferenceMetric

    qkov_config = QKOVConfig.from_model(model)
    qkov_metric = QKOVInterferenceMetric(qkov_config, fisher_collector)

    # Compute cross-task heatmaps
    if 'task_A' in results and 'task_B' in results:
        qkov_heatmap = qkov_metric.compute_heatmap(
            task_a='task_A',
            task_b='task_B',
            layers=list(range(qkov_config.num_layers)),
            heads=list(range(qkov_config.num_heads))
        )

        results['qkov_interference'] = qkov_heatmap
```

---

## Outputs for Paper Figures

### Figure 1: Per-Head Heatmap (Section 4.1)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Extract layer/head averages for block Q
Q_scores = heatmap['Q']['layer_head_avg']  # [n_layers, n_heads]

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(Q_scores, cmap='RdYlBu_r', annot=True, fmt='.2f',
            xticklabels=range(config.num_heads),
            yticklabels=range(config.num_layers))
plt.xlabel('Head')
plt.ylabel('Layer')
plt.title('M^Q_{ij,ℓ,h}: Query Projection Interference (Math vs Code)')
plt.savefig('figures/qkov_Q_heatmap.pdf')
```

### Figure 2: Block Comparison

```python
# Average across layers/heads for each block
block_means = {
    block: heatmap[block]['layer_head_avg'].mean()
    for block in ['Q', 'K', 'V', 'O']
}

# Bar plot
plt.figure(figsize=(8, 6))
plt.bar(block_means.keys(), block_means.values())
plt.ylabel('Mean Interference Score')
plt.title('Cross-Task Interference by Attention Block')
plt.savefig('figures/qkov_block_comparison.pdf')
```

### Table 1: Top Conflicting Heads

```python
# Extract top conflicts for Q block
top_Q = heatmap['Q']['top_conflicts'][:10]

# Print table
print("| Layer | Head | Score | p-value |")
print("|-------|------|-------|---------|")
for conf in top_Q:
    print(f"| {conf['layer']} | {conf['head']} | {conf['score']:.3f} | {conf.get('p_value', 'N/A')} |")
```

---

## Troubleshooting

### Issue: "Could not find parameter for L3 Q"

**Cause**: Parameter naming pattern doesn't match model architecture

**Fix**: Check actual parameter names:
```python
for name, param in model.named_parameters():
    if 'layers.3' in name and 'attn' in name:
        print(f"{name}: {param.shape}")
```

Add custom pattern to `QKOVIndexer._build_param_patterns()`:
```python
'Q': [
    # ... existing patterns
    r'your_custom_pattern\.(\d+)\.custom_q\.weight',
]
```

### Issue: "Shape mismatch when slicing"

**Cause**: Head dimension calculation incorrect

**Fix**: Verify head_dim:
```python
W_q = model.layers[0].self_attn.q_proj.weight
expected_rows = config.num_heads * config.head_dim
assert W_q.shape[0] == expected_rows, f"Expected {expected_rows}, got {W_q.shape[0]}"
```

### Issue: "All scores are zero"

**Cause**: Missing contribution cache or gradient storage

**Fix**: Enable storage in FisherCollector:
```python
fisher_collector = FisherCollector(
    enable_cross_task_analysis=True,  # ← Must be True
    gradient_memory_mb=100             # ← Sufficient budget
)
```

Check cache:
```python
print(f"Contributions cached: {len(fisher_collector.contribution_cache)}")
print(f"Gradients stored: {fisher_collector.gradient_manager.get_memory_usage()}")
```

---

## Performance Notes

**Memory scaling**:
- Contributions: `O(n_samples * n_params)` with compression
- Heatmap: `O(n_A * n_B * n_layers * n_heads * 4 blocks)` = ~400MB for 100×100 samples, 32 layers, 32 heads

**Optimizations**:
- Use `max_samples_per_task` to limit heatmap size
- Cache scores (automatic via `_score_cache`)
- Batch process sample pairs in parallel (TODO)

**Typical runtime** (single GPU):
- Per sample-pair: ~5ms
- 100×100 heatmap: ~50 seconds
- Statistical testing: +20 seconds (1000 permutations)

---

## References

- Olsson et al. (2022): Induction Heads
- Elhage et al. (2021): Mathematical Framework for Transformer Circuits
- Wang et al. (2022): Interpretability in the Wild
- Benjamini & Hochberg (1995): FDR Control
- Cohen (1988): Statistical Power Analysis

---

**Last Updated**: 2025-10-02
**Implementation**: `fisher/core/qkov_interference.py`
**Tests**: `fisher/tests/test_qkov_interference.py` (TODO)