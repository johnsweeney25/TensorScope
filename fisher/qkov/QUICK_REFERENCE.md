# QKOV Quick Reference

**One-page reference for Section 4.1 implementation**

## Imports

```python
from fisher.qkov import (
    QKOVConfig,                # Model configuration
    QKOVInterferenceMetric,    # Main metric
    QKOVStatistics,            # Statistical testing
)
```

## Basic Usage (3 Steps)

```python
# 1. Setup Fisher collector with cross-task enabled
from fisher.core.fisher_collector import FisherCollector

fisher_collector = FisherCollector(
    enable_cross_task_analysis=True,  # Required!
    gradient_memory_mb=100
)

# 2. Collect data for both tasks
fisher_collector.collect_fisher(model, math_batch, task='math', mode='ema')
fisher_collector.collect_fisher(model, code_batch, task='code', mode='ema')

# 3. Compute interference
config = QKOVConfig.from_model(model)
metric = QKOVInterferenceMetric(config, fisher_collector)

scores = metric.compute_sample_pair(
    task_a='math', sample_i=0,
    task_b='code', sample_j=0,
    layer=3, head=5
)
# Returns: {'Q': 0.42, 'K': 0.31, 'V': 0.18, 'O': 0.55}
```

## Full Heatmap

```python
heatmap = metric.compute_heatmap(
    task_a='math',
    task_b='code',
    layers=[3, 4, 5],           # Which layers
    heads=range(12),             # Which heads
    max_samples_per_task=100     # Sample limit
)

# Extract results
Q_scores = heatmap['Q']['layer_head_avg']  # [n_layers, n_heads]
top_conflicts = heatmap['Q']['top_conflicts'][:10]
```

## Statistical Testing

```python
stats = QKOVStatistics(fdr_alpha=0.05, n_permutations=1000)

results = stats.test_heatmap(heatmap, contribs, grads, fisher)

# Get significant conflicts (FDR-corrected)
sig_Q = results['fdr_corrected']['Q']
sig_K = results['fdr_corrected']['K']
sig_V = results['fdr_corrected']['V']
sig_O = results['fdr_corrected']['O']

print(f"Q: {len(sig_Q)} significant conflicts")
```

## Visualization

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Heatmap: layers × heads
plt.figure(figsize=(12, 8))
sns.heatmap(heatmap['Q']['layer_head_avg'],
            cmap='RdYlBu_r',
            annot=True,
            xticklabels=range(config.num_heads),
            yticklabels=range(config.num_layers))
plt.xlabel('Head')
plt.ylabel('Layer')
plt.title('M^Q: Query Projection Interference')
plt.savefig('Q_interference.pdf')

# Bar chart: block comparison
block_means = {b: heatmap[b]['layer_head_avg'].mean()
               for b in ['Q', 'K', 'V', 'O']}
plt.figure(figsize=(8, 6))
plt.bar(block_means.keys(), block_means.values())
plt.ylabel('Mean Interference')
plt.title('Interference by Block')
plt.savefig('block_comparison.pdf')
```

## Configuration

### Auto-Detect (Recommended)

```python
config = QKOVConfig.from_model(model)
```

### Manual

```python
config = QKOVConfig(
    num_layers=12,
    num_heads=12,
    head_dim=64,                 # d_k for Q/K
    v_head_dim=64,               # d_v for V (may differ!)
    hidden_dim=768,
    fused_qkv=False,             # True for GPT-2, ViT
    fused_qkv_transposed=False,  # True for GPT-2 Conv1D
    uses_gqa=False,              # True for LLaMA 2, Mistral
    num_kv_heads=None,           # For GQA: number of K/V heads
)
```

## Common Architectures

| Model     | fused_qkv | fused_qkv_transposed | uses_gqa | num_kv_heads | Notes |
|-----------|-----------|---------------------|----------|--------------|-------|
| GPT-2     | True      | True (Conv1D)       | False    | None         | Auto-detected |
| LLaMA 1   | False     | N/A                 | False    | None         | Split projections |
| LLaMA 2   | False     | N/A                 | True     | num_heads//4 | Split + GQA |
| Mistral   | False     | N/A                 | True     | 8            | Split + GQA |
| ViT       | True      | False (Linear)      | False    | None         | Fused, standard Linear |

## Parameters

### QKOVInterferenceMetric

```python
metric = QKOVInterferenceMetric(
    config=config,
    fisher_collector=fisher_collector,
    epsilon=1e-10,          # Numerical stability
    ridge_lambda=1e-8       # Ridge regularization
)
```

### QKOVStatistics

```python
stats = QKOVStatistics(
    fdr_alpha=0.05,         # FDR significance level
    n_permutations=1000,    # Permutation samples
    n_bootstrap=1000,       # Bootstrap samples
    min_effect_size=0.2,    # Cohen's d threshold
)
```

## Diagnostic Info

```python
# Enable debug logging
import logging
logging.getLogger('fisher.qkov').setLevel(logging.DEBUG)

# Check numerical health
score, diagnostics = metric.compute_block_head_score(...)
print(f"Fisher min: {diagnostics['fisher_min']:.2e}")
print(f"Contrib norm: {diagnostics['contrib_norm']:.2e}")
print(f"Grad norm: {diagnostics['grad_norm']:.2e}")
```

## Troubleshooting

### "Could not find parameter for L3 Q"

**Cause**: Architecture not recognized

**Fix**: Add pattern to `QKOVIndexer._build_param_patterns()` or check parameter names:

```python
for name, param in model.named_parameters():
    if 'layers.3' in name and 'attn' in name:
        print(f"{name}: {param.shape}")
```

### "All scores are zero"

**Cause**: Missing contribution cache or gradients

**Fix**: Enable cross-task analysis:

```python
fisher_collector = FisherCollector(
    enable_cross_task_analysis=True,  # ← Must be True!
)

# Verify storage
print(len(fisher_collector.contribution_cache))
print(fisher_collector.gradient_manager.get_memory_usage())
```

### "Shape mismatch when slicing"

**Cause**: Wrong head_dim or v_head_dim

**Fix**: Manually specify config or check weight shapes:

```python
W_q = model.layers[0].self_attn.q_proj.weight
print(f"Q weight: {W_q.shape}")
print(f"Expected rows: {config.num_heads * config.head_dim}")
```

## Formula Reference

```
M^B_{ij,ℓ,h} = ⟨C_i|_{B,ℓ,h} / (Î_n|_{B,ℓ,h} + ε), |g_j||_{B,ℓ,h}⟩
```

- **B**: Block (Q, K, V, or O)
- **i**: Sample index from task A
- **j**: Sample index from task B
- **ℓ**: Layer index
- **h**: Head index
- **C_i**: Per-sample contribution (gradient squared)
- **g_j**: Per-sample gradient
- **Î_n**: EMA Fisher (bias-corrected)
- **ε**: Numerical stability constant

## Files

```
fisher/qkov/
├── qkov_interference.py           # Core implementation
├── qkov_statistics.py             # Statistical testing
├── __init__.py                    # Public API
├── README.md                      # Module docs
├── QUICK_REFERENCE.md            # This file
├── MIGRATION.md                   # Import path changes
└── docs/
    ├── QKOV_ENGINEERING_NOTES.md       # Detailed guide
    └── QKOV_IMPLEMENTATION_SUMMARY.md  # Status

examples/
└── qkov_interference_example.py   # Working example
```

## Performance

- **Sample pair**: ~5ms
- **100×100 heatmap**: ~50s
- **+Statistical testing**: +20s

**Memory**: ~400MB for 100×100 samples, 32 layers, 32 heads

## Next Steps

1. Run example: `python examples/qkov_interference_example.py`
2. Read detailed guide: `fisher/qkov/docs/QKOV_ENGINEERING_NOTES.md`
3. Check implementation status: `fisher/qkov/docs/QKOV_IMPLEMENTATION_SUMMARY.md`

---

**Version**: 1.0.0
**Last Updated**: 2025-10-02
**Status**: Production-ready
