# Detailed Examples

Concrete code examples for cross-metric analysis.

[← Back to README](../README.md)

---

## Examples

### Cross-Metric Discovery: Fisher × Superposition

```python
from unified_model_analysis import UnifiedModelAnalyzer
from ICLRMetrics import ICLRMetrics

analyzer = UnifiedModelAnalyzer()
metrics = ICLRMetrics()

results = analyzer.analyze_models([model])

# Get metrics from single pass
fisher_spectrum = metrics.compute_fisher_spectrum(model, batch)  # Full eigenvalue distribution
superposition = results.get('superposition_regime')

# Correlate Fisher concentration with superposition by layer
import pandas as pd
df = pd.DataFrame({
    'layer': range(len(fisher_spectrum['per_layer_top_eig'])),
    'fisher_top_eig': fisher_spectrum['per_layer_top_eig'],
    'fisher_concentration': fisher_spectrum['per_layer_concentration'],  # Top-k / sum
    'superposition_score': [superposition[f'layer_{i}'] for i in range(len(superposition))]
})

correlation = df[['fisher_concentration', 'superposition_score']].corr(method='spearman')
print(f"Fisher concentration × Superposition: ρ = {correlation.iloc[0,1]:.3f}")

# Research question answered:
# "Do layers with concentrated Fisher (few dominant eigenvalues) show more superposition?"
```

### Cross-Metric Discovery: Geometry × Conflicts

```python
geometry = results.get('embedding_singularities')
conflicts = results.get('sample_conflicts')

# Do conflicting samples cluster near singularities?
singularity_density = geometry['density_per_token']
conflict_tokens = [c['token_id'] for c in conflicts['pairs']]

avg_density_conflicted = np.mean([singularity_density[t] for t in conflict_tokens])
avg_density_all = np.mean(list(singularity_density.values()))

print(f"Singularity density: conflicted={avg_density_conflicted:.3f}, all={avg_density_all:.3f}")

# Research question answered:
# "Do conflicting samples occur near geometric violations?"
```

### Engineering: LoRA Placement

```python
from peft import LoraConfig, get_peft_model

# Get Fisher importance from unified analysis
fisher_groups = results.get('grouped_fisher')
top_layers = sorted(fisher_groups, key=lambda k: fisher_groups[k]['mean'], reverse=True)[:4]

# Place LoRA only on high-Fisher layers
target_modules = [ln for ln in top_layers if 'self_attn' in ln or 'mlp' in ln]
config = LoraConfig(
    target_modules=target_modules,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)
model = get_peft_model(model, config)

# Freeze everything else
for name, p in model.named_parameters():
    if not any(t in name for t in target_modules):
        p.requires_grad_(False)

print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
```

### Engineering: Data Curation via Sample Conflicts

```python
# Get sample conflicts from unified analysis
conflicts = results.get('sample_conflicts')

# Filter by effect size and statistical significance
high_conflict_pairs = [
    c for c in conflicts['pairs']
    if c['effect_size'] > 0.5 and c['p_value'] < 0.01 and c['fdr_significant']
]

# Remove one sample from each high-conflict pair (keep lower loss)
samples_to_remove = set()
for conflict in high_conflict_pairs:
    if conflict['sample_a_loss'] > conflict['sample_b_loss']:
        samples_to_remove.add(conflict['sample_a_idx'])
    else:
        samples_to_remove.add(conflict['sample_b_idx'])

print(f"Removing {len(samples_to_remove)} conflicting samples")
clean_dataset = [s for i, s in enumerate(train_dataset) if i not in samples_to_remove]
```

---

