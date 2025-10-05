# Research Recipes

Detailed examples of how to use TensorScope for specific research tasks.

[← Back to README](../README.md)

---

## Research Recipes

### Training Dynamics & Optimization

**Gradient health (vanishing/exploding, SNR):**
```python
from GradientAnalysis import GradientAnalysis
grad = GradientAnalysis()
pathology = grad.compute_gradient_pathology(model, batch)
# Output: counts, SNR, dead neurons per layer
```

**Multi-task interference (where and how much):**
```python
conflict = grad.compute_gradient_conflict_pcgrad(model, task1_batch, task2_batch)
layer_conflicts = grad.compute_layer_gradient_alignment(model, task1_batch, task2_batch)
fisher_overlap = analyzer.compute_fisher_overlap(task1_data, task2_data)
# Output: conflict scores, per-layer angles, Fisher mask overlap (0-1)
```

**Sharpness & loss landscape:**
```python
from ICLRMetrics import ICLRMetrics
metrics = ICLRMetrics()
hessian_eigs = metrics.compute_hessian_eigenvalues_lanczos(model, batch, k=20)
landscape = metrics.compute_loss_landscape_2d(model, batch)
sharpness = metrics.compute_sam_sharpness_fixed(model, batch)
# Output: top eigenvalues, 2D grids, SAM deltas
```

### Fisher Information & Curvature

**Full curvature characterization (recommended):**
```python
from ICLRMetrics import ICLRMetrics
metrics = ICLRMetrics()

# Fisher spectrum (complete eigenvalue distribution)
spectrum = metrics.compute_fisher_spectrum(model, batch)
# Output: all eigenvalues, condition number, spectral concentration

# Top eigenvalues (memory-efficient for large models)
top_eigs = metrics.compute_fisher_eigenvalues_lanczos(model, batch, k=20)
# Output: top-k eigenvalues, eigenvectors, convergence info
# Saves ~48GB for 1.5B models vs full decomposition
```

**Coarse importance for LoRA/masking:**
```python
# Grouped Fisher (fast approximation for architecture decisions)
fisher_groups = analyzer.get_group_fisher(task, mode='accumulated')
top_layers = sorted(fisher_groups, key=fisher_groups.get, reverse=True)[:4]
# Output: per-layer/head importance + CIs (coarse, but fast)
```

**Task similarity (share or separate):**
```python
overlap = analyzer.compute_fisher_overlap(task1_data, task2_data)
# < 0.2: use adapters/separate heads
# > 0.8: shared backbone OK
```

**Curvature-aware pruning:**
```python
# Use spectral gaps to find safe pruning thresholds
spectrum = metrics.compute_fisher_spectrum(model, batch)
spectral_gap = spectrum['eigenvalues'][1] - spectrum['eigenvalues'][0]
# Large gap → safe to prune; small gap → be conservative
```

### Mechanistic Interpretability

**Attention circuits (induction, QK-OV):**
```python
from mechanistic.mechanistic_analyzer_unified import MechanisticAnalyzer
mech = MechanisticAnalyzer(model)
induction = mech.compute_induction_head_strength(model, batch)
qk_ov = mech.compute_qk_ov_pairing(model, batch)
# Output: strong induction heads, circuit pairings
```

**QKOV interference (which circuits conflict):**
```python
from fisher.qkov.qkov_interference import compute_heatmap
heatmap = compute_heatmap(model, task1_batch, task2_batch)
# Output: per-block/head interference scores
```

### Representation Analysis

**Similarity across checkpoints:**
```python
from ModularityMetrics import ModularityMetrics
from RepresentationAnalysisMetrics import RepresentationAnalysisMetrics
modularity = ModularityMetrics()
repr_metrics = RepresentationAnalysisMetrics()

cka = modularity.compute_linear_cka_per_layer(model1, model2, batch)
similarity = repr_metrics.compute_representational_similarity(model1, model2, batch)
# Output: per-layer CKA, representation similarity scores
```

**Superposition regime:**
```python
from superposition.metrics.paper_metrics import compute_superposition_regime
regime = compute_superposition_regime(model, batch)
# Output: polysemanticity scores, interference patterns
```

**Geometric stability:**
```python
from manifold_violations.embedding_singularity_metrics import compute_metrics
from manifold_violations.robinson_fiber_bundle_test import analyze_embedding_space
singularities = compute_metrics(model, batch)
fiber_test = analyze_embedding_space(model, batch)
# Output: singularity density, violation flags, transition radii
```

### Data Influence & Attribution

**Which samples help/harm (TracIn):**
```python
critical = metrics.find_critical_samples(
    model, train_dataset,
    checkpoints=['ckpt_1k.pt', 'ckpt_5k.pt', 'ckpt_10k.pt'],
    test_sample=test_batch
)
# Output: negative/positive influence indices, trajectories
```

**Sample-level conflicts:**
```python
conflicts = analyzer.analyze_sample_conflicts(task1_data, task2_data)
# Output: specific pairs (sample i, sample j), p-values, effect sizes, FDR flags
```

### Pruning & Lottery Tickets

**Iterative magnitude pruning:**
```python
from lottery_tickets.imp_wrapper import compute_iterative_magnitude_pruning
tickets = compute_iterative_magnitude_pruning(model, train_loader, sparsities=[0.5, 0.8, 0.9])
# Output: winning ticket masks at each sparsity
```

**Multiple importance metrics:**
```python
from lottery_tickets.importance_scoring import *
fisher_imp = compute_fisher_importance(model, batch)
taylor_imp = compute_taylor_importance(model, batch)
hybrid_imp = compute_hybrid_importance(model, batch)  # Combines multiple signals
```

---

