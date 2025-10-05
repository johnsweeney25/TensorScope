# TensorScope for Neuroscience & Cognitive Science

**The Problem:** Computational neuroscience labs typically analyze neural recordings and model internals with separate tools, making it hard to systematically compare biological and artificial neural networks. TensorScope provides a unified framework to compute model internals (Fisher information, geometry, circuits) in the same pipeline you use for neural analysis (RDMs, regression, manifolds).

---

## Table of Contents
- [Quick Start for Neuroscientists](#quick-start-for-neuroscientists)
- [Research Questions You Can Answer](#research-questions-you-can-answer)
  - [1. Representational Alignment](#1-representational-alignment)
  - [2. Neural Predictivity](#2-neural-predictivity)
  - [3. Polysemanticity & Superposition](#3-polysemanticity--superposition)
  - [4. Manifold Geometry](#4-manifold-geometry)
  - [5. Circuit Motifs](#5-circuit-motifs)
- [Example Workflows](#example-workflows)
- [Relevant Metrics](#relevant-metrics)
- [Integration with Neuroscience Tools](#integration-with-neuroscience-tools)
- [Publications Using TensorScope for Neuro-AI](#publications-using-tensorscope-for-neuro-ai)

---

## Quick Start for Neuroscientists

```python
from unified_model_analysis import UnifiedModelAnalyzer

# Analyze your model
analyzer = UnifiedModelAnalyzer()
results = analyzer.analyze_models([your_vision_model])

# Get representations for comparison with neural data
cka_similarity = results.get('cka_similarity')          # Compare layer representations
effective_rank = results.get('effective_rank')          # Dimensionality per layer
superposition = results.get('superposition_regime')     # Feature interference
geometry = results.get('embedding_singularities')       # Manifold violations
attention_patterns = results.get('attention_flow')      # Circuit structure

# Export for analysis in your neuroscience pipeline
results.save_report("model_analysis.json")
```

**What you get:**
- Layer-wise representational structure (CKA, RSA, effective rank)
- Geometric properties (singularities, curvature, manifold violations)
- Circuit analysis (attention patterns, induction heads, QK-OV pairing)
- Feature organization (superposition, polysemanticity, sparsity)
- All metrics computed under identical conditions (same batch, seed, device)

---

## Research Questions You Can Answer

### 1. Representational Alignment

**Question:** Does your model's representational geometry match neural recordings?

**What TensorScope provides:**
- **CKA (Centered Kernel Alignment)**: Compare layer-wise representations to neural RDMs
- **RSA (Representational Similarity Analysis)**: Correlation-based similarity between model and neural representational dissimilarity matrices
- **Trajectory tracking**: Analyze how alignment evolves across training checkpoints

**Example workflow:**
```python
# Compute model representations
results = analyzer.analyze_models([model])
model_rdm = results.get('cka_similarity')

# Compare to neural RDM (your data)
import numpy as np
from scipy.stats import spearmanr

neural_rdm = load_your_neural_rdm()  # Your neural recordings

# Find best-matching layer
alignment_scores = {}
for layer_name, layer_rdm in model_rdm.items():
    rho, p = spearmanr(layer_rdm.flatten(), neural_rdm.flatten())
    alignment_scores[layer_name] = {'rho': rho, 'p': p}

best_layer = max(alignment_scores, key=lambda k: alignment_scores[k]['rho'])
print(f"Best neural alignment: {best_layer} (ρ={alignment_scores[best_layer]['rho']:.3f})")
```

**Track across training:**
```python
# Analyze checkpoints
checkpoints = ['ckpt_1k.pt', 'ckpt_5k.pt', 'ckpt_10k.pt', 'final.pt']
results = analyzer.analyze_models(checkpoints)

# See when model becomes "brain-like"
alignment_trajectory = results.get('cka_trajectory')
# Plot: alignment_score vs training_step
```

**Why this matters:** Identifies which layers and training stages produce representations most similar to biological neural networks. Informs architecture design and training procedures to maximize brain-likeness.

---

### 2. Neural Predictivity

**Question:** Which model layers best predict neural/behavioral responses?

**What TensorScope provides:**
- **Layer-wise representations**: Extract features from each layer
- **Fisher importance**: Weight features by their importance to the model's task
- **Ablation analysis**: Test causal importance via Fisher-weighted ablations

**Example workflow:**
```python
# Step 1: Extract layer features
results = analyzer.analyze_models([model])
layer_features = results.get('layer_activations')  # [layers x samples x features]

# Step 2: Regress neural responses on layer features
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

neural_responses = load_your_neural_data()  # [samples x neurons]

# Test each layer
predictivity_scores = {}
for layer_name, features in layer_features.items():
    # Ridge regression
    ridge = Ridge(alpha=1.0)
    ridge.fit(features, neural_responses)
    r2 = ridge.score(features, neural_responses)
    predictivity_scores[layer_name] = r2

# Step 3: Validate with Fisher-weighted ablations
fisher_importance = results.get('grouped_fisher')

# Ablate high-Fisher vs low-Fisher features
high_fisher_features = get_top_k_features(fisher_importance, k=100)
low_fisher_features = get_bottom_k_features(fisher_importance, k=100)

# Re-run regression with ablated features
# Compare: does ablating high-Fisher features hurt neural predictivity more?
```

**Advanced: Partial Least Squares (PLS)**
```python
# PLS finds latent components that maximize covariance
pls = PLSRegression(n_components=10)
pls.fit(layer_features, neural_responses)

# Get latent components
model_components = pls.x_scores_  # Model latent space
neural_components = pls.y_scores_  # Neural latent space

# Analyze component structure
component_importance = np.abs(pls.x_weights_)  # Feature importance per component
```

**Why this matters:** Identifies which model features are most relevant for predicting neural activity. Fisher-weighted ablations test causal importance, not just correlation. Enables targeted model interventions to improve neural alignment.

---

### 3. Polysemanticity & Superposition

**Question:** How much superposition (multiple features per neuron) exists in your model vs. biological neurons?

**What TensorScope provides:**
- **Superposition strength**: Quantifies feature interference (how many features per dimension)
- **Feature sparsity**: Measures activation sparsity (L0/L1 norms)
- **Polysemanticity analysis**: Identifies neurons responding to multiple unrelated features

**Example workflow:**
```python
results = analyzer.analyze_models([model])

# Superposition metrics
superposition = results.get('superposition_regime')
sparsity = results.get('feature_sparsity')

# Per-layer analysis
for layer_name in superposition.keys():
    interference = superposition[layer_name]['interference_score']  # 0-1
    sparsity_l0 = sparsity[layer_name]['l0_norm']  # Active features
    
    print(f"{layer_name}:")
    print(f"  Superposition: {interference:.3f}")
    print(f"  Active features: {sparsity_l0:.1f}/{total_features}")
```

**Compare to neural selectivity:**
```python
# Your neural tuning curves
neural_selectivity = compute_selectivity_index(neural_responses)  # Your function

# Model feature selectivity
model_selectivity = compute_selectivity_index(layer_features)

# Compare distributions
from scipy.stats import ks_2samp
stat, p = ks_2samp(neural_selectivity, model_selectivity)
print(f"Selectivity distributions differ: p={p:.4f}")
```

**Relate to tuning curves:**
```python
# Identify polysemantic neurons (respond to multiple categories)
polysemantic_neurons = identify_polysemantic(layer_features, category_labels)

# Compare to neural polysemanticity
neural_polysemantic = identify_polysemantic(neural_responses, category_labels)

print(f"Model polysemantic: {len(polysemantic_neurons)}/{total_neurons}")
print(f"Neural polysemantic: {len(neural_polysemantic)}/{total_neural_units}")
```

**Why this matters:** Superposition is a key efficiency mechanism in both artificial and biological networks. Quantifying it enables direct comparison of coding strategies. High superposition may explain why both ANNs and brains can represent more features than they have neurons.

---

### 4. Manifold Geometry

**Question:** How does geometric structure relate to category separability and invariance?

**What TensorScope provides:**
- **Effective rank**: Intrinsic dimensionality of representations
- **Embedding singularities**: Geometric anomalies (Robinson et al. 2025)
- **Manifold curvature**: Ricci curvature for geometric characterization
- **Fiber bundle tests**: Statistical tests for manifold hypothesis

**Example workflow:**
```python
results = analyzer.analyze_models([model])

# Geometric metrics
effective_rank = results.get('effective_rank')
singularities = results.get('embedding_singularities')
curvature = results.get('manifold_curvature')

# Per-layer geometry
for layer_name in effective_rank.keys():
    rank = effective_rank[layer_name]
    n_singularities = singularities[layer_name]['count']
    mean_curvature = curvature[layer_name]['mean_ricci']
    
    print(f"{layer_name}:")
    print(f"  Effective rank: {rank:.1f}")
    print(f"  Singularities: {n_singularities}")
    print(f"  Mean curvature: {mean_curvature:.4f}")
```

**Link to category separability:**
```python
# Compute category separability in representation space
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(layer_features, category_labels)
separability = lda.score(layer_features, category_labels)

# Correlate with geometry
import pandas as pd
df = pd.DataFrame({
    'layer': layer_names,
    'effective_rank': [effective_rank[l] for l in layer_names],
    'singularities': [singularities[l]['count'] for l in layer_names],
    'separability': [separability_per_layer[l] for l in layer_names]
})

# Test correlations
from scipy.stats import spearmanr
rho_rank, p_rank = spearmanr(df['effective_rank'], df['separability'])
rho_sing, p_sing = spearmanr(df['singularities'], df['separability'])

print(f"Rank vs separability: ρ={rho_rank:.3f}, p={p_rank:.4f}")
print(f"Singularities vs separability: ρ={rho_sing:.3f}, p={p_sing:.4f}")
```

**Invariance analysis:**
```python
# Test invariance to transformations (rotation, translation, etc.)
transformed_features = apply_transformations(layer_features)  # Your function

# Measure geometric stability
from scipy.spatial.distance import cdist
original_distances = cdist(layer_features, layer_features)
transformed_distances = cdist(transformed_features, transformed_features)

invariance_score = np.corrcoef(
    original_distances.flatten(), 
    transformed_distances.flatten()
)[0, 1]

print(f"Geometric invariance: {invariance_score:.3f}")
```

**Why this matters:** Geometric structure constrains what representations can be learned and how they generalize. Singularities indicate regions of instability. Effective rank measures compression. Relating geometry to behavioral metrics (separability, invariance) connects representation structure to function.

---

### 5. Circuit Motifs

**Question:** Do attention circuits in transformers correspond to compositional/relational structure in neural circuits?

**What TensorScope provides:**
- **QK-OV pairing**: Identifies which attention heads compose (query-key → output-value chains)
- **Induction heads**: Detects in-context learning circuits
- **Attention flow patterns**: Analyzes information routing (previous token, skip connections, etc.)
- **Circuit specialization**: Measures task-specificity of attention heads

**Example workflow:**
```python
results = analyzer.analyze_models([transformer_model])

# Circuit analysis
qkov_pairing = results.get('qkov_pairing')
induction_heads = results.get('induction_head_strength')
attention_flow = results.get('attention_flow_patterns')

# Identify strong circuit motifs
strong_induction = [h for h, score in induction_heads.items() if score > 0.8]
print(f"Strong induction heads: {strong_induction}")

# QK-OV composition patterns
for head_pair, pairing_strength in qkov_pairing.items():
    if pairing_strength > 0.7:
        print(f"Strong composition: {head_pair} (strength={pairing_strength:.3f})")
```

**Relate to neural circuits:**
```python
# Your neural circuit analysis (e.g., from calcium imaging)
neural_circuit_graph = load_neural_connectivity()  # Adjacency matrix

# Model circuit graph (from attention patterns)
model_circuit_graph = build_attention_graph(attention_flow)

# Compare graph properties
from networkx import Graph, degree_centrality, betweenness_centrality

neural_graph = Graph(neural_circuit_graph)
model_graph = Graph(model_circuit_graph)

# Degree distributions
neural_degrees = list(degree_centrality(neural_graph).values())
model_degrees = list(degree_centrality(model_graph).values())

from scipy.stats import ks_2samp
stat, p = ks_2samp(neural_degrees, model_degrees)
print(f"Degree distributions differ: p={p:.4f}")

# Identify hub nodes (high betweenness)
neural_hubs = sorted(betweenness_centrality(neural_graph).items(), 
                     key=lambda x: x[1], reverse=True)[:10]
model_hubs = sorted(betweenness_centrality(model_graph).items(), 
                    key=lambda x: x[1], reverse=True)[:10]

print(f"Neural hubs: {neural_hubs}")
print(f"Model hubs: {model_hubs}")
```

**Compositional structure:**
```python
# Test if induction heads enable compositional generalization
# (similar to how neural circuits combine primitives)

# Ablate induction heads
ablated_model = ablate_heads(model, strong_induction)

# Test on compositional tasks
compositional_accuracy = test_compositional_generalization(ablated_model)
baseline_accuracy = test_compositional_generalization(model)

print(f"Compositional accuracy: {baseline_accuracy:.3f} → {compositional_accuracy:.3f}")
print(f"Induction heads necessary: {compositional_accuracy < baseline_accuracy}")
```

**Why this matters:** Attention mechanisms in transformers may implement similar compositional principles as neural circuits. QK-OV pairing resembles synaptic chains. Induction heads enable in-context learning, possibly analogous to rapid binding in hippocampus. Systematic comparison reveals shared computational motifs.

---

## Example Workflows

### Workflow 1: Brain-Score Style Benchmarking

```python
from unified_model_analysis import UnifiedModelAnalyzer

# Your models
models = [alexnet, resnet50, vit_base, your_custom_model]

# Analyze all models
analyzer = UnifiedModelAnalyzer()
all_results = {}

for model in models:
    results = analyzer.analyze_models([model])
    all_results[model.name] = results

# Compare to neural benchmarks
neural_benchmarks = load_neural_benchmarks()  # Your data

brain_scores = {}
for model_name, results in all_results.items():
    # CKA alignment
    cka = results.get('cka_similarity')
    alignment = compute_alignment(cka, neural_benchmarks['V1'])
    
    # Neural predictivity
    features = results.get('layer_activations')
    predictivity = compute_predictivity(features, neural_benchmarks['IT'])
    
    # Geometric similarity
    geometry = results.get('embedding_singularities')
    geo_score = compare_geometry(geometry, neural_benchmarks['geometry'])
    
    brain_scores[model_name] = {
        'alignment': alignment,
        'predictivity': predictivity,
        'geometry': geo_score,
        'overall': (alignment + predictivity + geo_score) / 3
    }

# Rank models
ranked = sorted(brain_scores.items(), key=lambda x: x[1]['overall'], reverse=True)
print("Brain-score rankings:")
for rank, (model_name, scores) in enumerate(ranked, 1):
    print(f"{rank}. {model_name}: {scores['overall']:.3f}")
```

### Workflow 2: Training Dynamics & Neural Alignment

```python
# Track how neural alignment evolves during training
checkpoints = [f'ckpt_step_{i}.pt' for i in [1000, 5000, 10000, 50000, 100000]]

results = analyzer.analyze_models(checkpoints)

# Extract alignment trajectory
alignment_over_time = []
for ckpt_results in results:
    cka = ckpt_results.get('cka_similarity')
    alignment = compute_alignment(cka, neural_rdm)
    alignment_over_time.append(alignment)

# Plot
import matplotlib.pyplot as plt
steps = [1000, 5000, 10000, 50000, 100000]
plt.plot(steps, alignment_over_time)
plt.xlabel('Training step')
plt.ylabel('Neural alignment (CKA)')
plt.title('When does model become brain-like?')
plt.savefig('alignment_trajectory.png')

# Identify critical transition point
max_increase_idx = np.argmax(np.diff(alignment_over_time))
critical_step = steps[max_increase_idx]
print(f"Largest alignment increase at step {critical_step}")
```

### Workflow 3: Cross-Species Comparison

```python
# Compare model to multiple species
species_data = {
    'macaque_V1': load_macaque_v1_data(),
    'macaque_IT': load_macaque_it_data(),
    'mouse_V1': load_mouse_v1_data(),
    'human_fMRI': load_human_fmri_data()
}

results = analyzer.analyze_models([model])

# Compute alignment to each species
species_alignment = {}
for species_name, neural_data in species_data.items():
    cka = results.get('cka_similarity')
    alignment = compute_alignment(cka, neural_data)
    species_alignment[species_name] = alignment

# Which species does model match best?
best_match = max(species_alignment, key=species_alignment.get)
print(f"Best match: {best_match} (alignment={species_alignment[best_match]:.3f})")

# Hierarchical clustering of alignments
from scipy.cluster.hierarchy import dendrogram, linkage
alignment_matrix = compute_alignment_matrix(species_data, model_representations)
linkage_matrix = linkage(alignment_matrix, method='ward')
dendrogram(linkage_matrix, labels=list(species_data.keys()) + ['model'])
plt.title('Representational similarity: Model vs. species')
plt.savefig('species_comparison.png')
```

---

## Relevant Metrics

**For representational alignment:**
- `cka_similarity`: Centered Kernel Alignment between layers
- `rsa_similarity`: Representational Similarity Analysis
- `block_cka_gap`: Alignment differences across model blocks

**For neural predictivity:**
- `layer_activations`: Features from each layer
- `grouped_fisher`: Importance weights for features
- `fisher_pruning_masks`: Identify critical features

**For polysemanticity:**
- `superposition_regime`: Feature interference quantification
- `feature_sparsity`: L0/L1 activation sparsity
- `vector_interference`: Dimensional overlap analysis

**For geometry:**
- `effective_rank`: Intrinsic dimensionality
- `embedding_singularities`: Geometric anomalies
- `manifold_curvature`: Ricci curvature per layer
- `robinson_fiber_test`: Manifold hypothesis testing

**For circuits:**
- `qkov_pairing`: Attention head composition
- `induction_head_strength`: In-context learning circuits
- `attention_flow_patterns`: Information routing
- `attention_entropy`: Attention distribution analysis

See [main README](../README.md) for complete metric list.

---

## Integration with Neuroscience Tools

TensorScope outputs are designed to integrate with standard neuroscience analysis pipelines:

**Export to MATLAB/Python:**
```python
results = analyzer.analyze_models([model])

# Save as JSON for MATLAB
results.save_report("model_analysis.json")

# Or export specific metrics as NumPy arrays
import numpy as np
np.save('layer_features.npy', results.get('layer_activations'))
np.save('cka_matrix.npy', results.get('cka_similarity'))
```

**Integration with Brain-Score:**
```python
# TensorScope features → Brain-Score benchmark
from brainscore import benchmark_pool

features = results.get('layer_activations')
brain_score = benchmark_pool['dicarlo.MajajHong2015'].score(features)
print(f"Brain-Score: {brain_score}")
```

**Integration with RSA Toolbox:**
```python
# Export RDMs in RSA Toolbox format
rdms = results.get('cka_similarity')

# Convert to RSA format
from rsatoolbox.rdm import RDMs
rsa_rdms = RDMs(
    dissimilarities=rdms,
    descriptors={'layer': layer_names}
)

# Now use RSA Toolbox functions
from rsatoolbox.inference import compare
comparison = compare(rsa_rdms, neural_rdms, method='corr')
```

**Integration with NMA (Neuromatch Academy) tools:**
```python
# Use with NMA dimensionality reduction tools
from nma_tools import compute_pca, compute_tsne

features = results.get('layer_activations')
pca_features = compute_pca(features, n_components=50)
tsne_embedding = compute_tsne(pca_features)

# Visualize alongside neural data
plot_neural_and_model_embeddings(neural_tsne, tsne_embedding)
```

---

## Publications Using TensorScope for Neuro-AI

*(This section will be populated as researchers publish work using TensorScope for neuroscience applications)*

**If you use TensorScope for neuroscience research, please let us know so we can list your work here!**

---

## Citation

If you use TensorScope for neuroscience research, please cite:

```bibtex
@software{tensorscope2024,
  title={TensorScope: Cross-Metric Neural Network Analysis},
  author={John Sweeney},
  year={2024},
  note={Infrastructure for systematic correlation discovery across optimization, representation, and interpretability metrics}
}
```

---

## Questions or Collaboration?

For neuroscience-specific questions or collaboration opportunities:
- Open an issue on GitHub with the `neuroscience` tag
- Email: [your email]
- We're particularly interested in validating these methods on real neural data!

---

**Back to [main README](../README.md)**
