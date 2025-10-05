# TensorScope for Neuroscience & Cognitive Science

**The Problem:** Computational neuroscience labs typically analyze neural recordings and model internals with separate tools. TensorScope provides model analysis metrics (Fisher information, geometry, circuits, superposition) that can be compared with neural data using your existing analysis pipeline.

**What's Built-In vs. What You Implement:**
- ✅ **TensorScope provides:** Model metrics (CKA, superposition, geometry, circuits, Fisher)
- 🔧 **You implement:** Comparison with your neural data (RDMs, regression, alignment)

---

## Table of Contents
- [Quick Start for Neuroscientists](#quick-start-for-neuroscientists)
- [What TensorScope Actually Provides](#what-tensorscope-actually-provides)
- [Research Questions You Can Investigate](#research-questions-you-can-investigate)
  - [1. Representational Alignment](#1-representational-alignment)
  - [2. Neural Predictivity](#2-neural-predictivity)
  - [3. Polysemanticity & Superposition](#3-polysemanticity--superposition)
  - [4. Manifold Geometry](#4-manifold-geometry)
  - [5. Circuit Motifs](#5-circuit-motifs)
- [Example Workflows](#example-workflows)
- [Integration with Neuroscience Tools](#integration-with-neuroscience-tools)

---

## Quick Start for Neuroscientists

```python
from unified_model_analysis import UnifiedModelAnalyzer

# Analyze your model
analyzer = UnifiedModelAnalyzer()
results = analyzer.analyze_models([your_vision_model])

# Get model metrics (built-in)
cka_similarity = results.get('cka_similarity')          # ✅ Layer-wise CKA
effective_rank = results.get('effective_rank')          # ✅ Dimensionality
superposition = results.get('superposition_regime')     # ✅ Feature interference
geometry = results.get('embedding_singularities')       # ✅ Manifold violations
circuits = results.get('qkov_pairing')                  # ✅ Attention circuits

# Export for your neural analysis pipeline
results.save_report("model_analysis.json")

# Now use YOUR tools to compare with neural data
# (sklearn, Brain-Score, RSA Toolbox, etc.)
```

---

## What TensorScope Actually Provides

### ✅ **Built-In Model Metrics:**
- **CKA similarity**: Layer-wise representational similarity (within model)
- **Superposition metrics**: Feature interference, sparsity, polysemanticity
- **Manifold geometry**: Singularities, curvature, effective rank
- **Circuit analysis**: QK-OV pairing, induction heads, attention flow
- **Fisher information**: Parameter importance, task overlap

### 🔧 **What You Need to Implement:**
- **Neural RDM comparison**: Use your neural data + scipy/sklearn
- **Ridge/PLS regression**: Use sklearn on extracted features
- **Brain-Score integration**: Use TensorScope outputs as inputs
- **RSA Toolbox export**: Convert TensorScope metrics to RSA format
- **Alignment scoring**: Compare model metrics to your neural benchmarks

**TensorScope is a model analysis tool, not a neuroscience analysis tool.** It provides the model half of the comparison; you provide the neural data half.

---

## Research Questions You Can Investigate

### 1. Representational Alignment

**Question:** Does your model's representational geometry match neural recordings?

**What TensorScope provides (built-in):**
- ✅ `compute_cka_similarity`: Layer-wise CKA within model
- ✅ `compute_block_cka_gap`: Block-wise representation similarity
- ✅ `compute_effective_rank`: Dimensionality per layer

**What you implement:**
- 🔧 Load your neural RDMs
- 🔧 Compute alignment scores (e.g., Spearman correlation)
- 🔧 Track alignment across training checkpoints

**Example workflow:**
```python
# Step 1: Get model representations (TensorScope)
results = analyzer.analyze_models([model])
model_cka = results.get('cka_similarity')  # Dict of layer similarities

# Step 2: Compare to neural data (YOUR CODE)
import numpy as np
from scipy.stats import spearmanr

# Load your neural RDM
neural_rdm = np.load('your_neural_rdm.npy')  # Your data

# Extract model layer representations
# NOTE: TensorScope doesn't provide direct layer activations export
# You'll need to extract them yourself using model hooks
from torch.nn import functional as F

def get_layer_activations(model, batch):
    """Helper function YOU implement"""
    activations = {}
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for layers you care about
    for name, module in model.named_modules():
        if 'layer' in name:  # Adjust for your architecture
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model(**batch)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activations

# Get activations
layer_acts = get_layer_activations(model, batch)

# Compute RDM for each layer
from scipy.spatial.distance import pdist, squareform

layer_rdms = {}
for layer_name, acts in layer_acts.items():
    # Flatten to [samples x features]
    acts_flat = acts.reshape(acts.shape[0], -1).cpu().numpy()
    # Compute pairwise distances
    distances = pdist(acts_flat, metric='correlation')
    rdm = squareform(distances)
    layer_rdms[layer_name] = rdm

# Compare to neural RDM
alignment_scores = {}
for layer_name, model_rdm in layer_rdms.items():
    rho, p = spearmanr(model_rdm.flatten(), neural_rdm.flatten())
    alignment_scores[layer_name] = {'rho': rho, 'p': p}

best_layer = max(alignment_scores, key=lambda k: alignment_scores[k]['rho'])
print(f"Best neural alignment: {best_layer} (ρ={alignment_scores[best_layer]['rho']:.3f})")
```

**Track across training:**
```python
# Analyze multiple checkpoints
checkpoints = ['ckpt_1k.pt', 'ckpt_5k.pt', 'ckpt_10k.pt', 'final.pt']

alignment_over_time = []
for ckpt_path in checkpoints:
    model = load_model(ckpt_path)  # Your loading function
    layer_acts = get_layer_activations(model, batch)
    
    # Compute alignment (your code from above)
    alignment = compute_alignment_to_neural_rdm(layer_acts, neural_rdm)
    alignment_over_time.append(alignment)

# Plot trajectory
import matplotlib.pyplot as plt
plt.plot([1000, 5000, 10000, 100000], alignment_over_time)
plt.xlabel('Training step')
plt.ylabel('Neural alignment (Spearman ρ)')
plt.title('When does model become brain-like?')
plt.savefig('alignment_trajectory.png')
```

**Why this matters:** Identifies which layers and training stages produce representations most similar to biological neural networks.

---

### 2. Neural Predictivity

**Question:** Which model layers best predict neural/behavioral responses?

**What TensorScope provides (built-in):**
- ✅ `grouped_fisher`: Fisher importance per layer/head
- ✅ `compute_effective_rank`: Layer dimensionality
- ✅ `compute_superposition_regime`: Feature organization

**What you implement:**
- 🔧 Extract layer activations (using hooks, see above)
- 🔧 Ridge/PLS regression (using sklearn)
- 🔧 Fisher-weighted ablations (mask features and re-run)

**Example workflow:**
```python
# Step 1: Extract layer features (YOUR CODE - see Section 1)
layer_features = get_layer_activations(model, batch)

# Step 2: Regress neural responses (YOUR CODE)
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score

neural_responses = load_your_neural_data()  # [samples x neurons]

# Test each layer
predictivity_scores = {}
for layer_name, features in layer_features.items():
    # Flatten features
    X = features.reshape(features.shape[0], -1).cpu().numpy()
    y = neural_responses
    
    # Ridge regression with cross-validation
    ridge = Ridge(alpha=1.0)
    scores = cross_val_score(ridge, X, y, cv=5, scoring='r2')
    predictivity_scores[layer_name] = scores.mean()

print(f"Best predictive layer: {max(predictivity_scores, key=predictivity_scores.get)}")

# Step 3: Validate with Fisher-weighted ablations (YOUR CODE)
results = analyzer.analyze_models([model])
fisher_importance = results.get('grouped_fisher')

# Identify high-Fisher features
# NOTE: Fisher is per-layer/head, not per-feature
# You'll need to map this to individual features based on your architecture
high_fisher_layers = sorted(fisher_importance.items(), 
                            key=lambda x: x[1], reverse=True)[:5]

print(f"High-Fisher layers: {[name for name, _ in high_fisher_layers]}")

# Ablate these layers and re-test neural predictivity
# (Implementation depends on your specific ablation strategy)
```

**Why this matters:** Identifies which model features are most relevant for predicting neural activity. Fisher importance provides a principled way to prioritize features for ablation studies.

---

### 3. Polysemanticity & Superposition

**Question:** How much superposition (multiple features per neuron) exists in your model vs. biological neurons?

**What TensorScope provides (built-in):**
- ✅ `compute_superposition_regime`: Quantifies feature interference
- ✅ `compute_feature_sparsity`: L0/L1 activation sparsity
- ✅ `compute_vector_interference`: Dimensional overlap

**What you implement:**
- 🔧 Compute selectivity indices for your neural data
- 🔧 Compare distributions (model vs. neural)
- 🔧 Identify polysemantic neurons in both

**Example workflow:**
```python
# Step 1: Get model superposition metrics (TensorScope)
results = analyzer.analyze_models([model])
superposition = results.get('superposition_regime')
sparsity = results.get('feature_sparsity')

# Per-layer analysis
for layer_name in superposition.keys():
    interference = superposition[layer_name]['interference_score']  # 0-1
    sparsity_l0 = sparsity[layer_name]['l0_norm']  # Active features
    
    print(f"{layer_name}:")
    print(f"  Superposition: {interference:.3f}")
    print(f"  Active features: {sparsity_l0:.1f}")

# Step 2: Compare to neural selectivity (YOUR CODE)
def compute_selectivity_index(responses, category_labels):
    """
    Compute selectivity index for each neuron.
    Higher = more selective (fewer categories activate neuron)
    """
    selectivity = []
    for neuron_idx in range(responses.shape[1]):
        neuron_resp = responses[:, neuron_idx]
        
        # Compute mean response per category
        category_means = []
        for cat in np.unique(category_labels):
            cat_mask = category_labels == cat
            category_means.append(neuron_resp[cat_mask].mean())
        
        # Selectivity = (max - mean) / (max + mean)
        max_resp = max(category_means)
        mean_resp = np.mean(category_means)
        si = (max_resp - mean_resp) / (max_resp + mean_resp + 1e-8)
        selectivity.append(si)
    
    return np.array(selectivity)

# Your neural data
neural_responses = load_your_neural_data()  # [samples x neurons]
category_labels = load_category_labels()     # [samples]

neural_selectivity = compute_selectivity_index(neural_responses, category_labels)

# Model features (extract using hooks)
layer_features = get_layer_activations(model, batch)
model_features = layer_features['layer_10']  # Pick a layer
model_selectivity = compute_selectivity_index(
    model_features.reshape(-1, model_features.shape[-1]).cpu().numpy(),
    category_labels
)

# Compare distributions
from scipy.stats import ks_2samp
stat, p = ks_2samp(neural_selectivity, model_selectivity)
print(f"Selectivity distributions differ: p={p:.4f}")

# Plot
import matplotlib.pyplot as plt
plt.hist(neural_selectivity, bins=50, alpha=0.5, label='Neural')
plt.hist(model_selectivity, bins=50, alpha=0.5, label='Model')
plt.xlabel('Selectivity Index')
plt.ylabel('Count')
plt.legend()
plt.savefig('selectivity_comparison.png')
```

**Why this matters:** Superposition is a key efficiency mechanism in both artificial and biological networks. Quantifying it enables direct comparison of coding strategies.

---

### 4. Manifold Geometry

**Question:** How does geometric structure relate to category separability and invariance?

**What TensorScope provides (built-in):**
- ✅ `compute_effective_rank`: Intrinsic dimensionality
- ✅ `embedding_singularities`: Geometric anomalies (Robinson et al.)
- ✅ `manifold_curvature`: Ricci curvature (if implemented)

**What you implement:**
- 🔧 Category separability analysis (LDA, SVM)
- 🔧 Invariance testing (transformations)
- 🔧 Correlation with behavioral metrics

**Example workflow:**
```python
# Step 1: Get model geometry (TensorScope)
results = analyzer.analyze_models([model])
effective_rank = results.get('effective_rank')
singularities = results.get('embedding_singularities')

# Per-layer geometry
for layer_name in effective_rank.keys():
    rank = effective_rank[layer_name]
    n_singularities = singularities.get(layer_name, {}).get('count', 0)
    
    print(f"{layer_name}:")
    print(f"  Effective rank: {rank:.1f}")
    print(f"  Singularities: {n_singularities}")

# Step 2: Link to category separability (YOUR CODE)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Extract layer features
layer_features = get_layer_activations(model, batch)
category_labels = load_category_labels()

separability_per_layer = {}
for layer_name, features in layer_features.items():
    X = features.reshape(features.shape[0], -1).cpu().numpy()
    
    lda = LinearDiscriminantAnalysis()
    lda.fit(X, category_labels)
    separability = lda.score(X, category_labels)
    separability_per_layer[layer_name] = separability

# Correlate geometry with separability
import pandas as pd
from scipy.stats import spearmanr

df = pd.DataFrame({
    'layer': list(effective_rank.keys()),
    'effective_rank': [effective_rank[l] for l in effective_rank.keys()],
    'singularities': [singularities.get(l, {}).get('count', 0) for l in effective_rank.keys()],
    'separability': [separability_per_layer.get(l, 0) for l in effective_rank.keys()]
})

rho_rank, p_rank = spearmanr(df['effective_rank'], df['separability'])
rho_sing, p_sing = spearmanr(df['singularities'], df['separability'])

print(f"Rank vs separability: ρ={rho_rank:.3f}, p={p_rank:.4f}")
print(f"Singularities vs separability: ρ={rho_sing:.3f}, p={p_sing:.4f}")
```

**Why this matters:** Geometric structure constrains what representations can be learned and how they generalize. Relating geometry to behavioral metrics connects representation structure to function.

---

### 5. Circuit Motifs

**Question:** Do attention circuits in transformers correspond to compositional structure?

**What TensorScope provides (built-in):**
- ✅ `qkov_pairing`: QK-OV composition patterns
- ✅ `induction_head_strength`: In-context learning circuits
- ✅ `attention_flow_patterns`: Information routing

**What you implement:**
- 🔧 Neural circuit graphs (from your data)
- 🔧 Graph property comparison (degree, betweenness)
- 🔧 Compositional generalization tests

**Example workflow:**
```python
# Step 1: Get model circuits (TensorScope)
results = analyzer.analyze_models([transformer_model])
qkov_pairing = results.get('qkov_pairing')
induction_heads = results.get('induction_head_strength')
attention_flow = results.get('attention_flow_patterns')

# Identify strong circuit motifs
strong_induction = [h for h, score in induction_heads.items() if score > 0.8]
print(f"Strong induction heads: {strong_induction}")

# Step 2: Compare to neural circuits (YOUR CODE)
# Load your neural connectivity data
neural_circuit_graph = load_neural_connectivity()  # Adjacency matrix

# Build model circuit graph from attention patterns
def build_attention_graph(attention_flow):
    """Convert attention patterns to graph"""
    # YOUR IMPLEMENTATION
    # Extract attention weights, threshold, create adjacency matrix
    pass

model_circuit_graph = build_attention_graph(attention_flow)

# Compare graph properties
import networkx as nx
from scipy.stats import ks_2samp

neural_graph = nx.Graph(neural_circuit_graph)
model_graph = nx.Graph(model_circuit_graph)

# Degree distributions
neural_degrees = list(nx.degree_centrality(neural_graph).values())
model_degrees = list(nx.degree_centrality(model_graph).values())

stat, p = ks_2samp(neural_degrees, model_degrees)
print(f"Degree distributions differ: p={p:.4f}")

# Hub identification
neural_hubs = sorted(nx.betweenness_centrality(neural_graph).items(), 
                     key=lambda x: x[1], reverse=True)[:10]
model_hubs = sorted(nx.betweenness_centrality(model_graph).items(), 
                    key=lambda x: x[1], reverse=True)[:10]

print(f"Neural hubs: {neural_hubs}")
print(f"Model hubs: {model_hubs}")
```

**Why this matters:** Attention mechanisms may implement similar compositional principles as neural circuits. Systematic comparison reveals shared computational motifs.

---

## Example Workflows

### Workflow 1: Brain-Score Style Benchmarking

```python
from unified_model_analysis import UnifiedModelAnalyzer

# Your models
models = [alexnet, resnet50, vit_base, your_custom_model]

# Analyze all models (TensorScope)
analyzer = UnifiedModelAnalyzer()
all_results = {}

for model in models:
    results = analyzer.analyze_models([model])
    all_results[model.name] = results

# Compare to neural benchmarks (YOUR CODE)
neural_benchmarks = load_neural_benchmarks()  # Your data

brain_scores = {}
for model_name, results in all_results.items():
    # Extract model metrics
    cka = results.get('cka_similarity')
    geometry = results.get('embedding_singularities')
    
    # YOUR COMPARISON CODE
    # Compute alignment, predictivity, geometric similarity
    # using your neural data and analysis functions
    
    brain_scores[model_name] = {
        'alignment': your_alignment_function(cka, neural_benchmarks),
        'predictivity': your_predictivity_function(model, neural_benchmarks),
        'geometry': your_geometry_function(geometry, neural_benchmarks),
    }

# Rank models
ranked = sorted(brain_scores.items(), 
                key=lambda x: sum(x[1].values()), reverse=True)
print("Brain-score rankings:")
for rank, (model_name, scores) in enumerate(ranked, 1):
    print(f"{rank}. {model_name}: {sum(scores.values()):.3f}")
```

---

## Integration with Neuroscience Tools

TensorScope outputs can be integrated with standard neuroscience analysis tools, but **you need to write the integration code**.

### Export to MATLAB/Python

```python
results = analyzer.analyze_models([model])

# Save as JSON for MATLAB
results.save_report("model_analysis.json")

# Or export specific metrics as NumPy arrays
import numpy as np
layer_features = get_layer_activations(model, batch)  # Your function
np.save('layer_features.npy', layer_features)

cka_matrix = results.get('cka_similarity')
# Convert dict to array for your specific use case
```

### Integration with Brain-Score

```python
# TensorScope provides model metrics
# You provide the Brain-Score integration

from brainscore import benchmark_pool

# Extract features using YOUR code (see Section 1)
features = get_layer_activations(model, batch)

# Use Brain-Score benchmark
brain_score = benchmark_pool['dicarlo.MajajHong2015'].score(features)
print(f"Brain-Score: {brain_score}")
```

### Integration with RSA Toolbox

```python
# Export RDMs for RSA Toolbox (YOUR CODE)

# Get model representations
layer_rdms = compute_layer_rdms(model, batch)  # Your function from Section 1

# Convert to RSA Toolbox format
from rsatoolbox.rdm import RDMs

rsa_rdms = RDMs(
    dissimilarities=np.array(list(layer_rdms.values())),
    descriptors={'layer': list(layer_rdms.keys())}
)

# Now use RSA Toolbox functions
from rsatoolbox.inference import compare
comparison = compare(rsa_rdms, neural_rdms, method='corr')
```

---

## Summary: What's Built-In vs. What You Implement

### ✅ **TensorScope Provides (Built-In):**
- CKA similarity (within model)
- Superposition & feature sparsity metrics
- Manifold geometry (singularities, effective rank)
- Circuit analysis (QK-OV, induction heads)
- Fisher information (importance, overlap)

### 🔧 **You Need to Implement:**
- Layer activation extraction (using PyTorch hooks)
- Neural RDM loading and comparison
- Ridge/PLS regression (using sklearn)
- Alignment scoring (using scipy)
- Brain-Score integration
- RSA Toolbox export
- Graph analysis for circuits
- Selectivity index computation

**TensorScope is a model analysis framework, not a neuroscience-specific tool.** It provides comprehensive model metrics that can be compared with neural data using standard neuroscience analysis libraries (sklearn, scipy, Brain-Score, RSA Toolbox).

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

**Back to [main README](../README.md)**