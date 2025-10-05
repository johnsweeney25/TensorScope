# Manifold Violations: Testing the Robinson Hypothesis in LLM Embeddings

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 🎯 Overview

This module implements the groundbreaking methodology from **"Token Embeddings Violate the Manifold Hypothesis"** (Robinson et al., 2025), revealing fundamental geometric instabilities in Large Language Model embeddings that challenge core assumptions about how LLMs represent language.

### Why This Matters

The manifold hypothesis—that high-dimensional data lies on lower-dimensional smooth manifolds—underpins most deep learning theory. Robinson et al. proved this assumption **fails** for LLM token embeddings, with profound implications:

- 🔴 **Prompt Instability**: Semantically equivalent prompts produce wildly different outputs
- 🔴 **Model Non-Portability**: Prompts that work on one model fail on another
- 🔴 **Polysemy Singularities**: Words with multiple meanings create "black holes" in embedding space

Our implementation provides tools to detect, measure, and mitigate these violations.

## 📚 Theoretical Foundation

### The Core Discovery

> *"For a smooth d-dimensional manifold, volume grows as V(r) ∝ r^d. We find that token embeddings show **increasing slopes** in log-log space, violating this fundamental relationship."* — Robinson et al.

### Key Mathematical Insight

In a proper fiber bundle (a weaker assumption than manifolds):
```
log V(r) = d · log r + const  (expected: constant slope d)
```

But in LLM embeddings:
```
d(log V(r))/d(log r) increases with r  (violation!)
```

This increasing slope indicates **geometric instability** causing unpredictable model behavior.

## 🚀 Quick Start

```python
from manifold_violations import RobinsonFiberBundleTest, compute_ricci_curvature_debiased
import numpy as np

# Test fiber bundle hypothesis on embeddings
embeddings = model.get_embeddings(tokens)  # Shape: (n_tokens, embed_dim)

# 1. Robinson Volume Growth Test
tester = RobinsonFiberBundleTest(significance_level=0.001)
result = tester.test_point(embeddings, point_idx=0)

if result.violates_hypothesis:
    print(f"❌ Violation detected! p-value: {result.p_value:.4f}")
    print(f"   Increasing slopes: {result.increasing_slopes}")
    print(f"   Transition at radius: {result.transition_radius:.2f}")

# 2. Ricci Curvature Analysis
ricci_mean, ricci_std = compute_ricci_curvature_debiased(
    embeddings,
    k_neighbors=5,
    n_samples=20
)
print(f"Ricci curvature: {ricci_mean:.4f} ± {ricci_std:.4f}")

# Interpretation:
# ricci > 0: Clustering (potential overfitting)
# ricci < 0: Diverging (maintaining diversity)
# ricci ≈ 0: Flat geometry (stable)
```

## 🏗️ Module Architecture

```
manifold_violations/
├── Core Implementation
│   ├── robinson_fiber_bundle_test.py    # Robinson volume growth analysis
│   ├── fiber_bundle_core.py            # Core fiber bundle algorithms
│   ├── polysemy_detector.py            # Clustering-based polysemy detection
│   └── singularity_mapper.py           # Comprehensive singularity profiling
│
├── Geometric Analysis
│   ├── tractable_manifold_curvature_fixed.py  # Ollivier-Ricci curvature
│   └── manifold_fiber_integration.py          # Integrated manifold metrics
│
├── Practical Applications
│   ├── prompt_robustness_analyzer.py   # Prompt stability assessment
│   ├── token_stability_analyzer.py     # Token-level stability analysis
│   └── training_singularity_dynamics.py # Training dynamics monitoring
│
├── Advanced Analysis
│   ├── token_dimension_classifier.py   # Dimension-based classification
│   └── tangent_space_analysis.py       # Local tangent space analysis
│
├── Integration
│   └── embedding_singularity_metrics.py # Unified metrics for main framework
│
└── Tests
    └── tests/                           # Comprehensive unit test suite
        ├── test_robinson_fiber_bundle.py
        ├── test_polysemy_detector.py
        └── test_method_comparison.py
```

## 🔬 Mathematical Methods

### 1. Volume Growth Analysis (Robinson's Method)

**Implementation**: `robinson_fiber_bundle_test.py`

```python
# Core algorithm from paper
def analyze_volume_growth(embeddings, point_idx):
    # Count neighbors within increasing radii
    for r in radii:
        volumes[i] = count_points_within_radius(r, excluding_center=True)

    # Transform to log-log space
    log_volumes = np.log(volumes)
    log_radii = np.log(radii)

    # Compute slopes via three-point centered differences
    slopes = centered_diff(log_volumes, log_radii)

    # Detect violations (increasing slopes)
    return detect_increasing_slopes(slopes)
```

**Paper Quote**: *"We use three-point centered differences for numerical stability"*

### 2. CFAR Detector for Statistical Rigor

**Implementation**: Lines 259-300 of `robinson_fiber_bundle_test.py`

```python
def cfar_detector(signal):
    """Constant False Alarm Rate detector from paper"""
    threshold = -stats.norm.ppf(significance_level / 2) * noise_level
    return detect_discontinuities(signal, threshold)
```

**Paper Quote**: *"We employ a CFAR detector to control false positives while detecting slope discontinuities"*

### 3. Ollivier-Ricci Curvature (Our Enhancement)

**Implementation**: `tractable_manifold_curvature_fixed.py`

```python
def compute_ricci_curvature_debiased(points):
    """
    κ(x,y) = 1 - W₁(μₓ, μᵧ)/d(x,y)

    Where W₁ is 1-Wasserstein distance between
    probability measures from lazy random walks
    """
    # Uses debiased Sinkhorn for accurate optimal transport
```

**Based on**: Ollivier (2007) "Ricci curvature of metric spaces"

### 4. Intrinsic Dimension Estimation

**Implementation**: Levina-Bickel MLE in `fiber_bundle_core.py`

```python
def estimate_intrinsic_dimension(points):
    """Maximum Likelihood Estimation of manifold dimension"""
    # d_hat = (k-1) / Σ log(r_k/r_i)
    return median([local_dim_estimates])  # Robust to outliers
```

**Based on**: Levina & Bickel (2005)

## 🔍 Polysemy Detection (Our Contribution)

While Robinson et al. showed that polysemous tokens create singularities, they didn't provide a detection method. We developed a **clustering-based polysemy detector** that directly identifies tokens with multiple meanings.

### ⚠️ IMPORTANT DISTINCTION

- **Robinson Test** (`robinson_fiber_bundle_test.py`): Statistical hypothesis testing on volume-radius curves
- **Polysemy Detector** (`polysemy_detector.py`): Clustering analysis of k-nearest neighbors (our method)

These are **different approaches** that may flag different tokens!

### Usage

```python
from manifold_violations import PolysemyDetector

# Our clustering-based approach (NOT the Robinson method)
detector = PolysemyDetector(
    n_neighbors=50,
    clustering_method='hierarchical',  # or 'dbscan'
    metric='cosine'
)

# Detect polysemy for a specific token
result = detector.detect_polysemy(embeddings, token_idx=42, token_str="bank")

print(f"Is polysemous: {result.is_polysemous}")
print(f"Number of meanings: {result.num_meanings}")
print(f"Confidence: {result.confidence:.3f}")  # Uncalibrated score
print(f"Type: {result.polysemy_type}")  # homonym/contranym/multi-sense

# Analyze entire vocabulary
analysis = detector.analyze_vocabulary(embeddings, sample_size=10000)
print(f"Polysemy rate: {analysis.polysemy_rate:.2%}")
```

### Why Not Just Use WordNet?

The polysemy detector shows **HOW** your specific model represents ambiguity:

- **Model-Specific**: GPT vs LLaMA represent "bank" differently internally
- **Subword Tokens**: Detects ambiguity in BPE fragments like "##able"
- **Novel Words**: Works on COVID-19, YOLO, emoji sequences
- **Quantitative**: Measures separation quality, not just binary polysemy
- **Data-Driven**: What the model actually learned from training data

### Comparison with Robinson Test

```python
# Use BOTH methods for comprehensive analysis
robinson_result = RobinsonFiberBundleTest().test_point(embeddings, token_idx)
polysemy_result = PolysemyDetector().detect_polysemy(embeddings, token_idx)

if robinson_result.violates_hypothesis:
    print("Robinson: Geometric irregularity detected")
if polysemy_result.is_polysemous:
    print("Clustering: Multiple semantic clusters detected")

# They test different properties and may disagree!
```

## 📊 Interpretation Guide

### Volume Growth Violations

| Pattern | Meaning | Implication |
|---------|---------|-------------|
| Increasing slopes | Fiber bundle violation | Semantic instability |
| Sharp transitions | Regime change | Different scaling laws |
| Multiple discontinuities | Severe irregularity | Highly unpredictable |

### Ricci Curvature Values

| Range | Geometry | Model Behavior |
|-------|----------|----------------|
| κ > 0.1 | Positive (spherical) | Representations clustering |
| -0.1 < κ < 0.1 | Nearly flat | Stable, Euclidean-like |
| κ < -0.1 | Negative (hyperbolic) | Representations diverging |

### P-Values and Statistical Significance

- **p < 0.001**: Strong evidence of violation (Robinson's threshold)
- **p < 0.01**: Significant violation
- **p < 0.05**: Marginal violation
- **p ≥ 0.05**: No significant violation detected

## ⚡ Performance & Complexity

### Time Complexity

| Method | Complexity | For n=10,000 points |
|--------|------------|---------------------|
| Volume Growth | O(n² · log n) | ~1 second |
| CFAR Detector | O(n · w) | ~0.1 seconds |
| Ricci Curvature | O(n² · k · iter) | ~10 seconds |
| Dimension Estimation | O(n · k log n) | ~0.5 seconds |
| Polysemy Detection | O(n · k²) | ~0.3 seconds |

### Memory Requirements

- Volume growth: O(n²) for distance matrix
- Ricci curvature: O(n·k) with sampling
- Polysemy detection: O(k²) for k neighbors
- Can process ~100k embeddings on 16GB RAM

### Optimization Tips

```python
# For large datasets, use sampling
result = tester.test_point(
    embeddings,
    point_idx=0,
    n_samples=1000  # Sample subset for efficiency
)

# Use GPU acceleration for distance computations
embeddings = embeddings.cuda()  # If available

# Precompute distance matrix for multiple tests
distances = cdist(embeddings, embeddings)
result = tester.test_point(embeddings, 0, precomputed_distances=distances)
```

## 🎯 Practical Applications

### 1. Prompt Stability Assessment

```python
from manifold_violations import PromptRobustnessAnalyzer

analyzer = PromptRobustnessAnalyzer()
risk_score = analyzer.assess_prompt(
    "What is the capital of France?",
    model=model,
    n_paraphrases=10
)

if risk_score > 0.7:
    print("⚠️ High instability risk - consider rephrasing")
```

### 2. Model Comparison

```python
# Compare embedding stability across models
violations_gpt = test_model(gpt_embeddings)
violations_llama = test_model(llama_embeddings)

print(f"GPT violations: {violations_gpt.sum()}/{len(embeddings)}")
print(f"LLaMA violations: {violations_llama.sum()}/{len(embeddings)}")
```

### 3. Training Monitoring

```python
from manifold_violations import training_singularity_dynamics

# Track geometric stability during training
for epoch in range(num_epochs):
    train_epoch()

    result = tester.test_checkpoint(model)
    if result.increasing_slopes:
        print(f"Warning: Geometric instability at epoch {epoch}")
```

## 📖 Complete Example

```python
import torch
from manifold_violations import (
    RobinsonFiberBundleTest,
    compute_manifold_metrics_fixed,
    SingularityMapper,
    PolysemyDetector
)

# Load model and get embeddings
model = AutoModel.from_pretrained("gpt2")
tokens = tokenizer("The quick brown fox", return_tensors="pt")
embeddings = model.get_input_embeddings()(tokens.input_ids)

# Reshape to (n_tokens, embed_dim)
embeddings = embeddings.squeeze(0).detach().numpy()

# 1. Comprehensive manifold analysis
metrics = compute_manifold_metrics_fixed(
    embeddings,
    n_samples=20,
    compute_dimension=True,
    compute_curvature=True
)

print(f"Intrinsic dimension: {metrics['intrinsic_dimension']:.1f}")
print(f"Ricci curvature: {metrics['ricci_curvature_mean']:.3f}")

# 2. Test specific tokens for violations
tester = RobinsonFiberBundleTest()
for i, token in enumerate(tokens.input_ids[0]):
    result = tester.test_point(embeddings, i)
    if result.violates_hypothesis:
        word = tokenizer.decode([token])
        print(f"Token '{word}' violates at index {i}")

# 3. Detect polysemy
detector = PolysemyDetector()
for i, token in enumerate(tokens.input_ids[0]):
    poly_result = detector.detect_polysemy(embeddings, i)
    if poly_result.is_polysemous:
        word = tokenizer.decode([token])
        print(f"Token '{word}' is polysemous: {poly_result.num_meanings} meanings")

# 4. Map singularities
mapper = SingularityMapper()
singularities = mapper.map_singularities(embeddings)
print(f"Found {len(singularities)} singular points")
```

## 🔧 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/manifold_violations.git
cd manifold_violations

# Install requirements
pip install -r requirements.txt

# Run tests
python -m pytest tests/
```

### Requirements

- Python ≥ 3.8
- NumPy ≥ 1.19
- SciPy ≥ 1.5
- scikit-learn ≥ 1.0
- PyTorch ≥ 1.9 (for Ricci curvature)
- matplotlib (for visualizations)

## 📚 References

### Core Paper
- Robinson, M., Dey, S., & Chiang, T. (2025). "Token embeddings violate the manifold hypothesis." *arXiv preprint arXiv:2504.01002*

### Mathematical Foundations
- Ollivier, Y. (2007). "Ricci curvature of metric spaces." *Comptes Rendus Mathématique*, 345(11), 643-646
- Levina, E., & Bickel, P. (2005). "Maximum likelihood estimation of intrinsic dimension." *NIPS*
- Mann, H. B. (1945). "Nonparametric tests against trend." *Econometrica*, 245-259

### Related Work
- Facco, E., et al. (2017). "Estimating the intrinsic dimension of datasets by a minimal neighborhood information." *Scientific Reports*

## 🤝 Contributing

We welcome contributions! Key areas for improvement:

1. **Cross-Model Comparison**: Tools for comparing singularities across models
2. **Optimization**: GPU kernels for faster distance computations
3. **Visualizations**: Interactive plots for exploring violations
4. **WordNet Integration**: Ground truth validation for polysemy detection

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

This implementation is based on the groundbreaking work by Michael Robinson, Sourya Dey, and Tony Chiang. Their paper fundamentally challenges our understanding of how LLMs represent language and opens new avenues for improving model stability.

---

*"The manifold hypothesis fails for LLM embeddings. This isn't a bug—it's a fundamental property we must understand and address."*