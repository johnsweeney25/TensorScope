# Attribution: Robinson et al. vs Our Extensions

## Clear Attribution of Methods

### From Robinson et al. (2024) Paper - EXACT IMPLEMENTATIONS

These methods are directly from the paper:

1. **Volume Growth Test** (`robinson_fiber_bundle_test.py`)
   - Log-log analysis of neighborhood growth
   - Formula: log(N(r)) vs log(r)
   - Paper Section 3.2

2. **Three-Point Centered Differences**
   - Method: `_compute_centered_slopes()`
   - Slope computation: `slope[i] = (log(N[i+1]) - log(N[i-1])) / (log(r[i+1]) - log(r[i-1]))`
   - Paper Equation 3

3. **CFAR Detector**
   - Method: `_cfar_detector()`
   - Constant False Alarm Rate for discontinuity detection
   - Paper Section 3.3

4. **Holm-Bonferroni Correction**
   - Method: `_holm_bonferroni_correction()`
   - Multiple testing correction
   - Paper mentions using this standard statistical method

5. **Core Test Logic**
   - Checking for increasing slopes as violation indicator
   - Two spatial regimes concept (small/large radius)
   - Paper's main theoretical contribution

### OUR ORIGINAL EXTENSIONS - NOT IN PAPER

These are our contributions beyond the paper:

1. **Local Signal Dimension Computation** (`robinson_fiber_bundle_test.py`)
   - Method: `_compute_local_signal_dimension()`
   - PCA-based implementation with entropy weighting
   - Paper mentions the CONCEPT but provides NO computation method
   - Our contribution: Practical algorithm to compute it

2. **Polysemy Detection Module** (`polysemy_detector.py`)
   - ENTIRE MODULE is our creation
   - Uses DBSCAN/hierarchical clustering
   - Paper observation: "polysemy creates singularities"
   - Our contribution: Algorithm to detect polysemous tokens

3. **Singularity Mapper** (`singularity_mapper.py`)
   - ENTIRE MODULE is our creation
   - Combines multiple detection methods
   - Classifies singularity types
   - Comprehensive profiling system

4. **Prompt Robustness Analyzer** (`prompt_robustness_analyzer.py`)
   - ENTIRE MODULE is our creation
   - Practical application for real prompts
   - Risk assessment and alternatives
   - Not mentioned in paper

5. **Geometric Fiber Bundle Test** (`fiber_bundle_core.py`)
   - ENTIRE MODULE is our creation
   - Complementary geometric approach
   - Tests tangent alignment, curvature regularity
   - Different from Robinson's volume-based approach

6. **Integration Framework** (`manifold_fiber_integration.py`)
   - Our system for combining approaches
   - Decision logic for method selection
   - Not in paper

### Conceptual vs Implementation Distinctions

| Concept | Robinson Paper | Our Implementation |
|---------|---------------|-------------------|
| Local signal dimension | Mentions concept, no formula | PCA + entropy-based calculation |
| Polysemy detection | States polysemy causes issues | Clustering algorithm to find it |
| Singularity types | Mentions existence | Full classification system |
| Practical impact | Theoretical discussion | Quantitative metrics |
| Alternative tokens | Not discussed | Similarity-based suggestions |
| Cross-model risk | Mentions differences | Risk quantification method |

### Paper Concepts We Haven't Fully Implemented

1. **Exact Signal/Noise Dimension Split**
   - Paper describes TWO separate dimensions
   - We compute combined dimension
   - TODO: Implement proper separation

2. **Visualization Tools**
   - Paper emphasizes log-log plots
   - We compute but don't visualize
   - TODO: Add plotting functions

### Citation Requirements

When using this code, please cite BOTH:

1. **For core methodology**:
```bibtex
@article{robinson2024token,
  title={Token embeddings violate the manifold hypothesis},
  author={Robinson, Michael and Dey, Sourya and Chiang, Tony},
  journal={arXiv preprint arXiv:2504.01002},
  year={2024}
}
```

2. **For extensions and practical tools**:
```
[Your publication here - these are original contributions]
- Local signal dimension computation algorithm
- Polysemy detection via clustering
- Singularity mapping framework
- Prompt robustness analysis tools
```

### Academic Integrity Note

We have been careful to:
- Implement Robinson's methods exactly as described
- Clearly mark our extensions
- Not claim their ideas as ours
- Add substantial novel contributions

The Robinson paper provides the theoretical foundation and core test.
Our work extends this into a practical diagnostic toolkit.