# TensorScope: Automated Neural Network Diagnostic Discovery

A framework that runs 70+ diagnostic metrics to discover statistical correlations between internal model states and training outcomes. Built for researchers exploring what internal patterns predict model success or failure.

## Core Value: Systematic Pattern Discovery

Instead of manually checking individual metrics, TensorScope enables:

1. **Comprehensive Diagnostic Collection**: Run all implemented metrics across your training trajectory
2. **Correlation Discovery** (upcoming): Automatically identify which metrics correlate with outcomes
3. **Hypothesis Generation**: Discover unexpected patterns between internal states and performance
4. **Reproducible Analysis**: Extensive test suite (500+ tests) ensures metric reliability

## What This Enables

With TensorScope's metric collection, researchers can explore questions like:
- Which internal metrics at step N predict final performance?
- What combinations of metrics indicate impending training collapse?
- Do certain metric patterns distinguish successful from failed runs?
- Which layers show early warning signs of catastrophic forgetting?

## Comprehensive Metric Collection

TensorScope implements 70+ diagnostic metrics from recent ML research, providing a unified interface for:

- **Gradient Diagnostics**: Pathology detection, conflict analysis (PCGrad), layer-wise alignment
- **Mechanistic Interpretability**: Induction heads (Olsson et al. 2022), QK-OV circuits, attention patterns
- **Multi-Task Analysis**: Task vectors (Ilharco et al. 2023), TIES-merging conflicts, Fisher information
- **Training Attribution**: TracIn implementation for data influence, critical sample identification
- **Loss Landscape**: Hessian analysis, mode connectivity, sharpness metrics
- **Information Theory**: Compression ratios, mutual information, channel capacity

Each metric serves as a potential feature for discovering diagnostic patterns. Even imperfect metrics may reveal useful correlations in specific contexts.

## Framework Capabilities

- **Unified API**: Consistent interface across all 70+ metrics
- **Memory Efficient**: Handles 70B+ parameter models with streaming implementations
- **Architecture Support**: GPT-2, LLaMA, Phi, GQA/MQA variants
- **Extensive Testing**: 500+ unit tests ensure metric correctness
- **Batched Analysis**: Run multiple metrics efficiently across checkpoints
- **Statistical Utilities**: Bootstrap CI, power analysis, multiple testing correction

## The Research Problem

Training failures lack systematic diagnosis:

- Models collapse without clear warning signals
- Multi-task interference is hard to measure
- Internal dynamics remain opaque during training
- No systematic way to correlate internal states with outcomes

TensorScope provides the measurement infrastructure to discover which internal patterns correlate with these failures, enabling data-driven hypothesis generation about training dynamics.

## What This Is and Isn't

✅ **What TensorScope IS:**
- A comprehensive metric collection for exploratory analysis
- A framework for discovering correlations between metrics and outcomes
- A tested implementation of published methods
- A tool for generating hypotheses about model behavior

❌ **What TensorScope ISN'T:**
- A proven diagnostic system (correlations must be validated)
- A replacement for TensorBoard/W&B (use alongside)
- Theoretically perfect (some metrics have known limitations)
- A magic solution (requires systematic experimentation)

## Testable Research Hypotheses

TensorScope enables testing hypotheses like:

- **Early Warning Signals**: Do any metrics at 10% training correlate with final task performance?
- **Failure Patterns**: Which metric combinations predict training collapse before loss diverges?
- **Multi-Task Interference**: Does gradient conflict score correlate with performance degradation?
- **Layer Attribution**: Which layers' metrics best predict overall model behavior?
- **Data Quality**: Do TracIn scores identify problematic training samples?

The framework provides the measurement infrastructure for systematic correlation studies between internal states and outcomes.

## Automated Correlation Discovery (Coming Soon)

The upcoming correlation module will automatically:

1. **Run Full Diagnostic Suite**: Execute all 70+ metrics across your checkpoints
2. **Statistical Analysis**: Compute correlations, p-values, and effect sizes
3. **Pattern Detection**: Identify metric combinations that predict outcomes
4. **Report Generation**: Produce actionable insights with statistical confidence

Example discovery pipeline:
```python
# Future API
from tensorscope import CorrelationDiscovery

discovery = CorrelationDiscovery()
results = discovery.analyze_training_run(
    checkpoints=["ckpt_1k.pt", "ckpt_5k.pt", "ckpt_10k.pt"],
    outcomes={"final_mmlu": 0.45, "collapsed": True},
    model_type="llama"
)

# Discovers patterns like:
# "Gradient conflict > 0.6 at step 5k correlates with collapse (r=0.78, p<0.01)"
# "Attention entropy < 2.0 in layers 20-30 predicts low MMLU (r=0.65, p<0.05)"
```

## Current Diagnostic Workflow

While the correlation module is in development, use the metrics directly:

```python
from GradientAnalysis import GradientAnalysis
from BombshellMetrics import BombshellMetrics

grad_analyzer = GradientAnalysis()
metrics = BombshellMetrics()

# 1. Initial Health Check
pathology = grad_analyzer.compute_gradient_pathology(model, batch)
if pathology['num_vanishing'] > 10:
    # → Initialization problem detected

# 2. If Multi-Task Training
conflict = grad_analyzer.compute_gradient_conflict_pcgrad(model, task1_batch, task2_batch)
if conflict['conflict_score'] > 0.5:
    # → Tasks are interfering, need architectural changes

# 3. Track Evolution
trajectory = grad_analyzer.compute_gradient_alignment_trajectory(
    [model_ckpt1, model_ckpt2, model_ckpt3],
    test_batches
)
# → Analyze gradient stability across checkpoints

# 4. Deep Dive
layer_analysis = grad_analyzer.compute_layer_gradient_alignment(model, task1_batch, task2_batch)
# → Pinpoint which layers have conflicts
```

Each diagnostic method addresses specific failure modes, and they work together to provide comprehensive training analysis.

## How TensorScope Complements Existing Tools

| Tool | Primary Focus | Key Question Answered | When to Use |
|------|--------------|----------------------|-------------|
| **TensorBoard** | Metric visualization | "What are my metrics?" | Monitoring training progress |
| **Weights & Biases** | Experiment tracking | "Which run performed best?" | Comparing experiments |
| **Captum** | Model interpretation | "What did my model learn?" | Understanding model decisions |
| **TensorScope** | Diagnostic discovery | "Which internal patterns correlate with outcomes?" | Discovering failure patterns |

TensorScope fills a gap in the ML tooling ecosystem by providing systematic diagnostics for training failures, complementing rather than replacing existing tools.

## Components

The framework includes three main modules:

### 1. Mechanistic Analyzer (`mechanistic_analyzer.py`)
- Complete implementation of induction head detection following Olsson et al. (2022)
- QK-OV circuit analysis for understanding attention computation paths
- Activation patching for causal validation
- Support for modern architectures including GQA/MQA

### 2. Training Dynamics (`training_dynamics.py`)
- Signal propagation analysis adapted from Schoenholz et al. (2017)
- Gradient pathology detection across 100+ layer networks
- Multi-task conflict analysis using gradient alignment
- Critical sample identification via TracIn

### 3. Metrics Collection (`training_metrics.py`)*
- Comprehensive suite of training diagnostics
- Memory-efficient implementations for large-scale analysis
- Unified interface for diverse metrics

*Currently named BombshellMetrics.py - should be renamed

### Implementation Transparency: Reimplementations vs. Extensions

**~85% of the codebase reimplements existing methods** from published papers. We provide these implementations for convenience and integration, but **the original papers should be cited**, not this repository.

#### Reimplemented Methods (Original Papers Should Be Cited)

| Method | Original Paper | Our Implementation | Status |
|--------|---------------|-------------------|---------|
| **PCGrad** | Yu et al. 2020 | `compute_gradient_conflict_pcgrad()` | 90% original + layer-wise extension |
| **TracIn** | Pruthi et al. 2020 | `find_critical_samples()` | 90% original + memory optimization |
| **Integrated Gradients** | Sundararajan et al. 2017 | `compute_integrated_gradients()` | 95% original |
| **CKA** | Kornblith et al. 2019 | `compute_linear_cka_per_layer()` | 100% original |
| **Lottery Tickets** | Frankle & Carbin 2019 | `LotteryTicketAnalysis` module | 95% original |
| **Task Arithmetic** | Ilharco et al. 2023 | `extract_task_vectors()` | 85% original |
| **TIES-Merging** | Yadav et al. 2023 | `analyze_ties_conflicts()` | 90% original |
| **Induction Heads** | Olsson et al. 2022 | `compute_induction_head_strength()` | 95% original |
| **Loss Landscapes** | Li et al. 2018 | `compute_loss_landscape_2d()` | 100% original |
| **SAM/Sharpness** | Foret et al. 2021 | `compute_sam_sharpness()` | 100% original |
| **Attention Rollout** | Abnar & Zuidema 2020 | `analyze_attention_flow()` | 100% original |

#### Our Extensions Beyond Original Papers (~15% of codebase)

| Extension | What We Added | Value |
|-----------|--------------|-------|
| **Layer-wise Gradient Conflicts** | PCGrad broken down per layer | Identifies specific layers for intervention |
| **Cross-checkpoint Trajectory** | Temporal gradient alignment tracking | Reveals training stability patterns |
| **Memory-Efficient TracIn** | Chunked processing for 70B+ models | Makes TracIn feasible at scale |
| **Instruction Template Sensitivity** | Teacher-forcing protocol with security | Rigorous format brittleness quantification |
| **Enhanced Intervention Vectors** | Statistical significance testing | Identifies meaningful weight changes |
| **QK-OV Circuit Pairing** | Coupled circuit analysis | Deeper mechanistic understanding |
| **Streaming OV→U** | Windowed processing for long sequences | Handles 10K+ token sequences |
| **Bootstrap CI for Metrics** | Statistical confidence intervals | Rigorous uncertainty quantification |
| **TIES Conflict Topology** | Layer-wise conflict distribution | Spatial understanding of interference |
| **Unified Training Dynamics** | Integrated phase transition detection | Holistic training evolution view |

#### Unique Ablation & Intervention Capabilities

These capabilities enable causal analysis and model surgery that isn't available in standard libraries:

| Capability | What It Enables | Implementation | Location | Not Available In |
|------------|----------------|----------------|----------|------------------|
| **Selective Head Ablation** | Disable specific attention heads for causal analysis | `freeze_attention_heads()` with zero/soft/identity modes | experimental_methods.py* | TransformerLens, Captum |
| **Null Space Projection** | Protect existing capabilities while updating model | `compute_null_space_projection()` with Fisher weighting | training_metrics.py* | Standard ML libraries |
| **Intervention Vector Analysis** | Find corrective directions between model states | `find_intervention_vectors_enhanced()` with statistical significance | training_metrics.py* | Existing interpretability tools |
| **Activation Patching** | Validate mechanistic hypotheses via causal intervention | `validate_with_activation_patching()` integrated with circuits | mechanistic_analyzer.py | Limited in other tools |
| **Causal Importance Analysis** | Rigorous importance scoring via systematic ablation | `compute_causal_necessity()` with bootstrap CI | information_metrics.py* | Beyond standard attribution |

*Note: Current filenames (FutureStudies.py → experimental_methods.py, BombshellMetrics.py → training_metrics.py, InformationTheoryMetrics.py → information_metrics.py) should be renamed for professionalism

These intervention methods enable researchers to:
- **Test causal hypotheses** about which components matter for specific behaviors
- **Protect capabilities** during continued training or fine-tuning
- **Repair models** by identifying and applying corrective weight updates
- **Validate mechanistic theories** through targeted interventions

⚠️ **Note**: Intervention methods in `FutureStudies.py` are experimental and modify model behavior. Always work with model copies, not originals.

### Method Origins & Transformer Applications

### Key Features
- Scales to 100+ layers (most tools handle <20)
- Unified interface for multiple analysis methods
- Optimized for transformers

*⚠️ **Signal Propagation Note**: The signal propagation metric was designed for feedforward networks. Its application to transformers is limited as it doesn't account for LayerNorm, skip connections, or attention mechanisms. Use as a rough diagnostic tool only, not for rigorous dynamical analysis.

## Quick Start

```python
from ICLRMetrics import ICLRMetrics
from InformationTheoryMetrics import InformationTheoryMetrics
from established_analysis import EstablishedAnalysisMethods

# Signal propagation analysis
info_metrics = InformationTheoryMetrics()
result = info_metrics.compute_signal_propagation_dynamics(model, batch)  # Layer-wise norm ratios
print(f"Signal propagation regime: {result['regime']}")

# Gradient conflict detection
iclr_metrics = ICLRMetrics()
conflict = iclr_metrics.compute_gradient_conflict_pcgrad(model, task1_batch, task2_batch)
print(f"Task conflict score: {conflict['conflict_score']:.3f}")

# Attribution and sensitivity analysis (replaces representation_analysis)
analyzer = EstablishedAnalysisMethods(model, tokenizer)
importance = analyzer.analyze_token_importance(inputs, position_of_interest=5)
print(f"Most important token: {importance['attributions'].argmax()}")
```


## Project Structure

### Core Analysis Modules (Main Framework)
- **`InformationTheoryMetrics.py`** (5,967 lines) - Comprehensive information-theoretic analysis: mutual information, compression, signal propagation (with limitations), phase transitions, channel capacity
- **`BombshellMetrics.py`** (5,566 lines) - Advanced training diagnostics: attention analysis, neuron importance, TracIn implementation, task vectors, TIES-merging analysis, intervention vectors
- **`ICLRMetrics.py`** (1,650 lines) - ICLR hypothesis testing: loss landscapes, mode connectivity, pruning analysis, attribution methods, Hessian analysis
- **`mechanistic_analyzer.py`** (3,674 lines) - Mechanistic interpretability: induction heads, QK-OV circuits, attention flow, activation patching, logit lens
- **`established_analysis.py`** (1,266 lines) - Attribution methods via Captum: integrated gradients, attention rollout, Jacobian analysis, layer-wise attribution

### Specialized Analysis Components
- **`GradientAnalysis.py`** (1,441 lines) - Consolidated gradient diagnostics: pathology detection, multi-task conflicts, PCGrad, alignment tracking
- **`ModularityMetrics.py`** (1,151 lines) - Task interference analysis: Fisher information, task vectors, elasticity, TIES-merging conflicts
- **`LotteryTicketAnalysis.py`** (1,148 lines) - Sparse network analysis: lottery tickets, pruning sensitivity, magnitude tracking, iterative pruning

### Catastrophic Forgetting Analysis
- **`CFA-2-complete.py`** (1,453 lines) - Complete catastrophic forgetting analyzer: integrates all metrics for systematic forgetting analysis across checkpoints
- **`cfa_statistical_utils.py`** (380 lines) - Statistical utilities: power analysis, bootstrap CI, multiple testing correction, Wasserstein distance

### Experimental & Future Work
- **`FutureStudies.py`** (374 lines) - ⚠️ EXPERIMENTAL/ALPHA: Head freezing interventions, causal analysis methods (use with extreme caution)

### Advanced Analysis Tools
- **`qkov_patching_dynamics.py`** - QK-OV circuit analysis with causal validation
- **`streaming_ovu.py`** - Memory-efficient computation for 10K+ token sequences
- **`base_model_comparison.py`** - Systematic model architecture comparison
- **`fishing_dashboard.py`** - Exploratory dashboard for anomaly detection
- **`cfa_statistical_utils.py`** - Statistical tools for rigorous analysis (bootstrap CI, power analysis)

### Example Scripts (examples/ directory)
- **`instruction_template_example.py`** - Demo usage of instruction template sensitivity analysis
- **`sanity_check_plots.py`** - Visualization examples for metric validation
- **`analyze_cfa2_simple.py`** - Example of analyzing CFA-2 metric execution coverage

### Validation & Testing

**Extensive Test Coverage (500+ tests):**
- **`test_suite.py`** - Main test suite with 100+ unit tests across all modules
- **`test_iclr_metrics_comprehensive.py`** - Validates loss landscape, gradient analysis
- **`test_mechanistic_analyzer_comprehensive.py`** - Tests induction heads, circuits
- **`test_information_theory.py`** - Validates MI estimation, compression metrics
- **`test_bombshell_smoke.py`** - Smoke tests for large-scale metrics
- **`audit_summary.py`** - Automated code quality and coverage reporting

Every metric is tested for:
- Correctness against known inputs/outputs
- Numerical stability
- Memory efficiency
- Architecture compatibility (GPT-2, LLaMA, etc.)

## Available Analysis Methods

Note: Most methods are implementations of published work. See attribution table for sources.
Methods are organized by their actual file location for easy reference.

### 1. Information Dynamics & Attribution Analysis

**InformationTheoryMetrics.py:**
- `compute_information_flow`: Layer-wise mutual information with multiple estimators
- `compute_practical_compression_ratio`: Compression analysis with multiple codecs
- `compute_variational_ib_probe`: Train probe to analyze representation compressibility
- `compute_channel_capacity`: Information channel capacity analysis
- `compute_plasticity_index`: Measure model adaptability
- `compute_redundancy_synergy`: Information decomposition analysis

**EstablishedAnalysisMethods.py (Captum-based):**
- `analyze_token_importance`: Token importance via Integrated Gradients
- `analyze_attention_flow`: Attention flow via Attention Rollout
- `compute_position_jacobian`: Exact position sensitivities via Jacobian
- `layer_wise_attribution`: Layer-wise attribution analysis
- `comprehensive_analysis`: Combined analysis pipeline

### 2. Signal Dynamics & Stability (InformationTheoryMetrics.py)
- `compute_signal_propagation_dynamics`: Layer-wise norm ratio analysis (⚠️ limited for transformers)
- `test_signal_propagation_stability`: Robustness under input perturbations
- `analyze_training_dynamics`: Grokking, phase transitions, changepoints
- `compute_alignment_fragility`: Measure model stability

### 3. Geometric & Topological Analysis
**ICLRMetrics.py:**
- `compute_loss_barrier`: Height of barrier between solutions
- `compute_loss_landscape_2d`: Visualize loss surface projections
- `compute_mode_connectivity`: Solution path analysis
- `compute_hessian_eigenvalues`: Local curvature analysis (standard method)
- `compute_hessian_eigenvalues_lanczos`: Efficient Hessian analysis for large models
- `compute_pruning_sensitivity`: Analyze model robustness to pruning
- `compute_integrated_gradients`: Attribution via integrated gradients
- `compute_attention_attribution`: Attention-based attribution

**ModularityMetrics.py:**
- `compute_effective_rank`: Intrinsic dimensionality
- `compute_sam_sharpness`: Sharpness-Aware Minimization metrics

### 4. Modularity & Representation Analysis (ModularityMetrics.py)
- `compute_block_cka_gap`: Block-wise CKA similarity for modularity detection
- `compute_linear_cka_per_layer`: Layer-wise representation similarity (CKA)
- `compute_fisher_weighted_damage`: Task interference via Fisher information
- `compute_fisher_damage_with_asymmetry`: Asymmetric task damage analysis
- `compute_subspace_distance`: Representation subspace comparison (CKA/Procrustes)
- `compute_sam_sharpness`: Sharpness-Aware Minimization metrics
- `compute_weight_space_distance`: Parameter space distance between models
- `compute_function_space_distance`: Functional behavior distance (JSD/MSE)
- `compute_elasticity`: Model recovery capacity after perturbation
- `compute_effective_rank`: Compute intrinsic dimensionality
- `compute_full_effective_rank`: Full effective rank analysis
- `update_fisher_ema`: Update Fisher information with EMA

### 5. **Optimization Diagnostics**

**Gradient Analysis (all in `GradientAnalysis.py`):**
- `compute_gradient_pathology`: Vanishing/exploding gradient detection
- `compute_gradient_conflict_pcgrad`: Multi-task interference patterns (PCGrad implementation)
- `compute_gradient_alignment_trajectory`: Gradient alignment across batches/checkpoints
- `compute_layer_gradient_alignment`: Per-layer gradient alignment analysis
- `compute_gradient_conflict_pair`: Pairwise gradient conflict computation
- `compute_gradient_conflict_matrix_multi`: Full conflict matrix for multiple tasks
- `compute_raw_gradient_conflict`: Memory-efficient gradient conflict with resampling

**Other Optimization Metrics:**
- `compute_hessian_eigenspectrum`: Training difficulty analysis
- `compute_fisher_information`: Natural gradient geometry
- `compute_sam_sharpness`: Sharpness-aware metrics

### 6. **Lottery Ticket & Pruning Analysis (LotteryTicketAnalysis.py)**
- `compute_ticket_overlap`: Measure overlap between lottery tickets
- `compute_pruning_robustness`: Test model robustness to pruning
- `compute_gradient_importance`: Gradient-based importance scoring
- `compute_iterative_magnitude_pruning`: Iterative magnitude-based pruning
- `compute_early_bird_tickets`: Find early-emerging lottery tickets
- `compute_layerwise_magnitude_ticket`: Layer-wise pruning analysis
- `compute_with_recovery`: Pruning with recovery analysis

### 7. **Attention & Transformer Analysis (BombshellMetrics.py)**
- `compute_attention_entropy`: Attention distribution analysis
- `compute_attention_drift`: Track attention pattern changes over time
- `compute_attention_concentration`: Measure attention focus

**Note**: For induction heads and mechanistic analysis, see mechanistic_analyzer.py (Section 11)

### 8. **Dead Neuron & Critical Sample Analysis (BombshellMetrics.py)**
- `compute_dead_neurons`: Identify inactive neurons across layers
- `compute_neuron_importance`: Rank neurons by task relevance
- `find_critical_samples`: Training data attribution via TracIn algorithm
  - Full TracIn implementation (Pruthi et al., 2020) with checkpoint support
  - Memory-efficient mode for large models (70B+)
  - Dynamic architecture detection (no hard-coded assumptions)
  - Robust checkpoint loading (HuggingFace, PyTorch, SafeTensors)
- `compute_representation_shift`: Track representation changes over time

### 9. **Task Interference & Forgetting Analysis (BombshellMetrics.py)**
- `extract_task_vectors`: Extract task-specific weight directions
- `analyze_ties_conflicts`: Analyze sign conflicts and redundancy (TIES-Merging)
- `find_intervention_vectors`: Generate protective vectors against forgetting
- `find_intervention_vectors_enhanced`: Advanced intervention vector generation
- `compute_retention_metrics`: Measure capability retention after training
- `compute_null_space_projection`: Project updates to preserve capabilities

### 10. **Experimental Methods (FutureStudies.py)**
⚠️ **WARNING: EXPERIMENTAL/ALPHA CODE - Use with extreme caution**
- `freeze_attention_heads`: Surgically disable specific attention heads for causal analysis
- Head intervention hooks for mechanistic studies
- **Note**: These methods modify model behavior and require careful handling

### 11. **Mechanistic Interpretability (mechanistic_analyzer.py)**
- `compute_induction_head_strength`: Detect and measure induction heads
- `compute_induction_ov_contribution`: Analyze OV circuit contributions
- `compute_qk_ov_pairing`: QK-OV circuit interaction analysis
- `compute_memory_efficient_ovu`: Memory-efficient OV→U computation
- `analyze_training_dynamics`: Track mechanistic changes during training
- `compute_attention_head_specialization`: Identify specialized attention patterns
- `analyze_attention_flow_patterns`: Trace information flow through attention
- `compute_logit_lens`: Intermediate layer predictions
- `compute_output_entropy`: Output distribution analysis
- `analyze_complete`: Run full mechanistic analysis pipeline

## 🔍 Gradient Analysis Functions - When to Use What

**Practical guide for choosing the right gradient analysis:**

### Decision Tree: Which Function Do I Need?

#### Checking if model is training properly
Use `compute_gradient_pathology`
- What it tells you: Are gradients vanishing or exploding
- When to use: First diagnostic when training fails
- Warning signs: `num_vanishing > 10` or `num_exploding > 5` layers
- Example scenario: Model loss stuck, accuracy not improving

#### Diagnosing multi-task training issues
Use `compute_gradient_conflict_pcgrad`
- What it tells you: Whether tasks have conflicting gradients
- When to use: Multi-task learning, instruction tuning, RLHF
- Warning signs: `conflict_score > 0.5` indicates opposing gradients
- Example scenario: Math performance drops when adding coding tasks

#### Tracking training evolution
Use `compute_gradient_alignment_trajectory`
- What it tells you: Gradient consistency across epochs
- When to use: Analyzing training stability, comparing checkpoints
- Warning signs: High variance in alignment indicates instability
- Example scenario: Comparing models at epochs 1, 5, 10

#### Locating layer-specific conflicts
Use `compute_layer_gradient_alignment`
- What it tells you: Which layers have gradient conflicts
- When to use: Debugging multi-task models, finding bottlenecks
- Warning signs: Sudden alignment drops at specific layers
- Example scenario: Tasks align until layer 67, then diverge

### Quick Reference Table

| Problem | Function to Use | Key Metric | Danger Zone |
|---------|----------------|------------|-------------|
| Training not converging | `compute_gradient_pathology` | `num_vanishing` | > 10 layers |
| Multi-task performance issues | `compute_gradient_conflict_pcgrad` | `conflict_score` | > 0.5 |
| Training instability | `compute_gradient_alignment_trajectory` | `alignment_variance` | > 0.3 |
| Need layer-specific debugging | `compute_layer_gradient_alignment` | `min_alignment` | < -0.2 |

These diagnostics help identify common training pathologies systematically.

### Real-World Usage Examples

```python
# Import the consolidated gradient analysis module
from GradientAnalysis import GradientAnalysis
grad_analyzer = GradientAnalysis()

# SCENARIO 1: "My model won't train"
pathology = grad_analyzer.compute_gradient_pathology(model, batch)
if pathology['num_vanishing'] > 10:
    print("Gradients vanishing - try different initialization or normalization")
if pathology['num_exploding'] > 5:
    print("Gradients exploding - reduce learning rate or add gradient clipping")

# SCENARIO 2: "Adding a new task hurt my existing tasks"
conflict = grad_analyzer.compute_gradient_conflict_pcgrad(model, math_batch, code_batch)
if conflict['conflict_score'] > 0.5:
    print(f"Tasks conflicting badly ({conflict['conflict_score']:.2f})")
    print("Consider: Separate task-specific layers or gradient surgery")

# SCENARIO 3: "Is my training stable?"
checkpoints = [model_epoch1, model_epoch5, model_epoch10]
trajectory = grad_analyzer.compute_gradient_alignment_trajectory(checkpoints, test_batches)
if trajectory['cross_checkpoint']['alignment_variance'] > 0.3:
    print("Training unstable - gradients changing direction too much")
    print("Consider: Lower learning rate or different optimizer")

# SCENARIO 4: "Which layers are causing problems?"
alignment = grad_analyzer.compute_layer_gradient_alignment(model, task1_batch, task2_batch)
problem_layers = [i for i, a in enumerate(alignment['per_layer']) if a < -0.2]
print(f"Gradient conflicts at layers: {problem_layers}")
print("Consider: Adding task-specific adapters at these layers")
```

## Signal Propagation Dynamics Analysis

⚠️ **Important Limitations**: This metric was designed for feedforward networks and has limited applicability to transformers. It measures simple norm ratios between layers but doesn't account for:
- LayerNorm (which normalizes activations)
- Skip connections (h + f(h) architecture)
- Attention mechanisms (dynamic connectivity)
- The thresholds (0.9/1.1) are arbitrary and not theoretically grounded for transformers

**Use as a rough diagnostic tool only**, not for rigorous dynamical analysis.

### Quick Start
```python
from InformationTheoryMetrics import InformationTheoryMetrics
metrics = InformationTheoryMetrics()

# Basic measurement (diagnostic only)
result = metrics.compute_signal_propagation_dynamics(model, batch)
print(f"Block gain: {result['block_gain']:.3f}")  # Simple ||layer_n||/||layer_{n-1}|| ratio
print(f"Regime: {result['regime']}")  # Based on arbitrary thresholds

# Stability testing (more useful as robustness metric)
stability = metrics.test_signal_propagation_stability(model, batch)
print(f"Stability score: {stability['stability_score']:.3f}")
```

### What This Actually Tells You
- **Extreme values** (<0.5 or >2.0): Possible initialization or normalization issues
- **Stability variance**: Relative model fragility under perturbations
- **NOT edge of chaos**: Despite the naming, this is not true dynamical analysis

### What This Can Be Used For
1. **Rough initialization check**: Detect extreme norm growth/decay
2. **Relative robustness comparison**: Compare models under perturbations
3. **NOT for rigorous analysis**: Do not use for publication-quality dynamics analysis

## Important Notes

### Attribution
- Most methods are implementations of published work (see table above)
- Use original paper citations, not this repository
- Some metrics are experimental combinations clearly marked

### Known Limitations
- Signal propagation: Designed for feedforward nets, limited value for transformers
- Some thresholds are empirical, not theoretically grounded
- Correlations discovered need independent validation
- FutureStudies.py contains experimental code that modifies models

### Validated Components
Through our 500+ test suite, we've validated:
- Gradient diagnostic accuracy
- TracIn implementation correctness
- Task vector extraction reliability
- Statistical utilities (bootstrap, power analysis)
- Memory efficiency for 70B+ models

## Installation

```bash
# Clone TensorScope
git clone https://github.com/[username]/TensorScope.git
cd TensorScope

# Install dependencies
pip install -r requirements.txt

# Optional: Install Captum for attribution methods
pip install captum  # For EstablishedAnalysisMethods
```

## Key Analysis Examples

### TracIn: Training Data Attribution

Identifies which training samples contribute most to model predictions:

```python
from BombshellMetrics import BombshellMetrics
metrics = BombshellMetrics()

# Memory-efficient TracIn for large models
results = metrics.find_critical_samples(
    model=base_model,
    dataset=train_samples[:1000],
    checkpoint_paths=['ckpt_1.pt', 'ckpt_2.pt', 'ckpt_3.pt'],
    test_sample=test_batch,
    memory_efficient=True,  # For 70B+ models
    full_tracin=True
)
```

### Task Vector Analysis

Extract and analyze task-specific directions:

```python
# Extract task vectors
task_vectors = metrics.extract_task_vectors(
    base_model=base_model,
    task_models={'math': math_model, 'code': code_model}
)

# Analyze conflicts using TIES-Merging insights
conflicts = metrics.analyze_ties_conflicts(task_vectors)
print(f"Sign conflict rate: {conflicts['sign_conflict_rate']:.2%}")
print(f"Redundancy rate: {conflicts['redundancy_rate']:.2%}")
```

### Catastrophic Forgetting Analysis (CFA-2-complete.py)

Complete framework for analyzing catastrophic forgetting:
- Integrates all metrics for systematic forgetting analysis
- Cross-checkpoint trajectory analysis
- Statistical utilities for rigorous testing
- Power analysis and bootstrap confidence intervals

## Understanding Key Metrics

| Metric | Typical Range | What It Actually Measures | Reliability |
|--------|--------------|--------------------------|-------------|
| Signal Propagation | [0.95, 1.05]* | Simple norm ratios | ⚠️ Low for transformers |
| Gradient Conflict | < 0.3 | Multi-task interference | ✅ Good |
| Gradient Pathology | < 10 vanishing | Layer-wise gradient health | ✅ Good |
| Attention Entropy | [2.0, 4.0] bits | Attention distribution spread | ✅ Good |
| TracIn Score | Context-dependent | Training sample influence | ✅ Good |
| Sign Conflict Rate | < 0.2 | Task vector disagreement | ✅ Good |
| Redundancy Rate | > 0.3 | Prunable parameters | ✅ Good |
| Fisher Damage | < 0.3 | Task interference estimate | ✅ Good |
| Lottery Ticket Sparsity | > 50% | Pruning potential | ✅ Good |

*Arbitrary thresholds, not theoretically grounded for transformers

## Original Papers & Citations

**Please cite the original papers, not this repository:**

- **Signal Propagation**: Inspired by Schoenholz et al. (2017). "Deep Information Propagation"
- **Information Flow**: Various MI estimation techniques (InfoNCE, MINE)  
- **Integrated Gradients**: Sundararajan et al. (2017). "Axiomatic Attribution for Deep Networks"
- **Gradient Conflict/PCGrad**: Yu et al. (2020). "Gradient Surgery for Multi-Task Learning"
- **Lottery Tickets**: Frankle & Carbin (2019). "The Lottery Ticket Hypothesis"
- **Neural Scaling Laws**: Kaplan et al. (2020). "Scaling Laws for Neural Language Models"
- **Grokking**: Power et al. (2022). "Grokking: Generalization Beyond Overfitting"
- **SAM**: Foret et al. (2021). "Sharpness-Aware Minimization"
- **TracIn**: Pruthi et al. (2020). "Estimating Training Data Influence by Tracing Gradient Descent"
- **Task Arithmetic**: Ilharco et al. (2023). "Editing Models with Task Arithmetic"
- **TIES-Merging**: Yadav et al. (2023). "Resolving Interference When Merging Models"

## Contributing

We welcome contributions to TensorScope! Key areas:
- Architecture-specific adapters (BERT, ViT, etc.)
- New correlation discoveries
- Efficiency improvements
- Visualization tools
- Documentation and tutorials

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

MIT License - See LICENSE file

## 📖 Citation

```bibtex
@software{tensorscope2024,
  title={TensorScope: Complete Neural Network Analysis Suite for Dynamics, Information Flow, Geometry & Emergent Phenomena},
  author={[Authors]},
  year={2024},
  url={https://github.com/[username]/TensorScope}
}
```

## 🙏 Acknowledgments

TensorScope builds on foundational work in information theory, statistical physics, and deep learning. Special thanks to the open-source community for making this comprehensive analysis possible.
