# TensorScope: Training Dynamics & Interpretability Analysis Framework

**Research automation toolkit** implementing 70+ metrics for interpretability and training dynamics analysis to systematically understand model behavior and discover correlations between early checkpoints and final outcomes.

**Key Value**: Run comprehensive diagnostics across training checkpoints and identify which internal metrics correlate with success or failure, enabling data-driven insights about model behavior.

## What This Actually Does - Concrete Examples

```python
from GradientAnalysis import GradientAnalysis
from BombshellMetrics import BombshellMetrics

analyzer = GradientAnalysis()
metrics = BombshellMetrics()

# Example 1: Why is my model not training?
pathology = analyzer.compute_gradient_pathology(model, batch)
# → Reveals: "15 layers have vanishing gradients" (initialization problem)

# Example 2: Why does adding task B hurt task A performance?
conflict = analyzer.compute_gradient_conflict_pcgrad(model, task_a_batch, task_b_batch)
# → Reveals: "Tasks have 0.8 gradient conflict score" (they're fighting each other)

# Example 3: Which training samples are causing problems?
critical = metrics.find_critical_samples(model, train_data, checkpoints)
# → Reveals: "Samples 457, 892, 1203 have negative influence" (corrupted data)

# Example 4: Will this model generalize or just memorize?
attention_patterns = metrics.compute_attention_entropy(model, batch)
# → Reveals: "Attention entropy < 2.0" (model is memorizing, not learning patterns)

# Example 5: Can I safely merge these task-specific models?
conflicts = metrics.analyze_ties_conflicts({'task1': model1, 'task2': model2})
# → Reveals: "65% weight sign conflicts in layer 23" (models will interfere if merged)
```

**The key insight**: These metrics, when measured at 10% training, often correlate strongly with final performance. TensorScope finds these correlations automatically.

## Comprehensive Metric Collection

TensorScope implements 70+ internal state measurements from recent ML research, providing a unified interface for:

- **Training Health Indicators**: Gradient pathology detection, vanishing/exploding gradients, layer-wise health metrics
- **Task Interference Measures**: Gradient conflict analysis (PCGrad), task vector alignment, Fisher information overlap
- **Mechanistic Interpretability**: Induction heads (Olsson et al. 2022), QK-OV circuits, attention patterns
- **Training Attribution**: TracIn implementation for data influence, critical sample identification
- **Loss Landscape Analysis**: Hessian eigenvalues, mode connectivity, sharpness metrics
- **Information Flow Metrics**: Compression ratios, mutual information, channel capacity

Each metric serves as a potential predictor of training outcomes. Even imperfect measurements may reveal statistically significant correlations with model success or failure.

## Key Terminology

**Internal State Metrics**: Measurable properties of your model during training (gradient norms, attention patterns, weight statistics). Not just loss/accuracy.

**Training Health Indicators**: Metrics that reveal if training is proceeding normally (gradient flow, learning dynamics, convergence patterns).

**Predictive Patterns**: Statistical relationships where early metrics (at 10% training) correlate with final outcomes (accuracy, collapse, generalization).

**Metric-Outcome Analysis**: The process of discovering which internal metrics at checkpoint N predict performance at checkpoint M (where M > N).

**Early Warning Signs**: Metrics that show problems before they manifest in loss curves (e.g., gradient conflicts appearing before accuracy drops).

**Interpretability Through Prediction**: Understanding model behavior by identifying which internal states predict future performance, making the "black box" transparent.

## Information Theory Context

Many of TensorScope's metrics have deep connections to information theory:

**Integrated Gradients**: Measures information gain as we interpolate from a baseline (low-information) input to the actual input. The path integral quantifies how much each feature contributes to the model's decision.

**Mutual Information**: Quantifies how much information representations at layer N contain about the input (I(X;T_n)) and output (I(T_n;Y)). The Information Bottleneck principle suggests good representations compress input while preserving output-relevant information.

**Attention Entropy**: Higher entropy means more uniform attention (exploring many tokens), while lower entropy indicates focused attention (exploiting specific tokens). This explore/exploit tradeoff reveals whether models are learning patterns or memorizing.

**Compression Metrics**: Based on Kolmogorov complexity - simpler (more compressible) representations often generalize better. High compression ratios suggest the model has found efficient encodings.

**TracIn (Influence Functions)**: Information-theoretic view of training - which samples contribute most information to the model's knowledge? Negative influence suggests harmful/contradictory information.

**Fisher Information**: The curvature of the KL divergence between model distributions. High Fisher information for a parameter means that parameter strongly affects the model's output distribution.

## Core Value: Systematic Pattern Discovery

Instead of manually checking individual metrics, TensorScope enables:

1. **Comprehensive Metric Collection**: Run all implemented training health indicators across your model checkpoints
2. **Metric-Outcome Analysis** (upcoming): Automatically identify which internal measurements predict final performance
3. **Hypothesis Generation**: Discover unexpected relationships between early training states and eventual success/failure
4. **Reproducible Analysis**: Extensive test suite (500+ tests) ensures metric reliability

## What This Enables

With TensorScope's metric collection, researchers can explore questions like:
- Which internal metrics at step N predict final performance?
- What combinations of metrics indicate impending training collapse?
- Do certain metric patterns distinguish successful from failed runs?
- Which layers show early warning signs of catastrophic forgetting?

## Framework Capabilities

- **Unified API**: Consistent interface across all 70+ metrics
- **Memory Efficient**: Handles 70B+ parameter models with streaming implementations
- **Architecture Support**: GPT-2, LLaMA, Phi, GQA/MQA variants
- **Extensive Testing**: 500+ unit tests ensure metric correctness
- **Batched Analysis**: Run multiple metrics efficiently across checkpoints
- **Statistical Utilities**: Bootstrap CI, power analysis, multiple testing correction

## Memory-Efficient Fisher Information

TensorScope features an optimized Fisher Information implementation with significant memory reduction, making Fisher-based analysis feasible for 70B+ parameter models.

### The Innovation: Group-Level Reduction
Instead of storing Fisher per-parameter (O(n²) memory), we aggregate to group-level statistics:
- **Linear/Conv layers**: Per-channel importance (out_channels dimension)
- **Attention layers**: Per-head importance (num_heads dimension)
- **Result**: Dramatic memory reduction while preserving importance structure

### Key Features:
- **True Fisher**: Samples from model distribution (theoretically sound)
- **K-FAC approximation**: Block-diagonal for natural gradient computation
- **Capacity metrics**: Eigenvalue-based measures (effective rank, PAC-Bayes)
- **Production scale**: Handles 70B+ models with streaming implementations

### Usage:
```python
from BombshellMetrics import BombshellMetrics

# Initialize with efficient Fisher
metrics = BombshellMetrics(
    fisher_reduction='group',      # Group-level aggregation
    fisher_storage='cpu_fp16'      # Additional 2x memory saving
)

# Collect Fisher information
metrics.update_fisher_ema(model, batch, task='pretrain')
fisher = metrics.get_group_fisher('pretrain', bias_corrected=True)

# For percolation experiments (concentration C)
channel_importance = fisher['layer.weight|channel']
top_k = int(len(channel_importance) * concentration_C)
important_channels = torch.topk(channel_importance, top_k).indices
```

### Performance Comparison:
| Method | Memory (1.3B model) | Memory (70B model) | Reduction vs Diagonal |
|--------|---------------------|-------------------|----------------------|
| Full Fisher | ~6.8 TB | ~196 PB | - |
| Diagonal | ~5.2 GB | ~280 GB | Baseline |
| **Group-level** | **~68 MB** | **~3.7 GB** | **~75x** |

### What This Enables:
- Real-time Fisher tracking during training
- Catastrophic forgetting detection at scale
- Natural gradient optimization via K-FAC
- Concentration-controlled perturbations for research

Implementation: `fisher_collector.py` | Documentation: [FISHER_DOCUMENTATION.md](FISHER_DOCUMENTATION.md)

## The Research Problem

Training failures lack systematic diagnosis:

- Models collapse without clear warning signals
- Multi-task interference is hard to measure
- Internal dynamics remain opaque during training
- No systematic way to correlate internal states with outcomes

TensorScope provides the measurement infrastructure to discover which internal patterns correlate with these failures, enabling data-driven hypothesis generation about training dynamics.

## What This Is and Isn't

✅ **What TensorScope IS:**
- A comprehensive toolkit for measuring 70+ internal model behaviors during training
- A framework for discovering which early training metrics predict final performance
- A tested implementation of state-of-the-art interpretability methods
- A hypothesis generator that reveals unexpected relationships between internal states and outcomes

❌ **What TensorScope ISN'T:**
- A crystal ball (discovered patterns must be validated on new training runs)
- A replacement for TensorBoard/W&B (complements them with deeper analysis)
- Theoretically perfect (some metrics have limitations, especially for transformers)
- A magic fix (identifies problems but you still need to solve them)

## Testable Research Hypotheses

TensorScope enables testing hypotheses like:

- **Early Warning Signals**: Do any metrics at 10% training correlate with final task performance?
- **Failure Patterns**: Which metric combinations predict training collapse before loss diverges?
- **Multi-Task Interference**: Does gradient conflict score correlate with performance degradation?
- **Layer Attribution**: Which layers' metrics best predict overall model behavior?
- **Data Quality**: Do TracIn scores identify problematic training samples?

The framework provides the measurement infrastructure for systematic correlation studies between internal states and outcomes.

## Automated Metric-Outcome Analysis (Coming Soon)

The upcoming metric-outcome analysis module will automatically:

1. **Run Full Metric Suite**: Execute all 70+ internal state measurements across your checkpoints
2. **Statistical Relationship Analysis**: Compute correlations between early metrics and final performance (with p-values and effect sizes)
3. **Predictive Pattern Detection**: Identify which metric combinations at early training predict success/failure
4. **Actionable Report Generation**: Show which internal states to monitor and when to intervene

Example predictive analysis pipeline:
```python
# Future API
from tensorscope import MetricOutcomeAnalyzer

analyzer = MetricOutcomeAnalyzer()
results = analyzer.analyze_training_run(
    checkpoints=["ckpt_1k.pt", "ckpt_5k.pt", "ckpt_10k.pt"],
    outcomes={"final_mmlu": 0.45, "collapsed": True},
    model_type="llama"
)

# Discovers predictive patterns like:
# "Gradient conflict > 0.6 at step 5k predicts collapse (correlation r=0.78, p<0.01)"
# "Attention entropy < 2.0 in layers 20-30 at 10% training predicts low final MMLU (r=0.65, p<0.05)"
```

## Unified Analysis Framework

The `unified_model_analysis.py` provides a comprehensive analysis pipeline that runs all 70+ metrics and generates statistical reports:

```python
from unified_model_analysis import UnifiedModelAnalyzer, ModelSpec, UnifiedConfig

# Configure analysis
config = UnifiedConfig(
    metrics_to_compute=['gradient_alignment', 'fisher_trace', 'elasticity_score'],
    generate_report=True,  # Generate PDF report
    save_results=True,
    output_dir='./analysis_results'
)

# Initialize analyzer
analyzer = UnifiedModelAnalyzer(config)

# Analyze models
specs = [
    ModelSpec(id='model_1k', path='checkpoint_1000.pt'),
    ModelSpec(id='model_5k', path='checkpoint_5000.pt'),
    ModelSpec(id='model_10k', path='checkpoint_10000.pt')
]

results = analyzer.analyze_models(specs)

# Analyze training trajectory
from unified_model_analysis import CheckpointSpec

trajectory_results = analyzer.analyze_trajectory([
    CheckpointSpec(path='ckpt_1k.pt', iteration=1000),
    CheckpointSpec(path='ckpt_5k.pt', iteration=5000),
    CheckpointSpec(path='ckpt_10k.pt', iteration=10000)
])

# Results include phase transitions, convergence points, and training dynamics
print(f"Phase transitions detected: {len(trajectory_results.phase_transitions)}")
print(f"Training dynamics: {trajectory_results.training_dynamics_analysis}")
```

## Statistical Report Generation

TensorScope includes comprehensive report generation with LaTeX/PDF output:

```python
from statistical_report_generator import StatisticalReportGenerator, ReportConfig

# Configure report
config = ReportConfig(
    style='technical',  # or 'neurips', 'ieee', 'executive'
    include_trajectory_analysis=True,
    include_correlations=True,
    figure_format='pdf'
)

# Generate report from analysis results
generator = StatisticalReportGenerator(config)
generator.add_results('analysis_results.json')
pdf_path = generator.generate_report()

# Report includes:
# - Statistical analysis with p-values and effect sizes
# - Correlation matrices with significance testing
# - Phase transition detection
# - Training dynamics visualization
# - Automatic citations to relevant papers
```

### Report Templates

Pre-configured LaTeX templates for different audiences:
- **technical_report.tex**: Detailed technical analysis
- **neurips_template.tex**: Conference paper format
- **ieee_template.tex**: IEEE journal format
- **executive_template.tex**: Executive summary format

## Quick Problem Identification Workflow

For immediate problem detection without full analysis:

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

Each analysis method targets specific failure modes, working together to provide comprehensive training interpretability.

## Why TensorScope Is Different

**The Problem**: Your model collapsed during training. You have the loss curve, but no idea why it happened or how to prevent it next time.

### What Existing Tools Do vs. What You Actually Need

| Tool | What It Shows You | What You Actually Need to Know |
|------|------------------|--------------------------------|
| **TensorBoard/W&B** | "Loss increased at step 10K" | WHY did loss increase? What internal state changed? |
| **Captum** | "This token influenced this prediction" | Which training dynamics led to this behavior? |
| **TransformerLens** | "Layer 23 implements an induction head" | Does this induction head predict training success? |

### TensorScope's Unique Approach: Correlation Discovery

**Instead of just showing you metrics, TensorScope discovers which internal states predict training outcomes.**

Here's what TensorScope actually does:

1. **Runs 70+ internal state measurements** on your checkpoints automatically:
   - Gradient conflicts between tasks
   - Attention pattern stability
   - Loss landscape sharpness
   - Dead neuron emergence
   - Information bottlenecks
   - And 65+ more...

2. **Correlates everything with outcomes**:
   - Which metrics at step 1K predict failure at step 10K?
   - What combinations of metrics indicate impending collapse?
   - Which layers show problems first?

3. **Provides actionable insights**:
   ```
   DISCOVERED: Layer 67 gradient conflict > 0.8 at step 1K
              correlates with training collapse (r=0.82, p<0.001)
   PATTERN: Seen in 8/10 similar failures
   ACTION: Monitor layer 67 conflicts; consider gradient surgery if >0.8
   ```

**Concrete Example:**
- **Without TensorScope**: "Training failed. Try different hyperparameters?"
- **With TensorScope**: "Gradient conflicts in layers 64-70 exceeded 0.8 at step 1K in all 5 failed runs, but stayed below 0.3 in successful runs. Add task-specific adapters at these layers."

No other tool systematically discovers correlations between internal model states and training outcomes.

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

### 3. Metrics Collection (`BombshellMetrics.py`)
- Comprehensive suite of failure prediction metrics
- Memory-efficient implementations for large-scale analysis
- Unified interface for diverse metrics

### 4. Superposition Analysis (`SuperpositionMetrics.py`)
- Feature interference and vector overlap analysis
- Power law scaling analysis for neural scaling laws
- Superposition strength quantification
- Feature frequency distribution analysis
- Representation capacity estimation

#### Reimplemented Methods (Please Cite Both Original Papers and TensorScope)

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
| **Superposition Analysis** | Liu et al. 2025; Anthropic 2022 | `SuperpositionMetrics` module | Novel implementation based on papers |

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

*⚠️ **Signal Propagation Note**: The signal propagation metric was designed for feedforward networks. Its application to transformers is limited as it doesn't account for LayerNorm, skip connections, or attention mechanisms. Use only as a rough health indicator, not for rigorous dynamical analysis.

## Quick Start

```python
from ICLRMetrics import ICLRMetrics
from InformationTheoryMetrics import InformationTheoryMetrics
from established_analysis import EstablishedAnalysisMethods
from SuperpositionMetrics import SuperpositionMetrics
from BombshellMetrics import BombshellMetrics

# Efficient Fisher Information with group-level reduction
bombshell = BombshellMetrics(fisher_reduction='group', fisher_storage='cpu_fp16')
bombshell.update_fisher_ema(model, batch, task='pretrain')
fisher = bombshell.get_group_fisher('pretrain', bias_corrected=True)
print(f"Fisher computed with {len(fisher)} groups (dramatic memory reduction)")

# Signal propagation analysis
info_metrics = InformationTheoryMetrics()
result = info_metrics.compute_signal_propagation_dynamics(model, batch)  # Layer-wise norm ratios
print(f"Signal propagation regime: {result['regime']}")

# Gradient conflict detection
iclr_metrics = ICLRMetrics()
conflict = iclr_metrics.compute_gradient_conflict_pcgrad(model, task1_batch, task2_batch)
print(f"Task conflict score: {conflict['conflict_score']:.3f}")

# Superposition and scaling analysis
sup_metrics = SuperpositionMetrics()
interference = sup_metrics.compute_vector_interference(model.embedding.weight)
print(f"Mean feature overlap: {interference['mean_overlap']:.4f}")
strength = sup_metrics.compute_superposition_strength(model, test_batch)
print(f"Superposition ratio: {strength['superposition_ratio']:.2f}x")

# Attribution and sensitivity analysis (replaces representation_analysis)
analyzer = EstablishedAnalysisMethods(model, tokenizer)
importance = analyzer.analyze_token_importance(inputs, position_of_interest=5)
print(f"Most important token: {importance['attributions'].argmax()}")
```


## Project Structure

### Core Analysis Modules (Main Framework)
- **`fisher_collector.py`** (795 lines) - Revolutionary group-level Fisher implementation with 100,000x memory reduction
- **`InformationTheoryMetrics.py`** (5,967 lines) - Information flow analysis: mutual information, compression dynamics, signal propagation (limited for transformers), phase transitions, channel capacity
- **`BombshellMetrics.py`** (5,566 lines) - Training failure predictors with efficient Fisher: attention entropy, dead neurons, TracIn attribution, task vectors, TIES conflicts, intervention analysis
- **`ICLRMetrics.py`** (1,650 lines) - Loss landscape and optimization metrics: mode connectivity, sharpness, pruning sensitivity, Hessian eigenvalues
- **`mechanistic_analyzer.py`** (3,674 lines) - Circuit-level interpretability: induction heads, QK-OV circuits, attention flow patterns, activation patching, logit lens
- **`established_analysis.py`** (1,266 lines) - Attribution methods via Captum: integrated gradients, attention rollout, Jacobian analysis, layer-wise attribution

### Specialized Analysis Components
- **`GradientAnalysis.py`** (1,441 lines) - Training health monitoring: gradient pathology detection, task interference measurement, PCGrad conflicts, alignment tracking
- **`ModularityMetrics.py`** (1,151 lines) - Task interference analysis with efficient Fisher: task vectors, elasticity, TIES-merging conflicts
- **`LotteryTicketAnalysis.py`** (1,148 lines) - Sparse network analysis: lottery tickets, pruning sensitivity, magnitude tracking, iterative pruning
- **`manifold_adaptive_sampling.py`** - Intelligent sample size selection for manifold analysis based on data size and time constraints
- **`manifold_violations/`** - Implementation of Robinson et al. (2024) "Token Embeddings Violate the Manifold Hypothesis": fiber bundle tests, Ricci curvature analysis, prompt robustness testing

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
- `compute_layer_mutual_information`: Layer-wise mutual information with multiple estimators
- `compute_practical_compression_ratio`: Compression analysis with multiple codecs
- `compute_variational_ib_probe`: Train probe to analyze representation compressibility
- `compute_channel_capacity`: Information channel capacity analysis
- `compute_plasticity_index`: Measure model adaptability
- `compute_redundancy_synergy`: Information decomposition analysis

#### Mutual Information Estimation Methods

The framework provides four different MI estimation methods, each with different tradeoffs:

| Method | Best For | Pros | Cons |
|--------|----------|------|------|
| **InfoNCE** (default) | High-dimensional representations | Stable, works in high-D, good bias-variance tradeoff | Provides lower bound only |
| **k-NN** | Low-moderate dimensions (<50D) | Non-parametric, no training needed | Struggles in high dimensions |
| **MINE** | Complex dependencies | Can provide tight bounds, flexible | High variance, can be unstable |
| **Binning** | Very low dimensions (<10D) | Fast, simple, interpretable | Severe curse of dimensionality |

**Quick Usage:**
```python
# Recommended default for neural networks
results = metrics.compute_layer_mutual_information(
    model, batch,
    method='infonce',  # Choose: 'infonce', 'knn', 'mine', 'binning'
    temperature=0.07,  # For InfoNCE
    n_negatives=64     # For InfoNCE
)
print(f"Mean layer MI: {results['mean_layer_mi']:.3f} nats")
print(f"Information bottleneck: {results['min_layer_mi']:.3f} nats")
```

**Decision Guide:**
- **High dimensions (>100D)**: Use `'infonce'` - designed for high-D neural representations
- **Low dimensions (<50D)**: Use `'knn'` if you have >1000 samples
- **Need tight bounds**: Use `'mine'` but expect longer compute time
- **Very low dimensions (<10D)**: Can use `'binning'` for fast estimates

**Recent Improvements (2024):**
- Fixed k-NN implementation to properly handle multivariate MI
- Fixed binning to compute true joint MI instead of averaging marginals
- Improved MINE initialization to reduce overestimation
- Added binning method to main interface
- Removed incorrect Gaussian approximation fallback

**Known Limitations:**
- All methods provide estimates, not exact MI
- InfoNCE: Lower bound only (underestimates true MI)
- k-NN: Uses PCA for dimension reduction in high-D (changes what's measured)
- MINE: Can be unstable, prone to overestimation
- Binning: Exponentially bad in high dimensions

For detailed guidance, see [MUTUAL_INFORMATION_GUIDE.md](MUTUAL_INFORMATION_GUIDE.md)

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

### 12. **Superposition and Scaling Analysis (SuperpositionMetrics.py)**
- `compute_vector_interference`: Measure feature vector overlaps and interference
- `compute_feature_frequency_distribution`: Analyze feature frequency with power law fitting
- `compute_superposition_strength`: Quantify degree of feature superposition
- `analyze_dimensional_scaling`: Study loss scaling with model dimension
- `compute_feature_sparsity`: Measure sparsity of feature representations
- `fit_scaling_law`: Fit power laws to loss vs model size (L ∝ N^(-α))
- `compute_representation_capacity`: Estimate feature packing capacity
- `analyze_feature_emergence`: Track feature organization during training

### 13. **Manifold Analysis & Violations**

**manifold_adaptive_sampling.py:**
- `get_adaptive_manifold_samples`: Intelligent sample size selection based on time budget ("fast", "balanced", "thorough")
- Balances statistical reliability vs computation time
- Adaptive sampling for manifold curvature analysis

**manifold_violations/ (Robinson et al. 2024 Implementation):**
- `RobinsonFiberBundleTest`: Test fiber bundle hypothesis on embeddings
- `compute_ricci_curvature_debiased`: Ollivier-Ricci curvature for geometric analysis
- `fiber_bundle_core.py`: Core fiber bundle hypothesis tests
- `singularity_mapper.py`: Detect polysemy singularities in embedding space
- `prompt_robustness_analyzer.py`: Test prompt stability across perturbations
- `token_stability_analyzer.py`: Token-level stability analysis
- `tractable_manifold_curvature_fixed.py`: Efficient curvature computation
- `training_singularity_dynamics.py`: Track singularities during training

Key insight: Robinson et al. proved the manifold hypothesis fails for LLM embeddings, causing:
- Prompt instability (semantically equivalent prompts → different outputs)
- Model non-portability (prompts fail across models)
- Polysemy singularities (multiple meanings create embedding "black holes")

## 🔍 Gradient Analysis Functions - When to Use What

**Practical guide for choosing the right gradient analysis:**

### Decision Tree: Which Function Do I Need?

#### Checking if model is training properly
Use `compute_gradient_pathology`
- What it tells you: Are gradients vanishing or exploding
- When to use: First check when training fails
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

These measurements help identify common training pathologies systematically.

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

**Use as a rough health check only**, not for rigorous dynamical analysis.

### Quick Start
```python
from InformationTheoryMetrics import InformationTheoryMetrics
metrics = InformationTheoryMetrics()

# Basic measurement (health check only)
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
- Gradient pathology detection accuracy
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

## Numerical Stability & Reproducibility

### SVD Configuration for Academic Publications

TensorScope prioritizes numerical stability and reproducibility for academic publications. When running experiments for ICLR/NeurIPS submissions:

```python
# For publication-quality results with guaranteed reproducibility
from InformationTheoryMetrics import InformationTheoryMetrics

info_metrics = InformationTheoryMetrics(
    seed=42,           # Fixed seed for reproducibility
    svd_driver='gesvd' # QR-based SVD for guaranteed convergence
)
```

#### SVD Driver Options

- **`auto`** (default): PyTorch chooses automatically, may show convergence warnings
- **`gesvd`**: Most accurate, guaranteed convergence, **recommended for publications**
- **`gesvdj`**: Faster but may not converge for ill-conditioned matrices
- **`gesvda`**: Fastest but approximate, use only for large-scale experiments

#### Understanding SVD Warnings

If you see warnings like:
```
torch.linalg.svd: During SVD computation... failed to converge
```

**This is not an error** - PyTorch is automatically using a more accurate fallback method. For publication, use `svd_driver='gesvd'` to avoid warnings and ensure reproducibility.

See [NUMERICAL_STABILITY.md](NUMERICAL_STABILITY.md) for detailed documentation on numerical considerations.

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

- **Signal Propagation**: Inspired by Schoenholz et al. (2017). "Deep Information Propagation"
- **InfoNCE**: Oord et al. (2018). "Representation Learning with Contrastive Predictive Coding"
- **MINE**: Belghazi et al. (2018). "Mutual Information Neural Estimation"
- **k-NN MI**: Kraskov et al. (2004). "Estimating mutual information" (via sklearn)
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
  url={https://github.com/johnsweeney25/TensorScope}
}
```

## 🙏 Acknowledgments

TensorScope builds on foundational work in information theory, statistical physics, and deep learning. Special thanks to the open-source community for making this comprehensive analysis possible.
