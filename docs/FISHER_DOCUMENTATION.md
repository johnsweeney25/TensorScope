# Fisher Information Matrix - Complete Documentation

## Overview

Production-ready Fisher Information Matrix implementation for parameter importance analysis in large language models. Includes Welford's algorithm for numerical stability, behavioral head categorization for circuit-level analysis, and integration with QKOV interference metrics.

## Table of Contents
1. [Quick Start](#quick-start)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Fisher Methods Comparison](#fisher-methods-comparison)
4. [7-Phase Analysis Pipeline](#7-phase-analysis-pipeline)
5. [Behavioral Head Categorization](#behavioral-head-categorization)
6. [Configuration Guide](#configuration-guide)
7. [Common Issues and Solutions](#common-issues-and-solutions)
8. [QKOV Integration](#qkov-integration)

---

## Quick Start

### Basic Usage

```python
from fisher.core.fisher_collector import FisherCollector

# Initialize Fisher collector
fisher_collector = FisherCollector(
    reduction='group',  # Group-level parameter reduction
    storage='cpu_fp16'  # Memory-efficient storage
)

# Collect Fisher information
fisher_collector.collect_fisher(model, batch, task='math')

# Get importance scores
importance = fisher_collector.get_group_fisher('math')
```

### Production Configuration

```python
from fisher.core.fisher_collector import FisherCollector
from mechanistic.mechanistic_analyzer_core import MechanisticAnalyzer

# Enhanced Fisher with behavioral head categorization
fisher_collector = FisherCollector(
    reduction='group',
    storage='cpu_fp16',
    ema_decay=0.99,
    enable_cross_task_analysis=True,
    gradient_memory_mb=50
)

# Optional: Add behavioral head analysis
mech_analyzer = MechanisticAnalyzer()
fisher_collector.set_mechanistic_analyzer(mech_analyzer)

# Collect Fisher for multiple tasks
fisher_collector.collect_fisher(model, math_batch, task='math')
fisher_collector.collect_fisher(model, code_batch, task='code')

# Compare tasks
comparison = fisher_collector.compare_task_fisher('math', 'code')
```

### Unified Analysis (Recommended)

```python
from unified_model_analysis import analyze_models, UnifiedConfig

# Complete Fisher analysis with all features
config = UnifiedConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    compute_fisher=True,
    compute_qkov=True,  # Enable QKOV interference
    
    # Fisher settings
    fisher_batch_size=128,
    fisher_n_samples=1000,
    compute_fisher_eigenvalues=True,
    enable_qkov_interference=True,
    enable_cross_task_conflicts=True,
    
    seed=42
)

results = analyze_models([model_path], config)
```

---

## Theoretical Foundation

### Empirical Fisher Information Matrix

The Fisher Information Matrix quantifies parameter importance:

```
F = E[∇log p(y|x,θ) ∇log p(y|x,θ)ᵀ]
```

**Implementation**: Uses **empirical Fisher** (ground-truth labels from training data):
```
F̂ = (1/N) Σᵢ ∇ℓᵢ ∇ℓᵢᵀ
```

**Numerical Stability**: Welford's algorithm for online mean/variance computation:
- Float64 precision for accumulation
- Token-normalized weighting for variable-length sequences
- Numerically stable against catastrophic cancellation

### Parameter Importance

Fisher diagonal Fᵢᵢ measures sensitivity of loss to parameter θᵢ:
- **High Fisher**: Parameter critical for task performance
- **Low Fisher**: Parameter can be pruned/merged safely

---

## Fisher Methods Comparison

| Method | Purpose | Use Case |
|--------|---------|----------|
| **Group Fisher** | Parameter importance | Pruning, merging, task arithmetic ✅ Primary |
| **KFAC** | Natural gradient | Second-order optimization |
| **Lanczos** | Eigen-spectrum | Optimization landscape analysis |
| **FisherSpectral** | Block-diagonal | Model capacity analysis |

**Note**: The unified analysis pipeline uses **Group Fisher** for all downstream analyses (Phases 2-7). Other methods are computed for comparison purposes only.

---

## 7-Phase Analysis Pipeline

The unified analysis system runs a comprehensive Fisher pipeline:

### Phase 1: Fisher Collection
- Enhanced Welford accumulation with token-aware weighting
- Optional behavioral head categorization
- Multi-architecture support (Qwen, LLaMA, GPT, GQA/MQA)

### Phase 2: Fisher Importance
- Parameter importance scores per task
- Layer-wise importance aggregation
- Task comparison metrics

### Phase 3: Pruning Masks
- Fisher-based binary masks for pruning
- Configurable sparsity levels
- Structured vs unstructured pruning

### Phase 4: Mask Overlap Analysis
- Quantifies parameter sharing between tasks
- Identifies conflict regions
- Generates merge recommendations

### Phase 5: Cross-task Conflict Detection
- Sample-level conflict identification
- Gradient-based interference detection
- Statistical significance testing

### Phase 6: QKOV Interference
- Circuit-level interference analysis
- Block-wise resolution (Q, K, V, O)
- Head-level attribution

### Phase 7: Additional Metrics
- Method comparison (Group/KFAC/Lanczos/Spectral)
- Fisher uncertainty quantification
- Advanced diagnostics

---

## Behavioral Head Categorization

### Overview

Fisher parameters can be grouped by **mechanistic function** rather than structural position:

**Traditional grouping**:
```
'model.layers.0.self_attn.q_proj.weight|head_0'
'model.layers.0.self_attn.q_proj.weight|head_1'
```

**Behavioral grouping**:
```
'model.layers.0.self_attn.q_proj.weight|induction_heads'
'model.layers.0.self_attn.q_proj.weight|positional_heads'
'model.layers.0.self_attn.q_proj.weight|content_heads'
```

### Usage

```python
from mechanistic.mechanistic_analyzer_core import MechanisticAnalyzer

# Enable behavioral categorization
mech_analyzer = MechanisticAnalyzer()
fisher_collector.set_mechanistic_analyzer(mech_analyzer)

# Fisher now groups by behavioral function
fisher_collector.collect_fisher(model, batch, task='math')

# Results use behavioral taxonomy
importance = fisher_collector.get_group_fisher('math')
# Keys like: 'layer.0.attn.q_proj|induction_heads'
```

### Head Types

- **Induction heads**: Copy and repeat patterns
- **Positional heads**: Attend to positional patterns  
- **Previous token heads**: Attend to previous token
- **Same token heads**: Self-attention patterns
- **Content heads**: Content-based attention
- **Mixed heads**: Multiple behaviors

### Benefits

- **Circuit-aware pruning**: Preserve functional circuits
- **Interference analysis**: Understand behavioral conflicts
- **Interpretable importance**: Know *why* parameters matter

---

## Configuration Guide

### FisherCollector Options

```python
fisher_collector = FisherCollector(
    reduction='group',           # 'param', 'group', 'block'
    storage='cpu_fp16',          # 'gpu_fp32', 'cpu_fp16', 'cpu_fp32'
    ema_decay=0.99,              # EMA decay factor
    computation_dtype='float32', # Numerical stability
    
    # Cross-task analysis
    enable_cross_task_analysis=True,
    gradient_memory_mb=50,
    
    # Advanced features
    n_bootstrap_samples=1000  # For uncertainty quantification
)
```

### UnifiedConfig Options

```python
config = UnifiedConfig(
    # Fisher computation
    fisher_batch_size=128,           # Batch size for Fisher
    fisher_n_samples=1000,           # Samples for estimation
    compute_fisher_eigenvalues=True, # Enable method comparison
    
    # Advanced features
    enable_qkov_interference=True,   # Circuit-level interference
    enable_cross_task_conflicts=True, # Sample-level conflicts
    
    # Reproducibility
    seed=42
)
```

### Architecture Support

**Automatic detection** for:
- **Standard**: LLaMA-style separate Q/K/V/O projections
- **Qwen**: `(hidden_size, hidden_size)` projection shapes
- **GQA/MQA**: Different KV head counts (`num_key_value_heads`)
- **GPT-fused**: Fused QKV projections (`attn.c_attn`)

**Graceful fallback** when head structure detection fails.

---

## Common Issues and Solutions

### Issue: Architecture Not Detected

**Symptoms**: Warning about "Could not separate heads" or fallback to channel reduction

**Solution**: Architecture-agnostic fallback is intentional and safe. For custom architectures, structure may not match expected patterns.

### Issue: High Memory Usage

**Solutions**:
```python
# Reduce batch size
config.fisher_batch_size = 64  # From default 128

# Use FP16 storage
fisher_collector = FisherCollector(storage='cpu_fp16')

# Limit gradient memory
fisher_collector = FisherCollector(gradient_memory_mb=25)
```

### Issue: Behavioral Categorization Not Working

**Solution**: Behavioral head categorization requires `mechanistic_analyzer`:
```python
fisher_collector.set_mechanistic_analyzer(mechanistic_analyzer)
```

Falls back gracefully to structural grouping if unavailable.

### Issue: NaN Fisher Values

**Causes**:
- Invalid model parameters (check for NaN in model)
- Out-of-bounds token IDs in batch
- Numerical overflow in gradients

**Solutions**:
- Enable gradient clipping
- Check input data validity
- Use `computation_dtype='float32'` for stability

---

## QKOV Integration

### Overview

QKOV (QK-OV Interference) provides circuit-level conflict detection using Fisher importance:

```python
from fisher.qkov import QKOVConfig, QKOVInterferenceMetric

# Auto-detect model configuration
qk_config = QKOVConfig.from_model(model)

# QKOV uses enhanced Fisher collector
interference_metric = QKOVInterferenceMetric(qk_config, fisher_collector)

# Compute circuit-level interference
interference = interference_metric.compute_interference('math', 'code')
```

### Behavioral Integration

With behavioral head categorization enabled:

```python
# Interference analysis per behavioral head type
for head_type in ['induction_heads', 'positional_heads', 'content_heads']:
    interference_by_type = interference_metric.get_interference_by_type(head_type)
    # Analyze which behavioral circuits conflict most
```

### Use Cases

- **Model merging**: Identify conflicting behavioral circuits
- **Task arithmetic**: Understand task vector interference
- **Pruning strategy**: Preserve important non-conflicting circuits
- **Optimization**: Avoid updating conflicting parameters

---

## References

### Key Papers

1. **Fisher Information**:
   - Martens & Grosse (2015): "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"
   - Kirkpatrick et al. (2017): "Overcoming catastrophic forgetting in neural networks"

2. **Numerical Stability**:
   - Welford (1962): "Note on a method for calculating corrected sums of squares and products"

3. **Task Interference**:
   - Ilharco et al. (2023): "Editing Models with Task Arithmetic"

### Related Documentation

- **QKOV Interference**: `fisher/qkov/README.md`
- **Unified Analysis**: `docs/UNIFIED_MODEL_ANALYSIS_DOCUMENTATION.md`
- **Mechanistic Analysis**: `mechanistic/README.md`

---

**Document Version**: 1.0  
**Last Updated**: 2025-10-06  
**Status**: ✅ Production Ready