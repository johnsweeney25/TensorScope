# Unified Model Analysis Framework - Complete Documentation

**Status**: ✅ **Production Ready** (ICML 2026)

## Overview

The Unified Model Analysis Framework is the **single entry point** for all model analysis tasks in this codebase. It provides a clean, unified interface to 50+ metrics across 8 specialized analysis modules, with built-in caching, memory management, and batch processing.

**Key Features**:
- **Single source of truth**: One configuration, one analysis call
- **Compute once, cache forever**: No redundant computations
- **Automatic batch sizing**: Adapts to your GPU memory
- **Memory-safe**: Handles 1.5B+ parameter models on 80GB GPUs
- **Reproducible**: Fixed seeds, deterministic algorithms
- **Extensible**: Easy to add new metrics

**File**: `unified_model_analysis.py` (9539 lines)

## Table of Contents
1. [Quick Start](#quick-start)
2. [Architecture Overview](#architecture-overview)
3. [Configuration System](#configuration-system)
4. [Main API Reference](#main-api-reference)
5. [Metric Categories](#metric-categories)
6. [Memory Management](#memory-management)
7. [Batch Size Configuration](#batch-size-configuration)
8. [Caching System](#caching-system)
9. [Common Usage Patterns](#common-usage-patterns)
10. [Troubleshooting](#troubleshooting)
11. [ICML Best Practices](#icml-best-practices)
12. [Extension Guide](#extension-guide)

---

## Quick Start

### Basic Usage

```python
from unified_model_analysis import analyze_models, UnifiedConfig

# Configure analysis
config = UnifiedConfig(
    # Model settings
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",

    # Batch sizes (automatically adapted if OOM)
    batch_size=32,              # Default batch size
    max_batch_size=256,         # For large datasets

    # Enable specific metrics
    compute_fisher=True,
    compute_loss_landscape=True,
    compute_gradients=True,

    # Reproducibility
    seed=42,

    # Output
    output_dir="./analysis_results",
    save_results=True
)

# Run analysis (computes once, caches results)
results = analyze_models(
    model_paths=["Qwen/Qwen2.5-Math-1.5B-Instruct"],
    config=config
)

# Access results
model_results = results.models["Qwen/Qwen2.5-Math-1.5B-Instruct"]
fisher_info = model_results.metrics["fisher_information"]
landscape = model_results.metrics["loss_landscape_2d"]
```

### Analyzing Multiple Checkpoints

```python
# Compare training checkpoints
checkpoints = [
    "checkpoint-1000",
    "checkpoint-2000",
    "checkpoint-3000"
]

config = UnifiedConfig(
    base_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    compute_all=True,  # Enable all metrics
    seed=42
)

results = analyze_models(checkpoints, config)

# Compare across checkpoints
for checkpoint in checkpoints:
    fisher = results.models[checkpoint].metrics["fisher_information"]
    print(f"{checkpoint}: Fisher trace = {fisher['trace']}")
```

### Memory-Constrained Systems

```python
# Configuration for 40GB GPU (e.g., A100)
config = UnifiedConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",

    # Conservative batch sizes
    batch_size=16,              # Reduced from 32
    max_batch_size=64,          # Reduced from 256

    # Memory-intensive metrics
    compute_loss_landscape=True,
    loss_landscape_n_points=19,  # Reduced from 25

    # Enable aggressive cleanup
    aggressive_cleanup=True
)

results = analyze_models(["model_path"], config)
```

---

## Architecture Overview

### Component Hierarchy

```
unified_model_analysis.py
│
├── analyze_models()              # Main entry point
│   └── UnifiedModelAnalyzer      # Orchestrates analysis
│       │
│       ├── MetricRegistry        # Registers all metrics
│       ├── ResultCache           # Caches computed results
│       ├── SimpleBatchManager    # Manages batch sizes
│       └── MetricContext         # Provides compute environment
│
└── Specialized Metric Modules:
    ├── BombshellMetrics          # Task vectors, TIES, etc.
    ├── GradientAnalysis          # Gradient flow, pathology
    ├── InformationTheoryMetrics  # Mutual information, PID
    ├── RepresentationAnalysisMetrics  # Dead neurons, superposition
    ├── ICLRMetrics               # Fisher, Hessian, loss landscape
    ├── MechanisticAnalyzer       # Attention patterns, induction
    ├── ModularityMetrics         # Modularity, separability
    └── LotteryTickets            # Lottery ticket hypothesis
```

### Data Flow

```
User Config → UnifiedConfig → MetricRegistry → Compute Metrics → Cache Results → Return AnalysisResults
                    ↓                ↓                ↓
                Validate        Select Enabled    Store in Cache
                                   Metrics
```

---

## Configuration System

### UnifiedConfig Class

The central configuration object for all analysis:

```python
@dataclass
class UnifiedConfig:
    # ========== Model Configuration ==========
    model_name: str = None                    # Model identifier
    base_model: str = None                    # Base model for checkpoints
    device: str = "auto"                      # "cuda", "cpu", or "auto"
    dtype: str = "auto"                       # "float32", "bfloat16", or "auto"

    # ========== Data Configuration ==========
    dataset_name: str = "openwebtext"         # Dataset for analysis
    dataset_split: str = "train"              # Dataset split
    num_samples: int = 1000                   # Number of samples to analyze
    max_seq_length: int = 512                 # Maximum sequence length

    # ========== Batch Size Configuration ==========
    batch_size: int = 32                      # Default batch size
    max_batch_size: int = 256                 # Max batch for large operations
    min_batch_size: int = 4                   # Min batch (OOM fallback)
    adaptive_batch_sizing: bool = True        # Auto-reduce on OOM

    # ========== Metric Selection ==========
    compute_all: bool = False                 # Enable all metrics
    compute_fisher: bool = False              # Fisher information
    compute_gradients: bool = False           # Gradient analysis
    compute_information_theory: bool = False  # MI, entropy, PID
    compute_representations: bool = False     # Dead neurons, superposition
    compute_mechanistic: bool = False         # Attention patterns
    compute_modularity: bool = False          # Modularity metrics
    compute_lottery: bool = False             # Lottery tickets
    compute_bombshell: bool = False           # Task vectors, TIES
    compute_loss_landscape: bool = False      # 2D loss landscape

    # ========== Fisher Information ==========
    fisher_n_samples: int = 1000              # Samples for Fisher estimation
    fisher_method: str = "empirical"          # "empirical" (recommended) or "exact"
    fisher_batch_size: int = 128              # Batch size for Fisher (H100-optimized)
    compute_fisher_eigenvalues: bool = True   # Enable advanced method comparison (KFAC/Lanczos)
    fisher_top_k: int = 20                    # Top K eigenvalues for Lanczos
    enable_qkov_interference: bool = True     # Enable circuit-level interference analysis
    enable_cross_task_conflicts: bool = True  # Enable sample-level conflict detection

    # ========== Loss Landscape ==========
    loss_landscape_n_points: int = 25         # Grid resolution (25×25)
    loss_landscape_span: float = 0.1          # Distance from origin
    loss_landscape_normalization: str = "layer"  # "filter", "layer", "global"
    loss_landscape_batches: int = 10          # Batches to average

    # ========== Gradient Analysis ==========
    gradient_trajectory_steps: int = 100      # Trajectory length
    compute_gradient_flow: bool = True        # Layer-wise flow
    compute_gradient_pathology: bool = True   # Gradient issues

    # ========== Memory Management ==========
    aggressive_cleanup: bool = True           # Clean after each metric
    enable_memory_tracking: bool = False      # Log memory usage
    max_memory_gb: float = 70.0               # Max GPU memory

    # ========== Reproducibility ==========
    seed: int = 42                            # Random seed
    deterministic: bool = True                # CUDA determinism

    # ========== Output Configuration ==========
    output_dir: str = "./analysis_results"    # Output directory
    save_results: bool = True                 # Save to disk
    save_plots: bool = False                  # Generate plots
    verbose: bool = True                      # Logging verbosity

    # ========== Advanced Options ==========
    use_cache: bool = True                    # Enable result caching
    force_recompute: bool = False             # Ignore cache
    skip_on_error: bool = True                # Continue on metric failure
    timeout_seconds: Optional[int] = None     # Per-metric timeout
```

### Creating Configurations

```python
# Minimal config (only required fields)
config = UnifiedConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    compute_fisher=True
)

# Full analysis config with Fisher enhancements
config = UnifiedConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    compute_fisher=True,
    compute_qkov=True,  # Enable QKOV interference analysis

    # Fisher-specific settings
    fisher_batch_size=128,           # H100-optimized for Fisher
    fisher_n_samples=1000,           # Samples for Fisher estimation
    compute_fisher_eigenvalues=True, # Enable method comparison
    enable_qkov_interference=True,   # Circuit-level interference
    enable_cross_task_conflicts=True, # Sample-level conflicts

    # General settings
    batch_size=32,
    seed=42,
    output_dir="./results"
)

# Memory-optimized config
config = UnifiedConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    batch_size=16,
    fisher_batch_size=64,  # Reduce Fisher batch size for memory
    max_batch_size=64,
    aggressive_cleanup=True,
    compute_loss_landscape=True,
    loss_landscape_n_points=19  # Smaller grid
)

# Load from JSON
with open("config.json") as f:
    config_dict = json.load(f)
config = UnifiedConfig(**config_dict)
```

---

## Main API Reference

### analyze_models()

**Main entry point** for all analysis tasks.

```python
def analyze_models(
    model_paths: List[str],
    config: UnifiedConfig = None
) -> AnalysisResults:
    """
    Analyze one or more models with unified configuration.

    Args:
        model_paths: List of model paths/identifiers
        config: Configuration object (uses defaults if None)

    Returns:
        AnalysisResults object with all computed metrics

    Raises:
        ValueError: If model_paths is empty or config is invalid
        RuntimeError: If critical computation fails
    """
```

**Example**:
```python
results = analyze_models(
    model_paths=["checkpoint-1000", "checkpoint-2000"],
    config=UnifiedConfig(compute_all=True, seed=42)
)

# Access results
for model_path in model_paths:
    model_results = results.models[model_path]
    print(f"\n{model_path}:")
    for metric_name, metric_value in model_results.metrics.items():
        print(f"  {metric_name}: {metric_value}")
```

### UnifiedModelAnalyzer Class

Low-level interface (advanced users only):

```python
analyzer = UnifiedModelAnalyzer(config=config)

# Load model
model, tokenizer = analyzer.load_model(model_path)

# Compute specific metric group
fisher_results = analyzer.compute_fisher_information(model, tokenizer)
gradient_results = analyzer.compute_gradient_analysis(model, tokenizer)
landscape_results = analyzer.compute_loss_landscape(model, tokenizer)

# Or compute all enabled metrics
all_results = analyzer.analyze_model(model_path)
```

---

## Metric Categories

### 1. Fisher Information & Hessian (`ICLRMetrics`, `BombshellMetrics`)

**Enabled by**: `compute_fisher=True`

**Metrics Computed**:
- `fisher_information`: Empirical Fisher Information Matrix
  - `trace`: Sum of diagonal elements
  - `eigenvalues`: Top K eigenvalues (via Lanczos)
  - `condition_number`: Optimization difficulty metric
  - `effective_rank`: Number of significant eigenvalues

- `fisher_importance`: Parameter importance scores per task
- `fisher_overlap`: Task interference quantification
- `qkov_interference`: Circuit-level conflict detection (if enabled)

**7-Phase Analysis Pipeline**:
1. Fisher collection (Welford accumulation, optional behavioral grouping)
2. Importance computation and task comparison
3. Pruning mask generation
4. Mask overlap analysis
5. Cross-task conflict detection
6. QKOV interference analysis
7. Method comparison (Group/KFAC/Lanczos/Spectral)

**Configuration**:
```python
config = UnifiedConfig(
    compute_fisher=True,
    fisher_batch_size=128,           # Batch size for Fisher
    fisher_n_samples=1000,           # Samples for estimation
    compute_fisher_eigenvalues=True, # Enable method comparison
    enable_qkov_interference=True,   # Circuit-level analysis
    enable_cross_task_conflicts=True # Sample-level conflicts
)
```

**See also**: `docs/FISHER_DOCUMENTATION.md` for complete Fisher guide

### 2. Loss Landscape (`ICLRMetrics`)

**Enabled by**: `compute_loss_landscape=True`

**Metrics Computed**:
- `loss_landscape_2d`: 2D visualization of loss surface
  - `grid_losses`: n_points × n_points array
  - `roughness`: Total variation (smoothness)
  - `loss_mean`, `loss_std`: Statistics
  - `orthogonality_check`: Direction quality
  - `n_batches_used`: Averaging count

**Configuration**:
```python
config = UnifiedConfig(
    compute_loss_landscape=True,
    loss_landscape_n_points=25,          # 25×25 grid (optimal)
    loss_landscape_span=0.1,             # Distance from origin
    loss_landscape_normalization="layer", # For transformers
    loss_landscape_batches=10             # Multi-batch averaging
)
```

**Batch Size**: Adaptive based on grid size
- 19×19: batch_size=32
- 25×25: batch_size=16 (default)
- 31×31: batch_size=8

**See**: `docs/LOSS_LANDSCAPE_2D_DOCUMENTATION.md`

### 3. Gradient Analysis (`GradientAnalysis`)

**Enabled by**: `compute_gradients=True`

**Metrics Computed**:
- `gradient_flow`: Layer-wise gradient statistics
  - `mean_gradient_norm`: Per-layer norms
  - `gradient_variance`: Gradient stability
  - `flow_interruptions`: Vanishing/exploding detection

- `gradient_pathology`: Gradient health metrics
  - `dead_neurons`: Neurons with zero gradients
  - `gradient_dispersion`: Spread of gradients
  - `effective_learning_rate`: Per-layer lr effectiveness

**Configuration**:
```python
config = UnifiedConfig(
    compute_gradients=True,
    gradient_trajectory_steps=100,  # Trajectory length
    compute_gradient_flow=True,
    compute_gradient_pathology=True
)
```

**Batch Size**: Uses `batch_size` (default: 32)

### 4. Information Theory (`InformationTheoryMetrics`)

**Enabled by**: `compute_information_theory=True`

**Metrics Computed**:
- `mutual_information`: MI between layers
- `entropy`: Layer-wise entropy
- `pid_analysis`: Partial Information Decomposition
  - `unique_x`, `unique_y`: Unique information
  - `redundancy`, `synergy`: Interaction types

**Configuration**:
```python
config = UnifiedConfig(
    compute_information_theory=True,
    batch_size=32
)
```

**Batch Size**: Uses `batch_size` (default: 32)

### 5. Representation Analysis (`RepresentationAnalysisMetrics`)

**Enabled by**: `compute_representations=True`

**Metrics Computed**:
- `dead_neurons`: Neurons with zero activation
- `superposition`: Polysemanticity analysis
- `feature_geometry`: Representation structure

**Configuration**:
```python
config = UnifiedConfig(
    compute_representations=True,
    batch_size=32
)
```

**Batch Size**: Uses `batch_size` (default: 32)

### 6. Mechanistic Interpretability (`MechanisticAnalyzer`)

**Enabled by**: `compute_mechanistic=True`

**Metrics Computed**:
- `attention_patterns`: Head specialization
- `induction_heads`: In-context learning
- `qk_ov_circuits`: Attention mechanisms

**Configuration**:
```python
config = UnifiedConfig(
    compute_mechanistic=True,
    batch_size=16  # Attention is memory-intensive
)
```

**Batch Size**: Uses `batch_size` (default: 16 for attention)

### 7. Modularity (`ModularityMetrics`)

**Enabled by**: `compute_modularity=True`

**Metrics Computed**:
- `modularity_score`: Network modularity
- `separability`: Task separability
- `task_interference`: Cross-task conflicts

**Configuration**:
```python
config = UnifiedConfig(
    compute_modularity=True,
    batch_size=32
)
```

**Batch Size**: Uses `batch_size` (default: 32)

### 8. Task Vectors (`BombshellMetrics`)

**Enabled by**: `compute_bombshell=True`

**Metrics Computed**:
- `task_vectors`: Parameter difference vectors
- `ties_merging`: TIES merge conflicts
- `fisher_merging`: Fisher-weighted merging

**Configuration**:
```python
config = UnifiedConfig(
    compute_bombshell=True,
    batch_size=32
)
```

**Batch Size**: Uses `batch_size` (default: 32)

### 9. Lottery Tickets (`LotteryTickets`)

**Enabled by**: `compute_lottery=True`

**Metrics Computed**:
- `pruning_robustness`: Mask stability
- `winning_tickets`: Lottery ticket analysis

**Configuration**:
```python
config = UnifiedConfig(
    compute_lottery=True,
    batch_size=32
)
```

**Batch Size**: Uses `batch_size` (default: 32)

---

## Memory Management

### Automatic Memory Management

The framework automatically manages memory through multiple strategies:

#### 1. Adaptive Batch Sizing

```python
# Automatically reduces batch size on OOM
config = UnifiedConfig(
    batch_size=32,              # Initial batch size
    min_batch_size=4,           # Don't go below this
    adaptive_batch_sizing=True  # Enable auto-reduction
)

# If OOM occurs:
# 32 → 16 → 8 → 4 (stops at min_batch_size)
```

#### 2. Aggressive Cleanup

```python
# Clean memory after each metric
config = UnifiedConfig(
    aggressive_cleanup=True  # Default
)

# This calls cleanup_memory() after each metric:
# - gc.collect()
# - torch.cuda.empty_cache()
# - torch.cuda.synchronize()
```

#### 3. Memory Tracking

```python
# Enable memory logging
config = UnifiedConfig(
    enable_memory_tracking=True,
    verbose=True
)

# Logs:
# [GPU Memory] Before Fisher: 12.3GB allocated, 67.7GB free
# [GPU Memory] After Fisher: 15.1GB allocated, 64.9GB free
```

### Manual Memory Management

```python
from unified_model_analysis import cleanup_memory

# Clean memory at any point
cleanup_memory(verbose=True, reason="After large computation")
```

### Memory Limits

```python
# Set hard memory limit
config = UnifiedConfig(
    max_memory_gb=70.0,  # Fail if exceeds 70GB
    skip_on_error=True   # Skip metric instead of crashing
)
```

---

## Batch Size Configuration

### Centralized Batch Size Management

All batch sizes are configured in `UnifiedConfig` and automatically applied to all metrics:

```python
config = UnifiedConfig(
    batch_size=32,           # Default for most metrics
    max_batch_size=256,      # For large datasets
    min_batch_size=4,        # OOM fallback
)
```

### Metric-Specific Overrides

Some metrics automatically adjust batch size:

| Metric | Batch Size Logic | Reason |
|--------|-----------------|--------|
| Loss Landscape (19×19) | 32 | Low memory overhead |
| **Loss Landscape (25×25)** | **16** | **Balanced (default)** |
| Loss Landscape (31×31) | 8 | High memory overhead |
| Attention Patterns | 16 | Attention is memory-intensive |
| Integrated Gradients | 32 (chunked) | Large n_steps × batch |
| Other Metrics | batch_size | User-configured |

### OOM Recovery

```python
# Automatic batch size reduction on OOM
try:
    result = compute_metric(batch_size=32)
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        # Automatically retries with batch_size=16
        result = compute_metric(batch_size=16)
```

### Batch Size Guidelines

| GPU Memory | Recommended Batch Size | Max Batch Size | Notes |
|------------|----------------------|---------------|-------|
| 80GB (H100) | 32 | 256 | Optimal for 1.5B models |
| 40GB (A100) | 16 | 128 | Reduce for memory-intensive metrics |
| 24GB (3090) | 8 | 64 | Conservative settings |
| 16GB (4090) | 4 | 32 | Minimal viable |

**See**: `docs/BATCH_SYSTEM_DOCUMENTATION.md`

---

## Caching System

### How Caching Works

Results are cached based on:
1. Model path/identifier
2. Metric name
3. Configuration hash (batch size, seed, etc.)

```python
# First call: Computes and caches
results1 = analyze_models(
    model_paths=["model_path"],
    config=UnifiedConfig(compute_fisher=True, seed=42)
)

# Second call: Returns cached result instantly
results2 = analyze_models(
    model_paths=["model_path"],
    config=UnifiedConfig(compute_fisher=True, seed=42)
)
```

### Cache Location

```
{output_dir}/
└── cache/
    └── {model_identifier}/
        ├── fisher_information_hash_{config_hash}.pkl
        ├── loss_landscape_2d_hash_{config_hash}.pkl
        └── ...
```

### Forcing Recomputation

```python
# Ignore cache and recompute
config = UnifiedConfig(
    compute_fisher=True,
    force_recompute=True  # Bypass cache
)

results = analyze_models(model_paths, config)
```

### Disabling Cache

```python
# Disable caching entirely
config = UnifiedConfig(
    compute_fisher=True,
    use_cache=False
)

results = analyze_models(model_paths, config)
```

### Cache Management

```python
from unified_model_analysis import ResultCache

# Clear cache for specific model
cache = ResultCache(cache_dir="./analysis_results/cache")
cache.clear_model_cache("model_path")

# Clear all caches
cache.clear_all()

# Get cache statistics
stats = cache.get_stats()
print(f"Cache size: {stats['total_size_mb']:.2f} MB")
print(f"Cached metrics: {stats['num_entries']}")
```

---

## Common Usage Patterns

### Pattern 1: Compare Training Checkpoints

```python
# Analyze checkpoints at different steps
checkpoints = [
    "checkpoint-1000",
    "checkpoint-2000",
    "checkpoint-3000",
    "checkpoint-4000"
]

config = UnifiedConfig(
    base_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    compute_fisher=True,
    compute_gradients=True,
    compute_loss_landscape=True,
    seed=42
)

results = analyze_models(checkpoints, config)

# Extract trajectory
fisher_traces = []
for checkpoint in checkpoints:
    fisher = results.models[checkpoint].metrics["fisher_information"]
    fisher_traces.append(fisher["trace"])

# Plot training trajectory
import matplotlib.pyplot as plt
plt.plot(checkpoints, fisher_traces)
plt.xlabel("Checkpoint")
plt.ylabel("Fisher Trace")
plt.title("Fisher Information During Training")
plt.show()
```

### Pattern 2: Multi-Model Comparison

```python
# Compare different model architectures
models = [
    "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-v0.1"
]

config = UnifiedConfig(
    compute_all=True,
    batch_size=32,
    seed=42
)

results = analyze_models(models, config)

# Compare Fisher traces
for model in models:
    fisher = results.models[model].metrics["fisher_information"]
    print(f"{model}: Fisher trace = {fisher['trace']:.2e}")
```

### Pattern 3: Ablation Studies

```python
# Test different hyperparameters
configs = [
    UnifiedConfig(loss_landscape_n_points=11, loss_landscape_batches=5),
    UnifiedConfig(loss_landscape_n_points=19, loss_landscape_batches=10),
    UnifiedConfig(loss_landscape_n_points=25, loss_landscape_batches=10),
    UnifiedConfig(loss_landscape_n_points=31, loss_landscape_batches=15),
]

for i, config in enumerate(configs):
    config.output_dir = f"./ablation_study/config_{i}"
    config.compute_loss_landscape = True
    results = analyze_models(["model_path"], config)

    landscape = results.models["model_path"].metrics["loss_landscape_2d"]
    print(f"Config {i}: Roughness = {landscape['roughness']:.4f}")
```

### Pattern 4: Error Handling

```python
# Graceful degradation on metric failures
config = UnifiedConfig(
    compute_all=True,
    skip_on_error=True,  # Continue on failure
    timeout_seconds=300,  # 5 min per metric
    verbose=True
)

results = analyze_models(["model_path"], config)

# Check which metrics succeeded
for metric_name, metric_value in results.models["model_path"].metrics.items():
    if isinstance(metric_value, dict) and "error" in metric_value:
        print(f"❌ {metric_name}: {metric_value['error']}")
    else:
        print(f"✅ {metric_name}: Success")
```

### Pattern 5: Batch Job Processing

```python
# Process multiple models in a batch job
import os
from pathlib import Path

checkpoint_dir = Path("./checkpoints")
checkpoints = sorted([str(p) for p in checkpoint_dir.glob("checkpoint-*")])

config = UnifiedConfig(
    compute_all=True,
    batch_size=16,  # Conservative for batch jobs
    save_results=True,
    output_dir="./batch_results",
    seed=42
)

# Process all checkpoints
for i, checkpoint in enumerate(checkpoints):
    print(f"\nProcessing {i+1}/{len(checkpoints)}: {checkpoint}")

    try:
        results = analyze_models([checkpoint], config)
        print(f"✅ {checkpoint} completed")
    except Exception as e:
        print(f"❌ {checkpoint} failed: {e}")
        continue
```

---

## Troubleshooting

### Issue 1: CUDA Out of Memory

**Symptoms**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# 1. Reduce batch size
config = UnifiedConfig(
    batch_size=16,  # Reduce from 32
    max_batch_size=64  # Reduce from 256
)

# 2. Enable aggressive cleanup
config.aggressive_cleanup = True

# 3. Reduce grid size for loss landscape
config.loss_landscape_n_points = 19  # Reduce from 25

# 4. Disable memory-intensive metrics
config.compute_mechanistic = False  # Attention is expensive
config.compute_loss_landscape = False
```

### Issue 2: Slow Computation

**Symptoms**: Analysis takes too long

**Solutions**:
```python
# 1. Reduce number of samples
config = UnifiedConfig(
    num_samples=500,  # Reduce from 1000
    fisher_n_samples=500  # Reduce from 1000
)

# 2. Reduce loss landscape resolution
config.loss_landscape_n_points = 19  # Faster than 25

# 3. Disable slow metrics
config.compute_lottery = False  # Pruning is slow
config.compute_mechanistic = False  # Attention is slow

# 4. Use caching
config.use_cache = True  # Default
```

### Issue 3: NaN or Inf Values

**Symptoms**: Metrics return NaN/Inf

**Solutions**:
```python
# 1. Check model is in eval mode (automatic)
# 2. Verify dataset has valid samples
# 3. Check for numerical instability

# Enable numerical debugging
config = UnifiedConfig(
    verbose=True,
    skip_on_error=True  # Skip problematic metrics
)

# Check specific metric
results = analyze_models(["model_path"], config)
if "error" in results.models["model_path"].metrics["fisher_information"]:
    print("Fisher computation failed")
```

### Issue 4: Reproducibility Issues

**Symptoms**: Different results with same seed

**Causes**:
1. CUDA non-determinism (inevitable, ~1e-7 variation)
2. Batch order changed
3. Different random state

**Solutions**:
```python
# Enable maximum determinism
config = UnifiedConfig(
    seed=42,
    deterministic=True,  # Enables torch.backends.cudnn.deterministic
    batch_size=32  # Keep consistent
)

# Note: CUDA non-determinism of O(1e-7) is acceptable for ICML
```

### Issue 5: Cache Not Working

**Symptoms**: Metrics recompute every time

**Causes**:
1. Config hash changed (different parameters)
2. Cache disabled
3. force_recompute=True

**Solutions**:
```python
# Verify caching is enabled
config = UnifiedConfig(
    use_cache=True,  # Default
    force_recompute=False  # Default
)

# Check cache directory exists
import os
os.makedirs(config.output_dir + "/cache", exist_ok=True)

# Verify config is identical
# Even small changes (batch_size, seed) create new cache entries
```

---

## ICML Best Practices

### 1. Reproducibility

```python
# ICML-ready reproducibility setup
config = UnifiedConfig(
    seed=42,                # Fixed seed
    deterministic=True,     # CUDA determinism
    batch_size=32,          # Document this value
    save_results=True,      # Save for supplementary material
    output_dir="./icml_results"
)
```

**In paper**: Document CUDA non-determinism (~1e-7 variation)

### 2. Multi-Batch Averaging

```python
# Reduce noise for publication-quality results
config = UnifiedConfig(
    loss_landscape_batches=10,  # 10× variance reduction
    fisher_n_samples=1000,      # More samples = better estimate
    seed=42
)
```

**In paper**: Report number of batches/samples used

### 3. Ablation Studies

```python
# Run ablations for key hyperparameters
ablation_configs = {
    "normalization": [
        UnifiedConfig(loss_landscape_normalization="filter"),
        UnifiedConfig(loss_landscape_normalization="layer"),
        UnifiedConfig(loss_landscape_normalization="global"),
    ],
    "grid_size": [
        UnifiedConfig(loss_landscape_n_points=11),
        UnifiedConfig(loss_landscape_n_points=19),
        UnifiedConfig(loss_landscape_n_points=25),
    ]
}

for ablation_name, configs in ablation_configs.items():
    for i, config in enumerate(configs):
        config.output_dir = f"./ablations/{ablation_name}/config_{i}"
        results = analyze_models(["model"], config)
```

**In paper**: Include ablation tables in appendix

### 4. Statistical Significance

```python
# Run multiple seeds for error bars
seeds = [42, 123, 456, 789, 999]
fisher_traces = []

for seed in seeds:
    config = UnifiedConfig(compute_fisher=True, seed=seed)
    results = analyze_models(["model"], config)
    fisher = results.models["model"].metrics["fisher_information"]
    fisher_traces.append(fisher["trace"])

# Report mean ± std
import numpy as np
mean_trace = np.mean(fisher_traces)
std_trace = np.std(fisher_traces)
print(f"Fisher trace: {mean_trace:.2e} ± {std_trace:.2e}")
```

**In paper**: Report mean ± std across seeds

### 5. Save Everything

```python
# Save all results and configurations for reproducibility
config = UnifiedConfig(
    save_results=True,
    save_plots=True,
    output_dir="./icml_submission"
)

results = analyze_models(models, config)

# Also save config
import json
with open("./icml_submission/config.json", "w") as f:
    json.dump(asdict(config), f, indent=2)
```

**In paper**: "Code and data available at [URL]"

---

## Extension Guide

### Adding a New Metric

#### Step 1: Create Metric Function

Add to appropriate module (e.g., `ICLRMetrics.py`):

```python
def compute_my_new_metric(
    self,
    model,
    data_batch: Dict[str, torch.Tensor],
    param1: int = 10,
    param2: float = 0.5
) -> Dict[str, Any]:
    """
    Compute my new metric.

    Args:
        model: PyTorch model
        data_batch: Input batch
        param1: Parameter 1
        param2: Parameter 2

    Returns:
        Dict with metric results
    """
    try:
        # Your computation here
        result = ...

        return {
            'metric_value': result,
            'param1_used': param1,
            'param2_used': param2
        }
    except Exception as e:
        logger.error(f"Error computing my_new_metric: {e}")
        return {'error': str(e)}
```

#### Step 2: Register in UnifiedConfig

Add configuration to `UnifiedConfig`:

```python
@dataclass
class UnifiedConfig:
    # ... existing fields ...

    # My New Metric
    compute_my_new_metric: bool = False
    my_new_metric_param1: int = 10
    my_new_metric_param2: float = 0.5
```

#### Step 3: Register in MetricRegistry

Add to `UnifiedModelAnalyzer._register_metrics()`:

```python
def _register_metrics(self):
    """Register all available metrics."""
    # ... existing registrations ...

    # My new metric
    if self.config.compute_my_new_metric:
        self.registry.register(
            name="my_new_metric",
            category="custom",
            compute_fn=self._compute_my_new_metric_wrapper,
            dependencies=[],
            batch_size=self.config.batch_size
        )
```

#### Step 4: Create Wrapper Function

Add wrapper in `UnifiedModelAnalyzer`:

```python
def _compute_my_new_metric_wrapper(
    self,
    model,
    tokenizer,
    data_loader
) -> Dict[str, Any]:
    """Wrapper for my_new_metric."""
    batch = next(iter(data_loader))
    batch = {k: v.to(self.device) for k, v in batch.items()}

    return self.iclr_metrics.compute_my_new_metric(
        model=model,
        data_batch=batch,
        param1=self.config.my_new_metric_param1,
        param2=self.config.my_new_metric_param2
    )
```

#### Step 5: Add Documentation

Update this documentation file with your metric.

#### Step 6: Add Tests

Create `test_my_new_metric.py`:

```python
from unified_model_analysis import analyze_models, UnifiedConfig

def test_my_new_metric():
    config = UnifiedConfig(
        model_name="gpt2",
        compute_my_new_metric=True,
        my_new_metric_param1=20,
        my_new_metric_param2=0.8
    )

    results = analyze_models(["gpt2"], config)
    metric = results.models["gpt2"].metrics["my_new_metric"]

    assert "metric_value" in metric
    assert metric["param1_used"] == 20
    assert metric["param2_used"] == 0.8
    print("✅ Test passed")

if __name__ == "__main__":
    test_my_new_metric()
```

---

## References

### Primary Implementation

- **File**: `unified_model_analysis.py` (9539 lines)
- **Main Entry**: `analyze_models()`
- **Main Class**: `UnifiedModelAnalyzer`

### Related Documentation

- **Loss Landscape**: `docs/LOSS_LANDSCAPE_2D_DOCUMENTATION.md`
- **Batch System**: `docs/BATCH_SYSTEM_DOCUMENTATION.md`
- **Fisher Information**: `docs/FISHER_EIGENVALUES_LANCZOS_DOCUMENTATION.md`
- **Lottery Tickets**: `docs/LOTTERY_TICKETS_DOCUMENTATION.md`
- **Manifold Metrics**: `docs/MANIFOLD_METRICS_DOCUMENTATION.md`
- **SAM Sharpness**: `docs/SAM_SHARPNESS_DOCUMENTATION.md`
- **Report Generation**: `docs/REPORT_GENERATION_DOCUMENTATION.md`

### Key Papers

1. **Fisher Information**:
   - Martens & Grosse (2015): "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"

2. **Loss Landscape**:
   - Li et al. (2018): "Visualizing the Loss Landscape of Neural Nets"

3. **Lottery Tickets**:
   - Frankle & Carbin (2019): "The Lottery Ticket Hypothesis"

4. **Task Vectors**:
   - Ilharco et al. (2023): "Editing Models with Task Arithmetic"

---

## Appendix: Complete Example

```python
#!/usr/bin/env python3
"""
Complete example: Analyze Qwen2.5-Math checkpoints for ICML submission.
"""

from unified_model_analysis import analyze_models, UnifiedConfig
import matplotlib.pyplot as plt
import numpy as np

# Configuration
checkpoints = [
    "Qwen/Qwen2.5-Math-1.5B-Instruct-checkpoint-1000",
    "Qwen/Qwen2.5-Math-1.5B-Instruct-checkpoint-2000",
    "Qwen/Qwen2.5-Math-1.5B-Instruct-checkpoint-3000",
]

config = UnifiedConfig(
    # Model settings
    base_model="Qwen/Qwen2.5-Math-1.5B-Instruct",
    device="cuda",
    dtype="bfloat16",

    # Enable key metrics
    compute_fisher=True,
    compute_qkov=True,  # Enable QKOV interference analysis
    compute_loss_landscape=True,
    compute_gradients=True,

    # Fisher settings (enhanced)
    fisher_batch_size=128,           # H100-optimized for Fisher
    fisher_n_samples=1000,           # Samples for Fisher estimation
    compute_fisher_eigenvalues=True, # Enable method comparison (KFAC/Lanczos)
    fisher_top_k=20,                 # Top K eigenvalues for Lanczos
    enable_qkov_interference=True,   # Circuit-level interference
    enable_cross_task_conflicts=True, # Sample-level conflicts

    # Loss landscape settings
    loss_landscape_n_points=25,
    loss_landscape_normalization="layer",
    loss_landscape_batches=10,

    # Batch sizes
    batch_size=32,
    max_batch_size=256,

    # Reproducibility
    seed=42,
    deterministic=True,

    # Output
    output_dir="./icml_analysis",
    save_results=True,
    save_plots=True,
    verbose=True
)

# Analyze all checkpoints
print("Analyzing checkpoints...")
results = analyze_models(checkpoints, config)

# Extract metrics (enhanced with Fisher improvements)
fisher_traces = []
fisher_condition_numbers = []
qk_interference_scores = []
roughnesses = []
gradient_norms = []

for checkpoint in checkpoints:
    model_results = results.models[checkpoint]

    # Enhanced Fisher metrics
    fisher = model_results.metrics["fisher_information"]
    fisher_traces.append(fisher["trace"])
    fisher_condition_numbers.append(fisher.get("condition_number", 1.0))

    # QKOV interference (new with our enhancements)
    if "qkov_interference" in model_results.metrics:
        qkov = model_results.metrics["qkov_interference"]
        # Average interference across behavioral head types
        interference_scores = []
        for block in ['Q', 'K', 'V', 'O']:
            if block in qkov and 'layer_head_avg' in qkov[block]:
                interference_scores.append(np.mean(qkov[block]['layer_head_avg']))
        qk_interference_scores.append(np.mean(interference_scores) if interference_scores else 0.0)

    # Loss landscape roughness
    landscape = model_results.metrics["loss_landscape_2d"]
    roughnesses.append(landscape["roughness"])

    # Gradient norm
    gradients = model_results.metrics["gradient_flow"]
    gradient_norms.append(np.mean(gradients["mean_gradient_norm"]))

# Plot results (enhanced with Fisher improvements)
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Fisher trace
axes[0, 0].plot(range(1, len(checkpoints)+1), fisher_traces, 'o-', color='blue')
axes[0, 0].set_xlabel("Checkpoint")
axes[0, 0].set_ylabel("Fisher Trace")
axes[0, 0].set_title("Fisher Information During Training")
axes[0, 0].grid(True)

# Fisher condition number (optimization difficulty)
axes[0, 1].plot(range(1, len(checkpoints)+1), fisher_condition_numbers, 'o-', color='red')
axes[0, 1].set_xlabel("Checkpoint")
axes[0, 1].set_ylabel("Condition Number")
axes[0, 1].set_title("Fisher Condition Number (Optimization Difficulty)")
axes[0, 1].grid(True)

# QKOV interference (circuit-level conflicts)
if qk_interference_scores and any(s > 0 for s in qk_interference_scores):
    axes[1, 0].plot(range(1, len(checkpoints)+1), qk_interference_scores, 'o-', color='green')
    axes[1, 0].set_xlabel("Checkpoint")
    axes[1, 0].set_ylabel("QKOV Interference")
    axes[1, 0].set_title("Circuit-Level Task Interference")
    axes[1, 0].grid(True)
else:
    axes[1, 0].text(0.5, 0.5, "QKOV Interference\n(Not Available)", ha='center', va='center', transform=axes[1, 0].transAxes)
    axes[1, 0].set_title("QKOV Interference")

# Loss landscape roughness
axes[1, 1].plot(range(1, len(checkpoints)+1), roughnesses, 'o-', color='orange')
axes[1, 1].set_xlabel("Checkpoint")
axes[1, 1].set_ylabel("Roughness")
axes[1, 1].set_title("Loss Landscape Smoothness")
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig("./icml_analysis/training_metrics.pdf")
plt.show()

print("\n✅ Analysis complete!")
print(f"Results saved to: {config.output_dir}")
print("\n📊 Key Insights:")
print(f"  • Fisher trace evolution: {fisher_traces[0]:.2e} → {fisher_traces[-1]:.2e}")
print(f"  • Condition number: {fisher_condition_numbers[0]:.2f} → {fisher_condition_numbers[-1]:.2f}")
if qk_interference_scores and any(s > 0 for s in qk_interference_scores):
    print(f"  • QKOV interference: {qk_interference_scores[0]:.3f} → {qk_interference_scores[-1]:.3f}")
print(f"  • Loss landscape roughness: {roughnesses[0]:.3f} → {roughnesses[-1]:.3f}")
```

---

**Document Version**: 2.0  
**Last Updated**: 2025-10-06  
**Status**: ✅ Production Ready (ICLR 2026)