# Fisher Information Matrix - Complete Documentation

**Version 2.0** | **Last Updated**: 2025-10-07 | **Status**: ✅ Production Ready

**Note**: All default Fisher quantities in this doc are **diagonal** (elementwise squares); full Fisher/KFAC are optional variants.

---

## Table of Contents
1. [Quick Start](#quick-start)
2. [Theoretical Foundation](#theoretical-foundation)
3. [Data Flow Architecture](#data-flow-architecture)
4. [7-Phase Analysis Pipeline](#7-phase-analysis-pipeline)
5. [Configuration Guide](#configuration-guide)
6. [Fisher Methods Comparison](#fisher-methods-comparison)
7. [Behavioral Head Categorization](#behavioral-head-categorization)
8. [QK-OV Integration (Phase 6)](#qk-ov-integration-phase-6)
9. [Common Issues and Solutions](#common-issues-and-solutions)
10. [Advanced Topics](#advanced-topics)

---

## Quick Start

### Basic Usage

```python
from fisher.core.fisher_collector import FisherCollector

# Initialize Fisher collector
fisher_collector = FisherCollector(
    reduction='group',  # Group-level parameter reduction
    storage_device='cpu',
    storage_dtype='float16'
)

# Collect Fisher information
fisher_collector.collect_fisher(model, batch, task='math')

# Get importance scores
importance = fisher_collector.get_group_fisher('math')
```

### Production Configuration (Recommended)

```python
from unified_model_analysis import analyze_models, UnifiedConfig

# Complete Fisher analysis with all features
config = UnifiedConfig(
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    
    # Enable Fisher analysis
    compute_fisher=True,
    fisher_batch_size=128,
    fisher_n_samples=1000,
    
    # ⚠️ CRITICAL: Enable for Phase 6 (QK-OV) and Phase 5 (conflicts)
    enable_cross_task_analysis=True,
    
    # Optional advanced features
    compute_fisher_eigenvalues=True,
    enable_cross_task_conflicts=True,
    
    seed=42
)

results = analyze_models([model_path], config)
```

**Key Setting**: `enable_cross_task_analysis=True` is **required** for:
- Phase 5: Cross-task conflict detection
- Phase 6: QK-OV interference analysis
- Per-sample contribution storage

---

## Theoretical Foundation

### Fisher Information Matrix

The Fisher Information Matrix quantifies how sensitive the model's predictions are to parameter changes:

**True Fisher (theoretical)**:
```
F = E_{x~D, y~p(y|x)}[(∇log p(y|x,θ)) (∇log p(y|x,θ))ᵀ]
```

**Empirical Fisher (our implementation)**:
```
F̂ = E_{x~D}[(∇log p(y*|x,θ)) (∇log p(y*|x,θ))ᵀ]
    ≈ (1/N) Σᵢ (∇log p(yᵢ*|xᵢ,θ)) (∇log p(yᵢ*|xᵢ,θ))ᵀ
```

where y* is the ground-truth label from training data.

### Per-Sample Contributions

For each sample i, we define the **contribution**:

```
C_i = (∇log p(yᵢ*|xᵢ,θ))²
```

This measures how much sample i "pulls" on each parameter. Key properties:

- **C_i ≥ 0**: Always non-negative
- **E[Cᵢ] = diag(F̂)**: The expected contribution equals the diagonal of the empirical Fisher
- **Var[C_i]**: Measures sample diversity

### Welford's Algorithm for Numerical Stability

Instead of naive accumulation, we use **Welford's algorithm** for computing F̂:

```python
# Initialize
mean = 0
M2 = 0
count = 0

# For each sample
for i in range(N):
    count += 1
    delta = sample_i - mean
    mean += delta / count
    delta2 = sample_i - mean
    M2 += delta * delta2
    
# Fisher estimate
fisher = mean  # Unbiased estimate
variance = M2 / (count - 1)  # Sample variance
```

**Why Welford?**
- ✅ **Numerically stable**: No catastrophic cancellation
- ✅ **Single pass**: Processes samples online
- ✅ **Float64 accumulation**: Prevents precision loss
- ✅ **Bias correction**: Automatically handles small sample sizes

---

## Data Flow Architecture

Understanding the data flow is **critical** for using Fisher correctly.

### Phase 1 Outputs: TWO Storage Systems

Phase 1 (Fisher Collection) produces **two different types of data**:

#### 1. `fisher_ema` (Group-Reduced Fisher)

**Purpose**: Efficient parameter importance for Phases 2-4

**Storage**:
```python
fisher_collector.fisher_ema[key] = group_fisher
# key format: 'task|param_name|group_type'
```

**Shape Examples**:
- Attention parameters: `[num_heads]` (e.g., `[16]` for 16 heads)
- MLP parameters: `[num_channels]` (e.g., `[4096]`)

**Computation** (Square-Then-Reduce):
```
1. Compute full parameter gradient: grad [4096, 4096]
2. Square elementwise: grad² → [4096, 4096]
3. Reduce to groups: grad² → [16] (by head)
4. Welford accumulate / Store in fisher_ema[key]
```

**Used By**: Phases 2, 3, 4 (mask generation, overlap analysis)

#### 2. `contribution_cache` (Full Parameter Tensors)

**Purpose**: Per-sample contributions for Phases 5 & 6

**Storage**:
```python
# Nested by task: contribution_cache[task][sample_key][param_name]
fisher_collector.contribution_cache[task][f"{task}_{sample_idx}"][param_name] = C_i
```

**Shape Examples**:
- Attention parameters: `[4096, 4096]` (FULL parameter tensor)
- MLP parameters: `[11008, 4096]` (FULL parameter tensor)

**Computation** (Square-Then-Store):
```
1. Compute full parameter gradient: grad [4096, 4096]
2. Square BEFORE reduction: grad² → [4096, 4096]
3. Normalize by tokens: grad² / num_tokens
4. Store in contribution_cache[task][sample][param]
```

**Important**: `gradient_manager` returns **token-sum** per-sample gradients to match the per-token normalization of contributions; do not use token-mean unless Fisher is also scaled accordingly.

**Used By**: 
- Phase 5: Cross-task conflict detection
- Phase 6: QK-OV interference (computes Fisher from these)

### Critical Data Flow Diagram

```
Phase 1: Fisher Collection
├─ For each sample i:
│  ├─ Forward pass → loss
│  ├─ Backward pass → gradients
│  │
│  ├─ Path A: Full Tensor Storage (if enable_cross_task_analysis=True)
│  │  ├─ grad² → [4096, 4096]
│  │  └─ Store in contribution_cache[task][i][param]  ← Phase 6 uses this!
│  │
│  └─ Path B: Group Reduction (Square-Then-Reduce)
│     ├─ grad → [4096, 4096]
│     ├─ Square elementwise → [4096, 4096]
│     ├─ Reduce to groups → [16]
│     ├─ Welford accumulate
│     └─ Store in fisher_ema[key]  ← Phases 2-4 use this

Phases 2-4: Use fisher_ema (group-reduced)
  ├─ Phase 2: Fisher importance scores
  ├─ Phase 3: Pruning masks
  └─ Phase 4: Mask overlap

Phase 5: Use contribution_cache (full tensors)
  └─ Detect cross-task sample conflicts

Phase 6: Use contribution_cache (full tensors)
  ├─ Compute Fisher: Î_n ≈ (1/n) Σ C_i
  └─ QK-OV block-wise interference
```

### Why Two Storage Systems?

**Design Rationale**:

1. **Memory Efficiency**: Most phases (2-4) don't need full tensors
   - `fisher_ema[param]`: 64 bytes (16 heads × 4 bytes)
   - `contribution_cache[sample][param]`: 32 MB (4096×4096×2 bytes)

2. **Different Granularity Needs**:
   - Phases 2-4: Need aggregated importance (heads, channels)
   - Phases 5-6: Need per-sample, full-parameter resolution

3. **Computational Efficiency**:
   - Group reduction in Phase 1 → faster Phases 2-4
   - Full storage in Phase 1 → enables advanced Phases 5-6

---

## 7-Phase Analysis Pipeline

### Phase 1: Fisher Collection with Welford

**What it does**:
- Computes Fisher Information using Welford's algorithm
- Stores group-reduced Fisher in `fisher_ema`
- Stores full per-sample contributions in `contribution_cache` (if enabled)

**Theoretical Foundation**:

**Empirical Fisher Information Matrix**:
```
F = E_x[(∇L(x, θ))²]  = (1/n) Σᵢ (∇L(xᵢ, θ))²
```

Where:
- L(x, θ): Loss for sample x with parameters θ
- ∇L: Gradient of loss w.r.t. parameters
- Diagonal approximation: F_ii only (ignoring cross-terms F_ij)

**Why Welford's Algorithm?**

Naïve one-pass summation (less stable):
```python
fisher = sum([grad**2 for grad in gradients]) / n
# Error: O(n·ε) ≈ 1e-5 for n=100
# Still unbiased, but numerically worse
```

Welford's algorithm (numerically superior):
```python
# Online update:
delta = x - mean
mean += delta / n
M2 += delta * (x - mean)
# Error: O(ε) ≈ 1e-7 (constant!)
```

**Numerical Stability** (Welford 1962):
- Standard accumulation: error grows with n
- Welford: error stays constant (independent of n)
- 100× precision improvement for large datasets
- Critical for ICML/ICLR publication standards

**Group Reduction** (Square-Then-Reduce):

To reduce memory, we aggregate parameters by "groups":
```python
# CRITICAL ORDER: Square FIRST, then reduce
# Example: Linear layer [4096, 4096] → 16 groups

# Step 1: Square elementwise (diagonal Fisher)
fisher_full = grad.pow(2)  # [4096, 4096]

# Step 2: Reduce to groups
fisher_reduced = [fisher_full[group_i].mean() for i in range(16)]  # [16]
```

**Why square-then-reduce?**
- Diagonal Fisher: E[g²], not (E[g])²
- Counterexample: g = [1, -1]
  - ❌ Reduce→Square: (1 + (-1))² = 0
  - ✅ Square→Reduce: 1² + (-1)² = 2

Benefits:
- Memory: O(num_groups) instead of O(num_params)
- Sufficient for Phases 2-4 (importance, pruning, overlap)
- **Note**: Both group Fisher and Phase 6 use diagonal Fisher (per-parameter variance), not full covariance

**Per-Sample Contributions** (if `enable_cross_task_analysis=True`):

Stores FULL per-sample squared gradients (square-then-store):
```python
# For each sample i, store C_i = (∇L_i)²  [full parameter tensor]
# Schema: contribution_cache[task][f"{task}_{i}"][param_name]
sample_key = f"{task}_{i}"
contribution_cache[task][sample_key][param_name] = grad_i.pow(2)  # [4096, 4096]
```

**Important**: Gradients are squared BEFORE storage, maintaining diagonal Fisher semantics.

**Token normalization**: Contributions are normalized by total active tokens to maintain comparable magnitudes across batches.

**Note on per-token Fisher**: The current approach computes `(1/T) * ||∇L_total||²` where `∇L_total` is the gradient of the summed loss. This is **not** the same as true per-token Fisher `(1/T) * Σ_t ||∇L_t||²` unless the loss is linear in each token. The current method is a computationally efficient approximation that may have bias for non-linear losses.

Critical for:
- Phase 5: Cross-task conflict detection (needs per-sample forensics)
- Phase 6: QK-OV interference (needs full-tensor Fisher computation)

**Key Parameters**:
```python
fisher_collector.compute_fisher_welford_batches(
    model=model,
    batches=batch_list,
    task='math',
    cache_gradients=enable_cross_task_analysis,  # CRITICAL
    show_progress=True
)
```

**What `enable_cross_task_analysis=True` triggers**:
```python
# In BombshellMetrics.compute_fisher_welford_batches():
micro_batch_size = 1 if (cache_gradients or enable_cross_task_analysis) else 10

# This enables:
# 1. Per-sample processing (micro_batch_size=1)
# 2. contribution_cache storage (full tensors)
# 3. gradient_manager storage (per-sample gradients)
```

**Trade-offs**:

| Mode | Memory | Speed | Enables |
|------|--------|-------|---------|
| `enable_cross_task_analysis=False` | Low | Fast | Phases 2-4 only |
| `enable_cross_task_analysis=True` | High | Slow | All 7 phases |

**Output**:
- `fisher_ema`: Dict[str, Tensor] - Group-reduced Fisher (all phases)
- `fisher_accumulated`: Dict[str, Dict[str, Tensor]] - Welford means per task
- `fisher_m2`: Dict[str, Dict[str, Tensor]] - Welford M2 (for variance)
- `contribution_cache`: Dict[str, Dict[str, Dict[str, Tensor]]] - Full contributions (Phase 5-6)
- `gradient_manager`: GradientComputationManager - Per-sample gradients (Phase 5)

**Implementation**: `fisher/core/fisher_collector.py`
- Welford update: `FisherCollector.update_fisher_welford()`
- Per-sample storage: `FisherCollector.update_fisher_welford()` (inline)
- Group reduction: `FisherCollector._reduce_to_groups()`

**Use cases**:
- All downstream phases (2-7) depend on Phase 1
- Foundation for all Fisher-based analyses

### Phase 2: Fisher Importance Analysis

**What it does**: Ranks parameters by importance using Fisher Information

**Theoretical Foundation**:

Fisher diagonal F_ii measures parameter sensitivity:
```
F_ii = E_x[(∂L(x,θ)/∂θ_i)²]
```

**Interpretation**:
- **High F_ii**: Parameter θ_i is critical for task performance
  - Small changes cause large loss increases
  - Should be preserved during pruning/merging
  
- **Low F_ii**: Parameter θ_i is less important
  - Can be pruned or merged safely
  - Loss is insensitive to this parameter

**Normalization**:
```python
# Per-layer normalization (makes layers comparable)
importance[layer] = F[layer] / sum(F[layer])

# Global normalization (makes all parameters comparable)
importance_global = F / sum(F)
```

**Uses**: `fisher_ema` (group-reduced)

**Output**:
- Parameter importance rankings (sorted by F_ii)
- Layer-wise importance aggregation
- Task comparison metrics
- Top-k critical parameters

**Implementation**: `BombshellMetrics.py` (inherits from `FisherCollector`)
- Gets group Fisher: `FisherCollector.get_group_fisher()`
- Per-layer organization: `BombshellMetrics.compute_fisher_importance()`
- Global normalization: `BombshellMetrics.compute_fisher_importance()`

**Use cases**:
- **Pruning**: Remove parameters with lowest F_ii
- **Merging**: Weight task vectors by F_ii
- **Continual learning**: Protect high-F_ii parameters (EWC)

### Phase 3: Pruning Mask Generation

**What it does**: Creates binary masks for Fisher-based pruning

**Theoretical Foundation**:

**Pruning criterion**: Remove parameters with lowest Fisher importance

```
mask_i = 1 if F_ii > threshold else 0
```

where threshold is determined by target sparsity.

**Why Fisher for pruning?**

1. **Optimal Quadratic Approximation** (LeCun et al. 1990):
   - Loss change under parameter deletion: ΔL ≈ (1/2) F_ii (Δθ_i)²
   - Setting θ_i = 0 minimizes loss increase when F_ii is small

2. **Second-Order Information**:
   - First-order (gradient): Where to move parameters
   - Second-order (Fisher): How sensitive is loss to movement
   - Fisher captures curvature of loss landscape

3. **Task-Specific Importance**:
   - Different tasks care about different parameters
   - Fisher reflects task-specific sensitivity

**Structured vs Unstructured Pruning**:

```python
# Unstructured: Prune individual weights
mask[i] = 1 if F_ii > threshold

# Structured: Prune entire groups (heads, channels)
mask[group] = 1 if mean(F[group]) > threshold
```

**Uses**: `fisher_ema` (group-reduced)

**Sparsity Levels**:
- 50%: Mild pruning (keeps most capacity)
- 70%: Moderate pruning (balanced)
- 90%: Aggressive pruning (may hurt performance)

**Output**:
- Binary masks per task: {param_name → {0,1}ⁿ}
- Configurable sparsity levels
- Per-layer sparsity statistics

**Implementation**: `BombshellMetrics.py`
- Quantile thresholding: `BombshellMetrics.get_fisher_pruning_masks()`
- Structured pruning: `BombshellMetrics.get_fisher_pruning_masks()`
- Unstructured pruning: `BombshellMetrics.get_fisher_pruning_masks()`

**Use cases**:
- **Model compression**: Reduce model size
- **Efficient inference**: Skip masked computations
- **Task-specific sub-networks**: Lottery ticket hypothesis

### Phase 4: Mask Overlap Analysis

**What it does**: Quantifies parameter sharing and conflicts between tasks

**Theoretical Foundation**:

**Overlap metric**:
```
overlap(A, B) = |mask_A ∩ mask_B| / |mask_A ∪ mask_B|
```

**Interpretation**:
- **High overlap (> 70%)**: Tasks share important parameters
  - Risk: Interference during multi-task learning
  - Solution: Careful learning rate scheduling
  
- **Medium overlap (40-70%)**: Partial parameter sharing
  - Some shared representations, some task-specific
  - Typical for related tasks
  
- **Low overlap (< 40%)**: Tasks use different parameters
  - Lower interference risk
  - Safe for task arithmetic / model merging

**Conflict Detection**:

```python
# Parameters important to both tasks = potential conflicts
conflicts = mask_A & mask_B

# Parameters unique to each task = safe
unique_A = mask_A & ~mask_B
unique_B = mask_B & ~mask_A
```

**Merge Recommendations**:

Based on overlap percentage:
- **> 70%**: High conflict → Use gradient surgery or task-specific adapters
- **40-70%**: Moderate conflict → Task arithmetic with careful weight selection
- **< 40%**: Low conflict → Safe to merge with task arithmetic

**Uses**: Masks from Phase 3

**Output**:
- Overlap percentage: scalar in [0, 1]
- Conflict regions: {param_name → conflict_score}
- Per-layer overlap analysis
- Merge recommendations: actionable strategies

**Implementation**: `BombshellMetrics.py`
- Mask intersection: `BombshellMetrics.compute_fisher_overlap()`
- Overlap computation: `BombshellMetrics.compute_fisher_overlap()`

**Use cases**:
- **Model merging**: Predict interference before merging
- **Multi-task learning**: Design task-specific adapters for high-conflict regions
- **Task arithmetic**: Weight task vectors by (1 - overlap)

### Phase 5: Cross-Task Sample Conflict Detection

**What it does**: Sample-level conflict forensics for multi-task learning

**Theoretical Foundation**:

**Key Insight**: Sample heterogeneity matters (Katharopoulos & Fleuret 2018)
- Within-task variance can exceed between-task variance
- Task A contains both "easy" and "hard" samples
- Hard samples from different tasks may conflict more than easy samples
- Batch averaging hides these sample-specific patterns

**Conflict Score** (Cosine similarity in natural gradient space):
```
# Step 1: Transform to natural gradient space (optional)
g̃_A = g_A / sqrt(F + ε)
g̃_B = g_B / sqrt(F + ε)

# Step 2: Compute cosine similarity
conflict(s_i, s_j) = cos(g̃_A, g̃_B) = (g̃_A · g̃_B) / (||g̃_A|| ||g̃_B||)
```

Where:
- s_i, s_j: Samples from tasks A and B
- g_A, g_B: Per-sample gradients ∇L(s_i, θ), ∇L(s_j, θ)
- F: Fisher information matrix (diagonal approximation)
- ε: Numerical stability (1e-8)
- Result: -1 (opposing) to +1 (aligned)

**Why Fisher-weighted (natural gradient space)?**
- Raw cosine similarity in Euclidean space can be misleading
- Fisher weighting accounts for parameter importance
- Natural gradient space = geometry-aware (Martens & Grosse 2015)
- Parameters with high Fisher weighted more in conflict detection
- More meaningful than unweighted gradient comparison

**Implementation** (from `cross_task_conflict_detector.py:172-233`):
```python
# Optional: Transform to natural gradient space
fisher_sqrt = torch.sqrt(fisher_matrix + 1e-8)
grad_a = grad_a / fisher_sqrt
grad_b = grad_b / fisher_sqrt

# Compute cosine similarity
cosine = F.cosine_similarity(grad_a.flatten(), grad_b.flatten())
```

**Statistical Significance**:

1. **Effect Size** (Vector effect magnitude):
   ```
   d = ||g̃_A - g̃_B||₂ / 2
   ```
   where g̃_A = g_A / ||g_A||₂, g̃_B = g_B / ||g_B||₂

   - d = 0.2: Small effect (subtle)
   - d = 0.5: Medium effect (visible)
   - d = 0.8: Large effect (severe)

   **Note**: This is L2 difference of unit-norm vectors, not standard Cohen's d (which uses pooled variance per component).

2. **Multiple Testing Correction**:
   - **FDR correction** (Benjamini-Hochberg): Controls false discovery rate
   - Proper p-values for thousands of comparisons
   - Bootstrap resampling for non-parametric tests

3. **Hypothesis Testing**:
   - H₀: Conflict score from random gradient fluctuations
   - H₁: Samples genuinely conflict (p < 0.05 with FDR)

**Forensic Claims Example**:
> "Sample 7 from Task A conflicts with Sample 23 from Task B on layer_3.qkv (p<0.001, d=0.73)"

**Uses**: `contribution_cache` + `gradient_manager` (full tensors)

**Requirements**:
- ⚠️ **REQUIRES** `enable_cross_task_analysis=True`

**Output**:
- Top conflicting sample pairs with statistical significance
- Conflict clusters (groups of mutually conflicting samples)
- Per-parameter conflict scores
- Forensic claims: "Sample X conflicts with Sample Y on parameter P (p<α)"
- Actionable recommendations:
  - Which samples to filter (conflict-based curriculum)
  - Which samples to reweight
  - Which model components are conflict hotspots

**Use cases**:
- **Curriculum learning**: Remove highly conflicting samples
- **Sample reweighting**: Down-weight conflicting pairs
- **Data augmentation**: Generate less-conflicting variants
- **Circuit analysis**: Identify which components cause interference

**Implementation**:
- `fisher/core/cross_task_conflict_detector.py`
- Conflict computation: `CrossTaskConflictDetector.compute_gradient_conflict()`
- Effect size: `CrossTaskConflictDetector.compute_gradient_conflict()`
- Statistical testing: `CrossTaskConflictDetector.apply_fdr_correction()`
- Main detection loop: `CrossTaskConflictDetector.detect_conflicts()`

**References**:
- Yu et al. (2020): "Gradient Surgery for Multi-Task Learning" (task-level)
- Katharopoulos & Fleuret (2018): "Not All Samples Are Created Equal"
- Martens & Grosse (2015): "Optimizing Neural Networks with KFAC"

### Phase 6: QK-OV Circuit-Level Interference

**What it does**: Analyzes interference at attention circuit level

**Uses**: `contribution_cache` (full tensors)

**How it works**:
```python
# 1. Get per-sample contribution for sample i
C_i = contribution_cache[task_a][f"{task_a}_{i}"][param_name]  # [4096, 4096]

# 2. Compute Fisher from all contributions
fisher = (1/n) Σ contribution_cache[task_a][f"{task_a}_{j}"][param_name]  # [4096, 4096]

# 3. Get gradient for sample j
g_j = gradient_manager.get_sample_gradients(task_b, j)[param_name]  # [4096, 4096]

# 4. Slice by QK-OV block and head
C_i_qkv = indexer.slice_tensor(C_i, layer, head, block='Q', param_name)
I_n_qkv = indexer.slice_tensor(fisher, layer, head, block='Q', param_name)
g_j_qkv = indexer.slice_tensor(g_j, layer, head, block='Q', param_name)

# 5. Compute interference score
M_ij = ⟨C_i_qkv / I_n_qkv, |g_j_qkv|⟩
```

**Requirements**:
- ⚠️ **REQUIRES** `enable_cross_task_analysis=True`
- ⚠️ **REQUIRES** Phase 1 to store full tensors in `contribution_cache`

**Output**:
- Interference scores per (Q, K, V, O) block
- Heatmaps across layers and heads
- Top conflicting block-head pairs
- GQA mapping: `kv_head = head // (H // H_kv)` included in heatmap JSON
- Gradient normalization info: `normalize_gj` setting used

**Numerical Health Diagnostics** (included in outputs):
- `min(Î)`, `median(Î)`, `max(Î)`: Fisher spectrum per slice
- `% below ε`: Percentage of parameters near zero Fisher
- `ridge_lambda_applied`: Actual ridge value after clamping
- `ridge_lambda_rel` (if using relative ridge)
- These help diagnose ill-conditioning and guide hyperparameter tuning

**Scale Invariance Note**: The default metric M scales linearly with loss scale; keep loss normalization fixed across tasks or use the natural-gradient variant (divide g_j by √Î) for scale invariance.

**Empty Task Guard**: If task_b has zero stored gradients, the metric returns zeros with an explicit warning message instead of taking means over empty axes.

**Key Innovation**: Uses **contribution-based Fisher** instead of `fisher_ema`:
```python
# OLD (broken): Used fisher_ema [16] - wrong shape!
# fisher = fisher_ema[param_name]  # ❌ Shape mismatch

# NEW (correct): Compute from contributions [4096, 4096]
fisher = _compute_fisher_from_contributions(task, param_name)  # ✅ Correct shape
```

### Phase 7: Fisher-Scaled Gradients (Optional)

**What it does**: Scales gradients by Fisher information for natural gradient descent

**Theoretical Foundation**:

**Natural Gradient** (Amari 1998):

Standard gradient descent:
```
θ_{t+1} = θ_t - η ∇L(θ_t)
```

Natural gradient descent:
```
θ_{t+1} = θ_t - η F^{-1} ∇L(θ_t)
```

Where F is the Fisher information matrix.

**Why natural gradients?**

1. **Parameter Space Invariance**:
   - Standard gradients depend on parameterization
   - Natural gradients are invariant to reparameterization
   - Follows steepest descent in distribution space (not parameter space)

2. **Adaptive Step Sizes**:
   - Large steps in flat directions (low Fisher)
   - Small steps in steep directions (high Fisher)
   - Automatically adapts to local geometry

3. **Faster Convergence** (Martens & Grosse 2015):
   - Natural gradients approximate Newton's method
   - Second-order optimization without full Hessian
   - Particularly effective for deep networks

**Diagonal Approximation**:

```
g_scaled = g / (F_diag + ε)
```

Where:
- F_diag: Diagonal Fisher from Phase 1
- ε: Numerical stability (prevents division by zero)

**Trade-off**:
- Full Fisher: O(p²) memory and computation
- Diagonal Fisher: O(p) memory, approximation
- KFAC: O(p) memory, better approximation

**Uses**: `fisher_accumulated` (from Phase 1) + fresh gradients

**When it runs**: Only if `'scale_by_fisher'` in metrics to compute

**Output**:
- Fisher-scaled gradients per parameter
- Uses Welford-accumulated Fisher for consistency
- Can be used directly for parameter updates

**Use cases**: 
- **Natural gradient descent**: Replace standard gradients
- **Task arithmetic**: Fisher-weight task vectors
- **Model merging**: Importance-weighted parameter averaging
- **Fine-tuning**: Adaptive learning rates per parameter

**Note**: This is an optional metric, not part of the core 6-phase pipeline

**Implementation**: `BombshellMetrics.py`
- Formula: `BombshellMetrics.scale_by_fisher()`
- Diagonal scaling: `BombshellMetrics.scale_by_fisher()`
- KFAC support: `BombshellMetrics.scale_by_fisher()`

**References**:
- Amari (1998): "Natural Gradient Works Efficiently in Learning"
- Martens & Grosse (2015): "Optimizing Neural Networks with KFAC"

---

### Fisher Method Comparison (Within Phase 1)

**When it runs**: During Phase 1, as diagnostics

**Methods compared**:
- **Group Fisher** (default): Parameter grouping by heads/channels
- **KFAC**: Kronecker-factored approximation
- **Lanczos**: Top-k eigenvalue approximation  
- **Spectral**: Block-diagonal Fisher

**Output**: Comparison table with:
- Computation time
- Memory usage
- Top eigenvalues (for spectral methods)
- Condition numbers
- Recommendation for best method

**Purpose**: Helps choose the right Fisher method for your use case

---

## Configuration Guide

### FisherCollector Configuration

```python
from fisher.core.fisher_collector import FisherCollector

fisher_collector = FisherCollector(
    # ═══════════════════════════════════════════════════════════
    # BASIC SETTINGS
    # ═══════════════════════════════════════════════════════════
    
    reduction='group',  # 'param', 'group', 'block'
    # - 'param': No reduction (full parameters)
    # - 'group': Reduce to groups (heads, channels)  ← RECOMMENDED
    # - 'block': Reduce to coarse blocks
    
    # ═══════════════════════════════════════════════════════════
    # MEMORY SETTINGS
    # ═══════════════════════════════════════════════════════════
    
    storage_device='cpu',  # 'cpu' or 'cuda'
    storage_dtype='float16',  # 'float16' or 'float32'
    # Tip: 'cpu'+'float16' reduces memory by 4x vs 'cuda'+'float32'
    
    # ═══════════════════════════════════════════════════════════
    # NUMERICAL SETTINGS
    # ═══════════════════════════════════════════════════════════
    
    computation_dtype='float32',  # Always use float32 for gradients
    ema_decay=0.99,  # EMA decay factor (0.99 = slow decay)
    
    # ═══════════════════════════════════════════════════════════
    # CROSS-TASK ANALYSIS (PHASES 5 & 6)
    # ═══════════════════════════════════════════════════════════
    
    enable_cross_task_analysis=True,  # ⚠️ REQUIRED for Phase 5 & 6
    # When True:
    # - Uses micro_batch_size=1 (per-sample processing)
    # - Stores full tensors in contribution_cache
    # - Enables gradient_manager
    
    gradient_memory_mb=50,  # Memory budget for gradient storage
    # Recommended: 50 MB for 1.5B models, 100 MB for 7B models
    
    # ═══════════════════════════════════════════════════════════
    # ADVANCED FEATURES
    # ═══════════════════════════════════════════════════════════
    
    use_ewc=False,  # EWC regularization
    n_bootstrap_samples=1000,  # Uncertainty quantification
    min_conflict_effect_size=0.01  # Conflict detection threshold
)
```

### UnifiedConfig Configuration

```python
from unified_model_analysis import UnifiedConfig

config = UnifiedConfig(
    # ═══════════════════════════════════════════════════════════
    # MODEL & DATA
    # ═══════════════════════════════════════════════════════════
    
    model_name="Qwen/Qwen2.5-Math-1.5B-Instruct",
    max_samples_per_dataset=1000,
    batch_size=4,  # Inference batch size
    
    # ═══════════════════════════════════════════════════════════
    # FISHER COMPUTATION
    # ═══════════════════════════════════════════════════════════
    
    compute_fisher=True,
    fisher_batch_size=128,  # Batch size for Fisher (not inference!)
    fisher_n_samples=1000,  # Total samples for Fisher estimation
    
    # ⚠️ CRITICAL for Phase 5 & 6
    enable_cross_task_analysis=True,
    
    # ═══════════════════════════════════════════════════════════
    # ADVANCED FISHER FEATURES
    # ═══════════════════════════════════════════════════════════
    
    compute_fisher_eigenvalues=True,  # Method comparison
    enable_cross_task_conflicts=True,  # Phase 5
    enable_qkov_interference=True,  # Phase 6
    
    # ═══════════════════════════════════════════════════════════
    # REPRODUCIBILITY
    # ═══════════════════════════════════════════════════════════
    # For deterministic results:
    # 1. model.eval() - disable dropout/DropPath
    # 2. Set seeds - torch, numpy, Python random
    # 3. torch.use_deterministic_algorithms(True) - optional
    # 4. Set CUBLAS_WORKSPACE_CONFIG=":16:8" - if using deterministic algos
    # 5. Fix dataloader shuffling - set shuffle=False or use fixed seed
    # 6. Set torch.backends.cudnn.benchmark=False - for determinism
    
    seed=42,
    verbose=True
)
```

### Memory vs Feature Trade-offs

| Configuration | Memory | Phase 1 | Phase 2-4 | Phase 5 | Phase 6 |
|---------------|--------|---------|-----------|---------|---------|
| **Minimal** | ~1 GB | ✅ | ✅ | ❌ | ❌ |
| `enable_cross_task_analysis=False` | | Group only | ✅ | No data | No data |
| **Standard** | ~5 GB | ✅ | ✅ | ✅ | ✅ |
| `enable_cross_task_analysis=True` | | Group + Full | ✅ | ✅ | ✅ |
| **Full Analysis** | ~10 GB | ✅ | ✅ | ✅ | ✅ |
| + eigenvalues, bootstrapping | | All methods | ✅ | ✅ | ✅ |

**Recommendation**: Always use `enable_cross_task_analysis=True` for research. The memory cost is worth it for Phases 5 & 6.

---

## Fisher Methods Comparison

### Method Overview

| Method | Purpose | Memory | Compute | Output Shape | Use Case |
|--------|---------|--------|---------|--------------|----------|
| **Group Fisher** | Parameter importance | Low | Fast | [groups] | ✅ Primary (all phases) |
| **KFAC** | Natural gradient | Medium | Medium | Block-diagonal | Second-order optimization |
| **Lanczos** | Eigenspectrum | Low | Slow | [k eigenvalues] | Landscape analysis |
| **Spectral** | Full block-diagonal | High | Medium | Block matrices | Capacity analysis |

### When to Use Each Method

**Group Fisher** (default):
```python
# Automatic in unified analysis
results = analyze_models([model], config)
fisher = results['fisher_importance']
```

**KFAC** (natural gradient descent):
```python
# Enable in config
config.compute_fisher_eigenvalues = True
results = analyze_models([model], config)
kfac_fisher = results['fisher_method_comparison']['kfac']
```

**Lanczos** (eigenvalue analysis):
```python
# For optimization landscape
config.compute_fisher_eigenvalues = True
results = analyze_models([model], config)
eigenvalues = results['fisher_method_comparison']['lanczos']['top_eigenvalues']
```

---

## Behavioral Head Categorization

### Overview

**Traditional grouping** (by structure):
```
'model.layers.0.self_attn.q_proj.weight|head'  # All heads treated the same
```

**Behavioral grouping** (by function):
```
'model.layers.0.self_attn.q_proj.weight|induction_heads'  # Grouped by behavior
'model.layers.0.self_attn.q_proj.weight|positional_heads'
'model.layers.0.self_attn.q_proj.weight|content_heads'
```

### Head Types

| Type | Behavior | QK Pattern | OV Pattern |
|------|----------|------------|------------|
| **Induction** | Copy & repeat | Prev token + offset | Copy value |
| **Positional** | Position-based | Attend to positions | Position encoding |
| **Previous token** | Look back | Attend to i-1 | Copy previous |
| **Same token** | Self-attention | Attend to self | Identity |
| **Content** | Content-based | Semantic similarity | Content transform |
| **Mixed** | Multiple behaviors | Varies | Varies |

### Integration with Fisher

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

### Benefits for Phase 6 (QK-OV)

**Without behavioral categorization**:
```
Phase 6 reports: "Head 5 in layer 3 has high Q interference"
→ Hard to interpret
```

**With behavioral categorization**:
```
Phase 6 reports: "Induction heads in layer 3 have high Q interference"
→ Immediately interpretable: induction mechanism conflicts!
```

---

## QK-OV Integration (Phase 6)

### Overview

Phase 6 provides **circuit-level interference analysis** using the attention mechanism taxonomy:

- **Q (Query)**: What to attend to
- **K (Key)**: What to match against
- **V (Value)**: What information to retrieve
- **O (Output)**: How to project back

### Theoretical Formula

For sample i from task A and sample j from task B:

```
M^B_{ij,ℓ,h} = Σ_{k ∈ B,ℓ,h} (C_{i,k} / (Î_{n,k} + ε)) · |g_{j,k}|
             = Σ_k (g²_{i,k} / (Î_k + ε)) · |g_{j,k}|
```

Where:
- B ∈ {Q, K, V, O}: Block type
- ℓ: Layer index
- h: Head index
- k: Parameter index within block/head slice
- C_{i,k} = g²_{i,k}: Per-sample contribution (squared gradient)
- Î_{n,k} ≈ (1/n) Σ_j C_{j,k}: Diagonal Fisher estimate
- |g_{j,k}|: Magnitude of task B gradient (unsigned)

### Metric Design (Important!)

**This metric is diagonal-Fisher, asymmetric, magnitude-weighted:**

#### 1. Diagonal Fisher (Not Full Covariance)

**What it uses**: Elementwise squares `C_i = g²_i`

**What it doesn't use**: Full covariance `g_i g_i^T`

**Why**: 
- ✅ Computationally tractable: O(p) memory vs O(p²) for full Fisher
- ✅ Standard approach: Used in EWC, continual learning, model merging
- ⚠️ Limitation: Ignores cross-parameter covariance within a block

**Implication**: Cross-parameter interactions are not captured. If two parameters always move together, this metric treats them independently.

#### 2. Asymmetric (Directional)

**Property**: M_ij ≠ M_ji

**Why**: Uses C_i for normalization but |g_j| for magnitude

**Interpretation**: Measures "how sample i from task A stresses parameters when sample j from task B updates them"

**Use case**: Directional interference detection (i interferes with j's update)

**Alternative**: For symmetric analysis (M_ij = M_ji), see signed/symmetric variants below

#### 3. Magnitude-Weighted (Unsigned)

**What it uses**: `|g_j|` (absolute value of gradients)

**What it doesn't capture**: Sign of interference (synergy vs conflict)

**Why**: Focuses on interference **strength**, not direction

**Implication**:
- ✅ Good for pruning/merging: Both positive and negative interference are problematic
- ⚠️ Misses synergy: Can't distinguish constructive (+) from destructive (-) interference

**Alternative**: For signed interference (captures ±conflict), see signed variant below

#### 4. Task-A Normalized

**Normalization**: Î computed from task A contributions only

**Why**: Reflects task A's parameter sensitivity distribution

**Implication**: Asymmetric by construction (task A vs task B)

**Alternative**: For symmetric normalization, pool Fisher from both tasks: `Î = (Î_A + Î_B) / 2`

### Alternative Metrics (For Ablation Studies)

#### Configuration API (Switching Variants)

```python
# In QKOVInterferenceMetric initialization
metric = QKOVInterferenceMetric(
    fisher_collector=bombshell,
    epsilon=1e-12,
    ridge_lambda=1e-12,
    ridge_lambda_rel=1e-6,  # Relative: λ = max(λ_min, λ_rel * median(Î))

    # Aggregation: 'mean' makes heads/blocks comparable regardless of size
    # Set 'sum' to recover raw totals
    aggregate='mean',  # or 'sum'

    # Gradient normalization for g_j (task B gradients)
    # 'none': No normalization (raw gradients)
    # 'per_token_mean': Divide by sequence length (sequence-invariant)
    # 'per_token_sum': Divide by total tokens (token-invariant)
    normalize_gj='none',  # 'none' | 'per_token_mean' | 'per_token_sum'

    # For ablation studies (default uses unsigned, asymmetric, diagonal)
    # variant='unsigned'  # 'signed' | 'symmetric'
    # fisher_pool='A'     # 'A' | 'B' | 'pooled'
)
```

**Aggregation Note**: By default we report the **mean** over the block/head slice to make heads and blocks comparable regardless of parameter count. Set `aggregate='sum'` to recover raw totals.

**GQA Mapping**: When pooling K/V heads for Grouped Query Attention, note that head h maps to kv-head floor(h / (H / H_kv)).

**Caching Costs**: A single full Fisher per (task, param) at 4096×4096 fp32 ≈ **67 MB**. For memory efficiency, consider: (i) caching **diagonals only** for non-sliced params, (ii) **compute-on-slice**: slice contributions then average, never materialize full Fisher for params you won't slice.

#### Signed Variant (Captures Synergy vs Conflict)

```python
M^B_{ij,ℓ,h} = Σ_k (g_{i,k} · g_{j,k}) / (Î_k + ε)
```

**Differences**:
- Uses **signed** gradients (not absolute value)
- Returns **positive** (synergy) or **negative** (conflict) scores

**Use case**: Understanding whether samples help or hinder each other

**Implementation note**: Requires storing signed per-sample gradients (not just squared contributions)

#### Symmetric Variant (M_ij = M_ji)

```python
M^B_{ij,ℓ,h} = Σ_k sqrt(C_{i,k} · C_{j,k}) / (Î_k + ε)
```

**Differences**:
- Treats both samples symmetrically
- M_ij = M_ji by construction

**Use case**: Measuring mutual interference or task similarity

#### Full Fisher Variant (Captures Covariance)

```python
M^B_{ij,ℓ,h} = g_i^T F^{-1} g_j
```

**Differences**:
- Uses full Fisher matrix (not diagonal)
- Captures cross-parameter covariance

**Challenge**: Requires O(p²) memory or KFAC/low-rank approximation

**Use case**: When parameter interactions are critical

### Implementation

```python
from fisher.qkov import QKOVConfig, QKOVInterferenceMetric

# 1. Ensure Phase 1 ran with enable_cross_task_analysis=True
config.enable_cross_task_analysis = True
results = analyze_models([model], config)

# 2. Create QK-OV metric
qkov_config = QKOVConfig.from_model(model)
metric = QKOVInterferenceMetric(
    config=qkov_config,
    fisher_collector=fisher_collector,
    epsilon=1e-10,
    ridge_lambda=1e-8,
    ridge_lambda_rel=1e-6,  # Relative ridge: λ = max(λ_min, λ_rel * median(Î))
    min_samples_for_fisher=10,  # Minimum samples for stable Fisher
    normalize_gj='none'  # 'none' | 'per_token_mean' | 'per_token_sum'
)

# 3. Compute interference for a sample pair
scores = metric.compute_sample_pair(
    task_a='math', sample_i=5,
    task_b='code', sample_j=12,
    layer=3, head=4
)
# Returns: {'Q': 0.42, 'K': 0.31, 'V': 0.18, 'O': 0.55}

# 4. Compute full heatmap
heatmap = metric.compute_heatmap(
    task_a='math',
    task_b='code',
    layers=list(range(model.config.num_hidden_layers)),
    heads=list(range(model.config.num_attention_heads)),
    max_samples_per_task=100
)

# 5. Analyze results
for block in ['Q', 'K', 'V', 'O']:
    avg_interference = heatmap[block]['layer_head_avg'].mean()
    print(f"{block} block average interference: {avg_interference:.4f}")
    
    top_conflicts = heatmap[block]['top_conflicts'][:5]
    print(f"Top 5 {block} conflicts:")
    for conf in top_conflicts:
        print(f"  Sample ({conf['sample_i']}, {conf['sample_j']}) "
              f"L{conf['layer']}H{conf['head']}: {conf['score']:.4f}")
```

### Contribution-Based Fisher (Key Innovation)

**Problem**: Phase 1's `fisher_ema` has wrong shape for QK-OV slicing
- `fisher_ema[param]`: Shape `[16]` (group-reduced by head)
- QK-OV needs: Shape `[4096, 4096]` (full parameter tensor)

**Solution**: Compute Fisher from stored contributions

```python
def _compute_fisher_from_contributions(self, task, param_name):
    """
    Compute Fisher as: Î_n ≈ (1/n) Σᵢ C_i
    
    This gives full parameter tensors [4096, 4096] for QK-OV slicing.
    """
    # Access nested schema: contribution_cache[task][sample_key][param_name]
    task_dict = self.fisher_collector.contribution_cache.get(task, {})
    contributions = []
    for sample_key, contribs in task_dict.items():
        if param_name in contribs:
            contributions.append(contribs[param_name])  # [4096, 4096]
    
    if not contributions:
        raise ValueError(f"No contributions for task={task}, param={param_name}")
    
    # Force fp32 on CPU for stable averaging
    contributions_fp32 = [c.detach().to(dtype=torch.float32, device='cpu') 
                          for c in contributions]
    
    # Fisher = mean of contributions (unbiased estimator)
    fisher_full = torch.stack(contributions_fp32, dim=0).mean(dim=0)  # [4096, 4096]
    return fisher_full
```

**Why this works**:
- ✅ Theoretically valid: E[Cᵢ] = diag(F̂) (definition of diagonal Fisher)
- ✅ Correct shapes: Full tensors for QK-OV slicing
- ✅ Q/K/V distinct: Contributions stored before group reduction
- ✅ Efficient: Cached after first computation

### Use Cases

**1. Model Merging**
```python
# Identify which attention mechanisms conflict
heatmap = metric.compute_heatmap('math', 'code')
most_conflicted_block = max(['Q', 'K', 'V', 'O'], 
                             key=lambda b: heatmap[b]['layer_head_avg'].mean())
# → Avoid merging parameters in most_conflicted_block
```

**2. Task Arithmetic**
```python
# Understand where task vectors interfere
interference = metric.compute_heatmap('task_a', 'task_b')
# → Scale down task vectors in high-interference regions
```

**3. Pruning Strategy**
```python
# Preserve non-conflicting important circuits
for block in ['Q', 'K', 'V', 'O']:
    interference = heatmap[block]['layer_head_avg']
    importance = fisher_collector.get_group_fisher('math')
    
    # Keep: high importance + low interference
    keep_mask = (importance > threshold) & (interference < conflict_threshold)
```

---

## Common Issues and Solutions

### Issue 1: "No contributions found for task=..."

**Symptom**:
```
ValueError: No contributions found for task='math', param='model.layers.0.self_attn.q_proj.weight'. 
Ensure Phase 1 ran with enable_cross_task_analysis=True.
```

**Cause**: Phase 1 didn't populate `contribution_cache`

**Solution**:
```python
# Set in config BEFORE running Phase 1
config.enable_cross_task_analysis = True

# Or directly in fisher_collector
fisher_collector = FisherCollector(
    enable_cross_task_analysis=True  # ← CRITICAL
)
```

### Issue 2: "QK-OV interference analysis failed"

**Symptom**: Phase 6 silently fails, log shows:
```
⚠️ QK-OV interference analysis failed: <error message>
```

**Common causes**:

**A. Missing contribution_cache**:
```python
# Fix: Enable cross-task analysis
config.enable_cross_task_analysis = True
```

**B. Insufficient samples**:
```python
# Fix: Increase sample count
config.fisher_n_samples = 1000  # Minimum recommended
```

**C. Architecture not detected**:
```python
# Check QKOVConfig auto-detection
try:
    qkov_config = QKOVConfig.from_model(model)
    print(f"Detected: {qkov_config.num_layers} layers, {qkov_config.num_heads} heads")
except ValueError as e:
    print(f"Auto-detection failed: {e}")
    # Provide manual config
```

### Issue 3: High Memory Usage

**Symptom**: OOM during Phase 1 with `enable_cross_task_analysis=True`

**Solutions**:

**A. Reduce sample count**:
```python
config.fisher_n_samples = 500  # From 1000
```

**B. Limit gradient memory**:
```python
fisher_collector = FisherCollector(
    gradient_memory_mb=25  # From default 50
)
```

**C. Use CPU storage**:
```python
fisher_collector = FisherCollector(
    storage_device='cpu',  # Offload to CPU
    storage_dtype='float16'  # Half precision
)
```

**D. Clear cache between tasks**:
```python
# After Phase 1 for task A, before task B
fisher_collector.contribution_cache.clear()
import gc; gc.collect()
torch.cuda.empty_cache()
```

### Issue 4: Architecture Not Detected

**Symptom**: 
```
Warning: Could not separate heads, falling back to channel reduction
```

**Cause**: Model architecture doesn't match expected patterns

**Solution**: This is **intentional graceful fallback**. The code still works correctly:
- Group Fisher: Falls back to channel/row reduction
- QK-OV: May not work if attention structure is non-standard

For custom architectures, structure may not match expected patterns. This is safe.

### Issue 5: NaN Fisher Values

**Symptom**: Fisher values contain NaN

**Causes & Solutions**:

**A. NaN in model parameters**:
```python
# Check model health
for name, param in model.named_parameters():
    if torch.isnan(param).any():
        print(f"NaN in {name}")
# → Fix model before computing Fisher
```

**B. Out-of-bounds tokens**:
```python
# Validate input data
assert batch['input_ids'].max() < model.config.vocab_size
assert batch['input_ids'].min() >= 0
```

**C. Gradient explosion**:
```python
# Enable gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

---

## Advanced Topics

### Welford Variance for Uncertainty Quantification

Phase 1 computes not just Fisher mean but also variance:

```python
# Access Welford statistics
fisher_mean = fisher_collector.fisher_accumulated[task][param]
fisher_m2 = fisher_collector.fisher_m2[task][param]
n_samples = fisher_collector.n_samples_seen[task]

# Compute standard error
fisher_variance = fisher_m2 / (n_samples - 1)
fisher_std_error = torch.sqrt(fisher_variance / n_samples)

# Confidence interval (95%)
lower_bound = fisher_mean - 1.96 * fisher_std_error
upper_bound = fisher_mean + 1.96 * fisher_std_error
```

### Token-Normalized Fisher

**Problem**: Variable-length sequences have different sample sizes

**Solution**: Token normalization

```python
# In Phase 1 (fisher_collector.py):
total_active_tokens = (batch['attention_mask'].sum() if 'attention_mask' in batch
                       else batch_size * seq_len)

# Normalize contribution by tokens
normalized_contribution = full_contribution / max(1, total_active_tokens)
```

**Effect**: Fisher values are now **per-token** importance, making samples comparable.

**Advanced: True Per-Token Fisher**

For theoretically accurate per-token Fisher (square-then-average per token):

```python
# loss per token
per_tok_loss = F.cross_entropy(logits.view(-1, V), labels.view(-1),
                               reduction='none', ignore_index=-100).view(B, T)

# pick active tokens, loop or vmap over tokens:
# for each active token t: backprop per_tok_loss[b, t], get grad_{b,t}
# then: C_{b,t} = grad_{b,t}**2  ;  C_b = mean_t C_{b,t}
```

The default path uses the cheaper approximation `(1/T) * ||∇L_total||²` which may have bias for non-linear losses.

### Bootstrap Confidence Intervals

For more robust uncertainty estimates:

```python
fisher_collector = FisherCollector(
    n_bootstrap_samples=1000
)

# After Phase 1
bootstrap_ci = fisher_collector.compute_bootstrap_confidence_intervals(task='math')
# Returns: Dict[param_name, (lower, upper)]
```

### Architecture-Specific Optimizations

**For Qwen models** (hidden_size × hidden_size projections):
```python
# Automatic fallback to row-wise reduction
# No configuration needed
```

**For GQA/MQA models** (Grouped-Query Attention):
```python
# Automatic detection from model.config.num_key_value_heads
# Fisher groups K/V by kv_head, Q by head
```

**For GPT-style fused QKV**:
```python
# Automatic detection of c_attn parameter
# Splits into Q, K, V blocks before reduction
```

---

## References

### Key Papers

1. **Fisher Information**:
   - Martens & Grosse (2015): "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"
   - Kirkpatrick et al. (2017): "Overcoming catastrophic forgetting in neural networks" (EWC)

2. **Numerical Stability**:
   - Welford (1962): "Note on a method for calculating corrected sums of squares and products"
   - Chan et al. (1983): "Updating Formulae and a Pairwise Algorithm for Computing Sample Variances"

3. **Task Interference**:
   - Ilharco et al. (2023): "Editing Models with Task Arithmetic"
   - Yadav et al. (2023): "Ties-Merging: Resolving Interference When Merging Models"

4. **Mechanistic Interpretability**:
   - Elhage et al. (2021): "A Mathematical Framework for Transformer Circuits"
   - Olsson et al. (2022): "In-context Learning and Induction Heads"

### Related Documentation

- **Phase 6 Details**: `docs/PHASE6_FIXED_IMPLEMENTATION.md`
- **QK-OV Integration**: `fisher/qkov/README.md`
- **Unified Analysis**: `docs/UNIFIED_MODEL_ANALYSIS_DOCUMENTATION.md`
- **Mechanistic Analysis**: `mechanistic/README.md`
- **Data Flow**: `docs/PHASE1_ALREADY_FEEDS_PHASE6.md`

---

## Sanity Tests for Production Deployment

### Core Correctness Tests

**Head additivity**: Sum of head-slice scores ≈ unsliced block score (within fp tolerance).

**Scale invariance (variant check)**: Multiply loss by (a); unsigned metric scales by (a), natural-gradient variant stays constant.

**Ablation validity**: Zero a head ⇒ its slice scores collapse.

**Symmetry check**: Symmetric variant gives (M_{ij}=M_{ji}).

### Numerical Stability Tests

**Welford stability**: Fisher estimate stable across different batch orders (shuffle invariance).

**Ridge robustness**: Metric scores don't explode for ultra-flat slices (Î ≈ 0).

**Token normalization**: Fisher values comparable across variable-length sequences.

**Memory bounds**: Peak GPU usage stays within 80% of available memory.

---

**Document Version**: 2.0
**Last Updated**: 2025-10-07
**Status**: ✅ Production Ready

**Key Changes from v1.0**:
- Added detailed data flow architecture
- Clarified two storage systems (fisher_ema vs contribution_cache)
- Explained contribution-based Fisher for Phase 6
- Better configuration guide with memory trade-offs
- Expanded Phase 6 (QK-OV) documentation
- Added more troubleshooting examples