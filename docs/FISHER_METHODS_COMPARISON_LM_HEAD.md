# Fisher Methods Comparison for lm_head Analysis

## Status: ✅ GROUP FISHER IS BETTER for Most Use Cases

After analyzing your codebase, **Group Fisher (FisherCollector) is the better choice** for lm_head analysis in most scenarios, NOT K-FAC with true Fisher.

---

## Quick Comparison Table

| Task | Best Method | Why | Second Best |
|------|-------------|-----|-------------|
| **Vocabulary pruning** | 🥇 **Group Fisher** | Per-token importance scores | Magnitude |
| **Fine-tuning with shift** | 🥇 **Group Fisher + Welford** | Tracks distribution changes with variance | K-FAC (periodic) |
| **Task arithmetic** | 🥇 **Group Fisher** | Per-parameter merging weights | K-FAC (block-level) |
| **Variance analysis** | 🥇 **Group Fisher + Welford** | Explicit variance via M2 | True Fisher (lower variance) |
| **Natural gradient optimization** | 🥇 **K-FAC** | Block-diagonal preconditioner | Group Fisher (diagonal) |
| **Loss landscape geometry** | 🥇 **K-FAC + Lanczos** | Captures interactions | FisherSpectral |

---

## What Each Method Actually Computes

### 1. Group Fisher (FisherCollector)

**What it gives you:**
```python
fisher_importance = {
    'lm_head.weight[0]': 0.023,  # Importance of token 0
    'lm_head.weight[1]': 0.891,  # Importance of token 1
    ...
    'lm_head.weight[50256]': 0.003,  # Importance of token 50256
}
```

**Key properties:**
- ✅ **Per-parameter (or per-group) importance scores**
- ✅ **Welford accumulation** → unbiased mean + variance
- ✅ **Token-weighted** → accounts for variable sequence lengths
- ✅ **CPU offloading** → fp16 storage, handles 50k vocab easily
- ✅ **True Fisher support** via `AdvancedFisherCollector`

**What it computes:**
```
F_ii = E_x[(∂L/∂θ_i)²]  (diagonal Fisher)
```

**Storage:** `O(#parameters)` - one value per parameter

---

### 2. K-FAC (KFACNaturalGradient)

**What it gives you:**
```python
kfac_factors = {
    'lm_head': {
        'A': activation_covariance,  # [hidden × hidden]
        'G': gradient_covariance,    # [vocab × vocab] OR Woodbury factors
    }
}
```

**Key properties:**
- ✅ **Block-diagonal approximation** → `F ≈ A ⊗ G`
- ✅ **Captures interactions** within layer (not across parameters)
- ✅ **Natural gradient** → second-order optimization
- ✅ **Woodbury for large vocab** → memory-efficient
- ⚠️ **NOT per-parameter** → block-level only

**What it computes:**
```
F_layer ≈ A ⊗ G
where A = activation covariance, G = gradient covariance
```

**Storage:** 
- Eigendecomp: `O(hidden² + vocab²)` - intractable for lm_head
- Woodbury: `O(vocab·T + T²)` - tractable, but still block-level

---

### 3. Lanczos Spectrum

**What it gives you:**
```python
spectrum = {
    'eigenvalues': [λ₁, λ₂, ..., λₖ],  # Top-k eigenvalues
    'condition_number': κ,
    'spectral_gap': gap
}
```

**Key properties:**
- ✅ **Extreme eigenvalues** → captures sharpest/flattest directions
- ✅ **One-shot** → no accumulation needed
- ✅ **Loss landscape analysis** → optimization diagnostics
- ⚠️ **Not per-parameter** → global curvature only

**Use case:** Understanding optimization landscape, not pruning

---

### 4. FisherSpectral (Block-Diagonal)

**What it gives you:**
```python
block_spectrum = {
    'lm_head': {
        'eigenvalues': [...],  # Per-block eigenvalues
        'capacity': capacity_estimate
    }
}
```

**Key properties:**
- ✅ **Per-block capacity** → model expressiveness
- ✅ **Cheap to compute** → per-layer eigendecomposition
- ⚠️ **Not per-parameter** → block-level only

**Use case:** Model capacity analysis, not pruning

---

## Use Case Analysis

### ✅ Use Case 1: Vocabulary Pruning (Remove Rare Tokens)

**Goal:** Identify which lm_head tokens can be pruned with minimal loss impact.

**Group Fisher wins:**
```python
from BombshellMetrics import BombshellMetrics

# Collect Fisher for lm_head
metrics = BombshellMetrics(reduction='param')  # Per-parameter
metrics.update_fisher_welford(model, batch, task='pretrain')

# Get pruning masks
masks = metrics.get_fisher_pruning_masks(
    task='pretrain',
    sparsity=0.1,  # Prune 10% least important tokens
    structured=True
)

# Prune lm_head based on importance
for name, mask in masks.items():
    if 'lm_head' in name:
        param = dict(model.named_parameters())[name]
        param.data *= mask  # Zero out unimportant tokens
```

**Why Group Fisher:**
- ✅ Gives importance **per vocabulary token** (what you need)
- ✅ Accumulated over many batches (reduces variance naturally)
- ✅ Welford variance tells you confidence in each importance score
- ✅ Already implemented and tested in your codebase

**Why NOT K-FAC:**
- ❌ Gives block-level `F ≈ A ⊗ G`, not per-token importance
- ❌ To extract per-token from K-FAC: `importance_i = diag(G)` (loses A information)
- ❌ More complex, same result as Group Fisher diagonal

**Why NOT K-FAC true Fisher:**
- ❌ Still block-level, not per-token
- ❌ 2× forward compute for negligible benefit (Group Fisher already accumulates)
- ❌ Over-engineering for this use case

---

### ✅ Use Case 2: Fine-Tuning with Distribution Shift

**Scenario:** Pre-train on web text, fine-tune on scientific papers. Need adaptive importance scores.

**Group Fisher + Welford wins:**
```python
# Pre-training phase
for batch in pretrain_loader:
    metrics.update_fisher_welford(model, batch, task='pretrain')

# Fine-tuning phase
for batch in scientific_loader:
    metrics.update_fisher_welford(model, batch, task='scientific')

# Compare Fisher distributions
fisher_pretrain = metrics.fisher_accumulated['pretrain']['lm_head.weight']
fisher_scientific = metrics.fisher_accumulated['scientific']['lm_head.weight']

# Identify tokens that became more important
importance_shift = fisher_scientific - fisher_pretrain

# High positive shift → scientific tokens
scientific_tokens = (importance_shift > threshold).nonzero()
```

**Why Group Fisher + Welford:**
- ✅ Tracks **per-token importance over time**
- ✅ Welford gives **variance** → confidence intervals for shifts
- ✅ Can detect when distribution changes (variance spikes)
- ✅ Unbiased accumulation (equal weight to all data)

**Why NOT K-FAC:**
- ⚠️ Periodic refresh (every N steps) → misses fine-grained changes
- ⚠️ Block-level → can't identify specific tokens that shifted
- ⚠️ No variance tracking (K-FAC doesn't use Welford)

**Why NOT K-FAC true Fisher:**
- ⚠️ 2× forward compute hurts fine-tuning throughput
- ⚠️ Lower variance per-sample, but Welford over many samples already achieves this
- ⚠️ Still block-level, not per-token

---

### ✅ Use Case 3: Task Arithmetic (Merge Models)

**Goal:** Merge pre-trained models using Fisher-weighted averaging.

**Group Fisher wins:**
```python
# Compute Fisher for each model
fisher_model1 = metrics.fisher_accumulated['model1']
fisher_model2 = metrics.fisher_accumulated['model2']

# Fisher-weighted merge
for name in model1.state_dict():
    if 'lm_head' in name:
        w1 = model1.state_dict()[name]
        w2 = model2.state_dict()[name]
        
        f1 = fisher_model1[name]
        f2 = fisher_model2[name]
        
        # Weight by importance
        weight = f1 / (f1 + f2 + 1e-8)
        merged = weight * w1 + (1 - weight) * w2
```

**Why Group Fisher:**
- ✅ Per-parameter merging weights
- ✅ Simple, interpretable
- ✅ Standard practice in task arithmetic literature

**Why NOT K-FAC:**
- ⚠️ Block-level → can't weight individual tokens differently
- ⚠️ Would need to extract diagonal from `A ⊗ G` (loses structure)

---

### ✅ Use Case 4: Variance-Critical Applications (Safety, Medical)

**Goal:** Conservative updates where variance matters.

**Group Fisher + Welford wins:**
```python
# Get Fisher with confidence intervals
fisher_mean = metrics.fisher_accumulated['task']['lm_head.weight']
fisher_variance = metrics.fisher_variance['task']['lm_head.weight']

# Compute 95% confidence interval for importance
std = torch.sqrt(fisher_variance)
lower = fisher_mean - 1.96 * std
upper = fisher_mean + 1.96 * std

# Conservative pruning: only prune tokens with high-confidence low importance
safe_to_prune = (upper < pruning_threshold)
```

**Why Group Fisher + Welford:**
- ✅ **Explicit variance** via M2 statistic
- ✅ Confidence intervals for importance scores
- ✅ Can quantify uncertainty in pruning decisions

**Why NOT K-FAC true Fisher:**
- ⚠️ True Fisher has lower **per-sample** variance
- ⚠️ But Welford over many samples achieves same effect
- ⚠️ True Fisher doesn't give you **variance estimates** (just lower variance)
- ⚠️ Still block-level, not per-parameter

---

## When K-FAC IS Better

### ✅ Use Case 5: Natural Gradient Optimization

**Goal:** Second-order optimization for faster convergence.

**K-FAC wins:**
```python
kfac = KFACNaturalGradient(damping=1e-4, update_freq=10)

# Collect factors
kfac.collect_kfac_factors(model, batch)

# Compute natural gradient
nat_grad = kfac.compute_natural_gradient(gradients, model)

# Apply preconditioned gradient
for name, param in model.named_parameters():
    if name in nat_grad:
        param.data -= lr * nat_grad[name]
```

**Why K-FAC:**
- ✅ Block-diagonal preconditioner `F^{-1}` 
- ✅ Captures **interactions** within layer
- ✅ Much faster than full second-order methods
- ✅ Standard for natural gradient optimization

**Why NOT Group Fisher:**
- ❌ Diagonal approximation → misses interactions
- ❌ Natural gradient: `F^{-1} ∇` needs inversion, not just diagonal
- ❌ Would be equivalent to RMSProp/Adam

---

### ✅ Use Case 6: Loss Landscape Geometry

**Goal:** Understand local curvature around parameters.

**K-FAC + Lanczos wins:**
```python
# Get extreme eigenvalues of Fisher
spectrum = lanczos_spectrum(model, batch, operator='fisher', k=20)

# Analyze landscape
max_eigenvalue = spectrum['eigenvalues'][0]  # Sharpness
condition_number = spectrum['condition_number']  # Ill-conditioning
spectral_gap = spectrum['spectral_gap']  # Discrete vs continuous spectrum
```

**Why K-FAC + Lanczos:**
- ✅ Captures **interactions** (not just diagonal)
- ✅ Eigenspectrum reveals landscape geometry
- ✅ Used in optimization analysis papers

**Why NOT Group Fisher:**
- ❌ Diagonal → misses off-diagonal structure
- ❌ Can't compute true eigenspectrum (diagonal ≠ eigendecomp)

---

## The K-FAC True Fisher Question

### When True Fisher for lm_head Would Help K-FAC

**Scenario:** Using K-FAC for natural gradient optimization on lm_head.

**Problem with empirical Fisher K-FAC:**
- Gradients only come from **observed tokens** in batch
- `G_empirical` has rank ≤ T (number of tokens)
- May miss curvature for **unobserved tokens**

**True Fisher advantage:**
- Samples from **model distribution** `p(y|x)`
- `G_true = diag(mean_probs) - U U^T` includes **all tokens**
- Better curvature estimate for natural gradient

**But still block-level:**
- True Fisher K-FAC still gives `F ≈ A ⊗ G_true`
- Can't extract per-token importance without losing structure
- Useful for optimization, not pruning

---

## Recommendation by Task

| Task | Method | Configuration |
|------|--------|---------------|
| **Vocabulary pruning** | Group Fisher | `reduction='param'`, Welford |
| **Token importance scoring** | Group Fisher | `reduction='param'`, Welford |
| **Task arithmetic** | Group Fisher | `reduction='param'`, Welford |
| **Fine-tuning adaptation** | Group Fisher | Track Fisher over time |
| **Confidence intervals** | Group Fisher + Welford | Use `fisher_variance` |
| **Natural gradient (lm_head)** | K-FAC Woodbury | `kfac_policy="all"` |
| **Optimization with low-confidence** | K-FAC true Fisher | If using natural gradient |
| **Loss landscape analysis** | K-FAC + Lanczos | Eigenspectrum |

---

## Code Examples

### Group Fisher for Vocabulary Pruning (RECOMMENDED)

```python
from BombshellMetrics import BombshellMetrics

# Initialize with per-parameter reduction
metrics = BombshellMetrics(reduction='param', storage='cpu_fp16')

# Accumulate Fisher over many batches (reduces variance)
for batch in dataloader:
    metrics.update_fisher_welford(model, batch, task='pretrain')

# Get importance scores for lm_head
fisher_lm_head = metrics.fisher_accumulated['pretrain']['lm_head.weight']
variance_lm_head = metrics.fisher_variance['pretrain']['lm_head.weight']

# Compute confidence intervals
std = torch.sqrt(variance_lm_head)
confidence_95 = 1.96 * std

# Identify tokens to prune (low importance, high confidence)
importance_threshold = 0.01
to_prune = (fisher_lm_head < importance_threshold) & (confidence_95 < 0.001)

print(f"Can safely prune {to_prune.sum().item()} / {len(to_prune)} tokens")
```

---

### K-FAC for Natural Gradient (If Optimizing)

```python
from fisher.kfac_utils import KFACNaturalGradient

# Initialize K-FAC
kfac = KFACNaturalGradient(
    damping=1e-4,
    update_freq=10,
    kfac_policy="all"  # Woodbury for lm_head
)

# Training loop with natural gradient
for step, batch in enumerate(dataloader):
    # Standard forward-backward
    loss = model(**batch).loss
    loss.backward()
    
    # Collect K-FAC factors periodically
    if step % 10 == 0:
        kfac.collect_kfac_factors(model, batch)
    
    # Compute natural gradient
    gradients = {name: param.grad for name, param in model.named_parameters()}
    nat_grad = kfac.compute_natural_gradient(gradients, model)
    
    # Apply natural gradient
    for name, param in model.named_parameters():
        if name in nat_grad:
            param.grad.copy_(nat_grad[name])
    
    optimizer.step()
```

---

### True Fisher via AdvancedFisherCollector (If Needed)

```python
from fisher.core.fisher_collector_advanced import AdvancedFisherCollector

# Initialize advanced collector
collector = AdvancedFisherCollector(reduction='param')

# Collect true Fisher (samples from model)
true_fisher = collector.collect_true_fisher(
    model, 
    batch, 
    task='pretrain',
    n_samples=10,  # Sample 10 labels per input
    temperature=1.0
)

# Compare to empirical Fisher
empirical_fisher = collector.update_fisher_welford(model, batch, task='pretrain')

# Analyze difference
for key in true_fisher:
    diff = true_fisher[key] - empirical_fisher[key]
    print(f"{key}: true={true_fisher[key].mean():.3f}, "
          f"empirical={empirical_fisher[key].mean():.3f}, "
          f"diff={diff.mean():.3f}")
```

---

## Why I Was Wrong Earlier

**My earlier claim:** "True Fisher for lm_head would help with pruning, fine-tuning, variance-critical apps"

**Why I was wrong:**

1. **Pruning needs per-parameter scores** → Group Fisher gives this, K-FAC doesn't (even with true Fisher)

2. **Variance reduction via accumulation** → Welford over many batches achieves same variance reduction as true Fisher, without 2× compute

3. **Distribution shift tracking** → Need per-parameter scores over time, not block-level curvature

4. **Confidence intervals** → Welford gives explicit variance, true Fisher just has lower per-sample variance

**Correct use of true Fisher:** Natural gradient optimization when model outputs are **low-confidence** (high entropy). Still block-level, not for pruning.

---

## Summary

**For lm_head analysis (pruning, importance, task arithmetic):**
✅ Use **Group Fisher (FisherCollector)** with Welford accumulation
- Per-parameter scores
- Explicit variance
- Already implemented
- Computationally efficient

**For lm_head optimization (natural gradient):**
✅ Use **K-FAC with empirical Fisher** (your current implementation)
- Block-diagonal preconditioner
- Woodbury for memory efficiency
- Standard practice

**True Fisher for lm_head:**
⚠️ Only useful if:
- Using natural gradient optimization
- Model outputs are low-confidence (high entropy)
- Want lower variance curvature estimates
- **NOT for pruning or per-token importance**

---

## Paper Recommendation

**Main results:** Use Group Fisher (Welford) for vocabulary analysis, K-FAC (empirical, Woodbury) for optimization.

**Appendix (optional):** Compare empirical vs true Fisher for natural gradient stability (if you're using natural gradient).

**Don't implement true Fisher for pruning** - it's the wrong tool for the job.

