# True Fisher for lm_head: Theory & Motivation

## Status: NOT IMPLEMENTED (Theory Only)

This document explains **why** we would implement true Fisher specifically for the `lm_head` layer, the theoretical foundation, and when it might be beneficial.

---

## Quick Summary

**Empirical Fisher** (what we use now):
- Computes gradients with respect to **observed data labels** `y_data`
- `F_empirical = E[∇log p(y_data|x) ∇log p(y_data|x)^T]`

**True Fisher** (optional for lm_head):
- Computes gradients with respect to **model-sampled labels** `y ~ p(y|x)`
- `F_true = E_y~p(y|x)[∇log p(y|x) ∇log p(y|x)^T]`

**Key Difference:**
- Empirical: "How sensitive is the model to the **actual** next tokens?"
- True: "How sensitive is the model to **all possible** next tokens, weighted by their probability?"

---

## Why This Matters for lm_head

### 1. **Structural Advantage: True Fisher is Diagonal-Minus-Low-Rank**

For softmax classification over vocabulary of size `V`, the true Fisher has special structure:

```
F_true = E[diag(p) - p p^T]
       = D̄ - U U^T
```

where:
- `D̄ = diag(mean_probs)` - diagonal matrix of average probabilities
- `U = [p_1, ..., p_T] / √T` - low-rank matrix of probability vectors

**Memory advantage:**
- Empirical Fisher: `G = U U^T` → rank ≤ T (number of tokens)
- True Fisher: `G = D - U U^T` → **diagonal plus low-rank**

**Why this helps:**
1. **Captures vocabulary-wide information**: Diagonal `D̄` encodes how often each vocab token is predicted
2. **Efficient inversion**: Woodbury identity for diagonal-minus-low-rank is well-conditioned
3. **Better curvature estimate**: Includes information about **all** tokens, not just observed ones

---

## Mathematical Foundation

### True Fisher for Softmax (Per-Token)

For a single token position with softmax output:

```
p_i = exp(z_i) / Σ_j exp(z_j)
```

The Fisher Information Matrix (FIM) for logits `z` is:

```
F = E_y~p(y|x)[∇log p(y|x) ∇log p(y|x)^T]
```

For softmax, this has a **known closed form**:

```
F = diag(p) - p p^T
```

**Proof sketch:**
```
∇log p(y=k|x) = e_k - p  (where e_k is one-hot for class k)

E_y[∇log p(y|x) ∇log p(y|x)^T] 
  = Σ_k p_k (e_k - p)(e_k - p)^T
  = Σ_k p_k (e_k e_k^T - e_k p^T - p e_k^T + p p^T)
  = diag(p) - p p^T
```

### Mini-Batch Averaging

Over `T` tokens:

```
G_true = (1/T) Σ_t [diag(p_t) - p_t p_t^T]
       = D̄ - (1/T) Σ_t p_t p_t^T
       = D̄ - U U^T
```

where:
- `D̄ = (1/T) Σ_t diag(p_t) = diag(mean(p_t))` - element-wise average
- `U = [p_1, ..., p_T] / √T` - stack of probability vectors

---

## Woodbury Inversion for Diagonal-Minus-Low-Rank

Given `G = D̄ - U U^T` with damping `λI`:

```
(D̄ + λI - U U^T)^{-1} = D_λ^{-1} + D_λ^{-1} U S^{-1} U^T D_λ^{-1}
```

where:
- `D_λ = D̄ + λI` (diagonal, easy to invert)
- `S = I - U^T D_λ^{-1} U` (small `T×T` matrix)

**Application to gradient Y:**

```
(G + λI)^{-1} Y = D_λ^{-1} ⊙ Y + D_λ^{-1} ⊙ [U @ (S^{-1} @ (U^T @ (D_λ^{-1} ⊙ Y)))]
```

**Complexity:**
- Storage: `O(V + oT + T²)` (diagonal + U + S_inv)
- Computation: `O(oTi + T²i)` per application

**Compare to empirical Fisher Woodbury:**
- Storage: `O(oT + T²)` (no diagonal, same U and S_inv)
- Computation: Same `O(oTi + T²i)`

---

## When True Fisher Helps

### 1. **Low-Confidence Predictions**

**Scenario:** Model outputs are **diffuse** (high entropy), many tokens have non-negligible probability.

**Problem with empirical Fisher:**
- Only captures curvature around the **single** observed token
- Ignores curvature from other plausible tokens

**True Fisher advantage:**
- Captures curvature weighted by **all** plausible tokens
- Better approximates the **full loss landscape**

**Example:**
```python
# Empirical Fisher: Only gradient for observed token "cat"
y_data = "cat"
grad_empirical = ∇log p("cat"|x)
G_empirical = grad_empirical @ grad_empirical^T

# True Fisher: Weighted average over all tokens
probs = {"cat": 0.3, "dog": 0.25, "bird": 0.2, "fish": 0.15, ...}
G_true = Σ p(y) * [∇log p(y|x) @ ∇log p(y|x)^T]
       = diag(probs) - probs @ probs^T
```

In this case, `G_true` captures that the model is **uncertain** and has significant curvature in multiple directions.

### 2. **Variance Reduction**

**Theory:** True Fisher has **lower variance** as an estimator of the Hessian.

**Why:**
- Empirical Fisher: Single sample `y_data` → high variance across batches
- True Fisher: Expectation over `p(y|x)` → variance only from `x` distribution

**Practical benefit:**
- More stable natural gradient updates
- Better second-order optimization (fewer oscillations)

**Caveat:** Requires **forward pass** to get probabilities → 2× compute

### 3. **Better for Rare Tokens**

**Problem:** Rare vocabulary tokens (e.g., technical terms) appear infrequently in training data.

**Empirical Fisher:**
- These tokens contribute to `G` only when they appear in `y_data`
- Their curvature is **underestimated** → poor natural gradient for rare tokens

**True Fisher:**
- If model assigns non-zero probability `p(rare_token|x) > 0`, it contributes to `G`
- Better coverage of vocabulary space

**When this matters:**
- Domain adaptation (scientific → general language)
- Vocabulary expansion (adding new tokens)
- Low-resource languages

---

## When True Fisher DOESN'T Help (and Why We Skip It)

### 1. **High-Confidence Predictions**

When the model is **confident** (low entropy):

```python
probs = {"the": 0.95, "a": 0.03, "an": 0.01, ...}
```

**True Fisher ≈ Empirical Fisher:**
- `diag(p) - p p^T ≈ diag(p)` when `p` is sharply peaked
- The low-rank term `p p^T` is dominated by the diagonal
- Both methods give similar curvature estimates

**Bottom line:** For well-trained models on in-distribution data, the difference is **small**.

### 2. **Computational Cost**

**Empirical Fisher:** 1 forward pass (for loss computation, already needed)

**True Fisher:** 2 forward passes:
1. Forward to get `logits` → compute `probs = softmax(logits)`
2. Forward again for loss (with sampled or actual labels)

**Cost:** **2× forward compute**, which is the bottleneck for large models.

### 3. **Implementation Complexity**

**Empirical Fisher:**
- Gradients come from standard backward pass
- No special handling needed

**True Fisher:**
- Must extract probabilities before loss
- Must handle diagonal `D̄` separately from low-rank `U U^T`
- More complex Woodbury formula (diagonal base, not identity)

**Engineering effort:** 3-4 hours to implement correctly.

### 4. **Standard Practice in K-FAC Literature**

**Empirical Fisher is standard:**
- Martens & Grosse (2015): "Optimizing Neural Networks with Kronecker-factored Approximate Curvature"
- Grosse & Martens (2016): "A Kronecker-factored approximate Fisher matrix for convolution layers"
- Ba et al. (2017): "Distributed Second-Order Optimization using Kronecker-Factored Approximations"

**True Fisher rarely used:**
- Requires model-specific knowledge (softmax structure)
- Extra compute not justified for standard training
- Empirical Fisher "good enough" for natural gradient

---

## When You SHOULD Implement True Fisher

### Use Case 1: **Second-Order Pruning**

**Goal:** Prune vocabulary tokens with minimal impact on loss.

**Why true Fisher:**
- Importance score needs curvature w.r.t. **all** tokens, not just observed
- Rare tokens might be pruned if only empirical Fisher is used

**Example:**
```python
# Empirical: "quantum" appears 10 times in 10k batches
# → Low importance (small gradient contribution)

# True: Model assigns p("quantum"|physics_context) = 0.05
# → Higher importance (significant curvature)
```

### Use Case 2: **Fine-Tuning with Distribution Shift**

**Scenario:** Pre-train on web text, fine-tune on scientific papers.

**Problem:** Pre-training empirical Fisher underestimates scientific token importance.

**Solution:** True Fisher during fine-tuning captures model's **current beliefs** about token probabilities, not just observed data.

### Use Case 3: **Variance-Critical Applications**

**Scenario:** Safety-critical deployment where optimizer stability is paramount.

**Example:** Medical NLP where updates must be **conservative** and **stable**.

**Benefit:** True Fisher's lower variance → more predictable updates.

---

## Implementation Sketch (What We'd Add)

### Step 1: Detect lm_head and Extract Probabilities

```python
if self.kfac_true_fisher_head and name.endswith("lm_head"):
    # Forward pass to get logits
    with torch.no_grad():
        logits = model.forward_to_layer(batch, layer=name)
        probs = torch.softmax(logits, dim=-1)  # [BS, seq, vocab]
    
    # Mask and flatten
    if 'attention_mask' in batch:
        mask = batch['attention_mask'].unsqueeze(-1)
        probs = probs * mask
    
    probs_flat = probs.reshape(-1, vocab_size)  # [T, vocab]
    T_effective = probs_flat.shape[0]
```

### Step 2: Build Diagonal and Low-Rank Components

```python
# Diagonal: mean probability across tokens
D_bar = probs_flat.mean(dim=0)  # [vocab]

# Low-rank: U = [p_1, ..., p_T] / sqrt(T)
U = (probs_flat / math.sqrt(T_effective)).to(dtype=torch.float16)  # [vocab, T]

# Damping
D_lambda = D_bar + self.damping_G  # [vocab]
D_lambda_inv = 1.0 / D_lambda  # [vocab]
```

### Step 3: Compute Woodbury Inverse Components

```python
# S = I - U^T D_λ^{-1} U (element-wise scaling)
U_scaled = U * D_lambda_inv.unsqueeze(1)  # Row-wise scaling [vocab, T]
S = torch.eye(T_effective, dtype=torch.float32, device=U.device)
S = S - (U.t().float() @ U_scaled.float())  # [T, T]

# Robust Cholesky inversion (same as empirical path)
S_inv = robust_cholesky_inverse(S, eps=self.kfac_eps)
```

### Step 4: Store Factors

```python
self.kfac_factors[name] = {
    ...
    'G_type': 'woodbury_true',
    'U': U,  # [vocab, T], fp16
    'D_lambda_inv': D_lambda_inv.to(dtype=torch.float32),  # [vocab], fp32
    'S_inv': S_inv,  # [T, T], fp32
    'T_effective': T_effective
}
```

### Step 5: Apply Natural Gradient

```python
if G_type == 'woodbury_true':
    # (D_λ - UU^T)^{-1} Y = D_λ^{-1} ⊙ Y + D_λ^{-1} ⊙ [U @ (S^{-1} @ (U^T @ (D_λ^{-1} ⊙ Y)))]
    
    D_lambda_inv = factors['D_lambda_inv'].to(target_device)
    U = factors['U'].to(target_device)
    S_inv = factors['S_inv'].to(target_device)
    
    # First term: element-wise scaling
    Y_scaled = Y * D_lambda_inv.unsqueeze(1)  # [vocab, in+1]
    
    # Second term: Woodbury correction
    Z = U.t().float() @ Y_scaled.float()  # [T, in+1]
    W = S_inv @ Z  # [T, in+1]
    correction = U.float() @ W  # [vocab, in+1]
    
    Y_G = Y_scaled + D_lambda_inv.unsqueeze(1) * correction
```

---

## Complexity Comparison

| Metric | Empirical Fisher (Woodbury) | True Fisher (Woodbury + Diag) |
|--------|------------------------------|-------------------------------|
| **Forward passes** | 1 | 2 (need probs before backward) |
| **Storage (G)** | `O(vocab·T + T²)` | `O(vocab + vocab·T + T²)` |
| **Inversion cost** | `O(vocab·T² + T³)` | `O(vocab·T² + T³)` (same) |
| **Application cost** | `O(vocab·T·hidden + T²·hidden)` | `O(vocab·hidden + vocab·T·hidden + T²·hidden)` |
| **Extra cost** | — | `+O(vocab·hidden)` per apply |

**Bottom line:** True Fisher adds:
- 1× forward pass overhead (significant for large models)
- `O(vocab)` storage for diagonal (negligible, ~200KB for 50k vocab)
- `O(vocab·hidden)` compute per application (~1% overhead)

---

## Recommendation

**For ICLR 2026 Main Results: SKIP True Fisher**

**Reasons:**
1. ✅ **Empirical Fisher is standard** in K-FAC literature
2. ✅ **2× forward compute** not justified for main experiments
3. ✅ **Implementation complexity** (3-4 hours) better spent elsewhere
4. ✅ **Minimal benefit** for well-trained models on in-distribution data

**For Appendix/Ablations: OPTIONAL**

**Include if:**
- Reviewers ask about variance reduction
- You have results showing optimizer instability with empirical Fisher
- You want to compare pruning/merging with true vs empirical importance

**How to frame in paper:**
> "We use empirical Fisher (standard in K-FAC literature) for computational efficiency. 
> True Fisher (with model-sampled labels) is a natural extension with lower variance 
> but requires 2× forward passes; we leave this for future work."

---

## References

**K-FAC with Empirical Fisher:**
- Martens & Grosse (2015). "Optimizing Neural Networks with Kronecker-factored Approximate Curvature." ICML.
- Ba et al. (2017). "Distributed Second-Order Optimization using Kronecker-Factored Approximations." ICLR.

**True Fisher Theory:**
- Amari (1998). "Natural Gradient Works Efficiently in Learning." Neural Computation.
- Pascanu & Bengio (2014). "Revisiting Natural Gradient for Deep Networks." ICLR Workshop.

**Diagonal-Minus-Low-Rank Inversion:**
- Woodbury (1950). "Inverting Modified Matrices." Memorandum Report 42.
- Henderson & Searle (1981). "On Deriving the Inverse of a Sum of Matrices." SIAM Review.

---

## TL;DR

**Why true Fisher for lm_head:**
- Captures curvature over **all** vocabulary tokens, not just observed
- Diagonal-minus-low-rank structure enables efficient Woodbury inversion
- Lower variance estimator → more stable natural gradient

**Why we skip it:**
- 2× forward compute (expensive for large models)
- Empirical Fisher "good enough" for standard training
- Implementation complexity not justified for main results
- Standard practice in K-FAC literature uses empirical

**Ship empirical Fisher, note true Fisher as future work.**
