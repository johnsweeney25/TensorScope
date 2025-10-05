# Metrics Catalog (Full)

This document lists the metrics available in TensorScope, organized by category. Use it as a reference; the README shows a curated subset.

Note: Function names follow the public API when available; many are exposed via `UnifiedModelAnalyzer` registry.

## Optimization & Curvature
- Fisher eigenvalues (Lanczos)
- Hessian eigenvalues (Lanczos)
- Spectrum comparison (Fisher vs Hessian vs K‑FAC)
- Loss barrier
- Mode connectivity
- Loss landscape (2D), directional losses
- SAM sharpness
- Pruning sensitivity

## Fisher Information
- Group Fisher (Welford accumulation)
- Group Fisher (EMA)
- One‑shot Fisher
- Fisher overlap (cross‑task similarity)
- Compare task Fisher
- Fisher‑weighted merge, scale‑by‑Fisher
- Fisher importance
- Fisher pruning masks
- Top Fisher directions
- Fisher uncertainty (variance/CI)

## Gradients & Training
- Gradient pathology (vanishing/exploding)
- Gradient signal‑to‑noise
- Raw gradient conflict (single‑scale, multiscale)
- PCGrad conflict (single‑scale, multiscale)
- Layer gradient alignment (single‑scale, multiscale)
- Gradient conflict pairs
- Gradient alignment trajectory (across models/runs)

## Attribution & Influence
- TracIn self‑influence (with checkpoints)
- Integrated gradients
- Attention attribution
- Causal necessity
- Find critical samples (influence)
- Extract task vectors

## Attention & Circuits
- Induction head strength
- QK‑OV pairing
- Attention head specialization
- Attention flow patterns
- Attention entropy, attention drift, attention concentration
- Logit lens

## Representation & Geometry
- CKA/RSA similarity
- Block CKA gap
- Effective rank, full effective rank
- Superposition: vector interference
- Superposition: feature frequency distribution
- Superposition: strength, dimensional scaling, feature sparsity
- Superposition: representation capacity, feature emergence
- Comprehensive superposition analysis
- Superposition trajectory
- Analyze model superposition

## Embeddings & Manifolds
- Embedding singularities (Robinson)
- Embedding singularity report
- Manifold metrics (fiber bundle tests, Ricci curvature)

## Information Theory & Dynamics
- Signal propagation (health check)
- Signal propagation stability
- Information flow
- Plasticity index
- Alignment fragility
- Practical compression ratio (int8+zlib)
- Parameter storage bits
- Layer mutual information (multiple estimators)
- Heuristic PID/MINMI
- Variational IB probe
- Analyze training dynamics, model behavior scales

## Lottery Tickets & Pruning
- Pruning robustness
- Layerwise magnitude tickets
- Gradient importance
- Fisher importance
- Early Bird tickets
- Lottery ticket quality
- Ticket overlap
- Iterative magnitude pruning (IMP)

## Modularity & Architecture
- Memory‑efficient OV→U
- Fisher‑weighted damage
- Damage asymmetry
- SAM sharpness (architecture‑aware)

## Documentation Links
- Fisher Lanczos: docs/FISHER_EIGENVALUES_LANCZOS_DOCUMENTATION.md
- SAM sharpness: docs/SAM_SHARPNESS_DOCUMENTATION.md
- Loss landscape: docs/LOSS_LANDSCAPE_2D_DOCUMENTATION.md
- TracIn: docs/TRACIN_DOCUMENTATION.md
- Manifold metrics: docs/MANIFOLD_METRICS_DOCUMENTATION.md
- MDL: docs/MDL_COMPLEXITY_IMPLEMENTATION.md
- PID/MINMI: docs/HEURISTIC_PID_MINMI_DOCUMENTATION.md

