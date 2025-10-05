# Interpretation Guide

How to interpret metric values and make decisions.

[← Back to README](../README.md)

---

## Quick Interpretation Guide

Fast reference for common decision points:

**Fisher Mask Overlap (Task Similarity):**
- `< 0.2`: Prefer adapters/LoRA or separate heads; avoid heavy shared updates
- `0.2 - 0.5`: Mixed strategy; share backbone, task-specific heads
- `0.5 - 0.8`: Shared backbone reasonable; monitor for conflicts
- `> 0.8`: Shared backbone OK; can prune shared parameters first

**Fisher Uncertainty (Confidence in Importance Estimates):**
- `> 30% groups with rel. std > 0.3`: Gather more batches before pruning/freezing
- `< 10% groups with rel. std > 0.3`: High confidence; safe to act on estimates
- High variance + low mean: Noisy signal, be conservative
- High variance + high mean: Genuinely variable importance, handle carefully

**Curvature Agreement (Cross-Validation Across Methods):**
- `Spectral vs Lanczos correlation ≥ 0.8`: Curvature estimates reliable; safe to use for thresholds
- `Correlation 0.5 - 0.8`: Moderate agreement; use conservative safety margins
- `Correlation < 0.5`: Low agreement; validate with LR sweeps or SAM before trusting

**Sample Conflict Severity:**
- `Effect size > 0.8 + FDR✓`: Strong conflict; prioritize removal/reweighting
- `Effect size 0.5 - 0.8 + FDR✓`: Moderate conflict; consider for curation
- `Effect size > 0.5 but FDR✗`: May be false positive; needs more data

**Superposition Regime (Feature Packing Density):**
- `High superposition + high Fisher`: Critical polysemantic features; preserve carefully
- `High superposition + low Fisher`: Safe to prune; less critical features
- `Low superposition + high Fisher`: Dedicated features; good pruning targets if sparse

---

