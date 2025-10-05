# API Reference

Output formats and data structures.

[← Back to README](../README.md)

---

## Artifacts & Formats

**Outputs from `UnifiedModelAnalyzer`:**
- Grouped Fisher: `{layer_name: {'mean': float, 'variance': float, 'ci_lower': float, 'ci_upper': float}}`
- Fisher masks: Binary/continuous masks (0-1) per parameter group
- Overlap matrices: Task similarity scores (0-1)
- Sample conflicts: `[{'sample_a_idx': int, 'sample_b_idx': int, 'p_value': float, 'effect_size': float, 'fdr_significant': bool}]`
- Spectra: Top eigenvalues, tridiagonal matrices, spectral gaps
- Geometry: Singularity density maps, violation flags, transition radii
- Superposition: Polysemanticity scores per layer/feature

**Export formats:**
- JSON (structured metrics)
- CSV (for spreadsheet analysis)
- NumPy arrays (for further computation)
- Visualizations (via matplotlib/seaborn)

---

