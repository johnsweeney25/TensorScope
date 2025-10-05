# Fisher Cross-Task Conflict Validation Tests

This directory contains validation experiments for the **ICLR 2026** cross-task sample conflict detection contribution.

## Quick Start

### Run Full Validation (for ICLR paper)

```bash
# On H100/A100 GPU (recommended)
python fisher/tests/run_iclr_validation.py \
    --model Qwen/Qwen2.5-Math-1.5B \
    --num-samples 768 \
    --num-seeds 5 \
    --removal-pct 0.05 \
    --output results_iclr.json \
    --plot results_iclr.png
```

**Runtime:** ~30 minutes on H100, ~45 minutes on A100

### Run Unit Tests (quick validation)

```bash
# Fast tests with synthetic data
python -m pytest fisher/tests/test_cross_task_conflict_validation.py -v

# Or use unittest
python -m unittest fisher.tests.test_cross_task_conflict_validation
```

**Runtime:** ~2 minutes

---

## What This Validates

### Core Claim (for ICLR paper)

> **Removing conflict-identified samples improves multi-task learning performance more than random sample removal.**

### Experimental Design

Three conditions (repeated across multiple random seeds):

1. **Baseline**: Train on all samples from both tasks
   - Reference performance

2. **Random Control**: Remove random X% of samples
   - Tests whether ANY filtering helps (curriculum effect)
   - Null hypothesis: random removal has no effect

3. **Conflict-based** (our method): Remove top X% most conflicting samples
   - Targeted removal based on detected conflicts
   - **Hypothesis**: Outperforms random removal

### Statistical Tests

- **Paired t-test**: Conflict vs Random (same seeds)
- **Effect size**: Cohen's d
- **Significance**: p < 0.05 with FDR correction
- **Power**: Minimum 5 random seeds

### Expected Results

For the claim to be **ICLR-valid**, we need:

✅ **Conflict filtering > Random filtering** (p < 0.05)
✅ **Effect size d ≥ 0.3** (medium effect)
✅ **Consistent across seeds** (low variance)

---

## Output Files

### `results_iclr.json`

Full experimental results in JSON format:

```json
{
  "configuration": {
    "model": "Qwen/Qwen2.5-Math-1.5B",
    "num_samples_per_task": 768,
    "num_seeds": 5,
    "removal_percentage": 0.05
  },
  "analysis": {
    "verdict": "PASS",
    "improvement_vs_random": 2.3,
    "p_value_vs_random": 0.012,
    "cohens_d_vs_random": 0.45
  },
  "raw_results": { ... }
}
```

### `results_iclr.png`

Bar chart with:
- Mean accuracy per condition
- Error bars (std dev)
- Significance annotation

### Console Output

```
ICLR VALIDATION VERDICT
================================================================================
✓ PASS: Conflict-based filtering significantly outperforms random
  Improvement: 2.30% (p=0.012, d=0.45)
```

---

## Files in This Directory

| File | Purpose |
|------|---------|
| `test_cross_task_conflict_validation.py` | Main validation framework + unit tests |
| `run_iclr_validation.py` | Executable script for ICLR experiments |
| `README.md` | This file |

---

## Usage for ICLR Paper

### 1. Run Validation

```bash
python fisher/tests/run_iclr_validation.py \
    --num-seeds 10 \
    --output paper_results.json \
    --plot paper_figure.png
```

### 2. Include in Paper

**Methods Section:**
```latex
We validate our conflict detection approach by comparing three conditions:
(1) baseline (all samples), (2) random filtering (removing 5\% of samples
uniformly at random), and (3) conflict-based filtering (removing the 5\%
most conflicting samples as identified by our method). Each condition was
evaluated across 10 random seeds with paired t-tests.
```

**Results Section:**
```latex
Conflict-based filtering achieved X.X\% accuracy (±Y.Y), significantly
outperforming random filtering (p=0.0XX, Cohen's d=0.XX). This represents
a Z.Z\% improvement over random removal, supporting our hypothesis that
sample-level conflict detection provides actionable insights for multi-task
learning.
```

**Figure Caption:**
```
Figure X: Validation of cross-task conflict detection. Removing samples
identified as conflicting (red) improves multi-task accuracy compared to
random sample removal (gray) and baseline (blue). Error bars show ±1 SD
across 10 random seeds. * indicates p<0.05.
```

### 3. Supplementary Materials

Include `paper_results.json` in supplementary materials for reproducibility.

---

## Ablation Studies

### Effect of Removal Percentage

```bash
for pct in 0.01 0.05 0.10 0.20; do
    python fisher/tests/run_iclr_validation.py \
        --removal-pct $pct \
        --output ablation_pct_${pct}.json
done
```

### Effect of Conflict Threshold

Modify `conflict_threshold` in `CrossTaskConflictDetector`:

```python
# In run_iclr_validation.py, add parameter:
conflict_detector = CrossTaskConflictDetector(
    ...,
    min_effect_size=args.effect_size  # Try: 0.2, 0.5, 0.8
)
```

---

## Troubleshooting

### "FAIL: Conflict-based filtering does not outperform random"

**Possible causes:**
1. Tasks are too similar (not enough conflicts)
2. Sample size too small (increase `--num-samples`)
3. Statistical power too low (increase `--num-seeds`)
4. Model already well-trained (try partially trained checkpoint)

**Solutions:**
- Try more diverse task pairs (math vs. code, not math vs. algebra)
- Use earlier checkpoints in training
- Increase sample size to 1000+ per task

### "MARGINAL: Improvement exists but lacks statistical power"

**Solutions:**
- Increase `--num-seeds` to 10-20
- Use paired t-test (already default)
- Report confidence intervals in paper

---

## Citation

If using this validation framework in your research:

```bibtex
@inproceedings{yourname2026crosstask,
  title={Sample-Level Cross-Task Conflict Detection for Multi-Task Learning},
  author={Your Name},
  booktitle={ICLR},
  year={2026}
}
```

---

## Questions?

See main documentation: `fisher/docs/FISHER_COMPLETE_SUMMARY.md`
