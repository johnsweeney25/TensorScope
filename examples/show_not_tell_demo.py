#!/usr/bin/env python3
"""
Show, don't tell: one-pass, many-lens demo on a tiny model.

This script:
1) Computes groupwise Fisher (Welford) for two synthetic tasks
2) Generates Fisher-based pruning masks and computes overlap
3) Compares curvature lenses (grouped_fisher, KFAC, spectral, Lanczos)

It uses a tiny MLP that accepts a batch dict with 'input_ids' and returns a scalar loss,
so it runs anywhere without external model dependencies.
"""
import torch
import torch.nn as nn
from types import SimpleNamespace

from BombshellMetrics import BombshellMetrics
from unified_model_analysis import UnifiedModelAnalyzer


class TinyModel(nn.Module):
    """Small MLP that consumes 'input_ids' and returns a scalar loss."""
    def __init__(self, input_dim=64, hidden=32, out=8):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden)
        self.fc2 = nn.Linear(hidden, out)

    def forward(self, input_ids=None, attention_mask=None, **kwargs):
        # Accepts dict-style calls: model(**batch)
        if input_ids is None:
            raise ValueError("Expected 'input_ids' in batch")
        x = input_ids.float()
        x = torch.relu(self.fc1(x))
        y = self.fc2(x)
        # Regression-to-zero toy loss
        return (y.pow(2).mean())


def make_batch(batch_size=8, dim=64, kind="math"):
    """Create synthetic batches with slightly different distributions."""
    g = torch.Generator().manual_seed(42 if kind == "math" else 7)
    x = torch.randn(batch_size, dim, generator=g)
    return {
        'input_ids': x,
        'attention_mask': torch.ones(batch_size, dim),
    }


def main():
    model = TinyModel()
    model.train()

    # 1) Groupwise Fisher (Welford) for two tasks
    bombshell = BombshellMetrics(fisher_mode='welford', enable_cross_task_analysis=False)

    math_batches = [make_batch(kind="math") for _ in range(2)]
    general_batches = [make_batch(kind="general") for _ in range(2)]

    ok_math = bombshell.compute_fisher_welford_batches(model, math_batches, task='math', show_progress=False)
    ok_general = bombshell.compute_fisher_welford_batches(model, general_batches, task='general', show_progress=False)

    print("\n=== Fisher (groupwise, Welford) ===")
    for task in ('math', 'general'):
        n = len(bombshell.fisher_accumulated.get(task, {}))
        print(f"Task '{task}': {n} group keys")

    # 2) Fisher-based masks + overlap
    masks_math = bombshell.get_fisher_pruning_masks(task='math', sparsity=0.5, structured=False)
    masks_general = bombshell.get_fisher_pruning_masks(task='general', sparsity=0.5, structured=False)

    if 'error' not in masks_math and 'error' not in masks_general:
        overlap = bombshell.compute_fisher_overlap(masks_math, masks_general)
        print("\n=== Fisher masks & overlap ===")
        print(f"Masks(math):   {len(masks_math)} params | Masks(general): {len(masks_general)} params")
        print(f"Mask overlap (fraction of elements kept in both): {overlap:.3f}")
    else:
        print("Mask generation skipped (insufficient Fisher data)")

    # 3) Curvature method comparison (grouped_fisher, KFAC, spectral, Lanczos)
    analyzer = UnifiedModelAnalyzer()
    context = SimpleNamespace()
    # Representative batch for spectral/Lanczos
    context.batch = make_batch()
    context.batches = [context.batch]
    # Optional: seed some KFAC factors in context if available (here we skip)

    task_batches_dict = {
        'math': [math_batches[0]],
        'general': [general_batches[0]],
    }

    comp = analyzer._compute_fisher_method_comparison(model, context, bombshell, task_batches_dict)
    print("\n=== Curvature lenses (comparison) ===")
    for k in ('grouped_fisher', 'kfac', 'spectral', 'lanczos'):
        entry = comp.get(k, {})
        print(f"{k:15s} -> status={entry.get('status','-')}, desc={entry.get('description','-')}")

    if 'comparison' in comp and 'eigenvalue_agreement' in comp['comparison']:
        agree = comp['comparison']['eigenvalue_agreement']
        print(f"Eigenvalue agreement (spectral vs lanczos): {agree['agreement_score']:.3f}")

    # 4) Simple recommendations (show, don't tell)
    print("\n=== Recommendations (pilot heuristics) ===")

    # Helper: estimate fraction of high-uncertainty groups via Welford
    def uncertainty_fraction(task: str, rel_std_threshold: float = 0.3) -> float:
        n = bombshell.n_samples_seen.get(task, 0)
        if n <= 1:
            return 1.0
        acc = bombshell.fisher_accumulated.get(task, {})
        m2 = bombshell.fisher_m2.get(task, {})
        hi = 0
        tot = 0
        eps = 1e-12
        for k, mean_t in acc.items():
            var_t = m2.get(k, torch.zeros_like(mean_t)) / (n - 1)
            rel_std = (var_t.clamp_min(0).sqrt()) / (mean_t.abs() + eps)
            # Count elements
            hi += (rel_std > rel_std_threshold).sum().item()
            tot += rel_std.numel()
        return (hi / max(tot, 1)) if tot > 0 else 1.0

    # A) Task sharing decision from mask overlap
    if 'error' not in masks_math and 'error' not in masks_general:
        if overlap < 0.2:
            print("- Low mask overlap (<0.2): prefer adapters/LoRA or separate heads; avoid heavy shared updates.")
        elif overlap > 0.8:
            print("- High mask overlap (>0.8): shared backbone is reasonable; prune shared parts first.")
        else:
            print("- Moderate overlap: mixed sharing likely; validate with small adapter layers.")
    else:
        print("- Masks unavailable: compute Fisher first (Welford), then derive masks.")

    # B) Uncertainty-aware decisions from Welford
    uf_math = uncertainty_fraction('math') if ok_math else 1.0
    uf_general = uncertainty_fraction('general') if ok_general else 1.0
    if max(uf_math, uf_general) > 0.3:
        print("- Fisher uncertainty is high (>30% groups with rel std > 0.3): gather more batches or delay pruning/regularization.")
    else:
        print("- Fisher uncertainty is low: safe to use masks/weights from grouped Fisher.")

    # C) Curvature validation via spectral/Lanczos agreement
    agree_score = None
    if 'comparison' in comp and 'eigenvalue_agreement' in comp['comparison']:
        agree_score = comp['comparison']['eigenvalue_agreement'].get('agreement_score', None)
    if agree_score is not None:
        if agree_score >= 0.8:
            print("- Curvature triangulation strong (agreement ≥ 0.8): grouped Fisher masks/thresholds likely reliable.")
        else:
            print("- Curvature triangulation weak: use conservative thresholds; validate with smaller steps.")
    else:
        print("- Curvature triangulation not available: spectral/Lanczos failed or skipped.")


if __name__ == "__main__":
    main()
