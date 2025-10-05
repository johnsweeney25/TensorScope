"""
QK-OV Interference Analysis Module

Implements Fisher-normalized, block-wise, head-resolved interference metrics
for analyzing cross-task conflicts at the circuit level.

Section 4.1: From Contributions to Circuit-Level Interference

Key Components:
---------------
- qkov_interference: Core metric implementation (M^B_{ij,ℓ,h})
- qkov_statistics: Statistical testing (permutation, FDR, bootstrap)

Usage:
------
    from fisher.qkov import QKOVConfig, QKOVInterferenceMetric, QKOVStatistics

    # Auto-detect model configuration
    config = QKOVConfig.from_model(model)

    # Setup metric
    metric = QKOVInterferenceMetric(config, fisher_collector)

    # Compute interference
    scores = metric.compute_sample_pair(
        task_a='math', sample_i=7,
        task_b='code', sample_j=23,
        layer=3, head=5
    )
    # Returns: {'Q': 0.42, 'K': 0.31, 'V': 0.18, 'O': 0.55}

    # Full heatmap
    heatmap = metric.compute_heatmap(
        task_a='math',
        task_b='code',
        layers=[3, 4, 5],
        heads=range(12)
    )

    # Statistical testing
    stats = QKOVStatistics(fdr_alpha=0.05)
    results = stats.test_heatmap(heatmap, contribs, grads, fisher)

Documentation:
--------------
- Engineering guide: fisher/qkov/docs/QKOV_ENGINEERING_NOTES.md
- Implementation summary: fisher/qkov/docs/QKOV_IMPLEMENTATION_SUMMARY.md
- Example: examples/qkov_interference_example.py

Author: ICLR 2026 Project
"""

from .qkov_interference import (
    QKOVConfig,
    QKOVIndexer,
    BlockHeadSlice,
    InterferenceScore,
    QKOVInterferenceMetric,
)

from .qkov_statistics import (
    QKOVStatistics,
    PermutationTestResult,
    ClusterResult,
)

__all__ = [
    # Configuration and indexing
    'QKOVConfig',
    'QKOVIndexer',
    'BlockHeadSlice',

    # Core metric
    'QKOVInterferenceMetric',
    'InterferenceScore',

    # Statistical testing
    'QKOVStatistics',
    'PermutationTestResult',
    'ClusterResult',
]

__version__ = '1.0.0'
