"""
Fisher Information Matrix Analysis Module
==========================================

A comprehensive suite for Fisher Information Matrix computation and analysis,
including advanced features like K-FAC approximation, spectral analysis,
unified Lanczos eigenvalue computation, and cross-task overlap analysis.

Main Components:
- FisherCollector: Basic Fisher matrix collection with EMA and group reduction
- AdvancedFisherCollector: Extended with true Fisher, K-FAC, and capacity metrics
- FisherSpectral: Spectral analysis with block-diagonal approximation
- Unified Lanczos: Robust eigenvalue computation for both Hessian and Fisher
- Overlap Analysis: Compare active Fisher groups across tasks to identify shared vs specialized parameters

Quick Start:
    from fisher import FisherCollector, AdvancedFisherCollector, analyze_fisher_overlap

    # Basic usage
    collector = FisherCollector()
    collector.collect_fisher(model, batch)
    fisher_values = collector.get_all_fisher()

    # Advanced usage with spectrum
    adv_collector = AdvancedFisherCollector(use_kfac=True)
    spectrum = adv_collector.lanczos_spectrum(model, batch, operator='ggn')

    # Overlap analysis between tasks
    overlap_results = analyze_fisher_overlap(welford_accumulators)
    print(f"Jaccard similarity: {overlap_results['summary']['average_jaccard_similarity']:.2%}")
"""

# Core collectors
from fisher.core.fisher_collector import FisherCollector
from fisher.core.fisher_collector_advanced import AdvancedFisherCollector

# Spectral analysis
from fisher.core.fisher_spectral import FisherSpectral, SpectralConfig

# Unified Lanczos system
from fisher.core.fisher_lanczos_unified import (
    compute_spectrum,
    LanczosConfig,
    LinOp,
    HessianOperator,
    GGNOperator,
    EmpiricalFisherOperator,
    KFACFisherOperator,
    create_operator,
    lanczos_algorithm
)

# Compatibility utilities
from fisher.core.fisher_compatibility import FisherCompatibilityMixin

# Overlap analysis
from fisher.core.overlap_analysis import (
    analyze_fisher_overlap,
    extract_active_groups,
    compute_overlap_statistics,
    analyze_parameter_type_overlap
)

__all__ = [
    # Main collectors
    'FisherCollector',
    'AdvancedFisherCollector',

    # Spectral analysis
    'FisherSpectral',
    'SpectralConfig',

    # Lanczos system
    'compute_spectrum',
    'LanczosConfig',
    'LinOp',
    'HessianOperator',
    'GGNOperator',
    'EmpiricalFisherOperator',
    'KFACFisherOperator',
    'create_operator',
    'lanczos_algorithm',

    # Compatibility
    'FisherCompatibilityMixin',

    # Overlap analysis
    'analyze_fisher_overlap',
    'extract_active_groups',
    'compute_overlap_statistics',
    'analyze_parameter_type_overlap',
]

# Version info
__version__ = '2.0.0'
__author__ = 'ICLR 2026 Project'