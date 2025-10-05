"""
Fisher Core Module
==================

Core implementations for Fisher Information Matrix computation and analysis.
"""

from .fisher_collector import FisherCollector
from .fisher_collector_advanced import AdvancedFisherCollector
from .fisher_spectral import FisherSpectral, SpectralConfig
from .fisher_lanczos_unified import (
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
from .fisher_compatibility import FisherCompatibilityMixin

__all__ = [
    'FisherCollector',
    'AdvancedFisherCollector',
    'FisherSpectral',
    'SpectralConfig',
    'compute_spectrum',
    'LanczosConfig',
    'LinOp',
    'HessianOperator',
    'GGNOperator',
    'EmpiricalFisherOperator',
    'KFACFisherOperator',
    'create_operator',
    'lanczos_algorithm',
    'FisherCompatibilityMixin',
]