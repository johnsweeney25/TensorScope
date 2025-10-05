"""
Superposition Analysis Package

A comprehensive framework for analyzing feature superposition in neural networks,
including metrics from recent papers on neural scaling laws and representation capacity.
"""

from .core.enhanced import SuperpositionMetrics, SuperpositionConfig, analyze_superposition
from .core.analyzer import SuperpositionAnalyzer, SuperpositionAnalysis, analyze_model_superposition_comprehensive
from .metrics.paper_metrics import PaperSuperpositionMetrics, analyze_model_superposition

# Optional alias for callers that migrated to the V2 naming
SuperpositionMetricsV2 = SuperpositionMetrics

__version__ = '2.0.0'

__all__ = [
    # Core classes
    'SuperpositionMetrics',
    'SuperpositionMetricsV2',
    'SuperpositionConfig',
    'SuperpositionAnalyzer',
    'SuperpositionAnalysis',

    # Paper-specific metrics
    'PaperSuperpositionMetrics',

    # Analysis functions
    'analyze_superposition',
    'analyze_model_superposition',
    'analyze_model_superposition_comprehensive',
]
