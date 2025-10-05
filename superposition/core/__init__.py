"""
Core superposition analysis modules.

Contains the main implementations of superposition metrics and analyzers.
"""

from .enhanced import SuperpositionMetrics, SuperpositionConfig, analyze_superposition

# Backwards-compatible alias for callers that referenced the V2 name explicitly
SuperpositionMetricsV2 = SuperpositionMetrics
from .analyzer import SuperpositionAnalyzer, SuperpositionAnalysis, analyze_model_superposition_comprehensive

__all__ = [
    'SuperpositionMetrics',
    'SuperpositionMetricsV2',
    'SuperpositionConfig',
    'SuperpositionAnalyzer',
    'SuperpositionAnalysis',
    'analyze_superposition',
    'analyze_model_superposition_comprehensive',
]
