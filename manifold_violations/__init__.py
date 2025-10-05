"""
Manifold Violations: Testing the Robinson Hypothesis in LLM Embeddings

This module implements methods from "Token Embeddings Violate the Manifold Hypothesis"
(Robinson et al., 2025), revealing geometric instabilities in LLM embeddings.

Main Components:
- EmbeddingSingularityMetrics: High-level GPU-accelerated interface for unified_model_analysis
- RobinsonFiberBundleTest: Core volume growth analysis
- SingularityMapper: Detect embedding singularities
- PolysemyDetector: Find polysemous tokens
"""

# Core analysis (Robinson paper)
from .robinson_fiber_bundle_test import RobinsonFiberBundleTest, RobinsonTestResult, analyze_embedding_space
# from .fiber_bundle_core import FiberBundleAnalyzer  # Not implemented yet
from .fiber_bundle_hypothesis_test import FiberBundleHypothesisTest, FiberBundleTestResult

# Singularity detection
from .singularity_mapper import SingularityMapper
from .polysemy_detector import PolysemyDetector, PolysemyAnalysis

# Geometric analysis
from .tractable_manifold_curvature_fixed import (
    compute_ricci_curvature_debiased,
    # compute_sectional_curvature_fixed,  # Not available
    compute_manifold_metrics_fixed
)

# Stability analysis
from .token_stability_analyzer import TokenStabilityAnalyzer
from .prompt_robustness_analyzer import PromptRobustnessAnalyzer
from .training_singularity_dynamics import TrainingSingularityTracker

# High-level interface (GPU-accelerated)
from .embedding_singularity_metrics import EmbeddingSingularityMetrics

# Integration utilities
from .manifold_fiber_integration import GeometricAnalyzer as ManifoldFiberIntegration

__all__ = [
    # Main interface
    'EmbeddingSingularityMetrics',

    # Robinson tests
    'RobinsonFiberBundleTest',
    'RobinsonTestResult',
    'analyze_embedding_space',

    # Fiber bundle
    # 'FiberBundleAnalyzer',  # Not implemented yet
    'FiberBundleHypothesisTest',
    'FiberBundleTestResult',

    # Singularity detection
    'SingularityMapper',
    'PolysemyDetector',
    'PolysemyAnalysis',

    # Geometric analysis
    'compute_ricci_curvature_debiased',
    # 'compute_sectional_curvature_fixed',  # Not available
    'compute_manifold_metrics_fixed',

    # Stability
    'TokenStabilityAnalyzer',
    'PromptRobustnessAnalyzer',
    'TrainingSingularityTracker',

    # Integration
    'ManifoldFiberIntegration',
]

__version__ = '1.0.0'