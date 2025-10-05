#!/usr/bin/env python3
"""
Integration module bridging manifold analysis and fiber bundle hypothesis testing.

This module provides a unified interface for geometric analysis of embeddings,
intelligently choosing between manifold-based and fiber bundle-based approaches
based on the geometric properties of the space.

Key insight: Fiber bundle tests are prerequisites that determine WHERE
manifold analysis is valid.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import warnings

# Import manifold analysis (using fixed versions)
from .tractable_manifold_curvature_fixed import (
    compute_ricci_curvature_debiased,
    compute_manifold_metrics_fixed
)

# Import fiber bundle analysis
from .fiber_bundle_hypothesis_test import (
    FiberBundleHypothesisTest,
    FiberBundleTestResult
)

# Import token stability if available
try:
    from token_stability_analyzer import TokenStabilityAnalyzer
    HAS_TOKEN_ANALYZER = True
except ImportError:
    HAS_TOKEN_ANALYZER = False


@dataclass
class UnifiedGeometricResult:
    """Unified result from geometric analysis."""
    # Fiber bundle test results
    passes_fiber_bundle_test: bool
    fiber_bundle_p_value: float
    fiber_bundle_regime: str
    irregularity_score: float

    # Manifold analysis (if applicable)
    has_manifold_structure: bool

    # Stability assessment
    geometry_type: str  # 'manifold', 'fiber_bundle', 'irregular'
    stability_score: float  # 0-1, higher is more stable
    confidence: float

    # Recommendations
    use_manifold_analysis: bool
    use_alternative_methods: bool
    recommended_approach: str

    # Optional fields with defaults
    ricci_curvature: Optional[Tuple[float, float]] = None  # (mean, std)
    sectional_curvature: Optional[Tuple[float, float]] = None
    intrinsic_dimension: Optional[float] = None


class GeometricAnalyzer:
    """
    Unified geometric analyzer combining manifold and fiber bundle approaches.

    This class intelligently selects appropriate geometric analysis methods
    based on the underlying structure of the data.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        manifold_confidence_threshold: float = 0.95,
        cache_results: bool = True
    ):
        """
        Initialize geometric analyzer.

        Args:
            alpha: Significance level for fiber bundle test
            manifold_confidence_threshold: P-value threshold for manifold analysis
            cache_results: Whether to cache analysis results
        """
        self.alpha = alpha
        self.manifold_threshold = manifold_confidence_threshold
        self.cache_results = cache_results

        # Initialize testers
        self.fiber_test = FiberBundleTest(alpha=alpha)

        # Cache
        self.cached_results: Dict[int, UnifiedGeometricResult] = {}

    def analyze_point(
        self,
        data: torch.Tensor,
        point_idx: int,
        compute_all_metrics: bool = True
    ) -> UnifiedGeometricResult:
        """
        Perform unified geometric analysis at a point.

        Args:
            data: (n_points, n_dims) tensor
            point_idx: Index of point to analyze
            compute_all_metrics: Whether to compute all possible metrics

        Returns:
            UnifiedGeometricResult with comprehensive analysis
        """
        # Check cache
        cache_key = point_idx
        if self.cache_results and cache_key in self.cached_results:
            return self.cached_results[cache_key]

        # Step 1: Fiber bundle test (prerequisite)
        # Convert to numpy if needed
        if hasattr(data, 'detach'):
            # Handle PyTorch tensors
            data_np = data.detach().cpu().numpy()
        elif isinstance(data, np.ndarray):
            # Already numpy
            data_np = data
        else:
            # Try to convert
            data_np = np.array(data)
        fb_result = self.fiber_test.test_point(data_np, point_idx)

        # Step 2: Determine if manifold analysis is appropriate
        passes_fb_test = not fb_result.reject_null
        use_manifold = passes_fb_test and fb_result.p_value > self.manifold_threshold

        # Step 3: Conditional manifold analysis
        ricci_curv = None
        sectional_curv = None
        intrinsic_dim = None

        if use_manifold and compute_all_metrics:
            try:
                # Get local neighborhood for manifold analysis
                k_neighbors = min(20, len(data) - 1)
                distances = torch.cdist(data, data)
                _, neighbor_idx = torch.topk(
                    distances[point_idx],
                    k_neighbors + 1,
                    largest=False
                )
                local_points = data[neighbor_idx[1:]]  # Exclude self

                # Compute manifold metrics
                ricci_curv = compute_ricci_curvature_tractable(
                    local_points,
                    n_samples=10
                )
                sectional_curv = compute_sectional_curvature_tractable(
                    local_points,
                    n_samples=10
                )
                intrinsic_dim = compute_intrinsic_dimension_twonn(local_points)

            except Exception as e:
                warnings.warn(f"Manifold analysis failed: {e}")
                use_manifold = False

        # Step 4: Determine geometry type
        if use_manifold:
            geometry_type = "manifold"
            has_manifold = True
        elif passes_fb_test:
            geometry_type = "fiber_bundle"
            has_manifold = False
        else:
            geometry_type = "irregular"
            has_manifold = False

        # Step 5: Calculate stability score
        stability_score = self._calculate_stability_score(
            fb_result, geometry_type, ricci_curv
        )

        # Step 6: Generate recommendations
        recommended_approach = self._get_recommended_approach(
            geometry_type, fb_result
        )

        # Create unified result
        result = UnifiedGeometricResult(
            passes_fiber_bundle_test=passes_fb_test,
            fiber_bundle_p_value=fb_result.p_value,
            fiber_bundle_regime=fb_result.regime,
            irregularity_score=fb_result.irregularity_score,
            has_manifold_structure=has_manifold,
            ricci_curvature=ricci_curv,
            sectional_curvature=sectional_curv,
            intrinsic_dimension=intrinsic_dim,
            geometry_type=geometry_type,
            stability_score=stability_score,
            confidence=fb_result.confidence,
            use_manifold_analysis=use_manifold,
            use_alternative_methods=(geometry_type == "irregular"),
            recommended_approach=recommended_approach
        )

        # Cache result
        if self.cache_results:
            self.cached_results[cache_key] = result

        return result

    def analyze_dataset(
        self,
        data: torch.Tensor,
        sample_size: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze geometric structure of entire dataset.

        Args:
            data: (n_points, n_dims) tensor
            sample_size: Number of points to sample
            verbose: Whether to print progress

        Returns:
            Summary statistics and classification
        """
        n_points = len(data)

        # Sample if needed
        if sample_size and sample_size < n_points:
            sample_idx = np.random.choice(n_points, sample_size, replace=False)
        else:
            sample_idx = list(range(n_points))

        if verbose:
            print(f"Analyzing {len(sample_idx)} points...")

        # Analyze each point
        results = []
        geometry_counts = {"manifold": 0, "fiber_bundle": 0, "irregular": 0}

        for i, idx in enumerate(sample_idx):
            if verbose and i % 50 == 0:
                print(f"  Progress: {i}/{len(sample_idx)}")

            result = self.analyze_point(data, idx, compute_all_metrics=False)
            results.append(result)
            geometry_counts[result.geometry_type] += 1

        # Calculate statistics
        n_analyzed = len(results)
        manifold_pct = geometry_counts["manifold"] / n_analyzed
        fiber_bundle_pct = geometry_counts["fiber_bundle"] / n_analyzed
        irregular_pct = geometry_counts["irregular"] / n_analyzed

        mean_stability = np.mean([r.stability_score for r in results])
        mean_irregularity = np.mean([r.irregularity_score for r in results])

        # Determine overall structure
        if manifold_pct > 0.7:
            overall_structure = "predominantly_manifold"
        elif irregular_pct > 0.5:
            overall_structure = "highly_irregular"
        elif fiber_bundle_pct > 0.5:
            overall_structure = "fiber_bundle_dominant"
        else:
            overall_structure = "mixed_geometry"

        summary = {
            "n_analyzed": n_analyzed,
            "geometry_distribution": {
                "manifold": manifold_pct,
                "fiber_bundle": fiber_bundle_pct,
                "irregular": irregular_pct
            },
            "geometry_counts": geometry_counts,
            "overall_structure": overall_structure,
            "mean_stability": mean_stability,
            "mean_irregularity": mean_irregularity,
            "recommendations": self._get_dataset_recommendations(
                overall_structure, manifold_pct, irregular_pct
            )
        }

        if verbose:
            self._print_summary(summary)

        return summary

    def analyze_layer_representations(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        layer_indices: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Analyze geometric structure across model layers.

        Args:
            model: Neural network model
            batch: Input batch
            layer_indices: Specific layers to analyze

        Returns:
            Layer-wise geometric analysis
        """
        # Get representations from model
        device = next(model.parameters()).device
        input_ids = batch['input_ids'].to(device)

        # Forward pass with hidden states
        with torch.no_grad():
            if hasattr(model, 'transformer'):
                outputs = model(input_ids, output_hidden_states=True)
                hidden_states = outputs.hidden_states
            else:
                raise ValueError("Model must output hidden states")

        # Select layers to analyze
        if layer_indices is None:
            layer_indices = list(range(len(hidden_states)))

        # Analyze each layer
        layer_results = {}

        for layer_idx in layer_indices:
            representations = hidden_states[layer_idx]

            # Flatten batch and sequence dimensions
            if len(representations.shape) == 3:
                batch_size, seq_len, hidden_dim = representations.shape
                points = representations.reshape(-1, hidden_dim)
            else:
                points = representations

            # Subsample for efficiency
            if len(points) > 1000:
                indices = torch.randperm(len(points))[:1000]
                points = points[indices]

            # Analyze geometric structure
            summary = self.analyze_dataset(
                points,
                sample_size=min(100, len(points)),
                verbose=False
            )

            layer_results[f"layer_{layer_idx}"] = summary

        # Analyze progression across layers
        progression = self._analyze_layer_progression(layer_results)

        return {
            "layer_results": layer_results,
            "progression": progression
        }

    def _calculate_stability_score(
        self,
        fb_result: FiberBundleTestResult,
        geometry_type: str,
        ricci_curv: Optional[Tuple[float, float]]
    ) -> float:
        """Calculate stability score from geometric properties."""
        # Base score from geometry type
        base_scores = {
            "manifold": 0.9,
            "fiber_bundle": 0.6,
            "irregular": 0.3
        }
        score = base_scores[geometry_type]

        # Adjust for fiber bundle test strength
        if fb_result.p_value > 0.99:
            score += 0.05
        elif fb_result.p_value < 0.01:
            score -= 0.1

        # Adjust for regime
        if fb_result.regime == "boundary":
            score -= 0.1

        # Adjust for curvature if available
        if ricci_curv is not None:
            mean_curv, std_curv = ricci_curv
            # High curvature variance indicates instability
            if std_curv > 0.5:
                score -= 0.1

        return float(np.clip(score, 0, 1))

    def _get_recommended_approach(
        self,
        geometry_type: str,
        fb_result: FiberBundleTestResult
    ) -> str:
        """Get recommended analysis approach."""
        if geometry_type == "manifold":
            return (
                "Use standard manifold analysis methods. "
                "Curvature metrics and dimension estimates are reliable."
            )
        elif geometry_type == "fiber_bundle":
            return (
                "Use fiber bundle aware methods. "
                "Standard manifold analysis may be unreliable. "
                "Consider multi-scale approaches."
            )
        else:  # irregular
            if fb_result.regime == "boundary":
                return (
                    "CAUTION: Boundary regime with high irregularity. "
                    "Avoid geometric methods. Use robust statistical approaches."
                )
            else:
                return (
                    "CAUTION: Irregular geometry detected. "
                    "Traditional manifold methods will fail. "
                    "Use distribution-free or topology-based methods."
                )

    def _get_dataset_recommendations(
        self,
        structure: str,
        manifold_pct: float,
        irregular_pct: float
    ) -> List[str]:
        """Generate recommendations for dataset analysis."""
        recommendations = []

        if structure == "predominantly_manifold":
            recommendations.append(
                "Dataset has good manifold structure. "
                "Standard dimensionality reduction and manifold learning methods appropriate."
            )
        elif structure == "highly_irregular":
            recommendations.append(
                "WARNING: Dataset violates manifold hypothesis. "
                f"{irregular_pct:.1%} of points are irregular."
            )
            recommendations.append(
                "Avoid PCA, t-SNE, UMAP for critical analysis. "
                "Consider robust methods or data cleaning."
            )
        elif structure == "fiber_bundle_dominant":
            recommendations.append(
                "Dataset has fiber bundle structure but not manifold. "
                "Use multi-scale analysis methods."
            )
        else:  # mixed
            recommendations.append(
                f"Mixed geometry: {manifold_pct:.1%} manifold, "
                f"{irregular_pct:.1%} irregular."
            )
            recommendations.append(
                "Consider analyzing different regions separately."
            )

        return recommendations

    def _analyze_layer_progression(
        self,
        layer_results: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Analyze how geometry changes across layers."""
        layers = sorted(layer_results.keys())

        manifold_progression = []
        stability_progression = []

        for layer in layers:
            result = layer_results[layer]
            manifold_progression.append(
                result["geometry_distribution"]["manifold"]
            )
            stability_progression.append(
                result["mean_stability"]
            )

        # Analyze trends
        manifold_trend = np.polyfit(range(len(layers)), manifold_progression, 1)[0]
        stability_trend = np.polyfit(range(len(layers)), stability_progression, 1)[0]

        return {
            "manifold_progression": manifold_progression,
            "stability_progression": stability_progression,
            "manifold_trend": "increasing" if manifold_trend > 0.01 else
                            "decreasing" if manifold_trend < -0.01 else "stable",
            "stability_trend": "increasing" if stability_trend > 0.01 else
                             "decreasing" if stability_trend < -0.01 else "stable",
            "interpretation": self._interpret_progression(
                manifold_trend, stability_trend
            )
        }

    def _interpret_progression(
        self,
        manifold_trend: float,
        stability_trend: float
    ) -> str:
        """Interpret layer progression trends."""
        if manifold_trend > 0.01 and stability_trend > 0.01:
            return (
                "Model progressively builds more regular geometric structure. "
                "Higher layers have better manifold properties."
            )
        elif manifold_trend < -0.01:
            return (
                "Model destroys manifold structure in higher layers. "
                "May indicate specialized processing or task-specific representations."
            )
        elif stability_trend < -0.01:
            return (
                "Stability decreases in higher layers. "
                "Model may be sensitive to input perturbations."
            )
        else:
            return "Geometric structure remains relatively consistent across layers."

    def _print_summary(self, summary: Dict[str, Any]) -> None:
        """Print analysis summary."""
        print("\n" + "=" * 50)
        print("GEOMETRIC STRUCTURE ANALYSIS")
        print("=" * 50)

        print(f"\nPoints analyzed: {summary['n_analyzed']}")
        print(f"Overall structure: {summary['overall_structure']}")

        print("\nGeometry distribution:")
        for geom_type, pct in summary['geometry_distribution'].items():
            print(f"  {geom_type:15s}: {pct:6.2%}")

        print(f"\nMean stability score: {summary['mean_stability']:.3f}")
        print(f"Mean irregularity: {summary['mean_irregularity']:.3f}")

        print("\nRecommendations:")
        for rec in summary['recommendations']:
            print(f"  • {rec}")


def integrated_geometric_analysis(
    data: Union[torch.Tensor, np.ndarray],
    sample_size: Optional[int] = None,
    compute_full_metrics: bool = True,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Perform integrated geometric analysis on data.

    This function provides a high-level interface for combined
    fiber bundle and manifold analysis.

    Args:
        data: Data points to analyze
        sample_size: Number of points to sample
        compute_full_metrics: Whether to compute all metrics
        verbose: Print progress

    Returns:
        Complete geometric analysis
    """
    # Convert to tensor if needed
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    # Initialize analyzer
    analyzer = GeometricAnalyzer()

    # Perform analysis
    results = analyzer.analyze_dataset(
        data,
        sample_size=sample_size,
        verbose=verbose
    )

    # Add detailed metrics for a sample if requested
    if compute_full_metrics and len(data) > 0:
        # Analyze first few points in detail
        detailed_results = []
        for i in range(min(5, len(data))):
            detailed = analyzer.analyze_point(data, i, compute_all_metrics=True)
            detailed_results.append({
                "point_idx": i,
                "geometry_type": detailed.geometry_type,
                "stability_score": detailed.stability_score,
                "intrinsic_dimension": detailed.intrinsic_dimension,
                "ricci_curvature": detailed.ricci_curvature
            })

        results["sample_detailed_results"] = detailed_results

    return results


if __name__ == "__main__":
    print("Manifold-Fiber Bundle Integration Module")
    print("=" * 50)

    # Example: Analyze synthetic data
    print("\nExample analysis on synthetic data...")

    # Create data with mixed geometry
    np.random.seed(42)

    # Region 1: Clean manifold
    t = np.linspace(0, 4*np.pi, 100)
    manifold_region = np.column_stack([
        np.sin(t),
        np.cos(t),
        t / 10
    ])

    # Region 2: Irregular
    irregular_region = np.random.randn(50, 3) * 2

    # Combine
    data = np.vstack([manifold_region, irregular_region])
    data_tensor = torch.from_numpy(data).float()

    # Run integrated analysis
    results = integrated_geometric_analysis(
        data_tensor,
        sample_size=50,
        compute_full_metrics=True,
        verbose=True
    )

    print("\n✅ Integration module ready for use!")