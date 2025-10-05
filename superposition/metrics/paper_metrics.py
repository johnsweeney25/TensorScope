"""
Extended SuperpositionMetrics to compute the specific metrics from
"Superposition Yields Robust Neural Scaling" (Liu et al., 2025)

This module adds methods to compute:
- ϕ₁/₂: Fraction of features with ∥Wᵢ∥₂ > 1/2
- ϕ₁: Fraction of features with ∥Wᵢ∥₂ > 1
- Weak vs Strong superposition classification
- Welch bound analysis for geometric constraints
"""

import torch
import numpy as np
from typing import Dict, Tuple, Optional, List
from ..core.enhanced import SuperpositionMetrics
import logging

logger = logging.getLogger(__name__)


class PaperSuperpositionMetrics(SuperpositionMetrics):
    """
    Extended metrics to compute the specific quantities from the paper.
    """

    def compute_superposition_regime(
        self,
        weight_matrix: torch.Tensor,
        threshold_half: float = 0.5,
        threshold_one: float = 1.0,
        compute_overlaps: bool = True
    ) -> Dict[str, any]:
        """
        Compute the paper's specific superposition metrics and classify the regime.

        From the paper:
        - ϕ₁/₂: Fraction of features with ∥Wᵢ∥₂ > 1/2 (represented features)
        - ϕ₁: Fraction of features with ∥Wᵢ∥₂ > 1 (strongly represented features)

        Args:
            weight_matrix: Weight matrix W of shape (n_features, n_dims)
            threshold_half: Threshold for ϕ₁/₂ (default 0.5 from paper)
            threshold_one: Threshold for ϕ₁ (default 1.0 from paper)
            compute_overlaps: Whether to compute geometric overlap statistics

        Returns:
            Dictionary containing:
            - phi_half: ϕ₁/₂ value (fraction with norm > 0.5)
            - phi_one: ϕ₁ value (fraction with norm > 1.0)
            - regime: 'weak', 'strong', or 'no' superposition
            - n_represented: Number of features with norm > 0.5
            - n_strongly_represented: Number of features with norm > 1.0
            - feature_norms: Array of all feature norms
            - geometric_overlaps: Overlap statistics if requested
            - welch_bound_ratio: How close overlaps are to Welch bound
        """
        weight_matrix = weight_matrix.to(self.device)
        n_features, n_dims = weight_matrix.shape

        # Compute L2 norms of each feature (row)
        feature_norms = torch.linalg.norm(weight_matrix, dim=1)

        # Compute ϕ₁/₂ and ϕ₁
        n_represented = (feature_norms > threshold_half).sum().item()
        n_strongly_represented = (feature_norms > threshold_one).sum().item()

        phi_half = n_represented / n_features
        phi_one = n_strongly_represented / n_features

        # Classify superposition regime
        regime = self._classify_regime(phi_half, phi_one, n_features, n_dims)

        results = {
            'phi_half': phi_half,
            'phi_one': phi_one,
            'regime': regime,
            'n_represented': n_represented,
            'n_strongly_represented': n_strongly_represented,
            'n_features': n_features,
            'n_dimensions': n_dims,
            'feature_norms': feature_norms.cpu().numpy(),
            'dimension_ratio': n_dims / n_features if n_features > 0 else 0
        }

        # Compute geometric overlaps if requested
        if compute_overlaps and n_represented > 1:
            overlap_results = self._compute_geometric_overlaps(
                weight_matrix, feature_norms, threshold_half
            )
            results.update(overlap_results)

        return results

    def _classify_regime(
        self,
        phi_half: float,
        phi_one: float,
        n_features: int,
        n_dims: int
    ) -> str:
        """
        Classify the superposition regime based on paper's definitions.

        - No superposition: phi_half ≈ m/n (only m features in m dimensions)
        - Weak superposition: phi_half > m/n but phi_one ≈ 0 (some features ignored)
        - Strong superposition: phi_half ≈ 1 (all features represented with interference)
        """
        if n_features == 0:
            return "no_superposition"

        dimension_ratio = n_dims / n_features

        # No superposition: represented features ≈ available dimensions
        if abs(phi_half - dimension_ratio) < 0.1:
            return "no_superposition"

        # Strong superposition: most features are represented
        elif phi_half > 0.8:  # Paper suggests ϕ₁/₂ ≈ 1 for strong
            return "strong_superposition"

        # Weak superposition: more than m features but not all
        elif phi_half > dimension_ratio + 0.1:
            return "weak_superposition"

        else:
            return "no_superposition"

    def _compute_geometric_overlaps(
        self,
        weight_matrix: torch.Tensor,
        feature_norms: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, any]:
        """
        Compute geometric overlap statistics for represented features.

        The paper shows that in strong superposition, overlaps scale as √(1/m),
        following the Welch bound for equal-angle tight frames.
        """
        # Get only represented features (norm > threshold)
        represented_mask = feature_norms > threshold
        represented_features = weight_matrix[represented_mask]

        if represented_features.shape[0] < 2:
            return {}

        # Normalize represented features
        represented_norms = feature_norms[represented_mask].unsqueeze(1)
        normalized_features = represented_features / represented_norms

        # Compute pairwise overlaps
        overlap_matrix = torch.matmul(normalized_features, normalized_features.T).abs()

        # Get off-diagonal elements
        n_rep = represented_features.shape[0]
        mask = ~torch.eye(n_rep, device=self.device, dtype=torch.bool)
        off_diagonal_overlaps = overlap_matrix[mask]

        # Compute statistics
        mean_overlap = off_diagonal_overlaps.mean().item()
        std_overlap = off_diagonal_overlaps.std().item()
        max_overlap = off_diagonal_overlaps.max().item()

        # Compute Welch bound (theoretical minimum for equal-angle tight frames)
        # For n vectors in d dimensions: overlap ≥ √((n-d)/(d(n-1)))
        n, d = represented_features.shape
        if n > d:
            welch_bound = np.sqrt((n - d) / (d * (n - 1)))
        else:
            welch_bound = 0.0

        # Check if overlaps follow √(1/m) scaling (strong superposition signature)
        expected_scaling = 1.0 / np.sqrt(d) if d > 0 else 0
        scaling_ratio = mean_overlap / expected_scaling if expected_scaling > 0 else 0

        return {
            'geometric_overlaps': {
                'mean_overlap': mean_overlap,
                'std_overlap': std_overlap,
                'max_overlap': max_overlap,
                'welch_bound': welch_bound,
                'welch_bound_ratio': mean_overlap / welch_bound if welch_bound > 0 else 0,
                'expected_scaling': expected_scaling,
                'scaling_ratio': scaling_ratio,
                'follows_sqrt_scaling': abs(scaling_ratio - 1.0) < 0.2
            }
        }

    def analyze_superposition_scaling(
        self,
        models_dict: Dict[str, torch.nn.Module],
        test_batch: Dict[str, torch.Tensor]
    ) -> Dict[str, any]:
        """
        Analyze how superposition regime affects scaling laws.

        The paper shows:
        - Strong superposition → robust 1/m scaling
        - Weak superposition → brittle, data-dependent scaling
        """
        scaling_results = self.analyze_dimensional_scaling(models_dict, test_batch)

        # For each model, determine its superposition regime
        regime_results = {}
        for size_key, model in models_dict.items():
            # Extract weight matrix (e.g., from embedding layer)
            weight_matrix = None
            for name, param in model.named_parameters():
                if 'embed' in name.lower() and 'weight' in name:
                    weight_matrix = param.data
                    break

            if weight_matrix is not None:
                regime = self.compute_superposition_regime(weight_matrix)
                regime_results[size_key] = regime

        # Analyze correlation between regime and scaling
        if 'alpha' in scaling_results and regime_results:
            strong_superposition_models = [
                key for key, res in regime_results.items()
                if res['regime'] == 'strong_superposition'
            ]
            weak_superposition_models = [
                key for key, res in regime_results.items()
                if res['regime'] == 'weak_superposition'
            ]

            scaling_results['regime_analysis'] = {
                'strong_superposition_models': strong_superposition_models,
                'weak_superposition_models': weak_superposition_models,
                'scaling_exponent': scaling_results['alpha'],
                'expected_strong_scaling': 1.0,  # Paper shows 1/m scaling
                'is_robust_scaling': abs(scaling_results['alpha'] - 1.0) < 0.2
            }

        scaling_results['regime_details'] = regime_results

        return scaling_results

    def compute_feature_representation_distribution(
        self,
        weight_matrix: torch.Tensor,
        n_bins: int = 50
    ) -> Dict[str, any]:
        """
        Compute the distribution of feature representation strengths.

        This helps visualize the transition between weak and strong superposition.
        """
        weight_matrix = weight_matrix.to(self.device)

        # Compute norms
        feature_norms = torch.linalg.norm(weight_matrix, dim=1)

        # Create histogram
        hist, bin_edges = np.histogram(
            feature_norms.cpu().numpy(),
            bins=n_bins,
            range=(0, feature_norms.max().item())
        )

        # Identify key thresholds
        n_below_half = (feature_norms < 0.5).sum().item()
        n_between = ((feature_norms >= 0.5) & (feature_norms < 1.0)).sum().item()
        n_above_one = (feature_norms >= 1.0).sum().item()

        return {
            'norm_histogram': {
                'counts': hist.tolist(),
                'bin_edges': bin_edges.tolist()
            },
            'threshold_counts': {
                'below_0.5': n_below_half,
                'between_0.5_and_1': n_between,
                'above_1.0': n_above_one
            },
            'statistics': {
                'mean_norm': feature_norms.mean().item(),
                'std_norm': feature_norms.std().item(),
                'min_norm': feature_norms.min().item(),
                'max_norm': feature_norms.max().item(),
                'median_norm': feature_norms.median().item()
            }
        }

    def verify_paper_predictions(
        self,
        weight_matrix: torch.Tensor,
        loss_values: Optional[List[float]] = None,
        model_dimensions: Optional[List[int]] = None
    ) -> Dict[str, any]:
        """
        Verify the paper's key predictions about superposition.

        Key predictions:
        1. Strong superposition → overlaps scale as √(1/m)
        2. Strong superposition → loss scales as 1/m
        3. Weak superposition → data-dependent, brittle scaling
        """
        results = {}

        # Check prediction 1: Overlap scaling
        regime_analysis = self.compute_superposition_regime(weight_matrix, compute_overlaps=True)
        results['regime'] = regime_analysis

        if 'geometric_overlaps' in regime_analysis:
            overlaps = regime_analysis['geometric_overlaps']
            results['prediction_1_overlap_scaling'] = {
                'follows_sqrt_scaling': overlaps['follows_sqrt_scaling'],
                'actual_vs_expected': overlaps['scaling_ratio'],
                'verified': overlaps['follows_sqrt_scaling']
            }

        # Check prediction 2: Loss scaling (if loss data provided)
        if loss_values is not None and model_dimensions is not None:
            scaling_fit = self.fit_scaling_law(model_dimensions, loss_values)

            if 'alpha' in scaling_fit:
                alpha = scaling_fit['alpha']
                results['prediction_2_loss_scaling'] = {
                    'scaling_exponent': alpha,
                    'expected_for_strong': 1.0,
                    'deviation': abs(alpha - 1.0),
                    'verified': abs(alpha - 1.0) < 0.2
                }

        # Check prediction 3: Regime determines scaling robustness
        if regime_analysis['regime'] == 'strong_superposition':
            results['prediction_3_robustness'] = {
                'regime': 'strong_superposition',
                'expected_behavior': 'robust_scaling',
                'phi_half': regime_analysis['phi_half'],
                'phi_one': regime_analysis['phi_one']
            }
        elif regime_analysis['regime'] == 'weak_superposition':
            results['prediction_3_robustness'] = {
                'regime': 'weak_superposition',
                'expected_behavior': 'brittle_data_dependent_scaling',
                'phi_half': regime_analysis['phi_half'],
                'phi_one': regime_analysis['phi_one']
            }

        return results


def analyze_model_superposition(model: torch.nn.Module, verbose: bool = True) -> Dict[str, any]:
    """
    Convenience function to analyze a model's superposition properties.

    Args:
        model: The model to analyze
        verbose: Whether to print results

    Returns:
        Complete superposition analysis including regime classification
    """
    analyzer = PaperSuperpositionMetrics()

    # Find weight matrix (e.g., embedding layer)
    weight_matrix = None
    layer_name = None

    for name, param in model.named_parameters():
        if 'embed' in name.lower() and 'weight' in name:
            weight_matrix = param.data
            layer_name = name
            break

    if weight_matrix is None:
        # Try first linear layer
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight_matrix = module.weight.data
                layer_name = name
                break

    if weight_matrix is None:
        return {'error': 'Could not find weight matrix to analyze'}

    # Compute all metrics
    results = {
        'layer_analyzed': layer_name,
        'regime_analysis': analyzer.compute_superposition_regime(weight_matrix),
        'norm_distribution': analyzer.compute_feature_representation_distribution(weight_matrix)
    }

    # Verify paper predictions
    verification = analyzer.verify_paper_predictions(weight_matrix)
    results['paper_verification'] = verification

    if verbose:
        regime = results['regime_analysis']
        print(f"\n=== Superposition Analysis for {layer_name} ===")
        print(f"Features: {regime['n_features']}, Dimensions: {regime['n_dimensions']}")
        print(f"Dimension ratio (m/n): {regime['dimension_ratio']:.3f}")
        print(f"\nKey Metrics:")
        print(f"  ϕ₁/₂ (features with ||W|| > 0.5): {regime['phi_half']:.3f}")
        print(f"  ϕ₁ (features with ||W|| > 1.0): {regime['phi_one']:.3f}")
        print(f"  Regime: {regime['regime'].replace('_', ' ').title()}")

        if 'geometric_overlaps' in regime:
            geo = regime['geometric_overlaps']
            print(f"\nGeometric Analysis:")
            print(f"  Mean overlap: {geo['mean_overlap']:.4f}")
            print(f"  Expected √(1/m): {geo['expected_scaling']:.4f}")
            print(f"  Follows √(1/m) scaling: {geo['follows_sqrt_scaling']}")
            print(f"  Welch bound: {geo['welch_bound']:.4f}")
            print(f"  Welch ratio: {geo['welch_bound_ratio']:.2f}")

        if 'paper_verification' in results:
            ver = results['paper_verification']
            print(f"\nPaper Predictions Verification:")
            if 'prediction_1_overlap_scaling' in ver:
                p1 = ver['prediction_1_overlap_scaling']
                print(f"  ✓ Overlap scaling: {'VERIFIED' if p1['verified'] else 'NOT VERIFIED'}")
            if 'prediction_2_loss_scaling' in ver:
                p2 = ver['prediction_2_loss_scaling']
                print(f"  ✓ Loss scaling (α={p2['scaling_exponent']:.2f}): {'VERIFIED' if p2['verified'] else 'NOT VERIFIED'}")

    return results


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    # Create a test model
    model = nn.Sequential(
        nn.Embedding(1000, 64),  # 1000 features in 64 dimensions
        nn.Linear(64, 128),
        nn.Linear(128, 1000)
    )

    # Analyze superposition
    results = analyze_model_superposition(model, verbose=True)

    # Check if model is in strong superposition
    if results['regime_analysis']['regime'] == 'strong_superposition':
        print("\n✓ Model is in STRONG SUPERPOSITION regime")
        print("  → Expect robust 1/m scaling laws")
        print("  → All features represented with controlled interference")
    elif results['regime_analysis']['regime'] == 'weak_superposition':
        print("\n⚠ Model is in WEAK SUPERPOSITION regime")
        print("  → Expect brittle, data-dependent scaling")
        print("  → Only frequent features well-represented")
