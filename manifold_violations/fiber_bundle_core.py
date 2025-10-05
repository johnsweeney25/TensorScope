#!/usr/bin/env python3
"""
Core mathematical implementation of Fiber Bundle Hypothesis Test.

This module contains pure mathematical functions for testing smooth fiber bundle
structure, with no dependencies on specific machine learning frameworks.

Based on: "Token embeddings violate the manifold hypothesis" (Robinson et al.)

Key concepts:
- A fiber bundle is a space that locally looks like a product space B × F
- More general than manifolds (which are fiber bundles with trivial fiber)
- Tests for local product structure and smooth transitions
"""

import numpy as np
from typing import Tuple, Optional, List, Dict
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist
from scipy.linalg import subspace_angles
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


@dataclass
class FiberBundleTestResult:
    """Results from fiber bundle hypothesis test."""
    # Core test results
    p_value: float
    reject_null: bool
    test_statistic: float

    # Component scores
    dimension_consistency: float
    curvature_regularity: float
    tangent_alignment: float
    regime_transition: float

    # Geometric properties
    local_dimension: float
    neighborhood_radius: float
    curvature_mean: float
    curvature_std: float

    # Regime identification
    regime: str  # 'small_radius', 'large_radius', 'boundary'

    # Stability assessment
    irregularity_score: float
    confidence: float


class FiberBundleTest:
    """
    Mathematical test for smooth fiber bundle structure.

    This class implements the core statistical test without any
    dependencies on specific embedding types or ML frameworks.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_neighbors_small: int = 10,
        n_neighbors_large: int = 50,
        n_bootstrap: int = 100,
        n_tangent_dims: int = 5
    ):
        """
        Initialize fiber bundle test.

        Args:
            alpha: Significance level for hypothesis testing
            n_neighbors_small: Neighbors for small radius regime
            n_neighbors_large: Neighbors for large radius regime
            n_bootstrap: Bootstrap samples for p-value estimation
            n_tangent_dims: Dimensions to use for tangent space estimation
        """
        self.alpha = alpha
        self.n_neighbors_small = n_neighbors_small
        self.n_neighbors_large = n_neighbors_large
        self.n_bootstrap = n_bootstrap
        self.n_tangent_dims = n_tangent_dims

    def test_point(
        self,
        data: np.ndarray,
        point_idx: int,
        precomputed_distances: Optional[np.ndarray] = None
    ) -> FiberBundleTestResult:
        """
        Test fiber bundle hypothesis at a specific point.

        Args:
            data: (n_points, n_dims) array of points
            point_idx: Index of point to test
            precomputed_distances: Optional (n_points, n_points) distance matrix

        Returns:
            FiberBundleTestResult with test statistics
        """
        n_points, n_dims = data.shape

        # Handle edge cases
        if n_points <= 1:
            # Single point - return neutral result
            return FiberBundleTestResult(
                p_value=0.5,
                reject_null=False,
                test_statistic=0.0,
                dimension_consistency=0.0,
                curvature_regularity=0.0,
                tangent_alignment=0.0,
                regime_transition=0.0,
                local_dimension=float(n_dims),
                neighborhood_radius=0.0,
                curvature_mean=0.0,
                curvature_std=0.0,
                regime="small_radius",
                irregularity_score=0.0,
                confidence=0.1
            )

        # Compute distances if not provided
        if precomputed_distances is None:
            distances = cdist(data, data, metric='euclidean')
        else:
            distances = precomputed_distances

        # Adjust neighbor counts for small datasets
        actual_neighbors_small = min(self.n_neighbors_small, n_points - 1)
        actual_neighbors_large = min(self.n_neighbors_large, n_points - 1)

        # Get neighborhoods at different scales
        small_nbr_idx, small_dists = self._get_k_neighbors(
            distances[point_idx], actual_neighbors_small
        )
        large_nbr_idx, large_dists = self._get_k_neighbors(
            distances[point_idx], actual_neighbors_large
        )

        # Component 1: Test dimension consistency across scales
        dim_score = self._test_dimension_consistency(
            data, small_nbr_idx, large_nbr_idx
        )

        # Component 2: Test curvature regularity
        curv_score, curv_mean, curv_std = self._test_curvature_regularity(
            data, point_idx, small_nbr_idx, distances
        )

        # Component 3: Test tangent space alignment (fiber structure)
        tangent_score = self._test_tangent_alignment(
            data, point_idx, small_nbr_idx
        )

        # Component 4: Test regime transition smoothness
        transition_score = self._test_regime_transition(
            small_dists, large_dists, n_dims
        )

        # Combine into overall test statistic
        test_statistic = self._combine_statistics(
            dim_score, curv_score, tangent_score, transition_score
        )

        # Estimate p-value via bootstrap
        p_value = self._bootstrap_p_value(
            data, test_statistic, distances
        )

        # Identify spatial regime
        regime = self._identify_regime(
            np.mean(small_dists), np.mean(large_dists), n_dims
        )

        # Calculate local dimension
        local_dim = self._estimate_intrinsic_dimension(data[small_nbr_idx])

        # Assess confidence in result
        confidence = self._calculate_confidence(
            p_value, len(small_nbr_idx), regime
        )

        return FiberBundleTestResult(
            p_value=p_value,
            reject_null=p_value < self.alpha,
            test_statistic=test_statistic,
            dimension_consistency=dim_score,
            curvature_regularity=curv_score,
            tangent_alignment=tangent_score,
            regime_transition=transition_score,
            local_dimension=local_dim,
            neighborhood_radius=float(np.mean(small_dists)),
            curvature_mean=curv_mean,
            curvature_std=curv_std,
            regime=regime,
            irregularity_score=test_statistic,
            confidence=confidence
        )

    def _get_k_neighbors(
        self,
        distances: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get k nearest neighbors excluding self."""
        # Sort distances and get indices
        sorted_idx = np.argsort(distances)
        # Exclude self (index 0 after sorting)
        neighbor_idx = sorted_idx[1:k+1]
        neighbor_dists = distances[neighbor_idx]
        return neighbor_idx, neighbor_dists

    def _test_dimension_consistency(
        self,
        data: np.ndarray,
        small_neighbors: np.ndarray,
        large_neighbors: np.ndarray
    ) -> float:
        """
        Test if intrinsic dimension is consistent across scales.

        In a fiber bundle, dimension should be locally constant.
        Large variations indicate irregularity.
        """
        # Estimate dimension at small scale
        dim_small = self._estimate_intrinsic_dimension(data[small_neighbors])

        # Estimate dimension at large scale
        dim_large = self._estimate_intrinsic_dimension(data[large_neighbors])

        # Relative difference (normalized by mean)
        mean_dim = (dim_small + dim_large) / 2
        if mean_dim > 0:
            inconsistency = abs(dim_small - dim_large) / mean_dim
        else:
            inconsistency = 0.0

        return float(inconsistency)

    def _test_curvature_regularity(
        self,
        data: np.ndarray,
        center_idx: int,
        neighbors: np.ndarray,
        distances: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Test if local neighborhood has regular geometry.

        NOTE: This is NOT Ricci curvature. It tests for local geometric regularity
        using angle distributions. True Ricci curvature requires optimal transport
        or heat kernel methods which are computationally expensive.

        Returns (irregularity_score, angle_variance, angle_std).
        """
        from scipy.spatial.distance import cosine
        from scipy.stats import ks_2samp

        # Use vectorized operations to compute angles
        center = data[center_idx]
        vectors = data[neighbors] - center

        # Normalize vectors using sklearn's normalize for numerical stability
        from sklearn.preprocessing import normalize
        normalized_vecs = normalize(vectors, norm='l2', axis=1)

        # Compute pairwise angles using dot products
        n_neighbors = len(neighbors)
        angles = []

        if n_neighbors >= 2:
            # Vectorized angle computation
            dot_products = normalized_vecs @ normalized_vecs.T
            # Clip for numerical stability before arccos
            dot_products = np.clip(dot_products, -1.0, 1.0)

            # Extract upper triangle (excluding diagonal)
            for i in range(n_neighbors):
                for j in range(i+1, n_neighbors):
                    angle = np.arccos(dot_products[i, j])
                    angles.append(angle)

        if len(angles) > 1:
            angles = np.array(angles)

            # Test for regularity: compare to uniform distribution
            # Generate uniform reference distribution
            uniform_angles = np.random.uniform(0, np.pi, size=max(100, len(angles)))

            # Kolmogorov-Smirnov test for distribution similarity
            ks_stat, ks_pval = ks_2samp(angles, uniform_angles)

            # Higher KS statistic means more irregular
            irregularity = float(ks_stat)

            # Return statistics
            return irregularity, float(np.mean(angles)), float(np.std(angles))

        return 0.0, 0.0, 0.0

    def _test_tangent_alignment(
        self,
        data: np.ndarray,
        center_idx: int,
        neighbors: np.ndarray
    ) -> float:
        """
        Test alignment of tangent spaces (fiber bundle property).

        In a smooth fiber bundle, nearby tangent spaces should align smoothly.
        """
        if len(neighbors) < 3:
            return 0.0

        # Estimate tangent space at center
        center_tangent = self._estimate_tangent_space(
            data[neighbors] - data[center_idx]
        )

        # For each neighbor, estimate its tangent space and compare
        misalignments = []

        for nbr_idx in neighbors[:min(5, len(neighbors))]:
            # Find neighbors of this neighbor
            nbr_point = data[nbr_idx]
            dists = np.linalg.norm(data - nbr_point, axis=1)
            nbr_neighbors = np.argsort(dists)[1:min(6, len(data))]

            if len(nbr_neighbors) < 3:
                continue

            # Estimate tangent space
            nbr_tangent = self._estimate_tangent_space(
                data[nbr_neighbors] - data[nbr_idx]
            )

            # Measure alignment using principal angles
            angles = subspace_angles(center_tangent, nbr_tangent)
            misalignment = np.mean(angles)
            misalignments.append(misalignment)

        if misalignments:
            return float(np.mean(misalignments))
        return 0.0

    def _test_regime_transition(
        self,
        small_dists: np.ndarray,
        large_dists: np.ndarray,
        dim: int
    ) -> float:
        """
        Test smoothness of transition between regimes.

        In a smooth fiber bundle, distance scaling should be predictable.
        """
        mean_small = np.mean(small_dists)
        mean_large = np.mean(large_dists)

        if mean_small == 0:
            return 0.0

        # Expected scaling for d-dimensional manifold
        # Based on volume growth: V(r) ∝ r^d
        ratio_expected = (len(large_dists) / len(small_dists)) ** (1.0 / dim)
        ratio_actual = mean_large / mean_small

        # Log-scale deviation
        if ratio_actual > 0 and ratio_expected > 0:
            deviation = abs(np.log(ratio_actual) - np.log(ratio_expected))
        else:
            deviation = 0.0

        return float(deviation)

    def _combine_statistics(
        self,
        dim_score: float,
        curv_score: float,
        tangent_score: float,
        transition_score: float
    ) -> float:
        """
        Combine component scores into overall test statistic.

        Uses weighted combination with normalization.
        """
        # Weights based on importance for fiber bundle structure
        weights = np.array([0.3, 0.25, 0.3, 0.15])
        scores = np.array([dim_score, curv_score, tangent_score, transition_score])

        # Normalize scores to [0, 1] using sigmoid-like transform
        normalized = scores / (1 + scores)

        # Weighted combination
        combined = np.sum(weights * normalized)

        return float(combined)

    def _bootstrap_p_value(
        self,
        data: np.ndarray,
        observed_stat: float,
        distances: np.ndarray
    ) -> float:
        """
        Estimate p-value using bootstrap under null hypothesis.

        Null: Data comes from smooth fiber bundle.
        """
        bootstrap_stats = []
        n_points = len(data)

        for _ in range(self.n_bootstrap):
            # Sample random point for null distribution
            rand_idx = np.random.randint(0, n_points)

            # Get its neighbors
            nbr_idx, _ = self._get_k_neighbors(
                distances[rand_idx], self.n_neighbors_small
            )

            # Quick test statistic (simplified for speed)
            dim = self._estimate_intrinsic_dimension(data[nbr_idx])
            angles = []

            for i in range(min(3, len(nbr_idx)-1)):
                v1 = data[nbr_idx[i]] - data[rand_idx]
                v2 = data[nbr_idx[i+1]] - data[rand_idx]
                cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
                angles.append(np.arccos(np.clip(cos_a, -1, 1)))

            angle_std = np.std(angles) if angles else 0
            quick_stat = (dim / data.shape[1]) * 0.5 + angle_std * 0.5
            bootstrap_stats.append(quick_stat)

        # P-value: proportion of bootstrap samples >= observed
        p_value = np.mean([s >= observed_stat for s in bootstrap_stats])

        # Ensure valid p-value
        p_value = np.clip(p_value, 1.0 / (self.n_bootstrap + 1), 1.0)

        return float(p_value)

    def _identify_regime(
        self,
        small_radius: float,
        large_radius: float,
        dim: int
    ) -> str:
        """Identify spatial regime based on neighborhood sizes."""
        # Characteristic length scale
        char_length = np.sqrt(dim)

        if small_radius < char_length / 10:
            return "small_radius"
        elif large_radius > char_length * 2:
            return "large_radius"
        else:
            return "boundary"

    def _estimate_intrinsic_dimension(self, points: np.ndarray) -> float:
        """
        Estimate intrinsic dimension using Maximum Likelihood Estimation (MLE).

        Uses the Levina-Bickel MLE method which is more robust than PCA
        for manifold dimension estimation.

        Reference: Levina & Bickel (2005) "Maximum Likelihood Estimation of
        Intrinsic Dimension"
        """
        if len(points) < 2:
            return float(points.shape[1])

        from sklearn.neighbors import NearestNeighbors
        from scipy.special import digamma

        n_points = len(points)
        k = min(10, n_points - 1)  # Number of neighbors

        # Use sklearn's NearestNeighbors for robust distance computation
        nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree')
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)

        # Remove self (first column) and get max distance to kth neighbor
        distances = distances[:, 1:]  # Remove self
        max_dists = distances[:, -1]  # Distance to kth neighbor

        # MLE dimension estimation
        # d_hat = -1 / (1/n * sum_i(log(T_k(i)/T_j(i))))
        dimensions = []
        for i in range(n_points):
            # Get distances to k neighbors for point i
            dists_i = distances[i]
            # Avoid log(0)
            dists_i = dists_i[dists_i > 0]
            if len(dists_i) > 1:
                # Compute local dimension estimate
                log_ratios = np.log(max_dists[i] / dists_i[:-1])
                if len(log_ratios) > 0 and np.sum(log_ratios) > 0:
                    local_dim = (len(log_ratios) - 1) / np.sum(log_ratios)
                    dimensions.append(local_dim)

        if dimensions:
            # Return median for robustness against outliers
            return float(np.median(dimensions))

        # Fallback to PCA if MLE fails
        try:
            pca = PCA()
            pca.fit(points - np.mean(points, axis=0))

            # Estimate dimension from explained variance
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            dim = np.argmax(cumsum >= 0.95) + 1

            return float(min(dim, points.shape[1]))
        except:
            return float(points.shape[1])

    def _estimate_tangent_space(
        self,
        centered_points: np.ndarray
    ) -> np.ndarray:
        """
        Estimate tangent space using PCA.

        Returns basis vectors for tangent space.
        """
        if len(centered_points) < 2:
            return np.eye(centered_points.shape[1])[:, :1]

        try:
            pca = PCA(n_components=min(self.n_tangent_dims,
                                      len(centered_points) - 1,
                                      centered_points.shape[1]))
            pca.fit(centered_points)
            return pca.components_.T
        except:
            # Fallback to identity
            return np.eye(centered_points.shape[1])[:, :self.n_tangent_dims]

    def _calculate_confidence(
        self,
        p_value: float,
        n_neighbors: int,
        regime: str
    ) -> float:
        """
        Calculate confidence in test result.

        Based on p-value strength and data availability.
        """
        # P-value contribution (stronger rejection = higher confidence)
        if p_value < 0.001:
            p_conf = 0.95
        elif p_value < 0.01:
            p_conf = 0.85
        elif p_value < 0.05:
            p_conf = 0.75
        elif p_value > 0.95:
            p_conf = 0.7  # Strong non-rejection also informative
        else:
            p_conf = 0.5

        # Sample size contribution
        sample_conf = min(n_neighbors / 20, 1.0)

        # Regime contribution (boundary is less certain)
        regime_conf = {"small_radius": 0.9, "large_radius": 0.9, "boundary": 0.7}[regime]

        # Combine factors
        confidence = p_conf * 0.5 + sample_conf * 0.3 + regime_conf * 0.2

        return float(confidence)


def batch_test_fiber_bundle(
    data: np.ndarray,
    test_indices: Optional[List[int]] = None,
    alpha: float = 0.05,
    n_neighbors_small: int = 10,
    n_neighbors_large: int = 50,
    n_bootstrap: int = 50,
    verbose: bool = False
) -> Dict[int, FiberBundleTestResult]:
    """
    Test multiple points for fiber bundle structure.

    Args:
        data: (n_points, n_dims) array
        test_indices: Indices to test (None = all)
        alpha: Significance level
        n_neighbors_small: Small radius neighbors
        n_neighbors_large: Large radius neighbors
        n_bootstrap: Bootstrap samples
        verbose: Print progress

    Returns:
        Dictionary mapping indices to test results
    """
    n_points = len(data)

    # Default to testing all points
    if test_indices is None:
        test_indices = list(range(n_points))

    # Pre-compute distance matrix for efficiency
    if verbose:
        print(f"Computing distance matrix for {n_points} points...")
    distances = cdist(data, data, metric='euclidean')

    # Initialize tester
    tester = FiberBundleTest(
        alpha=alpha,
        n_neighbors_small=n_neighbors_small,
        n_neighbors_large=n_neighbors_large,
        n_bootstrap=n_bootstrap
    )

    # Test each point
    results = {}
    for i, idx in enumerate(test_indices):
        if verbose and i % 10 == 0:
            print(f"Testing point {i+1}/{len(test_indices)}...")

        result = tester.test_point(data, idx, distances)
        results[idx] = result

    if verbose:
        n_rejected = sum(r.reject_null for r in results.values())
        print(f"\nRejected null hypothesis for {n_rejected}/{len(results)} points")
        print(f"Rejection rate: {n_rejected/len(results):.2%}")

    return results


if __name__ == "__main__":
    # Example usage
    print("Fiber Bundle Core Module")
    print("=" * 50)

    # Generate example data: mixture of manifold and non-manifold regions
    np.random.seed(42)

    # Smooth manifold region
    n_manifold = 100
    t = np.linspace(0, 4*np.pi, n_manifold)
    manifold_data = np.column_stack([
        np.sin(t) + 0.1*np.random.randn(n_manifold),
        np.cos(t) + 0.1*np.random.randn(n_manifold),
        t/10 + 0.1*np.random.randn(n_manifold)
    ])

    # Irregular region (violates fiber bundle)
    n_irregular = 50
    irregular_data = np.random.randn(n_irregular, 3) * 2

    # Combine
    data = np.vstack([manifold_data, irregular_data])

    print(f"Testing {len(data)} points (first {n_manifold} smooth, last {n_irregular} irregular)")
    print()

    # Run tests
    results = batch_test_fiber_bundle(
        data,
        test_indices=[0, 50, 99, 100, 125, 149],  # Sample from both regions
        verbose=True
    )

    print("\nSample results:")
    for idx, result in results.items():
        region = "smooth" if idx < n_manifold else "irregular"
        print(f"\nPoint {idx} ({region} region):")
        print(f"  p-value: {result.p_value:.4f}")
        print(f"  Reject null: {result.reject_null}")
        print(f"  Regime: {result.regime}")
        print(f"  Local dimension: {result.local_dimension:.2f}")