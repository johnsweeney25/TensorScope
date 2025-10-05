#!/usr/bin/env python3
"""
Tangent Space Analysis for Manifold Structure

This module provides proper tangent space construction and analysis,
adding mathematical rigor to the manifold violation detection framework.
"""

import numpy as np
import torch
from typing import Tuple, Dict, List, Optional
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class TangentSpaceAnalyzer:
    """
    Construct and analyze tangent spaces for manifold analysis.

    This provides the mathematical foundation missing from basic
    volume growth tests by properly constructing local linear approximations.
    """

    def __init__(self, k_neighbors: int = 20, epsilon: float = 1e-8):
        """
        Initialize tangent space analyzer.

        Args:
            k_neighbors: Number of neighbors for local tangent space
            epsilon: Small value for numerical stability
        """
        self.k_neighbors = k_neighbors
        self.epsilon = epsilon

    def estimate_tangent_space(
        self,
        points: np.ndarray,
        point_idx: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Estimate tangent space at a point using local PCA.

        This constructs the tangent space as the span of principal
        directions in the local neighborhood.

        Args:
            points: (N, D) array of points
            point_idx: Index of point to analyze

        Returns:
            tangent_basis: (D, d) orthonormal basis for tangent space
            eigenvalues: Eigenvalues indicating local variance
            local_dimension: Estimated local intrinsic dimension
        """
        n_points, ambient_dim = points.shape

        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, n_points))
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points[point_idx:point_idx+1])

        # Exclude the point itself
        neighbor_indices = indices[0, 1:]
        local_points = points[neighbor_indices]

        # Center points at query point
        centered = local_points - points[point_idx]

        # Perform PCA to find principal directions
        pca = PCA(n_components=min(len(neighbor_indices), ambient_dim))
        pca.fit(centered)

        # Estimate local dimension using explained variance
        explained_variance_ratio = pca.explained_variance_ratio_
        cumsum = np.cumsum(explained_variance_ratio)

        # Find dimension that explains 95% of variance
        local_dim = np.argmax(cumsum >= 0.95) + 1

        # Get tangent basis (principal components)
        tangent_basis = pca.components_[:local_dim].T

        return tangent_basis, pca.explained_variance_, float(local_dim)

    def compute_geodesic_distance(
        self,
        points: np.ndarray,
        idx1: int,
        idx2: int,
        n_steps: int = 10
    ) -> float:
        """
        Approximate geodesic distance between two points.

        Uses a simple shortest-path approximation through k-NN graph.

        Args:
            points: (N, D) array of points
            idx1, idx2: Indices of points
            n_steps: Number of interpolation steps

        Returns:
            Approximate geodesic distance
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        # Build k-NN graph
        n_points = len(points)
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors, n_points))
        nbrs.fit(points)

        # Get distances and connectivity
        distances, indices = nbrs.kneighbors(points)

        # Create sparse adjacency matrix
        row_ind = np.repeat(np.arange(n_points), self.k_neighbors)
        col_ind = indices.flatten()
        data = distances.flatten()

        graph = csr_matrix((data, (row_ind, col_ind)), shape=(n_points, n_points))

        # Compute shortest path
        dist_matrix = shortest_path(graph, directed=False, indices=[idx1, idx2])

        return dist_matrix[0, 1] if dist_matrix[0, 1] != np.inf else np.linalg.norm(points[idx1] - points[idx2])

    def parallel_transport_vector(
        self,
        points: np.ndarray,
        start_idx: int,
        end_idx: int,
        vector: np.ndarray
    ) -> np.ndarray:
        """
        Approximate parallel transport of a vector along the manifold.

        This is a simplified version that projects through tangent spaces.

        Args:
            points: (N, D) array of points
            start_idx: Starting point index
            end_idx: Ending point index
            vector: Vector to transport

        Returns:
            Transported vector at end point
        """
        # Get tangent spaces at both points
        basis_start, _, _ = self.estimate_tangent_space(points, start_idx)
        basis_end, _, _ = self.estimate_tangent_space(points, end_idx)

        # Project vector to start tangent space
        coeffs = basis_start.T @ vector

        # Reconstruct in end tangent space
        # This is simplified - true parallel transport would follow geodesic
        min_dim = min(len(coeffs), basis_end.shape[1])
        transported = basis_end[:, :min_dim] @ coeffs[:min_dim]

        return transported

    def compute_sectional_curvature(
        self,
        points: np.ndarray,
        point_idx: int,
        plane_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None
    ) -> float:
        """
        Estimate sectional curvature for a 2-plane in the tangent space.

        This uses the change in angle between vectors under parallel transport.

        Args:
            points: (N, D) array of points
            point_idx: Point at which to compute curvature
            plane_vectors: Optional pair of tangent vectors defining the plane

        Returns:
            Estimated sectional curvature
        """
        # Get tangent space
        tangent_basis, _, local_dim = self.estimate_tangent_space(points, point_idx)

        if local_dim < 2:
            return 0.0  # Not enough dimensions for curvature

        # Use first two principal directions if not specified
        if plane_vectors is None:
            v1 = tangent_basis[:, 0]
            v2 = tangent_basis[:, 1]
        else:
            v1, v2 = plane_vectors

        # Normalize vectors
        v1 = v1 / (np.linalg.norm(v1) + self.epsilon)
        v2 = v2 / (np.linalg.norm(v2) + self.epsilon)

        # Find nearby points for transport
        nbrs = NearestNeighbors(n_neighbors=5)
        nbrs.fit(points)
        _, indices = nbrs.kneighbors(points[point_idx:point_idx+1])

        # Transport vectors to neighbors and back
        angle_changes = []
        for neighbor_idx in indices[0, 1:]:
            # Transport v1 and v2 to neighbor and back
            v1_transported = self.parallel_transport_vector(points, point_idx, neighbor_idx, v1)
            v1_back = self.parallel_transport_vector(points, neighbor_idx, point_idx, v1_transported)

            v2_transported = self.parallel_transport_vector(points, point_idx, neighbor_idx, v2)
            v2_back = self.parallel_transport_vector(points, neighbor_idx, point_idx, v2_transported)

            # Measure angle change
            cos_angle_original = np.dot(v1, v2)
            cos_angle_transported = np.dot(v1_back, v2_back) / (
                np.linalg.norm(v1_back) * np.linalg.norm(v2_back) + self.epsilon
            )

            angle_change = np.arccos(np.clip(cos_angle_transported, -1, 1)) - \
                          np.arccos(np.clip(cos_angle_original, -1, 1))

            # Normalize by distance
            dist = np.linalg.norm(points[neighbor_idx] - points[point_idx])
            if dist > self.epsilon:
                angle_changes.append(angle_change / (dist ** 2))

        # Average curvature estimate
        return np.mean(angle_changes) if angle_changes else 0.0

    def analyze_manifold_structure(
        self,
        points: np.ndarray,
        n_samples: int = 10
    ) -> Dict[str, float]:
        """
        Comprehensive manifold structure analysis.

        Args:
            points: (N, D) array of points
            n_samples: Number of points to sample for analysis

        Returns:
            Dictionary of manifold metrics
        """
        n_points = len(points)
        sample_indices = np.random.choice(n_points, min(n_samples, n_points), replace=False)

        dimensions = []
        curvatures = []

        for idx in sample_indices:
            # Estimate local dimension
            _, _, local_dim = self.estimate_tangent_space(points, idx)
            dimensions.append(local_dim)

            # Estimate curvature
            curvature = self.compute_sectional_curvature(points, idx)
            curvatures.append(curvature)

        return {
            'mean_dimension': np.mean(dimensions),
            'std_dimension': np.std(dimensions),
            'mean_curvature': np.mean(curvatures),
            'std_curvature': np.std(curvatures),
            'dimension_variation': np.std(dimensions) / (np.mean(dimensions) + 1e-8),
            'is_flat': abs(np.mean(curvatures)) < 0.01
        }


def test_on_known_manifolds():
    """Test the analyzer on known manifolds."""

    analyzer = TangentSpaceAnalyzer()

    # Test 1: Sphere (should have constant positive curvature)
    print("Testing on 2-sphere embedded in 3D:")
    n_points = 500
    theta = np.random.uniform(0, np.pi, n_points)
    phi = np.random.uniform(0, 2*np.pi, n_points)
    sphere = np.column_stack([
        np.sin(theta) * np.cos(phi),
        np.sin(theta) * np.sin(phi),
        np.cos(theta)
    ])

    sphere_analysis = analyzer.analyze_manifold_structure(sphere)
    print(f"  Dimension: {sphere_analysis['mean_dimension']:.2f} (expected: 2)")
    print(f"  Curvature: {sphere_analysis['mean_curvature']:.4f} (expected: positive)")

    # Test 2: Swiss roll (should have zero curvature)
    print("\nTesting on Swiss roll:")
    t = np.random.uniform(0, 4*np.pi, n_points)
    height = np.random.uniform(0, 2, n_points)
    swiss_roll = np.column_stack([
        t * np.cos(t),
        height,
        t * np.sin(t)
    ])

    roll_analysis = analyzer.analyze_manifold_structure(swiss_roll)
    print(f"  Dimension: {roll_analysis['mean_dimension']:.2f} (expected: 2)")
    print(f"  Is flat: {roll_analysis['is_flat']} (expected: True)")

    # Test 3: Random high-dimensional data (should violate manifold structure)
    print("\nTesting on random 10D data:")
    random_data = np.random.randn(n_points, 10)

    random_analysis = analyzer.analyze_manifold_structure(random_data)
    print(f"  Dimension: {random_analysis['mean_dimension']:.2f} (expected: ~10)")
    print(f"  Dimension variation: {random_analysis['dimension_variation']:.2f} (expected: high)")


if __name__ == "__main__":
    test_on_known_manifolds()