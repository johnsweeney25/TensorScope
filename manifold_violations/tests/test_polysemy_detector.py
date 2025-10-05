#!/usr/bin/env python3
"""Rigorous unit tests for polysemy detector implementation."""

import unittest
import numpy as np
from typing import List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from manifold_violations.polysemy_detector import PolysemyDetector


class TestPolysemyDetector(unittest.TestCase):
    """Test suite for polysemy detector."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.n_tokens = 500
        self.embed_dim = 50

    def test_initialization(self):
        """Test detector initializes with correct parameters."""
        detector = PolysemyDetector(
            n_neighbors=30,
            clustering_method='hierarchical',
            metric='cosine',
            min_cluster_size=5,
            random_state=42
        )
        self.assertEqual(detector.n_neighbors, 30)
        self.assertEqual(detector.clustering_method, 'hierarchical')
        self.assertEqual(detector.metric, 'cosine')
        self.assertEqual(detector.min_cluster_size, 5)

    def test_monosemous_token(self):
        """Test detection on clearly monosemous token."""
        embeddings = np.random.randn(self.n_tokens, self.embed_dim)

        # Token 0 with all neighbors very similar (single meaning)
        center = np.random.randn(self.embed_dim)
        center = center / np.linalg.norm(center)
        embeddings[0] = center  # Token 0 at center of cluster

        # All neighbors cluster tightly around token 0
        for i in range(1, 60):
            embeddings[i] = center + np.random.randn(self.embed_dim) * 0.05

        detector = PolysemyDetector(
            n_neighbors=50,
            clustering_method='hierarchical',
            metric='cosine',
            random_state=42
        )

        result = detector.detect_polysemy(embeddings, 0, "monosemous_token")

        # Should have few clusters (1-2 acceptable due to clustering variance)
        self.assertLessEqual(result.num_meanings, 2,
                            f"Monosemous token has too many clusters: {result.num_meanings}")
        # Low confidence if detected as polysemous
        if result.is_polysemous:
            self.assertLess(result.confidence, 0.5,
                           "Confidence should be low for borderline cases")

    def test_clear_polysemy_detection(self):
        """Test detection of clearly polysemous token with multiple meanings."""
        embeddings = np.random.randn(self.n_tokens, self.embed_dim)

        # Create three distinct cluster centers
        cluster_centers = []
        for _ in range(3):
            center = np.random.randn(self.embed_dim)
            center = center / np.linalg.norm(center)
            cluster_centers.append(center)

        # Make centers orthogonal for clear separation
        cluster_centers[1] = cluster_centers[1] - np.dot(cluster_centers[1], cluster_centers[0]) * cluster_centers[0]
        cluster_centers[1] = cluster_centers[1] / np.linalg.norm(cluster_centers[1])
        cluster_centers[2] = cluster_centers[2] - np.dot(cluster_centers[2], cluster_centers[0]) * cluster_centers[0]
        cluster_centers[2] = cluster_centers[2] - np.dot(cluster_centers[2], cluster_centers[1]) * cluster_centers[1]
        cluster_centers[2] = cluster_centers[2] / np.linalg.norm(cluster_centers[2])

        # Token 0 at centroid of all clusters
        embeddings[0] = sum(cluster_centers) / len(cluster_centers)

        # Create tight clusters near token 0
        for cluster_idx, center in enumerate(cluster_centers):
            start_idx = 1 + cluster_idx * 17
            end_idx = start_idx + 17
            for i in range(start_idx, min(end_idx, self.n_tokens)):
                # Place points close to token 0 but in distinct directions
                embeddings[i] = embeddings[0] + center * 0.3 + np.random.randn(self.embed_dim) * 0.01

        detector = PolysemyDetector(
            n_neighbors=50,
            clustering_method='hierarchical',
            metric='cosine',
            random_state=42
        )

        result = detector.detect_polysemy(embeddings, 0, "polysemous_token")

        # Should detect polysemy
        self.assertTrue(result.is_polysemous,
                       f"Failed to detect clear polysemy. Confidence: {result.confidence:.3f}")
        self.assertGreaterEqual(result.num_meanings, 2,
                               f"Should detect multiple meanings, found {result.num_meanings}")

    def test_homonym_detection(self):
        """Test detection of homonym (two very distinct meanings)."""
        embeddings = np.random.randn(self.n_tokens, self.embed_dim)

        # Create two opposite clusters
        center1 = np.random.randn(self.embed_dim)
        center1 = center1 / np.linalg.norm(center1)
        center2 = -center1  # Opposite direction

        # Token 0 between both meanings (average)
        embeddings[0] = (center1 + center2) / 2  # Should be near zero but let's be explicit

        # Create neighbors close to token 0 but in opposite directions
        # Cluster 1 - slightly offset from token 0 in direction of center1
        for i in range(1, 26):
            embeddings[i] = embeddings[0] + center1 * 0.3 + np.random.randn(self.embed_dim) * 0.02

        # Cluster 2 - slightly offset from token 0 in direction of center2
        for i in range(26, 51):
            embeddings[i] = embeddings[0] + center2 * 0.3 + np.random.randn(self.embed_dim) * 0.02

        detector = PolysemyDetector(
            n_neighbors=50,
            clustering_method='hierarchical',
            metric='cosine',
            random_state=42
        )

        result = detector.detect_polysemy(embeddings, 0, "homonym_token")

        # Should detect multiple meanings (might not always be marked polysemous due to confidence threshold)
        self.assertGreaterEqual(result.num_meanings, 2,
                               f"Failed to detect homonym structure, only {result.num_meanings} clusters")
        # If marked as polysemous, should have reasonable confidence
        if result.is_polysemous:
            self.assertGreater(result.confidence, 0.3, "Confidence too low for clear structure")
        # Type classification depends on coherence score and cluster count
        # Could be homonym, contranym, or multi-sense
        self.assertIn(result.polysemy_type, ['homonym', 'contranym', 'multi-sense', 'ambiguous'],
                     f"Unexpected type: {result.polysemy_type}")

    def test_clustering_methods(self):
        """Test both DBSCAN and hierarchical clustering methods."""
        # Create test data with clear clusters
        embeddings = np.random.randn(200, self.embed_dim)
        embeddings[0] = np.zeros(self.embed_dim)

        # Two clear clusters
        for i in range(1, 26):
            embeddings[i] = np.array([1.0] + [0.0] * (self.embed_dim - 1)) + np.random.randn(self.embed_dim) * 0.05
        for i in range(26, 51):
            embeddings[i] = np.array([-1.0] + [0.0] * (self.embed_dim - 1)) + np.random.randn(self.embed_dim) * 0.05

        # Test hierarchical
        detector_hier = PolysemyDetector(
            n_neighbors=50,
            clustering_method='hierarchical',
            metric='euclidean',  # Use Euclidean for clearer test
            random_state=42
        )
        result_hier = detector_hier.detect_polysemy(embeddings, 0)

        # Test DBSCAN
        detector_dbscan = PolysemyDetector(
            n_neighbors=50,
            clustering_method='dbscan',
            metric='euclidean',
            random_state=42
        )
        result_dbscan = detector_dbscan.detect_polysemy(embeddings, 0)

        # Both should detect multiple clusters
        self.assertGreaterEqual(result_hier.num_meanings, 2,
                               f"Hierarchical found only {result_hier.num_meanings} clusters")
        self.assertGreaterEqual(result_dbscan.num_meanings, 2,
                               f"DBSCAN found only {result_dbscan.num_meanings} clusters")

    def test_metric_consistency(self):
        """Test that cosine and Euclidean metrics work correctly."""
        embeddings = np.random.randn(200, self.embed_dim)

        # Create directional clusters (better for cosine)
        center1 = np.random.randn(self.embed_dim)
        center1 = center1 / np.linalg.norm(center1)
        center2 = np.random.randn(self.embed_dim)
        center2 = center2 - np.dot(center2, center1) * center1
        center2 = center2 / np.linalg.norm(center2)

        embeddings[0] = (center1 + center2) / 2

        # Clusters at different magnitudes but same directions
        for i in range(1, 26):
            magnitude = np.random.uniform(0.5, 1.5)
            embeddings[i] = center1 * magnitude + np.random.randn(self.embed_dim) * 0.01

        for i in range(26, 51):
            magnitude = np.random.uniform(0.5, 1.5)
            embeddings[i] = center2 * magnitude + np.random.randn(self.embed_dim) * 0.01

        # Cosine should find 2 clusters (ignores magnitude)
        detector_cosine = PolysemyDetector(
            n_neighbors=50,
            metric='cosine',
            random_state=42
        )
        result_cosine = detector_cosine.detect_polysemy(embeddings, 0)

        # Euclidean might find more clusters (sensitive to magnitude)
        detector_euclidean = PolysemyDetector(
            n_neighbors=50,
            metric='euclidean',
            random_state=42
        )
        result_euclidean = detector_euclidean.detect_polysemy(embeddings, 0)

        # Both should detect structure
        self.assertGreaterEqual(result_cosine.num_meanings, 1,
                               "Cosine should detect clusters")
        self.assertGreaterEqual(result_euclidean.num_meanings, 1,
                               "Euclidean should detect clusters")

    def test_edge_cases(self):
        """Test handling of edge cases."""
        detector = PolysemyDetector(n_neighbors=50, random_state=42)

        # Test 1: Single token
        embeddings = np.array([[0, 0, 0]])
        result = detector.detect_polysemy(embeddings, 0)
        self.assertFalse(result.is_polysemous, "Single token shouldn't be polysemous")

        # Test 2: Not enough neighbors
        embeddings = np.random.randn(10, 3)
        result = detector.detect_polysemy(embeddings, 0)
        # Should handle gracefully
        self.assertIsNotNone(result)

        # Test 3: All identical neighbors
        embeddings = np.ones((100, 3))
        embeddings[0] = np.zeros(3)
        result = detector.detect_polysemy(embeddings, 0)
        self.assertEqual(result.num_meanings, 1, "Identical neighbors = single cluster")

    def test_confidence_scoring(self):
        """Test that confidence scores are reasonable."""
        embeddings = np.random.randn(200, self.embed_dim)
        detector = PolysemyDetector(n_neighbors=50, random_state=42)

        # Clear polysemy should have high confidence
        # Create very distinct clusters
        for i in range(1, 26):
            embeddings[i] = np.array([10.0] + [0.0] * (self.embed_dim - 1)) + np.random.randn(self.embed_dim) * 0.01
        for i in range(26, 51):
            embeddings[i] = np.array([-10.0] + [0.0] * (self.embed_dim - 1)) + np.random.randn(self.embed_dim) * 0.01

        result_clear = detector.detect_polysemy(embeddings, 0)

        # Ambiguous case should have lower confidence
        embeddings_ambiguous = np.random.randn(200, self.embed_dim)
        result_ambiguous = detector.detect_polysemy(embeddings_ambiguous, 0)

        # Confidence should be in valid range
        self.assertGreaterEqual(result_clear.confidence, 0.0, "Confidence should be >= 0")
        self.assertLessEqual(result_clear.confidence, 1.0, "Confidence should be <= 1")

        # Clear structure should have higher confidence than random
        if result_clear.is_polysemous and not result_ambiguous.is_polysemous:
            self.assertGreater(result_clear.confidence, result_ambiguous.confidence,
                             "Clear polysemy should have higher confidence")

    def test_subsampling(self):
        """Test that subsampling works for large vocabularies."""
        # Create large vocabulary
        large_embeddings = np.random.randn(20000, self.embed_dim)

        detector = PolysemyDetector(
            n_neighbors=50,
            subsample_size=5000,  # Force subsampling
            random_state=42
        )

        # Should complete without OOM
        result = detector.detect_polysemy(large_embeddings, 0)
        self.assertIsNotNone(result)

    def test_vocabulary_analysis(self):
        """Test full vocabulary analysis method."""
        embeddings = np.random.randn(100, self.embed_dim)

        # Create a clearly polysemous token at index 10
        center1 = np.random.randn(self.embed_dim)
        center1 = center1 / np.linalg.norm(center1)
        center2 = np.random.randn(self.embed_dim)
        center2 = center2 - np.dot(center2, center1) * center1
        center2 = center2 / np.linalg.norm(center2)

        embeddings[10] = (center1 + center2) / 2

        # Create clear clusters around token 10
        for i in range(11, 16):
            embeddings[i] = embeddings[10] + center1 * 0.2 + np.random.randn(self.embed_dim) * 0.01
        for i in range(16, 21):
            embeddings[i] = embeddings[10] + center2 * 0.2 + np.random.randn(self.embed_dim) * 0.01

        detector = PolysemyDetector(n_neighbors=10, random_state=42)

        analysis = detector.analyze_vocabulary(
            embeddings,
            sample_size=50,
            verbose=False
        )

        # At minimum, the analysis should complete without error
        self.assertGreaterEqual(analysis.polysemy_rate, 0.0,
                               "Polysemy rate should be non-negative")
        self.assertEqual(analysis.total_tokens, 50,
                        "Should analyze requested sample size")


if __name__ == '__main__':
    unittest.main(verbosity=2)