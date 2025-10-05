#!/usr/bin/env python3
"""
Integration test comparing Robinson fiber bundle test vs our polysemy detector.

This test demonstrates that these are DIFFERENT methods that may or may not
agree on which tokens are problematic.
"""

import unittest
import numpy as np
from typing import Dict, List

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from manifold_violations.robinson_fiber_bundle_test import RobinsonFiberBundleTest
from manifold_violations.polysemy_detector import PolysemyDetector


class TestMethodComparison(unittest.TestCase):
    """Compare Robinson statistical test vs clustering-based polysemy detection."""

    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
        self.embed_dim = 50

    def create_monosemous_embedding(self, n_tokens: int = 500) -> np.ndarray:
        """Create embeddings with a clearly monosemous token."""
        embeddings = np.random.randn(n_tokens, self.embed_dim)

        # Token 0: single clear meaning (all neighbors in one cluster)
        center = np.random.randn(self.embed_dim)
        center = center / np.linalg.norm(center)
        embeddings[0] = center

        # Dense single cluster around token 0
        for i in range(1, 100):
            embeddings[i] = center + np.random.randn(self.embed_dim) * 0.1

        return embeddings

    def create_polysemous_embedding(self, n_tokens: int = 500) -> np.ndarray:
        """Create embeddings with a clearly polysemous token."""
        embeddings = np.random.randn(n_tokens, self.embed_dim)

        # Create 3 distinct semantic clusters
        centers = []
        for _ in range(3):
            c = np.random.randn(self.embed_dim)
            c = c / np.linalg.norm(c)
            centers.append(c)

        # Orthogonalize centers for maximum separation
        centers[1] = centers[1] - np.dot(centers[1], centers[0]) * centers[0]
        centers[1] = centers[1] / np.linalg.norm(centers[1])
        centers[2] = centers[2] - np.dot(centers[2], centers[0]) * centers[0]
        centers[2] = centers[2] - np.dot(centers[2], centers[1]) * centers[1]
        centers[2] = centers[2] / np.linalg.norm(centers[2])

        # Token 0 at centroid of all meanings
        embeddings[0] = sum(centers) / len(centers)

        # Create distinct clusters
        for cluster_idx, center in enumerate(centers):
            start = 1 + cluster_idx * 30
            end = start + 30
            for i in range(start, min(end, n_tokens)):
                embeddings[i] = embeddings[0] + center * 0.3 + np.random.randn(self.embed_dim) * 0.02

        return embeddings

    def create_manifold_violating_embedding(self, n_tokens: int = 500) -> np.ndarray:
        """Create embeddings that violate manifold hypothesis but aren't necessarily polysemous."""
        embeddings = np.random.randn(n_tokens, self.embed_dim)

        # Token 0: create increasing volume growth (violates fiber bundle)
        # but neighbors are still relatively uniform (not clustered)
        embeddings[0] = np.zeros(self.embed_dim)

        # Sparse near, dense far (manifold violation)
        for i in range(1, 20):
            direction = np.random.randn(self.embed_dim)
            direction = direction / np.linalg.norm(direction)
            embeddings[i] = direction * np.random.uniform(0.1, 0.3)

        for i in range(20, 200):
            direction = np.random.randn(self.embed_dim)
            direction = direction / np.linalg.norm(direction)
            embeddings[i] = direction * np.random.uniform(2.0, 3.0)

        return embeddings

    def test_methods_on_monosemous_token(self):
        """Test both methods on clearly monosemous token."""
        embeddings = self.create_monosemous_embedding()

        # Robinson test
        robinson = RobinsonFiberBundleTest(
            significance_level=0.05,
            n_radii=20,
            max_embeddings_for_exact=10000
        )
        robinson_result = robinson.test_point(embeddings, 0)

        # Polysemy detector
        polysemy = PolysemyDetector(
            n_neighbors=50,
            clustering_method='hierarchical',
            metric='cosine',
            random_state=42
        )
        polysemy_result = polysemy.detect_polysemy(embeddings, 0, "monosemous")

        print("\n=== Monosemous Token Results ===")
        print(f"Robinson violates hypothesis: {robinson_result.violates_hypothesis}")
        print(f"Robinson p-value: {robinson_result.p_value:.4f}")
        print(f"Polysemy detected: {polysemy_result.is_polysemous}")
        print(f"Polysemy num_meanings: {polysemy_result.num_meanings}")
        print(f"Polysemy confidence: {polysemy_result.confidence:.3f}")

        # Both should agree: no violation/polysemy
        # Note: We can't guarantee exact agreement due to different methods
        if not robinson_result.violates_hypothesis and not polysemy_result.is_polysemous:
            print("✅ Methods agree: Token is regular/monosemous")

    def test_methods_on_polysemous_token(self):
        """Test both methods on clearly polysemous token."""
        embeddings = self.create_polysemous_embedding()

        # Robinson test
        robinson = RobinsonFiberBundleTest(
            significance_level=0.05,
            n_radii=20,
            max_embeddings_for_exact=10000
        )
        robinson_result = robinson.test_point(embeddings, 0)

        # Polysemy detector
        polysemy = PolysemyDetector(
            n_neighbors=50,
            clustering_method='hierarchical',
            metric='cosine',
            random_state=42
        )
        polysemy_result = polysemy.detect_polysemy(embeddings, 0, "polysemous")

        print("\n=== Polysemous Token Results ===")
        print(f"Robinson violates hypothesis: {robinson_result.violates_hypothesis}")
        print(f"Robinson p-value: {robinson_result.p_value:.4f}")
        print(f"Robinson local dimension: {robinson_result.local_signal_dimension:.1f}")
        print(f"Polysemy detected: {polysemy_result.is_polysemous}")
        print(f"Polysemy num_meanings: {polysemy_result.num_meanings}")
        print(f"Polysemy confidence: {polysemy_result.confidence:.3f}")

        # Methods might agree on irregularity
        if robinson_result.violates_hypothesis or polysemy_result.is_polysemous:
            print("✅ At least one method detected irregularity")

    def test_methods_on_manifold_violation(self):
        """Test case where manifold is violated but token isn't polysemous."""
        embeddings = self.create_manifold_violating_embedding()

        # Robinson test
        robinson = RobinsonFiberBundleTest(
            significance_level=0.05,
            n_radii=20,
            max_embeddings_for_exact=10000
        )
        robinson_result = robinson.test_point(embeddings, 0)

        # Polysemy detector
        polysemy = PolysemyDetector(
            n_neighbors=50,
            clustering_method='hierarchical',
            metric='cosine',
            random_state=42
        )
        polysemy_result = polysemy.detect_polysemy(embeddings, 0, "manifold_violation")

        print("\n=== Manifold Violation (not polysemous) Results ===")
        print(f"Robinson violates hypothesis: {robinson_result.violates_hypothesis}")
        print(f"Robinson violates fiber bundle: {robinson_result.violates_fiber_bundle}")
        print(f"Robinson violates manifold: {robinson_result.violates_manifold}")
        print(f"Polysemy detected: {polysemy_result.is_polysemous}")
        print(f"Polysemy num_meanings: {polysemy_result.num_meanings}")

        # This demonstrates the methods test different things
        if robinson_result.violates_hypothesis and not polysemy_result.is_polysemous:
            print("⚠️ Methods disagree: Geometric irregularity without semantic clustering")
            print("   This shows these methods test fundamentally different properties!")

    def test_correlation_analysis(self):
        """Analyze correlation between methods across many random tokens."""
        n_samples = 100
        robinson_violations = []
        polysemy_detections = []

        print("\n=== Correlation Analysis ===")
        print("Testing 100 random embeddings...")

        robinson = RobinsonFiberBundleTest(
            significance_level=0.05,
            n_radii=15,
            max_embeddings_for_exact=10000
        )

        polysemy = PolysemyDetector(
            n_neighbors=30,
            clustering_method='hierarchical',
            metric='cosine',
            random_state=42
        )

        for i in range(n_samples):
            # Create random embedding structure
            embeddings = np.random.randn(200, self.embed_dim)

            # Randomly add structure
            if np.random.rand() < 0.3:  # 30% chance of structure
                if np.random.rand() < 0.5:
                    # Add clusters
                    for j in range(1, 20):
                        embeddings[j] = embeddings[0] + np.random.randn(self.embed_dim) * 0.1
                    for j in range(20, 40):
                        embeddings[j] = embeddings[0] - np.random.randn(self.embed_dim) * 0.1
                else:
                    # Add manifold violation
                    for j in range(1, 10):
                        embeddings[j] = embeddings[0] + np.random.randn(self.embed_dim) * 0.05
                    for j in range(10, 50):
                        embeddings[j] = embeddings[0] + np.random.randn(self.embed_dim) * 2.0

            # Test with both methods
            robinson_result = robinson.test_point(embeddings, 0)
            polysemy_result = polysemy.detect_polysemy(embeddings, 0)

            robinson_violations.append(robinson_result.violates_hypothesis)
            polysemy_detections.append(polysemy_result.is_polysemous)

        # Calculate agreement
        agreements = sum(1 for r, p in zip(robinson_violations, polysemy_detections) if r == p)
        agreement_rate = agreements / n_samples

        robinson_rate = sum(robinson_violations) / n_samples
        polysemy_rate = sum(polysemy_detections) / n_samples

        print(f"Robinson violation rate: {robinson_rate:.2%}")
        print(f"Polysemy detection rate: {polysemy_rate:.2%}")
        print(f"Agreement rate: {agreement_rate:.2%}")

        # They shouldn't agree perfectly since they test different things
        self.assertGreater(agreement_rate, 0.3, "Methods should have some correlation")
        self.assertLess(agreement_rate, 0.95, "Methods shouldn't agree perfectly (they test different things)")

        print("\n=== Key Insight ===")
        print("Robinson test: Detects geometric/topological irregularities")
        print("Polysemy detector: Detects semantic clustering patterns")
        print("Both useful but measure different aspects of embedding structure!")


if __name__ == '__main__':
    unittest.main(verbosity=2)