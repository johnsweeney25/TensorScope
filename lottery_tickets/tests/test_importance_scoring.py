"""
Unit tests for importance scoring methods.
==========================================
Tests Fisher, Taylor, magnitude, and hybrid importance scoring.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lottery_tickets.importance_scoring import (
    compute_gradient_importance,
    compute_fisher_importance,
    compute_taylor_importance,
    compute_magnitude_importance,
    compute_gradient_norm_importance,
    compute_hybrid_importance
)
from lottery_tickets.utils import ensure_deterministic_pruning, create_model_wrapper


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class SimpleDataLoader:
    """Simple dataloader for testing."""
    def __init__(self, n_batches=5, batch_size=8, input_dim=10, output_dim=3):
        self.batches = []
        for _ in range(n_batches):
            batch = {
                'input_ids': torch.randn(batch_size, input_dim),
                'labels': torch.randint(0, output_dim, (batch_size,))
            }
            self.batches.append(batch)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self):
        return len(self.batches)


class TestFisherImportance(unittest.TestCase):
    """Test Fisher information importance scoring."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = SimpleModel()
        self.wrapped_model = create_model_wrapper(self.model)
        self.dataloader = SimpleDataLoader(n_batches=5, batch_size=8)

    def test_fisher_computation_basic(self):
        """Test basic Fisher importance computation."""
        scores = compute_fisher_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20,
            chunk_size=100_000
        )

        # Check all weight parameters have scores
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.assertIn(name, scores, f"Missing Fisher score for {name}")
                self.assertEqual(scores[name].shape, param.shape,
                               f"Score shape should match parameter shape for {name}")

    def test_fisher_mixed_precision(self):
        """Test Fisher with mixed precision (FP32 accumulation)."""
        scores = compute_fisher_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20,
            use_mixed_precision=True
        )

        # All scores should be non-negative (Fisher is PSD)
        for name, score_tensor in scores.items():
            self.assertTrue(torch.all(score_tensor >= 0),
                          f"Fisher scores should be non-negative for {name}")

    def test_fisher_gradient_clipping(self):
        """Test Fisher with gradient clipping for stability."""
        # Without clipping
        scores_no_clip = compute_fisher_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20,
            gradient_clip=0
        )

        # With clipping
        scores_clipped = compute_fisher_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20,
            gradient_clip=1.0
        )

        # Clipped scores should generally be smaller or equal
        for name in scores_no_clip:
            max_no_clip = scores_no_clip[name].max().item()
            max_clipped = scores_clipped[name].max().item()

            # Clipping should reduce or maintain max values
            self.assertLessEqual(max_clipped, max_no_clip * 1.1,
                               f"Clipped scores should not be much larger for {name}")

    def test_fisher_chunked_processing(self):
        """Test chunked processing for memory efficiency."""
        # Small chunks
        scores_small = compute_fisher_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20,
            chunk_size=1000  # Very small chunks
        )

        # Large chunks
        scores_large = compute_fisher_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20,
            chunk_size=1_000_000  # Large chunks
        )

        # Results should be similar regardless of chunk size
        for name in scores_small:
            diff = torch.abs(scores_small[name] - scores_large[name]).max().item()
            avg = (scores_small[name].abs().mean() + scores_large[name].abs().mean()).item() / 2

            if avg > 1e-10:  # Only check if values are non-negligible
                relative_diff = diff / avg
                self.assertLess(relative_diff, 0.1,
                              f"Chunk size should not affect results much for {name}")


class TestTaylorImportance(unittest.TestCase):
    """Test Taylor expansion importance scoring."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = SimpleModel()
        self.wrapped_model = create_model_wrapper(self.model)
        self.dataloader = SimpleDataLoader()

    def test_taylor_computation(self):
        """Test Taylor importance computation."""
        scores = compute_taylor_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20
        )

        # Check scores exist for all parameters
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                self.assertIn(name, scores)

                # Taylor scores should be non-negative (absolute values)
                self.assertTrue(torch.all(scores[name] >= 0),
                              f"Taylor scores should be non-negative for {name}")

    def test_taylor_vs_magnitude(self):
        """Test that Taylor differs from simple magnitude."""
        taylor_scores = compute_taylor_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20
        )

        magnitude_scores = compute_magnitude_importance(self.model)

        # Taylor should incorporate gradient information, so should differ
        for name in magnitude_scores:
            if name in taylor_scores:
                # Normalized comparison
                taylor_norm = taylor_scores[name] / (taylor_scores[name].max() + 1e-10)
                mag_norm = magnitude_scores[name] / (magnitude_scores[name].max() + 1e-10)

                # Should not be identical
                self.assertFalse(torch.allclose(taylor_norm, mag_norm, rtol=1e-2),
                               f"Taylor should differ from magnitude for {name}")


class TestMagnitudeImportance(unittest.TestCase):
    """Test magnitude-based importance."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = SimpleModel()

    def test_magnitude_computation(self):
        """Test basic magnitude importance."""
        scores = compute_magnitude_importance(self.model)

        # Should only include weight parameters
        for name in scores:
            self.assertIn('weight', name, f"Should only score weight parameters, got {name}")

        # Scores should be absolute values
        for name, param in self.model.named_parameters():
            if name in scores:
                expected = param.abs()
                self.assertTrue(torch.allclose(scores[name], expected),
                              f"Magnitude scores should be absolute values for {name}")

    def test_magnitude_excludes_bias(self):
        """Test that magnitude scoring excludes bias by default."""
        scores = compute_magnitude_importance(self.model)

        for name in scores:
            self.assertNotIn('bias', name, f"Should not include bias parameters, got {name}")


class TestGradientNormImportance(unittest.TestCase):
    """Test gradient norm importance."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = SimpleModel()
        self.wrapped_model = create_model_wrapper(self.model)
        self.dataloader = SimpleDataLoader()

    def test_gradient_norm_computation(self):
        """Test gradient norm importance."""
        scores = compute_gradient_norm_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20
        )

        # Check all parameters have scores
        for name, param in self.model.named_parameters():
            self.assertIn(name, scores, f"Missing gradient norm score for {name}")

            # Scores should be non-negative
            self.assertTrue(torch.all(scores[name] >= 0),
                          f"Gradient norms should be non-negative for {name}")


class TestHybridImportance(unittest.TestCase):
    """Test hybrid importance combining multiple methods."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = SimpleModel()
        self.wrapped_model = create_model_wrapper(self.model)
        self.dataloader = SimpleDataLoader()

    def test_hybrid_default_weights(self):
        """Test hybrid importance with default weights."""
        scores = compute_hybrid_importance(
            self.wrapped_model,
            self.dataloader,
            num_samples=20
        )

        # Check scores exist
        self.assertGreater(len(scores), 0, "Should compute hybrid scores")

        # Scores should be normalized to [0, 1]
        for name, score_tensor in scores.items():
            min_val = score_tensor.min().item()
            max_val = score_tensor.max().item()

            self.assertGreaterEqual(min_val, 0, f"Scores should be >= 0 for {name}")
            self.assertLessEqual(max_val, 1.01, f"Scores should be <= 1 for {name}")

    def test_hybrid_custom_weights(self):
        """Test hybrid importance with custom weights."""
        # Only magnitude
        scores_mag_only = compute_hybrid_importance(
            self.wrapped_model,
            self.dataloader,
            weights={'magnitude': 1.0, 'fisher': 0.0},
            num_samples=20
        )

        # Only Fisher
        scores_fisher_only = compute_hybrid_importance(
            self.wrapped_model,
            self.dataloader,
            weights={'magnitude': 0.0, 'fisher': 1.0},
            num_samples=20
        )

        # Should be different
        for name in scores_mag_only:
            if name in scores_fisher_only:
                self.assertFalse(
                    torch.allclose(scores_mag_only[name], scores_fisher_only[name]),
                    f"Different weight combinations should give different results for {name}"
                )

    def test_hybrid_normalization(self):
        """Test that hybrid scores are properly normalized."""
        scores = compute_hybrid_importance(
            self.wrapped_model,
            self.dataloader,
            weights={'magnitude': 0.3, 'fisher': 0.7},
            num_samples=20
        )

        for name, score_tensor in scores.items():
            # Check that normalization worked
            if score_tensor.numel() > 1:
                # Should have some variation (not all same value)
                std = score_tensor.std().item()
                self.assertGreater(std, 0, f"Scores should have variation for {name}")


class TestImportanceTypeSelection(unittest.TestCase):
    """Test importance type selection in main function."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = SimpleModel()
        self.wrapped_model = create_model_wrapper(self.model)
        self.dataloader = SimpleDataLoader()

    def test_all_importance_types(self):
        """Test all importance types through main function."""
        importance_types = ['fisher', 'taylor', 'magnitude', 'grad_norm']

        for imp_type in importance_types:
            with self.subTest(importance_type=imp_type):
                scores = compute_gradient_importance(
                    self.wrapped_model,
                    self.dataloader,
                    importance_type=imp_type,
                    num_samples=10
                )

                self.assertGreater(len(scores), 0,
                                 f"Should compute scores for {imp_type}")

    def test_invalid_importance_type(self):
        """Test that invalid importance type raises error."""
        with self.assertRaises(ValueError):
            compute_gradient_importance(
                self.wrapped_model,
                self.dataloader,
                importance_type='invalid_type'
            )


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTest(unittest.makeSuite(TestFisherImportance))
    suite.addTest(unittest.makeSuite(TestTaylorImportance))
    suite.addTest(unittest.makeSuite(TestMagnitudeImportance))
    suite.addTest(unittest.makeSuite(TestGradientNormImportance))
    suite.addTest(unittest.makeSuite(TestHybridImportance))
    suite.addTest(unittest.makeSuite(TestImportanceTypeSelection))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())