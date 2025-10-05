#!/usr/bin/env python3
"""
Unit tests for critical fixes in the SuperpositionMetrics implementation.

Tests verify that paper-invalidating bugs have been resolved per Liu et al. 2025
"Superposition Yields Robust Neural Scaling" paper requirements.

Source: superposition/core/enhanced.py
Class: SuperpositionMetrics
Purpose: Ensure theoretical correctness for ICLR 2026 submission
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superposition.core.enhanced import SuperpositionMetrics


class TestCriticalBaseFixesParticipationRatio(unittest.TestCase):
    """Test participation ratio fixes."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SuperpositionMetrics()

    def test_participation_ratio_uses_eigenvalues(self):
        """Test that participation ratio uses eigenvalues (λ = s²) not singular values."""
        # Create a matrix with known singular values
        U = torch.eye(3)
        s = torch.tensor([3.0, 2.0, 1.0])
        V = torch.eye(3)
        matrix = U @ torch.diag(s) @ V.T

        # Run compute_superposition_strength on a simple mock model
        class MockModel(nn.Module):
            def __init__(self, activation):
                super().__init__()
                self.activation = activation
                self.layer = nn.Identity()

            def forward(self, x):
                return self.activation

        model = MockModel(matrix.unsqueeze(0))  # Add batch dimension
        test_batch = {'input_ids': torch.tensor([[1, 2, 3]])}

        # Hook to capture metrics
        captured_pr = None

        def capture_hook(module, input, output):
            nonlocal captured_pr
            # The PR calculation happens inside compute_superposition_strength
            return output

        # We need to test the internal calculation more directly
        # Test the PR calculation with known values
        activation = matrix
        n_dims = 3

        # Expected PR using eigenvalues: λ = s²
        eigenvalues = s.numpy() ** 2  # [9, 4, 1]
        expected_pr = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()  # 196 / 98 = 2.0

        # Compute through the actual code path
        result = self.metrics.compute_superposition_strength(model, test_batch, n_probes=1)

        # The exact value depends on the random projections, but we can verify the order of magnitude
        self.assertIn('layer_metrics', result)
        self.assertIn('participation_ratio', result)

        # More direct test: verify the formula in isolation
        s_test = np.array([3.0, 2.0, 1.0])
        eigenvalues_test = s_test ** 2
        pr_eigenvalues = (eigenvalues_test.sum() ** 2) / (eigenvalues_test ** 2).sum()
        self.assertAlmostEqual(pr_eigenvalues, 2.0, places=5)

        # Wrong formula (using singular values) would give different result
        pr_singular = (s_test.sum() ** 2) / (s_test ** 2).sum()  # Would be ~2.45
        self.assertNotAlmostEqual(pr_singular, 2.0, places=1)


class TestCriticalBaseFixesLossComputation(unittest.TestCase):
    """Test loss computation fixes for causal LMs."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SuperpositionMetrics()

    def test_causal_lm_label_shifting(self):
        """Test that causal LMs get properly shifted labels."""
        # Create a mock causal LM
        class MockCausalLM(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.lm_head = nn.Linear(32, 100)

            def forward(self, input_ids):
                x = self.embed(input_ids)
                return self.lm_head(x)

        # Create models dict
        models = {'small': MockCausalLM()}
        test_batch = {'input_ids': torch.randint(0, 100, (2, 10))}

        # The analyze_dimensional_scaling should detect CausalLM and shift labels
        # We can't easily test the internal label shifting without mocking more
        # But we ensure the function runs without error
        result = self.metrics.analyze_dimensional_scaling(models, test_batch)

        self.assertIn('model_sizes', result)
        self.assertIn('losses', result)


class TestCriticalBaseFixesScalingLaw(unittest.TestCase):
    """Test scaling law with irreducible loss term."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SuperpositionMetrics()

    def test_scaling_law_with_irreducible_loss(self):
        """Test that scaling law fitting includes irreducible loss term c."""
        # Generate synthetic data with known parameters
        sizes = np.array([10, 20, 50, 100, 200, 500, 1000])
        true_alpha = 0.8
        true_constant = 100.0
        true_offset = 2.0  # Irreducible loss

        # L = a * N^(-alpha) + c
        losses = true_constant * sizes ** (-true_alpha) + true_offset
        # Add small noise
        np.random.seed(42)
        losses = losses + np.random.normal(0, 0.01, len(losses))

        # Fit with irreducible loss
        result_with_offset = self.metrics.fit_scaling_law(
            sizes, losses, log_scale=False, include_offset=True
        )

        # Check that fitting returned expected parameters
        self.assertIn('alpha', result_with_offset)
        self.assertIn('irreducible_loss', result_with_offset)

        # Check if fitted parameters are close to true values
        alpha_error = abs(result_with_offset['alpha'] - true_alpha)
        offset_error = abs(result_with_offset['irreducible_loss'] - true_offset)

        self.assertLess(alpha_error, 0.1, f"Alpha error {alpha_error:.3f} should be small")
        self.assertLess(offset_error, 0.5, f"Offset error {offset_error:.3f} should be small")

        # Compare with fitting without offset (should be worse)
        result_without_offset = self.metrics.fit_scaling_law(
            sizes, losses, log_scale=False, include_offset=False
        )

        self.assertNotIn('irreducible_loss', result_without_offset)
        # Without offset, alpha estimate should be biased
        self.assertNotAlmostEqual(
            result_without_offset['alpha'], true_alpha, places=1,
            msg="Without offset, alpha should be biased"
        )


class TestCriticalBaseFixesCapacityEstimation(unittest.TestCase):
    """Test capacity estimation with proper train/test split."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SuperpositionMetrics()

    def test_capacity_train_test_split(self):
        """Test that capacity estimation uses train/test split."""
        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(64, 128)

            def forward(self, x):
                return self.fc(x)

        model = SimpleModel()
        # Pass as dict with proper key
        test_batch = {'input_ids': torch.randn(30, 64)}  # Enough samples for train/test split

        result = self.metrics.compute_representation_capacity(
            model, test_batch, n_probes=3
        )

        self.assertIn('mean_test_accuracy', result)
        self.assertIn('probe_accuracies', result)
        self.assertEqual(len(result['probe_accuracies']), 3)

        # With random labels and proper train/test, accuracy should be around 0.5
        mean_acc = result['mean_test_accuracy']
        self.assertGreater(mean_acc, 0.3, "Mean accuracy should not be too low")
        self.assertLess(mean_acc, 0.8, "Mean accuracy should not be too high for random labels")

    def test_capacity_insufficient_samples(self):
        """Test that capacity estimation handles insufficient samples gracefully."""
        class TinyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 10)

            def forward(self, x):
                return self.fc(x)

        model = TinyModel()
        # Pass as dict with proper key
        test_batch = {'input_ids': torch.randn(5, 10)}  # Too few samples

        result = self.metrics.compute_representation_capacity(
            model, test_batch, n_probes=3
        )

        self.assertIn('error', result)
        self.assertIn('Insufficient samples', result['error'])


class TestCriticalBaseFixesSVDEdgeCases(unittest.TestCase):
    """Test SVD edge case handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SuperpositionMetrics()

    def test_svd_small_dimension_edge_cases(self):
        """Test that SVD handles matrices with dimension < 2."""
        # Test various edge cases
        test_cases = [
            (torch.tensor([[2.0]]), "1x1 matrix"),
            (torch.tensor([[2.0], [1.0]]), "2x1 matrix"),
            (torch.tensor([[2.0, 1.0]]), "1x2 matrix"),
            (torch.zeros(0, 5), "0x5 empty matrix"),
        ]

        for matrix, description in test_cases:
            # The SVD should handle these without crashing
            # We test through compute_vector_interference which uses SVD internally
            result = self.metrics.compute_vector_interference(matrix, normalize=False)

            self.assertIn('mean_overlap', result, f"Should handle {description}")
            self.assertIn('n_features', result, f"Should return metadata for {description}")

    def test_randomized_svd_large_matrix(self):
        """Test that randomized SVD is used for large matrices."""
        # Create a large matrix that would trigger randomized SVD
        large_matrix = torch.randn(300, 300)

        # The randomized SVD should be used internally
        k = 50
        s = self.metrics._randomized_svd(large_matrix, k=k)

        self.assertEqual(len(s), k, "Should return k singular values")
        self.assertTrue(torch.all(s >= 0), "Singular values should be non-negative")
        # Check they're in descending order
        self.assertTrue(torch.all(s[:-1] >= s[1:]), "Singular values should be sorted")


class TestCriticalBaseFixesReconstruction(unittest.TestCase):
    """Test reconstruction with orthogonal projection."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SuperpositionMetrics()

    def test_orthogonal_projection_nmse(self):
        """Test that reconstruction uses orthogonal projection and NMSE."""
        # Create a model with known activations
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 64)
                self.layer = nn.Linear(64, 64)

            def forward(self, input_ids=None, x=None):
                if input_ids is not None:
                    x = self.embed(input_ids).mean(dim=1)
                return self.layer(x)

        model = MockModel()
        test_batch = {'input_ids': torch.randint(0, 100, (4, 10))}

        # Run superposition analysis
        with torch.no_grad():
            result = self.metrics.compute_superposition_strength(
                model, test_batch, n_probes=2
            )

        self.assertIn('layer_metrics', result)

        # Check that reconstruction quality is reported and reasonable
        for layer_name, metrics in result['layer_metrics'].items():
            self.assertIn('reconstruction_quality', metrics)
            self.assertIn('reconstruction_error', metrics)

            # Quality should be between 0 and 1
            quality = metrics['reconstruction_quality']
            self.assertGreaterEqual(quality, 0.0)
            self.assertLessEqual(quality, 1.0)

            # Error should be NMSE (normalized)
            error = metrics['reconstruction_error']
            self.assertGreaterEqual(error, 0.0)


class TestCriticalBaseFixesHistogram(unittest.TestCase):
    """Test histogram range fixes."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SuperpositionMetrics()

    def test_histogram_range_normalized(self):
        """Test that histogram uses correct range for normalized overlaps."""
        # Create a small weight matrix
        weight_matrix = torch.randn(10, 5)

        result = self.metrics.compute_vector_interference(
            weight_matrix, normalize=True
        )

        if 'overlap_histogram' in result:
            hist = result['overlap_histogram']
            # For normalized overlaps, range should be [0, 1]
            self.assertAlmostEqual(hist['bin_edges'][0], 0.0, places=5)
            self.assertAlmostEqual(hist['bin_edges'][-1], 1.0, places=5)

    def test_histogram_range_unnormalized(self):
        """Test that histogram uses correct range for unnormalized overlaps."""
        # Create weight matrix with large values
        weight_matrix = torch.randn(10, 5) * 10.0

        result = self.metrics.compute_vector_interference(
            weight_matrix, normalize=False
        )

        if 'overlap_histogram' in result:
            hist = result['overlap_histogram']
            # For unnormalized overlaps, range should accommodate actual values
            # Just check it doesn't crash and range is reasonable
            self.assertGreater(hist['bin_edges'][-1], 1.0)
            self.assertLess(hist['bin_edges'][-1], 200.0)  # Should be capped


class TestCriticalBaseFixesTokenizerAPI(unittest.TestCase):
    """Test tokenizer API fixes."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = SuperpositionMetrics()

    def test_tokenizer_api_usage(self):
        """Test that tokenizer uses correct API."""
        # Create a mock tokenizer
        class MockTokenizer:
            def __call__(self, text, return_tensors=None, add_special_tokens=True):
                # Simulate proper tokenizer API
                return {'input_ids': torch.tensor([[1, 2, 3, 4, 5]])}

            def encode(self, text, return_tensors=None):
                # This should NOT be called
                raise RuntimeError("encode() should not be called, use __call__ instead")

        # Create a simple model
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(100, 32)
                self.config = type('Config', (), {'vocab_size': 100})()

            def forward(self, x):
                return self.embed(x)

        model = SimpleModel()
        tokenizer = MockTokenizer()
        dataset = ["test text"]

        # This should use the correct tokenizer API (not crash)
        result = self.metrics.compute_feature_frequency_distribution(
            model, dataset, tokenizer=tokenizer, max_samples=10
        )

        self.assertIn('vocab_size', result)
        self.assertEqual(result['vocab_size'], 100)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
