#!/usr/bin/env python3
"""
Unit tests for critical fixes in Enhanced SuperpositionMetrics implementation.

Tests verify that paper-invalidating bugs have been resolved in the enhanced version
which includes GPU optimizations, numerical stability improvements, and memory efficiency.

File: superposition/core/enhanced.py
Class: SuperpositionMetrics (enhanced version)
Purpose: Ensure theoretical correctness and performance optimizations for ICLR 2026
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superposition.core.enhanced import SuperpositionMetrics, SuperpositionConfig


class TestCriticalFixes(unittest.TestCase):
    """Test suite for critical fixes in SuperpositionMetrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SuperpositionConfig()
        self.metrics = SuperpositionMetrics(config=self.config)

    def test_participation_ratio_uses_eigenvalues(self):
        """Test that participation ratio uses eigenvalues (s²) not singular values."""
        # Create a simple matrix with known singular values
        U = torch.eye(3)
        s = torch.tensor([3.0, 2.0, 1.0])
        V = torch.eye(3)
        matrix = U @ torch.diag(s) @ V.T

        # Compute effective rank and participation ratio
        activation = matrix
        effective_rank, participation_ratio = self.metrics._compute_effective_rank(activation)

        # Expected PR using eigenvalues: λ = s²
        eigenvalues = s.numpy() ** 2  # [9, 4, 1]
        expected_pr_value = (eigenvalues.sum() ** 2) / (eigenvalues ** 2).sum()  # 196 / 98 = 2.0

        # Check that participation ratio matches eigenvalue-based calculation
        self.assertAlmostEqual(
            participation_ratio, expected_pr_value, places=2,
            msg="Participation ratio should use eigenvalues (s²) not singular values"
        )

    def test_svd_small_dimension_edge_case(self):
        """Test that SVD handles matrices with dimension < 2."""
        # Test 1x1 matrix
        tiny_matrix = torch.tensor([[2.0]])
        try:
            result = self.metrics._truncated_svd(tiny_matrix, k=1)
            self.assertEqual(len(result), 1, "1x1 matrix should return 1 singular value")
            self.assertFalse(torch.isnan(result).any(), "1x1 matrix should not return NaN")
        except Exception as e:
            self.fail(f"1x1 matrix raised exception: {e}")

        # Test 2x1 matrix
        small_matrix = torch.tensor([[2.0], [1.0]])
        try:
            result = self.metrics._truncated_svd(small_matrix, k=1)
            self.assertGreaterEqual(len(result), 1, "2x1 matrix should return at least 1 singular value")
            self.assertFalse(torch.isnan(result).any(), "2x1 matrix should not return NaN")
        except Exception as e:
            self.fail(f"2x1 matrix raised exception: {e}")

    def test_scaling_law_with_irreducible_loss(self):
        """Test that scaling law fitting includes irreducible loss term."""
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
        self.assertIn('alpha', result_with_offset, "Should return alpha parameter")
        self.assertIn('irreducible_loss', result_with_offset, "Should return irreducible loss")

        # Check if fitted parameters are close to true values
        alpha_error = abs(result_with_offset['alpha'] - true_alpha)
        offset_error = abs(result_with_offset['irreducible_loss'] - true_offset)

        self.assertLess(alpha_error, 0.1, f"Alpha error {alpha_error:.3f} should be small")
        self.assertLess(offset_error, 0.5, f"Offset error {offset_error:.3f} should be small")

    def test_normalized_reconstruction_error(self):
        """Test that reconstruction error is properly normalized."""
        # Create activations with different scales
        torch.manual_seed(42)
        small_scale = torch.randn(100, 50) * 0.1
        large_scale = torch.randn(100, 50) * 10.0

        # Compute reconstruction quality
        quality_small = self.metrics._estimate_reconstruction_quality(small_scale, n_probes=3)
        quality_large = self.metrics._estimate_reconstruction_quality(large_scale, n_probes=3)

        # Qualities should be similar since error is normalized
        quality_diff = abs(quality_small - quality_large)
        self.assertLess(
            quality_diff, 0.3,
            f"Normalized qualities should be similar (diff={quality_diff:.4f})"
        )

    def test_capacity_train_test_split(self):
        """Test that capacity estimation uses train/test split."""
        # Create synthetic hidden states
        torch.manual_seed(42)
        hidden_states = torch.randn(100, 20)

        # Run probe experiments
        probe_accuracies = self.metrics._run_probe_experiments(hidden_states, n_probes=5)

        self.assertEqual(len(probe_accuracies), 5, "Should run requested number of probes")

        # With random labels and proper train/test split, accuracy should be around 0.5
        mean_acc = np.mean(probe_accuracies)
        self.assertGreater(mean_acc, 0.3, "Mean accuracy should not be too low")
        self.assertLess(mean_acc, 0.8, "Mean accuracy should not be too high for random labels")

    def test_single_feature_metadata(self):
        """Test that single feature returns correct dimensions."""
        # Create single feature with known dimension
        n_dims = 128
        weight_matrix = torch.randn(1, n_dims)

        result = self.metrics.compute_vector_interference(weight_matrix)

        self.assertEqual(result['n_features'], 1, "Should have 1 feature")
        self.assertEqual(
            result['n_dimensions'], n_dims,
            f"Should have {n_dims} dimensions, got {result['n_dimensions']}"
        )

    def test_layer_selection_excludes_subcomponents(self):
        """Test that layer selection excludes sub-components like gate_proj, q_proj."""
        # Create mock model with sub-components
        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList()

                # Add a transformer layer with sub-components
                layer = nn.Module()
                layer.self_attn = nn.Module()
                layer.self_attn.q_proj = nn.Linear(256, 256)
                layer.self_attn.k_proj = nn.Linear(256, 256)
                layer.mlp = nn.Module()
                layer.mlp.gate_proj = nn.Linear(256, 512)
                layer.mlp.down_proj = nn.Linear(512, 256)

                self.model.layers.append(layer)
                self.model.embed_tokens = nn.Embedding(1000, 256)

            def forward(self, x):
                return torch.randn(1, 10, 1000)

        model = MockModel()
        test_batch = {'input_ids': torch.randint(0, 1000, (1, 10))}

        # Capture activations without specifying layers (should auto-detect)
        activations = self.metrics._capture_activations(
            model, test_batch['input_ids'], probe_layers=None
        )

        # Check that sub-components were excluded
        has_subcomponents = any(
            'gate_proj' in name or 'q_proj' in name or 'down_proj' in name
            for name in activations.keys()
        )

        self.assertFalse(
            has_subcomponents,
            "Sub-components like gate_proj, q_proj should be excluded from layer selection"
        )

    def test_svd_ill_conditioned_matrices(self):
        """Test that SVD handles ill-conditioned matrices with regularization."""
        # Create ill-conditioned matrix (rank-deficient)
        n = 100
        U = torch.randn(n, n)
        U, _ = torch.linalg.qr(U)

        # Create singular values with huge condition number
        s = torch.ones(n)
        s[-50:] = 1e-10  # Make last 50 singular values tiny
        S = torch.diag(s)
        V = torch.randn(n, n)
        V, _ = torch.linalg.qr(V)

        ill_conditioned = U @ S @ V.T

        try:
            # Try truncated SVD (should handle ill-conditioning)
            singular_values = self.metrics._truncated_svd(ill_conditioned, k=50)

            self.assertFalse(
                torch.isnan(singular_values).any(),
                "SVD should not return NaN for ill-conditioned matrix"
            )
            self.assertEqual(len(singular_values), 50, "Should return requested number of singular values")

        except Exception as e:
            self.fail(f"SVD raised exception for ill-conditioned matrix: {e}")

    def test_dtype_consistency_half_precision(self):
        """Test that dtype mismatches are handled for half precision models."""
        # Create half precision activation
        activation_half = torch.randn(32, 256).half()

        try:
            # Test reconstruction quality (should handle dtype conversion)
            error = self.metrics._estimate_reconstruction_quality(activation_half, n_probes=2)

            self.assertFalse(
                error != error,  # NaN check
                "Reconstruction should not return NaN for half precision"
            )
            self.assertGreaterEqual(error, 0, "Reconstruction quality should be non-negative")
            self.assertLessEqual(error, 1, "Reconstruction quality should be <= 1")

        except Exception as e:
            self.fail(f"dtype handling raised exception: {e}")


class TestMemoryLeakFix(unittest.TestCase):
    """Test memory leak fix in activation capture."""

    def test_memory_efficient_activation_capture(self):
        """Test that activation capture doesn't store full tensors when not needed."""
        # This test is a placeholder for the memory leak fix
        # The actual fix would compute metrics inside hooks and store only summaries
        # TODO: Implement memory-efficient activation capture
        pass


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)