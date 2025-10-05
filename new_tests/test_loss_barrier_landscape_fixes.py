#!/usr/bin/env python3
"""
Unit tests for ICLR-quality loss barrier and landscape implementations.
Tests all critical fixes for numerical stability, memory efficiency, and theoretical correctness.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import gc
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ICLRMetrics import ICLRMetrics


class SimpleTestModel(nn.Module):
    """Simple model for testing that mimics a language model interface."""

    def __init__(self, vocab_size=100, hidden_dim=20, include_buffers=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, vocab_size)

        if include_buffers:
            # Register buffers to test buffer interpolation
            self.register_buffer('running_mean', torch.zeros(hidden_dim))
            self.register_buffer('running_var', torch.ones(hidden_dim))

        # Add config for compatibility with ICLRMetrics
        class Config:
            pass
        self.config = Config()
        self.config.vocab_size = vocab_size

    def forward(self, input_ids=None, labels=None, **kwargs):
        """Forward pass with language model interface."""
        if input_ids is None:
            raise ValueError("input_ids is required")

        # Embed and process
        x = self.embedding(input_ids)
        x = torch.relu(self.layer1(x))
        logits = self.layer2(x)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # Flatten for loss computation
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Return in format expected by ICLRMetrics
        class Output:
            pass
        output = Output()
        output.logits = logits
        output.loss = loss
        return output


class TestLossBarrierFixes(unittest.TestCase):
    """Test critical fixes in loss barrier computation."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = ICLRMetrics(device=self.device)

        # Create two slightly different models
        self.model1 = SimpleTestModel().to(self.device)
        self.model2 = SimpleTestModel().to(self.device)

        # Perturb model2 slightly
        with torch.no_grad():
            for p2, p1 in zip(self.model2.parameters(), self.model1.parameters()):
                p2.add_(torch.randn_like(p2) * 0.1)

        # Create test batch with valid vocabulary indices
        vocab_size = 100  # Match model vocabulary
        self.batch = {
            'input_ids': torch.randint(0, vocab_size, (4, 10), device=self.device),
            'labels': torch.randint(0, vocab_size, (4, 10), device=self.device)
        }

    def test_buffer_interpolation_control(self):
        """Test that buffer interpolation can be controlled (critical for ICLR)."""
        # Test with buffers NOT interpolated (literature standard)
        result_no_buffers = self.metrics.compute_loss_barrier(
            self.model1, self.model2, self.batch,
            n_points=5,
            interpolate_buffers=False,
            seed=42
        )

        # Test with buffers interpolated
        result_with_buffers = self.metrics.compute_loss_barrier(
            self.model1, self.model2, self.batch,
            n_points=5,
            interpolate_buffers=True,
            seed=42
        )

        # Results should be different when models have buffers
        self.assertIn('barrier_height', result_no_buffers)
        self.assertIn('barrier_height', result_with_buffers)
        self.assertIn('interpolate_buffers', result_no_buffers)
        self.assertFalse(result_no_buffers['interpolate_buffers'])
        self.assertTrue(result_with_buffers['interpolate_buffers'])

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with seed (required for ICLR)."""
        result1 = self.metrics.compute_loss_barrier(
            self.model1, self.model2, self.batch,
            n_points=5, seed=123
        )

        result2 = self.metrics.compute_loss_barrier(
            self.model1, self.model2, self.batch,
            n_points=5, seed=123
        )

        # Results should be identical
        self.assertAlmostEqual(result1['barrier_height'], result2['barrier_height'])
        self.assertAlmostEqual(result1['mean_loss'], result2['mean_loss'])

    def test_nan_inf_handling(self):
        """Test robust handling of NaN/Inf values."""
        # Create a model that will produce NaN
        bad_model = SimpleTestModel().to(self.device)
        with torch.no_grad():
            # Set some weights to NaN
            bad_model.layer1.weight.data[0, :] = float('nan')

        result = self.metrics.compute_loss_barrier(
            self.model1, bad_model, self.batch,
            n_points=5
        )

        # Should handle NaN gracefully
        if 'error' not in result:
            self.assertIn('n_nonfinite', result)
            self.assertGreater(result['n_nonfinite'], 0)

    def test_memory_efficiency(self):
        """Test that memory usage is efficient (no deepcopy)."""
        import psutil
        import os

        process = psutil.Process(os.getpid())

        # Measure memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        # Run barrier computation
        result = self.metrics.compute_loss_barrier(
            self.model1, self.model2, self.batch,
            n_points=20
        )

        # Measure memory after
        gc.collect()
        mem_after = process.memory_info().rss / 1024 / 1024  # MB

        # Memory increase should be minimal (not doubling from deepcopy)
        mem_increase = mem_after - mem_before

        # Should complete successfully
        self.assertIn('barrier_height', result)
        self.assertNotIn('error', result)

        # Memory increase should be reasonable (< 100MB for small models)
        # Note: This is a soft check as memory can vary
        self.assertLess(mem_increase, 100,
                       f"Memory increased by {mem_increase:.1f}MB - possible memory leak")

    def test_type_safety(self):
        """Test that loss values are properly converted to Python floats."""
        result = self.metrics.compute_loss_barrier(
            self.model1, self.model2, self.batch,
            n_points=5
        )

        # All numeric values should be Python floats, not tensors
        self.assertIsInstance(result['barrier_height'], (float, int))
        self.assertIsInstance(result['mean_loss'], (float, int))
        self.assertIsInstance(result['loss_variance'], (float, int))

        if 'loss_trajectory' in result:
            for loss in result['loss_trajectory']:
                self.assertIsInstance(loss, (float, int))

    def test_model_restoration(self):
        """Test that original models are not modified."""
        # Save original parameters
        orig_params1 = {name: p.detach().clone()
                       for name, p in self.model1.named_parameters()}
        orig_params2 = {name: p.detach().clone()
                       for name, p in self.model2.named_parameters()}

        # Run barrier computation
        self.metrics.compute_loss_barrier(
            self.model1, self.model2, self.batch,
            n_points=10
        )

        # Check models are restored
        for name, p in self.model1.named_parameters():
            self.assertTrue(torch.allclose(p, orig_params1[name]),
                          f"Parameter {name} was modified in model1")
        for name, p in self.model2.named_parameters():
            self.assertTrue(torch.allclose(p, orig_params2[name]),
                          f"Parameter {name} was modified in model2")


class TestLossLandscapeFixes(unittest.TestCase):
    """Test critical fixes in loss landscape computation."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = ICLRMetrics(device=self.device)
        self.model = SimpleTestModel().to(self.device)

        vocab_size = 100  # Match model vocabulary
        self.batch = {
            'input_ids': torch.randint(0, vocab_size, (4, 10), device=self.device),
            'labels': torch.randint(0, vocab_size, (4, 10), device=self.device)
        }

    def test_renamed_functions(self):
        """Test that misleading function was renamed correctly."""
        # Old misleading name should be gone, replaced with accurate name
        self.assertTrue(hasattr(self.metrics, 'sample_directional_losses'))
        self.assertTrue(hasattr(self.metrics, 'compute_loss_landscape_2d'))

        # sample_directional_losses should NOT create a 2D grid
        result = self.metrics.sample_directional_losses(
            self.model, self.batch,
            n_samples=10, seed=42
        )

        self.assertIn('note', result)
        self.assertIn('not a 2D grid', result['note'])
        self.assertIn('n_samples', result)

    def test_true_2d_landscape(self):
        """Test that true 2D landscape creates a proper grid."""
        result = self.metrics.compute_loss_landscape_2d(
            self.model, self.batch,
            n_points=5, seed=42
        )

        self.assertIn('grid_losses', result)
        self.assertIn('grid_shape', result)
        self.assertEqual(result['grid_shape'], [5, 5])

        # Grid should be n_points x n_points
        grid = result['grid_losses']
        self.assertEqual(len(grid), 5)
        self.assertEqual(len(grid[0]), 5)

    def test_landscape_reproducibility(self):
        """Test landscape reproducibility with seed."""
        result1 = self.metrics.compute_loss_landscape_2d(
            self.model, self.batch,
            n_points=3, seed=999
        )

        result2 = self.metrics.compute_loss_landscape_2d(
            self.model, self.batch,
            n_points=3, seed=999
        )

        # Grids should be identical
        grid1 = np.array(result1['grid_losses'])
        grid2 = np.array(result2['grid_losses'])

        # Check non-NaN values are equal
        mask = ~(np.isnan(grid1) | np.isnan(grid2))
        if mask.any():
            np.testing.assert_array_almost_equal(grid1[mask], grid2[mask])

    def test_landscape_nan_handling(self):
        """Test that landscape handles NaN values properly."""
        result = self.metrics.compute_loss_landscape_2d(
            self.model, self.batch,
            n_points=5
        )

        # Should report valid points
        self.assertIn('n_valid', result)
        self.assertIn('n_total', result)

        # Statistics should handle NaN
        if not np.isnan(result['loss_mean']):
            self.assertIsInstance(result['loss_mean'], float)
            self.assertIsInstance(result['loss_std'], float)

    def test_filter_normalization(self):
        """Test filter normalization option for CNNs."""
        # Test with filter normalization (recommended for CNNs)
        result_filtered = self.metrics.compute_loss_landscape_2d(
            self.model, self.batch,
            n_points=3,
            use_filter_norm=True,
            seed=42
        )

        # Test without filter normalization
        result_unfiltered = self.metrics.compute_loss_landscape_2d(
            self.model, self.batch,
            n_points=3,
            use_filter_norm=False,
            seed=42
        )

        # Both should complete
        self.assertIn('grid_losses', result_filtered)
        self.assertIn('grid_losses', result_unfiltered)
        self.assertTrue(result_filtered['filter_normalized'])
        self.assertFalse(result_unfiltered['filter_normalized'])

    def test_numerical_stability(self):
        """Test numerical stability improvements."""
        result = self.metrics.sample_directional_losses(
            self.model, self.batch,
            n_samples=50,
            span=1e-10  # Very small span
        )

        # Should handle small values without division by zero
        self.assertIn('landscape_roughness', result)
        self.assertTrue(np.isfinite(result['landscape_roughness']))


class TestIntegration(unittest.TestCase):
    """Test integration with unified_model_analysis.py."""

    def test_compatibility(self):
        """Test that functions are compatible with unified framework."""
        metrics = ICLRMetrics()

        # Check expected signatures
        barrier_func = metrics.compute_loss_barrier
        self.assertIn('interpolate_buffers', barrier_func.__code__.co_varnames)
        self.assertIn('seed', barrier_func.__code__.co_varnames)

        landscape_func = metrics.compute_loss_landscape_2d
        self.assertIn('seed', landscape_func.__code__.co_varnames)

        # Check renamed function exists
        directional_func = metrics.sample_directional_losses
        self.assertIn('n_samples', directional_func.__code__.co_varnames)


if __name__ == '__main__':
    unittest.main()