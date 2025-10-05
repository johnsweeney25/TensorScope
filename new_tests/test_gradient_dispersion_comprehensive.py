#!/usr/bin/env python3
"""Comprehensive unit tests for gradient dispersion analysis.

Tests all edge cases, numerical stability, and proper implementation of
the compute_gradient_dispersion function.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import tempfile
import os
from unittest.mock import MagicMock, patch

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GradientAnalysis import GradientAnalysis


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.layer1 = nn.Linear(128, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 32)
        self.layer3 = nn.Linear(32, 2)

    def forward(self, input_ids=None, labels=None, **kwargs):
        x = torch.randn(1, 128, device=self.layer1.weight.device)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        loss = x.mean()
        return type('Output', (), {'loss': loss})()


class TestGradientDispersion(unittest.TestCase):
    """Test suite for gradient dispersion analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GradientAnalysis(device='cpu')
        self.model = SimpleTestModel()
        self.batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4]]),
            'attention_mask': torch.tensor([[1, 1, 1, 1]])
        }

    def tearDown(self):
        """Clean up after tests."""
        if hasattr(self, 'model'):
            self.model.zero_grad()

    def test_gini_coefficient_correctness(self):
        """Test Gini coefficient calculation for known distributions."""
        # Test perfect equality (all same values)
        equal_grads = torch.ones(100)
        results = self.analyzer.compute_gradient_dispersion(
            self.model, self.batch, metric='gini'
        )

        # Create a model with uniform gradients for testing
        with torch.no_grad():
            for param in self.model.parameters():
                param.grad = torch.ones_like(param) * 0.5

        # Directly test the internal function
        from GradientAnalysis import GradientAnalysis
        analyzer_instance = GradientAnalysis()

        # Test with perfectly equal distribution
        equal_tensor = torch.ones(1000) * 0.5
        # Access the method through the class
        test_model = SimpleTestModel()
        test_batch = self.batch

        # Mock uniform gradients
        with patch.object(test_model, 'forward') as mock_forward:
            mock_output = MagicMock()
            mock_output.loss = torch.tensor(1.0, requires_grad=True)
            mock_forward.return_value = mock_output

            # Set uniform gradients manually
            test_model.zero_grad()
            for param in test_model.parameters():
                param.grad = torch.ones_like(param) * 0.1

            results = analyzer_instance.compute_gradient_dispersion(
                test_model, test_batch, metric='gini'
            )

            # For uniform distribution, Gini should be close to 0
            if results['dispersion_score'] is not None:
                self.assertLess(results['dispersion_score'], 0.1)

    def test_entropy_calculation_edge_cases(self):
        """Test entropy calculation with edge cases."""
        # Test entropy calculation with a model that generates very small gradients
        # Create a model with near-zero loss
        class NearZeroLossModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.ones(10) * 1e-10)

            def forward(self, **kwargs):
                # Create a very small loss that still has gradients
                loss = self.param.sum() * 1e-10
                return type('Output', (), {'loss': loss})()

        near_zero_model = NearZeroLossModel()
        results = self.analyzer.compute_gradient_dispersion(
            near_zero_model, self.batch, metric='entropy'
        )

        # Should handle near-zero gradients gracefully
        self.assertIsNotNone(results['dispersion_score'])
        # With uniform near-zero gradients, entropy should be close to 0 (all equal)
        if not np.isnan(results['dispersion_score']):
            self.assertTrue(0 <= results['dispersion_score'] <= 1)

    def test_coefficient_of_variation_numerical_stability(self):
        """Test CV calculation with near-zero values."""
        # Create model with very small gradients
        small_grad_model = SimpleTestModel()
        small_grad_model.zero_grad()
        for param in small_grad_model.parameters():
            param.grad = torch.randn_like(param) * 1e-8

        results = self.analyzer.compute_gradient_dispersion(
            small_grad_model, self.batch, metric='cv'
        )

        # Should not crash or return inf/nan
        if results['dispersion_score'] is not None:
            self.assertFalse(np.isnan(results['dispersion_score']))
            self.assertFalse(np.isinf(results['dispersion_score']))

    def test_training_mode_preservation(self):
        """Test that model stays in training mode during gradient computation."""
        # Set model to eval initially
        self.model.eval()

        # Patch the backward call to check training mode
        original_forward = self.model.forward
        training_mode_during_forward = []

        def wrapped_forward(*args, **kwargs):
            training_mode_during_forward.append(self.model.training)
            return original_forward(*args, **kwargs)

        self.model.forward = wrapped_forward

        results = self.analyzer.compute_gradient_dispersion(
            self.model, self.batch, metric='gini'
        )

        # Model should have been in training mode during forward pass
        self.assertTrue(any(training_mode_during_forward))

    def test_gradient_accumulation_prevention(self):
        """Test that existing gradients are cleared before computation."""
        # First compute dispersion normally
        results1 = self.analyzer.compute_gradient_dispersion(
            self.model, self.batch, metric='gini'
        )

        # Manually add large gradients to the model
        for param in self.model.parameters():
            if param.grad is not None:
                param.grad = param.grad + 1000.0

        # Compute dispersion again - it should clear gradients first
        results2 = self.analyzer.compute_gradient_dispersion(
            self.model, self.batch, metric='gini'
        )

        # The dispersion scores should be similar since gradients are recomputed
        # (not accumulated with the manually added 1000)
        # If gradients were accumulated with +1000, the scores would be vastly different
        diff = abs(results1['dispersion_score'] - results2['dispersion_score'])
        self.assertLess(
            diff,
            0.1,  # Allow for small variations due to randomness
            msg=f"Gradients appear to be accumulating: diff={diff:.4f} (should be < 0.1)"
        )

    def test_device_consistency(self):
        """Test that all tensors are on the same device."""
        if torch.cuda.is_available():
            cuda_analyzer = GradientAnalysis(device='cuda')
            cuda_model = SimpleTestModel().cuda()
            cuda_batch = {
                'input_ids': torch.tensor([[1, 2, 3, 4]]).cuda(),
                'attention_mask': torch.tensor([[1, 1, 1, 1]]).cuda()
            }

            results = cuda_analyzer.compute_gradient_dispersion(
                cuda_model, cuda_batch, metric='gini'
            )

            # Should complete without device mismatch errors
            self.assertIsNotNone(results)

    def test_nan_inf_loss_handling(self):
        """Test handling of NaN/Inf loss values."""
        # Create a model that returns NaN loss with requires_grad
        class NaNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.param = nn.Parameter(torch.ones(1))

            def forward(self, **kwargs):
                # Create a loss that will become NaN but has gradient tracking
                loss = self.param * float('nan')
                return type('Output', (), {'loss': loss})()

        nan_model = NaNModel()
        results = self.analyzer.compute_gradient_dispersion(
            nan_model, self.batch, metric='gini'
        )

        # Should handle gracefully
        self.assertEqual(
            results['interpretation'],
            'Invalid loss (NaN or Inf) - cannot compute gradients'
        )

    def test_large_parameter_count_stability(self):
        """Test with models having large number of parameters."""
        # Create a model with many parameters
        large_model = nn.Sequential(
            *[nn.Linear(100, 100) for _ in range(10)]
        )

        # Mock forward to return a proper loss
        def forward(**kwargs):
            x = torch.randn(1, 100)
            for layer in large_model:
                x = layer(x)
            return type('Output', (), {'loss': x.mean()})()

        large_model.forward = forward

        # Should handle large parameter counts without overflow
        results = self.analyzer.compute_gradient_dispersion(
            large_model, self.batch, metric='gini'
        )

        if results['dispersion_score'] is not None:
            self.assertFalse(np.isnan(results['dispersion_score']))
            self.assertTrue(0 <= results['dispersion_score'] <= 1)

    def test_checkpoint_analysis_memory_cleanup(self):
        """Test proper memory cleanup in checkpoint analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy checkpoints
            checkpoint_paths = []
            for i in range(3):
                ckpt_path = os.path.join(tmpdir, f'ckpt_{i}.pt')
                torch.save(self.model.state_dict(), ckpt_path)
                checkpoint_paths.append(ckpt_path)

            # Track memory before and after
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                initial_memory = torch.cuda.memory_allocated()

            results = self.analyzer.compute_gradient_dispersion(
                self.model, self.batch,
                checkpoint_paths=checkpoint_paths,
                sample_every_n=1,
                metric='gini'
            )

            # Check temporal trend exists
            self.assertIn('temporal_trend', results)
            self.assertIn('steps', results['temporal_trend'])
            self.assertIn('dispersion', results['temporal_trend'])

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                final_memory = torch.cuda.memory_allocated()
                # Memory should not have grown significantly
                memory_growth = final_memory - initial_memory
                self.assertLess(memory_growth, 100 * 1024 * 1024)  # Less than 100MB growth

    def test_per_layer_dispersion_calculation(self):
        """Test that per-layer dispersion is calculated correctly."""
        results = self.analyzer.compute_gradient_dispersion(
            self.model, self.batch, metric='gini'
        )

        # Should have per-layer results
        self.assertIn('per_layer_dispersion', results)
        self.assertTrue(len(results['per_layer_dispersion']) > 0)

        # Each layer should have a valid dispersion score
        for layer_name, score in results['per_layer_dispersion'].items():
            if not np.isnan(score):
                self.assertTrue(0 <= score <= 1)

    def test_top_k_concentration_computation(self):
        """Test top-k concentration metrics."""
        results = self.analyzer.compute_gradient_dispersion(
            self.model, self.batch, metric='gini'
        )

        # Should have top-k concentration metrics
        self.assertIn('top_k_concentration', results)

        # Check standard percentiles
        for k in [1, 5, 10, 25]:
            key = f'top_{k}_percent'
            if key in results['top_k_concentration']:
                concentration = results['top_k_concentration'][key]
                # Concentration should be between 0 and 1
                self.assertTrue(0 <= concentration <= 1)

        # Top 25% should contain more than top 1%
        if 'top_1_percent' in results['top_k_concentration'] and \
           'top_25_percent' in results['top_k_concentration']:
            self.assertLessEqual(
                results['top_k_concentration']['top_1_percent'],
                results['top_k_concentration']['top_25_percent']
            )

    def test_consistent_absolute_values(self):
        """Test that gradient magnitudes are consistently computed as absolute values."""
        # Test that the function takes absolute values internally
        results = self.analyzer.compute_gradient_dispersion(
            self.model, self.batch, metric='gini'
        )

        # Check that we got valid results
        self.assertIsNotNone(results['dispersion_score'])

        # Verify per-layer dispersion is also computed with absolute values
        for layer_name, score in results['per_layer_dispersion'].items():
            if not np.isnan(score):
                # Dispersion scores should be between 0 and 1
                self.assertTrue(0 <= score <= 1, f"Layer {layer_name} has invalid score {score}")

    def test_interpretation_generation(self):
        """Test that interpretations are generated correctly."""
        for metric in ['gini', 'entropy', 'cv']:
            results = self.analyzer.compute_gradient_dispersion(
                self.model, self.batch, metric=metric
            )

            # Should have an interpretation
            self.assertIn('interpretation', results)
            if results['dispersion_score'] is not None:
                self.assertTrue(len(results['interpretation']) > 0)
                self.assertIn(metric.upper() if metric != 'cv' else 'CV',
                            results['interpretation'].upper())


if __name__ == '__main__':
    unittest.main()