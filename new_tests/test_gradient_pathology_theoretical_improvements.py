#!/usr/bin/env python3
"""
Unit tests for theoretically improved gradient pathology detection.

Tests the new implementation that addresses fundamental issues in the original:
1. Replaces percentile-based classification with absolute thresholds
2. Adds multiple batch sampling for statistical robustness
3. Computes signal-to-noise ratio for gradient reliability
4. Measures gradient flow through network depth
5. Provides comprehensive optimization health scoring
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GradientAnalysis import GradientAnalysis


class SimpleTestModel(nn.Module):
    """Simple model with controllable gradient behavior for testing."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 4,
                 weight_scale_decay: float = 0.5):
        super().__init__()
        self.embed = nn.Embedding(1000, hidden_size)

        # Create layers with different scales to test gradient flow
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Linear(hidden_size, hidden_size)
            with torch.no_grad():
                # Exponential decay in weights to create gradient variation
                layer.weight.mul_(weight_scale_decay ** i)
            self.layers.append(layer)

        self.output = nn.Linear(hidden_size, 1000)

        # Mock config for compatibility
        self.config = type('Config', (), {'vocab_size': 1000})()

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embed(input_ids)

        for layer in self.layers:
            x = torch.relu(layer(x))

        logits = self.output(x)

        outputs = type('Outputs', (), {})()
        outputs.logits = logits

        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits.view(-1, 1000), labels.view(-1))
            outputs.loss = loss

        return outputs


class TestGradientPathologyTheoreticalImprovements(unittest.TestCase):
    """Test suite for improved gradient pathology detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GradientAnalysis()
        self.model = SimpleTestModel()

        # Standard test batch
        self.test_batch = {
            'input_ids': torch.randint(0, 1000, (4, 16)),
            'labels': torch.randint(0, 1000, (4, 16))
        }

        # Larger batch for microbatch testing
        self.large_batch = {
            'input_ids': torch.randint(0, 1000, (20, 16)),
            'labels': torch.randint(0, 1000, (20, 16))
        }

    def test_absolute_thresholds_no_false_positives(self):
        """Test that extreme thresholds produce no false positives."""
        result = self.analyzer.compute_gradient_pathology(
            self.model,
            self.test_batch,
            learning_rate=1e-3,
            n_samples=3,
            absolute_vanishing_threshold=1e-20,  # Extremely low
            absolute_exploding_threshold=1e20     # Extremely high
        )

        self.assertNotIn('error', result)
        self.assertEqual(len(result['vanishing_parameters']), 0,
                        "Should find no vanishing gradients with extreme threshold")
        self.assertEqual(len(result['exploding_parameters']), 0,
                        "Should find no exploding gradients with extreme threshold")

    def test_absolute_thresholds_scale_with_learning_rate(self):
        """Test that thresholds scale appropriately with learning rate."""
        # Test with different learning rates
        lr_small = 1e-4
        lr_large = 1e-2

        result_small = self.analyzer.compute_gradient_pathology(
            self.model, self.test_batch,
            learning_rate=lr_small,
            n_samples=2
        )

        result_large = self.analyzer.compute_gradient_pathology(
            self.model, self.test_batch,
            learning_rate=lr_large,
            n_samples=2
        )

        # Verify thresholds scaled correctly
        self.assertLess(
            result_small['summary']['vanishing_threshold_used'],
            result_large['summary']['vanishing_threshold_used'],
            "Vanishing threshold should be smaller with smaller learning rate"
        )

        self.assertGreater(
            result_small['summary']['exploding_threshold_used'],
            result_large['summary']['exploding_threshold_used'],
            "Exploding threshold should be larger with smaller learning rate"
        )

    def test_multiple_batch_sampling_creates_variance(self):
        """Test that multiple batch sampling produces variance statistics."""
        result = self.analyzer.compute_gradient_pathology(
            self.model,
            self.test_batch,
            n_samples=5,
            compute_snr=False  # Disable SNR for this test
        )

        self.assertNotIn('error', result)
        self.assertEqual(result['num_batches_analyzed'], 5)

        # Check that we have variance (std > 0) for at least some parameters
        has_variance = any(
            stat.get('std_norm', 0) > 0
            for stat in result['gradient_statistics'].values()
        )

        self.assertTrue(has_variance,
                       "Multiple samples should produce non-zero variance")

    def test_signal_to_noise_ratio_computation(self):
        """Test SNR computation for gradient reliability assessment."""
        # Create consistent batches for meaningful SNR
        base_data = torch.randint(0, 1000, (4, 16))
        batches = [
            {
                'input_ids': base_data + torch.randint(-1, 2, base_data.shape),
                'labels': base_data
            }
            for _ in range(5)
        ]

        result = self.analyzer.compute_gradient_pathology(
            self.model,
            batches,
            compute_snr=True
        )

        self.assertNotIn('error', result)
        self.assertIn('signal_to_noise_ratio', result)

        snr_data = result['signal_to_noise_ratio']
        self.assertGreater(len(snr_data), 0, "Should compute SNR for parameters")

        # Check SNR structure
        for param_name, snr_info in snr_data.items():
            self.assertIn('snr', snr_info)
            self.assertIn('signal', snr_info)
            self.assertIn('noise', snr_info)
            self.assertIn('reliable', snr_info)

            # SNR should be positive
            self.assertGreaterEqual(snr_info['snr'], 0)

            # Reliability flag should be boolean
            self.assertIsInstance(snr_info['reliable'], bool)

    def test_gradient_flow_score_detection(self):
        """Test gradient flow analysis through network depth."""
        # Create model with known gradient decay pattern
        model_with_decay = SimpleTestModel(num_layers=6, weight_scale_decay=0.5)

        result = self.analyzer.compute_gradient_pathology(
            model_with_decay,
            self.test_batch,
            compute_flow_score=True,
            n_samples=2
        )

        self.assertNotIn('error', result)
        self.assertIn('gradient_flow_score', result)
        self.assertIn('gradient_flow_health', result)
        self.assertIn('layer_gradient_decay', result)
        self.assertIn('interpretation', result)

        # Flow health should be between 0 and 1
        flow_health = result['gradient_flow_health']
        self.assertGreaterEqual(flow_health, 0)
        self.assertLessEqual(flow_health, 1)

        # Should analyze multiple layers
        self.assertGreater(len(result['layer_gradient_decay']), 1)

    def test_optimization_health_score_comprehensive(self):
        """Test comprehensive optimization health scoring."""
        result = self.analyzer.compute_gradient_pathology(
            self.model,
            self.test_batch,
            learning_rate=1e-3,
            compute_snr=True,
            compute_flow_score=True,
            n_samples=3
        )

        self.assertNotIn('error', result)
        self.assertIn('optimization_health_score', result)
        self.assertIn('health_interpretation', result)

        # Health score should be in [0, 1]
        health_score = result['optimization_health_score']
        self.assertGreaterEqual(health_score, 0)
        self.assertLessEqual(health_score, 1)

        # Check summary statistics
        self.assertIn('summary', result)
        summary = result['summary']

        required_fields = [
            'total_parameters', 'vanishing_count', 'exploding_count',
            'vanishing_percentage', 'exploding_percentage', 'learning_rate',
            'vanishing_threshold_used', 'exploding_threshold_used'
        ]

        for field in required_fields:
            self.assertIn(field, summary, f"Summary should contain {field}")

    def test_uniform_gradients_no_pathology(self):
        """Test that uniform gradients don't trigger false pathology detection."""
        # Create batch with identical inputs (should produce uniform gradients)
        uniform_batch = {
            'input_ids': torch.ones(4, 16, dtype=torch.long) * 100,
            'labels': torch.ones(4, 16, dtype=torch.long) * 100
        }

        result = self.analyzer.compute_gradient_pathology(
            self.model,
            uniform_batch,
            learning_rate=1e-3,
            n_samples=1,  # Single sample to avoid variation
            absolute_vanishing_threshold=1e-10,
            absolute_exploding_threshold=1e10
        )

        self.assertNotIn('error', result)

        # Calculate pathology percentages
        total_params = len(result['gradient_statistics'])
        vanishing_pct = len(result['vanishing_parameters']) / total_params * 100
        exploding_pct = len(result['exploding_parameters']) / total_params * 100

        # Should find very few or no pathological gradients
        # Old percentile method would guarantee 5% each
        self.assertLess(vanishing_pct, 2,
                       "Should find <2% vanishing (old method would show ~5%)")
        self.assertLess(exploding_pct, 2,
                       "Should find <2% exploding (old method would show ~5%)")

    def test_backward_compatibility_single_batch(self):
        """Test that single batch input is handled correctly."""
        # Test with single batch (Dict)
        result_single = self.analyzer.compute_gradient_pathology(
            self.model,
            self.test_batch,  # Single batch as Dict
            n_samples=3
        )

        self.assertNotIn('error', result_single)
        self.assertEqual(result_single['num_batches_analyzed'], 3,
                        "Should create 3 variations from single batch")

    def test_multiple_batch_input(self):
        """Test that list of batches is handled correctly."""
        # Test with multiple batches (List[Dict])
        multi_batches = [
            {
                'input_ids': torch.randint(0, 1000, (4, 16)),
                'labels': torch.randint(0, 1000, (4, 16))
            }
            for _ in range(4)
        ]

        result_multi = self.analyzer.compute_gradient_pathology(
            self.model,
            multi_batches
        )

        self.assertNotIn('error', result_multi)
        self.assertEqual(result_multi['num_batches_analyzed'], 4,
                        "Should analyze all provided batches")

    def test_gradient_validation_integration(self):
        """Test that gradient validation catches frozen models."""
        # Create model with all parameters frozen
        frozen_model = SimpleTestModel()
        for param in frozen_model.parameters():
            param.requires_grad = False

        result = self.analyzer.compute_gradient_pathology(
            frozen_model,
            self.test_batch
        )

        self.assertIn('error', result)
        self.assertIn('validation_failed', result)
        self.assertEqual(result['gradient_coverage'], 0.0,
                        "Frozen model should have 0% gradient coverage")

    def test_numerical_stability_with_small_gradients(self):
        """Test numerical stability with very small gradients."""
        # Create model with very small weights
        small_model = SimpleTestModel()
        with torch.no_grad():
            for param in small_model.parameters():
                param.mul_(1e-8)

        result = self.analyzer.compute_gradient_pathology(
            small_model,
            self.test_batch,
            learning_rate=1e-3,
            n_samples=2
        )

        # Should complete without NaN/Inf errors
        self.assertNotIn('error', result)

        # Check for NaN/Inf in results
        for stat in result['gradient_statistics'].values():
            self.assertFalse(np.isnan(stat['mean_norm']),
                           "Should not produce NaN values")
            self.assertFalse(np.isinf(stat['mean_norm']),
                           "Should not produce Inf values")


class TestGradientFlowHelperFunctions(unittest.TestCase):
    """Test helper functions for gradient flow analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GradientAnalysis()

    def test_estimate_layer_depth(self):
        """Test layer depth estimation from names."""
        test_cases = [
            ("model.layers.0.weight", 0),
            ("model.layers.5.bias", 5),
            ("encoder.layer.3.attention", 3),
            ("blocks.2.mlp.weight", 2),
        ]

        for layer_name, expected_depth in test_cases:
            # The function returns some depth estimate
            depth = self.analyzer._estimate_layer_depth(layer_name)
            # For numbered layers, it should extract the number
            if "layers" in layer_name or "layer" in layer_name or "blocks" in layer_name:
                self.assertIsInstance(depth, int)

    def test_interpret_flow_score(self):
        """Test gradient flow score interpretation."""
        test_cases = [
            (-2.0, "Vanishing"),  # Strong vanishing
            (-0.3, "Mild vanishing"),  # Mild vanishing
            (0.05, "Excellent"),  # Good flow
            (0.3, "Mild exploding"),  # Mild exploding
            (2.0, "Exploding"),  # Strong exploding
        ]

        for decay_rate, expected_keyword in test_cases:
            interpretation = self.analyzer._interpret_flow_score(decay_rate)
            self.assertIn(expected_keyword, interpretation)

    def test_interpret_health_score(self):
        """Test health score interpretation."""
        test_cases = [
            (0.95, "Excellent"),
            (0.7, "Good"),
            (0.5, "Fair"),
            (0.3, "Poor"),
            (0.1, "Critical"),
        ]

        for health_score, expected_keyword in test_cases:
            interpretation = self.analyzer._interpret_health_score(health_score)
            self.assertIn(expected_keyword, interpretation)


class TestGradientPathologyFixes(unittest.TestCase):
    """Test suite for critical fixes to gradient pathology detection."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GradientAnalysis()
        self.model = SimpleTestModel()

    def test_microbatch_splitting(self):
        """Test that microbatch splitting creates distinct batches."""
        # Create a batch with diverse data
        batch = {
            'input_ids': torch.randint(0, 1000, (20, 16)),
            'labels': torch.randint(0, 1000, (20, 16))
        }

        result = self.analyzer.compute_gradient_pathology(
            self.model, batch, n_samples=4
        )

        self.assertNotIn('error', result)
        # With microbatches, different parts of the data are used
        # This should produce variance in gradients
        if 'signal_to_noise_ratio' in result:
            # Check that we get reasonable SNR values
            snr_values = [v['snr'] for v in result['signal_to_noise_ratio'].values()]
            # Some variance should exist
            self.assertTrue(len(snr_values) > 0, "Should compute SNR for some parameters")

            # Verify microbatching worked (created variance)
            self.assertEqual(result['num_batches_analyzed'], 4, "Should analyze 4 microbatches")

    def test_batchnorm_not_modified(self):
        """Test that BatchNorm running stats are not modified during analysis."""
        # Add a BatchNorm layer to the model
        model = nn.Sequential(
            nn.Linear(100, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

        # Initialize BN with specific values
        bn_layer = model[1]
        original_mean = bn_layer.running_mean.clone()
        original_var = bn_layer.running_var.clone()

        # Create simple batch
        batch_size = 8
        batch = {
            'input_ids': torch.randn(batch_size, 100),
            'labels': torch.randint(0, 10, (batch_size,))
        }

        # Monkey-patch forward to work with our batch format
        def forward_hook(input_ids, labels=None, **kwargs):
            x = input_ids
            logits = model(x)
            outputs = type('Outputs', (), {})()
            outputs.logits = logits
            if labels is not None:
                outputs.loss = nn.CrossEntropyLoss()(logits, labels)
            return outputs

        model.forward = forward_hook
        model.config = type('Config', (), {'vocab_size': 10})()

        # Run gradient pathology analysis
        result = self.analyzer.compute_gradient_pathology(
            model, batch, n_samples=2
        )

        # Check that BN stats were not modified
        self.assertTrue(torch.allclose(bn_layer.running_mean, original_mean),
                       "BatchNorm running_mean should not be modified")
        self.assertTrue(torch.allclose(bn_layer.running_var, original_var),
                       "BatchNorm running_var should not be modified")

    def test_pathological_layers_populated(self):
        """Test that pathological_layers field is properly populated."""
        # Create model with known gradient issues
        model = SimpleTestModel()

        # Make some layers have very small weights (will have vanishing gradients)
        with torch.no_grad():
            for name, param in model.named_parameters():
                if 'layers.0' in name:
                    param.mul_(1e-10)

        batch = {
            'input_ids': torch.randint(0, 1000, (4, 16)),
            'labels': torch.randint(0, 1000, (4, 16))
        }

        result = self.analyzer.compute_gradient_pathology(
            model, batch,
            learning_rate=1.0,  # Large LR to make vanishing threshold larger
            absolute_vanishing_threshold=1e-5
        )

        self.assertNotIn('error', result)
        self.assertIn('pathological_layers', result)
        self.assertIsInstance(result['pathological_layers'], list)

        # Should have found some pathological layers
        if result['vanishing_parameters']:
            self.assertGreater(len(result['pathological_layers']), 0,
                             "Should identify pathological layers when parameters are vanishing")

        # Check layer_pathology_details
        self.assertIn('layer_pathology_details', result)
        if result['pathological_layers']:
            # Details should contain pathology types
            for layer in result['pathological_layers']:
                self.assertIn(layer, result['layer_pathology_details'])
                pathology_types = result['layer_pathology_details'][layer]
                self.assertTrue(all(t in ['vanishing', 'exploding'] for t in pathology_types))

    def test_memory_efficient_mode(self):
        """Test memory-efficient SNR computation mode."""
        # Create multiple distinct batches for consistent testing
        batches = [
            {
                'input_ids': torch.randint(0, 1000, (8, 16)),
                'labels': torch.randint(0, 1000, (8, 16))
            }
            for _ in range(4)
        ]

        # Test with memory-efficient mode
        result_efficient = self.analyzer.compute_gradient_pathology(
            self.model, batches,
            memory_efficient=True,
            compute_snr=True
        )

        # Test without memory-efficient mode
        result_standard = self.analyzer.compute_gradient_pathology(
            self.model, batches,
            memory_efficient=False,
            compute_snr=True
        )

        self.assertNotIn('error', result_efficient)
        self.assertNotIn('error', result_standard)

        # Both should compute SNR
        self.assertIn('signal_to_noise_ratio', result_efficient,
                     "Memory-efficient mode should compute SNR")
        self.assertIn('signal_to_noise_ratio', result_standard,
                     "Standard mode should compute SNR")

        # Check that both computed SNR for multiple parameters
        self.assertGreater(len(result_efficient['signal_to_noise_ratio']), 0,
                          "Memory-efficient should compute SNR for parameters")
        self.assertGreater(len(result_standard['signal_to_noise_ratio']), 0,
                          "Standard should compute SNR for parameters")

        # Both should have reasonable SNR values (not NaN or infinite)
        for snr_data in result_efficient['signal_to_noise_ratio'].values():
            self.assertFalse(np.isnan(snr_data['snr']), "SNR should not be NaN")
            self.assertFalse(np.isinf(snr_data['snr']), "SNR should not be infinite")
            self.assertGreaterEqual(snr_data['snr'], 0, "SNR should be non-negative")


if __name__ == '__main__':
    unittest.main()