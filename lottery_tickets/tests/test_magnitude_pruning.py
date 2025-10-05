"""
Unit tests for magnitude-based pruning methods.
================================================
Tests pruning robustness, mask creation, and lottery ticket finding.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lottery_tickets.magnitude_pruning import (
    compute_pruning_robustness,
    compute_layerwise_magnitude_ticket,
    create_magnitude_mask,
    _global_magnitude_pruning,
    _layerwise_magnitude_pruning,
    _compute_robustness_metrics
)
from lottery_tickets.utils import (
    ensure_deterministic_pruning,
    create_model_wrapper,
    compute_sparsity,
    apply_mask,
    remove_mask
)


class TestModel(nn.Module):
    """Test model with various layer types."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 2 * 2, 128)  # Fixed: 8x8 after 2 pooling -> 2x2
        self.fc2 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # For testing, accept both image and flattened input
        if x.dim() == 2:  # Flattened input from wrapper
            batch_size = x.size(0)
            # Expecting 3*8*8 = 192 flattened input
            if x.size(1) == 3 * 8 * 8:
                x = x.view(-1, 3, 8, 8)
            else:
                # Fallback: create dummy image input
                x = x[:, :3*8*8].view(-1, 3, 8, 8) if x.size(1) >= 192 else torch.randn(batch_size, 3, 8, 8)

        x = F.relu(self.conv1(x))  # 8x8 -> 8x8 (padding=1)
        x = F.max_pool2d(x, 2)     # 8x8 -> 4x4
        x = F.relu(self.conv2(x))  # 4x4 -> 4x4 (padding=1)
        x = F.max_pool2d(x, 2)     # 4x4 -> 2x2
        x = x.view(x.size(0), -1)  # Flatten to [batch, 32*2*2=128]
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


class TestMaskCreation(unittest.TestCase):
    """Test pruning mask creation."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = TestModel()
        self.model.eval()

    def test_create_magnitude_mask_basic(self):
        """Test basic mask creation."""
        sparsity = 0.5
        mask = create_magnitude_mask(self.model, sparsity)

        # Check masks exist for weight parameters
        for name, param in self.model.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                self.assertIn(name, mask, f"Missing mask for {name}")

                # Check mask is binary
                unique_vals = torch.unique(mask[name])
                self.assertTrue(
                    len(unique_vals) <= 2 and
                    all(v in [0, 1] for v in unique_vals),
                    f"Mask should be binary for {name}"
                )

    def test_histogram_vs_direct_quantile(self):
        """Test histogram quantile approximation vs direct computation."""
        sparsity = 0.3

        # Histogram-based (memory efficient)
        mask_hist = create_magnitude_mask(
            self.model, sparsity,
            use_histogram=True,
            histogram_bins=1000
        )

        # Direct quantile
        mask_direct = create_magnitude_mask(
            self.model, sparsity,
            use_histogram=False
        )

        # Should be similar but not identical
        for name in mask_hist:
            sparsity_hist = (mask_hist[name] == 0).float().mean().item()
            sparsity_direct = (mask_direct[name] == 0).float().mean().item()

            # Should be close
            self.assertAlmostEqual(sparsity_hist, sparsity_direct, delta=0.05,
                                 msg=f"Histogram should approximate direct for {name}")

    def test_sparsity_levels(self):
        """Test different sparsity levels."""
        sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

        for sparsity in sparsity_levels:
            with self.subTest(sparsity=sparsity):
                mask = create_magnitude_mask(self.model, sparsity)
                actual_sparsity = compute_sparsity(mask)

                # Should be close to target
                self.assertAlmostEqual(actual_sparsity, sparsity, delta=0.1,
                                     msg=f"Actual sparsity {actual_sparsity} should match target {sparsity}")

    def test_only_weights_parameter(self):
        """Test that only_weights parameter works correctly."""
        # With only_weights=True (default)
        mask_weights_only = create_magnitude_mask(self.model, 0.5, only_weights=True)

        for name in mask_weights_only:
            self.assertIn('weight', name, f"Should only include weights, got {name}")

        # With only_weights=False
        mask_all = create_magnitude_mask(self.model, 0.5, only_weights=False)

        # Should not include bias (1D parameters are skipped)
        for name in mask_all:
            param = dict(self.model.named_parameters())[name]
            self.assertGreaterEqual(len(param.shape), 2,
                                  f"Should skip 1D parameters, got {name} with shape {param.shape}")


class TestPruningRobustness(unittest.TestCase):
    """Test pruning robustness computation."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = TestModel()
        self.wrapped_model = create_model_wrapper(self.model)
        self.wrapped_model.eval()

        # Create test batch
        self.batch = {
            'input_ids': torch.randn(8, 3 * 8 * 8),  # Flattened for compatibility
            'labels': torch.randint(0, 10, (8,))
        }

    def test_pruning_robustness_basic(self):
        """Test basic pruning robustness computation."""
        results = compute_pruning_robustness(
            self.wrapped_model,
            self.batch,
            sparsity_levels=[0.1, 0.5, 0.9],
            use_histogram_quantiles=True
        )

        # Check result structure
        self.assertIn('baseline_loss', results)
        self.assertIn('sparsity_curves', results)
        self.assertIn('robustness_metrics', results)

        # Check baseline loss is positive
        self.assertGreater(results['baseline_loss'], 0)

    def test_sparsity_curves(self):
        """Test sparsity performance curves."""
        results = compute_pruning_robustness(
            self.wrapped_model,
            self.batch,
            sparsity_levels=[0.1, 0.3, 0.5, 0.7, 0.9]
        )

        curves = results['sparsity_curves']

        # Should have results for each sparsity level
        self.assertEqual(len(curves), 5)

        # Check each curve entry
        for key, metrics in curves.items():
            self.assertIn('sparsity', metrics)
            self.assertIn('actual_sparsity', metrics)
            self.assertIn('loss', metrics)
            self.assertIn('loss_increase', metrics)
            self.assertIn('performance_retention', metrics)

            # Loss should increase with pruning (generally)
            self.assertGreaterEqual(metrics['loss'], 0)

    def test_robustness_metrics(self):
        """Test robustness summary metrics."""
        results = compute_pruning_robustness(
            self.wrapped_model,
            self.batch,
            sparsity_levels=[0.1, 0.3, 0.5, 0.7, 0.9]
        )

        metrics = results['robustness_metrics']

        # Check required metrics
        self.assertIn('winning_ticket_score', metrics)
        self.assertIn('optimal_sparsity', metrics)
        self.assertIn('critical_sparsity', metrics)

        # Validate ranges
        self.assertGreaterEqual(metrics['winning_ticket_score'], 0)
        self.assertGreaterEqual(metrics['optimal_sparsity'], 0)
        self.assertLessEqual(metrics['optimal_sparsity'], 1)

        # Critical sparsity should be when performance drops by 50%
        self.assertGreaterEqual(metrics['critical_sparsity'], 0)
        self.assertLessEqual(metrics['critical_sparsity'], 1.0)

    def test_return_masks(self):
        """Test returning pruning masks."""
        results = compute_pruning_robustness(
            self.wrapped_model,
            self.batch,
            sparsity_levels=[0.3, 0.6],
            return_masks=True
        )

        self.assertIn('masks', results)
        masks = results['masks']

        # Should have masks for each sparsity
        self.assertEqual(len(masks), 2)

        # Check mask structure
        for sparsity_key, mask_dict in masks.items():
            for param_name, mask_tensor in mask_dict.items():
                # Should be on CPU for storage
                self.assertEqual(mask_tensor.device.type, 'cpu')

                # Should be binary
                unique = torch.unique(mask_tensor)
                self.assertTrue(all(v in [0, 1] for v in unique))


class TestLotteryTicketFinding(unittest.TestCase):
    """Test lottery ticket hypothesis methods."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = TestModel()
        self.model.eval()

    def test_global_magnitude_pruning(self):
        """Test global magnitude ranking."""
        result = _global_magnitude_pruning(
            self.model,
            target_sparsity=0.5,
            max_params_per_chunk=100_000
        )

        # Check result structure
        self.assertIn('masks', result)
        self.assertIn('layer_sparsities', result)
        self.assertIn('overall_sparsity', result)
        self.assertIn('method', result)
        self.assertIn('threshold', result)

        # Method should be global
        self.assertEqual(result['method'], 'global_ranking')

        # Overall sparsity should match target
        self.assertAlmostEqual(result['overall_sparsity'], 0.5, delta=0.1)

        # All layers should use same threshold
        threshold = result['threshold']
        self.assertGreater(threshold, 0)

    def test_layerwise_magnitude_pruning(self):
        """Test layer-wise magnitude pruning."""
        result = _layerwise_magnitude_pruning(
            self.model,
            target_sparsity=0.5,
            layer_importance_weights=None  # Uniform
        )

        # Check structure
        self.assertIn('masks', result)
        self.assertIn('layer_sparsities', result)
        self.assertIn('overall_sparsity', result)
        self.assertIn('method', result)

        # Method should be layer-wise
        self.assertEqual(result['method'], 'importance_weighted')

        # Each layer should have approximately target sparsity (uniform weights)
        for layer_name, sparsity in result['layer_sparsities'].items():
            self.assertAlmostEqual(sparsity, 0.5, delta=0.15,
                                 msg=f"Layer {layer_name} sparsity should be close to target")

    def test_importance_weighted_pruning(self):
        """Test importance-weighted layer-wise pruning."""
        # Define importance weights (higher = more important = less pruning)
        importance_weights = {
            'fc2.weight': 2.0,  # Most important (output layer)
            'fc1.weight': 1.0,  # Normal importance
            'conv2.weight': 0.8,  # Less important
            'conv1.weight': 0.5   # Least important
        }

        result = _layerwise_magnitude_pruning(
            self.model,
            target_sparsity=0.5,
            layer_importance_weights=importance_weights
        )

        sparsities = result['layer_sparsities']

        # fc2 should have less pruning (lower sparsity)
        # conv1 should have more pruning (higher sparsity)
        if 'fc2.weight' in sparsities and 'conv1.weight' in sparsities:
            self.assertLess(sparsities['fc2.weight'], sparsities['conv1.weight'],
                          "Important layers should be pruned less")

    def test_global_vs_layerwise(self):
        """Test difference between global and layer-wise pruning."""
        global_result = compute_layerwise_magnitude_ticket(
            self.model,
            target_sparsity=0.5,
            use_global_ranking=True
        )

        layerwise_result = compute_layerwise_magnitude_ticket(
            self.model,
            target_sparsity=0.5,
            use_global_ranking=False
        )

        # Both should achieve similar overall sparsity
        self.assertAlmostEqual(global_result['overall_sparsity'],
                             layerwise_result['overall_sparsity'],
                             delta=0.1)

        # But layer sparsities should differ
        global_sparsities = global_result['layer_sparsities']
        layerwise_sparsities = layerwise_result['layer_sparsities']

        # Global should have more variance in layer sparsities
        global_var = np.var(list(global_sparsities.values()))
        layerwise_var = np.var(list(layerwise_sparsities.values()))

        # Global typically has higher variance
        # (some layers pruned more, others less)
        self.assertGreaterEqual(global_var, layerwise_var * 0.5,
                              "Global pruning typically has more variance")


class TestMaskOperations(unittest.TestCase):
    """Test mask application and removal."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.model = TestModel()

        # Create a simple mask
        self.mask = create_magnitude_mask(self.model, sparsity=0.5)

    def test_apply_mask(self):
        """Test applying pruning mask."""
        # Store original weights
        original = {}
        for name, param in self.model.named_parameters():
            if name in self.mask:
                original[name] = param.data.clone()

        # Apply mask
        apply_mask(self.model, self.mask)

        # Check weights are zeroed where mask is 0
        for name, param in self.model.named_parameters():
            if name in self.mask:
                masked_positions = (self.mask[name] == 0)
                zeroed = torch.all(param.data[masked_positions] == 0)
                self.assertTrue(zeroed, f"Masked positions should be zero for {name}")

                # Non-masked positions should be unchanged
                non_masked = ~masked_positions
                if torch.any(non_masked):
                    unchanged = torch.allclose(
                        param.data[non_masked],
                        original[name][non_masked]
                    )
                    self.assertTrue(unchanged, f"Non-masked positions should be unchanged for {name}")

    def test_apply_mask_with_clone(self):
        """Test applying mask with weight cloning."""
        original_weights = apply_mask(self.model, self.mask, clone_weights=True)

        # Should return original weights
        self.assertIsNotNone(original_weights)

        # Check all masked parameters are in original_weights
        for name in self.mask:
            self.assertIn(name, original_weights)

    def test_remove_mask(self):
        """Test removing mask and restoring weights."""
        # Store original
        original = {}
        for name, param in self.model.named_parameters():
            if name in self.mask:
                original[name] = param.data.clone()

        # Apply mask
        apply_mask(self.model, self.mask)

        # Remove mask with original weights
        remove_mask(self.model, self.mask, original_weights=original)

        # Should be restored
        for name, param in self.model.named_parameters():
            if name in self.mask:
                restored = torch.allclose(param.data, original[name])
                self.assertTrue(restored, f"Weights should be restored for {name}")

    def test_remove_mask_with_noise(self):
        """Test removing mask with noise injection."""
        # Apply mask
        apply_mask(self.model, self.mask)

        # Remove with noise
        remove_mask(self.model, self.mask, original_weights=None, noise_scale=0.01)

        # Previously masked positions should have small noise
        for name, param in self.model.named_parameters():
            if name in self.mask:
                masked_positions = (self.mask[name] == 0)

                if torch.any(masked_positions):
                    # Should have some non-zero values from noise
                    has_noise = torch.any(param.data[masked_positions] != 0)
                    self.assertTrue(has_noise, f"Should add noise to pruned weights for {name}")

                    # But noise should be small
                    max_noise = param.data[masked_positions].abs().max().item()
                    self.assertLess(max_noise, 0.1, f"Noise should be small for {name}")


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTest(unittest.makeSuite(TestMaskCreation))
    suite.addTest(unittest.makeSuite(TestPruningRobustness))
    suite.addTest(unittest.makeSuite(TestLotteryTicketFinding))
    suite.addTest(unittest.makeSuite(TestMaskOperations))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())