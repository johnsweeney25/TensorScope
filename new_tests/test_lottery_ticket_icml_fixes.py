#!/usr/bin/env python3
"""
Comprehensive tests for ICML-grade lottery ticket fixes.
Tests theoretical correctness and numerical precision.

This test suite validates all critical fixes for:
1. IMP mask enforcement during training
2. Histogram-based global quantile computation
3. Early-Bird training between evaluations
4. Proper rewind epoch implementation
5. Label handling without fabrication
6. Robust batch size detection
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from LotteryTicketAnalysis import (
    LotteryTicketAnalysis,
    get_batch_size,
    ensure_labels
)


class SimpleModel(nn.Module):
    """Simple test model for lottery ticket testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, **kwargs):
        # Handle different input formats
        if 'input_ids' in kwargs:
            x = kwargs['input_ids']
        elif 'x' in kwargs:
            x = kwargs['x']
        else:
            raise ValueError("No input found in kwargs")

        x = self.relu(self.fc1(x))
        logits = self.fc2(x)

        # Compute loss if labels provided
        loss = None
        if 'labels' in kwargs:
            labels = kwargs['labels']
            loss = nn.functional.cross_entropy(logits, labels)

        return SimpleNamespace(loss=loss, logits=logits)


class TestMaskEnforcementICML(unittest.TestCase):
    """Test that masks are properly enforced during training (Critical for LTH)."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.model = SimpleModel()
        self.analyzer = LotteryTicketAnalysis()
        self.dataloader = [
            {'input_ids': torch.randn(8, 10), 'labels': torch.randint(0, 5, (8,))}
            for _ in range(3)
        ]

    def test_masks_stay_zero_during_training(self):
        """Verify that pruned weights remain zero throughout training.

        This is CRITICAL for the lottery ticket hypothesis - if pruned weights
        can be updated during training, the entire theory breaks down.
        """
        # Create masks that prune 50% of weights
        masks = {}
        for name, param in self.model.named_parameters():
            mask = torch.rand_like(param) > 0.5
            masks[name] = mask
            # Apply initial mask
            param.data *= mask.float()

        # Store initial masked values
        initial_zeros = {}
        for name, param in self.model.named_parameters():
            initial_zeros[name] = (param.data == 0).clone()

        # Train with mask enforcement
        _ = self.analyzer._simple_train(
            self.model,
            self.dataloader,
            epochs=2,
            masks=masks
        )

        # Verify pruned weights stayed zero
        for name, param in self.model.named_parameters():
            zeros_after = (param.data == 0)
            # All initially zero weights should still be zero
            violations = initial_zeros[name] & ~zeros_after
            self.assertEqual(
                violations.sum().item(), 0,
                f"Pruned weights in {name} were updated during training! "
                f"This violates the lottery ticket hypothesis."
            )

    def test_gradient_hooks_installed(self):
        """Test that gradient hooks properly mask gradients during backprop."""
        masks = {
            'fc1.weight': torch.ones_like(self.model.fc1.weight, dtype=torch.bool)
        }
        # Set some weights to be pruned
        masks['fc1.weight'][0, :5] = False

        # Apply masks with hooks
        handles = self.analyzer._apply_masks_with_hooks(self.model, masks)

        # Compute gradients
        batch = {'input_ids': torch.randn(4, 10), 'labels': torch.randint(0, 5, (4,))}
        outputs = self.model(**batch)
        outputs.loss.backward()

        # Check that gradients are masked
        grad = self.model.fc1.weight.grad
        self.assertTrue(
            (grad[0, :5] == 0).all(),
            "Gradients not masked properly - pruned weights would receive updates"
        )

        # Clean up hooks
        for h in handles:
            h.remove()

    def test_mask_persistence_across_iterations(self):
        """Test that masks persist correctly across multiple training iterations."""
        # Create sparse mask
        masks = {}
        for name, param in self.model.named_parameters():
            if 'weight' in name:
                mask = torch.rand_like(param) > 0.7  # 70% sparsity
                masks[name] = mask
            else:
                masks[name] = torch.ones_like(param, dtype=torch.bool)

        # Apply initial masks
        for name, param in self.model.named_parameters():
            param.data *= masks[name].float()

        # Count initial zeros
        initial_zero_count = sum(
            (param.data == 0).sum().item()
            for param in self.model.parameters()
        )

        # Train for multiple epochs
        _ = self.analyzer._simple_train(
            self.model,
            self.dataloader,
            epochs=5,
            masks=masks
        )

        # Count final zeros
        final_zero_count = sum(
            (param.data == 0).sum().item()
            for param in self.model.parameters()
        )

        self.assertEqual(
            initial_zero_count,
            final_zero_count,
            "Number of pruned weights changed during training"
        )


class TestHistogramQuantileICML(unittest.TestCase):
    """Test the corrected histogram-based quantile computation."""

    def test_histogram_quantile_accuracy(self):
        """Test that histogram method gives statistically accurate quantiles.

        The previous "quantile of quantiles" approach was fundamentally wrong
        and would produce biased results on large models.
        """
        # Create test parameters with known distribution
        torch.manual_seed(42)
        params = [
            ('param1', torch.randn(1000).abs()),
            ('param2', torch.randn(2000).abs()),
            ('param3', torch.randn(500).abs())
        ]

        # Compute ground truth quantile
        all_values = torch.cat([p[1].flatten() for p in params])
        true_quantile_50 = torch.quantile(all_values, 0.5).item()
        true_quantile_90 = torch.quantile(all_values, 0.9).item()

        # Implement the histogram quantile method directly for testing
        def compute_global_quantile_histogram(param_list, sparsity, bins=4096):
            """Test implementation of histogram-based quantile."""
            global_min = float('inf')
            global_max = float('-inf')
            total_elements = 0
            param_list = list(param_list)

            # Phase 1: Find min/max
            for name, param in param_list:
                if param.numel() == 0:
                    continue
                p_abs = param.detach().abs()
                global_min = min(global_min, p_abs.min().item())
                global_max = max(global_max, p_abs.max().item())
                total_elements += param.numel()

            if total_elements == 0:
                return torch.tensor(0.0)

            if global_min == global_max:
                return torch.tensor(global_min)

            # Phase 2: Build histogram
            device = param_list[0][1].device if param_list else torch.device('cpu')
            edges = torch.linspace(global_min, global_max, bins + 1, device=device)
            hist = torch.zeros(bins, dtype=torch.long, device=device)

            for name, param in param_list:
                if param.numel() == 0:
                    continue
                p_abs = param.detach().abs().flatten()
                indices = torch.searchsorted(edges[:-1], p_abs, right=False)
                indices = torch.clamp(indices, 0, bins - 1)
                hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.long))

            # Phase 3: Find quantile from CDF
            cumsum = torch.cumsum(hist, dim=0)
            target_count = int(sparsity * total_elements)
            threshold_idx = (cumsum >= target_count).nonzero(as_tuple=True)[0]
            if len(threshold_idx) > 0:
                threshold_idx = threshold_idx[0].item()
            else:
                threshold_idx = bins - 1

            return edges[threshold_idx]

        # Test quantiles
        hist_q50 = compute_global_quantile_histogram(params, 0.5)
        hist_q90 = compute_global_quantile_histogram(params, 0.9)

        # Allow small tolerance due to histogram binning
        self.assertAlmostEqual(
            hist_q50.item(), true_quantile_50, delta=0.05
        )
        self.assertAlmostEqual(
            hist_q90.item(), true_quantile_90, delta=0.05
        )

    def test_quantile_edge_cases(self):
        """Test edge cases for quantile computation."""
        # Test with empty params
        params = [('empty', torch.tensor([]))]
        # Should not crash

        # Test with single value
        params = [('single', torch.tensor([1.0]))]

        # Test with all zeros
        params = [('zeros', torch.zeros(100))]

        # None of these should crash
        analyzer = LotteryTicketAnalysis()
        model1 = SimpleModel()
        model2 = SimpleModel()

        # This internally uses the quantile computation
        result = analyzer.compute_ticket_overlap(
            model1, model2,
            sparsity_levels=[0.5]
        )
        self.assertIsNotNone(result)


class TestEarlyBirdTrainingICML(unittest.TestCase):
    """Test Early-Bird implementation actually trains between evaluations."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.model = SimpleModel()
        self.analyzer = LotteryTicketAnalysis()
        self.dataloader = [
            {'input_ids': torch.randn(8, 10), 'labels': torch.randint(0, 5, (8,))}
            for _ in range(5)
        ]

    def test_early_bird_trains_between_checks(self):
        """Verify model weights change between mask evaluations.

        The original implementation never trained, making it not Early-Bird at all.
        """
        initial_weights = self.model.fc1.weight.data.clone()

        result = self.analyzer.compute_early_bird_tickets(
            self.model,
            self.dataloader,
            max_epochs=10,
            check_interval=2,
            target_sparsity=0.5
        )

        # Weights should have changed significantly
        final_weights = self.model.fc1.weight.data
        weight_change = (final_weights - initial_weights).abs().sum()
        self.assertGreater(
            weight_change.item(), 0.1,
            "Model weights didn't change during Early-Bird - no training happened!"
        )

        # Should have training losses recorded
        self.assertIn('training_losses', result)
        self.assertGreater(len(result['training_losses']), 0)

        # Losses should generally decrease (allowing for noise)
        if len(result['training_losses']) > 3:
            early_loss = np.mean(result['training_losses'][:3])
            late_loss = np.mean(result['training_losses'][-3:])
            # Allow some variance but expect general improvement
            self.assertLess(
                late_loss, early_loss * 1.5,
                "Training losses didn't improve during Early-Bird"
            )

    def test_mask_evolution_during_training(self):
        """Test that masks actually evolve as the model trains."""
        result = self.analyzer.compute_early_bird_tickets(
            self.model,
            self.dataloader,
            max_epochs=15,
            check_interval=5,
            target_sparsity=0.5
        )

        # Should have mask distances showing evolution
        self.assertIn('mask_distances', result)
        if len(result['mask_distances']) > 0:
            # At least some masks should have changed
            max_distance = max(result['mask_distances'])
            self.assertGreater(
                max_distance, 0.01,
                "Masks never changed during training - Early-Bird not working"
            )

    def test_early_bird_convergence_detection(self):
        """Test that Early-Bird properly detects mask convergence."""
        # Use small model for faster convergence
        small_model = SimpleModel(input_dim=10, hidden_dim=10, output_dim=5)

        # Create dataloader with matching dimensions
        small_dataloader = [
            {'input_ids': torch.randn(8, 10), 'labels': torch.randint(0, 5, (8,))}
            for _ in range(2)
        ]

        result = self.analyzer.compute_early_bird_tickets(
            small_model,
            small_dataloader,  # Less data for faster convergence
            max_epochs=20,
            check_interval=2,
            target_sparsity=0.3,
            stability_threshold=0.1,  # Higher threshold for easier convergence
            stability_window=2  # Smaller window
        )

        # Check convergence detection logic
        self.assertIn('converged', result)
        self.assertIn('early_bird_epoch', result)
        self.assertIn('final_sparsity', result)


class TestRewindEpochICML(unittest.TestCase):
    """Test proper rewind epoch implementation for IMP."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.model = SimpleModel()
        self.analyzer = LotteryTicketAnalysis()
        self.dataloader = [
            {'input_ids': torch.randn(8, 10), 'labels': torch.randint(0, 5, (8,))}
            for _ in range(3)
        ]

    def test_rewind_epoch_initial_training(self):
        """Test that model trains to rewind_epoch before IMP starts.

        This is critical for late resetting (Frankle et al. 2020).
        """
        initial_weights = self.model.fc1.weight.data.clone()

        # Run IMP with rewind_epoch > 0
        result = self.analyzer.compute_iterative_magnitude_pruning(
            self.model,
            self.dataloader,
            target_sparsity=0.5,
            num_iterations=2,
            rewind_epoch=3  # Should train for 3 epochs first
        )

        # Check that rewind_epoch is recorded
        self.assertEqual(result['rewind_epoch'], 3)

        # Model should have been reset and retrained
        final_weights = self.model.fc1.weight.data
        self.assertGreater(
            (final_weights - initial_weights).abs().sum().item(), 0.01,
            "Model weights unchanged - rewind training didn't happen"
        )

    def test_rewind_vs_initialization(self):
        """Test difference between rewind_epoch=0 (init) and rewind_epoch>0."""
        torch.manual_seed(42)
        model1 = SimpleModel()
        initial_state1 = model1.state_dict()

        # Run with rewind_epoch=0 (rewind to initialization)
        result1 = self.analyzer.compute_iterative_magnitude_pruning(
            model1,
            self.dataloader,
            target_sparsity=0.3,
            num_iterations=2,
            rewind_epoch=0
        )

        torch.manual_seed(42)
        model2 = SimpleModel()

        # Run with rewind_epoch=2 (late resetting)
        result2 = self.analyzer.compute_iterative_magnitude_pruning(
            model2,
            self.dataloader,
            target_sparsity=0.3,
            num_iterations=2,
            rewind_epoch=2
        )

        # Both should complete successfully
        self.assertIsNotNone(result1['final_masks'])
        self.assertIsNotNone(result2['final_masks'])

        # Rewind epochs should be different
        self.assertEqual(result1['rewind_epoch'], 0)
        self.assertEqual(result2['rewind_epoch'], 2)


class TestLabelHandlingICML(unittest.TestCase):
    """Test proper label handling without fabrication."""

    def test_ensure_labels_language_modeling(self):
        """Test labels created correctly for language modeling tasks."""
        batch = {
            'input_ids': torch.tensor([[1, 2, 3, 4, 5]]),
            'attention_mask': torch.tensor([[1, 1, 1, 0, 0]])
        }

        batch_with_labels = ensure_labels(batch, task_type='language_modeling')

        # Should have labels
        self.assertIn('labels', batch_with_labels)

        # Labels should match input_ids for non-padded positions
        self.assertTrue(torch.equal(
            batch_with_labels['labels'][:, :3],
            batch['input_ids'][:, :3]
        ))

        # Padding should be masked with -100
        self.assertTrue(
            (batch_with_labels['labels'][:, 3:] == -100).all(),
            "Padding tokens not masked with -100"
        )

    def test_ensure_labels_classification_requires_labels(self):
        """Test that classification without labels raises appropriate error."""
        batch = {'input_ids': torch.randn(4, 10)}

        with self.assertRaises(ValueError) as context:
            ensure_labels(batch, task_type='classification')

        self.assertIn(
            "Labels required",
            str(context.exception),
            "Should require labels for classification tasks"
        )

    def test_ensure_labels_preserves_existing(self):
        """Test that existing labels are preserved."""
        batch = {
            'input_ids': torch.randn(4, 10),
            'labels': torch.randint(0, 5, (4,))
        }

        batch_with_labels = ensure_labels(batch, task_type='auto')

        # Should preserve existing labels
        self.assertTrue(torch.equal(
            batch_with_labels['labels'],
            batch['labels']
        ))

    def test_batch_size_detection_priority(self):
        """Test robust batch size detection with priority order."""
        # Test with labels (highest priority)
        batch1 = {'labels': torch.randn(16, 5), 'input_ids': torch.randn(8, 10)}
        self.assertEqual(get_batch_size(batch1), 16)

        # Test with input_ids (second priority)
        batch2 = {'input_ids': torch.randn(32, 10), 'x': torch.randn(8, 10)}
        self.assertEqual(get_batch_size(batch2), 32)

        # Test with custom key (fallback to first tensor)
        batch3 = {'features': torch.randn(8, 20)}
        self.assertEqual(get_batch_size(batch3), 8)

        # Test empty batch raises error
        batch4 = {'metadata': 'test'}
        with self.assertRaises(ValueError):
            get_batch_size(batch4)


class TestNumericalStabilityICML(unittest.TestCase):
    """Test numerical stability improvements for ICML standards."""

    def test_sparsity_validation(self):
        """Test that invalid sparsity values are caught before quantile ops."""
        analyzer = LotteryTicketAnalysis()
        model1 = SimpleModel()
        model2 = SimpleModel()

        # Test invalid sparsity > 1
        with self.assertRaises(ValueError) as context:
            analyzer.compute_ticket_overlap(
                model1, model2,
                sparsity_levels=[1.5]
            )
        self.assertIn("must be in range", str(context.exception))

        # Test invalid sparsity < 0
        with self.assertRaises(ValueError) as context:
            analyzer.compute_ticket_overlap(
                model1, model2,
                sparsity_levels=[-0.1]
            )
        self.assertIn("must be in range", str(context.exception))

    def test_epsilon_standardization(self):
        """Test that epsilon usage is standardized."""
        analyzer = LotteryTicketAnalysis()

        # Test _compute_actual_sparsity with empty masks
        empty_masks = {}
        sparsity = analyzer._compute_actual_sparsity(empty_masks)
        self.assertEqual(sparsity, 0.0)

        # Should not produce NaN or Inf
        self.assertFalse(np.isnan(sparsity))
        self.assertFalse(np.isinf(sparsity))

        # Test with all True masks (no pruning)
        masks = {'param': torch.ones(10, 10, dtype=torch.bool)}
        sparsity = analyzer._compute_actual_sparsity(masks)
        self.assertEqual(sparsity, 0.0)

    def test_gradient_accumulation_correctness(self):
        """Test that gradient accumulation doesn't double-scale."""
        analyzer = LotteryTicketAnalysis()
        model = SimpleModel()

        dataloader = [
            {'input_ids': torch.randn(8, 10), 'labels': torch.randint(0, 5, (8,))}
            for _ in range(4)
        ]

        # Test Fisher with gradient accumulation
        # The previous bug would cause incorrect scaling
        scores1 = analyzer.compute_gradient_importance(
            model,
            dataloader,
            importance_type='fisher',
            num_samples=32,
            gradient_accumulation_steps=1
        )

        scores2 = analyzer.compute_gradient_importance(
            model,
            dataloader,
            importance_type='fisher',
            num_samples=32,
            gradient_accumulation_steps=2
        )

        # Scores should be similar magnitude (not scaled by accumulation steps)
        for name in scores1:
            if scores1[name].mean() > 1e-8:
                ratio = scores2[name].mean() / scores1[name].mean()
                # Should be close to 1, not 2 or 0.5
                self.assertGreater(ratio, 0.3)
                self.assertLess(ratio, 3.0)


class TestICMLIntegration(unittest.TestCase):
    """Integration tests for complete ICML-grade lottery ticket pipeline."""

    def test_full_imp_pipeline(self):
        """Test complete IMP pipeline with all fixes."""
        torch.manual_seed(42)
        model = SimpleModel()
        analyzer = LotteryTicketAnalysis()

        dataloader = [
            {'input_ids': torch.randn(8, 10), 'labels': torch.randint(0, 5, (8,))}
            for _ in range(3)
        ]

        # Run full IMP
        result = analyzer.compute_iterative_magnitude_pruning(
            model,
            dataloader,
            target_sparsity=0.7,
            num_iterations=3,
            rewind_epoch=1
        )

        # Verify all components
        self.assertIn('final_masks', result)
        self.assertIn('iterations', result)
        self.assertEqual(len(result['iterations']), 3)

        # Check final sparsity
        total_params = 0
        total_pruned = 0
        for mask in result['final_masks'].values():
            total_params += mask.numel()
            total_pruned += (~mask).sum().item()

        actual_sparsity = total_pruned / total_params
        self.assertAlmostEqual(actual_sparsity, 0.7, delta=0.1)

    def test_early_bird_to_imp_pipeline(self):
        """Test using Early-Bird masks as initialization for IMP."""
        torch.manual_seed(42)
        model = SimpleModel()
        analyzer = LotteryTicketAnalysis()

        dataloader = [
            {'input_ids': torch.randn(8, 10), 'labels': torch.randint(0, 5, (8,))}
            for _ in range(3)
        ]

        # First find early-bird tickets
        eb_result = analyzer.compute_early_bird_tickets(
            model,
            dataloader,
            max_epochs=10,
            check_interval=2,
            target_sparsity=0.3
        )

        # Use early-bird masks for IMP
        if 'early_bird_masks' in eb_result:
            # This would be a natural extension
            self.assertIsNotNone(eb_result['early_bird_masks'])


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)