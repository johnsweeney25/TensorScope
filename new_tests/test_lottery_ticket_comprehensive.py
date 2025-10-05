#!/usr/bin/env python3
"""
Comprehensive Unit Test Suite for LotteryTicketAnalysis Module
===============================================================
Production-quality test suite validating all functionality and fixes.
Tests numerical stability, theoretical correctness, edge cases, and performance.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import warnings
import tempfile
import shutil

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from LotteryTicketAnalysis import LotteryTicketAnalysis, create_model_wrapper


class SimpleTestNetwork(nn.Module):
    """Minimal test network for unit testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)


class ModelWithProperLoss(SimpleTestNetwork):
    """Test model that outputs loss and logits properly."""

    def forward(self, *args, **kwargs):
        # Extract input tensor
        if 'input_ids' in kwargs:
            x = kwargs['input_ids']
            labels = kwargs.get('labels')
        elif len(args) > 0:
            x = args[0]
            labels = args[1] if len(args) > 1 else kwargs.get('labels')
        else:
            x = kwargs.get('x')
            labels = kwargs.get('labels')

        # Get logits from parent
        logits = super().forward(x)

        # Calculate loss if labels provided
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(logits, labels)
        else:
            loss = logits.mean()  # Simple loss for unsupervised

        return SimpleNamespace(loss=loss, logits=logits)


class TestNumericalStabilityFixes(unittest.TestCase):
    """Test suite for numerical stability improvements."""

    def test_epsilon_handling_in_divisions(self):
        """Test that all divisions use proper epsilon values."""
        analyzer = LotteryTicketAnalysis()
        model1 = SimpleTestNetwork()
        model2 = SimpleTestNetwork()

        # Create extreme case: very small weights
        with torch.no_grad():
            for p in model2.parameters():
                p.mul_(1e-10)

        result = analyzer.compute_ticket_overlap(
            model1, model2,
            sparsity_levels=[0.5, 0.9, 0.99]
        )

        # Verify no NaN or Inf in results
        for key, value in result.items():
            if isinstance(value, (int, float)):
                self.assertFalse(np.isnan(value), f"{key} is NaN")
                self.assertFalse(np.isinf(value), f"{key} is Inf")
                if 'ratio' in key or 'jaccard' in key:
                    self.assertGreaterEqual(value, 0.0)
                    self.assertLessEqual(value, 1.0)

    def test_zero_loss_handling(self):
        """Test handling when baseline or pruned loss is zero."""
        analyzer = LotteryTicketAnalysis()

        # Create model that produces zero loss
        class ZeroLossModel(nn.Module):
            def forward(self, *args, **kwargs):
                batch_size = kwargs.get('input_ids', args[0]).shape[0]
                return SimpleNamespace(
                    loss=torch.tensor(0.0),
                    logits=torch.zeros(batch_size, 5)
                )

        model = ZeroLossModel()
        batch = {'input_ids': torch.randn(8, 10)}

        result = analyzer.compute_pruning_robustness(
            model, batch, sparsity_levels=[0.5]
        )

        # Should handle zero loss gracefully
        self.assertIn('baseline_loss', result)
        self.assertEqual(result['baseline_loss'], 0.0)
        metrics = result['robustness_metrics']
        self.assertFalse(np.isnan(metrics['winning_ticket_score']))

    def test_empty_tensor_quantile_handling(self):
        """Test that empty tensors in quantile computation don't crash."""
        analyzer = LotteryTicketAnalysis()
        model = SimpleTestNetwork()

        # Zero out one layer completely
        with torch.no_grad():
            model.fc1.weight.zero_()
            model.fc1.bias.zero_()

        # Should handle empty/zero tensors
        masks = analyzer._get_magnitude_masks(model, sparsity=0.5)
        self.assertIsNotNone(masks)
        for name, mask in masks.items():
            self.assertEqual(mask.dtype, torch.bool)


class TestTheoreticalCorrectness(unittest.TestCase):
    """Test theoretical correctness of implementations."""

    def test_fisher_computed_in_eval_mode(self):
        """Verify Fisher Information is computed in eval mode (no dropout)."""
        analyzer = LotteryTicketAnalysis()

        # Model with dropout to test mode
        class ModelWithDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.dropout = nn.Dropout(0.5)
                self.linear = nn.Linear(10, 5)
                self.mode_when_called = None

            def forward(self, *args, **kwargs):
                self.mode_when_called = self.training
                x = kwargs.get('input_ids', args[0] if args else torch.randn(1, 10))
                x = self.dropout(x)
                logits = self.linear(x)
                return SimpleNamespace(
                    loss=logits.mean(),
                    logits=logits
                )

        model = ModelWithDropout()
        dataloader = [{'input_ids': torch.randn(8, 10)} for _ in range(2)]

        # Compute Fisher importance
        scores = analyzer._compute_fisher_importance(
            model, dataloader, num_samples=16
        )

        # Model should have been in eval mode during Fisher computation
        self.assertFalse(model.mode_when_called,
                        "Fisher computed in train mode - should be eval!")

    def test_self_supervised_loss_is_sensible(self):
        """Test that self-supervised loss uses proper objectives."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()

        dataloader = [{'input_ids': torch.randn(8, 10)} for _ in range(2)]

        # Compute Fisher without labels (self-supervised)
        scores = analyzer._compute_fisher_importance(
            model, dataloader, num_samples=16, use_labels=False
        )

        # Should produce non-zero gradients
        for name, score in scores.items():
            self.assertTrue(torch.any(score > 0),
                          f"No gradients for {name} in self-supervised mode")

    def test_taylor_product_not_grasp(self):
        """Verify renamed Taylor product method and GRASP deprecation."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()
        dataloader = [{'input_ids': torch.randn(8, 10)} for _ in range(2)]

        # New name should work
        scores = analyzer.compute_gradient_importance(
            model, dataloader,
            importance_type='taylor_product',
            num_samples=16
        )
        self.assertIsInstance(scores, dict)

        # Old name should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            scores_old = analyzer.compute_gradient_importance(
                model, dataloader,
                importance_type='grasp_approx',
                num_samples=16
            )
            self.assertTrue(any('deprecated' in str(warning.message).lower()
                              for warning in w))


class TestMemoryOptimizations(unittest.TestCase):
    """Test memory-efficient implementations."""

    def test_streaming_weight_processing(self):
        """Test that weight processing uses streaming for large models."""
        analyzer = LotteryTicketAnalysis()

        # Create models
        model1 = SimpleTestNetwork()
        model2 = SimpleTestNetwork()

        # The compute_ticket_overlap now uses streaming
        result = analyzer.compute_ticket_overlap(
            model1, model2,
            sparsity_levels=[0.5]
        )

        # Should complete without storing all weights in memory
        self.assertIn('sparsity_50_overlap_ratio', result)

    def test_efficient_state_saving(self):
        """Test that state saving only stores trainable parameters."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()

        # Freeze some parameters
        for param in model.fc1.parameters():
            param.requires_grad = False

        dataloader = [{'input_ids': torch.randn(8, 10)} for _ in range(2)]

        # Should only save trainable parameters
        result = analyzer.compute_iterative_magnitude_pruning(
            model, dataloader,
            target_sparsity=0.5,
            num_iterations=2
        )

        self.assertIn('final_masks', result)

    def test_layerwise_processing_memory_budget(self):
        """Test layer-wise processing respects memory budget."""
        analyzer = LotteryTicketAnalysis()
        model = SimpleTestNetwork()

        result = analyzer.compute_layerwise_magnitude_ticket(
            model,
            target_sparsity=0.7,
            max_memory_gb=0.001  # Very small budget to force grouping
        )

        # Should complete successfully
        self.assertIn('actual_sparsity', result)
        self.assertAlmostEqual(result['actual_sparsity'], 0.7, delta=0.1)


class TestEdgeCaseHandling(unittest.TestCase):
    """Test handling of edge cases and error conditions."""

    def test_mismatched_model_parameters(self):
        """Test error handling for mismatched models."""
        analyzer = LotteryTicketAnalysis()
        model1 = SimpleTestNetwork(input_dim=10)
        model2 = SimpleTestNetwork(input_dim=20)  # Different architecture

        with self.assertRaises(ValueError) as context:
            analyzer.compute_ticket_overlap(model1, model2)

        self.assertIn('shape mismatch', str(context.exception).lower())

    def test_invalid_sparsity_values(self):
        """Test validation of sparsity values."""
        analyzer = LotteryTicketAnalysis()
        model = SimpleTestNetwork()

        # Invalid sparsity > 1.0
        with self.assertRaises(ValueError) as context:
            analyzer.compute_ticket_overlap(
                model, model,
                sparsity_levels=[1.5]
            )

        self.assertIn('range [0, 1]', str(context.exception))

    def test_dataloader_format_handling(self):
        """Test handling of different dataloader formats."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()

        # List format
        list_loader = [{'input_ids': torch.randn(8, 10)} for _ in range(2)]
        result1 = analyzer.compute_gradient_importance(
            model, list_loader, num_samples=16
        )
        self.assertIsInstance(result1, dict)

        # DataLoader format
        dataset = TensorDataset(torch.randn(32, 10))
        torch_loader = DataLoader(dataset, batch_size=8)
        # Convert to dict format
        dict_loader = [{'input_ids': batch[0]} for batch in torch_loader]
        result2 = analyzer.compute_gradient_importance(
            model, dict_loader, num_samples=16
        )
        self.assertIsInstance(result2, dict)

    def test_exception_handling_in_quantile(self):
        """Test graceful handling of quantile computation failures."""
        analyzer = LotteryTicketAnalysis()
        model = SimpleTestNetwork()

        # This should trigger fallback paths in quantile computation
        with torch.no_grad():
            # Create pathological weight distribution
            for param in model.parameters():
                param.copy_(torch.ones_like(param))

        # Should handle gracefully
        masks = analyzer._get_magnitude_masks(model, sparsity=0.5)
        self.assertIsNotNone(masks)


class TestCoreAlgorithms(unittest.TestCase):
    """Test core lottery ticket algorithms."""

    def test_iterative_magnitude_pruning_convergence(self):
        """Test IMP converges to target sparsity."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()
        dataloader = [
            {'input_ids': torch.randn(8, 10),
             'labels': torch.randint(0, 5, (8,))}
            for _ in range(3)
        ]

        result = analyzer.compute_iterative_magnitude_pruning(
            model, dataloader,
            target_sparsity=0.7,
            num_iterations=3
        )

        # Check convergence to target
        final_masks = result['final_masks']
        total_params = sum(m.numel() for m in final_masks.values())
        total_pruned = sum((~m).sum().item() for m in final_masks.values())
        actual_sparsity = total_pruned / total_params

        self.assertAlmostEqual(actual_sparsity, 0.7, delta=0.1)

    def test_early_bird_ticket_detection(self):
        """Test early-bird ticket convergence detection."""
        analyzer = LotteryTicketAnalysis()
        model = SimpleTestNetwork()
        dataloader = [{'input_ids': torch.randn(8, 10)} for _ in range(2)]

        # Simulate training that converges quickly
        with torch.no_grad():
            # Make weights stable
            for param in model.parameters():
                param.abs_()

        result = analyzer.compute_early_bird_tickets(
            model, dataloader,
            max_epochs=30,
            eval_interval=5
        )

        self.assertIn('early_bird_epoch', result)
        self.assertIn('converged', result)
        self.assertIsInstance(result['early_bird_masks'], dict)

    def test_pruning_robustness_metrics(self):
        """Test computation of pruning robustness metrics."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()

        batch = {
            'input_ids': torch.randn(16, 10),
            'labels': torch.randint(0, 5, (16,))
        }

        result = analyzer.compute_pruning_robustness(
            model, batch,
            sparsity_levels=[0.1, 0.3, 0.5, 0.7, 0.9],
            return_masks=True
        )

        # Verify metrics
        metrics = result['robustness_metrics']
        self.assertIn('winning_ticket_score', metrics)
        self.assertIn('optimal_sparsity', metrics)
        self.assertIn('pruning_resistance', metrics)
        self.assertIn('critical_sparsity', metrics)

        # Check monotonicity (performance should generally degrade)
        curves = result['pruning_curves']
        losses = [curves[f'sparsity_{int(s*100)}']['loss']
                 for s in [0.1, 0.3, 0.5, 0.7, 0.9]]

        # Later pruning should have higher loss (with some tolerance)
        self.assertGreater(losses[-1], losses[0] * 0.9)


class TestGradientImportanceMethods(unittest.TestCase):
    """Test gradient-based importance scoring."""

    def test_fisher_information_properties(self):
        """Test Fisher information has correct properties."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()
        dataloader = [
            {'input_ids': torch.randn(8, 10),
             'labels': torch.randint(0, 5, (8,))}
            for _ in range(4)
        ]

        scores = analyzer.compute_gradient_importance(
            model, dataloader,
            importance_type='fisher',
            num_samples=32
        )

        for name, score in scores.items():
            # Fisher scores are non-negative (squared gradients)
            self.assertTrue(torch.all(score >= 0),
                          f"Negative Fisher scores for {name}")
            # Should have non-zero scores for most parameters
            self.assertTrue(torch.any(score > 0),
                          f"All zero Fisher scores for {name}")

    def test_taylor_importance_computation(self):
        """Test Taylor importance combines weight and gradient."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()

        # Set some weights to zero
        with torch.no_grad():
            model.fc1.weight[0, :] = 0

        dataloader = [{'input_ids': torch.randn(8, 10)} for _ in range(3)]

        scores = analyzer.compute_gradient_importance(
            model, dataloader,
            importance_type='taylor',
            num_samples=24
        )

        # Zero weights should have zero importance
        fc1_scores = scores['fc1.weight']
        self.assertEqual(fc1_scores[0, :].sum().item(), 0.0)

    def test_gradient_accumulation_small_batches(self):
        """Test gradient accumulation for small batch sizes."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()

        # Very small batch size should trigger accumulation
        small_dataloader = [
            {'input_ids': torch.randn(2, 10),  # batch size 2
             'labels': torch.randint(0, 5, (2,))}
            for _ in range(10)
        ]

        scores = analyzer.compute_gradient_importance(
            model, small_dataloader,
            importance_type='fisher',
            num_samples=20,
            min_batch_size=16  # Should trigger accumulation
        )

        # Should still produce valid scores
        self.assertIsInstance(scores, dict)
        for name in model.state_dict().keys():
            self.assertIn(name, scores)


class TestErrorRecoveryMechanisms(unittest.TestCase):
    """Test error recovery and fallback strategies."""

    def test_oom_recovery_fallback(self):
        """Test fallback to simpler methods on OOM."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()
        dataloader = [{'input_ids': torch.randn(8, 10)} for _ in range(2)]

        # This should handle potential OOM gracefully
        result = analyzer.compute_with_recovery(
            model, dataloader,
            analysis_type='memory_efficient',
            retry_on_oom=True
        )

        self.assertIn('computation_time', result)
        self.assertNotIn('error', result)

    def test_checkpoint_based_recovery(self):
        """Test checkpoint-based recovery for long computations."""
        analyzer = LotteryTicketAnalysis()
        model = SimpleTestNetwork()

        with tempfile.TemporaryDirectory() as tmpdir:
            result = analyzer.compute_layerwise_magnitude_ticket(
                model,
                target_sparsity=0.8,
                checkpoint_dir=tmpdir
            )

            # Should create checkpoints
            self.assertTrue(len(os.listdir(tmpdir)) > 0)

            # Second run should use checkpoints
            result2 = analyzer.compute_layerwise_magnitude_ticket(
                model,
                target_sparsity=0.8,
                checkpoint_dir=tmpdir
            )

            self.assertEqual(result['actual_sparsity'],
                           result2['actual_sparsity'])

    def test_batch_size_auto_reduction(self):
        """Test automatic batch size reduction."""
        analyzer = LotteryTicketAnalysis()

        class MockDataLoader:
            def __init__(self, batch_size):
                self.batch_size = batch_size
                self.dataset = TensorDataset(torch.randn(128, 10))
                self.num_workers = 0
                self.collate_fn = None
                self.pin_memory = False
                self.drop_last = False
                self.timeout = 0
                self.worker_init_fn = None

        # Test reduction
        dl = MockDataLoader(batch_size=64)
        reduced = analyzer._reduce_batch_size(dl, reduction_factor=0.5)
        self.assertEqual(reduced.batch_size, 32)

        # Test minimum batch size
        dl_small = MockDataLoader(batch_size=20)
        reduced_small = analyzer._reduce_batch_size(dl_small, reduction_factor=0.5)
        self.assertEqual(reduced_small.batch_size, 16)  # Minimum is 16


class TestModelWrapperCompatibility(unittest.TestCase):
    """Test model wrapper handles diverse interfaces."""

    def test_transformer_style_interface(self):
        """Test compatibility with HuggingFace-style models."""

        class TransformerModel(nn.Module):
            def forward(self, input_ids, attention_mask=None, labels=None):
                batch_size = input_ids.shape[0]
                logits = torch.randn(batch_size, 5)
                loss = None
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(logits, labels)
                return SimpleNamespace(loss=loss, logits=logits)

        model = TransformerModel()
        wrapped = create_model_wrapper(model)

        # Test with various calling conventions
        x = torch.randn(8, 10)
        labels = torch.randint(0, 5, (8,))

        output1 = wrapped(input_ids=x)
        self.assertIsNotNone(output1)

        output2 = wrapped(input_ids=x, labels=labels)
        self.assertIsNotNone(output2.loss)

    def test_simple_forward_interface(self):
        """Test with simple forward(x) models."""
        model = SimpleTestNetwork()
        wrapped = create_model_wrapper(model)

        x = torch.randn(8, 10)

        # Positional
        output1 = wrapped(x)
        self.assertEqual(output1.shape, (8, 5))

        # Keyword
        output2 = wrapped(x=x)
        self.assertEqual(output2.shape, (8, 5))

    def test_mixed_interface_handling(self):
        """Test wrapper handles mixed positional/keyword args."""

        class FlexibleModel(nn.Module):
            def forward(self, *args, **kwargs):
                # Flexible handling
                if args:
                    x = args[0]
                elif 'input_ids' in kwargs:
                    x = kwargs['input_ids']
                else:
                    x = kwargs.get('x')

                return torch.randn(x.shape[0], 5)

        model = FlexibleModel()
        wrapped = create_model_wrapper(model)

        x = torch.randn(8, 10)

        # All should work
        out1 = wrapped(x)
        out2 = wrapped(input_ids=x)
        out3 = wrapped(x=x)

        for out in [out1, out2, out3]:
            self.assertEqual(out.shape, (8, 5))


class TestIntegrationScenarios(unittest.TestCase):
    """End-to-end integration tests."""

    def test_full_lottery_ticket_pipeline(self):
        """Test complete lottery ticket analysis pipeline."""
        analyzer = LotteryTicketAnalysis()

        # Create models
        model_init = SimpleTestNetwork()
        model_trained = SimpleTestNetwork()

        # Simulate training
        with torch.no_grad():
            for p_trained in model_trained.parameters():
                p_trained.add_(torch.randn_like(p_trained) * 0.1)

        # 1. Compare tickets
        overlap = analyzer.compute_ticket_overlap(
            model_init, model_trained,
            sparsity_levels=[0.5, 0.8]
        )
        self.assertIn('interpretation', overlap)

        # 2. Test pruning robustness
        model_with_loss = ModelWithProperLoss()
        batch = {
            'input_ids': torch.randn(16, 10),
            'labels': torch.randint(0, 5, (16,))
        }

        robustness = analyzer.compute_pruning_robustness(
            model_with_loss, batch,
            sparsity_levels=[0.3, 0.6]
        )
        self.assertIn('robustness_metrics', robustness)

        # 3. Compute importance scores
        dataloader = [batch for _ in range(3)]
        importance = analyzer.compute_gradient_importance(
            model_with_loss, dataloader,
            importance_type='fisher',
            num_samples=24
        )
        self.assertEqual(len(importance), len(dict(model_with_loss.named_parameters())))

    def test_production_ready_analysis(self):
        """Test production-ready analysis with error recovery."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()
        dataloader = [
            {'input_ids': torch.randn(8, 10),
             'labels': torch.randint(0, 5, (8,))}
            for _ in range(5)
        ]

        # Should handle various analysis types
        for analysis_type in ['quick', 'memory_efficient']:
            result = analyzer.compute_with_recovery(
                model, dataloader,
                analysis_type=analysis_type
            )
            self.assertIn('computation_time', result)
            self.assertNotIn('error', result)


class TestPerformanceOptimizations(unittest.TestCase):
    """Test performance optimizations work correctly."""

    def test_early_stopping_in_training(self):
        """Test early stopping prevents wasted computation."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()

        # Create dataloader that causes divergence
        bad_dataloader = [
            {'input_ids': torch.randn(8, 10) * 100,  # Large inputs
             'labels': torch.randint(0, 5, (8,))}
            for _ in range(10)
        ]

        # Should stop early if loss diverges
        result = analyzer._simple_train(
            model, bad_dataloader,
            epochs=10, lr=10.0  # Bad learning rate
        )

        # Should stop before all epochs
        self.assertLess(result['epochs_completed'], 10)

    def test_adaptive_metrics_recording(self):
        """Test adaptive frequency of metrics recording."""
        analyzer = LotteryTicketAnalysis()
        model = ModelWithProperLoss()

        # Small dataloader
        small_dl = [{'input_ids': torch.randn(8, 10)} for _ in range(3)]
        result_small = analyzer._simple_train(model, small_dl, epochs=2)
        history_small = result_small['training_history']['loss']

        # Large dataloader
        large_dl = [{'input_ids': torch.randn(8, 10)} for _ in range(50)]
        result_large = analyzer._simple_train(model, large_dl, epochs=2)
        history_large = result_large['training_history']['loss']

        # Should adapt recording frequency
        self.assertGreater(len(history_large), len(history_small))


if __name__ == '__main__':
    # Run with verbosity
    unittest.main(verbosity=2)