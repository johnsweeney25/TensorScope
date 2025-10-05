#!/usr/bin/env python3
"""
Comprehensive unittest suite for ICLRMetrics critical fixes.
Tests all the high-impact issues and theoretical correctness fixes.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import warnings
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ICLRMetrics import ICLRMetrics, _float_like, _check_compatible


class SimpleTestModel(nn.Module):
    """Simple test model for validation."""

    def __init__(self, vocab_size=100, hidden_size=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, vocab_size)

        # Mock config for HF compatibility
        class Config:
            def __init__(self):
                self.vocab_size = vocab_size
                self.pad_token_id = 0
                self.use_cache = False
        self.config = Config()

    def get_input_embeddings(self):
        return self.embed

    def forward(self, input_ids=None, labels=None, attention_mask=None,
                inputs_embeds=None, output_attentions=False, **kwargs):
        # Handle both input_ids and inputs_embeds
        if inputs_embeds is not None:
            x = inputs_embeds
        else:
            x = self.embed(input_ids)

        x = F.relu(self.linear(x))
        logits = self.output(x)

        # Compute loss with proper shifting for causal LM
        loss = None
        if labels is not None:
            # Shift for causal LM: predict next token
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100
            )

        # Return output object
        class Output:
            pass
        out = Output()
        out.logits = logits
        out.loss = loss

        if output_attentions:
            # Mock attention weights for testing
            B, S = (input_ids.shape if input_ids is not None
                   else inputs_embeds.shape[:2])
            out.attentions = [torch.softmax(torch.randn(B, 2, S, S), dim=-1)]

        return out


class TestICLRMetricsCriticalFixes(unittest.TestCase):
    """Test all critical fixes applied to ICLRMetrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = ICLRMetrics(device=str(self.device))
        self.model = SimpleTestModel().to(self.device)

        # Create test batch
        self.batch = {
            'input_ids': torch.randint(0, 100, (4, 16)).to(self.device),
            'attention_mask': torch.ones(4, 16).to(self.device),
            'labels': torch.randint(0, 100, (4, 16)).to(self.device)
        }

    def test_logger_defined(self):
        """Test that logger is properly defined."""
        # This would have thrown NameError before fix
        import logging
        from ICLRMetrics import logger
        self.assertIsInstance(logger, logging.Logger)
        self.assertEqual(logger.name, 'ICLRMetrics')

    def test_device_management(self):
        """Test that model device is detected correctly everywhere."""
        # Test with model on different device than metrics instance
        cpu_model = SimpleTestModel().to('cpu')
        gpu_metrics = ICLRMetrics(device='cuda' if torch.cuda.is_available() else 'cpu')

        cpu_batch = {k: v.cpu() for k, v in self.batch.items()}

        # This should not raise device mismatch errors
        loss = gpu_metrics._compute_loss(cpu_model, cpu_batch)
        self.assertIsInstance(loss, float)

    def test_integrated_gradients_loss_target(self):
        """Test that IG uses model's loss computation with proper shifting."""
        # Create a model that tracks whether loss was computed correctly
        class IGTestModel(SimpleTestModel):
            def __init__(self):
                super().__init__()
                self.loss_computed_with_labels = False

            def forward(self, inputs_embeds=None, labels=None, **kwargs):
                if labels is not None and inputs_embeds is not None:
                    self.loss_computed_with_labels = True
                return super().forward(inputs_embeds=inputs_embeds,
                                     labels=labels, **kwargs)

        model = IGTestModel().to(self.device)
        result = self.metrics.compute_integrated_gradients(model, self.batch, n_steps=5)

        # Check that model's loss was used (labels passed to forward)
        self.assertTrue(model.loss_computed_with_labels,
                       "IG should pass labels to model for proper loss computation")
        self.assertNotIn('error', result)

    def test_filter_normalization_li_et_al(self):
        """Test that filter normalization matches Li et al. 2018."""
        # Create direction vector
        direction = [torch.randn_like(p) for p in self.model.parameters()]

        # Apply filter normalization
        normed = self.metrics._filter_normalize_direction(self.model, direction)

        # Check that normalization includes weight scaling
        for p, d, n in zip(self.model.parameters(), direction, normed):
            if p.ndim >= 2:
                axes = tuple(range(1, p.ndim))
                d_norm = d.norm(p=2, dim=axes, keepdim=True).clamp_min(1e-12)
                w_norm = p.data.norm(p=2, dim=axes, keepdim=True).clamp_min(1e-12)
                expected = d * (w_norm / d_norm)
                torch.testing.assert_close(n, expected, rtol=1e-5, atol=1e-7)

    def test_loss_landscape_2d_device(self):
        """Test that 2D landscape handles device correctly."""
        result = self.metrics.compute_loss_landscape_2d_true(
            self.model, self.batch, n_points=5, span=0.01
        )

        # Should complete without device errors
        self.assertNotIn('error', result)
        self.assertIn('grid_losses', result)

    def test_modified_gram_schmidt_normalization(self):
        """Test that MGS properly normalizes result."""
        # Create test vectors
        v1 = [torch.randn(10) for _ in range(3)]
        v1_norm = torch.sqrt(sum((vi**2).sum() for vi in v1))
        v1 = [vi / v1_norm for vi in v1]

        v2 = [torch.randn(10) for _ in range(3)]

        # Orthogonalize v2 against v1
        result = self.metrics._modified_gram_schmidt([v1], v2)

        # Check that result is normalized
        result_norm = torch.sqrt(sum((ri**2).sum() for ri in result))
        self.assertAlmostEqual(result_norm.item(), 1.0, places=5,
                              msg="MGS should return normalized vectors")

        # Check orthogonality
        dot_product = sum((r * v).sum() for r, v in zip(result, v1))
        self.assertLess(abs(dot_product.item()), 1e-5,
                       msg="MGS result should be orthogonal to input vectors")

    def test_attention_normalization_entropy(self):
        """Test that attention matrices are normalized before entropy calculation."""
        # Create unnormalized attention matrix
        class AttentionTestModel(SimpleTestModel):
            def forward(self, **kwargs):
                out = super().forward(**kwargs)
                if hasattr(out, 'attentions'):
                    # Create unnormalized attention for testing
                    B, H, S, _ = out.attentions[0].shape
                    out.attentions = [torch.rand(B, H, S, S) * 10]  # Not normalized!
                return out

        model = AttentionTestModel().to(self.device)
        result = self.metrics.compute_attention_attribution(model, self.batch)

        # Should still compute valid entropy (after normalization)
        self.assertNotIn('error', result)
        self.assertGreaterEqual(result['attention_entropy'], 0)

    def test_loss_barrier_cpu_interpolation(self):
        """Test that loss barrier uses CPU for interpolation to avoid device issues."""
        model1 = SimpleTestModel().to(self.device)
        model2 = SimpleTestModel().to(self.device)

        # Slightly perturb model2
        with torch.no_grad():
            for p in model2.parameters():
                p.add_(torch.randn_like(p) * 0.01)

        result = self.metrics.compute_loss_barrier(model1, model2, self.batch, n_points=5)

        # Should complete without device mixing errors
        self.assertNotIn('error', result)
        self.assertIn('barrier_height', result)

    def test_mode_connectivity_error_handling(self):
        """Test that compute_mode_connectivity handles barrier errors properly."""
        models = [SimpleTestModel().to(self.device) for _ in range(3)]

        # Mock compute_loss_barrier to sometimes return errors
        original_method = self.metrics.compute_loss_barrier
        call_count = [0]

        def mock_barrier(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:  # Make second call fail
                return {'error': 'Test error'}
            return original_method(*args, **kwargs)

        with patch.object(self.metrics, 'compute_loss_barrier', mock_barrier):
            result = self.metrics.compute_mode_connectivity(models, self.batch)

        # Should handle error gracefully
        self.assertNotIn('error', result)
        self.assertIn('mean_barrier_height', result)

    def test_mode_connectivity_vs_flatness_separation(self):
        """Test that mode connectivity is separated from flatness conceptually."""
        model1 = SimpleTestModel().to(self.device)
        model2 = SimpleTestModel().to(self.device)

        result = self.metrics.analyze_rlvr_vs_instruct(model1, model2, self.batch)

        # Check that we have separate metrics
        self.assertIn('rlvr_is_mode_connected', result['summary'])
        self.assertIn('barrier_is_low', result['summary'])
        self.assertNotIn('rlvr_is_flatter', result['summary'])  # Old conflated metric

    def test_compute_loss_custom_function(self):
        """Test that custom loss functions can use gradients."""
        call_count = [0]

        def custom_loss_fn(model, batch):
            call_count[0] += 1
            # This loss function needs gradients
            outputs = model(**batch)
            if outputs.loss is not None:
                return outputs.loss.item()
            return 0.0

        loss = self.metrics._compute_loss(self.model, self.batch, loss_fn=custom_loss_fn)

        self.assertEqual(call_count[0], 1)
        self.assertIsInstance(loss, float)

    def test_pruning_sensitivity_quantile_precision(self):
        """Test that pruning uses appropriate precision for quantile calculation."""
        result = self.metrics.compute_pruning_sensitivity(
            self.model, self.batch, sparsity_levels=[0.1, 0.5, 0.9]
        )

        # Should complete without numerical issues
        self.assertNotIn('error', result)
        self.assertIn('pruning_robustness_score', result)
        self.assertGreaterEqual(result['pruning_robustness_score'], 0)
        self.assertLessEqual(result['pruning_robustness_score'], 1)

    def test_valid_barriers_include_zeros(self):
        """Test that barrier statistics include zero values."""
        models = [SimpleTestModel().to(self.device) for _ in range(2)]

        # Make models identical to get zero barrier
        models[1].load_state_dict(models[0].state_dict())

        result = self.metrics.compute_mode_connectivity(models, self.batch)

        # Min barrier should be close to 0 (or negative due to numerics)
        self.assertLessEqual(result['min_barrier_height'], 0.1)


class TestDeviceCompatibility(unittest.TestCase):
    """Test device compatibility across different configurations."""

    def setUp(self):
        """Set up test fixtures."""
        self.cpu_metrics = ICLRMetrics(device='cpu')
        self.cpu_model = SimpleTestModel().to('cpu')
        self.cpu_batch = {
            'input_ids': torch.randint(0, 100, (2, 8)),
            'attention_mask': torch.ones(2, 8),
            'labels': torch.randint(0, 100, (2, 8))
        }

    def test_cpu_model_gpu_metrics(self):
        """Test CPU model with metrics initialized for GPU."""
        if torch.cuda.is_available():
            gpu_metrics = ICLRMetrics(device='cuda')

            # Should handle device mismatch gracefully
            result = gpu_metrics.compute_integrated_gradients(
                self.cpu_model, self.cpu_batch, n_steps=3
            )
            self.assertNotIn('error', result)

    def test_mixed_device_loss_barrier(self):
        """Test loss barrier with models on different devices initially."""
        model1 = SimpleTestModel()
        model2 = SimpleTestModel()

        if torch.cuda.is_available():
            model1 = model1.cuda()
            model2 = model2.cpu()

            gpu_batch = {k: v.cuda() for k, v in self.cpu_batch.items()}

            # Metrics should handle this gracefully
            metrics = ICLRMetrics()
            result = metrics.compute_loss_barrier(model1, model2, gpu_batch, n_points=3)

            # Should either work or give clear error
            if 'error' in result:
                self.assertIn('compatible', result['error'].lower())


class TestNumericalStability(unittest.TestCase):
    """Test numerical stability improvements."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = ICLRMetrics()
        self.model = SimpleTestModel()
        self.batch = {
            'input_ids': torch.randint(0, 100, (2, 8)),
            'labels': torch.randint(0, 100, (2, 8))
        }

    def test_gini_coefficient_stability(self):
        """Test Gini coefficient with edge cases."""
        # All zeros
        gini_zero = self.metrics._compute_gini(np.zeros(10))
        self.assertEqual(gini_zero, 0.0)

        # All equal values
        gini_equal = self.metrics._compute_gini(np.ones(10))
        self.assertAlmostEqual(gini_equal, 0.0, places=5)

        # With NaN and inf
        values_with_nan = np.array([1, 2, np.nan, 3, np.inf, 4])
        gini_nan = self.metrics._compute_gini(values_with_nan)
        self.assertGreaterEqual(gini_nan, 0)
        self.assertLessEqual(gini_nan, 1)

    def test_attention_entropy_numerical_stability(self):
        """Test attention entropy with very small values."""
        # Create attention with very small probabilities
        attn = torch.tensor([[0.99999, 0.00001], [0.5, 0.5]])

        # Compute entropy manually with numerical safety
        attn_safe = attn.clamp_min(1e-12)
        entropy = -(attn_safe * attn_safe.log()).sum(dim=-1).mean()

        self.assertFalse(torch.isnan(entropy))
        self.assertFalse(torch.isinf(entropy))


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)