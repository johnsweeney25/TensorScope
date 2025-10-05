#!/usr/bin/env python3
"""
Unit tests for ICLRMetrics audit fixes.
Tests the critical issues identified in the intern's audit.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ICLRMetrics_fixes import ICLRMetricsFixes


class SimpleTestModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.bias = nn.Parameter(torch.randn(vocab_size))
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'pad_token_id': 0,
            'use_cache': False
        })()

    def get_input_embeddings(self):
        return self.embedding

    def forward(self, input_ids=None, inputs_embeds=None, labels=None, attention_mask=None,
               output_attentions=False, use_cache=False, **kwargs):
        if inputs_embeds is None:
            inputs_embeds = self.embedding(input_ids)

        # Apply norm
        x = self.norm(inputs_embeds)

        # Simple processing
        logits = self.linear(x) + self.bias

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Create mock attention weights if requested
        attentions = None
        if output_attentions:
            batch_size, seq_len = inputs_embeds.shape[:2]
            # Create normalized attention matrix
            attn = torch.softmax(torch.randn(batch_size, 1, seq_len, seq_len), dim=-1)
            attentions = [attn]

        return type('Output', (), {
            'loss': loss,
            'logits': logits,
            'attentions': attentions
        })()


class TestTrainingModeRestoration(unittest.TestCase):
    """Test that training mode is always restored."""

    def setUp(self):
        self.model = SimpleTestModel()
        self.metrics = ICLRMetricsFixes()
        self.metrics._compute_loss = lambda m, b, fn=None: 1.0
        self.metrics._compute_gini = lambda x: 0.5

    def test_ig_training_mode_on_error(self):
        """Test IG restores training mode on error."""
        self.model.train()
        invalid_batch = {}  # Missing input_ids

        result = self.metrics.compute_integrated_gradients(self.model, invalid_batch)

        self.assertTrue(self.model.training, "Model should be in training mode after error")
        self.assertIn('error', result, "Should return error for invalid input")

    def test_ig_eval_mode_on_error(self):
        """Test IG restores eval mode on error."""
        self.model.eval()
        invalid_batch = {}

        result = self.metrics.compute_integrated_gradients(self.model, invalid_batch)

        self.assertFalse(self.model.training, "Model should be in eval mode after error")
        self.assertIn('error', result, "Should return error for invalid input")

    def test_attention_training_mode_on_error(self):
        """Test attention restores training mode on error."""
        self.model.train()
        invalid_batch = {}

        result = self.metrics.compute_attention_attribution(self.model, invalid_batch)

        self.assertTrue(self.model.training, "Model should be in training mode after error")
        self.assertIn('error', result, "Should return error for invalid input")

    def test_pruning_training_mode_on_completion(self):
        """Test pruning restores training mode after completion."""
        self.model.train()
        batch = {'input_ids': torch.randint(0, 100, (2, 10))}

        result = self.metrics.compute_pruning_sensitivity(self.model, batch, sparsity_levels=[0.5])

        self.assertTrue(self.model.training, "Model should be in training mode after completion")
        self.assertIn('baseline_loss', result, "Should return valid results")


class TestTensorToScalarConversion(unittest.TestCase):
    """Test tensor comparison fixes."""

    def test_attention_no_tensor_comparison_error(self):
        """Test that attention computation doesn't fail on tensor comparison."""
        model = SimpleTestModel()
        metrics = ICLRMetricsFixes()
        batch = {'input_ids': torch.randint(0, 100, (2, 10))}

        # This should not raise an error about tensor in if statement
        try:
            result = metrics.compute_attention_attribution(model, batch)
            # Test passes if no exception is raised
            self.assertTrue(True, "No tensor comparison error")
        except RuntimeError as e:
            if "ambiguous" in str(e).lower():
                self.fail(f"Tensor comparison issue still present: {e}")
            else:
                raise


class TestIntegratedGradientsImprovements(unittest.TestCase):
    """Test IG improvements."""

    def setUp(self):
        self.model = SimpleTestModel()
        self.metrics = ICLRMetricsFixes()
        self.metrics._compute_gini = lambda x: 0.5
        self.batch = {'input_ids': torch.randint(0, 100, (2, 10))}

    def test_completeness_residual_computation(self):
        """Test completeness residual is computed."""
        target_spec = {
            'type': 'predicted',
            'compute_completeness': True
        }

        result = self.metrics.compute_integrated_gradients(
            self.model, self.batch,
            n_steps=10,
            target_spec=target_spec
        )

        self.assertIn('completeness_residual', result, "Should compute completeness residual")
        if result['completeness_residual'] is not None:
            self.assertIsInstance(result['completeness_residual'], float,
                                "Completeness residual should be a float")
            self.assertGreaterEqual(result['completeness_residual'], 0,
                                  "Completeness residual should be non-negative")

    def test_midpoint_rule_option(self):
        """Test midpoint rule can be selected."""
        result = self.metrics.compute_integrated_gradients(
            self.model, self.batch,
            n_steps=10,
            step_rule='midpoint'
        )

        self.assertEqual(result.get('step_rule'), 'midpoint', "Should use midpoint rule")

    def test_standard_rule_option(self):
        """Test standard rule can be selected."""
        result = self.metrics.compute_integrated_gradients(
            self.model, self.batch,
            n_steps=10,
            step_rule='standard'
        )

        self.assertEqual(result.get('step_rule'), 'standard', "Should use standard rule")

    def test_float32_accumulation_no_nan(self):
        """Test float32 accumulation prevents NaN values."""
        result = self.metrics.compute_integrated_gradients(
            self.model, self.batch,
            n_steps=10
        )

        if 'error' not in result:
            self.assertFalse(np.isnan(result['mean_attribution']),
                           "Should not have NaN values with float32 accumulation")

    def test_baseline_policy_zeros(self):
        """Test zero baseline policy."""
        result = self.metrics.compute_integrated_gradients(
            self.model, self.batch,
            baseline_policy='zeros',
            n_steps=5
        )

        self.assertEqual(result.get('baseline_policy'), 'zeros',
                        "Should use zeros baseline policy")

    def test_baseline_policy_pad(self):
        """Test PAD token baseline policy."""
        result = self.metrics.compute_integrated_gradients(
            self.model, self.batch,
            baseline_policy='pad',
            n_steps=5
        )

        self.assertEqual(result.get('baseline_policy'), 'pad',
                        "Should use pad baseline policy")

    def test_custom_target_spec(self):
        """Test custom target specification."""
        def custom_target(outputs, batch):
            return outputs.logits.mean()

        target_spec = {
            'type': 'custom',
            'fn': custom_target
        }

        result = self.metrics.compute_integrated_gradients(
            self.model, self.batch,
            n_steps=5,
            target_spec=target_spec
        )

        self.assertNotIn('error', result, "Should handle custom target")


class TestPruningScopeRestrictions(unittest.TestCase):
    """Test pruning scope filters."""

    def setUp(self):
        self.model = SimpleTestModel()
        self.metrics = ICLRMetricsFixes()
        self.metrics._compute_loss = lambda m, b, fn=None: 1.0
        self.batch = {'input_ids': torch.randint(0, 100, (2, 10))}

    def test_exclude_all_special_params(self):
        """Test excluding bias, norm, and embedding layers."""
        result = self.metrics.compute_pruning_sensitivity(
            self.model, self.batch,
            sparsity_levels=[0.5],
            include_bias=False,
            include_norm=False,
            include_emb=False
        )

        self.assertIn('n_prunable_params', result, "Should report prunable params count")
        self.assertEqual(result['n_prunable_params'], 1,
                        "Should only prune linear.weight")

    def test_include_all_params(self):
        """Test including all parameter types."""
        result = self.metrics.compute_pruning_sensitivity(
            self.model, self.batch,
            sparsity_levels=[0.5],
            include_bias=True,
            include_norm=True,
            include_emb=True
        )

        self.assertGreater(result['n_prunable_params'], 1,
                          "Should prune more params with inclusions")

    def test_pruning_scope_metadata(self):
        """Test that pruning scope is recorded."""
        result = self.metrics.compute_pruning_sensitivity(
            self.model, self.batch,
            sparsity_levels=[0.5],
            include_bias=False,
            include_norm=True,
            include_emb=False
        )

        self.assertIn('pruning_scope', result, "Should include pruning scope metadata")
        scope = result['pruning_scope']
        self.assertFalse(scope['include_bias'])
        self.assertTrue(scope['include_norm'])
        self.assertFalse(scope['include_emb'])


class TestAttentionRolloutNormalization(unittest.TestCase):
    """Test attention rollout normalization."""

    def test_rollout_entropy_valid_range(self):
        """Test rollout entropy is in valid range."""
        model = SimpleTestModel()
        metrics = ICLRMetricsFixes()
        batch = {'input_ids': torch.randint(0, 100, (2, 10))}

        result = metrics.compute_attention_attribution(model, batch, alpha=0.5)

        if 'error' not in result:
            entropy = result['rollout_entropy']
            seq_len = 10
            max_entropy = np.log(seq_len)

            self.assertGreaterEqual(entropy, 0,
                                  "Entropy should be non-negative")
            self.assertLessEqual(entropy, max_entropy + 0.1,
                               f"Entropy should be <= log({seq_len})")

    def test_alpha_parameter(self):
        """Test alpha parameter is respected."""
        model = SimpleTestModel()
        metrics = ICLRMetricsFixes()
        batch = {'input_ids': torch.randint(0, 100, (2, 10))}

        result = metrics.compute_attention_attribution(model, batch, alpha=0.7)

        self.assertEqual(result.get('alpha'), 0.7,
                        "Should use specified alpha value")

    def test_source_index_parameter(self):
        """Test source_index parameter is respected."""
        model = SimpleTestModel()
        metrics = ICLRMetricsFixes()
        batch = {'input_ids': torch.randint(0, 100, (2, 10))}

        result = metrics.compute_attention_attribution(model, batch, source_index=3)

        self.assertEqual(result.get('source_index'), 3,
                        "Should use specified source index")


class TestRobustnessToEdgeCases(unittest.TestCase):
    """Test robustness to edge cases."""

    def test_empty_batch(self):
        """Test handling of empty batch."""
        model = SimpleTestModel()
        metrics = ICLRMetricsFixes()
        metrics._compute_gini = lambda x: 0.5

        empty_batch = {'input_ids': torch.zeros(0, 0, dtype=torch.long)}

        result = metrics.compute_integrated_gradients(model, empty_batch)
        self.assertIn('error', result, "Should handle empty tensor gracefully")

    def test_out_of_vocab_tokens(self):
        """Test handling of out-of-vocabulary tokens."""
        model = SimpleTestModel()
        metrics = ICLRMetricsFixes()
        metrics._compute_gini = lambda x: 0.5

        # Create batch with out-of-range token IDs
        batch = {'input_ids': torch.tensor([[150, 200, 99, 98]])}  # vocab_size is 100

        result = metrics.compute_integrated_gradients(model, batch, n_steps=5)

        # Should clamp tokens and continue
        self.assertNotIn('error', result, "Should handle out-of-vocab tokens by clamping")

    def test_baseline_without_input_ids(self):
        """Test baseline handling when input_ids missing."""
        model = SimpleTestModel()
        metrics = ICLRMetricsFixes()
        metrics._compute_gini = lambda x: 0.5

        batch = {'input_ids': torch.randint(0, 100, (2, 10))}
        baseline = {'attention_mask': torch.ones(2, 10)}  # Missing input_ids

        result = metrics.compute_integrated_gradients(
            model, batch,
            baseline_batch=baseline,
            n_steps=5
        )

        # Should fall back to zero embeddings
        self.assertNotIn('error', result, "Should handle baseline without input_ids")


if __name__ == '__main__':
    unittest.main(verbosity=2)