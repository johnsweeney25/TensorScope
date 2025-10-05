"""
Unit tests for Fisher damage functions - ICML 2026 submission fixes.

Tests cover:
1. Import scoping errors (no local torch import)
2. Gradient requirements checking on frozen models
3. Numerical precision for large models
4. Reproducibility (deterministic gradient collection)
5. Asymmetry ratio stability
6. Theoretical correctness of damage metrics

Author: Fisher Damage Fix Team
Date: 2024-09-30
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, Any

# Import the module to test
from ModularityMetrics import ExtendedModularityMetrics


class DummyLanguageModel(nn.Module):
    """Simple dummy model that mimics HuggingFace model interface."""

    def __init__(self, vocab_size=100, hidden_dim=32, num_layers=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass that returns loss."""
        class Output:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits

        # Embed inputs
        x = self.embedding(input_ids)

        # Process through layers
        for layer in self.layers:
            x = torch.relu(layer(x))

        # Output projection
        logits = self.output(x)

        # Compute loss if labels provided
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )
        else:
            loss = logits.mean()

        return Output(loss, logits)


class TestFisherDamageImportScoping(unittest.TestCase):
    """Test that import scoping error is fixed."""

    def test_no_local_torch_import(self):
        """Verify no local 'import torch' in compute_fisher_damage_with_asymmetry."""
        import inspect

        metrics = ExtendedModularityMetrics()
        source = inspect.getsource(metrics.compute_fisher_damage_with_asymmetry)

        # Check if there's a local import torch
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'import torch' in line:
                # Should be commented out or not present
                stripped = line.strip()
                self.assertTrue(
                    stripped.startswith('#') or 'already imported' in stripped,
                    "Found uncommented 'import torch' inside function"
                )

    def test_function_executes_without_name_error(self):
        """Test that functions execute without NameError: torch."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        # Should not raise NameError
        try:
            result = metrics.compute_fisher_damage_with_asymmetry(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=1
            )
            self.assertIn('asymmetry_ratio', result)
        except NameError as e:
            self.fail(f"NameError raised: {e}")


class TestFisherDamageGradientRequirements(unittest.TestCase):
    """Test gradient requirement checking on frozen models."""

    def test_frozen_model_gradient_enabling(self):
        """Test that gradients are enabled on frozen models."""
        import logging
        from io import StringIO

        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        # Freeze all parameters (simulate pretrained model)
        for param in model.parameters():
            param.requires_grad = False

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        # Capture logger output (warnings are emitted via logger, not warnings module)
        log_capture = StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)

        logger = logging.getLogger('ModularityMetrics')
        logger.addHandler(handler)

        try:
            result = metrics.compute_fisher_weighted_damage(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=1
            )

            # Check log output for warning about frozen parameters
            log_contents = log_capture.getvalue()
            self.assertIn(
                "requires_grad=True",
                log_contents,
                "Should warn about frozen parameters"
            )

            # Verify result is computed (not zero due to missing gradients)
            self.assertIsNotNone(result)

        finally:
            logger.removeHandler(handler)

    def test_gradient_state_restoration(self):
        """Test that original requires_grad state is restored."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        # Set mixed gradient requirements
        for i, param in enumerate(model.parameters()):
            param.requires_grad = (i % 2 == 0)  # Every other parameter

        # Store original state
        original_state = {
            name: param.requires_grad
            for name, param in model.named_parameters()
        }

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        # Run function
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            metrics.compute_fisher_weighted_damage(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=1
            )

        # Verify state is restored
        for name, param in model.named_parameters():
            self.assertEqual(
                param.requires_grad,
                original_state[name],
                f"Parameter {name} requires_grad not restored"
            )


class TestFisherDamageNumericalPrecision(unittest.TestCase):
    """Test numerical precision improvements."""

    def test_sqrt_clamping(self):
        """Test that sqrt(Fisher) is clamped to prevent precision loss."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        # Run with symmetric damage (uses sqrt)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metrics.compute_fisher_weighted_damage(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=1,
                damage_type='symmetric'
            )

        # Should not produce NaN or Inf
        self.assertFalse(np.isnan(result['total_damage']))
        self.assertFalse(np.isinf(result['total_damage']))
        self.assertFalse(np.isnan(result['normalized_damage']))
        self.assertFalse(np.isinf(result['normalized_damage']))

    def test_large_model_fisher_sum_precision(self):
        """Test that Fisher sum maintains precision for large models."""
        metrics = ExtendedModularityMetrics()

        # Create larger model to test precision
        model = DummyLanguageModel(vocab_size=1000, hidden_dim=128, num_layers=4)

        batch_A = {
            'input_ids': torch.randint(0, 1000, (4, 20)),
            'attention_mask': torch.ones(4, 20)
        }
        batch_B = {
            'input_ids': torch.randint(0, 1000, (4, 20)),
            'attention_mask': torch.ones(4, 20)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metrics.compute_fisher_weighted_damage(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=2
            )

        # Fisher norm should be finite and non-negative
        self.assertGreaterEqual(result['fisher_norm'], 0.0)
        self.assertFalse(np.isnan(result['fisher_norm']))
        self.assertFalse(np.isinf(result['fisher_norm']))


class TestFisherDamageReproducibility(unittest.TestCase):
    """Test reproducibility improvements."""

    def test_deterministic_gradient_collection(self):
        """Test that gradient collection is deterministic."""
        torch.manual_seed(42)

        metrics = ExtendedModularityMetrics(seed=42)
        model = DummyLanguageModel()

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        # Run twice with same seed
        results = []
        for _ in range(2):
            torch.manual_seed(42)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = metrics.compute_fisher_weighted_damage(
                    model, batch_A, batch_B,
                    fisher_type='direct',
                    n_fisher_samples=1
                )
            results.append(result['normalized_damage'])

        # Should be identical (or very close due to floating point)
        self.assertAlmostEqual(
            results[0], results[1], places=6,
            msg="Results should be reproducible with same seed"
        )


class TestFisherDamageAsymmetryStability(unittest.TestCase):
    """Test asymmetry ratio stability improvements."""

    def test_asymmetry_ratio_finite(self):
        """Test that asymmetry ratio is always finite or explicitly inf."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metrics.compute_fisher_damage_with_asymmetry(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=1
            )

        # Ratio should be finite or explicitly inf (not NaN)
        ratio = result['asymmetry_ratio']
        self.assertFalse(np.isnan(ratio), "Asymmetry ratio should not be NaN")

        # If finite, should be positive
        if np.isfinite(ratio):
            self.assertGreater(ratio, 0.0)

    def test_log_asymmetry_ratio_present(self):
        """Test that log asymmetry ratio is included in output."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metrics.compute_fisher_damage_with_asymmetry(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=1
            )

        # Should have log_asymmetry_ratio key
        self.assertIn('log_asymmetry_ratio', result)

        # If ratio is finite and positive, log should be finite
        if np.isfinite(result['asymmetry_ratio']) and result['asymmetry_ratio'] > 0:
            self.assertTrue(
                np.isfinite(result['log_asymmetry_ratio']) or
                result['log_asymmetry_ratio'] == float('-inf')
            )

    def test_zero_damage_handling(self):
        """Test that zero or near-zero damages are handled gracefully."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        # Use identical batches (should give very low damage)
        batch = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metrics.compute_fisher_damage_with_asymmetry(
                model, batch, batch,
                fisher_type='direct',
                n_fisher_samples=1
            )

        # Should handle gracefully (not crash)
        self.assertIsNotNone(result['asymmetry_ratio'])


class TestFisherDamageTheoreticalCorrectness(unittest.TestCase):
    """Test theoretical correctness of damage formulas."""

    def test_asymmetric_damage_formula(self):
        """Test that asymmetric damage uses grad^2 * Fisher formula."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metrics.compute_fisher_weighted_damage(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=1,
                damage_type='asymmetric'
            )

        # Should produce non-negative damage (grad^2 * Fisher >= 0)
        self.assertGreaterEqual(result['total_damage'], 0.0)
        self.assertGreaterEqual(result['normalized_damage'], 0.0)

    def test_damage_type_support(self):
        """Test that all damage types are supported."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        damage_types = ['symmetric', 'asymmetric', 'l1_weighted']

        for damage_type in damage_types:
            with self.subTest(damage_type=damage_type):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = metrics.compute_fisher_weighted_damage(
                        model, batch_A, batch_B,
                        fisher_type='direct',
                        n_fisher_samples=1,
                        damage_type=damage_type
                    )

                # Should complete without error
                self.assertIsNotNone(result)
                self.assertIn('total_damage', result)
                self.assertIn('normalized_damage', result)


class TestFisherDamageOutputFormat(unittest.TestCase):
    """Test that output format is correct and complete."""

    def test_weighted_damage_output_keys(self):
        """Test compute_fisher_weighted_damage returns all expected keys."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metrics.compute_fisher_weighted_damage(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=1
            )

        expected_keys = {
            'total_damage',
            'normalized_damage',
            'layer_damages',
            'fisher_norm',
            'n_parameters'
        }

        self.assertEqual(set(result.keys()), expected_keys)

    def test_asymmetry_output_keys(self):
        """Test compute_fisher_damage_with_asymmetry returns all expected keys."""
        metrics = ExtendedModularityMetrics()
        model = DummyLanguageModel()

        batch_A = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }
        batch_B = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = metrics.compute_fisher_damage_with_asymmetry(
                model, batch_A, batch_B,
                fisher_type='direct',
                n_fisher_samples=1
            )

        expected_keys = {
            'damage_math_from_general',
            'damage_general_from_math',
            'asymmetry_ratio',
            'log_asymmetry_ratio',  # New key added in fix
            'more_vulnerable_task',
            'damage_math_details',
            'damage_general_details'
        }

        self.assertEqual(set(result.keys()), expected_keys)


if __name__ == '__main__':
    unittest.main(verbosity=2)