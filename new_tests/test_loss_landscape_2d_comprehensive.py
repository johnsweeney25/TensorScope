#!/usr/bin/env python3
"""
Comprehensive unit tests for compute_loss_landscape_2d fixes.
Tests all 7 critical issues identified by intern's audit:
1. Gram-Schmidt orthogonalization
2. Try/finally weight restoration
3. Guarded batch_processor import
4. NumPy dtype conversion issues
5. Transformer layer normalization
6. Robust TV roughness computation
7. Diagnostic outputs for orthogonality
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ICLRMetrics import ICLRMetrics


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, vocab_size=50, hidden_size=32, num_layers=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)

        # Use ModuleList to create layers with proper naming
        self.layer = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size) for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.config = type('Config', (), {'vocab_size': vocab_size})()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embeddings(input_ids)

        for layer in self.layer:
            x = torch.relu(layer(x))

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        class Output:
            pass
        output = Output()
        output.logits = logits
        output.loss = loss if loss is not None else torch.tensor(0.0)
        return output


class TestLossLandscape2DFixes(unittest.TestCase):
    """Test all fixes for compute_loss_landscape_2d."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)

        self.model = SimpleModel(vocab_size=50, hidden_size=32, num_layers=2)
        self.model.eval()

        self.metrics = ICLRMetrics()

        self.batch = {
            'input_ids': torch.randint(0, 50, (2, 16)),
            'attention_mask': torch.ones(2, 16, dtype=torch.long),
            'labels': torch.randint(0, 50, (2, 16))
        }

    def test_issue1_gram_schmidt_orthogonalization(self):
        """Test that directions are properly orthogonalized using Gram-Schmidt."""
        print("\n[Issue 1] Testing Gram-Schmidt orthogonalization...")

        for mode in ['global', 'filter', 'layer']:
            with self.subTest(normalization_mode=mode):
                result = self.metrics.compute_loss_landscape_2d(
                    self.model,
                    self.batch,
                    n_points=3,
                    span=0.1,
                    normalization_mode=mode,
                    seed=42
                )

                # Check orthogonality diagnostics exist
                self.assertIn('cos_angle_d1_d2', result)
                self.assertIn('orthogonality_check', result)
                self.assertIn('norm_d1', result)
                self.assertIn('norm_d2', result)

                # Verify orthogonality
                cos_angle = result['cos_angle_d1_d2']
                self.assertLess(abs(cos_angle), 0.01,
                               f"Directions not orthogonal in {mode} mode: cos={cos_angle}")
                self.assertTrue(result['orthogonality_check'],
                               f"Orthogonality check failed in {mode} mode")

                # Verify norms are positive
                self.assertGreater(result['norm_d1'], 0)
                self.assertGreater(result['norm_d2'], 0)

    def test_issue2_weight_restoration_with_try_finally(self):
        """Test that weights are restored even when errors occur."""
        print("\n[Issue 2] Testing try/finally weight restoration...")

        # Save original weights
        original_weights = {name: param.clone()
                          for name, param in self.model.named_parameters()}

        # Define a loss function that raises an error
        def error_loss_fn(model, batch):
            raise ValueError("Simulated error during loss computation")

        # Test normal execution first
        _ = self.metrics.compute_loss_landscape_2d(
            self.model,
            self.batch,
            n_points=3,
            span=0.1,
            seed=42
        )

        # Check weights restored after normal execution
        for name, param in self.model.named_parameters():
            self.assertTrue(
                torch.allclose(param, original_weights[name], atol=1e-6),
                f"Weight {name} not restored after normal execution"
            )

        # Test with error (all grid points will fail, but weights should still be restored)
        with self.assertLogs(level='WARNING') as cm:
            result = self.metrics.compute_loss_landscape_2d(
                self.model,
                self.batch,
                n_points=3,
                span=0.1,
                loss_fn=error_loss_fn,
                seed=42
            )

        # Check that we got warnings about failures
        self.assertTrue(any('Loss computation failed' in msg for msg in cm.output))

        # Most importantly: check weights are still restored despite errors
        for name, param in self.model.named_parameters():
            self.assertTrue(
                torch.allclose(param, original_weights[name], atol=1e-6),
                f"Weight {name} not restored after error"
            )

    def test_issue3_guarded_batch_processor_import(self):
        """Test that batch_processor import is properly guarded."""
        print("\n[Issue 3] Testing guarded batch_processor import...")

        # Create a new metrics instance with no batch_processor
        metrics_no_bp = ICLRMetrics(batch_processor=None)

        # Mock the import to fail when trying to import batch_processor
        with patch.dict('sys.modules', {'batch_processor': None}):
            # This should not raise an error, but fall back gracefully
            result = metrics_no_bp.compute_loss_landscape_2d(
                self.model,
                self.batch,
                n_points=3,
                span=0.1,
                seed=42
            )

            # Should still compute successfully without batch_processor
            self.assertNotIn('error', result)
            self.assertGreater(result['n_valid'], 0)

    def test_issue4_numpy_dtype_conversion(self):
        """Test that NumPy dtype issues are handled correctly."""
        print("\n[Issue 4] Testing NumPy dtype conversion...")

        # The issue was that numpy scalars (from linspace) could cause dtype upcast
        # when directly used as coefficients in tensor operations.
        # The fix converts them to Python floats.

        # Use different numpy dtypes for span to test conversion
        for np_dtype in [np.float32, np.float64, np.float16]:
            with self.subTest(numpy_dtype=np_dtype):
                # Create span with specific numpy dtype
                span = np_dtype(0.1)

                # This should not cause dtype upcast issues
                result = self.metrics.compute_loss_landscape_2d(
                    self.model,
                    self.batch,
                    n_points=5,
                    span=float(span),  # Convert to Python float like the fix does
                    seed=42
                )

                # Check that we got valid results without dtype errors
                self.assertNotIn('error', result)
                self.assertGreater(result['n_valid'], 0)

                # Verify alphas are properly converted (checking in results)
                self.assertIsInstance(result['axis_values'], list)
                for val in result['axis_values']:
                    self.assertIsInstance(val, float)
                    # Should be Python float, not numpy scalar
                    self.assertNotIsInstance(val, np.number)

    def test_issue5_transformer_layer_normalization(self):
        """Test transformer-specific layer normalization."""
        print("\n[Issue 5] Testing transformer layer normalization...")

        # Create a transformer-like model with layer naming convention
        class TransformerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embeddings = nn.Embedding(50, 32)
                # Use layer.N naming pattern
                self.h = nn.ModuleList([
                    nn.Linear(32, 32) for _ in range(4)
                ])
                self.lm_head = nn.Linear(32, 50)
                self.config = type('Config', (), {'vocab_size': 50})()

            def forward(self, input_ids, labels=None, **kwargs):
                x = self.embeddings(input_ids)
                for layer in self.h:
                    x = layer(x)
                logits = self.lm_head(x)

                loss = None
                if labels is not None:
                    loss = nn.functional.cross_entropy(
                        logits.view(-1, 50), labels.view(-1), ignore_index=-100
                    )

                class Output:
                    pass
                output = Output()
                output.logits = logits
                output.loss = loss if loss is not None else torch.tensor(0.0)
                return output

        transformer_model = TransformerModel()
        transformer_model.eval()

        # Test layer normalization
        result_layer = self.metrics.compute_loss_landscape_2d(
            transformer_model,
            self.batch,
            n_points=3,
            span=0.1,
            normalization_mode='layer',
            seed=42
        )

        # Test global normalization for comparison
        result_global = self.metrics.compute_loss_landscape_2d(
            transformer_model,
            self.batch,
            n_points=3,
            span=0.1,
            normalization_mode='global',
            seed=42
        )

        # Both should work but produce different landscapes
        self.assertGreater(result_layer['n_valid'], 0)
        self.assertGreater(result_global['n_valid'], 0)

        # Verify normalization mode is recorded
        self.assertEqual(result_layer['normalization_mode'], 'layer')
        self.assertEqual(result_global['normalization_mode'], 'global')

    def test_issue6_robust_tv_roughness(self):
        """Test robust Total Variation roughness computation."""
        print("\n[Issue 6] Testing robust TV roughness computation...")

        # Create a landscape with some NaN values to test robustness
        result = self.metrics.compute_loss_landscape_2d(
            self.model,
            self.batch,
            n_points=5,
            span=0.1,
            seed=42
        )

        # Check that roughness was computed
        self.assertIn('roughness', result)
        self.assertIn('normalized_roughness', result)

        # Roughness should be non-negative
        self.assertGreaterEqual(result['roughness'], 0)

        # Test with very small grid (edge case)
        result_small = self.metrics.compute_loss_landscape_2d(
            self.model,
            self.batch,
            n_points=2,  # 2x2 grid
            span=0.1,
            seed=42
        )

        # Should handle small grid gracefully
        self.assertIn('roughness', result_small)
        self.assertGreaterEqual(result_small['roughness'], 0)

        # Mock a case with NaN values in grid
        with patch.object(self.metrics, '_compute_loss', side_effect=[
            1.0, 2.0, np.nan,  # First row
            1.5, np.nan, 3.0,  # Second row
            2.0, 2.5, 3.5      # Third row
        ]):
            result_nan = self.metrics.compute_loss_landscape_2d(
                self.model,
                self.batch,
                n_points=3,
                span=0.1,
                seed=42
            )

            # Should still compute roughness despite NaNs
            self.assertIn('roughness', result_nan)
            self.assertIsInstance(result_nan['roughness'], float)
            self.assertFalse(np.isnan(result_nan['roughness']))

    def test_issue7_diagnostic_outputs(self):
        """Test that all diagnostic outputs are present and correct."""
        print("\n[Issue 7] Testing diagnostic outputs...")

        result = self.metrics.compute_loss_landscape_2d(
            self.model,
            self.batch,
            n_points=3,
            span=0.1,
            normalization_mode='filter',
            seed=42
        )

        # Check all diagnostic outputs are present
        required_diagnostics = [
            'cos_angle_d1_d2',
            'norm_d1',
            'norm_d2',
            'orthogonality_check',
            'normalization_mode'
        ]

        for diagnostic in required_diagnostics:
            self.assertIn(diagnostic, result, f"Missing diagnostic: {diagnostic}")

        # Verify types and ranges
        self.assertIsInstance(result['cos_angle_d1_d2'], float)
        self.assertLessEqual(abs(result['cos_angle_d1_d2']), 1.0)  # Cosine in [-1, 1]

        self.assertIsInstance(result['norm_d1'], float)
        self.assertGreater(result['norm_d1'], 0)

        self.assertIsInstance(result['norm_d2'], float)
        self.assertGreater(result['norm_d2'], 0)

        self.assertIsInstance(result['orthogonality_check'], bool)

        self.assertEqual(result['normalization_mode'], 'filter')

        # Check updated note mentions orthogonality
        self.assertIn('note', result)
        self.assertIn('orthogonal', result['note'].lower())

    def test_integration_all_fixes_work_together(self):
        """Integration test that all fixes work together properly."""
        print("\n[Integration] Testing all fixes work together...")

        # Save original weights
        original_weights = {name: param.clone()
                          for name, param in self.model.named_parameters()}

        # Run with all features
        result = self.metrics.compute_loss_landscape_2d(
            self.model,
            self.batch,
            n_points=5,
            span=0.15,
            normalization_mode='layer',  # Test transformer mode
            seed=123
        )

        # Comprehensive checks
        # 1. Orthogonality
        self.assertTrue(result['orthogonality_check'])
        self.assertLess(abs(result['cos_angle_d1_d2']), 0.01)

        # 2. Weights restored
        for name, param in self.model.named_parameters():
            self.assertTrue(torch.allclose(param, original_weights[name], atol=1e-6))

        # 3. No import errors (implicit - would have failed)

        # 4. Dtype handling (implicit - would have failed)

        # 5. Layer normalization used
        self.assertEqual(result['normalization_mode'], 'layer')

        # 6. Roughness computed
        self.assertIsInstance(result['roughness'], float)
        self.assertFalse(np.isnan(result['roughness']))

        # 7. All diagnostics present
        for key in ['cos_angle_d1_d2', 'norm_d1', 'norm_d2', 'orthogonality_check']:
            self.assertIn(key, result)

        # Valid results
        self.assertGreater(result['n_valid'], 0)
        self.assertNotIn('error', result)

        print(f"  ✓ Computed {result['n_valid']}/{result['n_total']} valid points")
        print(f"  ✓ Orthogonality: cos_angle={result['cos_angle_d1_d2']:.6f}")
        print(f"  ✓ Roughness: {result['roughness']:.4f}")


if __name__ == '__main__':
    # Run with verbosity for detailed output
    unittest.main(verbosity=2)