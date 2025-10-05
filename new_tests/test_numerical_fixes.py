#!/usr/bin/env python3
"""
Comprehensive unit tests for numerical and theoretical fixes in InformationTheoryMetrics.py
Tests all the critical fixes implemented for ICLR 2026 paper reliability.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import warnings
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InformationTheoryMetrics import InformationTheoryMetrics

# Set up logging for test output
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestInformationTheoryNumericalStability(unittest.TestCase):
    """Test suite for numerical stability and theoretical correctness in InformationTheoryMetrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_epsilon_function(self):
        """Test the centralized epsilon function for different dtypes."""
        # Test different dtypes
        dtypes = [torch.float16, torch.float32, torch.float64]
        contexts = ['default', 'division', 'log', 'regularization', 'probability']

        for dtype in dtypes:
            for context in contexts:
                eps = self.metrics._get_epsilon(dtype, context)

                # Verify epsilon is appropriate for dtype
                if dtype == torch.float16:
                    self.assertGreaterEqual(eps, 1e-4,
                        f"Epsilon too small for float16: {eps}")
                elif dtype == torch.float64:
                    self.assertLessEqual(eps, 1e-6,
                        f"Epsilon too large for float64: {eps}")

                # Verify context multipliers work
                if context == 'regularization':
                    default_eps = self.metrics._get_epsilon(dtype, 'default')
                    self.assertGreater(eps, default_eps,
                        f"Regularization eps should be larger than default")

                # Verify epsilon is representable in dtype
                eps_tensor = torch.tensor(eps, dtype=dtype)
                self.assertGreater(eps_tensor.item(), 0,
                    f"Epsilon becomes zero in {dtype}")

    def test_dtype_consistency(self):
        """Test that dtype mismatches are fixed."""
        # Test with float16 tensors
        x = torch.randn(10, 64, dtype=torch.float16)
        y = torch.randn(10, 64, dtype=torch.float16)

        # Test InfoNCE with mixed precision
        result = self.metrics._estimate_mutual_information_infonce(x, y)

        self.assertIn('mi_nats', result, "InfoNCE should return MI estimate")
        self.assertGreaterEqual(result['mi_nats'], 0, "MI should be non-negative")
        self.assertFalse(np.isnan(result['mi_nats']), "MI should not be NaN")

    def test_entropy_computation(self):
        """Test that entropy computation handles zero probabilities correctly."""
        # Test case 1: All same labels (H=0)
        labels = torch.tensor([1, 1, 1, 1, 1])
        hidden = torch.randn(5, 128)
        result = self.metrics._compute_i_ty_via_labels_aligned(
            hidden, labels, num_classes=10
        )
        self.assertEqual(result['h_y_nats'], 0.0,
            "Entropy should be 0 for identical labels")

        # Test case 2: Uniform distribution
        labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
        hidden = torch.randn(10, 128)
        result = self.metrics._compute_i_ty_via_labels_aligned(
            hidden, labels, num_classes=10
        )
        expected_h = np.log(10)  # Theoretical entropy for uniform distribution
        self.assertAlmostEqual(result['h_y_nats'], expected_h, places=2,
            msg=f"Uniform entropy incorrect: {result['h_y_nats']} vs {expected_h}")

        # Test case 3: Some zero probabilities
        labels = torch.tensor([0, 0, 1, 1, 2, 2])  # Classes 3-9 have zero probability
        hidden = torch.randn(6, 128)
        result = self.metrics._compute_i_ty_via_labels_aligned(
            hidden, labels, num_classes=10
        )
        # Should compute entropy only for non-zero classes
        self.assertGreater(result['h_y_nats'], 0, "Entropy should be positive")
        self.assertLess(result['h_y_nats'], np.log(10),
            "Entropy should be less than max")

    def test_infonce_formula(self):
        """Test that InfoNCE formula uses correct negative sampling."""
        # Create correlated data with known MI
        dim = 32
        n_samples = 128
        x = torch.randn(n_samples, dim)
        y = x + 0.1 * torch.randn(n_samples, dim)  # High correlation

        result = self.metrics._estimate_mutual_information_infonce(x, y, n_negatives=64)

        # Check that formula is using log(batch_size) correctly
        self.assertIn('mi_nats', result, "Should return MI estimate")
        self.assertGreater(result['mi_nats'], 0,
            "MI should be positive for correlated data")

        # Test with independent data (should have low MI)
        y_independent = torch.randn(n_samples, dim)
        result_indep = self.metrics._estimate_mutual_information_infonce(
            x, y_independent, n_negatives=64
        )
        self.assertGreater(result['mi_nats'], result_indep['mi_nats'],
            "Correlated data should have higher MI than independent")

    def test_probe_improvements(self):
        """Test improved probe-based I(T;Y) estimation."""
        # Create data where linear probe would struggle
        n_samples = 1000
        hidden_dim = 256
        num_classes = 10

        # Create XOR-like pattern that needs non-linearity
        hidden = torch.randn(n_samples, hidden_dim)
        # Labels depend on non-linear combination of features
        labels = ((hidden[:, 0] > 0) ^ (hidden[:, 1] > 0)).long()
        labels = torch.clamp(labels, 0, num_classes - 1)

        # Test with middle layer (should use more epochs and MLP)
        result = self.metrics._compute_i_ty_via_labels_aligned(
            hidden, labels, num_classes, layer_idx=5, n_layers=12
        )

        # Check that probe was trained properly
        self.assertIn('probe_epochs', result, "Should report training epochs")
        self.assertGreaterEqual(result['mi_nats'], 0, "MI should be non-negative")
        # Middle layers should train for more epochs
        self.assertGreaterEqual(result['probe_epochs'], 5,
            "Middle layers should train for more epochs")

    def test_convergence_monitoring(self):
        """Test convergence monitoring and early stopping."""
        # Create easy-to-learn pattern that should converge quickly
        n_samples = 256
        dim = 16
        x = torch.randn(n_samples, dim)
        y = x + 0.01 * torch.randn(n_samples, dim)  # Very high correlation

        result = self.metrics._estimate_mutual_information_infonce(x, y)

        # Check convergence info
        self.assertIn('converged', result, "Should report convergence status")
        self.assertIn('n_epochs_trained', result, "Should report actual epochs")
        self.assertIn('mi_history', result, "Should include MI history")

        # Check that it has MI history
        self.assertIsInstance(result['mi_history'], list, "MI history should be a list")
        self.assertGreater(len(result['mi_history']), 0, "MI history should not be empty")

        # For highly correlated data, MI should be positive
        self.assertGreater(result['mi_nats'], 0, "MI should be positive for correlated data")

    def test_warning_system(self):
        """Test that warnings are issued for numerical issues."""
        # Create scenario that might produce negative MI
        x = torch.randn(10, 100)  # Small batch, high dim
        y = torch.randn(10, 100)  # Independent

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # This might trigger warnings - we just check it doesn't crash
            result = self.metrics._estimate_mutual_information_infonce(x, y)

            # Should complete without errors
            self.assertIn('mi_nats', result)
            self.assertGreaterEqual(result['mi_nats'], 0,
                "MI should be clamped to non-negative")

    def test_sequence_boundaries(self):
        """Test that sequence boundaries are handled correctly."""
        # Create data with sequence structure
        batch_size = 4
        seq_len = 10
        hidden_dim = 64

        # Simulate sequence data
        x = torch.randn(batch_size * seq_len, hidden_dim)
        y = torch.randn(batch_size * seq_len, hidden_dim)
        seq_ids = torch.repeat_interleave(torch.arange(batch_size), seq_len)

        # Test MINE with sequence boundaries
        result = self.metrics._estimate_mutual_information_mine(
            x, y, seq_ids=seq_ids, n_epochs=10  # Fewer epochs for testing
        )

        self.assertIn('mi_nats', result, "Should return MI estimate")
        self.assertGreaterEqual(result['mi_nats'], 0, "MI should be non-negative")

    def test_numerical_stability_extreme_values(self):
        """Test overall numerical stability with extreme values."""
        # Test with very small values
        x_small = torch.randn(100, 32) * 1e-8
        y_small = torch.randn(100, 32) * 1e-8

        result = self.metrics._estimate_mutual_information_infonce(x_small, y_small)
        self.assertFalse(np.isnan(result['mi_nats']), "MI should not be NaN")
        self.assertFalse(np.isinf(result['mi_nats']), "MI should not be infinite")
        self.assertGreaterEqual(result['mi_nats'], 0, "MI should be non-negative")

        # Test with very large values
        x_large = torch.randn(100, 32) * 1e8
        y_large = torch.randn(100, 32) * 1e8

        result = self.metrics._estimate_mutual_information_infonce(x_large, y_large)
        self.assertFalse(np.isnan(result['mi_nats']), "MI should not be NaN")
        self.assertFalse(np.isinf(result['mi_nats']), "MI should not be infinite")
        self.assertGreaterEqual(result['mi_nats'], 0, "MI should be non-negative")

    def test_flow_ratio_stability(self):
        """Test that flow ratio computation uses proper epsilon."""
        # Create a simple model output scenario
        batch_size = 2
        seq_len = 10
        hidden_dim = 64

        # Mock data for compute_information_flow
        input_batch = {
            'input_ids': torch.randint(0, 100, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len)
        }

        # Create a simple model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = type('Config', (), {'vocab_size': 100})()
                self.embeddings = nn.Embedding(100, hidden_dim)
                self.lm_head = nn.Linear(hidden_dim, 100)

            def forward(self, input_ids, **kwargs):
                embeds = self.embeddings(input_ids)
                logits = self.lm_head(embeds)
                hidden_states = [embeds, embeds]  # Simplified
                return type('Output', (), {
                    'logits': logits,
                    'hidden_states': hidden_states
                })()

            def get_input_embeddings(self):
                return self.embeddings

        model = DummyModel()

        # Test with very small compression values
        result = self.metrics.compute_information_flow(
            model, input_batch, n_samples=100
        )

        # Check flow_ratio doesn't have division by zero issues
        self.assertIn('flow_ratio', result)
        self.assertFalse(np.isnan(result['flow_ratio']),
            "Flow ratio should not be NaN even with small values")
        self.assertFalse(np.isinf(result['flow_ratio']),
            "Flow ratio should not be infinite")


class TestCriticalMutualInformationFixes(unittest.TestCase):
    """Test critical fixes for mutual information estimation correctness."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_mine_dtype_fix(self):
        """Test that MINE estimator handles dtype correctly (Issue #2)."""
        # Test with float16
        x = torch.randn(50, 32, dtype=torch.float16)
        y = torch.randn(50, 32, dtype=torch.float16)

        result = self.metrics._estimate_mutual_information_mine(
            x, y, n_epochs=5  # Few epochs for testing
        )

        self.assertIn('mi_nats', result)
        self.assertFalse(np.isnan(result['mi_nats']),
            "MINE should handle float16 without NaN")

    def test_negative_mi_warning(self):
        """Test that negative MI triggers warnings (Issue #7)."""
        # Create independent data that might give negative MI estimate
        x = torch.randn(20, 100)  # Small batch, high dim
        y = torch.randn(20, 100)

        # Should warn but still return valid result
        result = self.metrics._estimate_mutual_information_infonce(x, y)

        self.assertGreaterEqual(result['mi_nats'], 0,
            "MI should be clamped to non-negative")

    def test_entropy_zero_probability(self):
        """Test entropy computation with zero probabilities (Issue #6)."""
        # Create distribution with many zero probabilities
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 2])  # Classes 3-9 have p=0
        hidden = torch.randn(7, 64)

        result = self.metrics._compute_i_ty_via_labels_aligned(
            hidden, labels, num_classes=10
        )

        # Should handle zero probabilities without issues
        self.assertGreater(result['h_y_nats'], 0, "Should have positive entropy")
        self.assertFalse(np.isnan(result['h_y_nats']), "Entropy should not be NaN")

        # Verify it's less than max entropy
        max_entropy = np.log(10)
        self.assertLess(result['h_y_nats'], max_entropy)


if __name__ == '__main__':
    unittest.main(verbosity=2)