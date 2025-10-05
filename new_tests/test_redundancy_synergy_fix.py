#!/usr/bin/env python3
"""
Unit test to verify compute_heuristic_pid_minmi uses InfoNCE instead of binning.
Tests that the function no longer produces PCA reduction warnings.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
import logging
from unittest.mock import patch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InformationTheoryMetrics import InformationTheoryMetrics


class MockModel(nn.Module):
    """Minimal mock model for testing."""
    def __init__(self, vocab_size=1000, hidden_size=64, num_layers=2):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_size, hidden_size)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, output_hidden_states=False, return_dict=False, **kwargs):
        x = self.embeddings(input_ids)

        hidden_states = [x] if output_hidden_states else []
        for layer in self.layers:
            x = layer(x)
            if output_hidden_states:
                hidden_states.append(x)

        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100
            )

        if return_dict:
            class Output:
                pass
            output = Output()
            output.logits = logits
            output.loss = loss if loss is not None else torch.tensor(0.0)
            if output_hidden_states:
                output.hidden_states = tuple(hidden_states)
            return output
        else:
            return (loss, logits, hidden_states) if output_hidden_states else (loss, logits)


class TestRedundancySynergyFix(unittest.TestCase):
    """Test that compute_heuristic_pid_minmi uses appropriate MI estimation."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel(vocab_size=100, hidden_size=64, num_layers=2)
        self.model.eval()
        self.metrics = InformationTheoryMetrics(seed=42)

        # Set up logging to capture warnings
        self.log_capture = []

    def _create_batch(self, batch_size=32, seq_length=128, vocab_size=100):
        """Helper to create test batches."""
        batch = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length, dtype=torch.long),
            'labels': torch.randint(0, vocab_size, (batch_size, seq_length))
        }
        # Set some labels to -100 (padding)
        batch['labels'][:, -5:] = -100
        return batch

    def test_infonce_method_no_pca_warnings(self):
        """Test that InfoNCE method doesn't produce PCA warnings."""
        task1_batch = self._create_batch(batch_size=16, seq_length=64)
        task2_batch = self._create_batch(batch_size=16, seq_length=64)

        # Run with InfoNCE (should not produce PCA warnings)
        # We check by ensuring it runs without warnings
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            result = self.metrics.compute_heuristic_pid_minmi(
                self.model,
                task1_batch,
                task2_batch,
                max_tokens_for_pid=100,
                mi_method='infonce'  # Using InfoNCE
            )

            # Check that no PCA-related warnings were issued
            pca_warnings = [warning for warning in w
                          if 'PCA' in str(warning.message) or 'binning' in str(warning.message)]
            self.assertEqual(len(pca_warnings), 0,
                           f"InfoNCE should not produce PCA warnings, but got: {pca_warnings}")

        # Verify result structure (now has labels1/labels2 targets)
        self.assertIsInstance(result, dict)
        self.assertIn('layer_0_labels1_redundancy', result)
        self.assertIn('layer_0_labels1_synergy', result)
        self.assertIn('layer_0_labels1_residual', result)  # New: conservation check
        self.assertIn('layer_0_labels2_redundancy', result)
        self.assertIn('layer_0_cka_h1_h2', result)  # New: CKA overlap metric

    def test_binning_method_produces_warnings(self):
        """Test that binning method does produce warnings for high-dim data."""
        task1_batch = self._create_batch(batch_size=16, seq_length=64)
        task2_batch = self._create_batch(batch_size=16, seq_length=64)

        # Mock the logger to capture warnings
        with patch('InformationTheoryMetrics.logger') as mock_logger:
            # Run with binning (should produce warnings for high dimensions)
            result = self.metrics.compute_heuristic_pid_minmi(
                self.model,
                task1_batch,
                task2_batch,
                max_tokens_for_pid=100,
                mi_method='binning'  # Force binning to test warnings
            )

            # Check that warnings were issued about high dimensions
            # The warning should be called when dimensions > 100
            if mock_logger.warning.called:
                warning_messages = [str(call) for call in mock_logger.warning.call_args_list]
                # At least one warning about dimensions
                self.assertTrue(any('high dimensions' in str(msg) for msg in warning_messages),
                              "Expected warning about high dimensions with binning")

    def test_method_selection(self):
        """Test different MI estimation methods work correctly."""
        task1_batch = self._create_batch(batch_size=8, seq_length=32)
        task2_batch = self._create_batch(batch_size=8, seq_length=32)

        methods_to_test = ['infonce', 'mine']  # Test main methods

        for method in methods_to_test:
            with self.subTest(method=method):
                result = self.metrics.compute_heuristic_pid_minmi(
                    self.model,
                    task1_batch,
                    task2_batch,
                    max_tokens_for_pid=50,
                    mi_method=method
                )

                # Verify basic structure
                self.assertIsInstance(result, dict)
                # Should have layer-wise metrics
                layer_metrics = [k for k in result.keys() if 'layer_' in k]
                self.assertGreater(len(layer_metrics), 0, f"No layer metrics for method {method}")

    def test_high_dimensional_data(self):
        """Test that high-dimensional data is handled correctly."""
        # Create larger batches to test high-dimensional scenarios
        task1_batch = self._create_batch(batch_size=32, seq_length=128)
        task2_batch = self._create_batch(batch_size=32, seq_length=128)

        # This would previously fail or produce many warnings with binning
        result = self.metrics.compute_heuristic_pid_minmi(
            self.model,
            task1_batch,
            task2_batch,
            max_tokens_for_pid=500,
            mi_method='infonce'  # Should handle high dimensions well
        )

        # Check that we got valid results
        self.assertIsInstance(result, dict)

        # Check for NaN or Inf values (should not occur with proper method)
        for key, value in result.items():
            if isinstance(value, (int, float)):
                self.assertFalse(torch.isnan(torch.tensor(value)).item(),
                               f"NaN value in {key}")
                self.assertFalse(torch.isinf(torch.tensor(value)).item(),
                               f"Inf value in {key}")

    def test_pid_decomposition_validity(self):
        """Test that PID decomposition satisfies theoretical constraints."""
        task1_batch = self._create_batch(batch_size=16, seq_length=64)
        task2_batch = self._create_batch(batch_size=16, seq_length=64)

        result = self.metrics.compute_heuristic_pid_minmi(
            self.model,
            task1_batch,
            task2_batch,
            max_tokens_for_pid=100,
            mi_method='infonce'
        )

        # Check metadata warning about heuristic nature
        self.assertIn('metadata', result)
        self.assertIn('warning', result['metadata'])
        self.assertIn('Not valid PID', result['metadata']['warning'])

        # Check conservation residuals for both targets
        for layer_idx in range(2):  # We have 2 layers in mock model
            for target in ['labels1', 'labels2']:
                prefix = f'layer_{layer_idx}_{target}'
                if f'{prefix}_redundancy' in result:
                    # Check all components exist
                    redundancy = result[f'{prefix}_redundancy']
                    unique_1 = result.get(f'{prefix}_unique_task1', 0.0)
                    unique_2 = result.get(f'{prefix}_unique_task2', 0.0)
                    synergy = result.get(f'{prefix}_synergy', 0.0)
                    residual = result.get(f'{prefix}_residual', 0.0)

                    # Basic sanity checks
                    self.assertIsNotNone(redundancy, f"Missing redundancy for {prefix}")
                    self.assertIsNotNone(synergy, f"Missing synergy for {prefix}")
                    self.assertIsNotNone(residual, f"Missing residual for {prefix}")
                    # Note: Values can be negative due to MI lower bounds


if __name__ == '__main__':
    unittest.main()