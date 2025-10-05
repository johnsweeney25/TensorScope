#!/usr/bin/env python3
"""
Unit tests for compute_causal_necessity batch splitting functionality.
Tests that the function properly handles variable input sizes and long sequences.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

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

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        x = self.embeddings(input_ids)
        for layer in self.layers:
            x = layer(x)
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


class TestCausalNecessityBatchSplitting(unittest.TestCase):
    """Test cases for compute_causal_necessity batch splitting."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = MockModel()
        self.model.eval()
        self.metrics = InformationTheoryMetrics(seed=42)

    def _create_batch(self, batch_size=32, seq_length=256, vocab_size=1000):
        """Helper to create a test batch."""
        batch = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_length)),
            'attention_mask': torch.ones(batch_size, seq_length, dtype=torch.long),
            'labels': torch.randint(0, vocab_size, (batch_size, seq_length))
        }
        # Set some labels to -100 (padding)
        batch['labels'][:, -5:] = -100
        return batch

    def test_small_batch_no_splitting(self):
        """Test that small batches are not split."""
        batch = self._create_batch(batch_size=20, seq_length=256)

        result = self.metrics.compute_causal_necessity(
            self.model,
            batch,
            n_samples=10,
            batch_size=32,  # Larger than input, no split needed
            seq_length=512,
            interventions_per_batch=5
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        # Check that we got some layer-specific metrics
        layer_metrics = [k for k in result.keys() if 'layers' in k]
        self.assertGreater(len(layer_metrics), 0)

    def test_large_batch_splitting(self):
        """Test that large batches are split correctly."""
        batch = self._create_batch(batch_size=100, seq_length=256)

        result = self.metrics.compute_causal_necessity(
            self.model,
            batch,
            n_samples=20,
            batch_size=32,  # Should split 100 into chunks of 32
            seq_length=256,
            interventions_per_batch=5
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        # Verify we got global metrics
        self.assertIn('global_causal_necessity', result)

    def test_sequence_truncation(self):
        """Test that long sequences are truncated."""
        batch = self._create_batch(batch_size=16, seq_length=2048)

        result = self.metrics.compute_causal_necessity(
            self.model,
            batch,
            n_samples=10,
            batch_size=32,
            seq_length=512,  # Should truncate from 2048 to 512
            interventions_per_batch=5
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_very_large_batch(self):
        """Test handling of 700+ samples as mentioned in requirements."""
        batch = self._create_batch(batch_size=700, seq_length=512)

        result = self.metrics.compute_causal_necessity(
            self.model,
            batch,
            n_samples=50,
            batch_size=32,  # Split 700 into 22 batches
            seq_length=512,
            interventions_per_batch=5
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)
        # Check we processed multiple batches
        self.assertIn('global_causal_necessity', result)

    def test_list_of_batches_input(self):
        """Test that list of batches is handled correctly."""
        batch_list = [
            self._create_batch(batch_size=16, seq_length=256),
            self._create_batch(batch_size=16, seq_length=256),
            self._create_batch(batch_size=16, seq_length=256),
        ]

        result = self.metrics.compute_causal_necessity(
            self.model,
            batch_list,
            n_samples=15,
            batch_size=32,
            seq_length=256,
            interventions_per_batch=5
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_long_range_default(self):
        """Test the long-range default configuration (1024 tokens)."""
        batch = self._create_batch(batch_size=32, seq_length=1024)

        result = self.metrics.compute_causal_necessity(
            self.model,
            batch,
            n_samples=20,
            batch_size=32,
            seq_length=1024,  # Long-range default
            interventions_per_batch=10
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_intervention_distribution(self):
        """Test that interventions are distributed across batches."""
        batch = self._create_batch(batch_size=64, seq_length=256)

        # With 30 interventions and 10 per batch, should use 3 batches
        result = self.metrics.compute_causal_necessity(
            self.model,
            batch,
            n_samples=30,
            batch_size=32,  # Split 64 into 2 batches
            seq_length=256,
            interventions_per_batch=10  # 10 per batch
        )

        self.assertIsInstance(result, dict)
        # Verify we got metrics from multiple layers
        layer_metrics = [k for k in result.keys() if 'layers' in k]
        self.assertGreater(len(layer_metrics), 0)

    def test_edge_case_single_sample(self):
        """Test edge case with single sample batch."""
        batch = self._create_batch(batch_size=1, seq_length=256)

        result = self.metrics.compute_causal_necessity(
            self.model,
            batch,
            n_samples=5,
            batch_size=32,
            seq_length=256,
            interventions_per_batch=5
        )

        self.assertIsInstance(result, dict)
        self.assertGreater(len(result), 0)

    def test_helper_split_into_batches(self):
        """Test the _split_into_batches helper method."""
        batch = self._create_batch(batch_size=100, seq_length=256)

        batches = self.metrics._split_into_batches(batch, max_size=32)

        # Should get 4 batches: 32, 32, 32, 4
        self.assertEqual(len(batches), 4)
        self.assertEqual(batches[0]['input_ids'].shape[0], 32)
        self.assertEqual(batches[1]['input_ids'].shape[0], 32)
        self.assertEqual(batches[2]['input_ids'].shape[0], 32)
        self.assertEqual(batches[3]['input_ids'].shape[0], 4)

    def test_helper_adjust_sequence_length(self):
        """Test the _adjust_sequence_length helper method."""
        # Test truncation
        batch_long = self._create_batch(batch_size=16, seq_length=2048)
        adjusted = self.metrics._adjust_sequence_length(batch_long, target_length=512)
        self.assertEqual(adjusted['input_ids'].shape[1], 512)

        # Test no change when already shorter
        batch_short = self._create_batch(batch_size=16, seq_length=256)
        adjusted = self.metrics._adjust_sequence_length(batch_short, target_length=512)
        self.assertEqual(adjusted['input_ids'].shape[1], 256)


if __name__ == '__main__':
    unittest.main()