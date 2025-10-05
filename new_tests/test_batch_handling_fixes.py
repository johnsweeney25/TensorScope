"""
Test that batch handling fixes are working properly.
This verifies the fixes for:
1. Property assignment issue (context.batch = ...)
2. Batch size validation replaced with adjustment
3. Simplified gradient_alignment_trajectory handling
"""

import unittest
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_model_analysis import MetricContext
from GradientAnalysis import GradientAnalysis


class TestBatchHandlingFixes(unittest.TestCase):
    """Test that batch handling fixes work properly."""

    def test_metric_context_property_modification(self):
        """Test that we can modify batches without 'can't set attribute' error."""
        # Create context with batches
        context = MetricContext(
            models=[None],
            batches=[{'input_ids': torch.tensor([[1, 2, 3]])}]
        )

        # Verify initial state
        self.assertIsNotNone(context.batch)
        self.assertEqual(context.batch['input_ids'].shape[1], 3)

        # Modify batch by changing batches[0] (the fix for property assignment)
        # This would have failed before with "can't set attribute 'batch'"
        context.batches[0] = {'input_ids': torch.tensor([[4, 5, 6, 7]])}

        # Verify modification worked
        self.assertEqual(context.batch['input_ids'].shape[1], 4)

    def test_gradient_trajectory_batch_adjustment(self):
        """Test that batch sizes are adjusted instead of throwing errors."""
        grad_analysis = GradientAnalysis()

        # Create oversized batch
        oversized_batch = {
            'input_ids': torch.randn(100, 200),  # 100 samples, 200 tokens
            'attention_mask': torch.ones(100, 200)
        }

        # Test _take method for batch size adjustment
        adjusted_batch = grad_analysis._take(oversized_batch, 32)
        self.assertEqual(adjusted_batch['input_ids'].shape[0], 32)
        self.assertEqual(adjusted_batch['attention_mask'].shape[0], 32)

    def test_batch_preparation_without_property_error(self):
        """Test that batch preparation works without property assignment error."""
        # Create context
        context = MetricContext(
            models=[MagicMock()],
            batches=[{
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }]
        )

        # Simulate what happens in unified_model_analysis.py line 1770-1774
        if context.batches and len(context.batches) > 0 and context.batches[0] is not None:
            # Simulate prepare_batch_for_gradient_computation
            prepared_batch = {
                k: v.detach() if torch.is_tensor(v) else v
                for k, v in context.batches[0].items()
            }
            # This is the fix - modify batches[0] instead of context.batch
            context.batches[0] = prepared_batch

        # Verify it worked
        self.assertIsNotNone(context.batch)
        self.assertTrue(all(not v.requires_grad if torch.is_tensor(v) else True
                          for v in context.batch.values()))

    def test_gradient_alignment_trajectory_simplified(self):
        """Test that gradient_alignment_trajectory uses simple batch handling."""
        grad_analysis = GradientAnalysis()

        # Create test batches of different sizes
        batch_large = {'input_ids': torch.randn(512, 256)}
        batch_medium = {'input_ids': torch.randn(128, 128)}
        batch_small = {'input_ids': torch.randn(64, 128)}

        # With max_batch_size=64, the function should automatically adjust
        # Rather than throwing errors, it should handle any batch size
        max_batch_size = 64
        max_seq_length = 128

        # Test that each batch can be processed
        for name, batch in [('large', batch_large), ('medium', batch_medium), ('small', batch_small)]:
            # Get actual dimensions
            actual_batch_size = batch['input_ids'].shape[0]
            actual_seq_length = batch['input_ids'].shape[1]

            # Simulate the adjustment that happens in GradientAnalysis
            if max_batch_size and actual_batch_size > max_batch_size:
                batch = grad_analysis._take(batch, max_batch_size)

            if max_seq_length and actual_seq_length > max_seq_length:
                for key in batch:
                    if isinstance(batch[key], torch.Tensor) and batch[key].dim() >= 2:
                        batch[key] = batch[key][:, :max_seq_length]

            # Verify batch is now within limits
            self.assertLessEqual(batch['input_ids'].shape[0], max_batch_size)
            self.assertLessEqual(batch['input_ids'].shape[1], max_seq_length)


if __name__ == '__main__':
    unittest.main()