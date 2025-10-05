"""
Comprehensive unit tests for _prepare_batch_for_config function.
Tests theory correctness, numerical precision, and edge cases.
"""

import unittest
import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GradientAnalysis import GradientAnalysis


class TestBatchPreparation(unittest.TestCase):
    """Test suite for batch preparation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GradientAnalysis(device='cpu')
        torch.manual_seed(42)

    def create_test_batch(self, batch_size=4, seq_len=10, include_special_keys=False):
        """Create a test batch with various tensor types."""
        batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'attention_mask': torch.ones(batch_size, seq_len, dtype=torch.long),
        }

        if include_special_keys:
            batch['labels'] = torch.randint(0, 1000, (batch_size, seq_len))
            batch['position_ids'] = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
            batch['token_type_ids'] = torch.zeros(batch_size, seq_len, dtype=torch.long)

        return batch

    # ========== Input Validation Tests ==========

    def test_negative_dimensions_raise_error(self):
        """Test that negative dimensions raise ValueError."""
        batch = self.create_test_batch()

        with self.assertRaises(ValueError) as context:
            self.analyzer._prepare_batch_for_config(batch, -1, 10)
        self.assertIn("must be positive", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.analyzer._prepare_batch_for_config(batch, 4, -1)
        self.assertIn("must be positive", str(context.exception))

    def test_non_integer_dimensions_raise_error(self):
        """Test that non-integer dimensions raise ValueError."""
        batch = self.create_test_batch()

        with self.assertRaises(ValueError) as context:
            self.analyzer._prepare_batch_for_config(batch, 4.5, 10)
        self.assertIn("must be integers", str(context.exception))

        with self.assertRaises(ValueError) as context:
            self.analyzer._prepare_batch_for_config(batch, 4, 10.5)
        self.assertIn("must be integers", str(context.exception))

    def test_empty_batch_handling(self):
        """Test handling of empty or malformed batches."""
        # Empty batch
        result = self.analyzer._prepare_batch_for_config({}, 4, 10)
        self.assertEqual(result, {})

        # Batch without input_ids
        batch = {'attention_mask': torch.ones(4, 10)}
        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)
        self.assertEqual(result, batch)

    # ========== Batch Size Adjustment Tests ==========

    def test_batch_size_reduction_deterministic(self):
        """Test that batch size reduction is deterministic with seed."""
        batch = self.create_test_batch(batch_size=8, seq_len=10)

        # Run twice with same seed
        result1 = self.analyzer._prepare_batch_for_config(
            batch, 4, 10, sampling_seed=42
        )
        result2 = self.analyzer._prepare_batch_for_config(
            batch, 4, 10, sampling_seed=42
        )

        self.assertTrue(torch.equal(result1['input_ids'], result2['input_ids']))
        self.assertEqual(result1['input_ids'].shape[0], 4)

    def test_batch_size_duplication_warning(self):
        """Test that batch size duplication raises error by default."""
        batch = self.create_test_batch(batch_size=2, seq_len=10)

        # Should raise error without allow_duplication
        with self.assertRaises(ValueError) as context:
            self.analyzer._prepare_batch_for_config(batch, 4, 10)
        self.assertIn("Sample duplication would bias gradients", str(context.exception))

        # Should work with allow_duplication
        result = self.analyzer._prepare_batch_for_config(
            batch, 4, 10, allow_duplication=True
        )
        self.assertEqual(result['input_ids'].shape[0], 4)

    def test_batch_duplication_creates_correct_pattern(self):
        """Test that batch duplication cycles through samples correctly."""
        batch = self.create_test_batch(batch_size=3, seq_len=10)

        result = self.analyzer._prepare_batch_for_config(
            batch, 7, 10, allow_duplication=True, sampling_seed=42
        )

        # Check that samples are cycled: [0,1,2,0,1,2,0]
        expected_pattern = [0, 1, 2, 0, 1, 2, 0]
        for i in range(7):
            expected_idx = expected_pattern[i]
            self.assertTrue(torch.equal(
                result['input_ids'][i],
                batch['input_ids'][expected_idx]
            ))

    # ========== Sequence Length Adjustment Tests ==========

    def test_sequence_truncation(self):
        """Test that sequences are truncated correctly."""
        batch = self.create_test_batch(batch_size=4, seq_len=20)

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        self.assertEqual(result['input_ids'].shape[1], 10)
        # Check that first 10 tokens are preserved
        self.assertTrue(torch.equal(
            result['input_ids'],
            batch['input_ids'][:, :10]
        ))

    def test_sequence_padding_right(self):
        """Test right padding for sequences."""
        batch = self.create_test_batch(batch_size=4, seq_len=5)

        result = self.analyzer._prepare_batch_for_config(
            batch, 4, 10, pad_token_id=999, padding_side='right'
        )

        self.assertEqual(result['input_ids'].shape[1], 10)
        # Check that original tokens are preserved
        self.assertTrue(torch.equal(
            result['input_ids'][:, :5],
            batch['input_ids']
        ))
        # Check padding tokens
        self.assertTrue(torch.all(result['input_ids'][:, 5:] == 999))

    def test_sequence_padding_left(self):
        """Test left padding for sequences."""
        batch = self.create_test_batch(batch_size=4, seq_len=5)

        result = self.analyzer._prepare_batch_for_config(
            batch, 4, 10, pad_token_id=999, padding_side='left'
        )

        self.assertEqual(result['input_ids'].shape[1], 10)
        # Check that original tokens are preserved
        self.assertTrue(torch.equal(
            result['input_ids'][:, 5:],
            batch['input_ids']
        ))
        # Check padding tokens
        self.assertTrue(torch.all(result['input_ids'][:, :5] == 999))

    # ========== Special Keys Handling Tests ==========

    def test_attention_mask_padding(self):
        """Test that attention mask is padded with zeros."""
        batch = self.create_test_batch(batch_size=4, seq_len=5)

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        # Check that attention mask is padded with 0s
        self.assertTrue(torch.all(result['attention_mask'][:, 5:] == 0))
        self.assertTrue(torch.all(result['attention_mask'][:, :5] == 1))

    def test_labels_padding(self):
        """Test that labels are padded with -100."""
        batch = self.create_test_batch(batch_size=4, seq_len=5, include_special_keys=True)

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        # Check that labels are padded with -100
        self.assertTrue(torch.all(result['labels'][:, 5:] == -100))
        # Original labels preserved
        self.assertTrue(torch.equal(
            result['labels'][:, :5],
            batch['labels']
        ))

    def test_position_ids_padding(self):
        """Test that position_ids continue the sequence."""
        batch = self.create_test_batch(batch_size=4, seq_len=5, include_special_keys=True)

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        # Check that position_ids continue from 5 to 9
        expected_positions = torch.arange(5, 10).unsqueeze(0).expand(4, -1)
        self.assertTrue(torch.equal(
            result['position_ids'][:, 5:],
            expected_positions
        ))

    def test_token_type_ids_padding(self):
        """Test that token_type_ids are padded with the last type."""
        batch = self.create_test_batch(batch_size=4, seq_len=5, include_special_keys=True)
        # Set some token_type_ids to 1
        batch['token_type_ids'][:2, :] = 1

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        # Check that padding matches last token type
        self.assertTrue(torch.all(result['token_type_ids'][:2, 5:] == 1))
        self.assertTrue(torch.all(result['token_type_ids'][2:, 5:] == 0))

    # ========== Device Handling Tests ==========

    def test_mixed_device_handling(self):
        """Test handling of tensors on different devices."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        batch = self.create_test_batch(batch_size=4, seq_len=10)
        # Put attention_mask on different device
        batch['attention_mask'] = batch['attention_mask'].cuda()

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        # All tensors should be on the same device
        devices = [v.device for v in result.values() if torch.is_tensor(v)]
        self.assertEqual(len(set(devices)), 1)

    # ========== Dtype Preservation Tests ==========

    def test_dtype_preservation(self):
        """Test that dtypes are preserved during padding."""
        batch = {
            'input_ids': torch.randint(0, 1000, (4, 5), dtype=torch.int32),
            'attention_mask': torch.ones(4, 5, dtype=torch.int8),
            'embeddings': torch.randn(4, 5, 768, dtype=torch.float16),
        }

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        self.assertEqual(result['input_ids'].dtype, torch.int32)
        self.assertEqual(result['attention_mask'].dtype, torch.int8)
        self.assertEqual(result['embeddings'].dtype, torch.float16)

    # ========== Numerical Precision Tests ==========

    def test_no_numerical_overflow(self):
        """Test that large batch sizes don't cause overflow."""
        batch = self.create_test_batch(batch_size=2, seq_len=10)

        # This should not overflow
        result = self.analyzer._prepare_batch_for_config(
            batch, 10000, 10, allow_duplication=True
        )

        self.assertEqual(result['input_ids'].shape[0], 10000)
        self.assertFalse(torch.any(torch.isnan(result['input_ids'].float())))

    def test_gradient_bias_from_duplication(self):
        """Test that duplicated samples produce identical gradients (demonstrating bias)."""
        torch.manual_seed(42)

        # Create a simple model for gradient testing
        model = torch.nn.Linear(10, 1)

        # Original batch
        original_batch = torch.randn(2, 10)

        # Duplicated batch (simulating what the function does)
        duplicated_batch = original_batch.repeat(2, 1)  # [0, 1, 0, 1]

        # Compute gradients for original
        model.zero_grad()
        loss_original = model(original_batch).sum()
        loss_original.backward()
        grad_original = model.weight.grad.clone()

        # Compute gradients for duplicated
        model.zero_grad()
        loss_duplicated = model(duplicated_batch).sum()
        loss_duplicated.backward()
        grad_duplicated = model.weight.grad.clone()

        # Gradient should be 2x for duplicated (demonstrating bias)
        self.assertTrue(torch.allclose(grad_duplicated, 2 * grad_original, rtol=1e-5))

    # ========== Edge Cases ==========

    def test_single_sample_batch(self):
        """Test handling of single-sample batches."""
        batch = self.create_test_batch(batch_size=1, seq_len=10)

        result = self.analyzer._prepare_batch_for_config(
            batch, 4, 10, allow_duplication=True
        )

        self.assertEqual(result['input_ids'].shape[0], 4)
        # All samples should be identical
        for i in range(1, 4):
            self.assertTrue(torch.equal(
                result['input_ids'][i],
                result['input_ids'][0]
            ))

    def test_exact_match_dimensions(self):
        """Test that exact match dimensions return unchanged batch."""
        batch = self.create_test_batch(batch_size=4, seq_len=10)
        original_input_ids = batch['input_ids'].clone()

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        self.assertTrue(torch.equal(result['input_ids'], original_input_ids))

    def test_non_tensor_values_preserved(self):
        """Test that non-tensor values in batch are preserved."""
        batch = self.create_test_batch(batch_size=4, seq_len=10)
        batch['metadata'] = {'key': 'value'}
        batch['list_data'] = [1, 2, 3]

        result = self.analyzer._prepare_batch_for_config(batch, 2, 5)

        self.assertEqual(result['metadata'], {'key': 'value'})
        self.assertEqual(result['list_data'], [1, 2, 3])

    def test_3d_tensor_padding(self):
        """Test padding of 3D tensors (e.g., embeddings)."""
        batch = {
            'input_ids': torch.randint(0, 1000, (4, 5)),
            'hidden_states': torch.randn(4, 5, 768),
        }

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        self.assertEqual(result['hidden_states'].shape, (4, 10, 768))
        # Check that padding is zeros
        self.assertTrue(torch.all(result['hidden_states'][:, 5:, :] == 0))

    # ========== Default Pad Token Tests ==========

    def test_default_pad_token_warning(self):
        """Test that default pad token usage is logged."""
        batch = self.create_test_batch(batch_size=4, seq_len=5)

        # Should use default (0) when not specified
        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        # Check padding uses default
        self.assertTrue(torch.all(result['input_ids'][:, 5:] == 0))

    def test_custom_pad_token_from_attribute(self):
        """Test that pad token can be read from analyzer attribute."""
        self.analyzer._pad_token_id = 50256  # GPT-2 pad token
        batch = self.create_test_batch(batch_size=4, seq_len=5)

        result = self.analyzer._prepare_batch_for_config(batch, 4, 10)

        # Should use attribute pad token
        self.assertTrue(torch.all(result['input_ids'][:, 5:] == 50256))


if __name__ == '__main__':
    unittest.main()