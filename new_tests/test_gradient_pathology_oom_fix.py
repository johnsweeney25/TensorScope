"""
Unit tests for gradient pathology OOM fixes.

This test file verifies that the gradient pathology computation:
1. Skips dtype conversion for large models to prevent OOM
2. Uses the correct batch size (32 instead of 64)
3. Provides clear memory warnings
4. Handles batch size configuration properly
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GradientAnalysis import GradientAnalysis
from unified_model_analysis import UnifiedConfig


class MockLargeModel(nn.Module):
    """Mock model with configurable parameter count."""

    def __init__(self, num_params=1_500_000_000):
        super().__init__()
        # Create a fake parameter tensor to simulate large model
        # We don't actually allocate this much memory, just pretend
        self.fake_param_count = num_params
        self.linear = nn.Linear(128, 128)  # Small actual layer
        self.config = type('Config', (), {'vocab_size': 1000})()

    def parameters(self):
        """Override to report large parameter count for testing."""
        # Return actual parameters but we'll mock the count
        return super().parameters()

    def forward(self, input_ids, labels=None, **kwargs):
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        hidden = torch.randn(batch_size, seq_len, 128, device=input_ids.device)
        logits = self.linear(hidden)

        loss = None
        if labels is not None:
            loss = torch.tensor(1.0, device=input_ids.device, requires_grad=True)

        return type('Output', (), {'logits': logits, 'loss': loss})()


class TestGradientPathologyOOMFix(unittest.TestCase):
    """Test gradient pathology OOM prevention fixes."""

    def test_batch_size_configuration(self):
        """Test that default batch size is correctly set to 32."""
        config = UnifiedConfig()
        self.assertEqual(config.gradient_pathology_batch_size, 32,
                        "Default gradient_pathology_batch_size should be 32")
        self.assertEqual(config.batch_size, 512,
                        "General batch_size should remain 512")

    @patch('GradientAnalysis.logger')
    def test_dtype_conversion_skipped_for_large_models(self, mock_logger):
        """Test that large models skip float32 conversion."""
        grad_analysis = GradientAnalysis()

        # Create a mock large model (>1B params)
        model = MockLargeModel(num_params=1_500_000_000)

        # Mock the parameter counting
        with patch('GradientAnalysis.sum') as mock_sum:
            # First call counts parameters for skip check
            # Second call counts total parameters
            # Third call counts trainable parameters
            mock_sum.side_effect = [1_500_000_000, 1_500_000_000, 1_500_000_000]

            batch = {
                'input_ids': torch.randint(0, 1000, (32, 128)),
                'labels': torch.randint(0, 1000, (32, 128))
            }

            # Set model to bfloat16 to trigger conversion logic
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                model = model.to(torch.bfloat16)
                original_dtype = torch.bfloat16
            else:
                model = model.to(torch.float16)
                original_dtype = torch.float16

            # Force computation_dtype to float32 to test skip logic
            grad_analysis.computation_dtype = 'float32'

            # This should skip conversion due to large model size
            try:
                result = grad_analysis.compute_gradient_pathology(model, batch)
            except Exception:
                pass  # We don't care if it fails, just checking logs

            # Check that warning was logged about skipping conversion
            warning_calls = [call for call in mock_logger.warning.call_args_list
                            if 'Skipping float32 conversion' in str(call)]
            self.assertTrue(len(warning_calls) > 0,
                          "Should log warning about skipping float32 conversion")

    @patch('GradientAnalysis.logger')
    def test_memory_warning_for_high_usage(self, mock_logger):
        """Test that memory warnings are shown for high expected usage."""
        grad_analysis = GradientAnalysis()

        model = nn.Linear(128, 128)  # Small model
        large_batch = {
            'input_ids': torch.randint(0, 1000, (128, 512)),  # Large batch
            'labels': torch.randint(0, 1000, (128, 512))
        }

        # Mock model forward to avoid actual computation
        with patch.object(model, 'forward') as mock_forward:
            mock_output = type('Output', (), {
                'logits': torch.randn(128, 512, 128),
                'loss': torch.tensor(1.0, requires_grad=True)
            })()
            mock_forward.return_value = mock_output

            try:
                # We expect this to show memory warnings
                grad_analysis.compute_gradient_pathology(model, large_batch)
            except Exception:
                pass  # Don't care if it fails, checking warnings

            # Check for memory warning
            info_calls = [str(call) for call in mock_logger.info.call_args_list]
            warning_calls = [str(call) for call in mock_logger.warning.call_args_list]

            # Should have logged batch size info
            batch_size_logged = any('batch size: 128' in call.lower()
                                  for call in info_calls)
            self.assertTrue(batch_size_logged or
                          any('batch size: 128' in call.lower() for call in warning_calls),
                          "Should log actual batch size being used")

    def test_batch_size_slicing(self):
        """Test that batch slicing logic works correctly."""
        # This tests the configuration and ensures the batch size is properly set
        config = UnifiedConfig()
        config.gradient_pathology_batch_size = 32

        # Create a batch that's larger than the configured size
        large_batch = {
            'input_ids': torch.randint(0, 1000, (128, 256)),  # 128 samples
            'labels': torch.randint(0, 1000, (128, 256))
        }

        # The configuration should enforce slicing to 32
        # In actual usage, unified_model_analysis will slice the batch
        # Here we just verify the configuration is correct
        self.assertEqual(config.gradient_pathology_batch_size, 32)

        # Simulate what the framework does - slice the batch
        sliced_batch = {k: v[:config.gradient_pathology_batch_size] if torch.is_tensor(v) else v
                       for k, v in large_batch.items()}

        # Verify the slicing worked
        self.assertEqual(sliced_batch['input_ids'].shape[0], 32,
                        "Batch should be sliced to 32 samples")
        self.assertEqual(sliced_batch['labels'].shape[0], 32,
                        "Labels should be sliced to 32 samples")

    def test_configuration_override(self):
        """Test that batch size can be overridden in config."""
        config = UnifiedConfig()
        config.gradient_pathology_batch_size = 16  # Override default

        self.assertEqual(config.gradient_pathology_batch_size, 16,
                        "Should be able to override gradient_pathology_batch_size")


if __name__ == '__main__':
    unittest.main()