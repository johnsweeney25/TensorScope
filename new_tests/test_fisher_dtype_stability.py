#!/usr/bin/env python3
"""
Unit tests for Fisher computation dtype stability and numerical precision handling.

Tests the dtype conversion pipeline for Fisher Information Matrix computation,
particularly for models with known numerical instabilities (e.g., Qwen models in float16).

Key test areas:
- BFloat16 computation for numerical stability
- Proper fallback to float32 with warnings when bfloat16 unavailable
- Cross-model comparability through consistent precision
- Integration with UnifiedModelAnalyzer and MetricRegistry

This ensures Fisher metrics work correctly across different hardware configurations
and model architectures without NaN losses or numerical overflow/underflow.
"""

import unittest
import torch
import torch.nn as nn
import logging
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisher.core.fisher_collector import FisherCollector
from BombshellMetrics import BombshellMetrics
from GradientAnalysis import GradientAnalysis
from unified_model_analysis import UnifiedConfig, MetricRegistry, UnifiedModelAnalyzer, ModelSpec

# Set up logging
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


class SimpleTestModel(nn.Module):
    """Simple model for testing Fisher computation."""
    def __init__(self, vocab_size=100, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embed(input_ids)
        x = x.mean(dim=1)  # Simple pooling
        logits = self.layer(x)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )

        # Return in Transformers-style format
        return type('Output', (), {'loss': loss, 'logits': logits})()


class TestFisherDtypeStability(unittest.TestCase):
    """Test suite for Fisher computation dtype stability."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = SimpleTestModel()
        self.batch = self._create_test_batch()

    def _create_test_batch(self):
        """Create a simple test batch."""
        batch_size = 4
        seq_length = 8
        return {
            'input_ids': torch.randint(0, 100, (batch_size, seq_length)),
            'labels': torch.randint(0, 100, (batch_size, seq_length))
        }

    def test_fisher_bfloat16_initialization(self):
        """Test FisherCollector initializes with bfloat16 configuration."""
        fisher = FisherCollector(computation_dtype='bfloat16')

        # Check that dtype is set (will be float32 on CPU)
        self.assertIsNotNone(fisher.computation_dtype)

        # On CPU or non-bf16 GPU, should fall back to float32
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            target_dtype = fisher._get_target_dtype()
            self.assertEqual(target_dtype, torch.float32,
                           "Should fall back to float32 on CPU/non-bf16 hardware")

    def test_fisher_no_nan_losses(self):
        """Test that Fisher computation doesn't produce NaN losses."""
        fisher = FisherCollector(computation_dtype='bfloat16')

        # Forward pass
        outputs = self.model(**self.batch)
        loss = outputs.loss

        # Check loss is valid
        self.assertFalse(torch.isnan(loss).item(),
                        "Loss should not be NaN")

        # Compute gradients
        loss.backward()

        # Update Fisher
        fisher.update_fisher_ema(self.model, self.batch)

        # Check Fisher values exist and are valid
        fisher_storage = fisher._fisher_ema_storage
        self.assertGreater(len(fisher_storage), 0,
                          "Fisher storage should not be empty")

        # Check for NaN in Fisher values
        has_nan = any(torch.isnan(v).any().item() for v in fisher_storage.values())
        self.assertFalse(has_nan, "Fisher values should not contain NaN")

    def test_bombshell_metrics_dtype_propagation(self):
        """Test that BombshellMetrics correctly propagates computation_dtype."""
        bombshell = BombshellMetrics(computation_dtype='bfloat16')

        # Check initialization succeeded
        self.assertIsNotNone(bombshell.computation_dtype)

        # Test Fisher computation through BombshellMetrics
        outputs = self.model(**self.batch)
        loss = outputs.loss
        loss.backward()

        bombshell.update_fisher_ema(self.model, self.batch)

        # Check Fisher was computed
        fisher_storage = bombshell._fisher_ema_storage
        self.assertGreater(len(fisher_storage), 0,
                          "BombshellMetrics should compute Fisher values")

    def test_gradient_analysis_dtype_handling(self):
        """Test GradientAnalysis dtype configuration."""
        grad = GradientAnalysis(computation_dtype='bfloat16')

        # Check dtype is set appropriately
        self.assertIsNotNone(grad._target_dtype)

        # On CPU, should use float32
        if not torch.cuda.is_available():
            self.assertEqual(grad._target_dtype, torch.float32,
                           "Should use float32 on CPU")

    def test_metric_registry_initialization(self):
        """Test MetricRegistry initializes without AttributeError."""
        config = UnifiedConfig(
            computation_dtype='bfloat16',
            force_computation_dtype=True,
            gradient_pathology_batch_size=128,
            random_seed=42
        )

        # This should not raise AttributeError
        try:
            registry = MetricRegistry(config=config)
            self.assertIsNotNone(registry, "Registry should initialize")
        except AttributeError as e:
            self.fail(f"MetricRegistry raised AttributeError: {e}")

    def test_batch_size_preservation(self):
        """Test that gradient pathology batch size is preserved at 128."""
        config = UnifiedConfig(
            computation_dtype='bfloat16',
            force_computation_dtype=True,
            gradient_pathology_batch_size=128,
            gradient_batch_size=128
        )

        # Check config values
        self.assertEqual(config.gradient_pathology_batch_size, 128,
                        "Gradient pathology batch size should be 128")
        self.assertEqual(config.gradient_batch_size, 128,
                        "Gradient batch size should be 128")

    def test_hardware_fallback_warnings(self):
        """Test that appropriate warnings are shown for hardware limitations."""
        import io
        import contextlib

        # Capture log output
        log_capture = io.StringIO()
        handler = logging.StreamHandler(log_capture)
        handler.setLevel(logging.WARNING)

        fisher_logger = logging.getLogger('fisher.core.fisher_collector')
        fisher_logger.addHandler(handler)

        try:
            # Create FisherCollector with bfloat16
            fisher = FisherCollector(computation_dtype='bfloat16')

            # Get target dtype (triggers warnings)
            target_dtype = fisher._get_target_dtype()

            # Check for warnings if not on bf16-capable hardware
            if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
                log_output = log_capture.getvalue()
                self.assertIn("BFloat16 requested but", log_output,
                             "Should warn about bfloat16 fallback")
                self.assertIn("COMPARABILITY WARNING", log_output,
                             "Should warn about comparability issues")
        finally:
            fisher_logger.removeHandler(handler)

    def test_qwen_specific_handling(self):
        """Test handling of Qwen-like models with known float16 issues."""
        # Simulate Qwen model scenario
        config = UnifiedConfig(
            computation_dtype='bfloat16',  # Use bfloat16 for stability
            force_computation_dtype=True,
            use_float16=False  # Don't use float16 due to Qwen issues
        )

        fisher = FisherCollector(computation_dtype=config.computation_dtype)

        # Verify not using float16
        target_dtype = fisher._get_target_dtype()
        self.assertNotEqual(target_dtype, torch.float16,
                           "Should not use float16 for Qwen-like models")

        # Should use bfloat16 or float32
        self.assertIn(target_dtype, [torch.bfloat16, torch.float32],
                     "Should use bfloat16 or float32 for numerical stability")


class TestIntegrationWithUnifiedAnalyzer(unittest.TestCase):
    """Test integration with UnifiedModelAnalyzer."""

    def test_unified_analyzer_with_dtype_config(self):
        """Test UnifiedModelAnalyzer with dtype configuration."""
        config = UnifiedConfig(
            base_model='gpt2',  # Use small model for testing
            computation_dtype='bfloat16',
            force_computation_dtype=True,
            gradient_pathology_batch_size=128,
            batch_size=2,
            metrics_to_compute=['update_fisher_ema'],
            skip_expensive=True,
            skip_fisher_metrics=False,
            device='cpu'
        )

        try:
            analyzer = UnifiedModelAnalyzer(config)
            self.assertIsNotNone(analyzer, "Analyzer should initialize")

            # Check registry is properly configured
            self.assertIsNotNone(analyzer.registry, "Registry should exist")

            # Check gradient module has correct configuration
            grad_module = analyzer.registry.modules.get('gradient')
            if grad_module and hasattr(grad_module, 'computation_dtype'):
                self.assertIsNotNone(grad_module.computation_dtype,
                                   "Gradient module should have computation_dtype")
        except Exception as e:
            self.fail(f"UnifiedModelAnalyzer initialization failed: {e}")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)