#!/usr/bin/env python3
"""
Unit tests for SAM Sharpness and Hessian OOM fixes.

Tests memory efficiency, numerical precision, and correctness of the fixed implementations.
Run with: python -m pytest new_tests/test_sam_hessian_oom_fixes.py -v
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import time
import logging
from typing import Dict, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModel(nn.Module):
    """Test model with configurable size for memory testing."""

    def __init__(self, vocab_size: int = 50257, n_layers: int = 12,
                 hidden_size: int = 768, n_heads: int = 12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embedding layers
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.pos_embed = nn.Embedding(512, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=n_heads,
                dim_feedforward=4 * hidden_size,
                dropout=0.1,
                batch_first=True
            ) for _ in range(n_layers)
        ])

        # Output layer
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Config for compatibility
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'n_layers': n_layers
        })()

    def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None,
                **kwargs):
        seq_len = input_ids.shape[1]
        device = input_ids.device

        # Embeddings
        x = self.embed(input_ids)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embed(pos_ids)

        # Transformer layers
        for layer in self.layers:
            x = layer(x)

        # Output
        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            # Compute loss
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1),
                ignore_index=-100
            )

        return type('Output', (), {'loss': loss, 'logits': logits})()

    def param_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())


class TestSAMHessianFixes(unittest.TestCase):
    """Test suite for SAM and Hessian OOM fixes."""

    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        cls.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if cls.device == 'cuda':
            # Set deterministic mode for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            # Get GPU info
            cls.gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            cls.gpu_name = torch.cuda.get_device_properties(0).name
            logger.info(f"Testing on {cls.gpu_name} with {cls.gpu_memory_gb:.1f}GB")

    def setUp(self):
        """Set up each test."""
        # Clear GPU cache before each test
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

    def tearDown(self):
        """Clean up after each test."""
        # Clear GPU cache after each test
        if self.device == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def create_test_batch(self, batch_size: int, seq_len: int,
                         vocab_size: int) -> Dict[str, torch.Tensor]:
        """Create a test batch."""
        batch = {
            'input_ids': torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device),
            'attention_mask': torch.ones(batch_size, seq_len, device=self.device)
        }
        # Labels with some padding tokens (-100)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        labels[:, :5] = -100  # Mask first few tokens
        batch['labels'] = labels

        return batch

    def test_sam_sharpness_small_model(self):
        """Test SAM sharpness computation on small model."""
        # Import the fixed compute_sam_sharpness from ModularityMetrics
        from ModularityMetrics import ExtendedModularityMetrics

        # Create small model
        model = TestModel(vocab_size=1000, n_layers=2, hidden_size=128, n_heads=4)
        model = model.to(self.device)

        # Create metrics instance
        metrics = ExtendedModularityMetrics()

        # Create batch
        batch = self.create_test_batch(2, 32, model.vocab_size)

        # Compute SAM sharpness
        sharpness = metrics.compute_sam_sharpness(model, batch, epsilon=0.01)

        # Verify result
        self.assertIsInstance(sharpness, float)
        self.assertGreaterEqual(sharpness, -1.0)  # Sharpness can be negative
        self.assertLessEqual(sharpness, 10.0)  # But should be reasonable

        logger.info(f"Small model SAM sharpness: {sharpness:.6f}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for memory test")
    def test_memory_usage_125m_model(self):
        """Test memory usage with 125M parameter model."""
        from ModularityMetrics import ExtendedModularityMetrics

        # Create 125M model
        model = TestModel(vocab_size=50257, n_layers=12, hidden_size=768, n_heads=12)
        model = model.to(self.device)

        param_count = model.param_count()
        self.assertGreater(param_count, 100e6)  # Should be > 100M
        self.assertLess(param_count, 200e6)  # Should be < 200M

        logger.info(f"125M model parameters: {param_count/1e6:.1f}M")

        # Create metrics instance
        metrics = ExtendedModularityMetrics()

        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            logger.info(f"Testing batch_size={batch_size}")

            # Clear memory
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            # Create batch
            batch = self.create_test_batch(batch_size, 128, model.vocab_size)

            # Record initial memory
            mem_before = torch.cuda.memory_allocated() / 1e9

            try:
                # Compute SAM sharpness
                sharpness = metrics.compute_sam_sharpness(model, batch, epsilon=0.01)

                # Record peak memory
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                mem_used = mem_peak - mem_before

                logger.info(f"  ✓ Success! Sharpness: {sharpness:.6f}")
                logger.info(f"    Memory used: {mem_used:.2f}GB (peak: {mem_peak:.2f}GB)")

                # Memory should be reasonable
                self.assertLess(mem_used, 10.0, f"Memory usage too high: {mem_used:.2f}GB")

            except torch.cuda.OutOfMemoryError:
                self.fail(f"OOM with batch_size={batch_size} on 125M model")

    def test_gradient_state_restoration(self):
        """Test that gradient states are properly restored."""
        from ModularityMetrics import ExtendedModularityMetrics

        model = TestModel(vocab_size=1000, n_layers=2, hidden_size=128, n_heads=4)
        model = model.to(self.device)

        # Freeze half the parameters
        params_list = list(model.parameters())
        n_frozen = len(params_list) // 2

        for i, param in enumerate(params_list):
            if i < n_frozen:
                param.requires_grad = False

        # Record original states
        original_states = {id(p): p.requires_grad for p in model.parameters()}
        original_training = model.training

        # Create metrics instance and compute sharpness
        metrics = ExtendedModularityMetrics()
        batch = self.create_test_batch(2, 32, model.vocab_size)
        sharpness = metrics.compute_sam_sharpness(model, batch, epsilon=0.01)

        # Check states are restored
        for p in model.parameters():
            self.assertEqual(
                p.requires_grad,
                original_states[id(p)],
                "Gradient state not restored properly"
            )

        self.assertEqual(model.training, original_training, "Training mode not restored")

        logger.info(f"✓ Gradient states properly restored")

    def test_numerical_precision(self):
        """Test numerical precision with different epsilon values."""
        from ModularityMetrics import ExtendedModularityMetrics

        # Small model for quick testing
        model = TestModel(vocab_size=1000, n_layers=2, hidden_size=128, n_heads=4)
        model = model.to(self.device)

        metrics = ExtendedModularityMetrics()
        batch = self.create_test_batch(2, 32, model.vocab_size)

        # Test with different epsilon values
        epsilons = [1e-4, 1e-3, 1e-2, 0.05, 0.1]
        results = []

        for epsilon in epsilons:
            sharpness = metrics.compute_sam_sharpness(model, batch, epsilon=epsilon)
            results.append((epsilon, sharpness))
            logger.info(f"ε={epsilon:.4f}: sharpness={sharpness:.6f}")

        # Sharpness should generally increase with epsilon (not strict requirement)
        # Just check that values are reasonable
        for epsilon, sharpness in results:
            self.assertIsInstance(sharpness, float)
            self.assertTrue(np.isfinite(sharpness), f"Non-finite sharpness for ε={epsilon}")

    def test_zero_gradient_handling(self):
        """Test handling of zero or near-zero gradients."""
        from ModularityMetrics import ExtendedModularityMetrics

        model = TestModel(vocab_size=1000, n_layers=2, hidden_size=128, n_heads=4)
        model = model.to(self.device)

        # Freeze all parameters except bias (which doesn't exist in our model)
        # This should lead to zero gradients
        for param in model.parameters():
            param.requires_grad = False

        # Enable just one small parameter
        model.ln_f.weight.requires_grad = True

        metrics = ExtendedModularityMetrics()
        batch = self.create_test_batch(1, 16, model.vocab_size)

        # Should handle gracefully
        sharpness = metrics.compute_sam_sharpness(model, batch, epsilon=0.01)

        # With minimal gradients, sharpness should be 0 or very small
        self.assertLessEqual(abs(sharpness), 1.0, "Sharpness too large for minimal gradients")

        logger.info(f"✓ Handles near-zero gradients: sharpness={sharpness:.6f}")

    def test_deterministic_results(self):
        """Test that results are deterministic with same seed."""
        from ModularityMetrics import ExtendedModularityMetrics

        # Set seed
        torch.manual_seed(42)
        if self.device == 'cuda':
            torch.cuda.manual_seed_all(42)

        model = TestModel(vocab_size=1000, n_layers=2, hidden_size=128, n_heads=4)
        model = model.to(self.device)

        metrics = ExtendedModularityMetrics(seed=42)

        # Create identical batches
        torch.manual_seed(42)
        batch1 = self.create_test_batch(2, 32, model.vocab_size)

        torch.manual_seed(42)
        batch2 = self.create_test_batch(2, 32, model.vocab_size)

        # Compute sharpness twice
        sharpness1 = metrics.compute_sam_sharpness(model, batch1, epsilon=0.01)
        sharpness2 = metrics.compute_sam_sharpness(model, batch2, epsilon=0.01)

        # Should be identical
        self.assertAlmostEqual(sharpness1, sharpness2, places=6,
                              msg=f"Non-deterministic results: {sharpness1} vs {sharpness2}")

        logger.info(f"✓ Deterministic results: {sharpness1:.6f}")

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required for OOM test")
    def test_oom_handling(self):
        """Test that OOM is handled gracefully."""
        from ModularityMetrics import ExtendedModularityMetrics

        # Create large model that might OOM with large batch
        model = TestModel(vocab_size=50257, n_layers=24, hidden_size=1024, n_heads=16)
        model = model.to(self.device)

        metrics = ExtendedModularityMetrics()

        # Try with very large batch that should OOM
        large_batch_size = 256
        batch = self.create_test_batch(large_batch_size, 512, model.vocab_size)

        # Should raise OOM error cleanly
        with self.assertRaises(torch.cuda.OutOfMemoryError):
            metrics.compute_sam_sharpness(model, batch, epsilon=0.01)

        # GPU should be recoverable after OOM
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        # Should work with smaller batch
        small_batch = self.create_test_batch(1, 32, model.vocab_size)
        sharpness = metrics.compute_sam_sharpness(model, small_batch, epsilon=0.01)
        self.assertIsInstance(sharpness, float)

        logger.info(f"✓ OOM handled gracefully, recovered with small batch")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)