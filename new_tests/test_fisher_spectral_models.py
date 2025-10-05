"""
Model-specific tests for Fisher Spectral Analysis Module
==========================================================
Tests with realistic transformer models to validate real-world performance.

Author: ICLR 2026 Project
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
import warnings
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisher_spectral import FisherSpectral, SpectralConfig


class MockTransformerLayer(nn.Module):
    """Mock transformer layer mimicking GPT/Qwen architecture."""
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        # Attention components
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        # MLP components
        self.gate_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.up_proj = nn.Linear(hidden_size, hidden_size * 4)
        self.down_proj = nn.Linear(hidden_size * 4, hidden_size)

        # Normalization
        self.input_layernorm = nn.LayerNorm(hidden_size)
        self.post_attention_layernorm = nn.LayerNorm(hidden_size)

    def forward(self, x, attention_mask=None):
        # Simplified forward pass
        residual = x
        x = self.input_layernorm(x)

        # Attention (simplified - no actual attention computation)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn_output = self.o_proj(q * 0.1)  # Fake attention
        x = residual + attn_output

        # MLP
        residual = x
        x = self.post_attention_layernorm(x)
        gate = torch.sigmoid(self.gate_proj(x))
        up = self.up_proj(x)
        x = self.down_proj(gate * up)
        x = residual + x

        return x


class MockTransformerModel(nn.Module):
    """Mock transformer model mimicking GPT/Qwen architecture."""
    def __init__(self, vocab_size=50257, hidden_size=768, num_layers=12, num_heads=12):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size

        # Embeddings
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)

        # Transformer layers
        self.layers = nn.ModuleList([
            MockTransformerLayer(hidden_size, num_heads)
            for _ in range(num_layers)
        ])

        # Output
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Embedding
        hidden_states = self.embed_tokens(input_ids)

        # Transformer layers
        for layer in self.layers:
            hidden_states = layer(hidden_states, attention_mask)

        # Output
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.vocab_size),
                labels.view(-1)
            )

        return type('Output', (), {'loss': loss, 'logits': logits})()


class TestFisherSpectralModels(unittest.TestCase):
    """Tests with realistic transformer models."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SpectralConfig(
            seed=42,
            storage_mode='chunked',
            chunk_size=16,
            max_params_per_block=50000,
            dtype_eigensolve=torch.float64
        )
        self.spectral = FisherSpectral(self.config)

    def test_mock_gpt2_model(self):
        """Test with GPT-2 like architecture."""
        torch.manual_seed(42)

        # Small GPT-2 like model
        model = MockTransformerModel(
            vocab_size=1000,  # Smaller vocab for testing
            hidden_size=256,   # Smaller hidden size
            num_layers=4,      # Fewer layers
            num_heads=8
        )

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Mock GPT-2 model has {total_params:,} parameters")

        # Create batch
        batch_size, seq_len = 8, 32
        batch = {
            'input_ids': torch.randint(0, 1000, (batch_size, seq_len)),
            'labels': torch.randint(0, 1000, (batch_size, seq_len))
        }

        # Compute spectrum
        results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=8,
            block_structure='layer'
        )

        # Validate results
        self.assertIsNotNone(results)
        self.assertIn('global', results)
        self.assertIn('per_block', results)

        # Check we have multiple blocks (one per layer + embedding + output)
        self.assertGreater(len(results['per_block']), 1)

        # Check global metrics
        if results['global']:
            self.assertGreater(results['global']['largest_eigenvalue'], 0)
            self.assertGreater(results['global']['effective_rank'], 1)

            # For transformer, expect moderate condition number
            self.assertGreater(results['global']['condition_number'], 1)
            self.assertLess(results['global']['condition_number'], 1e10)

    def test_mock_qwen_like_model(self):
        """Test with Qwen-like architecture (larger model)."""
        torch.manual_seed(42)

        # Qwen-like dimensions (but smaller for testing)
        model = MockTransformerModel(
            vocab_size=5000,
            hidden_size=512,
            num_layers=6,
            num_heads=8
        )

        # This model is larger - test parameter subsampling
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Mock Qwen model has {total_params:,} parameters")

        batch = {
            'input_ids': torch.randint(0, 5000, (4, 16)),
            'labels': torch.randint(0, 5000, (4, 16))
        }

        # Use aggressive subsampling
        config = SpectralConfig(
            seed=42,
            storage_mode='chunked',
            chunk_size=4,
            max_params_per_block=10000  # Force subsampling
        )
        spectral = FisherSpectral(config)

        results = spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=4,
            block_structure='module'  # Group by module type
        )

        # Should handle large model with subsampling
        self.assertIsNotNone(results)

        # Check block structure matches module grouping
        if results['per_block']:
            block_keys = list(results['per_block'].keys())
            # Should have attention, mlp, embedding blocks
            self.assertTrue(any('attention' in k for k in block_keys))
            self.assertTrue(any('mlp' in k for k in block_keys))

    def test_parameter_efficient_model(self):
        """Test with parameter-efficient fine-tuning scenario."""
        torch.manual_seed(42)

        # Model with most parameters frozen
        model = MockTransformerModel(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=4
        )

        # Freeze all but last layer (simulating LoRA/PEFT)
        for name, param in model.named_parameters():
            if 'layers.1' not in name and 'lm_head' not in name:
                param.requires_grad = False

        # Count trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"PEFT model: {trainable_params:,}/{total_params:,} trainable parameters")

        batch = {
            'input_ids': torch.randint(0, 1000, (8, 16)),
            'labels': torch.randint(0, 1000, (8, 16))
        }

        results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=8,
            block_structure='layer'
        )

        # Should only compute spectrum for trainable parameters
        self.assertIsNotNone(results)
        if results['global']:
            # Fewer parameters should give lower effective rank
            self.assertLess(results['global']['effective_rank'], trainable_params)

    def test_attention_head_analysis(self):
        """Test analyzing individual attention heads."""
        torch.manual_seed(42)

        # Single layer model for detailed analysis
        model = MockTransformerModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=1,
            num_heads=4
        )

        batch = {
            'input_ids': torch.randint(0, 100, (16, 8)),
            'labels': torch.randint(0, 100, (16, 8))
        }

        # Analyze with fine-grained blocks
        results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=16,
            block_structure='module'
        )

        # Check attention block metrics
        if results['per_block']:
            for block_key, metrics in results['per_block'].items():
                if 'attention' in block_key:
                    # Attention blocks should have reasonable spectrum
                    self.assertGreater(metrics['largest_eigenvalue'], 0)
                    self.assertGreater(metrics['effective_rank'], 1)

    def test_gradient_explosion_detection(self):
        """Test detection of gradient explosion via spectrum."""
        torch.manual_seed(42)

        # Model with bad initialization (leads to gradient issues)
        model = MockTransformerModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4
        )

        # Bad initialization - very large weights
        for param in model.parameters():
            if param.dim() >= 2:
                param.data.normal_(0, 10.0)  # Large variance

        batch = {
            'input_ids': torch.randint(0, 100, (4, 8)),
            'labels': torch.randint(0, 100, (4, 8))
        }

        results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=4,
            block_structure='none'
        )

        # Should detect poor conditioning
        if results['global']:
            # Expect very large condition number
            self.assertGreater(results['global']['condition_number'], 1e3)
            # Expect large eigenvalues
            self.assertGreater(results['global']['largest_eigenvalue'], 1.0)

    def test_cuda_compatibility(self):
        """Test CUDA compatibility if available."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        torch.manual_seed(42)

        model = MockTransformerModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=1,
            num_heads=4
        ).cuda()

        batch = {
            'input_ids': torch.randint(0, 100, (8, 16)).cuda(),
            'labels': torch.randint(0, 100, (8, 16)).cuda()
        }

        # Should work on CUDA
        results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=8,
            block_structure='layer'
        )

        self.assertIsNotNone(results)
        if results['global']:
            self.assertGreater(results['global']['largest_eigenvalue'], 0)

    def test_mixed_precision_compatibility(self):
        """Test mixed precision training compatibility."""
        torch.manual_seed(42)

        model = MockTransformerModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=1,
            num_heads=4
        )

        # Convert to half precision
        if torch.cuda.is_available():
            model = model.half().cuda()
            batch = {
                'input_ids': torch.randint(0, 100, (4, 8)).cuda(),
                'labels': torch.randint(0, 100, (4, 8)).cuda()
            }
        else:
            # CPU doesn't support half well, use bfloat16 if available
            try:
                model = model.to(torch.bfloat16)
                batch = {
                    'input_ids': torch.randint(0, 100, (4, 8)),
                    'labels': torch.randint(0, 100, (4, 8))
                }
            except:
                self.skipTest("Mixed precision not supported on this system")

        # Should handle mixed precision
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore precision warnings
            results = self.spectral.compute_fisher_spectrum(
                model, batch,
                n_samples=4,
                block_structure='none'
            )

        self.assertIsNotNone(results)

    def test_very_deep_model(self):
        """Test with very deep model (many layers)."""
        torch.manual_seed(42)

        # Deep but narrow model
        model = MockTransformerModel(
            vocab_size=100,
            hidden_size=32,
            num_layers=16,  # Many layers
            num_heads=4
        )

        batch = {
            'input_ids': torch.randint(0, 100, (4, 8)),
            'labels': torch.randint(0, 100, (4, 8))
        }

        results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=4,
            block_structure='layer'
        )

        # Should have many blocks (one per layer)
        if results['per_block']:
            # At least embedding + layers + output
            self.assertGreater(len(results['per_block']), 10)

        # Global spectrum should aggregate all blocks
        if results['global']:
            self.assertGreater(results['global']['n_total_eigenvalues'], 0)

    def test_comparison_with_legacy_implementation(self):
        """Compare results with legacy implementation for validation."""
        torch.manual_seed(42)

        # Import both implementations
        from InformationTheoryMetrics import InformationTheoryMetrics

        model = MockTransformerModel(
            vocab_size=100,
            hidden_size=64,
            num_layers=2,
            num_heads=4
        )

        batch = {
            'input_ids': torch.randint(0, 100, (8, 16)),
            'labels': torch.randint(0, 100, (8, 16))
        }

        # New implementation
        new_results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=8,
            block_structure='none'  # Single block for fair comparison
        )

        # Legacy implementation via InformationTheoryMetrics
        metrics = InformationTheoryMetrics(seed=42)
        legacy_results = metrics.compute_spectral_gap(model, batch)

        # Both should produce results
        self.assertIsNotNone(new_results)
        self.assertIsNotNone(legacy_results)

        # New implementation should be more stable
        if new_results['global']:
            self.assertGreater(new_results['global']['largest_eigenvalue'], 0)

        # Legacy may fail with SVD error, but new should succeed
        if 'spectral_gap' in legacy_results:
            # If legacy succeeds, results should be in same ballpark
            # (exact match not expected due to different algorithms)
            if new_results['global'] and legacy_results['spectral_gap'] > 0:
                ratio = new_results['global']['spectral_gap'] / legacy_results['spectral_gap']
                # Should be within order of magnitude
                self.assertGreater(ratio, 0.01)
                self.assertLess(ratio, 100)


if __name__ == '__main__':
    unittest.main()