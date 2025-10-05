"""
Comprehensive test suite for SuperpositionMetrics module.

Tests all methods for correctness, numerical stability, and edge cases.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict
import warnings

from superposition.core.enhanced import SuperpositionMetrics, analyze_superposition


class DummyModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, vocab_size: int = 100, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(n_layers)
        ])
        self.output = nn.Linear(hidden_dim, vocab_size)

        # Add config attribute for compatibility
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_dim,
            'n_layers': n_layers
        })()

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.output(x)


class TestSuperpositionMetrics(unittest.TestCase):
    """Test suite for SuperpositionMetrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = SuperpositionMetrics(device=self.device)
        self.model = DummyModel().to(self.device)

        # Create test batch
        self.batch_size = 8
        self.seq_length = 16
        self.test_batch = {
            'input_ids': torch.randint(0, 100, (self.batch_size, self.seq_length)).to(self.device),
            'labels': torch.randint(0, 100, (self.batch_size, self.seq_length)).to(self.device)
        }

    def test_vector_interference_basic(self):
        """Test basic vector interference computation."""
        # Create test weight matrix
        n_features, n_dims = 50, 32
        weight_matrix = torch.randn(n_features, n_dims)

        # Compute interference
        result = self.metrics.compute_vector_interference(
            weight_matrix,
            normalize=True,
            return_full_matrix=False
        )

        # Check outputs
        self.assertIn('mean_overlap', result)
        self.assertIn('std_overlap', result)
        self.assertIn('max_overlap', result)
        self.assertIn('effective_orthogonality', result)

        # Check value ranges
        self.assertGreaterEqual(result['mean_overlap'], 0.0)
        self.assertLessEqual(result['mean_overlap'], 1.0)
        self.assertGreaterEqual(result['effective_orthogonality'], 0.0)
        self.assertLessEqual(result['effective_orthogonality'], 1.0)

    def test_vector_interference_orthogonal(self):
        """Test interference with orthogonal vectors."""
        # Create orthogonal vectors
        n_features = 3
        weight_matrix = torch.eye(n_features)

        result = self.metrics.compute_vector_interference(
            weight_matrix,
            normalize=True,
            exclude_diagonal=True
        )

        # Orthogonal vectors should have zero overlap
        self.assertAlmostEqual(result['mean_overlap'], 0.0, places=5)
        self.assertAlmostEqual(result['effective_orthogonality'], 1.0, places=5)

    def test_vector_interference_parallel(self):
        """Test interference with parallel vectors."""
        # Create parallel vectors
        base_vector = torch.randn(32)
        weight_matrix = base_vector.unsqueeze(0).repeat(10, 1)

        result = self.metrics.compute_vector_interference(
            weight_matrix,
            normalize=True,
            exclude_diagonal=True
        )

        # Parallel vectors should have perfect overlap
        self.assertAlmostEqual(result['mean_overlap'], 1.0, places=5)
        self.assertAlmostEqual(result['effective_orthogonality'], 0.0, places=5)

    def test_vector_interference_large_matrix(self):
        """Test interference with large matrix (batched computation)."""
        # Create large matrix to trigger batching
        n_features, n_dims = 2000, 512
        weight_matrix = torch.randn(n_features, n_dims)

        result = self.metrics.compute_vector_interference(
            weight_matrix,
            normalize=True,
            batch_size=500  # Force batching
        )

        # Check that batched computation works
        self.assertIn('mean_overlap', result)
        self.assertGreater(result['n_features'], 0)

    def test_feature_frequency_distribution(self):
        """Test feature frequency distribution computation."""
        # Create mock dataset
        dataset = [
            {'input_ids': torch.tensor([1, 2, 3, 1, 2, 1])},
            {'input_ids': torch.tensor([1, 4, 5, 1, 1, 2])}
        ]

        result = self.metrics.compute_feature_frequency_distribution(
            self.model,
            dataset,
            fit_power_law=True
        )

        # Check outputs
        self.assertIn('token_frequencies', result)
        self.assertIn('normalized_frequencies', result)
        self.assertIn('entropy', result)
        self.assertIn('gini_coefficient', result)

        # Check that frequencies sum to approximately 1
        total = sum(result['normalized_frequencies'])
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_feature_frequency_power_law(self):
        """Test power law fitting for frequency distribution."""
        # Create power law distribution
        vocab_size = 1000
        ranks = np.arange(1, vocab_size + 1)
        frequencies = 1.0 / (ranks ** 1.5)  # Known power law

        # Create mock dataset matching this distribution
        tokens = []
        for i, freq in enumerate(frequencies):
            count = int(freq * 10000)
            tokens.extend([i] * count)

        dataset = [{'input_ids': torch.tensor(tokens[:1000])}]

        result = self.metrics.compute_feature_frequency_distribution(
            self.model,
            dataset,
            fit_power_law=True
        )

        # Check power law fitting
        if 'power_law_alpha' in result and result['power_law_alpha'] is not None:
            # Should be close to 1.5
            self.assertGreater(result['power_law_r_squared'], 0.8)

    def test_superposition_strength(self):
        """Test superposition strength quantification."""
        result = self.metrics.compute_superposition_strength(
            self.model,
            self.test_batch,
            n_probes=5
        )

        # Check outputs
        self.assertIn('superposition_ratio', result)
        self.assertIn('effective_rank', result)
        self.assertIn('reconstruction_quality', result)
        self.assertIn('layer_metrics', result)

        # Check value ranges
        self.assertGreater(result['superposition_ratio'], 0.0)
        self.assertGreater(result['effective_rank'], 0.0)
        self.assertGreaterEqual(result['reconstruction_quality'], 0.0)
        self.assertLessEqual(result['reconstruction_quality'], 1.0)

    def test_dimensional_scaling(self):
        """Test dimensional scaling analysis."""
        # Create models of different sizes
        models_dict = {
            '32': DummyModel(hidden_dim=32).to(self.device),
            '64': DummyModel(hidden_dim=64).to(self.device),
            '128': DummyModel(hidden_dim=128).to(self.device)
        }

        result = self.metrics.analyze_dimensional_scaling(
            models_dict,
            self.test_batch
        )

        # Check outputs
        self.assertIn('model_sizes', result)
        self.assertIn('losses', result)

        # Should have results for each model
        self.assertEqual(len(result['model_sizes']), 3)
        self.assertEqual(len(result['losses']), 3)

        # Check for scaling law fitting
        if 'scaling_exponent' in result:
            self.assertIsInstance(result['scaling_exponent'], float)

    def test_feature_sparsity(self):
        """Test feature sparsity computation."""
        # Create test activations with known sparsity
        n_samples, n_features = 100, 50

        # Create sparse activations
        activations = torch.zeros(n_samples, n_features)
        # Only 10% of features active
        n_active = int(0.1 * n_features)
        for i in range(n_samples):
            active_idx = torch.randperm(n_features)[:n_active]
            activations[i, active_idx] = torch.randn(n_active)

        result = self.metrics.compute_feature_sparsity(
            activations,
            threshold=0.01,
            relative_threshold=False
        )

        # Check outputs
        self.assertIn('sparsity', result)
        self.assertIn('gini_coefficient', result)
        self.assertIn('l0_norm', result)
        self.assertIn('l1_l2_ratio', result)

        # Check that sparsity is approximately 0.9
        self.assertGreater(result['sparsity'], 0.85)
        self.assertLess(result['sparsity'], 0.95)

    def test_fit_scaling_law(self):
        """Test scaling law fitting."""
        # Generate data following known power law
        sizes = np.array([100, 200, 400, 800, 1600])
        alpha = 0.75
        constant = 10.0
        losses = constant * sizes ** (-alpha) + np.random.normal(0, 0.01, len(sizes))

        result = self.metrics.fit_scaling_law(
            sizes,
            losses,
            log_scale=True,
            return_confidence=True
        )

        # Check outputs
        self.assertIn('alpha', result)
        self.assertIn('constant', result)
        self.assertIn('r_squared', result)

        # Check that fitted values are close to true values
        self.assertAlmostEqual(result['alpha'], alpha, delta=0.1)
        self.assertAlmostEqual(result['constant'], constant, delta=2.0)
        self.assertGreater(result['r_squared'], 0.9)

    def test_representation_capacity(self):
        """Test representation capacity estimation."""
        result = self.metrics.compute_representation_capacity(
            self.model,
            self.test_batch,
            probe_dim=10,
            n_probes=5
        )

        # Check outputs
        self.assertIn('estimated_capacity', result)
        self.assertIn('probe_success_rate', result)
        self.assertIn('capacity_ratio', result)

        # Check value ranges
        self.assertGreaterEqual(result['probe_success_rate'], 0.0)
        self.assertLessEqual(result['probe_success_rate'], 1.0)
        self.assertGreater(result['capacity_ratio'], 0.0)

    def test_feature_emergence(self):
        """Test feature emergence tracking."""
        # Create checkpoints with increasing organization
        checkpoints = []
        for i in range(3):
            model = DummyModel(hidden_dim=64).to(self.device)
            # Modify weights to simulate training progress
            with torch.no_grad():
                for param in model.parameters():
                    param.data *= (i + 1) * 0.5
            checkpoints.append(model)

        result = self.metrics.analyze_feature_emergence(
            checkpoints,
            self.test_batch,
            checkpoint_steps=[0, 100, 200]
        )

        # Check outputs
        self.assertIn('overlap_evolution', result)
        self.assertIn('effective_dimension_evolution', result)
        self.assertIn('sparsity_evolution', result)
        self.assertIn('overlap_trend', result)
        self.assertIn('emergence_rate', result)

        # Should have trajectory for each checkpoint
        self.assertEqual(len(result['overlap_evolution']), 3)
        self.assertEqual(len(result['checkpoint_steps']), 3)

    def test_analyze_superposition_convenience(self):
        """Test convenience function for quick analysis."""
        # Get weight matrix from model
        weight_matrix = self.model.embedding.weight.data

        result = analyze_superposition(
            self.model,
            self.test_batch,
            weight_matrix=weight_matrix,
            verbose=False
        )

        # Check that all analyses were run
        self.assertIn('superposition', result)
        self.assertIn('interference', result)
        self.assertIn('sparsity', result)

        # Check nested structure
        self.assertIn('superposition_ratio', result['superposition'])
        self.assertIn('mean_overlap', result['interference'])
        self.assertIn('gini_coefficient', result['sparsity'])

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with empty weight matrix
        empty_matrix = torch.zeros(0, 10)
        result = self.metrics.compute_vector_interference(empty_matrix)
        self.assertEqual(result['n_features'], 0)

        # Test with single feature
        single_feature = torch.randn(1, 10)
        result = self.metrics.compute_vector_interference(single_feature)
        self.assertEqual(result['n_features'], 1)

        # Test with invalid sizes for scaling law
        result = self.metrics.fit_scaling_law([], [], log_scale=True)
        self.assertIn('error', result)

        # Test with NaN values
        sizes = [100, 200, 300]
        losses = [1.0, np.nan, 0.5]
        result = self.metrics.fit_scaling_law(sizes, losses, log_scale=True)
        # Should handle NaN gracefully

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Very small values
        small_matrix = torch.randn(10, 10) * 1e-10
        result = self.metrics.compute_vector_interference(small_matrix, normalize=True)
        self.assertFalse(np.isnan(result['mean_overlap']))

        # Very large values
        large_matrix = torch.randn(10, 10) * 1e10
        result = self.metrics.compute_vector_interference(large_matrix, normalize=True)
        self.assertFalse(np.isnan(result['mean_overlap']))

        # Mixed scales
        mixed_matrix = torch.randn(10, 10)
        mixed_matrix[0] *= 1e-10
        mixed_matrix[1] *= 1e10
        result = self.metrics.compute_vector_interference(mixed_matrix, normalize=True)
        self.assertFalse(np.isnan(result['mean_overlap']))


class TestIntegration(unittest.TestCase):
    """Integration tests with different model types."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = SuperpositionMetrics(device=self.device)

    def test_with_transformer_model(self):
        """Test with a transformer-like model."""
        # Create a more complex model
        class TransformerBlock(nn.Module):
            def __init__(self, hidden_dim):
                super().__init__()
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4)
                self.norm1 = nn.LayerNorm(hidden_dim)
                self.ffn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 4, hidden_dim)
                )
                self.norm2 = nn.LayerNorm(hidden_dim)

            def forward(self, x):
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + attn_out)
                # FFN
                ffn_out = self.ffn(x)
                x = self.norm2(x + ffn_out)
                return x

        class TransformerModel(nn.Module):
            def __init__(self, vocab_size=100, hidden_dim=64, n_layers=2):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, hidden_dim)
                self.blocks = nn.ModuleList([
                    TransformerBlock(hidden_dim) for _ in range(n_layers)
                ])
                self.output = nn.Linear(hidden_dim, vocab_size)
                self.config = type('Config', (), {'vocab_size': vocab_size})()

            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = x.transpose(0, 1)  # (seq, batch, hidden)
                for block in self.blocks:
                    x = block(x)
                x = x.transpose(0, 1)  # (batch, seq, hidden)
                return self.output(x)

        model = TransformerModel().to(self.device)
        test_batch = {
            'input_ids': torch.randint(0, 100, (4, 8)).to(self.device)
        }

        # Test superposition analysis
        result = self.metrics.compute_superposition_strength(
            model,
            test_batch,
            n_probes=3
        )

        self.assertIn('superposition_ratio', result)
        self.assertIn('layer_metrics', result)
        self.assertGreater(len(result['layer_metrics']), 0)


if __name__ == '__main__':
    unittest.main()
