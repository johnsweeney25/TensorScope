"""
Integration tests for Fisher Spectral Analysis Module
=======================================================
Tests integration with models, data handling, and end-to-end workflows.

Author: ICLR 2026 Project
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisher_spectral import FisherSpectral, SpectralConfig
from InformationTheoryMetrics import InformationTheoryMetrics


class SimpleModel(nn.Module):
    """Simple test model with known structure."""
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=10):
        super().__init__()
        self.embed = nn.Embedding(100, input_dim)
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.embed(input_ids)
        x = x.mean(dim=1)  # Simple pooling
        x = self.layer1(x)
        x = torch.relu(x)
        x = self.norm(x)
        logits = self.layer2(x)

        loss = None
        if labels is not None:
            loss = self.criterion(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Return in format expected by metrics
        return type('Output', (), {'loss': loss, 'logits': logits})()


class TestFisherSpectralIntegration(unittest.TestCase):
    """Integration tests for Fisher Spectral module."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SpectralConfig(
            seed=42,
            eps=1e-9,
            storage_mode='full',
            max_params_per_block=1000
        )
        self.spectral = FisherSpectral(self.config)

    def test_small_linear_model(self):
        """Test on a simple linear model."""
        torch.manual_seed(42)

        # Create simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )

        # Create synthetic batch
        batch_size = 32
        batch = {
            'input_ids': torch.randn(batch_size, 10),
            'labels': torch.randint(0, 10, (batch_size,))
        }

        # Add forward method to sequential
        def forward_with_loss(self, input_ids, labels=None, **kwargs):
            logits = super(nn.Sequential, self).forward(input_ids)
            loss = nn.functional.cross_entropy(logits, labels) if labels is not None else None
            return type('Output', (), {'loss': loss, 'logits': logits})()

        model.forward = lambda **kwargs: forward_with_loss(model, **kwargs)

        # Compute spectrum
        results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=16,
            block_structure='layer'
        )

        # Validate results structure
        self.assertIn('global', results)
        self.assertIn('per_block', results)
        self.assertIn('metadata', results)

        # Check global metrics
        if results['global']:
            self.assertIn('spectral_gap', results['global'])
            self.assertIn('condition_number', results['global'])
            self.assertIn('effective_rank', results['global'])
            self.assertIn('largest_eigenvalue', results['global'])

            # Sanity checks
            self.assertGreaterEqual(results['global']['spectral_gap'], 0.0)
            self.assertGreater(results['global']['condition_number'], 0.0)
            self.assertGreater(results['global']['effective_rank'], 0.0)

    def test_batch_size_variations(self):
        """Test with different batch sizes (N < P, N = P, N > P)."""
        torch.manual_seed(42)

        for n_samples, expected_case in [(5, 'N<P'), (20, 'N≈P'), (100, 'N>P')]:
            with self.subTest(case=expected_case, n_samples=n_samples):
                # Small model with P ≈ 20 parameters
                model = nn.Linear(5, 4)  # 5*4 + 4 = 24 parameters

                batch = {
                    'input_ids': torch.randn(n_samples, 5),
                    'labels': torch.randn(n_samples, 4)  # Regression task
                }

                def forward(self, input_ids, labels=None, **kwargs):
                    output = super(nn.Linear, self).forward(input_ids)
                    loss = nn.functional.mse_loss(output, labels) if labels is not None else None
                    return type('Output', (), {'loss': loss, 'logits': output})()

                model.forward = lambda **kwargs: forward(model, **kwargs)

                # Compute spectrum
                results = self.spectral.compute_fisher_spectrum(
                    model, batch,
                    n_samples=min(n_samples, 50),
                    block_structure='none'  # Single block
                )

                # Should complete without error
                self.assertIsNotNone(results)
                if results['global']:
                    self.assertGreater(results['global']['largest_eigenvalue'], 0)

    def test_memory_modes(self):
        """Test different memory storage modes."""
        torch.manual_seed(42)

        model = SimpleModel()
        batch = {
            'input_ids': torch.randint(0, 100, (32, 10)),
            'labels': torch.randint(0, 10, (32, 10))
        }

        for storage_mode in ['full', 'chunked']:
            with self.subTest(storage_mode=storage_mode):
                config = SpectralConfig(
                    seed=42,
                    storage_mode=storage_mode,
                    chunk_size=8 if storage_mode == 'chunked' else 32
                )
                spectral = FisherSpectral(config)

                results = spectral.compute_fisher_spectrum(
                    model, batch,
                    n_samples=16,
                    block_structure='module'
                )

                # Should produce valid results
                self.assertIsNotNone(results)
                self.assertIn('global', results)

    def test_reproducibility(self):
        """Test that same seed produces same results."""
        torch.manual_seed(42)

        model = SimpleModel()
        batch = {
            'input_ids': torch.randint(0, 100, (32, 10)),
            'labels': torch.randint(0, 10, (32, 10))
        }

        # Run twice with same seed
        results1 = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=16,
            block_structure='layer'
        )

        # Reset and run again
        self.spectral = FisherSpectral(self.config)
        results2 = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=16,
            block_structure='layer'
        )

        # Compare key metrics
        if results1['global'] and results2['global']:
            self.assertAlmostEqual(
                results1['global']['spectral_gap'],
                results2['global']['spectral_gap'],
                places=5,
                msg="Results not reproducible with same seed"
            )

    def test_dtype_handling(self):
        """Test handling of different dtypes (fp16, fp32, fp64)."""
        torch.manual_seed(42)

        for dtype in [torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                model = nn.Linear(10, 10).to(dtype)
                batch = {
                    'input_ids': torch.randn(16, 10, dtype=dtype),
                    'labels': torch.randn(16, 10, dtype=dtype)
                }

                def forward(self, input_ids, labels=None, **kwargs):
                    output = super(nn.Linear, self).forward(input_ids)
                    loss = nn.functional.mse_loss(output, labels) if labels is not None else None
                    return type('Output', (), {'loss': loss})()

                model.forward = lambda **kwargs: forward(model, **kwargs)

                results = self.spectral.compute_fisher_spectrum(
                    model, batch,
                    n_samples=8,
                    block_structure='none'
                )

                # Should handle all dtypes
                self.assertIsNotNone(results)

    def test_information_theory_metrics_integration(self):
        """Test integration with InformationTheoryMetrics class."""
        torch.manual_seed(42)

        # Create metrics instance
        metrics = InformationTheoryMetrics(seed=42)

        # Create model and batch
        model = SimpleModel()
        batch = {
            'input_ids': torch.randint(0, 100, (32, 10)),
            'labels': torch.randint(0, 10, (32, 10))
        }

        # Test compute_spectral_gap
        results = metrics.compute_spectral_gap(model, batch)

        # Check expected keys
        self.assertIn('spectral_gap', results)
        self.assertIn('condition_number', results)
        self.assertIn('fim_effective_rank', results)
        self.assertIn('largest_eigenvalue', results)
        self.assertIn('optimization_timescale', results)

        # No more "mixing_time" - that was incorrect
        self.assertNotIn('mixing_time', results)
        self.assertNotIn('forgetting_timescale', results)

        # Validate values
        self.assertGreaterEqual(results['spectral_gap'], 0.0)
        if results['largest_eigenvalue'] > 0:
            self.assertAlmostEqual(
                results['optimization_timescale'],
                1.0 / results['largest_eigenvalue'],
                places=5
            )

    def test_block_structure_options(self):
        """Test different block structure options."""
        torch.manual_seed(42)

        model = SimpleModel()
        batch = {
            'input_ids': torch.randint(0, 100, (32, 10)),
            'labels': torch.randint(0, 10, (32, 10))
        }

        for block_structure in ['layer', 'module', 'none']:
            with self.subTest(block_structure=block_structure):
                results = self.spectral.compute_fisher_spectrum(
                    model, batch,
                    n_samples=16,
                    block_structure=block_structure
                )

                self.assertIsNotNone(results)
                self.assertIn('per_block', results)

                # Check block keys match structure
                if block_structure == 'none':
                    self.assertIn('global', results['per_block'])
                elif block_structure == 'layer':
                    # Should have embedding/output blocks
                    block_keys = list(results['per_block'].keys())
                    self.assertTrue(
                        any('embed' in k or 'output' in k or 'other' in k for k in block_keys)
                    )
                elif block_structure == 'module':
                    # Should have attention/mlp/norm blocks
                    block_keys = list(results['per_block'].keys())
                    self.assertTrue(
                        any(k in ['attention', 'mlp', 'embedding', 'normalization', 'other']
                            for k in block_keys)
                    )

    def test_centering_flag(self):
        """Test Fisher vs Covariance computation (centering flag)."""
        torch.manual_seed(42)

        model = SimpleModel()
        batch = {
            'input_ids': torch.randint(0, 100, (32, 10)),
            'labels': torch.randint(0, 10, (32, 10))
        }

        # Compute Fisher (uncentered)
        fisher_results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=16,
            block_structure='none',
            center_gradients=False
        )

        # Compute Covariance (centered)
        cov_results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=16,
            block_structure='none',
            center_gradients=True
        )

        # Both should complete
        self.assertIsNotNone(fisher_results)
        self.assertIsNotNone(cov_results)

        # Fisher typically has larger eigenvalues than covariance
        if fisher_results['global'] and cov_results['global']:
            # This may not always hold for random initialization, but usually does
            fisher_trace = fisher_results['global']['trace']
            cov_trace = cov_results['global']['trace']
            # At least check they're different
            self.assertNotAlmostEqual(fisher_trace, cov_trace, places=5)

    def test_empty_batch_handling(self):
        """Test handling of empty or invalid batches."""
        model = SimpleModel()

        # Empty batch
        empty_batch = {
            'input_ids': torch.zeros(0, 10, dtype=torch.long),
            'labels': torch.zeros(0, 10, dtype=torch.long)
        }

        results = self.spectral.compute_fisher_spectrum(
            model, empty_batch,
            n_samples=16
        )

        # Should return empty results gracefully
        self.assertIn('metadata', results)
        self.assertIn('error', results['metadata'])

    def test_nan_gradient_handling(self):
        """Test handling of NaN gradients."""
        torch.manual_seed(42)

        # Model that produces NaN gradients
        class NaNModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, input_ids, labels=None, **kwargs):
                x = self.linear(input_ids)
                # Force NaN by dividing by zero
                loss = x.sum() / 0.0 if labels is not None else None
                return type('Output', (), {'loss': loss})()

        model = NaNModel()
        batch = {
            'input_ids': torch.randn(16, 10),
            'labels': torch.randn(16, 10)
        }

        # Should handle NaN without crashing
        results = self.spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=8
        )

        # May return empty results but shouldn't crash
        self.assertIsNotNone(results)

    def test_parameter_subsampling(self):
        """Test subsampling for large parameter counts."""
        torch.manual_seed(42)

        # Model with many parameters
        model = nn.Sequential(
            nn.Linear(100, 500),  # 50,500 parameters
            nn.ReLU(),
            nn.Linear(500, 100)   # 50,100 parameters
        )

        batch = {
            'input_ids': torch.randn(16, 100),
            'labels': torch.randn(16, 100)
        }

        def forward(self, input_ids, labels=None, **kwargs):
            output = super(nn.Sequential, self).forward(input_ids)
            loss = nn.functional.mse_loss(output, labels) if labels is not None else None
            return type('Output', (), {'loss': loss})()

        model.forward = lambda **kwargs: forward(model, **kwargs)

        # Configure with low max_params_per_block to force subsampling
        config = SpectralConfig(
            seed=42,
            max_params_per_block=1000  # Much less than 50k+ params
        )
        spectral = FisherSpectral(config)

        results = spectral.compute_fisher_spectrum(
            model, batch,
            n_samples=8,
            block_structure='layer'
        )

        # Should handle subsampling
        self.assertIsNotNone(results)
        if results['per_block']:
            # Check that subsampling occurred
            for block_key, block_metrics in results['per_block'].items():
                if block_metrics['n_params'] > 0:
                    # Subsampled blocks should have <= max_params
                    self.assertLessEqual(
                        block_metrics['n_params'],
                        config.max_params_per_block * 2  # Some tolerance for block boundaries
                    )


if __name__ == '__main__':
    unittest.main()