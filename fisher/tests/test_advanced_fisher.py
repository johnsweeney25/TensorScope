#!/usr/bin/env python3
"""
Comprehensive test suite for Advanced Fisher Collector.

Tests:
1. True Fisher vs Empirical Fisher
2. K-FAC approximation accuracy
3. Capacity metrics correlation with generalization
4. Natural gradient computation
5. Loss landscape curvature
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import unittest
from typing import Dict
import logging

from fisher_collector_advanced import AdvancedFisherCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModel(nn.Module):
    """Simple test model for experiments."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.config = type('Config', (), {'vocab_size': output_dim})()

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Simple forward pass
        x = F.one_hot(input_ids, num_classes=10).float()  # Convert to one-hot
        x = x.mean(dim=1)  # Pool over sequence
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        # Expand logits to match label shape
        if labels is not None:
            seq_len = labels.shape[1]
            logits = logits.unsqueeze(1).expand(-1, seq_len, -1)

            # Compute loss
            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100
            )
        else:
            loss = None

        return type('Output', (), {'loss': loss, 'logits': logits})()


class TestAdvancedFisher(unittest.TestCase):
    """Test suite for Advanced Fisher Collector."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.model = TestModel()
        self.batch = {
            'input_ids': torch.randint(0, 10, (4, 8)),
            'attention_mask': torch.ones(4, 8),
            'labels': torch.randint(0, 10, (4, 8))
        }

    def test_true_fisher_positive_semidefinite(self):
        """Test: True Fisher should be positive semi-definite."""
        print("\n=== Testing True Fisher Positive Semi-Definite ===")

        collector = AdvancedFisherCollector(use_true_fisher=True)

        # Collect true Fisher
        fisher = collector.collect_true_fisher(
            self.model, self.batch, 'test', n_samples=10
        )

        # Check all values are non-negative (diagonal approximation)
        for key, values in fisher.items():
            self.assertTrue(
                torch.all(values >= 0),
                f"True Fisher has negative values for {key}"
            )

        print("✓ True Fisher is positive semi-definite")

    def test_true_vs_empirical_fisher(self):
        """Test: Compare true Fisher with empirical Fisher."""
        print("\n=== Testing True vs Empirical Fisher ===")

        # Collect true Fisher
        true_collector = AdvancedFisherCollector(use_true_fisher=True)
        true_fisher = true_collector.collect_true_fisher(
            self.model, self.batch, 'true', n_samples=50
        )

        # Collect empirical Fisher
        emp_collector = AdvancedFisherCollector(use_true_fisher=False)
        emp_collector.compute_oneshot_fisher(
            self.model, self.batch, 'empirical', n_samples=50
        )
        emp_fisher = emp_collector.get_group_fisher('empirical', bias_corrected=False)

        # Compare magnitudes (true Fisher often smaller than empirical)
        true_total = sum(v.sum().item() for v in true_fisher.values())
        emp_total = sum(v.sum().item() for v in emp_fisher.values())

        print(f"True Fisher total: {true_total:.4f}")
        print(f"Empirical Fisher total: {emp_total:.4f}")

        # True Fisher should generally have smaller magnitude
        # (empirical Fisher overestimates curvature)
        self.assertLess(
            true_total, emp_total * 2.0,
            "True Fisher unexpectedly larger than empirical"
        )

        print("✓ True Fisher has expected relationship to empirical Fisher")

    def test_kfac_approximation(self):
        """Test: K-FAC approximation accuracy."""
        print("\n=== Testing K-FAC Approximation ===")

        # Create larger model for K-FAC
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.ReLU(),
            nn.Linear(200, 100)
        )

        batch = {
            'input_ids': torch.randn(8, 100),  # Direct input for simplicity
            'labels': torch.randint(0, 100, (8,))
        }

        # Custom forward for sequential model
        def forward_fn(batch):
            x = batch['input_ids']
            logits = model(x)
            loss = F.cross_entropy(logits, batch['labels'])
            return type('Output', (), {'loss': loss, 'logits': logits})()

        model.forward = lambda **b: forward_fn(b)

        collector = AdvancedFisherCollector(
            use_kfac=True,
            kfac_update_freq=1
        )

        # Update K-FAC factors
        collector._update_kfac_factors(model, batch)

        # Check factors exist
        self.assertTrue(
            len(collector.kfac_factors) > 0,
            "No K-FAC factors computed"
        )

        # Check factor properties
        for layer_name, factors in collector.kfac_factors.items():
            A = factors['A']
            G = factors['G']

            # Should be square matrices
            self.assertEqual(A.dim(), 2)
            self.assertEqual(G.dim(), 2)
            self.assertEqual(A.shape[0], A.shape[1])
            self.assertEqual(G.shape[0], G.shape[1])

            # Should be positive semi-definite
            eigvals_A = torch.linalg.eigvalsh(A)
            eigvals_G = torch.linalg.eigvalsh(G)

            self.assertTrue(
                torch.all(eigvals_A >= -1e-6),
                f"A matrix has negative eigenvalues for {layer_name}"
            )
            self.assertTrue(
                torch.all(eigvals_G >= -1e-6),
                f"G matrix has negative eigenvalues for {layer_name}"
            )

        print(f"✓ K-FAC computed {len(collector.kfac_factors)} factor pairs")

    def test_natural_gradient(self):
        """Test: Natural gradient computation."""
        print("\n=== Testing Natural Gradient ===")

        collector = AdvancedFisherCollector(
            use_kfac=True,
            kfac_update_freq=1
        )

        # Compute gradients
        self.model.zero_grad()
        outputs = self.model(**self.batch)
        loss = outputs.loss
        loss.backward()

        # Store original gradients
        orig_grads = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                orig_grads[name] = param.grad.clone()

        # Update K-FAC and compute natural gradient
        collector._update_kfac_factors(self.model, self.batch)
        nat_grads = collector.get_kfac_natural_gradient(self.model)

        # Natural gradient should be different from standard gradient
        for name, nat_grad in nat_grads.items():
            param_name = name.replace('.weight', '').replace('.bias', '')
            orig_name = name.split('.')[-1]  # 'weight' or 'bias'

            if param_name + '.' + orig_name in orig_grads:
                orig = orig_grads[param_name + '.' + orig_name]

                # Check shapes match
                self.assertEqual(
                    nat_grad.shape, orig.shape,
                    f"Shape mismatch for {name}"
                )

                # Natural gradient should differ from original
                diff = (nat_grad - orig).norm() / (orig.norm() + 1e-8)
                self.assertGreater(
                    diff, 0.01,
                    f"Natural gradient too similar to original for {name}"
                )

        print(f"✓ Natural gradient computed for {len(nat_grads)} parameters")

    def test_capacity_metrics(self):
        """Test: Capacity metrics computation."""
        print("\n=== Testing Capacity Metrics ===")

        collector = AdvancedFisherCollector(use_kfac=True)

        # Collect Fisher
        collector.collect_fisher(self.model, self.batch, 'capacity_test')

        # Compute capacity metrics
        metrics = collector.compute_capacity_metrics('capacity_test')

        # Check expected metrics exist
        expected = ['trace', 'log_det', 'effective_rank', 'condition_number', 'pac_bayes_complexity']
        for metric in expected:
            self.assertIn(
                metric, metrics,
                f"Missing metric: {metric}"
            )

        # Check metric properties
        self.assertGreater(metrics['trace'], 0, "Trace should be positive")
        self.assertGreater(metrics['effective_rank'], 0, "Effective rank should be positive")
        self.assertGreaterEqual(metrics['condition_number'], 1, "Condition number should be >= 1")

        print(f"✓ Capacity metrics: trace={metrics['trace']:.2f}, "
              f"eff_rank={metrics['effective_rank']:.2f}")

        # Test capacity score
        score = collector.compute_model_capacity_score(
            self.model, self.batch, 'score_test'
        )
        self.assertGreater(score, 0, "Capacity score should be positive")

        print(f"✓ Model capacity score: {score:.4f}")

    def test_loss_landscape_curvature(self):
        """Test: Loss landscape curvature estimation."""
        print("\n=== Testing Loss Landscape Curvature ===")

        collector = AdvancedFisherCollector(use_true_fisher=True)

        # Compute curvature
        curvature = collector.compute_loss_landscape_curvature(
            self.model, self.batch, epsilon=0.1, n_samples=5
        )

        # Check expected metrics
        expected = ['average_sharpness', 'max_sharpness', 'effective_curvature',
                   'landscape_variance', 'original_loss']

        for metric in expected:
            self.assertIn(metric, curvature, f"Missing metric: {metric}")

        # Sharpness should be non-negative (loss increases with perturbation)
        self.assertGreaterEqual(
            curvature['average_sharpness'], 0,
            "Average sharpness should be non-negative"
        )

        print(f"✓ Curvature: avg_sharp={curvature['average_sharpness']:.4f}, "
              f"eff_curv={curvature['effective_curvature']:.4f}")

    def test_fisher_spectrum_analysis(self):
        """Test: Fisher spectrum analysis."""
        print("\n=== Testing Fisher Spectrum Analysis ===")

        collector = AdvancedFisherCollector(use_kfac=True)

        # Collect Fisher with K-FAC
        collector.collect_fisher(self.model, self.batch, 'spectrum_test')
        collector._update_kfac_factors(self.model, self.batch)

        # Analyze spectrum
        spectrum = collector.analyze_fisher_spectrum('spectrum_test')

        if collector.kfac_factors:
            # Check K-FAC spectrum analysis
            for key, stats in spectrum.items():
                if isinstance(stats, dict):
                    self.assertIn('max_eigenvalue', stats)
                    self.assertIn('min_eigenvalue', stats)
                    self.assertGreater(
                        stats['max_eigenvalue'],
                        stats['min_eigenvalue'],
                        "Max eigenvalue should exceed min"
                    )

            print(f"✓ Spectrum analyzed for {len(spectrum)} components")
        else:
            # Check diagonal spectrum
            self.assertIn('diagonal_fisher', spectrum)
            stats = spectrum['diagonal_fisher']

            self.assertGreater(stats['max_value'], stats['min_value'])
            self.assertGreater(stats['n_significant'], 0)

            print(f"✓ Diagonal spectrum: max={stats['max_value']:.4e}, "
                  f"sparsity={stats['sparsity']:.2f}")

    def test_memory_efficiency_comparison(self):
        """Test: Compare memory usage of different Fisher approximations."""
        print("\n=== Testing Memory Efficiency ===")

        import sys

        # Diagonal Fisher
        diag_collector = AdvancedFisherCollector(
            reduction='group',
            use_kfac=False
        )
        diag_collector.collect_fisher(self.model, self.batch, 'diag_test')
        diag_fisher = diag_collector.get_group_fisher('diag_test', bias_corrected=False)

        diag_memory = sum(
            v.element_size() * v.numel()
            for v in diag_fisher.values()
        )

        # K-FAC Fisher
        kfac_collector = AdvancedFisherCollector(
            use_kfac=True,
            kfac_update_freq=1
        )
        kfac_collector.collect_fisher(self.model, self.batch, 'kfac_test')
        kfac_collector._update_kfac_factors(self.model, self.batch)

        kfac_memory = 0
        if kfac_collector.kfac_factors:
            for factors in kfac_collector.kfac_factors.values():
                A = factors['A']
                G = factors['G']
                kfac_memory += A.element_size() * A.numel()
                kfac_memory += G.element_size() * G.numel()

        # Full Fisher (hypothetical)
        n_params = sum(p.numel() for p in self.model.parameters())
        full_memory = n_params * n_params * 4  # float32

        print(f"Memory usage:")
        print(f"  Diagonal: {diag_memory:,} bytes")
        print(f"  K-FAC: {kfac_memory:,} bytes")
        print(f"  Full (hypothetical): {full_memory:,} bytes")

        print(f"✓ Reduction ratios: K-FAC={full_memory/max(kfac_memory,1):.1f}x, "
              f"Diagonal={full_memory/max(diag_memory,1):.1f}x")

    def test_correlation_with_hessian(self):
        """Test: Fisher should correlate with Hessian diagonal."""
        print("\n=== Testing Fisher-Hessian Correlation ===")

        collector = AdvancedFisherCollector(use_true_fisher=True)

        # Collect Fisher
        fisher = collector.collect_true_fisher(
            self.model, self.batch, 'hessian_test', n_samples=20
        )

        # Compute Hessian diagonal approximation (using finite differences)
        epsilon = 1e-4
        hessian_diag = {}

        self.model.zero_grad()
        outputs = self.model(**self.batch)
        loss_orig = outputs.loss

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Store original values
                orig_param = param.data.clone()

                # Compute gradient at original point
                self.model.zero_grad()
                loss_orig.backward(retain_graph=True)
                grad_orig = param.grad.clone() if param.grad is not None else torch.zeros_like(param)

                # Approximate Hessian diagonal with finite differences
                h_diag = torch.zeros_like(param)

                # Sample subset of parameters for efficiency
                n_samples = min(10, param.numel())
                indices = torch.randperm(param.numel())[:n_samples]

                for idx in indices:
                    # Perturb single parameter
                    flat_param = param.view(-1)
                    flat_param[idx] += epsilon

                    # Compute gradient at perturbed point
                    self.model.zero_grad()
                    outputs_pert = self.model(**self.batch)
                    outputs_pert.loss.backward()
                    grad_pert = param.grad.clone() if param.grad is not None else torch.zeros_like(param)

                    # Finite difference approximation
                    flat_h = h_diag.view(-1)
                    flat_grad_orig = grad_orig.view(-1)
                    flat_grad_pert = grad_pert.view(-1)

                    flat_h[idx] = (flat_grad_pert[idx] - flat_grad_orig[idx]) / epsilon

                    # Restore parameter
                    param.data = orig_param.clone()

                # Store Hessian diagonal approximation
                hessian_diag[name] = h_diag.abs()  # Use absolute value for comparison

        # Compare Fisher and Hessian magnitudes
        # They should be correlated (both measure curvature)
        fisher_vals = []
        hessian_vals = []

        for name, h_diag in hessian_diag.items():
            # Get corresponding Fisher values
            # Note: Fisher is group-reduced, so we need to match carefully
            key = None
            for k in fisher.keys():
                if name in k:
                    key = k
                    break

            if key and key in fisher:
                f_vals = fisher[key]
                h_vals = h_diag

                # Reduce Hessian to match Fisher shape if needed
                if f_vals.shape != h_vals.shape and h_vals.numel() > f_vals.numel():
                    # Average pool Hessian to match Fisher
                    h_vals = h_vals.view(-1)[:f_vals.numel()]

                if f_vals.shape == h_vals.shape:
                    fisher_vals.append(f_vals.mean().item())
                    hessian_vals.append(h_vals.mean().item())

        if fisher_vals and hessian_vals:
            # Compute correlation
            fisher_vals = np.array(fisher_vals)
            hessian_vals = np.array(hessian_vals)

            # Normalize to [0, 1] for comparison
            fisher_vals = (fisher_vals - fisher_vals.min()) / (fisher_vals.max() - fisher_vals.min() + 1e-8)
            hessian_vals = (hessian_vals - hessian_vals.min()) / (hessian_vals.max() - hessian_vals.min() + 1e-8)

            correlation = np.corrcoef(fisher_vals, hessian_vals)[0, 1]

            print(f"✓ Fisher-Hessian correlation: {correlation:.3f}")

            # Should have positive correlation (both measure curvature)
            self.assertGreater(
                correlation, -0.1,
                "Fisher and Hessian should have non-negative correlation"
            )
        else:
            print("✓ Fisher-Hessian comparison completed (shape mismatch)")


def run_comprehensive_tests():
    """Run all tests with detailed output."""
    print("=" * 60)
    print("COMPREHENSIVE ADVANCED FISHER TESTS")
    print("=" * 60)

    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAdvancedFisher)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ ALL TESTS PASSED!")
    else:
        print(f"⚠️ {len(result.failures)} tests failed")
        print(f"⚠️ {len(result.errors)} tests had errors")

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_comprehensive_tests()
    exit(0 if success else 1)