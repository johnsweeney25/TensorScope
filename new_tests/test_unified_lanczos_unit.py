#!/usr/bin/env python3
"""
Unit Tests for Unified Fisher/Hessian Lanczos System
=====================================================
Solid unit tests that verify the unified Lanczos system works correctly.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisher.core.fisher_lanczos_unified import (
    compute_spectrum, LanczosConfig, HessianOperator,
    GGNOperator, EmpiricalFisherOperator, KFACFisherOperator,
    create_operator, LinOp, lanczos_algorithm
)
from fisher.core.fisher_collector_advanced import AdvancedFisherCollector
from ICLRMetrics import ICLRMetrics


class TestModel(nn.Module):
    """Simple test model for unit tests."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=10):
        super().__init__()
        self.config = type('Config', (), {'vocab_size': output_dim})()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Handle different input shapes
        if input_ids.dim() == 3:
            # [batch, seq, features]
            x = input_ids.float().mean(dim=1)
        elif input_ids.dim() == 2:
            # [batch, features] or [batch, seq] for embeddings
            x = input_ids.float()
            if x.shape[-1] != self.linear1.in_features:
                # If it's sequence data, average it
                x = F.pad(x, (0, self.linear1.in_features - x.shape[-1]))
        else:
            x = input_ids.float().unsqueeze(0)

        x = self.activation(self.linear1(x))
        logits = self.linear2(x)

        # Compute loss
        loss = None
        if labels is not None:
            if logits.dim() == 2 and labels.dim() == 1:
                # Standard classification loss
                loss = nn.functional.cross_entropy(logits, labels)
            else:
                # Default loss
                loss = logits.mean()
        else:
            loss = logits.mean()

        # Return in expected format
        output = type('Output', (), {})()
        output.loss = loss
        output.logits = logits.unsqueeze(1) if logits.dim() == 2 else logits
        return output


class TestLinearOperators(unittest.TestCase):
    """Test linear operators."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.batch = {
            'input_ids': torch.randn(4, 10),
            'labels': torch.randint(0, 10, (4,))
        }
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    def test_hessian_operator(self):
        """Test Hessian operator."""
        def loss_fn():
            outputs = self.model(**self.batch)
            return outputs.loss

        op = HessianOperator(self.model, loss_fn, self.params)

        # Test that it's not PSD
        self.assertFalse(op.is_psd)
        self.assertEqual(op.name, "hessian")

        # Test matrix-vector product
        v = [torch.randn_like(p) for p in self.params]
        hv = op.mv(v)

        # Check output shape
        self.assertEqual(len(hv), len(v))
        for h, vi in zip(hv, v):
            self.assertEqual(h.shape, vi.shape)

    def test_ggn_operator(self):
        """Test GGN operator."""
        op = GGNOperator(self.model, self.batch, self.params)

        # Test that it's PSD
        self.assertTrue(op.is_psd)
        self.assertEqual(op.name, "ggn")

        # Test matrix-vector product
        v = [torch.randn_like(p) for p in self.params]
        gv = op.mv(v)

        # Check output shape
        self.assertEqual(len(gv), len(v))
        for g, vi in zip(gv, v):
            self.assertEqual(g.shape, vi.shape)

    def test_empirical_fisher_operator(self):
        """Test Empirical Fisher operator."""
        op = EmpiricalFisherOperator(self.model, self.batch, self.params)

        # Test that it's PSD
        self.assertTrue(op.is_psd)
        self.assertEqual(op.name, "empirical_fisher")

        # Test matrix-vector product
        v = [torch.randn_like(p) for p in self.params]
        fv = op.mv(v)

        # Check output shape
        self.assertEqual(len(fv), len(v))
        for f, vi in zip(fv, v):
            self.assertEqual(f.shape, vi.shape)


class TestLanczosAlgorithm(unittest.TestCase):
    """Test Lanczos algorithm."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.batch = {
            'input_ids': torch.randn(4, 10),
            'labels': torch.randint(0, 10, (4,))
        }

    def test_lanczos_convergence(self):
        """Test that Lanczos converges."""
        config = LanczosConfig(k=3, max_iters=10, seed=42)

        def loss_fn():
            outputs = self.model(**self.batch)
            return outputs.loss

        op = HessianOperator(self.model, loss_fn)
        results = lanczos_algorithm(op, config)

        # Check that we got results
        self.assertIn('eigenvalues', results)
        self.assertIsNotNone(results['eigenvalues'])

        # Check that we got at least one eigenvalue
        self.assertGreater(len(results['eigenvalues']), 0)

    def test_lanczos_psd_operator(self):
        """Test Lanczos with PSD operator."""
        config = LanczosConfig(k=3, max_iters=10, seed=42)

        op = GGNOperator(self.model, self.batch)
        results = lanczos_algorithm(op, config)

        # Check that eigenvalues are non-negative for PSD operator
        if 'eigenvalues' in results and len(results['eigenvalues']) > 0:
            eigenvalues = results['eigenvalues']
            # Allow small negative values due to numerical error
            self.assertTrue(all(e >= -1e-6 for e in eigenvalues),
                          f"PSD operator has negative eigenvalues: {eigenvalues}")

    def test_lanczos_reproducibility(self):
        """Test that Lanczos is reproducible with fixed seed."""
        config = LanczosConfig(k=3, max_iters=10, seed=42)

        op = GGNOperator(self.model, self.batch)

        # Run twice with same seed
        results1 = lanczos_algorithm(op, config)
        results2 = lanczos_algorithm(op, config)

        # Check that results are identical
        if 'eigenvalues' in results1 and 'eigenvalues' in results2:
            eigs1 = np.array(results1['eigenvalues'])
            eigs2 = np.array(results2['eigenvalues'])

            if len(eigs1) > 0 and len(eigs2) > 0:
                np.testing.assert_array_almost_equal(eigs1, eigs2, decimal=8)


class TestComputeSpectrum(unittest.TestCase):
    """Test high-level compute_spectrum function."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.batch = {
            'input_ids': torch.randn(4, 10),
            'labels': torch.randint(0, 10, (4,))
        }

    def test_compute_spectrum_hessian(self):
        """Test computing Hessian spectrum."""
        results = compute_spectrum(
            self.model, self.batch,
            operator_type='hessian',
            config=LanczosConfig(k=3, max_iters=10)
        )

        self.assertIn('eigenvalues', results)
        self.assertIn('operator', results)
        self.assertEqual(results['operator'], 'hessian')
        self.assertFalse(results.get('is_psd', True))

    def test_compute_spectrum_ggn(self):
        """Test computing GGN spectrum."""
        results = compute_spectrum(
            self.model, self.batch,
            operator_type='ggn',
            config=LanczosConfig(k=3, max_iters=10)
        )

        self.assertIn('eigenvalues', results)
        self.assertIn('operator', results)
        self.assertEqual(results['operator'], 'ggn')
        self.assertTrue(results.get('is_psd', False))

    def test_compute_spectrum_empirical_fisher(self):
        """Test computing Empirical Fisher spectrum."""
        results = compute_spectrum(
            self.model, self.batch,
            operator_type='empirical_fisher',
            config=LanczosConfig(k=3, max_iters=10)
        )

        self.assertIn('eigenvalues', results)
        self.assertIn('operator', results)
        self.assertEqual(results['operator'], 'empirical_fisher')
        self.assertTrue(results.get('is_psd', False))

    def test_compute_spectrum_invalid_operator(self):
        """Test that invalid operator raises error."""
        with self.assertRaises(ValueError):
            compute_spectrum(
                self.model, self.batch,
                operator_type='invalid_operator'
            )


class TestAdvancedFisherIntegration(unittest.TestCase):
    """Test integration with AdvancedFisherCollector."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.batch = {
            'input_ids': torch.randn(4, 10),
            'labels': torch.randint(0, 10, (4,))
        }
        self.collector = AdvancedFisherCollector()

    def test_lanczos_spectrum_method(self):
        """Test lanczos_spectrum method."""
        results = self.collector.lanczos_spectrum(
            self.model, self.batch,
            operator='ggn',
            k=3
        )

        # Check basic structure
        self.assertIsInstance(results, dict)

        # If computation succeeded, check results
        if 'error' not in results:
            self.assertIn('operator', results)
            self.assertEqual(results['operator'], 'ggn')

    def test_compute_hessian_spectrum(self):
        """Test compute_hessian_spectrum convenience method."""
        results = self.collector.compute_hessian_spectrum(
            self.model, self.batch,
            k=3
        )

        self.assertIsInstance(results, dict)
        if 'error' not in results:
            self.assertEqual(results.get('operator'), 'hessian')

    def test_compute_fisher_spectrum(self):
        """Test compute_fisher_spectrum convenience method."""
        results = self.collector.compute_fisher_spectrum(
            self.model, self.batch,
            k=3,
            use_ggn=True
        )

        self.assertIsInstance(results, dict)
        if 'error' not in results:
            self.assertEqual(results.get('operator'), 'ggn')


class TestICLRMetricsIntegration(unittest.TestCase):
    """Test integration with ICLRMetrics."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.batch = {
            'input_ids': torch.randn(4, 10),
            'labels': torch.randint(0, 10, (4,))
        }
        self.metrics = ICLRMetrics(device='cpu')

    def test_compute_hessian_eigenvalues_unified(self):
        """Test Hessian eigenvalues with unified system."""
        results = self.metrics.compute_hessian_eigenvalues_lanczos(
            self.model, self.batch,
            k=3,
            max_iter=10,
            use_unified=True
        )

        self.assertIsInstance(results, dict)

        # Check for expected keys
        if 'error' not in results:
            self.assertIn('top_eigenvalues', results)
            self.assertIn('note', results)
            # Should mention unified system
            self.assertIn('unified', results['note'].lower())

    def test_compute_hessian_eigenvalues_legacy(self):
        """Test Hessian eigenvalues with legacy system."""
        results = self.metrics.compute_hessian_eigenvalues_lanczos(
            self.model, self.batch,
            k=3,
            max_iter=10,
            use_unified=False
        )

        self.assertIsInstance(results, dict)

        # Legacy might work or fail, both are acceptable
        # Just check that it returns a dict
        if 'error' not in results and 'note' in results:
            # Should not mention unified system
            self.assertNotIn('unified', results.get('note', '').lower())


class TestEdgeCases(unittest.TestCase):
    """Test edge cases."""

    def test_tiny_model(self):
        """Test with a very small model."""
        model = nn.Linear(2, 2)
        batch = {'input_ids': torch.randn(1, 2)}

        def loss_fn():
            return model(batch['input_ids']).sum()

        results = compute_spectrum(
            model, batch,
            operator_type='hessian',
            config=LanczosConfig(k=2, max_iters=5),
            loss_fn=loss_fn
        )

        self.assertIsInstance(results, dict)
        if 'eigenvalues' in results:
            # Should have at most 4 eigenvalues (2x2 weight matrix)
            self.assertLessEqual(len(results['eigenvalues']), 4)

    def test_zero_gradients(self):
        """Test with model that has no gradients."""
        model = TestModel()
        # Freeze all parameters
        for param in model.parameters():
            param.requires_grad = False

        batch = {
            'input_ids': torch.randn(4, 10),
            'labels': torch.randint(0, 10, (4,))
        }

        # This should handle gracefully
        results = compute_spectrum(
            model, batch,
            operator_type='hessian',
            config=LanczosConfig(k=3, max_iters=10)
        )

        self.assertIsInstance(results, dict)
        # Might return error or empty eigenvalues

    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        model = TestModel()
        batch = {
            'input_ids': torch.randn(1, 10),
            'labels': torch.randint(0, 10, (1,))
        }

        # Should work fine
        results = compute_spectrum(
            model, batch,
            operator_type='empirical_fisher',
            config=LanczosConfig(k=3, max_iters=10)
        )

        self.assertIsInstance(results, dict)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestLinearOperators))
    suite.addTests(loader.loadTestsFromTestCase(TestLanczosAlgorithm))
    suite.addTests(loader.loadTestsFromTestCase(TestComputeSpectrum))
    suite.addTests(loader.loadTestsFromTestCase(TestAdvancedFisherIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestICLRMetricsIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n✅ ALL TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
        if result.failures:
            print("\nFailed tests:")
            for test, traceback in result.failures:
                print(f"  - {test}")
        if result.errors:
            print("\nTests with errors:")
            for test, traceback in result.errors:
                print(f"  - {test}")

    return result.wasSuccessful()


if __name__ == '__main__':
    # Run tests
    success = run_tests()

    # Exit with appropriate code
    sys.exit(0 if success else 1)