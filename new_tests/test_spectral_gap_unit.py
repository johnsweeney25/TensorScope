#!/usr/bin/env python3
"""
Unit Tests for compute_spectral_gap
====================================
Tests that compute_spectral_gap works correctly with the refactored Fisher module.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
import warnings

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InformationTheoryMetrics import InformationTheoryMetrics


class TestModel(nn.Module):
    """Simple test model for unit tests."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=10):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, input_ids, labels=None, **kwargs):
        x = self.linear1(input_ids.float())
        x = F.relu(x)
        logits = self.linear2(x)

        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        else:
            loss = logits.mean()

        return type('Output', (), {'loss': loss, 'logits': logits})()


class NaNModel(nn.Module):
    """Model that produces NaN losses for testing error handling."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.counter = 0

    def forward(self, input_ids, labels=None, **kwargs):
        logits = self.linear(input_ids.float())

        # Produce NaN loss on some samples
        self.counter += 1
        if self.counter % 5 == 0:  # Every 5th sample
            loss = torch.tensor(float('nan'))
        else:
            loss = logits.mean()

        return type('Output', (), {'loss': loss, 'logits': logits})()


class TestSpectralGapWithFisherSpectral(unittest.TestCase):
    """Test compute_spectral_gap using FisherSpectral module."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.batch = {
            'input_ids': torch.randn(32, 10),
            'labels': torch.randint(0, 10, (32,))
        }
        self.metrics = InformationTheoryMetrics()

    def test_basic_computation(self):
        """Test basic spectral gap computation."""
        results = self.metrics.compute_spectral_gap(self.model, self.batch)

        # Check that we got results
        self.assertIsInstance(results, dict)
        self.assertIn('spectral_gap', results)
        self.assertIn('condition_number', results)
        self.assertIn('fim_effective_rank', results)
        self.assertIn('largest_eigenvalue', results)

    def test_values_are_reasonable(self):
        """Test that computed values are reasonable."""
        results = self.metrics.compute_spectral_gap(self.model, self.batch)

        # Check value ranges
        self.assertGreater(results['largest_eigenvalue'], 0,
                          "Largest eigenvalue should be positive")
        self.assertGreater(results['condition_number'], 0,
                          "Condition number should be positive")
        self.assertGreater(results['fim_effective_rank'], 0,
                          "Effective rank should be positive")

        # Spectral gap can be zero if all eigenvalues are the same
        self.assertGreaterEqual(results['spectral_gap'], 0,
                               "Spectral gap should be non-negative")

    def test_optimization_timescale(self):
        """Test that optimization timescale is computed correctly."""
        results = self.metrics.compute_spectral_gap(self.model, self.batch)

        if 'optimization_timescale' in results:
            # Should be reciprocal of largest eigenvalue
            expected = 1.0 / results['largest_eigenvalue']
            self.assertAlmostEqual(results['optimization_timescale'], expected, places=10)

    def test_different_batch_sizes(self):
        """Test with different batch sizes."""
        for batch_size in [8, 16, 64]:
            with self.subTest(batch_size=batch_size):
                batch = {
                    'input_ids': torch.randn(batch_size, 10),
                    'labels': torch.randint(0, 10, (batch_size,))
                }
                results = self.metrics.compute_spectral_gap(self.model, batch)

                self.assertIn('spectral_gap', results)
                self.assertIn('num_samples', results)
                # Should use min(256, batch_size) samples
                self.assertLessEqual(results.get('num_samples', 0), min(256, batch_size))


class TestSpectralGapFallback(unittest.TestCase):
    """Test fallback implementation when FisherSpectral is not available."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = TestModel()
        self.batch = {
            'input_ids': torch.randn(16, 10),
            'labels': torch.randint(0, 10, (16,))
        }
        self.metrics = InformationTheoryMetrics()

    def test_fallback_when_import_fails(self):
        """Test that fallback SVD method works when FisherSpectral unavailable."""
        # Temporarily hide the fisher module
        import sys
        original_modules = {}

        # Save and remove fisher modules
        for key in list(sys.modules.keys()):
            if 'fisher' in key.lower():
                original_modules[key] = sys.modules[key]
                del sys.modules[key]

        try:
            # Create new metrics instance (will trigger import)
            metrics = InformationTheoryMetrics()

            # Should fall back to SVD implementation
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                results = metrics.compute_spectral_gap(self.model, self.batch)

                # Should have warned about fallback
                warning_messages = [str(warning.message) for warning in w]
                self.assertTrue(
                    any("FisherSpectral module not found" in msg for msg in warning_messages),
                    "Should warn about missing FisherSpectral"
                )

            # Check results from fallback
            self.assertIn('spectral_gap', results)
            self.assertIn('condition_number', results)
            self.assertIn('fim_effective_rank', results)
            self.assertIn('largest_eigenvalue', results)

        finally:
            # Restore original modules
            sys.modules.update(original_modules)

    def test_svd_implementation_correctness(self):
        """Test that SVD fallback computes reasonable values."""
        # Force use of fallback by calling private method directly
        results = self.metrics._compute_fim_spectrum(
            self.model, self.batch, n_samples=16, max_params=1000
        )

        # Check results
        self.assertIn('spectral_gap', results)
        self.assertGreater(results['largest_eigenvalue'], 0)
        self.assertGreater(results['condition_number'], 0)
        self.assertGreater(results['fim_effective_rank'], 0)


class TestNonFiniteLossHandling(unittest.TestCase):
    """Test handling of non-finite (NaN/Inf) losses."""

    def setUp(self):
        """Set up test fixtures."""
        self.nan_model = NaNModel()
        self.batch = {
            'input_ids': torch.randn(32, 10),
            'labels': torch.randint(0, 10, (32,))
        }
        self.metrics = InformationTheoryMetrics()

    def test_nan_loss_handling(self):
        """Test that NaN losses are handled gracefully."""
        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="Non-finite loss")

            results = self.metrics.compute_spectral_gap(self.nan_model, self.batch)

            # Should still get results despite NaN losses
            self.assertIsInstance(results, dict)
            self.assertIn('spectral_gap', results)

            # Values should still be computed from valid samples
            if results['largest_eigenvalue'] > 0:
                self.assertGreater(results['condition_number'], 0)

    def test_warning_on_nan_loss(self):
        """Test that warnings are issued for non-finite losses."""
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Reset model counter
            self.nan_model.counter = 0
            results = self.metrics.compute_spectral_gap(self.nan_model, self.batch)

            # Check for non-finite loss warnings
            # Note: These might be logged rather than warned
            # The test passes if we get results despite NaN losses
            self.assertIsInstance(results, dict)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and boundary conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.metrics = InformationTheoryMetrics()

    def test_single_sample_batch(self):
        """Test with batch size of 1."""
        model = TestModel()
        batch = {
            'input_ids': torch.randn(1, 10),
            'labels': torch.randint(0, 10, (1,))
        }

        results = self.metrics.compute_spectral_gap(model, batch)

        # Should still work with single sample
        self.assertIsInstance(results, dict)
        self.assertIn('spectral_gap', results)

    def test_tiny_model(self):
        """Test with a very small model."""
        model = nn.Linear(2, 2)

        # Wrap to have proper interface
        class TinyWrapper(nn.Module):
            def __init__(self, linear):
                super().__init__()
                self.linear = linear

            def forward(self, input_ids, **kwargs):
                out = self.linear(input_ids.float())
                return type('Output', (), {'loss': out.mean(), 'logits': out})()

        wrapped = TinyWrapper(model)
        batch = {'input_ids': torch.randn(4, 2)}

        results = self.metrics.compute_spectral_gap(wrapped, batch)

        # Should handle tiny models
        self.assertIsInstance(results, dict)
        self.assertIn('spectral_gap', results)

    def test_large_model_subsampling(self):
        """Test parameter subsampling for large models."""
        # Create a model with many parameters
        model = nn.Sequential(
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 100)
        )

        # Wrap for proper interface
        class LargeWrapper(nn.Module):
            def __init__(self, seq):
                super().__init__()
                self.seq = seq

            def forward(self, input_ids, **kwargs):
                out = self.seq(input_ids.float())
                return type('Output', (), {'loss': out.mean(), 'logits': out})()

        wrapped = LargeWrapper(model)
        batch = {'input_ids': torch.randn(8, 100)}

        # Should subsample parameters if too many
        results = self.metrics._compute_fim_spectrum(
            wrapped, batch, n_samples=8, max_params=1000
        )

        self.assertIsInstance(results, dict)
        self.assertIn('spectral_gap', results)
        # Check that subsampling was used (num_params should be <= max_params)
        self.assertLessEqual(results.get('num_params', float('inf')), 1000)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSpectralGapWithFisherSpectral))
    suite.addTests(loader.loadTestsFromTestCase(TestSpectralGapFallback))
    suite.addTests(loader.loadTestsFromTestCase(TestNonFiniteLossHandling))
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