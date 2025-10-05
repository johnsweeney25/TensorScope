"""
Test suite for SuperpositionAnalyzer to verify optimization and caching.

Converted to unittest framework for consistency.
"""

import unittest
import torch
import torch.nn as nn
import time
import numpy as np
import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from superposition.core.analyzer import SuperpositionAnalyzer, analyze_model_superposition_comprehensive


class TestComprehensiveAnalysis(unittest.TestCase):
    """Test comprehensive superposition analysis functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_comprehensive_analysis(self):
        """Test the comprehensive analysis method."""
        # Create test weight matrix (100 features in 20 dimensions)
        weight_matrix = torch.randn(100, 20)

        # Run comprehensive analysis
        result = self.analyzer.compute_comprehensive_superposition_analysis(
            weight_matrix,
            return_matrices=True
        )

        # Check all fields are present
        self.assertTrue(hasattr(result, 'phi_half'), "Missing phi_half")
        self.assertTrue(hasattr(result, 'phi_one'), "Missing phi_one")
        self.assertTrue(hasattr(result, 'regime'), "Missing regime")
        self.assertTrue(hasattr(result, 'mean_overlap'), "Missing mean_overlap")
        self.assertTrue(hasattr(result, 'welch_bound'), "Missing welch_bound")

        # Verify results are in expected ranges
        self.assertGreaterEqual(result.phi_half, 0.0)
        self.assertLessEqual(result.phi_half, 1.0)
        self.assertGreaterEqual(result.phi_one, 0.0)
        self.assertLessEqual(result.phi_one, 1.0)
        self.assertIn(result.regime, ['no_superposition', 'weak_superposition', 'strong_superposition'])


class TestCaching(unittest.TestCase):
    """Test caching functionality and efficiency."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_caching_efficiency(self):
        """Test that caching reduces computation time."""
        # Create larger test matrix
        weight_matrix = torch.randn(500, 100)

        # First run (cache miss)
        start_time = time.time()
        result1 = self.analyzer.compute_comprehensive_superposition_analysis(weight_matrix)
        first_run_time = time.time() - start_time

        # Second run (should use cache)
        start_time = time.time()
        result2 = self.analyzer.compute_comprehensive_superposition_analysis(weight_matrix)
        cached_run_time = time.time() - start_time

        # Check results are identical
        self.assertEqual(result1.phi_half, result2.phi_half, "Results differ between runs")
        self.assertEqual(result1.mean_overlap, result2.mean_overlap, "Results differ between runs")

        # Check cache statistics
        cache_stats = self.analyzer.get_cache_statistics()
        self.assertGreater(cache_stats['total_hits'], 0, "No cache hits recorded")

        # Cached run should be faster (allow some variance)
        self.assertLess(cached_run_time, first_run_time * 0.8,
                       f"Caching not effective: first={first_run_time:.4f}s, cached={cached_run_time:.4f}s")

    def test_no_duplication(self):
        """Test that norms are computed only once."""
        # Clear cache to start fresh
        self.analyzer.clear_cache()

        weight_matrix = torch.randn(50, 30)

        # Run analysis
        result = self.analyzer.compute_comprehensive_superposition_analysis(weight_matrix)

        # Check cache was populated correctly
        cache_stats = self.analyzer.get_cache_statistics()
        self.assertEqual(cache_stats['detailed_misses']['norms'], 1, "Norms computed more than once")
        self.assertEqual(cache_stats['detailed_misses']['overlaps'], 1, "Overlaps computed more than once")

    def test_cache_clearing(self):
        """Test cache clearing functionality."""
        weight_matrix = torch.randn(20, 10)

        # Populate cache
        self.analyzer.compute_comprehensive_superposition_analysis(weight_matrix)

        # Clear cache
        self.analyzer.clear_cache()

        # Check cache is empty
        cache_stats = self.analyzer.get_cache_statistics()
        self.assertEqual(cache_stats['cache_sizes']['norms'], 0)
        self.assertEqual(cache_stats['cache_sizes']['overlaps'], 0)


class TestOptimizedMethods(unittest.TestCase):
    """Test optimized method implementations."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_optimized_vector_interference(self):
        """Test the optimized vector interference method."""
        weight_matrix = torch.randn(100, 50)

        # Test with return_norms
        result = self.analyzer.compute_vector_interference_optimized(
            weight_matrix,
            return_norms=True
        )

        self.assertIn('feature_norms', result, "Norms not returned")
        self.assertIsNotNone(result['feature_norms'], "Norms are None")
        self.assertEqual(len(result['feature_norms']), 100, "Wrong number of norms")

        # Verify norm values are reasonable
        mean_norm = np.mean(result['feature_norms'])
        self.assertGreater(mean_norm, 0, "Mean norm should be positive")

    def test_optimized_without_norms(self):
        """Test optimized method without returning norms."""
        weight_matrix = torch.randn(50, 25)

        result = self.analyzer.compute_vector_interference_optimized(
            weight_matrix,
            return_norms=False
        )

        self.assertNotIn('feature_norms', result, "Norms should not be returned")
        self.assertIn('mean_overlap', result, "Missing mean_overlap")


class TestRegimeClassification(unittest.TestCase):
    """Test superposition regime classification."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_no_superposition_regime(self):
        """Test classification of no superposition regime."""
        # More dims than features - should have no superposition
        weight_matrix = torch.randn(50, 100)
        # Make first 50 features strong, rest weak
        weight_matrix[:50] *= 2.0

        result = self.analyzer.compute_comprehensive_superposition_analysis(weight_matrix)

        # With more dimensions than features, shouldn't be strong superposition
        self.assertNotEqual(result.regime, "strong_superposition")

    def test_strong_superposition_regime(self):
        """Test classification of strong superposition regime."""
        # Many more features than dims with all normalized
        n_features, n_dims = 1000, 100
        weight_matrix = torch.randn(n_features, n_dims)
        weight_matrix = weight_matrix / weight_matrix.norm(dim=1, keepdim=True)

        result = self.analyzer.compute_comprehensive_superposition_analysis(weight_matrix)

        # Should detect strong superposition
        self.assertEqual(result.regime, "strong_superposition")
        self.assertGreater(result.phi_half, 0.8)  # Most features represented


class TestModelAnalysis(unittest.TestCase):
    """Test analysis on neural network models."""

    def test_model_layer_analysis(self):
        """Test analysis on a real model."""
        # Create test model
        model = nn.Sequential(
            nn.Embedding(500, 64),  # 500 features in 64 dimensions
            nn.Linear(64, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 500)
        )

        # Analyze all layers
        results = analyze_model_superposition_comprehensive(model, verbose=False)

        # Should analyze 4 layers
        self.assertEqual(len(results), 4, "Should analyze all 4 layers")

        # Check each layer has valid results
        for layer_name, analysis in results.items():
            self.assertTrue(hasattr(analysis, 'regime'))
            self.assertTrue(hasattr(analysis, 'phi_half'))
            self.assertTrue(hasattr(analysis, 'phi_one'))
            self.assertIn(analysis.regime, ['no_superposition', 'weak_superposition', 'strong_superposition'])


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_empty_matrix(self):
        """Test handling of empty matrix."""
        empty = torch.zeros(0, 10)
        result = self.analyzer.compute_comprehensive_superposition_analysis(empty)

        self.assertEqual(result.n_features, 0)
        self.assertEqual(result.phi_half, 0)
        self.assertEqual(result.phi_one, 0)

    def test_single_feature(self):
        """Test handling of single feature."""
        single = torch.randn(1, 10)
        result = self.analyzer.compute_comprehensive_superposition_analysis(single)

        self.assertEqual(result.n_features, 1)
        # Single feature should have phi values based on its norm
        self.assertGreaterEqual(result.phi_half, 0)
        self.assertLessEqual(result.phi_half, 1)

    def test_large_matrix_batching(self):
        """Test handling of very large matrix with batching."""
        large = torch.randn(1000, 500)
        result = self.analyzer.compute_comprehensive_superposition_analysis(large, batch_size=100)

        self.assertEqual(result.n_features, 1000)
        self.assertEqual(result.n_dimensions, 500)

        # Should complete without memory errors
        self.assertIsNotNone(result.mean_overlap)
        self.assertIsNotNone(result.regime)

    def test_zero_norm_features(self):
        """Test handling of features with zero norm."""
        weight_matrix = torch.randn(10, 5)
        weight_matrix[5:] = 0  # Set half to zero

        result = self.analyzer.compute_comprehensive_superposition_analysis(weight_matrix)

        # Should handle zero-norm features gracefully
        self.assertEqual(result.n_features, 10)
        self.assertEqual(result.phi_half, 0.5)  # Half have non-zero norm


class TestDeviceCompatibility(unittest.TestCase):
    """Test CPU/GPU device compatibility."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.analyzer = SuperpositionAnalyzer(device=self.device)

    def test_device_handling(self):
        """Test analysis works on configured device."""
        weight_matrix = torch.randn(50, 25, device=self.device)

        result = self.analyzer.compute_comprehensive_superposition_analysis(weight_matrix)

        # Should complete successfully
        self.assertIsNotNone(result)
        self.assertEqual(result.n_features, 50)

    def test_mixed_device_inputs(self):
        """Test handling of inputs on different devices."""
        if torch.cuda.is_available():
            cpu_analyzer = SuperpositionAnalyzer(device=torch.device('cpu'))
            gpu_matrix = torch.randn(30, 15, device='cuda')

            # Should handle device transfer
            result = cpu_analyzer.compute_comprehensive_superposition_analysis(gpu_matrix)
            self.assertIsNotNone(result)


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()

    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestComprehensiveAnalysis))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestCaching))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestOptimizedMethods))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestRegimeClassification))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestModelAnalysis))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestDeviceCompatibility))

    return suite


if __name__ == '__main__':
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())