#!/usr/bin/env python3
"""
Unit tests for manifold_adaptive_sampling.py functions.

Tests comprehensive functionality including:
- get_adaptive_manifold_samples: Adaptive sample size selection
- safe_compute_n_pairs: Overflow-safe pair computation
- estimate_manifold_computation_time: Time complexity estimation
- validate_sample_size: Statistical validation
- compute_robust_cv: Robust coefficient of variation
- compute_bootstrap_ci: Bootstrap confidence intervals

Each test validates both correctness and edge case handling.
"""

import unittest
import numpy as np
import torch
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from manifold_adaptive_sampling import (
    get_adaptive_manifold_samples,
    safe_compute_n_pairs,
    validate_sample_size,
    compute_robust_cv,
    compute_bootstrap_ci,
    estimate_manifold_computation_time,
    MIN_STATISTICAL_SAMPLES,
    MAX_SAFE_N_POINTS,
    NORMAL_CI_Z_SCORE
)


class TestSafeComputeNPairs(unittest.TestCase):
    """Test safe_compute_n_pairs function for overflow protection."""

    def test_normal_computation(self):
        """Test normal pair computation for reasonable inputs."""
        self.assertEqual(safe_compute_n_pairs(5), 10)  # 5*4/2 = 10
        self.assertEqual(safe_compute_n_pairs(10), 45)  # 10*9/2 = 45
        self.assertEqual(safe_compute_n_pairs(100), 4950)  # 100*99/2 = 4950

    def test_edge_cases(self):
        """Test edge cases: 0, 1, 2 points."""
        self.assertEqual(safe_compute_n_pairs(0), 0)
        self.assertEqual(safe_compute_n_pairs(1), 0)
        self.assertEqual(safe_compute_n_pairs(2), 1)

    def test_negative_input(self):
        """Test that negative input raises ValueError."""
        with self.assertRaises(ValueError) as context:
            safe_compute_n_pairs(-1)
        self.assertIn("non-negative", str(context.exception))

    def test_overflow_protection(self):
        """Test that very large inputs raise overflow error."""
        # Value larger than MAX_SAFE_N_POINTS should raise error
        with self.assertRaises(ValueError) as context:
            safe_compute_n_pairs(MAX_SAFE_N_POINTS + 1)
        self.assertIn("overflow", str(context.exception))

    def test_boundary_value(self):
        """Test computation at the safety boundary."""
        # Should work at or below MAX_SAFE_N_POINTS
        try:
            result = safe_compute_n_pairs(1000)  # Well below limit
            self.assertIsInstance(result, int)
            self.assertGreater(result, 0)
        except ValueError:
            self.fail("Should not raise error for safe values")


class TestGetAdaptiveManifoldSamples(unittest.TestCase):
    """Test adaptive sample size selection."""

    def test_basic_sampling_strategies(self):
        """Test different time budget strategies."""
        n_points = 500

        # Fast mode should give fewer samples
        fast_samples = get_adaptive_manifold_samples(n_points, "fast")
        balanced_samples = get_adaptive_manifold_samples(n_points, "balanced")
        thorough_samples = get_adaptive_manifold_samples(n_points, "thorough")

        self.assertLessEqual(fast_samples, balanced_samples)
        self.assertLessEqual(balanced_samples, thorough_samples)

    def test_skip_expensive_flag(self):
        """Test that skip_expensive forces fast mode."""
        n_points = 500
        normal_thorough = get_adaptive_manifold_samples(n_points, "thorough", skip_expensive=False)
        skipped_thorough = get_adaptive_manifold_samples(n_points, "thorough", skip_expensive=True)

        # When skip_expensive=True, should use fast mode regardless
        self.assertLess(skipped_thorough, normal_thorough)

    def test_dataset_size_awareness(self):
        """Test adaptive behavior for different dataset sizes."""
        # Small dataset should get higher relative sampling
        small_samples = get_adaptive_manifold_samples(50, "balanced", dataset_size_aware=True)
        small_pairs = safe_compute_n_pairs(50)
        small_coverage = small_samples / small_pairs if small_pairs > 0 else 0

        # Large dataset should get lower relative sampling
        large_samples = get_adaptive_manifold_samples(5000, "balanced", dataset_size_aware=True)
        large_pairs = safe_compute_n_pairs(5000)
        large_coverage = large_samples / large_pairs if large_pairs > 0 else 0

        self.assertGreater(small_coverage, large_coverage)

    def test_bounds_enforcement(self):
        """Test that min/max sample bounds are enforced."""
        n_points = 1000

        # Test minimum bound
        samples = get_adaptive_manifold_samples(n_points, "fast", min_samples=100, max_samples=1000)
        self.assertGreaterEqual(samples, 100)

        # Test maximum bound
        samples = get_adaptive_manifold_samples(n_points, "thorough", min_samples=10, max_samples=50)
        self.assertLessEqual(samples, 50)

    def test_invalid_bounds(self):
        """Test that invalid bounds raise errors."""
        with self.assertRaises(ValueError):
            get_adaptive_manifold_samples(100, min_samples=100, max_samples=50)

        with self.assertRaises(ValueError):
            get_adaptive_manifold_samples(100, min_samples=-10)

    def test_edge_cases(self):
        """Test edge cases for n_points."""
        # Zero points
        self.assertEqual(get_adaptive_manifold_samples(0), 0)

        # One point
        self.assertEqual(get_adaptive_manifold_samples(1), 0)

        # Two points (only one pair possible)
        samples = get_adaptive_manifold_samples(2)
        self.assertLessEqual(samples, 1)

        # Negative points should raise error
        with self.assertRaises(ValueError):
            get_adaptive_manifold_samples(-1)

    def test_overflow_handling(self):
        """Test handling of very large n_points."""
        # Should fallback gracefully for very large values
        samples = get_adaptive_manifold_samples(int(1e10), "fast")
        self.assertIsInstance(samples, int)
        self.assertGreater(samples, 0)


class TestEstimateManifoldComputationTime(unittest.TestCase):
    """Test computation time estimation."""

    def test_basic_estimation(self):
        """Test basic time estimation returns expected fields."""
        result = estimate_manifold_computation_time(1000, 100)

        self.assertIn("distance_matrix", result)
        self.assertIn("ricci_curvature", result)
        self.assertIn("intrinsic_dimension", result)
        self.assertIn("total_estimated", result)
        self.assertIn("formatted", result)
        self.assertIn("complexity", result)

    def test_device_scaling(self):
        """Test that GPU estimates are faster than CPU."""
        cpu_time = estimate_manifold_computation_time(1000, 100, device="cpu")
        gpu_time = estimate_manifold_computation_time(1000, 100, device="cuda")

        self.assertLess(gpu_time["total_estimated"], cpu_time["total_estimated"])

    def test_uncertainty_bounds(self):
        """Test inclusion of uncertainty bounds."""
        result = estimate_manifold_computation_time(1000, 100, include_uncertainty=True)

        self.assertIn("confidence_interval", result)
        self.assertIn("lower", result["confidence_interval"])
        self.assertIn("upper", result["confidence_interval"])

        # Upper bound should be greater than lower bound
        self.assertGreater(
            result["confidence_interval"]["upper"],
            result["confidence_interval"]["lower"]
        )

    def test_time_formatting(self):
        """Test time formatting for different durations."""
        # Small time (seconds)
        result = estimate_manifold_computation_time(100, 10)
        self.assertTrue(result["formatted"].endswith("s"))

        # Large dataset should show warning
        result = estimate_manifold_computation_time(10000, 10000)
        if result["total_estimated"] > 300:
            self.assertIn("warning", result)

    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        result = estimate_manifold_computation_time(-1, 100)
        self.assertIn("error", result)

        result = estimate_manifold_computation_time(100, -1)
        self.assertIn("error", result)


class TestValidateSampleSize(unittest.TestCase):
    """Test sample size validation."""

    def test_quality_assessment(self):
        """Test quality assessment for different sample sizes."""
        # Poor quality for very small samples
        result = validate_sample_size(10, 1000)
        self.assertEqual(result["quality"], "poor")

        # Acceptable quality for minimum samples
        result = validate_sample_size(MIN_STATISTICAL_SAMPLES, 1000)
        self.assertEqual(result["quality"], "acceptable")

        # Good/excellent quality for large samples
        result = validate_sample_size(500, 1000)
        self.assertIn(result["quality"], ["good", "excellent", "acceptable"])

    def test_statistical_warnings(self):
        """Test that appropriate warnings are generated."""
        # Warning for samples below minimum
        result = validate_sample_size(15, 1000)
        self.assertTrue(any("below" in w for w in result["warnings"]))

        # Warning for very low coverage
        result = validate_sample_size(30, 10000)
        self.assertTrue(any("coverage" in w.lower() for w in result["warnings"]))

    def test_statistical_power_analysis(self):
        """Test statistical power calculations."""
        result = validate_sample_size(100, 1000)

        self.assertIn("statistical_power", result)
        power = result["statistical_power"]

        # Power should be between 0 and 1
        self.assertLessEqual(power["small_effect"], 1.0)
        self.assertLessEqual(power["medium_effect"], 1.0)
        self.assertLessEqual(power["large_effect"], 1.0)

        # Large effects should have higher power than small
        self.assertGreaterEqual(power["large_effect"], power["small_effect"])

    def test_bootstrap_vs_normal(self):
        """Test difference between bootstrap and normal methods."""
        result_bootstrap = validate_sample_size(50, 1000, use_bootstrap=True)
        result_normal = validate_sample_size(50, 1000, use_bootstrap=False)

        self.assertEqual(result_bootstrap["method"], "bootstrap")
        self.assertEqual(result_normal["method"], "normal")

    def test_edge_cases(self):
        """Test edge cases for validation."""
        # Zero samples
        result = validate_sample_size(0, 100)
        self.assertFalse(result["valid"])

        # Zero points
        result = validate_sample_size(100, 0)
        self.assertFalse(result["valid"])

        # Negative inputs
        result = validate_sample_size(-1, 100)
        self.assertIn("error", result)


class TestComputeRobustCV(unittest.TestCase):
    """Test robust coefficient of variation computation."""

    def test_normal_cv(self):
        """Test CV for normal cases."""
        values = np.array([1, 2, 3, 4, 5])
        cv = compute_robust_cv(values)
        expected_cv = np.std(values) / np.mean(values)
        self.assertAlmostEqual(cv, expected_cv, places=5)

    def test_zero_mean_handling(self):
        """Test handling of zero or near-zero mean."""
        values = np.array([-1, 0, 1])  # Mean = 0

        # Relative method should not return inf
        cv_relative = compute_robust_cv(values, handle_zero_mean="relative")
        self.assertFalse(np.isinf(cv_relative))

        # Undefined method should return inf for non-zero std
        cv_undefined = compute_robust_cv(values, handle_zero_mean="undefined")
        self.assertTrue(np.isinf(cv_undefined))

        # Epsilon method should return finite value
        cv_epsilon = compute_robust_cv(values, handle_zero_mean="epsilon")
        self.assertFalse(np.isinf(cv_epsilon))

    def test_empty_array(self):
        """Test handling of empty array."""
        cv = compute_robust_cv(np.array([]))
        self.assertTrue(np.isnan(cv))

    def test_constant_values(self):
        """Test CV for constant values (zero variance)."""
        values = np.array([5, 5, 5, 5])
        cv = compute_robust_cv(values)
        self.assertEqual(cv, 0.0)


class TestComputeBootstrapCI(unittest.TestCase):
    """Test bootstrap confidence interval computation."""

    def test_basic_bootstrap(self):
        """Test basic bootstrap CI computation."""
        np.random.seed(42)
        values = np.random.randn(50)
        lower, upper = compute_bootstrap_ci(values)

        # CI should contain the mean
        mean = np.mean(values)
        self.assertLessEqual(lower, mean)
        self.assertGreaterEqual(upper, mean)

        # Upper should be greater than lower
        self.assertGreater(upper, lower)

    def test_confidence_levels(self):
        """Test different confidence levels."""
        np.random.seed(42)
        values = np.random.randn(50)

        ci_90 = compute_bootstrap_ci(values, confidence_level=0.90)
        ci_95 = compute_bootstrap_ci(values, confidence_level=0.95)
        ci_99 = compute_bootstrap_ci(values, confidence_level=0.99)

        # Higher confidence should give wider intervals
        width_90 = ci_90[1] - ci_90[0]
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]

        self.assertLessEqual(width_90, width_95)
        self.assertLessEqual(width_95, width_99)

    def test_methods(self):
        """Test different bootstrap methods."""
        np.random.seed(42)
        values = np.random.randn(50)

        ci_percentile = compute_bootstrap_ci(values, method="percentile")
        ci_bca = compute_bootstrap_ci(values, method="bca")

        # Both should return valid intervals
        self.assertEqual(len(ci_percentile), 2)
        self.assertEqual(len(ci_bca), 2)

        # Invalid method should raise error
        with self.assertRaises(ValueError):
            compute_bootstrap_ci(values, method="invalid")

    def test_edge_cases(self):
        """Test edge cases for bootstrap CI."""
        # Empty array
        lower, upper = compute_bootstrap_ci(np.array([]))
        self.assertTrue(np.isnan(lower))
        self.assertTrue(np.isnan(upper))

        # Single value
        lower, upper = compute_bootstrap_ci(np.array([5.0]))
        self.assertEqual(lower, 5.0)
        self.assertEqual(upper, 5.0)

    def test_reproducibility(self):
        """Test that bootstrap is reproducible with fixed seed."""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

        # Same seed should give same results
        ci1 = compute_bootstrap_ci(values)
        ci2 = compute_bootstrap_ci(values)

        self.assertAlmostEqual(ci1[0], ci2[0], places=5)
        self.assertAlmostEqual(ci1[1], ci2[1], places=5)


class TestIntegrationScenarios(unittest.TestCase):
    """Integration tests for realistic usage scenarios."""

    def test_small_dataset_workflow(self):
        """Test typical workflow for small dataset."""
        n_points = 50

        # Get adaptive samples
        n_samples = get_adaptive_manifold_samples(n_points, "balanced")

        # Validate the sample size
        validation = validate_sample_size(n_samples, n_points)

        # Should have reasonable quality
        self.assertIn(validation["quality"], ["acceptable", "good", "excellent"])
        self.assertTrue(validation["valid"])

        # Estimate computation time
        time_est = estimate_manifold_computation_time(n_points, n_samples)
        self.assertGreater(time_est["total_estimated"], 0)

    def test_large_dataset_workflow(self):
        """Test typical workflow for large dataset."""
        n_points = 10000

        # Get adaptive samples with fast mode
        n_samples = get_adaptive_manifold_samples(n_points, "fast")

        # Should respect computational constraints
        self.assertLess(n_samples, 1000)  # Should not be too large

        # Validate
        validation = validate_sample_size(n_samples, n_points)

        # Coverage should be low but valid
        self.assertLess(validation["coverage_percent"], 1.0)
        self.assertGreater(n_samples, 0)

    def test_statistical_analysis_workflow(self):
        """Test statistical analysis of sampling results."""
        # Simulate multiple measurements
        np.random.seed(42)
        measurements = np.random.randn(30) * 0.1 - 0.5  # Simulated Ricci values

        # Compute robust CV
        cv = compute_robust_cv(measurements)
        self.assertGreater(cv, 0)

        # Compute bootstrap CI
        ci = compute_bootstrap_ci(measurements)
        self.assertGreater(ci[1], ci[0])

        # Validate if we have enough samples
        validation = validate_sample_size(len(measurements), 1000)
        self.assertEqual(validation["n_samples"], len(measurements))


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)