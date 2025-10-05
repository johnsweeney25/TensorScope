#!/usr/bin/env python3
"""
Unit tests for _interpret_multiscale_results and _check_position_dependency functions
Testing the fixes for configuration mismatch, numerical precision, and statistical analysis
"""
import unittest
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GradientAnalysis import GradientAnalysis


class TestGradientMultiscaleInterpretation(unittest.TestCase):
    """Test suite for multiscale gradient analysis interpretation functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GradientAnalysis()

    def test_interpret_empty_results(self):
        """Test handling of empty results."""
        result = self.analyzer._interpret_multiscale_results({})
        self.assertEqual(result, "No results to interpret")

    def test_interpret_single_data_point(self):
        """Test interpretation with single data point."""
        # Test high conflict
        results = {
            'tokens_0_to_256': {'conflict_score': 0.8}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        self.assertIn("High conflict", interpretation)
        self.assertIn("256 tokens", interpretation)

        # Test low conflict
        results = {
            'tokens_0_to_256': {'conflict_score': 0.1}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        self.assertIn("Low conflict", interpretation)

        # Test moderate conflict
        results = {
            'tokens_0_to_256': {'conflict_score': 0.35}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        self.assertIn("Moderate conflict", interpretation)

    def test_interpret_with_nan_values(self):
        """Test handling of NaN values in results."""
        results = {
            'tokens_0_to_256': {'conflict_score': np.nan},
            'tokens_0_to_512': {'conflict_score': 0.3},
            'tokens_0_to_1024': {'conflict_score': 0.4}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        # Should skip NaN and process valid values
        self.assertNotIn("nan", interpretation.lower())
        self.assertIn("512", interpretation)

    def test_interpret_increasing_trend(self):
        """Test detection of increasing conflict trend."""
        results = {
            'tokens_0_to_128': {'conflict_score': 0.1},
            'tokens_0_to_256': {'conflict_score': 0.3},
            'tokens_0_to_512': {'conflict_score': 0.5},
            'tokens_0_to_1024': {'conflict_score': 0.7}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        # Should detect increasing trend
        self.assertTrue(
            "increases with" in interpretation.lower() or
            "increasing" in interpretation.lower() or
            "Conflict increases with length" in interpretation
        )

    def test_interpret_decreasing_trend(self):
        """Test detection of decreasing conflict trend."""
        results = {
            'tokens_0_to_128': {'conflict_score': 0.7},
            'tokens_0_to_256': {'conflict_score': 0.5},
            'tokens_0_to_512': {'conflict_score': 0.3},
            'tokens_0_to_1024': {'conflict_score': 0.1}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        # Should detect decreasing trend or early concentration
        self.assertTrue(
            "concentrated early" in interpretation or
            "decreases with" in interpretation.lower()
        )

    def test_interpret_peak_in_middle(self):
        """Test detection of conflict peak in middle ranges."""
        results = {
            'tokens_0_to_128': {'conflict_score': 0.2},
            'tokens_0_to_256': {'conflict_score': 0.3},
            'tokens_0_to_512': {'conflict_score': 0.8},  # Peak
            'tokens_0_to_768': {'conflict_score': 0.4},
            'tokens_0_to_1024': {'conflict_score': 0.2}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        # Should detect peak
        self.assertIn("peak", interpretation.lower())
        self.assertIn("512", interpretation)

    def test_interpret_high_conflict_throughout(self):
        """Test detection of consistently high conflict."""
        results = {
            'tokens_0_to_128': {'conflict_score': 0.7},
            'tokens_0_to_256': {'conflict_score': 0.8},
            'tokens_0_to_512': {'conflict_score': 0.75},
            'tokens_0_to_1024': {'conflict_score': 0.85}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        self.assertIn("Strong conflict throughout", interpretation)

    def test_interpret_minimal_conflict(self):
        """Test detection of consistently low conflict."""
        results = {
            'tokens_0_to_128': {'conflict_score': 0.05},
            'tokens_0_to_256': {'conflict_score': 0.08},
            'tokens_0_to_512': {'conflict_score': 0.1},
            'tokens_0_to_1024': {'conflict_score': 0.09}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        self.assertIn("Minimal conflict", interpretation)

    def test_interpret_invalid_keys(self):
        """Test handling of invalid/malformed keys."""
        results = {
            'invalid_key': {'conflict_score': 0.5},
            'tokens_0_to_256': {'conflict_score': 0.3},
            'tokens_0_to_abc': {'conflict_score': 0.4},  # Invalid number
            'tokens_0_to_512': {'conflict_score': 0.5}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        # Should process valid keys only
        self.assertIn("256", interpretation)
        self.assertIn("512", interpretation)
        self.assertNotIn("abc", interpretation)

    def test_check_position_dependency_insufficient_data(self):
        """Test position dependency check with insufficient data."""
        results = {
            'tokens_0_to_256': {'conflict_score': 0.5}
        }
        dependency = self.analyzer._check_position_dependency(results)
        self.assertFalse(dependency['position_dependent'])
        self.assertEqual(dependency['confidence'], 0.0)
        self.assertIn('reason', dependency)

    def test_check_position_dependency_increasing(self):
        """Test detection of increasing position dependency."""
        results = {
            'tokens_0_to_128': {'conflict_score': 0.1},
            'tokens_0_to_256': {'conflict_score': 0.2},
            'tokens_0_to_512': {'conflict_score': 0.3},
            'tokens_0_to_1024': {'conflict_score': 0.4}
        }
        dependency = self.analyzer._check_position_dependency(results)
        self.assertEqual(dependency['pattern'], 'increasing')
        self.assertTrue(dependency['position_dependent'])
        self.assertGreater(dependency['confidence'], 0.5)

    def test_check_position_dependency_with_nan(self):
        """Test position dependency with NaN values."""
        results = {
            'tokens_0_to_128': {'conflict_score': np.nan},
            'tokens_0_to_256': {'conflict_score': 0.3},
            'tokens_0_to_512': {'conflict_score': 0.4},
            'tokens_0_to_1024': {'conflict_score': 0.5}
        }
        dependency = self.analyzer._check_position_dependency(results)
        # Should skip NaN and still detect pattern
        self.assertIsNotNone(dependency['pattern'])
        self.assertIn('confidence', dependency)

    def test_check_position_dependency_mixed_pattern(self):
        """Test detection of mixed patterns."""
        results = {
            'tokens_0_to_128': {'conflict_score': 0.2},
            'tokens_0_to_256': {'conflict_score': 0.5},
            'tokens_0_to_512': {'conflict_score': 0.3},
            'tokens_0_to_1024': {'conflict_score': 0.6}
        }
        dependency = self.analyzer._check_position_dependency(results)
        # Pattern should not be strictly increasing/decreasing
        self.assertNotEqual(dependency['pattern'], 'increasing')
        self.assertNotEqual(dependency['pattern'], 'decreasing')

    def test_check_position_dependency_dynamic_configs(self):
        """Test that function works with any valid config names."""
        results = {
            'tokens_0_to_100': {'conflict_score': 0.2},
            'tokens_0_to_200': {'conflict_score': 0.3},
            'tokens_0_to_300': {'conflict_score': 0.4},
            'tokens_0_to_400': {'conflict_score': 0.5}
        }
        dependency = self.analyzer._check_position_dependency(results)
        # Should still work with non-standard token counts
        self.assertIsNotNone(dependency)
        self.assertIn('pattern', dependency)
        self.assertIn('confidence', dependency)

    def test_numerical_precision_comparison(self):
        """Test numerical precision in comparisons."""
        # Test values that are very close (should be treated as equal)
        results = {
            'tokens_0_to_128': {'conflict_score': 0.2000000001},
            'tokens_0_to_256': {'conflict_score': 0.2000000002},
            'tokens_0_to_512': {'conflict_score': 0.2000000003}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        # Should recognize these as essentially constant
        self.assertIn("consistent", interpretation.lower())

    def test_edge_case_all_zeros(self):
        """Test handling of all zero conflict scores."""
        results = {
            'tokens_0_to_128': {'conflict_score': 0.0},
            'tokens_0_to_256': {'conflict_score': 0.0},
            'tokens_0_to_512': {'conflict_score': 0.0}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        self.assertIn("Minimal conflict", interpretation)

    def test_missing_conflict_score_field(self):
        """Test handling when conflict_score field is missing."""
        results = {
            'tokens_0_to_128': {'other_field': 0.5},  # Missing conflict_score
            'tokens_0_to_256': {'conflict_score': 0.3},
            'tokens_0_to_512': {'conflict_score': 0.4}
        }
        interpretation = self.analyzer._interpret_multiscale_results(results)
        # Should process only valid entries
        self.assertIn("256", interpretation)
        self.assertNotIn("128", interpretation)


if __name__ == '__main__':
    unittest.main()