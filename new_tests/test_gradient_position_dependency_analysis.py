"""
Unit tests for GradientAnalysis._check_position_dependency function.
Tests comprehensive position dependency detection, confidence calculation,
correlation analysis, and numerical precision handling.
"""

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GradientAnalysis import GradientAnalysis


class TestCheckPositionDependency(unittest.TestCase):
    """Test suite for the _check_position_dependency function."""

    def setUp(self):
        """Set up test environment."""
        self.gradient_analysis = GradientAnalysis()

    def test_insufficient_data_points(self):
        """Test handling of insufficient data points (< 2)."""
        # Test with no data
        results = {}
        output = self.gradient_analysis._check_position_dependency(results)
        self.assertFalse(output['position_dependent'])
        self.assertEqual(output['confidence'], 0.0)
        self.assertEqual(output['reason'], 'Insufficient data points')

        # Test with only one data point
        results = {
            'tokens_0_to_256': {'global_conflict': 0.5}
        }
        output = self.gradient_analysis._check_position_dependency(results)
        self.assertFalse(output['position_dependent'])
        self.assertEqual(output['confidence'], 0.0)

    def test_nan_and_inf_handling(self):
        """Test that NaN and inf values are properly filtered out."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.3},
            'tokens_0_to_512': {'global_conflict': np.nan},
            'tokens_0_to_1024': {'global_conflict': np.inf},
            'tokens_0_to_1536': {'global_conflict': 0.5},
            'tokens_0_to_2048': {'global_conflict': -np.inf},
            'tokens_0_to_3072': {'global_conflict': 0.7}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        # Should only have 3 valid data points (0.3, 0.5, 0.7)
        self.assertEqual(output['num_data_points'], 3)
        self.assertAlmostEqual(output['mean_conflict'], 0.5, places=2)

    def test_score_range_validation(self):
        """Test that scores outside [0, 2] range are handled with warnings."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.3},
            'tokens_0_to_512': {'global_conflict': -0.5},  # Invalid: negative
            'tokens_0_to_1024': {'global_conflict': 3.0},   # Invalid: > 2
            'tokens_0_to_1536': {'global_conflict': 0.8}
        }

        with self.assertWarns(UserWarning):
            output = self.gradient_analysis._check_position_dependency(results)

        # Should only have 2 valid data points (0.3, 0.8)
        self.assertEqual(output['num_data_points'], 2)

    def test_monotonic_increasing_pattern(self):
        """Test detection of strictly increasing conflict pattern."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.1},
            'tokens_0_to_512': {'global_conflict': 0.3},
            'tokens_0_to_1024': {'global_conflict': 0.5},
            'tokens_0_to_1536': {'global_conflict': 0.7},
            'tokens_0_to_2048': {'global_conflict': 0.9}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        self.assertTrue(output['position_dependent'])
        self.assertEqual(output['pattern'], 'increasing')
        self.assertGreater(output['confidence'], 0.5)
        self.assertIn('Conflict increases with sequence length', output['interpretation'])

    def test_monotonic_decreasing_pattern(self):
        """Test detection of strictly decreasing conflict pattern."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.9},
            'tokens_0_to_512': {'global_conflict': 0.7},
            'tokens_0_to_1024': {'global_conflict': 0.5},
            'tokens_0_to_1536': {'global_conflict': 0.3}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        self.assertTrue(output['position_dependent'])
        self.assertEqual(output['pattern'], 'decreasing')
        self.assertGreater(output['confidence'], 0.5)
        self.assertIn('Conflict decreases with sequence length', output['interpretation'])

    def test_weak_monotonicity_with_equality(self):
        """Test weak monotonicity detection (allows equal consecutive values)."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.3},
            'tokens_0_to_512': {'global_conflict': 0.3},  # Equal to previous
            'tokens_0_to_1024': {'global_conflict': 0.5},
            'tokens_0_to_1536': {'global_conflict': 0.5}  # Equal to previous
        }

        output = self.gradient_analysis._check_position_dependency(results)

        # Should still detect as increasing due to weak monotonicity
        self.assertEqual(output['pattern'], 'increasing')

    def test_mixed_pattern(self):
        """Test detection of mixed (non-monotonic) conflict pattern."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.3},
            'tokens_0_to_512': {'global_conflict': 0.7},
            'tokens_0_to_1024': {'global_conflict': 0.2},
            'tokens_0_to_1536': {'global_conflict': 0.8}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        self.assertIn(output['pattern'], ['mixed', 'correlated', 'anti-correlated'])
        self.assertIsNotNone(output['correlation']['pearson_r'])

    def test_position_independent_low_variance(self):
        """Test position-independent detection when variance is very low."""
        # All values nearly identical
        results = {
            'tokens_0_to_256': {'global_conflict': 0.500},
            'tokens_0_to_512': {'global_conflict': 0.501},
            'tokens_0_to_1024': {'global_conflict': 0.499},
            'tokens_0_to_1536': {'global_conflict': 0.500}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        self.assertFalse(output['position_dependent'])
        self.assertLess(output['coefficient_of_variation'], 0.1)
        self.assertIn('Position-independent', output['interpretation'])

    def test_coefficient_of_variation_calculation(self):
        """Test correct calculation of coefficient of variation."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.2},
            'tokens_0_to_512': {'global_conflict': 0.4},
            'tokens_0_to_1024': {'global_conflict': 0.6}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        scores = [0.2, 0.4, 0.6]
        expected_cv = np.std(scores) / np.mean(scores)
        self.assertAlmostEqual(output['coefficient_of_variation'], expected_cv, places=5)

    def test_correlation_analysis(self):
        """Test Pearson and Spearman correlation calculations."""
        # Create data with strong positive correlation
        results = {
            'tokens_0_to_256': {'global_conflict': 0.1},
            'tokens_0_to_512': {'global_conflict': 0.25},
            'tokens_0_to_1024': {'global_conflict': 0.45},
            'tokens_0_to_1536': {'global_conflict': 0.6},
            'tokens_0_to_2048': {'global_conflict': 0.8}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        # Check correlation values
        self.assertIsNotNone(output['correlation']['pearson_r'])
        self.assertIsNotNone(output['correlation']['spearman_r'])
        self.assertGreater(output['correlation']['pearson_r'], 0.9)  # Strong positive correlation
        self.assertLess(output['correlation']['pearson_p'], 0.05)  # Significant

    def test_confidence_calculation_high_variance(self):
        """Test that high variance leads to high confidence in position dependency."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.1},
            'tokens_0_to_512': {'global_conflict': 0.9},
            'tokens_0_to_1024': {'global_conflict': 0.2},
            'tokens_0_to_1536': {'global_conflict': 1.8}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        # High variance should lead to higher confidence
        self.assertGreater(output['confidence'], 0.3)
        self.assertGreater(output['coefficient_of_variation'], 0.5)

    def test_token_count_extraction(self):
        """Test robust extraction of token counts from config names."""
        # Standard format
        results = {
            'tokens_0_to_256': {'global_conflict': 0.3},
            'tokens_0_to_1024': {'global_conflict': 0.5}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        # Should extract 256 and 1024
        self.assertEqual(output['num_data_points'], 2)
        self.assertIn('256T', output['interpretation'])
        self.assertIn('1024T', output['interpretation'])

    def test_interpretation_all_data_points(self):
        """Test that interpretation shows all data points when <= 6."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.1},
            'tokens_0_to_512': {'global_conflict': 0.2},
            'tokens_0_to_1024': {'global_conflict': 0.3},
            'tokens_0_to_1536': {'global_conflict': 0.4}
        }

        output = self.gradient_analysis._check_position_dependency(results)

        # Should show all 4 points
        for token_count in ['256T', '512T', '1024T', '1536T']:
            self.assertIn(token_count, output['interpretation'])

    def test_interpretation_sampled_data_points(self):
        """Test that interpretation intelligently samples when > 6 data points."""
        results = {
            'tokens_0_to_256': {'global_conflict': 0.1},
            'tokens_0_to_512': {'global_conflict': 0.2},
            'tokens_0_to_1024': {'global_conflict': 0.3},
            'tokens_0_to_1536': {'global_conflict': 0.4},
            'tokens_0_to_2048': {'global_conflict': 0.5},
            'tokens_0_to_3072': {'global_conflict': 0.6}
        }

        # Add more to trigger sampling
        for i in range(4000, 5000, 100):
            results[f'tokens_0_to_{i}'] = {'global_conflict': 0.7}

        output = self.gradient_analysis._check_position_dependency(results)

        # Should have ellipsis for sampling
        self.assertIn('...', output['interpretation'])

    def test_confidence_levels_interpretation(self):
        """Test correct interpretation of confidence levels."""
        # High confidence case
        results_high = {
            'tokens_0_to_256': {'global_conflict': 0.1},
            'tokens_0_to_512': {'global_conflict': 0.5},
            'tokens_0_to_1024': {'global_conflict': 0.9}
        }
        output_high = self.gradient_analysis._check_position_dependency(results_high)
        self.assertIn('High confidence', output_high['interpretation'])

        # Low confidence case - needs truly low variance and no clear pattern
        results_low = {
            'tokens_0_to_256': {'global_conflict': 0.5},
            'tokens_0_to_512': {'global_conflict': 0.502}  # Very minimal change
        }
        output_low = self.gradient_analysis._check_position_dependency(results_low)
        # With such minimal variance, confidence should be low or moderate
        self.assertTrue(
            'Low confidence' in output_low['interpretation'] or
            'Moderate confidence' in output_low['interpretation']
        )

    def test_alternative_key_names(self):
        """Test that function works with both 'global_conflict' and 'conflict_score' keys."""
        results = {
            'tokens_0_to_256': {'conflict_score': 0.3},  # Using conflict_score
            'tokens_0_to_512': {'global_conflict': 0.5},  # Using global_conflict
            'tokens_0_to_1024': {'conflict_score': 0.7}   # Using conflict_score
        }

        output = self.gradient_analysis._check_position_dependency(results)

        self.assertEqual(output['num_data_points'], 3)
        self.assertAlmostEqual(output['mean_conflict'], 0.5, places=2)

    def test_floating_point_precision(self):
        """Test handling of floating point precision in monotonicity checks."""
        # Values that might cause floating point comparison issues
        results = {
            'tokens_0_to_256': {'global_conflict': 0.1 + 0.1 + 0.1},  # 0.3 with potential rounding
            'tokens_0_to_512': {'global_conflict': 0.3},
            'tokens_0_to_1024': {'global_conflict': 0.3 + 1e-7}  # Very small increase
        }

        output = self.gradient_analysis._check_position_dependency(results)

        # Should handle floating point comparison correctly
        self.assertEqual(output['pattern'], 'increasing')  # Due to weak monotonicity with tolerance


if __name__ == '__main__':
    unittest.main(verbosity=2)