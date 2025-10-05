"""
Unit tests for the _compute_consensus_score function in GradientAnalysis.

Tests cover:
- Division by zero protection
- Input validation (NaN, Inf, negative weights)
- Score range validation and clamping
- Partial failure handling
- Edge cases (single config, all failures, etc.)
- Numerical precision and stability
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from GradientAnalysis import GradientAnalysis


class TestGradientConsensusScore(unittest.TestCase):
    """Test suite for _compute_consensus_score function."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = GradientAnalysis()

    def test_normal_weighted_average(self):
        """Test normal weighted average computation."""
        results = {
            'config1': {'overall_conflict': 0.3, 'weight': 0.5},
            'config2': {'overall_conflict': 0.7, 'weight': 0.3},
            'config3': {'overall_conflict': 0.5, 'weight': 0.2}
        }
        # Expected: (0.3*0.5 + 0.7*0.3 + 0.5*0.2) / (0.5 + 0.3 + 0.2)
        #         = (0.15 + 0.21 + 0.10) / 1.0 = 0.46
        score = self.analyzer._compute_consensus_score(results)
        self.assertAlmostEqual(score, 0.46, places=4)

    def test_division_by_zero_protection(self):
        """Test that division by zero is handled gracefully."""
        # All weights are zero
        results = {
            'config1': {'overall_conflict': 0.5, 'weight': 0.0},
            'config2': {'overall_conflict': 0.7, 'weight': 0.0}
        }
        score = self.analyzer._compute_consensus_score(results)
        self.assertTrue(np.isnan(score))

    def test_negative_weight_filtering(self):
        """Test that negative weights are filtered out."""
        results = {
            'config1': {'overall_conflict': 0.5, 'weight': 0.5},
            'config2': {'overall_conflict': 0.9, 'weight': -0.3},  # Invalid
            'config3': {'overall_conflict': 0.3, 'weight': 0.5}
        }
        # Only config1 and config3 should be used
        score = self.analyzer._compute_consensus_score(results)
        expected = (0.5 * 0.5 + 0.3 * 0.5) / 1.0
        self.assertAlmostEqual(score, expected, places=4)

    def test_nan_score_filtering(self):
        """Test that NaN scores are filtered out."""
        results = {
            'config1': {'overall_conflict': 0.4, 'weight': 0.3},
            'config2': {'overall_conflict': float('nan'), 'weight': 0.4},  # Invalid
            'config3': {'overall_conflict': 0.6, 'weight': 0.3}
        }
        score = self.analyzer._compute_consensus_score(results)
        expected = (0.4 * 0.3 + 0.6 * 0.3) / 0.6
        self.assertAlmostEqual(score, expected, places=4)

    def test_inf_score_filtering(self):
        """Test that infinite scores are filtered out."""
        results = {
            'config1': {'overall_conflict': 0.2, 'weight': 0.5},
            'config2': {'overall_conflict': float('inf'), 'weight': 0.3},  # Invalid
            'config3': {'overall_conflict': float('-inf'), 'weight': 0.2}  # Invalid
        }
        score = self.analyzer._compute_consensus_score(results)
        # Only config1 should be valid
        self.assertAlmostEqual(score, 0.2, places=4)

    def test_score_clamping(self):
        """Test that out-of-range scores are clamped to [0, 1]."""
        results = {
            'config1': {'overall_conflict': -0.5, 'weight': 0.3},  # Will be clamped to 0
            'config2': {'overall_conflict': 1.5, 'weight': 0.3},   # Will be clamped to 1
            'config3': {'overall_conflict': 0.5, 'weight': 0.4}
        }
        with patch('GradientAnalysis.logger') as mock_logger:
            score = self.analyzer._compute_consensus_score(results)
            # Verify warnings were logged for out-of-range scores
            self.assertEqual(mock_logger.warning.call_count, 2)

        # All scores should be used after clamping
        expected = (0.0 * 0.3 + 1.0 * 0.3 + 0.5 * 0.4) / 1.0
        self.assertAlmostEqual(score, expected, places=4)
        # Verify final result is in [0, 1]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_error_entries_skipped(self):
        """Test that entries with 'error' key are skipped."""
        results = {
            'config1': {'overall_conflict': 0.4, 'weight': 0.5},
            'config2': {'error': 'Computation failed'},  # Should be skipped
            'config3': {'overall_conflict': 0.6, 'weight': 0.5}
        }
        score = self.analyzer._compute_consensus_score(results)
        expected = (0.4 * 0.5 + 0.6 * 0.5) / 1.0
        self.assertAlmostEqual(score, expected, places=4)

    def test_all_configs_failed(self):
        """Test behavior when all configurations fail."""
        results = {
            'config1': {'error': 'OOM'},
            'config2': {'error': 'Computation failed'},
            'config3': {'error': 'Invalid input'}
        }
        with patch('GradientAnalysis.logger') as mock_logger:
            score = self.analyzer._compute_consensus_score(results)
            self.assertTrue(np.isnan(score))
            # Verify warning was logged
            mock_logger.warning.assert_called_once()

    def test_single_valid_config(self):
        """Test with only one valid configuration."""
        results = {
            'config1': {'overall_conflict': 0.75, 'weight': 0.42}
        }
        score = self.analyzer._compute_consensus_score(results)
        self.assertAlmostEqual(score, 0.75, places=4)

    def test_empty_results(self):
        """Test with empty results dictionary."""
        results = {}
        score = self.analyzer._compute_consensus_score(results)
        self.assertTrue(np.isnan(score))

    def test_missing_fields(self):
        """Test configs missing required fields are skipped."""
        results = {
            'config1': {'overall_conflict': 0.5},  # Missing weight
            'config2': {'weight': 0.3},  # Missing overall_conflict
            'config3': {'overall_conflict': 0.7, 'weight': 1.0}  # Valid
        }
        score = self.analyzer._compute_consensus_score(results)
        # Only config3 should be used
        self.assertAlmostEqual(score, 0.7, places=4)

    def test_very_small_weights(self):
        """Test numerical stability with very small weights."""
        results = {
            'config1': {'overall_conflict': 0.5, 'weight': 1e-11},
            'config2': {'overall_conflict': 0.7, 'weight': 1e-12}
        }
        # Total weight below threshold, should return NaN
        score = self.analyzer._compute_consensus_score(results)
        self.assertTrue(np.isnan(score))

    def test_weight_normalization(self):
        """Test that weights don't need to sum to 1 for correct computation."""
        # Using weights that sum to 2.0
        results = {
            'config1': {'overall_conflict': 0.2, 'weight': 0.8},
            'config2': {'overall_conflict': 0.8, 'weight': 1.2}
        }
        score = self.analyzer._compute_consensus_score(results)
        expected = (0.2 * 0.8 + 0.8 * 1.2) / 2.0
        self.assertAlmostEqual(score, expected, places=4)

    def test_numerical_precision(self):
        """Test numerical precision with many small contributions."""
        # Create many configs with small weights
        results = {}
        n_configs = 100
        for i in range(n_configs):
            results[f'config{i}'] = {
                'overall_conflict': 0.5 + 0.001 * i,
                'weight': 1.0 / n_configs
            }

        score = self.analyzer._compute_consensus_score(results)
        # Score should be close to average of conflicts
        expected_avg = sum(0.5 + 0.001 * i for i in range(n_configs)) / n_configs
        self.assertAlmostEqual(score, expected_avg, places=4)
        # Result should be in valid range
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_mixed_valid_invalid_configs(self):
        """Test with a mix of valid and invalid configurations."""
        results = {
            'valid1': {'overall_conflict': 0.3, 'weight': 0.2},
            'invalid1': {'overall_conflict': float('nan'), 'weight': 0.15},
            'valid2': {'overall_conflict': 0.7, 'weight': 0.3},
            'error1': {'error': 'Failed'},
            'invalid2': {'overall_conflict': 0.5, 'weight': -1.0},
            'valid3': {'overall_conflict': 0.5, 'weight': 0.5}
        }

        score = self.analyzer._compute_consensus_score(results)
        # Only valid1, valid2, valid3 should be used
        expected = (0.3 * 0.2 + 0.7 * 0.3 + 0.5 * 0.5) / (0.2 + 0.3 + 0.5)
        self.assertAlmostEqual(score, expected, places=4)


if __name__ == '__main__':
    unittest.main()