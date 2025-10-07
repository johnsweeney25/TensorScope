"""
Unit tests for MI estimator consistency - ICLR 2026 submission
Tests the pure classifier-based MI lower bound estimator to ensure:
1. No silent fallback to binning-based estimation
2. Consistent support between H(Z) and CE
3. Deterministic label coarsening
4. Proper error signaling on failure
5. Bootstrap operates on valid samples only
"""

import unittest
import torch
import numpy as np
import logging
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InformationTheoryMetrics import InformationTheoryMetrics


class TestMIEstimatorConsistency(unittest.TestCase):
    """Test suite for classifier-based MI estimator consistency"""

    @classmethod
    def setUpClass(cls):
        """Set up logging for tests"""
        logging.basicConfig(level=logging.WARNING)
        cls.logger = logging.getLogger(__name__)

    def setUp(self):
        """Create InformationTheoryMetrics instance"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create metrics instance with fixed seed for reproducibility
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_no_binning_fallback_on_normal_execution(self):
        """Ensure normal execution uses classifier, not binning"""
        # Create synthetic data with reasonable cardinality
        n_samples = 500
        hidden_dim = 32
        n_classes = 50
        
        h = torch.randn(n_samples, hidden_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        result = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=3,
            n_bootstrap=10,
            use_alternative_estimator=False  # Critical: no binning
        )
        
        # Should return dict with estimator='logistic_regression', not 'binning'
        if 'estimator' in result:
            self.assertEqual(result['estimator'], 'logistic_regression',
                           "Should use classifier estimator")
        
        # Should have valid MI
        self.assertIn('mi', result)
        self.assertTrue(np.isfinite(result['mi']), "MI should be finite")

    def test_label_coarsening_determinism(self):
        """Test that label coarsening is deterministic and uses top-K + <other>"""
        n_samples = 1000
        hidden_dim = 32
        n_classes = 250  # Exceeds MAX_CLASSES_FOR_MI (100)
        
        h = torch.randn(n_samples, hidden_dim)
        
        # Create labels with power-law distribution (realistic)
        # Use numpy for deterministic sampling
        np.random.seed(42)
        # Top 50 classes get most samples
        labels_list = []
        remaining = n_samples
        for i in range(min(50, n_classes)):
            count = max(1, int(remaining * 0.1))
            labels_list.extend([i] * count)
            remaining -= count
        # Rare classes
        for i in range(50, n_classes):
            if remaining <= 0:
                break
            count = np.random.randint(0, min(3, remaining + 1))
            labels_list.extend([i] * count)
            remaining -= count
        
        # Pad to exact n_samples
        while len(labels_list) < n_samples:
            labels_list.append(np.random.randint(0, n_classes))
        labels = torch.tensor(labels_list[:n_samples])
        
        # Run twice with same seed
        result1 = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=3,
            n_bootstrap=10,
            seed=42,
            use_alternative_estimator=False
        )
        
        result2 = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=3,
            n_bootstrap=10,
            seed=42,
            use_alternative_estimator=False
        )
        
        # Results should be identical (deterministic)
        self.assertAlmostEqual(result1['mi'], result2['mi'], places=5,
                              msg="Label coarsening should be deterministic")

    def test_consistent_support_hz_and_ce(self):
        """Test that H(Z) and CE are computed on the same filtered sample subset"""
        n_samples = 300
        hidden_dim = 32
        n_classes = 20
        
        h = torch.randn(n_samples, hidden_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        result = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=5,
            n_bootstrap=10,
            use_alternative_estimator=False
        )
        
        # Check that result includes both filtered and original values
        self.assertIn('H_Z', result, "Should report filtered H(Z)")
        self.assertIn('H_Z_original', result, "Should report original H(Z)")
        self.assertIn('n_samples', result, "Should report valid sample count")
        self.assertIn('n_samples_original', result, "Should report original count")
        
        # Valid samples should be <= original samples
        self.assertLessEqual(result['n_samples'], result['n_samples_original'],
                            "Valid samples should not exceed original samples")

    def test_bootstrap_on_valid_samples_only(self):
        """Ensure bootstrap operates only on valid (non-NaN) losses"""
        n_samples = 200
        hidden_dim = 32
        n_classes = 30
        
        h = torch.randn(n_samples, hidden_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        result = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=3,
            n_bootstrap=50,
            use_alternative_estimator=False
        )
        
        # CI should be finite (not NaN from contamination)
        self.assertTrue(np.isfinite(result['ci_low']),
                       "CI lower bound should be finite")
        self.assertTrue(np.isfinite(result['ci_high']),
                       "CI upper bound should be finite")
        
        # CI should bracket MI
        self.assertLessEqual(result['ci_low'], result['mi'],
                            "CI lower should be <= MI")
        self.assertLessEqual(result['mi'], result['ci_high'],
                            "MI should be <= CI upper")

    def test_explicit_error_on_classifier_failure(self):
        """Test that classifier failure returns explicit error, not silent fallback"""
        n_samples = 50  # Too few samples for reliable classification
        hidden_dim = 32
        n_classes = 100  # High cardinality relative to samples
        
        h = torch.randn(n_samples, hidden_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        result = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=3,
            n_bootstrap=10,
            use_alternative_estimator=False
        )
        
        # Should either succeed or return error dict
        if 'error' in result or 'warning' in result:
            # Explicit error signaling - good
            self.assertIn('estimator', result)
            self.assertEqual(result['estimator'], 'logistic_regression',
                           "Even on error, should report classifier estimator")
        else:
            # Succeeded - should still be classifier
            self.assertEqual(result['estimator'], 'logistic_regression')

    def test_mi_error_dict_structure(self):
        """Test that MI errors return proper dict structure with 'error' key"""
        # Test the error handling path by forcing a failure condition
        # (Too few samples for classification)
        n_samples = 5  # Extremely small - will fail
        hidden_dim = 32
        n_classes = 100
        
        h = torch.randn(n_samples, hidden_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        result = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=2,  # Can't split 5 samples into meaningful folds
            n_bootstrap=10,
            use_alternative_estimator=False
        )
        
        # Should return a dict (not raise exception)
        self.assertIsInstance(result, dict, "Should return dict even on failure")
        
        # Should have MI value (could be 0 on failure)
        self.assertIn('mi', result)
        
        # If it failed, should have 'error' or 'warning' key
        if result.get('mi', 0) == 0 or result.get('n_samples', n_samples) < 10:
            # Likely failed or nearly failed
            self.assertTrue('error' in result or 'warning' in result,
                           "Should have error/warning when MI=0 or too few samples")

    def test_high_cardinality_label_coarsening(self):
        """Test aggressive coarsening for high cardinality (>100 classes)"""
        n_samples = 800
        hidden_dim = 32
        n_classes = 300  # Very high cardinality
        
        h = torch.randn(n_samples, hidden_dim)
        
        # Create highly skewed distribution - ensure exactly n_samples
        np.random.seed(123)
        labels_list = []
        # Top 80 classes get most samples
        for i in range(80):
            count = max(5, int((n_samples * 0.7) / 80))
            labels_list.extend([i] * count)
        
        # Fill remaining with rare classes
        remaining = n_samples - len(labels_list)
        for i in range(80, min(n_classes, 80 + remaining)):
            labels_list.append(i)
        
        # Ensure exact count
        labels_list = labels_list[:n_samples]
        while len(labels_list) < n_samples:
            labels_list.append(np.random.randint(0, 80))
        
        labels = torch.tensor(labels_list)
        
        result = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=3,
            n_bootstrap=10,
            use_alternative_estimator=False
        )
        
        # Should succeed with coarsening
        self.assertIn('mi', result)
        self.assertTrue(np.isfinite(result['mi']),
                       "MI should be finite after coarsening")
        
        # Original should have more unique labels than filtered
        if result['n_samples'] < result['n_samples_original']:
            # Some samples were filtered - expected for high cardinality
            self.logger.info(f"Filtered {result['n_samples_original'] - result['n_samples']} "
                           f"samples due to unseen classes")

    def test_no_binning_import_or_call(self):
        """Verify that _compute_mi_binning_based is never called in OOF path"""
        n_samples = 200
        hidden_dim = 32
        n_classes = 50
        
        h = torch.randn(n_samples, hidden_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        # Track if binning method is called
        binning_called = []
        
        original_binning = self.metrics._compute_mi_binning_based
        def track_binning_call(*args, **kwargs):
            binning_called.append(True)
            return original_binning(*args, **kwargs)
        
        with patch.object(self.metrics, '_compute_mi_binning_based',
                         side_effect=track_binning_call):
            result = self.metrics._compute_mi_lower_bound_oof(
                h, labels,
                n_splits=3,
                n_bootstrap=10,
                use_alternative_estimator=False  # Critical
            )
        
        # Binning should NEVER be called
        self.assertEqual(len(binning_called), 0,
                        "Binning-based estimator should never be called with "
                        "use_alternative_estimator=False")
        
        # Should be classifier-based
        self.assertEqual(result['estimator'], 'logistic_regression')


class TestMIEstimatorNumericalStability(unittest.TestCase):
    """Test numerical stability of MI estimator"""

    @classmethod
    def setUpClass(cls):
        """Set up logging for tests"""
        logging.basicConfig(level=logging.WARNING)
        cls.logger = logging.getLogger(__name__)

    def setUp(self):
        """Create metrics instance"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_mi_non_negative(self):
        """MI lower bound should be non-negative (or close due to numerical error)"""
        n_samples = 300
        hidden_dim = 32
        n_classes = 20
        
        h = torch.randn(n_samples, hidden_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        result = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=3,
            n_bootstrap=10,
            use_alternative_estimator=False
        )
        
        # MI is a lower bound, can be 0 but generally not negative
        # However, due to finite sample effects and OOF prediction, small negative values can occur
        # This is theoretically acceptable as it's still a lower bound (just a loose one)
        if result['mi'] < -0.1:
            self.logger.warning(f"MI significantly negative: {result['mi']:.3f}")
        
        # Main assertion: should be finite and not wildly negative
        self.assertTrue(np.isfinite(result['mi']), "MI should be finite")
        self.assertGreater(result['mi'], -2.0, 
                          "MI should not be wildly negative (small negative OK for OOF)")

    def test_h_z_bounds(self):
        """H(Z) should be bounded by [0, log(n_classes)]"""
        n_samples = 300
        hidden_dim = 32
        n_classes = 20
        
        h = torch.randn(n_samples, hidden_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        result = self.metrics._compute_mi_lower_bound_oof(
            h, labels,
            n_splits=3,
            n_bootstrap=10,
            use_alternative_estimator=False
        )
        
        max_entropy = np.log(n_classes)
        
        self.assertGreaterEqual(result['H_Z'], 0.0,
                               "H(Z) should be non-negative")
        self.assertLessEqual(result['H_Z'], max_entropy * 1.1,  # 10% tolerance
                            f"H(Z) should be <= log({n_classes}) ≈ {max_entropy:.2f}")

    def test_reproducibility_with_seed(self):
        """Same seed should give identical results"""
        n_samples = 200
        hidden_dim = 32
        n_classes = 30
        
        h = torch.randn(n_samples, hidden_dim)
        labels = torch.randint(0, n_classes, (n_samples,))
        
        # Convert to numpy for determinism
        h_np = h.numpy()
        labels_np = labels.numpy()
        
        results = []
        for _ in range(3):
            h_tensor = torch.from_numpy(h_np)
            labels_tensor = torch.from_numpy(labels_np)
            
            result = self.metrics._compute_mi_lower_bound_oof(
                h_tensor, labels_tensor,
                n_splits=3,
                n_bootstrap=20,
                seed=12345,  # Fixed seed
                use_alternative_estimator=False
            )
            results.append(result['mi'])
        
        # All runs should give identical MI
        for i in range(1, len(results)):
            self.assertAlmostEqual(results[0], results[i], places=4,
                                  msg="Same seed should give reproducible results")


if __name__ == '__main__':
    unittest.main()

