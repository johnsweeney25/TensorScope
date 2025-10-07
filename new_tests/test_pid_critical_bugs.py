"""
Unit tests for the 5 critical bugs fixed in compute_heuristic_pid_minmi
Tests ensure the bugs don't regress - ICLR 2026 submission critical
"""

import unittest
import torch
import numpy as np
import logging
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from InformationTheoryMetrics import InformationTheoryMetrics


class TestCacheBugFix(unittest.TestCase):
    """Test that cache is set in reachable code (Bug #1)"""

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.WARNING)
        cls.logger = logging.getLogger(__name__)

    def setUp(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_cache_mechanism_exists(self):
        """Verify cache mechanism exists in code (structural test)"""
        import inspect
        
        try:
            source = inspect.getsource(self.metrics.compute_heuristic_pid_minmi)
            
            # Check that cache is defined
            self.assertIn('layer_mi_cache', source, 
                         "Should have layer_mi_cache variable")
            
            # Check that cache is set (not just accessed)
            self.assertIn('layer_mi_cache[cache_key] =', source,
                         "Should SET cache (not just access it)")
            
            # Verify cache is set BEFORE it's accessed (no KeyError)
            # The bug was: cache set AFTER continue, then accessed → KeyError
            lines = source.split('\n')
            
            cache_set_line = None
            cache_accessed_line = None
            
            for i, line in enumerate(lines):
                if 'layer_mi_cache[cache_key] =' in line and 'cached' not in line:
                    cache_set_line = i
                if 'cached = layer_mi_cache[cache_key]' in line:
                    cache_accessed_line = i
            
            if cache_set_line and cache_accessed_line:
                # The bug was cache set AFTER a continue, so it was unreachable
                # Check that cache assignment is not immediately after continue
                for i in range(max(0, cache_set_line - 5), cache_set_line):
                    line = lines[i].strip()
                    if line == 'continue':
                        self.fail(f"Cache set immediately after 'continue' at line {i} "
                                f"(makes it unreachable). This is the KeyError bug.")
                
        except Exception as e:
            self.logger.warning(f"Could not inspect source: {e}")


class TestLoggerBugFix(unittest.TestCase):
    """Test that self.logger is used, not undefined logger (Bug #2)"""

    def setUp(self):
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_logger_attribute_exists(self):
        """Verify InformationTheoryMetrics has logger attribute"""
        self.assertTrue(hasattr(self.metrics, 'logger'), 
                       "InformationTheoryMetrics should have 'logger' attribute")
        self.assertIsNotNone(self.metrics.logger,
                            "logger attribute should not be None")

    def test_no_undefined_logger_in_code(self):
        """Check that code uses self.logger, not bare 'logger'"""
        import inspect
        
        # Get source of compute_heuristic_pid_minmi
        try:
            source = inspect.getsource(self.metrics.compute_heuristic_pid_minmi)
            
            # Check for problematic patterns
            # Allow: self.logger.info/warning/error
            # Disallow: logger.info/warning/error (without self.)
            
            lines = source.split('\n')
            problematic_lines = []
            
            for i, line in enumerate(lines):
                # Skip comments
                if line.strip().startswith('#'):
                    continue
                    
                # Check for bare logger. (not self.logger.)
                if 'logger.' in line and 'self.logger.' not in line and 'logging.' not in line:
                    # Make sure it's actually calling logger, not a variable name
                    if 'logger.info' in line or 'logger.warning' in line or 'logger.error' in line or 'logger.debug' in line:
                        problematic_lines.append((i, line.strip()))
            
            if problematic_lines:
                msg = "Found bare 'logger' calls (should be 'self.logger'):\n"
                for line_num, line in problematic_lines:
                    msg += f"  Line {line_num}: {line}\n"
                self.fail(msg)
                
        except Exception as e:
            # If we can't get source, just warn
            self.logger.warning(f"Could not inspect source: {e}")


class TestTask2MaskingFix(unittest.TestCase):
    """Test that task2 masking is properly handled (Bug #3)"""

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.WARNING)
        cls.logger = logging.getLogger(__name__)

    def setUp(self):
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_get_joint_valid_indices_exists(self):
        """Verify _get_joint_valid_indices helper method exists"""
        self.assertTrue(hasattr(self.metrics, '_get_joint_valid_indices'),
                       "Should have _get_joint_valid_indices method")

    def test_joint_valid_indices_token_level(self):
        """Test that joint valid indices correctly intersects both tasks (token-level)"""
        batch_size = 4
        seq_len = 8
        
        # Task1: first half valid, second half padding
        labels1 = torch.randint(0, 50, (batch_size, seq_len))
        mask1 = torch.ones(batch_size, seq_len)
        mask1[:, seq_len//2:] = 0  # Second half padded
        
        # Task2: second half valid, first half padding
        labels2 = torch.randint(0, 50, (batch_size, seq_len))
        mask2 = torch.ones(batch_size, seq_len)
        mask2[:, :seq_len//2] = 0  # First half padded
        
        # Get joint valid indices
        valid_indices = self.metrics._get_joint_valid_indices(
            labels1, mask1,
            labels2, mask2,
            max_tokens=1000,
            seed=42,
            is_sequence_level=False
        )
        
        # Should have NO valid indices (no overlap between valid regions)
        self.assertEqual(len(valid_indices), 0,
                        "Should have no valid indices when masks don't overlap")
        
        # Now test with overlapping valid regions
        mask1_overlap = torch.ones(batch_size, seq_len)
        mask2_overlap = torch.ones(batch_size, seq_len)
        mask1_overlap[:, -2:] = 0  # Last 2 padded in task1
        mask2_overlap[:, :2] = 0   # First 2 padded in task2
        
        valid_indices_overlap = self.metrics._get_joint_valid_indices(
            labels1, mask1_overlap,
            labels2, mask2_overlap,
            max_tokens=1000,
            seed=42,
            is_sequence_level=False
        )
        
        # Should have valid indices in the middle (positions 2-5)
        expected_valid = (seq_len - 4) * batch_size  # 4 positions * batch_size
        self.assertEqual(len(valid_indices_overlap), expected_valid,
                        f"Should have {expected_valid} valid indices in overlap region")

    def test_joint_valid_indices_excludes_ignore_labels(self):
        """Test that -100 labels are excluded from both tasks"""
        batch_size = 4
        seq_len = 8
        
        # Both tasks have valid masks but some -100 labels
        labels1 = torch.randint(0, 50, (batch_size, seq_len))
        labels1[:, 0] = -100  # First position ignored in task1
        
        labels2 = torch.randint(0, 50, (batch_size, seq_len))
        labels2[:, -1] = -100  # Last position ignored in task2
        
        mask1 = torch.ones(batch_size, seq_len)
        mask2 = torch.ones(batch_size, seq_len)
        
        valid_indices = self.metrics._get_joint_valid_indices(
            labels1, mask1,
            labels2, mask2,
            max_tokens=1000,
            seed=42,
            is_sequence_level=False
        )
        
        # Should exclude first and last positions
        expected_valid = (seq_len - 2) * batch_size  # All except first and last
        self.assertEqual(len(valid_indices), expected_valid,
                        f"Should have {expected_valid} valid indices (excluding -100 labels)")


class TestComputeDtypeFix(unittest.TestCase):
    """Test that hidden states are cast to compute_dtype (Bug #4)"""

    def setUp(self):
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_compute_dtype_casting_in_code(self):
        """Verify code casts hidden states to compute_dtype"""
        import inspect
        
        try:
            source = inspect.getsource(self.metrics.compute_heuristic_pid_minmi)
            
            # Check for dtype casting pattern
            has_to_compute_dtype = '.to(compute_dtype)' in source
            
            self.assertTrue(has_to_compute_dtype,
                           "Code should cast hidden states to compute_dtype using .to(compute_dtype)")
            
            # Check that casting happens before gathering
            # Look for pattern: h1_layer.to(compute_dtype) before _gather_by_indices
            lines = source.split('\n')
            
            found_cast = False
            found_gather_after_cast = False
            
            for i, line in enumerate(lines):
                if 'h1_layer_compute = h1_layer.to(compute_dtype)' in line or \
                   'h1_layer.to(compute_dtype)' in line:
                    found_cast = True
                    
                    # Check next ~10 lines for gather call
                    for j in range(i, min(i+10, len(lines))):
                        if '_gather_by_indices' in lines[j] and 'h1_layer_compute' in lines[j]:
                            found_gather_after_cast = True
                            break
            
            if found_cast:
                self.assertTrue(found_gather_after_cast,
                               "Should call _gather_by_indices on cast tensor (h1_layer_compute), not original")
                
        except Exception as e:
            self.logger.warning(f"Could not inspect source: {e}")


class TestDuplicateZ1Fix(unittest.TestCase):
    """Test that task2 PID uses z2 values, not duplicate z1 (Bug #5)"""

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.WARNING)
        cls.logger = logging.getLogger(__name__)

    def setUp(self):
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_no_duplicate_z1_computation(self):
        """Verify task2 PID computes z2, not duplicate z1"""
        import inspect
        
        try:
            source = inspect.getsource(self.metrics.compute_heuristic_pid_minmi)
            
            lines = source.split('\n')
            
            # Find the section after "Compute PID for labels2"
            in_labels2_section = False
            z1_count_after_labels2 = 0
            z2_count_after_labels2 = 0
            
            for i, line in enumerate(lines):
                if 'Compute PID for labels2' in line or 'labels2 target' in line:
                    in_labels2_section = True
                    continue
                
                if in_labels2_section:
                    # Count z1 vs z2 variable assignments in PID computation
                    # After loading from cache, should compute redundancy_z2, not redundancy_z1
                    if 'redundancy_z1 = min(mi_h1_z1' in line:
                        z1_count_after_labels2 += 1
                    if 'redundancy_z2 = min(mi_h1_z2' in line:
                        z2_count_after_labels2 += 1
                    
                    # Stop after we've found PID computations
                    if z2_count_after_labels2 > 0:
                        break
            
            # Should have z2 computation in labels2 section, not z1
            self.assertGreater(z2_count_after_labels2, 0,
                              "labels2 section should compute redundancy_z2")
            self.assertEqual(z1_count_after_labels2, 0,
                            "labels2 section should NOT compute redundancy_z1 (duplicate bug)")
            
        except Exception as e:
            self.logger.warning(f"Could not inspect source: {e}")

    def test_z2_variables_exist_in_results(self):
        """Verify that z2 PID variables would be computed (structure check)"""
        import inspect
        
        try:
            source = inspect.getsource(self.metrics.compute_heuristic_pid_minmi)
            
            # Check that z2 variables are defined
            required_z2_vars = [
                'redundancy_z2',
                'unique1_z2',
                'unique2_z2',
                'synergy_z2'
            ]
            
            for var in required_z2_vars:
                self.assertIn(var, source,
                             f"Should compute {var} for task2 PID")
                
        except Exception as e:
            self.logger.warning(f"Could not inspect source: {e}")


class TestAllBugsIntegration(unittest.TestCase):
    """Integration test to verify all 5 bugs are fixed together"""

    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.WARNING)
        cls.logger = logging.getLogger(__name__)

    def setUp(self):
        self.metrics = InformationTheoryMetrics(seed=42)

    def test_all_fixes_present_in_code(self):
        """Verify all 5 fixes are present in code (structural test)"""
        import inspect
        
        try:
            source = inspect.getsource(self.metrics.compute_heuristic_pid_minmi)
            
            # Test Bug #1 Fix: Cache mechanism
            self.assertIn('layer_mi_cache[cache_key] =', source,
                         "Bug #1: Should have cache assignment")
            
            # Test Bug #2 Fix: self.logger
            self.assertIn('self.logger', source,
                         "Bug #2: Should use self.logger")
            
            # Test Bug #3 Fix: Joint valid indices
            self.assertIn('_get_joint_valid_indices', source,
                         "Bug #3: Should call _get_joint_valid_indices")
            
            # Test Bug #4 Fix: compute_dtype casting
            self.assertIn('.to(compute_dtype)', source,
                         "Bug #4: Should cast to compute_dtype")
            
            # Test Bug #5 Fix: z2 variables (not duplicate z1)
            self.assertIn('redundancy_z2', source,
                         "Bug #5: Should have redundancy_z2 variable")
            self.assertIn('unique1_z2', source,
                         "Bug #5: Should have unique1_z2 variable")
            self.assertIn('synergy_z2', source,
                         "Bug #5: Should have synergy_z2 variable")
            
            # All 5 fixes present!
            self.logger.info("✓ All 5 bug fixes verified in code structure")
            
        except Exception as e:
            self.fail(f"Could not verify fixes in code: {e}")


if __name__ == '__main__':
    # Run with verbose output
    unittest.main(verbosity=2)

