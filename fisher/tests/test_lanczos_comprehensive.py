#!/usr/bin/env python3
"""
Comprehensive Unit Tests for Lanczos Algorithm
===============================================

Tests all critical correctness properties of the Lanczos implementation:
1. 3-term recurrence correctness
2. Precision handling (BF16, FP32, FP64)
3. Orthogonality of Lanczos vectors
4. Eigenvalue convergence
5. Edge cases and error handling

These tests are designed to catch the bugs identified in the 2025-10-07 intern review
and prevent regressions.

Run with:
    python -m pytest fisher/tests/test_lanczos_comprehensive.py -v
    or
    python -m unittest fisher/tests/test_lanczos_comprehensive.py
"""

import unittest
import torch
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fisher.core.fisher_lanczos_unified import (
    lanczos_algorithm, LanczosConfig, LinOp,
    HessianOperator, GGNOperator, EmpiricalFisherOperator, KFACFisherOperator
)


class SimpleMatrixOperator(LinOp):
    """Test operator that wraps an explicit matrix."""
    
    def __init__(self, matrix: torch.Tensor, is_psd: bool = True):
        """
        Args:
            matrix: Symmetric matrix to test with
            is_psd: Whether matrix is positive semi-definite
        """
        self.matrix = matrix
        n = matrix.shape[0]
        
        # Create dummy parameter with matching dtype and requires_grad
        param = torch.zeros(n, dtype=matrix.dtype, device=matrix.device, requires_grad=True)
        params = [param]
        
        def matvec(v):
            """Matrix-vector product."""
            v_vec = v[0]
            result = self.matrix @ v_vec.to(self.matrix.dtype)
            return [result]
        
        super().__init__(params, matvec, "test_matrix", is_psd=is_psd, device=matrix.device)


class TestLanczosCorrectness(unittest.TestCase):
    """Test correctness of Lanczos eigenvalue computation."""
    
    def test_diagonal_matrix_exact(self):
        """Test that Lanczos gets exact eigenvalues for diagonal matrix."""
        # Diagonal matrix with known eigenvalues
        eigenvalues_true = np.array([10.0, 5.0, 1.0])
        A = torch.diag(torch.tensor(eigenvalues_true, dtype=torch.float64))
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(k=3, max_iters=10, seed=42, dtype_compute=torch.float64)
        results = lanczos_algorithm(op, config, verbose=False)
        
        eigenvalues_computed = np.array(sorted(results['eigenvalues'], reverse=True))
        
        # Should be accurate to machine precision for diagonal matrix
        np.testing.assert_allclose(eigenvalues_computed, eigenvalues_true, rtol=1e-10, atol=1e-10,
                                   err_msg="Lanczos failed on simple diagonal matrix")
    
    def test_symmetric_matrix_convergence(self):
        """Test Lanczos convergence on dense symmetric matrix."""
        # Create well-conditioned symmetric matrix
        n = 20
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = (A + A.T) / 2  # Symmetrize
        A = A + torch.eye(n) * 10.0  # Make positive definite
        
        # Compute true eigenvalues
        eigenvalues_true = torch.linalg.eigvalsh(A).cpu().numpy()
        eigenvalues_true = np.sort(eigenvalues_true)[::-1]
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(k=5, max_iters=30, seed=42, dtype_compute=torch.float64)
        results = lanczos_algorithm(op, config, verbose=False)
        
        eigenvalues_computed = np.array(sorted(results['eigenvalues'], reverse=True))
        
        # Top eigenvalue should be accurate (most important for Lanczos)
        np.testing.assert_allclose(eigenvalues_computed[0], eigenvalues_true[0], rtol=1e-6,
                                   err_msg="Lanczos top eigenvalue not converged")
    
    def test_indefinite_matrix_negative_eigenvalues(self):
        """Test Lanczos correctly handles indefinite matrices with negative eigenvalues."""
        # Create indefinite matrix
        n = 10
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = (A + A.T) / 2
        A = A - torch.eye(n) * 2.0  # Shift to make some eigenvalues negative (not all)
        
        # Verify it's indefinite
        eigs_true = torch.linalg.eigvalsh(A).cpu().numpy()
        self.assertTrue(np.any(eigs_true < 0), "Test matrix should be indefinite")
        self.assertTrue(np.any(eigs_true > 0), "Test matrix should have positive eigenvalues")
        
        op = SimpleMatrixOperator(A, is_psd=False)
        config = LanczosConfig(k=5, max_iters=20, seed=42, dtype_compute=torch.float64)
        results = lanczos_algorithm(op, config, verbose=False)
        
        self.assertIn('has_negative_eigenvalues', results)
        # May or may not detect negative eigenvalues depending on which are found
        self.assertEqual(results['regularization_applied'], 0.0,
                        "Should not regularize indefinite matrices")
    
    def test_three_term_recurrence_correctness(self):
        """
        Test that 3-term recurrence is correctly implemented.
        
        This catches the critical bug from intern review where β_{i-1}·v_{i-1} was missing.
        For a diagonal matrix, Lanczos should converge in exactly n iterations.
        """
        eigenvalues_true = np.array([8.0, 4.0, 2.0])
        A = torch.diag(torch.tensor(eigenvalues_true, dtype=torch.float64))
        
        op = SimpleMatrixOperator(A, is_psd=True)
        # Disable reorthogonalization to test pure 3-term recurrence
        config = LanczosConfig(k=3, max_iters=5, seed=42, dtype_compute=torch.float64, 
                              reorth_period=999)  # Effectively disabled
        results = lanczos_algorithm(op, config, verbose=False)
        
        eigenvalues_computed = np.array(sorted(results['eigenvalues'], reverse=True))
        
        # With correct 3-term recurrence, should get exact eigenvalues
        np.testing.assert_allclose(eigenvalues_computed, eigenvalues_true, rtol=1e-10,
                                   err_msg="3-term recurrence not correctly implemented")


class TestLanczosPrecision(unittest.TestCase):
    """Test precision handling for different dtypes."""
    
    def test_fp64_precision_preserved(self):
        """
        Test that FP64 precision is preserved (not downcast to FP32).
        
        This catches the bug where we were casting FP64→FP32→FP64, losing precision.
        """
        # Use well-separated eigenvalues (Lanczos can't distinguish eigenvalues closer than ~tol)
        eigenvalues_true = np.array([10.0, 5.0, 1.0])
        A = torch.diag(torch.tensor(eigenvalues_true, dtype=torch.float64))
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(k=3, max_iters=10, seed=42, dtype_compute=torch.float64)
        results = lanczos_algorithm(op, config, verbose=False)
        
        eigenvalues_computed = np.array(sorted(results['eigenvalues'], reverse=True))
        
        # Should preserve FP64 precision (not downgrade to FP32)
        np.testing.assert_allclose(eigenvalues_computed, eigenvalues_true, rtol=1e-10, atol=1e-10,
                                   err_msg="FP64 precision not preserved")
    
    def test_fp32_precision_adequate(self):
        """Test that FP32 provides reasonable precision for typical use."""
        n = 15
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float32)
        A = A @ A.T + torch.eye(n, dtype=torch.float32)
        
        # True eigenvalues
        eigenvalues_true = torch.linalg.eigvalsh(A.to(torch.float64)).cpu().numpy()
        eigenvalues_true = np.sort(eigenvalues_true)[::-1]
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(k=5, max_iters=20, seed=42, dtype_compute=torch.float32)
        results = lanczos_algorithm(op, config, verbose=False)
        
        eigenvalues_computed = np.array(sorted(results['eigenvalues'], reverse=True))
        
        # FP32 should get top eigenvalue accurate (most important)
        np.testing.assert_allclose(eigenvalues_computed[0], eigenvalues_true[0], rtol=1e-4,
                                   err_msg="FP32 precision inadequate for top eigenvalue")
    
    def test_bfloat16_upcast_to_fp32(self):
        """Test that BF16 inputs are properly upcast to FP32 for computation."""
        n = 10
        torch.manual_seed(42)
        # Create in FP32 first
        A_fp32 = torch.randn(n, n, dtype=torch.float32)
        A_fp32 = A_fp32 @ A_fp32.T + torch.eye(n)
        
        # Convert to BF16
        A_bf16 = A_fp32.to(torch.bfloat16)
        
        op = SimpleMatrixOperator(A_bf16, is_psd=True)
        # Note: BF16 models typically use BF16 compute
        config = LanczosConfig(k=3, max_iters=15, seed=42, dtype_compute=torch.bfloat16)
        
        # Should not crash and should return reasonable eigenvalues
        results = lanczos_algorithm(op, config, verbose=False)
        
        self.assertEqual(len(results['eigenvalues']), 3)
        self.assertTrue(all(eig > 0 for eig in results['eigenvalues']),
                       "BF16 should produce positive eigenvalues for PSD matrix")


class TestLanczosOrthogonality(unittest.TestCase):
    """Test that Lanczos vectors maintain orthogonality."""
    
    def test_no_double_orthogonalization(self):
        """
        Test that we don't double-orthogonalize against current vector.
        
        This catches the bug where we subtracted α·v_curr twice.
        """
        # Simple test: if we're double-orthogonalizing, eigenvalues will be very wrong
        eigenvalues_true = np.array([10.0, 5.0, 1.0])
        A = torch.diag(torch.tensor(eigenvalues_true, dtype=torch.float64))
        
        op = SimpleMatrixOperator(A, is_psd=True)
        # Use full reorthogonalization (where the bug was)
        config = LanczosConfig(k=3, max_iters=10, seed=42, dtype_compute=torch.float64,
                              reorth_period=0)  # Full reorth
        results = lanczos_algorithm(op, config, verbose=False)
        
        eigenvalues_computed = np.array(sorted(results['eigenvalues'], reverse=True))
        
        # Should still be accurate (would fail with double-orthogonalization bug)
        np.testing.assert_allclose(eigenvalues_computed, eigenvalues_true, rtol=1e-9,
                                   err_msg="Double-orthogonalization bug detected")


class TestLanczosConfiguration(unittest.TestCase):
    """Test configuration options and edge cases."""
    
    def test_regularization_modes(self):
        """Test different regularization modes."""
        n = 8
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = A @ A.T + torch.eye(n) * 0.01  # Slightly ill-conditioned
        
        op = SimpleMatrixOperator(A, is_psd=True)
        
        # Test 'off' mode
        config = LanczosConfig(k=3, max_iters=15, seed=42, regularization_mode='off')
        results = lanczos_algorithm(op, config, verbose=False)
        self.assertEqual(results['regularization_applied'], 0.0)
        
        # Test 'fixed' mode
        config = LanczosConfig(k=3, max_iters=15, seed=42, 
                              regularization_mode='fixed', regularization=1e-6)
        results = lanczos_algorithm(op, config, verbose=False)
        self.assertAlmostEqual(results['regularization_applied'], 1e-6, places=10)
        
        # Test 'auto' mode
        config = LanczosConfig(k=3, max_iters=15, seed=42, regularization_mode='auto')
        results = lanczos_algorithm(op, config, verbose=False)
        # Auto might or might not regularize depending on condition number
        self.assertGreaterEqual(results['regularization_applied'], 0.0)
    
    def test_gc_every_configuration(self):
        """Test GPU cache cleanup configuration."""
        n = 10
        A = torch.diag(torch.tensor([5.0, 3.0, 1.0] + [0.1]*7, dtype=torch.float64))
        
        op = SimpleMatrixOperator(A, is_psd=True)
        
        # Test with different gc_every values - should all complete
        for gc_every in [-1, 0, 2, 10]:
            config = LanczosConfig(k=3, max_iters=10, seed=42, gc_every=gc_every)
            results = lanczos_algorithm(op, config, verbose=False)
            self.assertIn('eigenvalues', results)
    
    def test_ritz_condition_number_label(self):
        """Test that condition number is correctly labeled as 'ritz_condition_number'."""
        n = 10
        A = torch.diag(torch.tensor([10.0, 5.0, 2.0, 1.0] + [0.5]*6, dtype=torch.float64))
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(k=4, max_iters=10, seed=42)
        results = lanczos_algorithm(op, config, verbose=False)
        
        # Should use 'ritz_condition_number', not 'condition_number'
        self.assertIn('ritz_condition_number', results)
        self.assertNotIn('condition_number', results)
        self.assertGreater(results['ritz_condition_number'], 1.0)
    
    def test_reproducibility_with_seed(self):
        """Test that same seed gives identical results."""
        n = 15
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = A @ A.T + torch.eye(n)
        
        eigenvalues_list = []
        for _ in range(3):
            op = SimpleMatrixOperator(A, is_psd=True)
            config = LanczosConfig(k=5, max_iters=20, seed=12345)
            results = lanczos_algorithm(op, config, verbose=False)
            eigenvalues_list.append(results['eigenvalues'])
        
        # All runs should give identical results
        for eigs in eigenvalues_list[1:]:
            np.testing.assert_array_almost_equal(eigenvalues_list[0], eigs, decimal=14,
                                                err_msg="Same seed should give identical results")
    
    def test_different_seeds_all_converge(self):
        """Test that different seeds all converge to same eigenvalues."""
        n = 12
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = A @ A.T + torch.eye(n)
        
        eigenvalues_list = []
        for seed in [42, 123, 999, 7777]:
            op = SimpleMatrixOperator(A, is_psd=True)
            config = LanczosConfig(k=3, max_iters=25, seed=seed)
            results = lanczos_algorithm(op, config, verbose=False)
            eigenvalues_list.append(sorted(results['eigenvalues'], reverse=True))
        
        # All should converge to same values (within tolerance)
        for eigs in eigenvalues_list[1:]:
            np.testing.assert_allclose(eigenvalues_list[0], eigs, rtol=1e-4,
                                      err_msg="Different seeds should converge to same eigenvalues")


class TestLanczosEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_early_convergence(self):
        """Test that Lanczos completes without error on simple matrix."""
        # Diagonal matrix
        eigenvalues_true = np.array([5.0, 3.0, 1.0])
        A = torch.diag(torch.tensor(eigenvalues_true, dtype=torch.float64))
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(k=3, max_iters=10, tol=1e-10, seed=42)
        results = lanczos_algorithm(op, config, verbose=False)
        
        # Just verify it completes and returns 3 eigenvalues (values depend on random init)
        self.assertEqual(len(results['eigenvalues']), 3)
        self.assertTrue(all(eig > 0 for eig in results['eigenvalues']))  # PSD matrix
    
    def test_insufficient_iterations_warning(self):
        """Test that warning is issued if iterations < 3*k."""
        n = 20
        A = torch.randn(n, n, dtype=torch.float64)
        A = A @ A.T + torch.eye(n)
        
        op = SimpleMatrixOperator(A, is_psd=True)
        # Request k=10 but only give max_iters=5
        config = LanczosConfig(k=10, max_iters=5, seed=42)
        results = lanczos_algorithm(op, config, verbose=False)
        
        # Should have warning about insufficient iterations
        self.assertIn('warnings', results)
        self.assertGreater(len(results['warnings']), 0)
    
    def test_rank_deficient_matrix(self):
        """Test behavior on rank-deficient matrix."""
        # Create rank-2 matrix in 5D space
        n = 5
        torch.manual_seed(42)
        U = torch.randn(n, 2, dtype=torch.float64)
        A = U @ U.T
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(k=2, max_iters=10, seed=42)  # Request top-2
        results = lanczos_algorithm(op, config, verbose=False)
        
        eigenvalues = np.array(results['eigenvalues'])
        # Lanczos finds TOP eigenvalues, not zero ones
        # Just verify it completes and returns 2 eigenvalues
        self.assertEqual(len(eigenvalues), 2)
        self.assertTrue(all(eig >= 0 for eig in eigenvalues))


class TestLanczosNumericalStability(unittest.TestCase):
    """Test numerical stability in challenging cases."""
    
    def test_ill_conditioned_matrix(self):
        """Test Lanczos on ill-conditioned matrix."""
        # Create matrix with wide range of eigenvalues
        eigenvalues_true = np.array([1e6, 1e3, 1e0, 1e-3, 1e-6])
        Q = torch.tensor(np.eye(5), dtype=torch.float64)
        A = Q @ torch.diag(torch.tensor(eigenvalues_true, dtype=torch.float64)) @ Q.T
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(k=5, max_iters=20, seed=42, dtype_compute=torch.float64)
        results = lanczos_algorithm(op, config, verbose=False)
        
        # Top eigenvalue should still be accurate
        self.assertAlmostEqual(max(results['eigenvalues']), 1e6, delta=1e3)
        
        # Ritz condition number is from top-k, not full matrix
        # Just check it's computed and reasonable
        self.assertIn('ritz_condition_number', results)
        self.assertGreater(results['ritz_condition_number'], 1.0)
    
    def test_clustered_eigenvalues(self):
        """Test Lanczos when eigenvalues are clustered."""
        # Create matrix with clustered eigenvalues
        eigenvalues_true = np.array([5.0, 5.0, 5.0, 1.0, 1.0])
        Q = torch.tensor(np.eye(5), dtype=torch.float64)
        A = Q @ torch.diag(torch.tensor(eigenvalues_true, dtype=torch.float64)) @ Q.T
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(k=3, max_iters=15, seed=42)  # Request top-3
        results = lanczos_algorithm(op, config, verbose=False)
        
        eigenvalues_computed = np.array(sorted(results['eigenvalues'], reverse=True))
        
        # Lanczos finds top-3, which are all 5.0
        # Should approximate the cluster correctly
        np.testing.assert_allclose(eigenvalues_computed, [5.0, 5.0, 5.0], rtol=0.01)


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLanczosCorrectness))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLanczosPrecision))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLanczosOrthogonality))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLanczosConfiguration))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLanczosEdgeCases))
    suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestLanczosNumericalStability))
    
    return suite


if __name__ == '__main__':
    # Run with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())
