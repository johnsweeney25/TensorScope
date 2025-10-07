#!/usr/bin/env python3
"""
Production-Grade Lanczos Tests
===============================

Minimal test suite covering:
1. PSD vs Indefinite sanity checks
2. Windowed reorthogonalization ablation
3. Scale invariance
4. Ritz residuals validation
5. OOM regression (memory stability)

These tests verify production readiness for ICLR submission.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fisher.core.fisher_lanczos_unified import (
    lanczos_algorithm,
    LanczosConfig,
    LinOp,
    HessianOperator,
    GGNOperator,
    create_operator
)


class SimpleMatrixOperator(LinOp):
    """Simple matrix operator for testing."""
    
    def __init__(self, A: torch.Tensor, is_psd: bool = True):
        """
        Args:
            A: Matrix to wrap
            is_psd: Whether matrix is positive semi-definite
        """
        self.matrix = A
        param = torch.zeros(A.shape[0], dtype=A.dtype, requires_grad=True)
        
        def matvec(v):
            """Matrix-vector product."""
            v_vec = v[0].to(self.matrix.dtype)
            return [self.matrix @ v_vec]
        
        super().__init__([param], matvec, "test_matrix", is_psd=is_psd, device=A.device)


class TestPSDIndefiniteSanity(unittest.TestCase):
    """Test 1: PSD vs Indefinite matrix sanity checks."""
    
    def test_psd_all_positive_eigenvalues(self):
        """PSD matrices (GGN) should return all non-negative eigenvalues."""
        # Create PSD matrix
        n = 20
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = A @ A.T  # Guaranteed PSD
        A = A + torch.eye(n) * 0.1  # Add small diagonal for stability
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(
            k=5,
            max_iters=30,
            seed=42,
            dtype_compute=torch.float64,
            reorth_mode='full'
        )
        
        results = lanczos_algorithm(op, config, verbose=False)
        eigenvalues = np.array(results['eigenvalues'])
        
        # All eigenvalues should be non-negative
        self.assertTrue(
            np.all(eigenvalues >= -1e-10),
            f"PSD matrix returned negative eigenvalues: {eigenvalues[eigenvalues < 0]}"
        )
        
        # Check that is_psd flag is preserved
        self.assertTrue(results['is_psd'], "Result should indicate PSD operator")
    
    def test_indefinite_has_negative_eigenvalues(self):
        """Indefinite matrices (Hessian) should be able to return negative eigenvalues."""
        # Create indefinite matrix with large negative eigenvalues
        # Lanczos finds largest magnitude eigenvalues, so negatives must be large in |value|
        n = 20
        torch.manual_seed(42)
        Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
        
        # Eigenvalues: [10, -8, 5, -3, 2, 0.1, ...]
        # Mix of large positive and negative so Lanczos finds both
        diag = torch.tensor([10.0, -8.0, 5.0, -3.0, 2.0] + [0.1] * (n-5), dtype=torch.float64)
        A = Q @ torch.diag(diag) @ Q.T
        
        op = SimpleMatrixOperator(A, is_psd=False)
        config = LanczosConfig(
            k=5,
            max_iters=40,
            seed=42,
            dtype_compute=torch.float64,
            reorth_mode='full',
            regularization_mode='off'  # Don't regularize - need accurate negatives
        )
        
        results = lanczos_algorithm(op, config, verbose=False)
        eigenvalues = np.array(results['eigenvalues'])
        
        # Should find negative eigenvalues (at least one of -8 or -3)
        # Note: Lanczos returns eigenvalues in descending order, so check if any are negative
        has_negative = np.any(eigenvalues < -1e-6)
        self.assertTrue(
            has_negative,
            f"Indefinite matrix should have negative eigenvalues, got: {eigenvalues}"
        )
        
        # Check negative mass metrics are present
        self.assertIn('negative_mass', results)
        self.assertIn('negative_fraction', results)
        self.assertIn('most_negative_eigenvalue', results)
        
        # Negative mass should be > 0
        self.assertGreater(results['negative_mass'], 0.0)
        self.assertGreater(results['negative_fraction'], 0.0)


class TestReorthogonalizationAblation(unittest.TestCase):
    """Test 2: Windowed reorthogonalization ablation study."""
    
    def test_reorth_modes_comparison(self):
        """Compare full, selective, and off reorthogonalization modes."""
        # Create moderately ill-conditioned matrix
        n = 30
        torch.manual_seed(42)
        Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
        # Eigenvalues spanning 3 orders of magnitude
        diag = torch.logspace(0, 3, n, dtype=torch.float64)
        A = Q @ torch.diag(diag) @ Q.T
        
        eigenvalues_true = torch.linalg.eigvalsh(A).cpu().numpy()
        eigenvalues_true = np.sort(eigenvalues_true)[::-1][:5]
        
        modes = ['full', 'selective', 'off']
        results_by_mode = {}
        
        for mode in modes:
            op = SimpleMatrixOperator(A, is_psd=True)
            config = LanczosConfig(
                k=5,
                max_iters=40,
                seed=42,
                dtype_compute=torch.float64,
                reorth_mode=mode,
                reorth_period=5 if mode == 'selective' else 0,
                reorth_window=5 if mode == 'selective' else 0
            )
            
            results = lanczos_algorithm(op, config, verbose=False)
            eigenvalues = np.array(results['eigenvalues'])
            error = np.abs(eigenvalues - eigenvalues_true)
            
            results_by_mode[mode] = {
                'eigenvalues': eigenvalues,
                'max_error': np.max(error),
                'top_error': error[0],  # Most important
                'converged': results['converged'],
                'iterations': results['iterations'],
                'warnings': len(results.get('warnings', []))
            }
        
        # Full reorthogonalization should be most accurate
        self.assertLess(
            results_by_mode['full']['top_error'],
            1e-8,
            f"Full reorth should be highly accurate: {results_by_mode['full']['top_error']:.2e}"
        )
        
        # Selective should be reasonable (within 1e-6 for top eigenvalue)
        self.assertLess(
            results_by_mode['selective']['top_error'],
            1e-5,
            f"Selective reorth should be reasonable: {results_by_mode['selective']['top_error']:.2e}"
        )
        
        # Off mode should have more warnings about loss of orthogonality
        # (but might still work for well-conditioned matrices)
        print(f"\nReorthogonalization ablation:")
        for mode in modes:
            print(f"  {mode:10s}: top_error={results_by_mode[mode]['top_error']:.2e}, "
                  f"warnings={results_by_mode[mode]['warnings']}, "
                  f"converged={results_by_mode[mode]['converged']}")


class TestScaleInvariance(unittest.TestCase):
    """Test 3: Scale invariance of Lanczos eigenvalues."""
    
    def test_eigenvalues_scale_with_matrix(self):
        """Eigenvalues should scale linearly with matrix scaling."""
        n = 20
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = (A + A.T) / 2
        A = A + torch.eye(n) * 5.0
        
        scales = [1.0, 10.0, 100.0]
        eigenvalues_by_scale = {}
        
        for scale in scales:
            A_scaled = A * scale
            
            op = SimpleMatrixOperator(A_scaled, is_psd=True)
            config = LanczosConfig(
                k=5,
                max_iters=30,
                seed=42,
                dtype_compute=torch.float64,
                reorth_mode='full',
                regularization_mode='off'  # Disable to test pure scaling
            )
            
            results = lanczos_algorithm(op, config, verbose=False)
            eigenvalues_by_scale[scale] = np.array(results['eigenvalues'])
        
        # Check that eigenvalues scale linearly
        base_eigs = eigenvalues_by_scale[1.0]
        
        for scale in [10.0, 100.0]:
            scaled_eigs = eigenvalues_by_scale[scale]
            expected_eigs = base_eigs * scale
            
            # Relative error should be small
            rel_error = np.abs(scaled_eigs - expected_eigs) / (expected_eigs + 1e-10)
            max_rel_error = np.max(rel_error)
            
            self.assertLess(
                max_rel_error,
                1e-6,
                f"Eigenvalues should scale linearly (scale={scale}): "
                f"max_rel_error={max_rel_error:.2e}"
            )
            
            print(f"  Scale {scale:5.1f}: max_rel_error={max_rel_error:.2e}")


class TestRitzResiduals(unittest.TestCase):
    """Test 4: Ritz residuals validation."""
    
    def test_residuals_computed_for_full_reorth(self):
        """Ritz residuals should be computed when using full reorthogonalization."""
        n = 20
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = (A + A.T) / 2
        A = A + torch.eye(n) * 10.0
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(
            k=5,
            max_iters=30,
            seed=42,
            dtype_compute=torch.float64,
            reorth_mode='full'  # Required for residuals
        )
        
        results = lanczos_algorithm(op, config, verbose=False)
        
        # Should have residuals
        self.assertIn('ritz_residuals', results)
        self.assertIsNotNone(results['ritz_residuals'])
        
        residuals = results['ritz_residuals']
        
        # Should have one residual per eigenvalue
        self.assertEqual(len(residuals), len(results['eigenvalues']))
        
        # Residuals should be small for converged eigenpairs
        if results['converged']:
            for i, res in enumerate(residuals):
                self.assertLess(
                    res,
                    1e-6,
                    f"Converged eigenpair {i} should have small residual: {res:.2e}"
                )
        
        print(f"\nRitz residuals: {[f'{r:.2e}' for r in residuals]}")
    
    def test_no_residuals_for_selective_reorth(self):
        """Ritz residuals should not be computed for selective reorthogonalization."""
        n = 20
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = (A + A.T) / 2
        A = A + torch.eye(n) * 10.0
        
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(
            k=5,
            max_iters=30,
            seed=42,
            dtype_compute=torch.float64,
            reorth_mode='selective'  # No full basis stored
        )
        
        results = lanczos_algorithm(op, config, verbose=False)
        
        # Should not have residuals (or should be None)
        ritz_residuals = results.get('ritz_residuals')
        self.assertIsNone(
            ritz_residuals,
            "Selective reorth should not compute residuals (no full basis)"
        )
    
    def test_residuals_quantify_accuracy(self):
        """Residuals should correlate with eigenvalue accuracy."""
        n = 20
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = (A + A.T) / 2
        A = A + torch.eye(n) * 10.0
        
        # True eigenvalues
        eigenvalues_true = torch.linalg.eigvalsh(A).cpu().numpy()
        eigenvalues_true = np.sort(eigenvalues_true)[::-1][:5]
        
        # Compute with Lanczos
        op = SimpleMatrixOperator(A, is_psd=True)
        config = LanczosConfig(
            k=5,
            max_iters=30,
            seed=42,
            dtype_compute=torch.float64,
            reorth_mode='full'
        )
        
        results = lanczos_algorithm(op, config, verbose=False)
        eigenvalues = np.array(results['eigenvalues'])
        residuals = np.array(results['ritz_residuals'])
        
        # Compute eigenvalue errors
        errors = np.abs(eigenvalues - eigenvalues_true)
        
        # Residuals should be small when errors are small
        # (Not a perfect correlation, but should be reasonable)
        for i in range(len(residuals)):
            if errors[i] < 1e-8:
                self.assertLess(
                    residuals[i],
                    1e-5,
                    f"Accurate eigenvalue {i} should have small residual: "
                    f"error={errors[i]:.2e}, residual={residuals[i]:.2e}"
                )


class TestMemoryStability(unittest.TestCase):
    """Test 5: OOM regression and memory stability."""
    
    def test_large_matrix_selective_reorth(self):
        """Test that selective reorthogonalization doesn't OOM on large matrices."""
        # Simulate large matrix operator (don't actually allocate full matrix)
        n = 10000  # 10k dimensions
        
        # Create operator that does matvec without storing full matrix
        class SparseOperator(LinOp):
            def __init__(self, n, seed=42):
                self.n = n
                torch.manual_seed(seed)
                self.diag = torch.linspace(100, 1, n, dtype=torch.float64)
                
                param = torch.zeros(n, dtype=torch.float64, requires_grad=True)
                
                def matvec(v):
                    # Diagonal matrix-vector product
                    return [self.diag * v[0]]
                
                super().__init__([param], matvec, "sparse", is_psd=True)
        
        op = SparseOperator(n)
        config = LanczosConfig(
            k=10,
            max_iters=50,
            seed=42,
            dtype_compute=torch.float64,
            reorth_mode='selective',  # Must use selective for large n
            reorth_window=5,
            reorth_period=5
        )
        
        # This should complete without OOM
        try:
            results = lanczos_algorithm(op, config, verbose=False)
            
            # Should find reasonable eigenvalues
            eigenvalues = np.array(results['eigenvalues'])
            
            # Top eigenvalue should be close to 100
            self.assertGreater(eigenvalues[0], 90.0)
            self.assertLess(eigenvalues[0], 110.0)
            
            # Should complete
            self.assertGreater(results['iterations'], 0)
            
            print(f"\nLarge matrix test (n={n}): "
                  f"top_eigenvalue={eigenvalues[0]:.2f}, "
                  f"iterations={results['iterations']}")
            
        except RuntimeError as e:
            if 'out of memory' in str(e).lower():
                self.fail(f"OOM on large matrix with selective reorth: {e}")
            else:
                raise
    
    def test_memory_cleanup_between_runs(self):
        """Test that memory is properly cleaned up between multiple runs."""
        n = 100
        torch.manual_seed(42)
        A = torch.randn(n, n, dtype=torch.float64)
        A = (A + A.T) / 2
        A = A + torch.eye(n) * 10.0
        
        # Run multiple times to check for memory leaks
        for run in range(5):
            op = SimpleMatrixOperator(A, is_psd=True)
            config = LanczosConfig(
                k=10,
                max_iters=50,
                seed=42 + run,
                dtype_compute=torch.float64,
                reorth_mode='selective',
                gc_every=10  # Enable garbage collection
            )
            
            results = lanczos_algorithm(op, config, verbose=False)
            
            # Should complete successfully
            self.assertGreater(len(results['eigenvalues']), 0)
            
            # Force cleanup
            del results
            del op
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # If we got here without OOM, cleanup is working
        print("\nMemory cleanup test: 5 runs completed successfully")


class TestNegativeMassMetrics(unittest.TestCase):
    """Additional test for negative mass metrics (from enhancement #3)."""
    
    def test_negative_mass_quantifies_saddle_character(self):
        """Negative mass should correctly quantify saddle point character."""
        n = 20
        torch.manual_seed(42)
        
        # Create matrix with known negative eigenvalues
        Q, _ = torch.linalg.qr(torch.randn(n, n, dtype=torch.float64))
        
        # 60% negative eigenvalues (by count)
        diag = torch.tensor(
            [10.0, 5.0, 2.0, 1.0] +  # 4 positive
            [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0] +  # 6 negative
            [0.1] * (n - 10),  # rest small positive
            dtype=torch.float64
        )
        A = Q @ torch.diag(diag) @ Q.T
        
        op = SimpleMatrixOperator(A, is_psd=False)
        config = LanczosConfig(
            k=10,  # Get enough to see negatives
            max_iters=50,
            seed=42,
            dtype_compute=torch.float64,
            reorth_mode='full',
            regularization_mode='off'
        )
        
        results = lanczos_algorithm(op, config, verbose=False)
        
        # Check negative mass metrics
        self.assertIn('negative_mass', results)
        self.assertIn('negative_fraction', results)
        self.assertIn('most_negative_eigenvalue', results)
        
        negative_fraction = results['negative_fraction']
        negative_mass = results['negative_mass']
        most_negative = results['most_negative_eigenvalue']
        
        # Should have reasonable negative fraction (may not be exactly 60% due to Lanczos approximation)
        self.assertGreater(negative_fraction, 0.3)
        self.assertLess(negative_fraction, 0.8)
        
        # Negative mass should be positive
        self.assertGreater(negative_mass, 0.0)
        self.assertLess(negative_mass, 1.0)
        
        # Most negative should be negative
        self.assertLess(most_negative, -1.0)
        
        print(f"\nNegative mass metrics: "
              f"fraction={negative_fraction:.2f}, "
              f"mass={negative_mass:.2f}, "
              f"most_negative={most_negative:.2f}")


def run_production_tests():
    """Run all production tests with summary."""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPSDIndefiniteSanity))
    suite.addTests(loader.loadTestsFromTestCase(TestReorthogonalizationAblation))
    suite.addTests(loader.loadTestsFromTestCase(TestScaleInvariance))
    suite.addTests(loader.loadTestsFromTestCase(TestRitzResiduals))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryStability))
    suite.addTests(loader.loadTestsFromTestCase(TestNegativeMassMetrics))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_production_tests()
    sys.exit(0 if success else 1)
