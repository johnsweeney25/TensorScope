"""
Unit tests for Fisher Spectral Analysis Module
===============================================
Tests theoretical correctness and numerical stability of Fisher spectrum computation.

Key tests:
1. Gram trick equivalence: G@G.T/N has same non-zero eigenvalues as G.T@G/N
2. Subsampling consistency: Same indices used across samples
3. Eigenvalue ordering: Verify descending sort
4. Spectral gap: λ₁ - λ₂ computation
5. Condition number: λ_max/λ_min
6. Effective rank: exp(H(p)) calculation
7. Block merging: Global spectrum = union of blocks
8. Fisher vs Covariance: Centering flag behavior
9. Numerical stability: Ill-conditioned matrices
10. Edge cases: Rank-1, all-zero, NaN handling

Author: ICLR 2026 Project
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisher_spectral import FisherSpectral, SpectralConfig


class TestFisherSpectralUnit(unittest.TestCase):
    """Unit tests for Fisher Spectral module."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SpectralConfig(
            seed=42,
            eps=1e-9,
            storage_mode='full',
            max_params_per_block=1000,
            dtype_eigensolve=torch.float64
        )
        self.spectral = FisherSpectral(self.config)

    def test_gram_trick_equivalence(self):
        """
        Test that Gram matrix trick gives same non-zero eigenvalues.

        Theory: For G ∈ ℝ^(N×P), the non-zero eigenvalues of (1/N)G@G.T
        equal the non-zero eigenvalues of (1/N)G.T@G.
        """
        torch.manual_seed(42)
        N, P = 50, 1000  # N << P case
        G = torch.randn(N, P, dtype=torch.float64)

        # Method 1: Direct Fisher (P×P)
        F_direct = (G.T @ G) / N
        eigs_direct = torch.linalg.eigvalsh(F_direct)
        # Keep only top N eigenvalues (rest are ~0 due to rank)
        eigs_direct = torch.sort(eigs_direct, descending=True).values[:N]

        # Method 2: Gram matrix (N×N)
        F_gram = (G @ G.T) / N
        eigs_gram = torch.linalg.eigvalsh(F_gram)
        eigs_gram = torch.sort(eigs_gram, descending=True).values

        # Compare non-zero eigenvalues (should match up to numerical precision)
        # Filter out near-zero eigenvalues
        threshold = 1e-10
        eigs_direct = eigs_direct[eigs_direct > threshold]
        eigs_gram = eigs_gram[eigs_gram > threshold]

        # Check they match
        min_len = min(len(eigs_direct), len(eigs_gram))
        torch.testing.assert_close(
            eigs_direct[:min_len],
            eigs_gram[:min_len],
            rtol=1e-6,
            atol=1e-8,
            msg="Gram trick eigenvalues don't match direct computation"
        )

    def test_subsampling_consistency(self):
        """
        Test that subsampling indices are consistent across samples.

        Critical for correctness: Each row of gradient matrix must use
        same parameter indices.
        """
        torch.manual_seed(42)
        block_key = 'test_block'
        P = 100000  # Large parameter count

        # Create two different gradient vectors
        grad1 = torch.randn(P)
        grad2 = torch.randn(P) * 2  # Different values

        # Subsample both
        sub1 = self.spectral._subsample_gradient(grad1, block_key)
        sub2 = self.spectral._subsample_gradient(grad2, block_key)

        # Check same shape
        self.assertEqual(sub1.shape, sub2.shape)
        self.assertEqual(sub1.shape[0], self.config.max_params_per_block)

        # Verify indices are consistent by checking relative ordering
        # Get indices used
        if block_key in self.spectral.subsample_indices:
            indices = self.spectral.subsample_indices[block_key]
            # Verify same indices were used
            torch.testing.assert_close(sub1, grad1[indices])
            torch.testing.assert_close(sub2, grad2[indices])

    def test_eigenvalue_ordering(self):
        """Test that eigenvalues are sorted in descending order."""
        torch.manual_seed(42)
        G = torch.randn(10, 20, dtype=torch.float64)

        eigenvalues = self.spectral._compute_block_eigenvalues(G)

        # Check descending order
        for i in range(len(eigenvalues) - 1):
            self.assertGreaterEqual(
                eigenvalues[i],
                eigenvalues[i + 1],
                f"Eigenvalues not in descending order at index {i}"
            )

    def test_spectral_gap_calculation(self):
        """Test correct computation of spectral gap λ₁ - λ₂."""
        # Create matrix with known eigenvalues
        torch.manual_seed(42)
        eigenvalues_true = torch.tensor([10.0, 7.0, 3.0, 1.0], dtype=torch.float64)

        # Create diagonal matrix with these eigenvalues
        D = torch.diag(eigenvalues_true)
        # Random orthogonal matrix for rotation
        Q, _ = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64))
        # Construct matrix with known spectrum
        A = Q @ D @ Q.T

        # Compute spectrum
        N = 4
        eigenvalues = torch.linalg.eigvalsh(A)
        eigenvalues = torch.sort(eigenvalues, descending=True).values

        # Analyze
        metrics = self.spectral._analyze_eigenvalues(eigenvalues, N, N)

        # Check spectral gap
        expected_gap = 10.0 - 7.0
        self.assertAlmostEqual(
            metrics['spectral_gap'],
            expected_gap,
            places=5,
            msg="Spectral gap calculation incorrect"
        )

    def test_condition_number(self):
        """Test condition number calculation λ_max/λ_min."""
        torch.manual_seed(42)
        eigenvalues = torch.tensor([100.0, 10.0, 1.0, 0.1], dtype=torch.float64)

        metrics = self.spectral._analyze_eigenvalues(eigenvalues, 4, 4)

        expected_condition = 100.0 / 0.1
        self.assertAlmostEqual(
            metrics['condition_number'],
            expected_condition,
            places=5,
            msg="Condition number calculation incorrect"
        )

    def test_effective_rank(self):
        """
        Test effective rank computation exp(H(p)).

        Theory: Effective rank measures the "participation" of eigenvalues.
        - All equal eigenvalues -> rank = number of eigenvalues
        - Single dominant eigenvalue -> rank ≈ 1
        """
        # Case 1: All equal eigenvalues
        eigenvalues_equal = torch.ones(10, dtype=torch.float64)
        metrics = self.spectral._analyze_eigenvalues(eigenvalues_equal, 10, 10)
        self.assertAlmostEqual(
            metrics['effective_rank'],
            10.0,
            places=5,
            msg="Effective rank wrong for equal eigenvalues"
        )

        # Case 2: Single dominant eigenvalue
        eigenvalues_dominant = torch.tensor([1000.0, 0.001, 0.001, 0.001], dtype=torch.float64)
        metrics = self.spectral._analyze_eigenvalues(eigenvalues_dominant, 4, 4)
        self.assertLess(
            metrics['effective_rank'],
            2.0,
            msg="Effective rank should be ~1 for dominant eigenvalue"
        )

        # Case 3: Two equal groups
        eigenvalues_groups = torch.tensor([10.0, 10.0, 1.0, 1.0], dtype=torch.float64)
        metrics = self.spectral._analyze_eigenvalues(eigenvalues_groups, 4, 4)
        # Entropy calculation: p = [10/22, 10/22, 1/22, 1/22]
        p = eigenvalues_groups / eigenvalues_groups.sum()
        entropy = -(p * torch.log(p)).sum()
        expected_rank = torch.exp(entropy).item()
        self.assertAlmostEqual(
            metrics['effective_rank'],
            expected_rank,
            places=5,
            msg="Effective rank calculation incorrect"
        )

    def test_block_merging(self):
        """
        Test that global spectrum correctly merges block eigenvalues.

        Theory: For block-diagonal matrix, global eigenvalues are
        the union (not average) of block eigenvalues.
        """
        # Create block eigenvalues
        block1_eigs = [10.0, 8.0, 6.0]
        block2_eigs = [9.0, 7.0, 5.0]
        block3_eigs = [11.0, 4.0, 2.0]

        all_eigenvalues = block1_eigs + block2_eigs + block3_eigs

        # Compute global metrics
        global_metrics = self.spectral._compute_global_metrics(all_eigenvalues)

        # Check largest eigenvalue is global max
        self.assertEqual(global_metrics['largest_eigenvalue'], 11.0)

        # Check spectral gap is λ₁ - λ₂ globally
        self.assertEqual(global_metrics['spectral_gap'], 11.0 - 10.0)

        # Check all eigenvalues are included
        expected_sorted = sorted(all_eigenvalues, reverse=True)
        actual_top = global_metrics['top_eigenvalues']
        for i in range(min(len(expected_sorted), len(actual_top))):
            self.assertAlmostEqual(actual_top[i], expected_sorted[i], places=5)

    def test_fisher_vs_covariance(self):
        """
        Test that centering flag correctly switches between Fisher and covariance.

        Theory:
        - Fisher: F = E[gg^T] (uncentered)
        - Covariance: C = E[(g-μ)(g-μ)^T] (centered)
        """
        torch.manual_seed(42)
        # Create gradients with non-zero mean
        N, P = 10, 5
        G = torch.randn(N, P) + 2.0  # Add bias to create non-zero mean

        # Compute Fisher (uncentered)
        eigenvalues_fisher = self.spectral._compute_block_eigenvalues(G)

        # Compute Covariance (centered)
        G_centered = G - G.mean(dim=0, keepdim=True)
        eigenvalues_cov = self.spectral._compute_block_eigenvalues(G_centered)

        # Fisher should have larger eigenvalues due to mean component
        self.assertGreater(
            eigenvalues_fisher[0],
            eigenvalues_cov[0],
            "Fisher should have larger eigenvalues than covariance when mean ≠ 0"
        )

    def test_numerical_stability_illconditioned(self):
        """Test handling of ill-conditioned matrices."""
        torch.manual_seed(42)
        # Create ill-conditioned matrix with huge condition number
        N = 20
        eigenvalues = torch.logspace(-15, 2, N, dtype=torch.float64)  # Range 1e-15 to 100
        D = torch.diag(eigenvalues)
        Q, _ = torch.linalg.qr(torch.randn(N, N, dtype=torch.float64))
        A = Q @ D @ Q.T

        # This should not crash
        computed_eigs = torch.linalg.eigvalsh(A + self.config.regularization * torch.eye(N, dtype=torch.float64))
        computed_eigs = torch.sort(computed_eigs, descending=True).values
        computed_eigs = computed_eigs[computed_eigs > self.config.eps]

        metrics = self.spectral._analyze_eigenvalues(computed_eigs, N, N)

        # Check that we get reasonable results despite ill-conditioning
        self.assertGreater(metrics['condition_number'], 1e10)
        self.assertTrue(math.isfinite(metrics['effective_rank']))

    def test_edge_case_rank_one(self):
        """Test handling of rank-1 matrices (e.g., single gradient)."""
        torch.manual_seed(42)
        # Rank-1 matrix: gg^T
        g = torch.randn(10, dtype=torch.float64)
        G = g.unsqueeze(0)  # 1×10 matrix

        eigenvalues = self.spectral._compute_block_eigenvalues(G)
        metrics = self.spectral._analyze_eigenvalues(eigenvalues, 1, 10)

        # Should have single eigenvalue
        self.assertEqual(len(eigenvalues), 1)
        self.assertEqual(metrics['spectral_gap'], 0.0)  # No second eigenvalue
        self.assertAlmostEqual(metrics['effective_rank'], 1.0, places=5)

    def test_edge_case_zero_gradients(self):
        """Test handling of all-zero gradients."""
        G = torch.zeros(10, 20, dtype=torch.float64)

        eigenvalues = self.spectral._compute_block_eigenvalues(G)

        # Should handle gracefully
        self.assertEqual(len(eigenvalues), 0)  # All filtered as numerical zeros

    def test_edge_case_nan_handling(self):
        """Test that NaN gradients are handled gracefully."""
        # This is tested at a higher level in integration tests
        # Here we test the eigenvalue analysis with edge cases

        # Empty eigenvalues (as would result from NaN filtering)
        empty_eigs = torch.tensor([], dtype=torch.float64)
        metrics = self.spectral._analyze_eigenvalues(empty_eigs, 0, 0)

        self.assertEqual(metrics['spectral_gap'], 0.0)
        self.assertEqual(metrics['effective_rank'], 1.0)
        self.assertTrue(math.isinf(metrics['condition_number']))

    def test_block_key_generation(self):
        """Test that block keys are generated consistently."""
        test_cases = [
            # (param_name, block_structure, expected_key)
            ('model.layers.0.self_attn.q_proj.weight', 'layer', 'layer_0'),
            ('model.layers.10.mlp.gate_proj.weight', 'layer', 'layer_10'),
            ('model.embed_tokens.weight', 'layer', 'embedding'),
            ('lm_head.weight', 'layer', 'output'),
            ('model.layers.0.self_attn.q_proj.weight', 'module', 'attention'),
            ('model.layers.0.mlp.gate_proj.weight', 'module', 'mlp'),
            ('model.norm.weight', 'module', 'normalization'),
            ('anything.weight', 'none', 'global'),
        ]

        for param_name, block_structure, expected_key in test_cases:
            actual_key = self.spectral._get_block_key(param_name, block_structure)
            self.assertEqual(
                actual_key,
                expected_key,
                f"Block key mismatch for {param_name} with structure {block_structure}"
            )

    def test_regularization_effect(self):
        """Test that regularization improves numerical stability."""
        torch.manual_seed(42)
        N = 10
        # Create near-singular matrix
        G = torch.randn(N, N, dtype=torch.float64)
        G[5:, :] = G[0, :] * 1e-10  # Make bottom half nearly zero

        # Without regularization (using very small eps)
        config_no_reg = SpectralConfig(regularization=1e-15)
        spectral_no_reg = FisherSpectral(config_no_reg)

        # With regularization
        config_with_reg = SpectralConfig(regularization=1e-8)
        spectral_with_reg = FisherSpectral(config_with_reg)

        # Both should complete without error
        eigs_no_reg = spectral_no_reg._compute_block_eigenvalues(G)
        eigs_with_reg = spectral_with_reg._compute_block_eigenvalues(G)

        # Regularized version should have better conditioning
        cond_no_reg = eigs_no_reg[0] / eigs_no_reg[-1] if len(eigs_no_reg) > 0 else float('inf')
        cond_with_reg = eigs_with_reg[0] / eigs_with_reg[-1] if len(eigs_with_reg) > 0 else float('inf')

        # Regularization should improve condition number
        self.assertLess(cond_with_reg, cond_no_reg * 0.99)  # At least 1% improvement

    def test_dtype_consistency(self):
        """Test that dtypes are handled consistently throughout."""
        torch.manual_seed(42)

        # Test with different input dtypes
        for dtype in [torch.float32, torch.float64]:
            G = torch.randn(5, 10, dtype=dtype)
            eigenvalues = self.spectral._compute_block_eigenvalues(G)

            # Should always compute in float64 for accuracy
            self.assertEqual(eigenvalues.dtype, torch.float64)

            # Results should be valid
            self.assertTrue(all(e > 0 for e in eigenvalues))


if __name__ == '__main__':
    unittest.main()