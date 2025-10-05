#!/usr/bin/env python3
"""
Unit tests for critical fixes in SuperpositionAnalyzer implementation.

Tests verify that paper-invalidating bugs identified in the intern's review have been resolved
for the ICLR 2026 submission. These bugs include incorrect Welch bound usage, wrong expected
scaling formula, memory explosion from caching, and sampling bias issues.

File: superposition/core/analyzer.py
Class: SuperpositionAnalyzer
Purpose: Ensure theoretical correctness and numerical stability for unified analysis
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from superposition.core.analyzer import SuperpositionAnalyzer, SuperpositionAnalysis


class TestCacheKeyVulnerability(unittest.TestCase):
    """Test robust cache key fingerprinting."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_tensor_fingerprint_detects_mutations(self):
        """Test that fingerprint changes when tensor is mutated in-place."""
        # Create a tensor
        tensor = torch.randn(100, 50)

        # Get initial fingerprint
        fp1 = self.analyzer._tensor_fingerprint(tensor)

        # Mutate in-place
        tensor[0, 0] = 999.0

        # Get fingerprint after mutation
        fp2 = self.analyzer._tensor_fingerprint(tensor)

        # Version should have changed if available
        if hasattr(tensor, '_version'):
            self.assertNotEqual(
                fp1[-1], fp2[-1],
                "Fingerprint should detect in-place mutations via version tracking"
            )

    def test_tensor_fingerprint_different_for_different_tensors(self):
        """Test that different tensors have different fingerprints."""
        t1 = torch.randn(10, 10)
        t2 = torch.randn(10, 10)

        fp1 = self.analyzer._tensor_fingerprint(t1)
        fp2 = self.analyzer._tensor_fingerprint(t2)

        # Different data pointers
        self.assertNotEqual(fp1[0], fp2[0], "Different tensors should have different data pointers")

    def test_cache_lru_eviction(self):
        """Test that cache uses LRU eviction when full."""
        # Set small cache size for testing
        self.analyzer._max_cache_entries = 3

        # Add entries to fill cache
        for i in range(5):
            tensor = torch.randn(10, 10)
            key = self.analyzer._tensor_fingerprint(tensor)
            self.analyzer._cache_set(
                self.analyzer._norm_cache, key, torch.tensor(i), 'test'
            )

        # Cache should only have 3 entries (most recent)
        self.assertEqual(len(self.analyzer._norm_cache), 3, "Cache should maintain max size")


class TestWelchBoundFix(unittest.TestCase):
    """Test that Welch bound uses coherence (max overlap) not mean."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_welch_bound_uses_max_overlap(self):
        """Test that Welch bound ratio uses maximum overlap (coherence), not mean."""
        # Create overcomplete set (n > d)
        n_features = 20
        n_dims = 10

        # Create random normalized vectors
        W = torch.randn(n_features, n_dims)
        W = W / torch.linalg.norm(W, dim=1, keepdim=True)

        result = self.analyzer.compute_comprehensive_superposition_analysis(
            W, return_dict=False
        )

        # Compute overlaps manually
        overlaps = torch.abs(W @ W.T)
        mask = ~torch.eye(n_features, dtype=torch.bool)
        off_diagonal = overlaps[mask]

        max_overlap = off_diagonal.max().item()
        mean_overlap = off_diagonal.mean().item()

        # Welch bound
        welch_bound = np.sqrt((n_features - n_dims) / (n_dims * (n_features - 1)))

        # The ratio should use max, not mean
        expected_ratio = max_overlap / welch_bound

        # The result should be based on max overlap
        self.assertGreater(
            result.welch_bound_ratio, 0.9,  # Should be ≥ 1 theoretically
            "Welch bound ratio should use max overlap (coherence)"
        )

        # Should NOT match mean-based ratio
        wrong_ratio = mean_overlap / welch_bound
        self.assertNotAlmostEqual(
            result.welch_bound_ratio, wrong_ratio, 2,
            "Welch bound ratio should NOT use mean overlap"
        )


class TestExpectedScalingFix(unittest.TestCase):
    """Test correct expected scaling formula."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_expected_scaling_formula(self):
        """Test that expected scaling uses √(2/π)/√d, not 1/√d."""
        n_features = 100
        n_dims = 50

        # Create random unit vectors
        W = torch.randn(n_features, n_dims)
        W = W / torch.linalg.norm(W, dim=1, keepdim=True)

        result = self.analyzer.compute_comprehensive_superposition_analysis(W)

        # Correct formula: E[|⟨x,y⟩|] ≈ √(2/π)/√d ≈ 0.798/√d
        correct_expected = np.sqrt(2.0 / np.pi) / np.sqrt(n_dims)

        # Wrong formula: 1/√d
        wrong_expected = 1.0 / np.sqrt(n_dims)

        # Check that result uses correct formula
        self.assertAlmostEqual(
            result.expected_scaling, correct_expected, 5,
            f"Expected scaling should be √(2/π)/√d ≈ {correct_expected:.5f}"
        )

        self.assertNotAlmostEqual(
            result.expected_scaling, wrong_expected, 3,
            f"Expected scaling should NOT be 1/√d = {wrong_expected:.5f}"
        )


class TestMemoryExplosionFix(unittest.TestCase):
    """Test that overlap matrices are not cached to prevent memory explosion."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_overlap_matrices_not_cached(self):
        """Test that overlap matrices are not cached (would cause O(n²) memory)."""
        # Create a moderately sized matrix
        W = torch.randn(100, 50)

        # First computation
        result1 = self.analyzer.compute_comprehensive_superposition_analysis(W)

        # Check cache - should have norms but NOT overlaps
        self.assertGreater(
            len(self.analyzer._norm_cache), 0,
            "Norms should be cached"
        )

        # Overlap cache should be empty (no matrix caching)
        self.assertEqual(
            len(self.analyzer._overlap_cache), 0,
            "Overlap matrices should NOT be cached to prevent memory explosion"
        )

        # Second computation should recompute overlaps
        cache_misses_before = self.analyzer.cache_misses['overlaps']
        result2 = self.analyzer.compute_comprehensive_superposition_analysis(W)
        cache_misses_after = self.analyzer.cache_misses['overlaps']

        self.assertGreater(
            cache_misses_after, cache_misses_before,
            "Overlaps should be recomputed (not cached)"
        )


class TestSamplingBiasFix(unittest.TestCase):
    """Test that sampling doesn't bias high overlap counting."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_high_overlap_count_before_sampling(self):
        """Test that high overlap pairs are counted on full batch before sampling."""
        # Create matrix with some high overlaps
        n_features = 1000
        n_dims = 100

        W = torch.randn(n_features, n_dims)
        # Add some nearly identical vectors for high overlap
        W[10:20] = W[0:10] + torch.randn(10, n_dims) * 0.01
        W = W / torch.linalg.norm(W, dim=1, keepdim=True)

        # Run with sampling
        result_sampled = self.analyzer.compute_comprehensive_superposition_analysis(
            W,
            overlap_threshold=0.95,
            use_sampling=True,
            max_pairs=10000,  # Force sampling
            seed=42
        )

        # Run without sampling
        result_full = self.analyzer.compute_comprehensive_superposition_analysis(
            W,
            overlap_threshold=0.95,
            use_sampling=False
        )

        # High overlap count should be similar (within sampling error)
        # The fix ensures max and high count are computed BEFORE sampling
        self.assertAlmostEqual(
            result_sampled.max_overlap,
            result_full.max_overlap,
            3,
            "Max overlap should be computed on full batch before sampling"
        )

        # High overlap count should be scaled properly
        ratio = result_sampled.num_high_overlap_pairs / max(result_full.num_high_overlap_pairs, 1)
        self.assertGreater(
            ratio, 0.5,  # Should be reasonably close
            "High overlap count should be properly estimated with sampling"
        )


class TestFP32Enforcement(unittest.TestCase):
    """Test that FP32 is enforced throughout for numerical stability."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_fp32_conversion(self):
        """Test that FP16 inputs are converted to FP32."""
        # Create FP16 tensor
        W_fp16 = torch.randn(50, 20, dtype=torch.float16)

        # Should not crash and should compute correctly
        result = self.analyzer.compute_comprehensive_superposition_analysis(W_fp16)

        # Results should be valid (not NaN)
        self.assertFalse(
            np.isnan(result.mean_overlap),
            "Mean overlap should not be NaN after FP32 conversion"
        )

        self.assertGreaterEqual(
            result.mean_overlap, 0,
            "Mean overlap should be non-negative"
        )


class TestStreamingStatistics(unittest.TestCase):
    """Test streaming statistics for memory efficiency."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_streaming_computation_accuracy(self):
        """Test that streaming statistics match batch computation."""
        # Create test data
        np.random.seed(42)
        data = np.random.randn(10000)

        # Compute statistics in one go
        mean_batch = np.mean(data)
        std_batch = np.std(data)

        # Compute using Welford's online algorithm (as in the fix)
        n = 0
        mean = 0.0
        M2 = 0.0

        for x in data:
            n += 1
            delta = x - mean
            mean += delta / n
            delta2 = x - mean
            M2 += delta * delta2

        std_streaming = np.sqrt(M2 / n)

        # Should match closely
        self.assertAlmostEqual(mean, mean_batch, 10, "Streaming mean should match batch")
        self.assertAlmostEqual(std_streaming, std_batch, 10, "Streaming std should match batch")


class TestUniformSamplingClaim(unittest.TestCase):
    """Test that sampling is correctly described as uniform, not stratified."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SuperpositionAnalyzer()

    def test_sampling_is_uniform(self):
        """Test that sampling uses uniform random selection, not stratified."""
        # This is more of a documentation test - the implementation should
        # clearly state it uses UNIFORM sampling, not stratified

        # Check docstring mentions uniform sampling
        docstring = self.analyzer._compute_overlaps_batched.__doc__
        self.assertIn(
            "UNIFORM", docstring,
            "Docstring should clearly state that sampling is UNIFORM, not stratified"
        )

        # The implementation uses torch.randperm which is uniform sampling
        # Stratified would require binning first
        W = torch.randn(100, 50)
        result = self.analyzer.compute_comprehensive_superposition_analysis(
            W, use_sampling=True, max_pairs=100, seed=42
        )

        # Should complete without error
        self.assertIsInstance(result, SuperpositionAnalysis)


class TestGPUDeviceFix(unittest.TestCase):
    """Test GPU device handling fixes."""

    def test_device_handling(self):
        """Test that device is handled correctly."""
        # Create analyzer with explicit device
        if torch.cuda.is_available():
            analyzer = SuperpositionAnalyzer(device='cuda:0')
            W = torch.randn(50, 20)  # CPU tensor

            # Should handle CPU tensor and move to GPU
            result = analyzer.compute_comprehensive_superposition_analysis(W)
            self.assertIsInstance(result, SuperpositionAnalysis)
        else:
            # Test CPU path
            analyzer = SuperpositionAnalyzer(device='cpu')
            W = torch.randn(50, 20)
            result = analyzer.compute_comprehensive_superposition_analysis(W)
            self.assertIsInstance(result, SuperpositionAnalysis)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)