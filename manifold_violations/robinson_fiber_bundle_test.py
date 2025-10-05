#!/usr/bin/env python3
"""
Fiber Bundle Hypothesis Test for Embedding Spaces (AUDITED)

NOTE: This implementation is inspired by research on manifold violations in embeddings.
The original paper reference (arxiv:2504.01002)
This module implements geometric tests for manifold structure violations.

Key Innovation:
- Tests volume growth patterns in log-log space
- Detects slope increases that violate fiber bundle structure
- Uses CFAR detector for statistically rigorous slope change detection

The test examines how the number of tokens within radius r scales with r.
For a proper fiber bundle, this should follow predictable patterns:
- Small radius: O(r^d) where d = local manifold dimension
- Large radius: O((r-ρ)^d_base) where d_base = base manifold dimension
- Slopes should be piecewise linear and NON-INCREASING

Violations (increasing slopes) indicate semantic instability.
"""

import numpy as np
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass
from scipy import stats
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
import warnings
import matplotlib.pyplot as plt

# Suppress expected scipy warnings for degenerate geometric cases
#
# MATHEMATICAL CONTEXT:
# The Robinson et al. fiber bundle test analyzes log-log volume growth patterns
# to detect manifold structure violations. The test assumes embeddings form a
# well-behaved geometric structure, but real token embeddings can violate this.
#
# These warnings occur in edge cases that are mathematically expected:
#
# 1. Division by zero in log-space slope computation:
#    - When log(radii[i+1]) ≈ log(radii[i]), denominator → 0
#    - Occurs when embeddings are highly discrete/quantized
#    - Example: Binary embeddings with only a few distinct distances
#
# 2. Constant values in statistical tests:
#    - Mann-Kendall test for monotonic trend requires variance
#    - When all slopes are identical, rank correlation is undefined
#    - Mathematically: correlation = cov(X,Y)/(σ_X * σ_Y), where σ = 0
#
# 3. Log of zero volumes:
#    - While we add 1e-10 to avoid log(0), numerical precision can still cause issues
#    - Occurs in extremely sparse regions where no neighbors exist at certain radii
#
# The Robinson paper assumes sufficient data density for meaningful volume measurements.
# These warnings indicate edge cases where that assumption breaks down, which may
# itself be informative about the embedding structure (e.g., extreme sparsity or
# discretization). The test continues despite warnings, using the resulting NaN/Inf
# values naturally in subsequent computations (NaN propagates, affecting p-values).
warnings.filterwarnings('ignore', message='invalid value encountered in divide', module='scipy.stats')
warnings.filterwarnings('ignore', message='divide by zero encountered', module='scipy.stats')


@dataclass
class RobinsonTestResult:
    """Results from fiber bundle hypothesis test.

    Implements dual rejection criteria from the paper:
    - Rejects manifold if slopes don't change between regimes
    - Rejects fiber bundle if slopes increase
    """
    # Dual violation flags (matching paper's flowchart)
    violates_manifold: bool  # No slope change detected
    violates_fiber_bundle: bool  # Increasing slopes detected
    violates_hypothesis: bool  # Either violation (legacy compatibility)
    p_value: float

    # Volume growth analysis
    radii: np.ndarray
    volumes: np.ndarray  # Number of points within each radius
    log_radii: np.ndarray
    log_volumes: np.ndarray

    # Slope analysis
    slopes: np.ndarray
    slope_changes: np.ndarray
    increasing_slopes: bool
    max_slope_increase: float

    # Regime identification
    small_radius_slope: float
    large_radius_slope: float
    transition_radius: float

    # Statistical details
    cfar_threshold: float
    detected_discontinuities: List[int]
    holm_bonferroni_enabled: bool  # Whether multiple testing correction is enabled (applied at vocabulary level)

    # Local signal dimension (semantic flexibility)
    local_signal_dimension: float

    # Diagnostic PCA dimensions (not from paper, but useful)
    signal_dimension: float  # PCA dimension far from point (semantic variation)
    noise_dimension: float   # PCA dimension near point (local variation)


class RobinsonFiberBundleTest:
    """
    Implementation of fiber bundle hypothesis test for high-dimensional data.

    This test analyzes volume growth patterns to detect violations
    of the fiber bundle hypothesis in embedding spaces.
    """

    def __init__(
        self,
        significance_level: float = 0.001,  # Paper uses 10^-3
        min_radius: float = 0.1,
        max_radius: float = 10.0,
        n_radii: int = 50,
        cfar_window: int = 5,
        cfar_guard: int = 2,
        use_holm_bonferroni: bool = True,
        max_embeddings_for_exact: int = 10000,  # For computational efficiency
        bootstrap_samples: int = 10000,  # Bootstrap sample size for statistical validity
        seed: Optional[int] = None  # ICML FIX: Seed for reproducibility
    ):
        """
        Initialize Robinson fiber bundle test.

        Statistical Note (ICML Documentation):
            For large vocabularies (>max_embeddings_for_exact), we use bootstrap sampling
            to estimate volume growth patterns. This maintains statistical validity while
            improving computational efficiency from O(n²) to O(n*bootstrap_samples).

            **Bootstrap Sampling Impact:**
            1. **Unbiased Estimation**: Volume counts are scaled by (n-1)/sample_size to
               provide unbiased estimates of true volume growth.
            2. **Statistical Power**: Effective sample size for hypothesis testing is
               reduced from n to bootstrap_samples. This increases Type II error rate
               (false negatives) but maintains α = 0.001 for Type I errors.
            3. **Reproducibility**: When seed parameter is provided, bootstrap sampling
               is deterministic. Without seed, results will vary across runs for large
               vocabularies (>10k tokens).
            4. **Recommendation for ICML**: For reproducible results, always specify seed.
               For final publication results, consider running multiple seeds and reporting
               confidence intervals or using conservative interpretation.

        Args:
            significance_level: Statistical significance (paper uses 0.001)
            min_radius: Minimum radius to test
            max_radius: Maximum radius to test
            n_radii: Number of radius values to sample
            cfar_window: Window size for CFAR detector
            cfar_guard: Guard cells for CFAR detector
            use_holm_bonferroni: Apply Holm-Bonferroni correction (should be at vocab level)
            max_embeddings_for_exact: Threshold for exact vs bootstrap (default: 10000)
            bootstrap_samples: Sample size for bootstrap (default: 10000)
            seed: Random seed for reproducibility (default: None = non-deterministic)
        """
        self.significance_level = significance_level
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.n_radii = n_radii
        self.cfar_window = cfar_window
        self.cfar_guard = cfar_guard
        self.use_holm_bonferroni = use_holm_bonferroni
        self.max_embeddings_for_exact = max_embeddings_for_exact
        self.bootstrap_samples = bootstrap_samples
        self.seed = seed  # ICML FIX: Store seed for reproducibility

        # ICML FIX: Initialize random state for reproducibility
        if seed is not None:
            self.rng = np.random.RandomState(seed)
        else:
            self.rng = np.random

        # Generate radius values (logarithmically spaced)
        self.radii = np.logspace(
            np.log10(min_radius),
            np.log10(max_radius),
            n_radii
        )

    def test_point(
        self,
        embeddings: np.ndarray,
        point_idx: int,
        precomputed_distances: Optional[np.ndarray] = None
    ) -> RobinsonTestResult:
        """
        Test fiber bundle hypothesis at a specific point.

        This is the core test from the paper - analyzes how
        neighborhood volume grows with radius.

        Args:
            embeddings: (n_points, n_dims) embedding matrix
            point_idx: Index of point to test
            precomputed_distances: Optional distance matrix

        Returns:
            RobinsonTestResult with detailed analysis
        """
        # Check input precision and warn if too low
        if embeddings.dtype == np.float16:
            warnings.warn(
                "Robinson fiber bundle test with float16 data: "
                "Results may be unreliable. The test requires float32+ for accurate "
                "log-log slope analysis and distance calculations in high dimensions.",
                UserWarning
            )
        n_points, n_dims = embeddings.shape

        # Compute distances if not provided
        sample_indices = None  # Track for PCA indexing fix
        if precomputed_distances is None:
            # Use bootstrap sampling for large vocabularies
            if n_points <= self.max_embeddings_for_exact:
                # Exact computation for small vocabularies
                distances_to_point = np.linalg.norm(
                    embeddings - embeddings[point_idx], axis=1
                )
            else:
                # Bootstrap sampling for large vocabularies (statistically valid)
                # This provides an unbiased estimate of volume growth patterns
                # ICML FIX: Sample from indices EXCLUDING the point itself to avoid bias
                # Use self.rng for reproducibility when seed is set
                all_other_indices = np.setdiff1d(np.arange(n_points), [point_idx])
                sample_size = min(self.bootstrap_samples, len(all_other_indices))
                sample_indices = self.rng.choice(
                    all_other_indices,
                    size=sample_size,
                    replace=False
                )

                # Compute distances only to sampled points
                sampled_embeddings = embeddings[sample_indices]
                sampled_distances = np.linalg.norm(
                    sampled_embeddings - embeddings[point_idx],
                    axis=1
                )

                # Create a sparse distance array for volume computation
                # FIX: Correct scaling factor for unbiased estimate
                # We sampled from n-1 points (excluding center), so scale by (n-1)/sample_size
                scaling_factor = (n_points - 1) / sample_size
                distances_to_point = sampled_distances
        else:
            distances_to_point = precomputed_distances[point_idx]
            scaling_factor = 1.0

        # Auto-scale radii based on actual distance distribution
        # This ensures we capture the relevant scale for each embedding
        positive_distances = distances_to_point[distances_to_point > 0]
        if len(positive_distances) > 0:
            # Use 5th nearest neighbor (or 1st percentile) as minimum radius
            # and 90th percentile as maximum radius for robust scaling
            sorted_distances = np.sort(positive_distances)
            r_min = sorted_distances[min(4, len(sorted_distances) // 100)]  # 5th neighbor or 1st percentile
            r_max = np.percentile(sorted_distances, 90)

            # Guard against degenerate cases and numerical issues
            if not np.isfinite(r_min) or not np.isfinite(r_max) or r_max <= r_min:
                # Fall back to default radii if data-driven approach fails
                radii = self.radii
            else:
                # Generate radii in this data-driven range
                radii = np.logspace(np.log10(r_min), np.log10(r_max), self.n_radii)
        else:
            # Fallback to default if no positive distances
            radii = self.radii

        # Count points within each radius (volume growth)
        # CRITICAL FIX: Exclude the center point itself (distance = 0)
        # PERFORMANCE FIX: Fully vectorized volume counting
        positive_mask = distances_to_point > 0
        positive_distances = distances_to_point[positive_mask]

        if len(positive_distances) > 0:
            # Fully vectorized counting using searchsorted
            sorted_dists = np.sort(positive_distances)
            counts = np.searchsorted(sorted_dists, radii, side='right').astype(float)
            if n_points > self.max_embeddings_for_exact and precomputed_distances is None:
                volumes = counts * scaling_factor
            else:
                volumes = counts
        else:
            # No positive distances, all volumes are 0
            volumes = np.zeros(len(radii))

        # Mask early radii until we have enough neighbors (k=5 minimum)
        min_neighbors = 5
        valid_mask = volumes >= min_neighbors
        if not np.any(valid_mask):
            # If we don't have enough neighbors, use all points but warn
            valid_mask = volumes > 0

        # Only use radii with sufficient statistics
        radii = radii[valid_mask]
        volumes = volumes[valid_mask]

        # Edge case: No valid radii at all
        if len(radii) == 0:
            # Return default result for degenerate case
            return RobinsonTestResult(
                violates_manifold=False,
                violates_fiber_bundle=False,
                violates_hypothesis=False,
                p_value=1.0,
                radii=np.array([1.0]),
                volumes=np.array([0.0]),
                log_radii=np.array([0.0]),
                log_volumes=np.array([0.0]),
                slopes=np.array([0.0]),
                slope_changes=np.array([]),
                increasing_slopes=False,
                max_slope_increase=0.0,
                small_radius_slope=0.0,
                large_radius_slope=0.0,
                transition_radius=1.0,
                cfar_threshold=self._compute_cfar_threshold(),
                detected_discontinuities=[],
                holm_bonferroni_enabled=False,  # Correction should be done at vocabulary level, not per-token
                local_signal_dimension=float(embeddings.shape[1]),
                signal_dimension=float(embeddings.shape[1] // 2),
                noise_dimension=float(embeddings.shape[1] // 4)
            )

        if len(radii) < 10:
            # Not enough data points for reliable analysis
            warnings.warn(
                f"Only {len(radii)} radii have sufficient neighbors for analysis. "
                "Results may be unreliable.",
                UserWarning
            )

        # Compute log-log representation
        log_radii = np.log(radii)
        # ICML FIX: Use machine epsilon instead of magic constant
        log_volumes = np.log(volumes + np.finfo(np.float64).eps)  # Avoid log(0)

        # Estimate slopes using three-point centered differences
        slopes = self._compute_centered_slopes(log_radii, log_volumes)

        # Detect slope changes
        slope_changes = np.diff(slopes)

        # Apply CFAR detector for discontinuity detection
        discontinuities = self._cfar_detector(slope_changes)

        # Check for increasing slopes (violation indicator)
        increasing_slopes = self._detect_increasing_slopes(
            slopes, discontinuities
        )

        # Estimate reach for valid hypothesis testing
        estimated_reach = self._estimate_reach(log_radii, log_volumes, radii, discontinuities)

        # Identify regime transition
        transition_idx = self._find_regime_transition(slopes)

        # Compute regime-specific slopes
        if transition_idx > 0:
            small_radius_slope = np.mean(slopes[:transition_idx])
            large_radius_slope = np.mean(slopes[transition_idx:])
            transition_radius = radii[transition_idx]  # FIX: Use local radii, not self.radii
        else:
            small_radius_slope = np.mean(slopes[:len(slopes)//2])
            large_radius_slope = np.mean(slopes[len(slopes)//2:])
            transition_radius = radii[len(slopes)//2]  # FIX: Use local radii, not self.radii

        # Compute p-value
        p_value = self._compute_p_value(
            slopes, slope_changes, discontinuities
        )

        # No double correction - p-value is already computed correctly
        # Multiple testing correction should be applied at the vocabulary level,
        # not per-token (as per paper's methodology)

        # Compute local signal dimension (new!)
        # FIX: Pass sample_indices when using bootstrap sampling
        local_signal_dim = self._compute_local_signal_dimension(
            embeddings, point_idx, distances_to_point, sample_indices
        )

        # PAPER-COMPLIANT: Compute separate signal and noise dimensions
        # FIX: Pass sample_indices when using bootstrap sampling
        signal_dim, noise_dim = self._compute_signal_noise_dimensions(
            embeddings, point_idx, distances_to_point, transition_radius, sample_indices
        )

        # Implement dual rejection logic from paper's flowchart
        # First check: Does slope change between regimes?
        has_significant_regime_change = self._detect_significant_regime_change(
            slopes, transition_idx
        )

        # Don't make strong conclusions with very sparse data
        # Need at least 10 valid radii for reliable hypothesis testing
        if len(radii) < 10:
            violates_manifold = False
            violates_fiber_bundle = False
        # Apply decision tree from paper with reach gating:
        # Only reject hypotheses within the estimated reach
        elif transition_radius <= estimated_reach:
            # Within valid testing region - apply tests
            if not has_significant_regime_change:
                # No slope change -> Reject manifold hypothesis
                violates_manifold = True
                violates_fiber_bundle = False
            elif increasing_slopes:
                # Slope increases -> Reject fiber bundle hypothesis
                violates_manifold = False
                violates_fiber_bundle = True
            else:
                # Slope decreases or stays constant -> Accept fiber bundle
                violates_manifold = False
                violates_fiber_bundle = False
        else:
            # Beyond reach - be conservative and don't reject
            # (per paper Section 4.1: rejections beyond reach may be spurious)
            violates_manifold = False
            violates_fiber_bundle = False

        # FIX: Remove p_value from decision to avoid bypassing reach gating
        # Legacy: combined violation (for backward compatibility)
        violates = violates_manifold or violates_fiber_bundle

        return RobinsonTestResult(
            violates_manifold=violates_manifold,
            violates_fiber_bundle=violates_fiber_bundle,
            violates_hypothesis=violates,
            p_value=p_value,
            radii=radii,
            volumes=volumes,
            log_radii=log_radii,
            log_volumes=log_volumes,
            slopes=slopes,
            slope_changes=slope_changes,
            increasing_slopes=increasing_slopes,
            max_slope_increase=np.max(slope_changes) if len(slope_changes) > 0 else 0,
            small_radius_slope=small_radius_slope,
            large_radius_slope=large_radius_slope,
            transition_radius=transition_radius,
            cfar_threshold=self._compute_cfar_threshold(),
            detected_discontinuities=discontinuities,
            holm_bonferroni_enabled=False,  # Correction should be done at vocabulary level, not per-token
            local_signal_dimension=local_signal_dim,  # New field!
            signal_dimension=signal_dim,  # PAPER-COMPLIANT
            noise_dimension=noise_dim     # PAPER-COMPLIANT
        )

    def _compute_centered_slopes(
        self,
        log_radii: np.ndarray,
        log_volumes: np.ndarray
    ) -> np.ndarray:
        """
        Compute slopes using three-point centered differences with explicit division-by-zero handling.

        Generic numerical differentiation method for log-log data.
        ICML FIX: Added explicit checks to prevent division by zero.
        """
        n = len(log_radii)
        slopes = np.zeros(n, dtype=float)
        eps = np.finfo(np.float64).eps  # Machine epsilon

        if n == 1:
            slopes[0] = 0.0
            return slopes

        if n == 2:
            # Only two points: use forward difference
            denominator = log_radii[1] - log_radii[0]
            if abs(denominator) < eps:
                slopes[:] = 0.0  # Degenerate case: radii are identical
            else:
                slopes[:] = (log_volumes[1] - log_volumes[0]) / denominator
            return slopes

        # Handle boundaries with forward/backward differences
        # Forward difference at start
        denom = log_radii[1] - log_radii[0]
        if abs(denom) >= eps:
            slopes[0] = (log_volumes[1] - log_volumes[0]) / denom
        else:
            slopes[0] = 0.0  # Degenerate case

        # Backward difference at end
        denom = log_radii[-1] - log_radii[-2]
        if abs(denom) >= eps:
            slopes[-1] = (log_volumes[-1] - log_volumes[-2]) / denom
        else:
            slopes[-1] = 0.0  # Degenerate case

        # Three-point centered differences for interior points
        for i in range(1, n-1):
            denom = log_radii[i+1] - log_radii[i-1]
            if abs(denom) >= eps:
                slopes[i] = (log_volumes[i+1] - log_volumes[i-1]) / denom
            else:
                slopes[i] = 0.0  # Degenerate case: nearly identical radii

        return slopes

    def _cfar_detector(
        self,
        signal: np.ndarray
    ) -> List[int]:
        """
        CFAR-style detector for discontinuities.

        Heuristic detector for significant slope changes while controlling false positives.
        Note: This is not from the Robinson paper, but a generic signal processing technique.
        """
        n = len(signal)
        if n < 2 * self.cfar_window + 2 * self.cfar_guard + 1:
            return []

        discontinuities = []
        threshold_multiplier = -stats.norm.ppf(self.significance_level)  # One-sided test

        for i in range(self.cfar_window + self.cfar_guard,
                      n - self.cfar_window - self.cfar_guard):
            # Left window
            left_start = i - self.cfar_window - self.cfar_guard
            left_end = i - self.cfar_guard
            left_samples = signal[left_start:left_end]

            # Right window
            right_start = i + self.cfar_guard + 1
            right_end = i + self.cfar_guard + self.cfar_window + 1
            right_samples = signal[right_start:right_end]

            # Estimate noise level
            noise_samples = np.concatenate([left_samples, right_samples])
            # ICML FIX: Use machine epsilon
            noise_level = np.std(noise_samples) + np.finfo(np.float64).eps  # Avoid division by zero

            # CFAR threshold
            threshold = threshold_multiplier * noise_level

            # One-sided test for POSITIVE jumps only (slope increases)
            # We only care about violations (increasing slopes)
            if signal[i] > threshold:
                discontinuities.append(i)

        return discontinuities

    def _detect_increasing_slopes(
        self,
        slopes: np.ndarray,
        discontinuities: List[int]
    ) -> bool:
        """
        Check if slopes increase through discontinuities.

        This is the key violation indicator in the paper.
        """
        if len(discontinuities) == 0:
            # No discontinuities, check overall trend
            return self._is_increasing_trend(slopes)

        # Check slope changes at discontinuities
        for disc_idx in discontinuities:
            if disc_idx > 0 and disc_idx < len(slopes) - 1:
                # Compare slopes before and after discontinuity
                before = np.mean(slopes[max(0, disc_idx-3):disc_idx])
                after = np.mean(slopes[disc_idx+1:min(len(slopes), disc_idx+4)])

                # ICML FIX: Use noise-scaled test consistent with significance_level
                # Compute local noise level from slopes
                local_noise = np.std(slopes[max(0, disc_idx-5):min(len(slopes), disc_idx+5)]) + np.finfo(np.float64).eps
                zcrit = -stats.norm.ppf(self.significance_level)  # one-sided z-critical value

                if (after - before) > zcrit * local_noise:  # Significant increase
                    return True

        return False

    def _is_increasing_trend(self, slopes: np.ndarray) -> bool:
        """Check if slopes have statistically significant increasing trend.

        Uses Mann-Kendall test as per Robinson paper's formal approach.
        Returns True only if p-value < significance_level (10⁻³).
        """
        if len(slopes) < 3:
            return False

        # Use Mann-Kendall test for monotonic increasing trend
        from scipy.stats import kendalltau, norm

        x = np.arange(len(slopes))
        tau, _ = kendalltau(x, slopes, method="auto")

        # Compute one-sided p-value for increasing trend
        n = len(slopes)
        var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1)) if n > 1 else np.inf

        if not np.isfinite(var_tau) or var_tau == 0:
            return False

        z = tau / np.sqrt(var_tau)
        p_value = 1.0 - norm.cdf(z)  # One-sided test for increase

        # Robinson uses α = 10⁻³
        return p_value < self.significance_level

    def _detect_significant_regime_change(self, slopes: np.ndarray, transition_idx: int) -> bool:
        """
        Detect if there's a statistically significant change in slope between regimes.

        This determines whether we have a true fiber bundle structure (with regime change)
        or just a manifold (constant slope).

        Args:
            slopes: Array of computed slopes
            transition_idx: Index where regime transition occurs

        Returns:
            True if significant regime change detected, False otherwise
        """
        if len(slopes) < 5 or transition_idx < 2 or transition_idx >= len(slopes) - 2:
            return False  # Not enough data

        # Compare slopes before and after transition
        before_slopes = slopes[:transition_idx]
        after_slopes = slopes[transition_idx:]

        if len(before_slopes) < 2 or len(after_slopes) < 2:
            return False

        # Compute statistics for each regime
        mean_before = np.mean(before_slopes)
        mean_after = np.mean(after_slopes)
        std_before = np.std(before_slopes) if len(before_slopes) > 1 else 0
        std_after = np.std(after_slopes) if len(after_slopes) > 1 else 0

        # Robinson paper: Use proper statistical test with p < 10⁻³
        # Welch's t-test for unequal variances
        from scipy import stats

        # Need at least 2 samples in each group
        if len(before_slopes) < 2 or len(after_slopes) < 2:
            return False

        # Proper two-sample t-test
        t_stat, p_value = stats.ttest_ind(before_slopes, after_slopes, equal_var=False)

        # Robinson uses α = 10⁻³ for all tests
        return p_value < self.significance_level

    def _estimate_reach(
        self,
        log_radii: np.ndarray,
        log_volumes: np.ndarray,
        radii: np.ndarray,
        discontinuities: List[int]
    ) -> float:
        """
        Estimate the embedded reach for valid hypothesis testing.

        The reach is the maximum radius within which the manifold hypothesis
        is valid. Beyond the reach, the test may produce spurious rejections.

        Conservative approach: use the first major discontinuity or the radius
        where curvature changes significantly.
        """
        if len(discontinuities) > 0:
            # Use first significant discontinuity as reach estimate
            first_disc_idx = min(discontinuities)
            if first_disc_idx < len(radii):
                return radii[first_disc_idx]

        # Alternative: use radius at 50th percentile as conservative estimate
        # This ensures we're testing in the dense region of the embedding
        if len(radii) > 0:
            return radii[len(radii) // 2]

        # Fallback to maximum radius if no better estimate
        return radii[-1] if len(radii) > 0 else 10.0

    def _find_regime_transition(self, slopes: np.ndarray) -> int:
        """
        Find transition between small and large radius regimes using change-point detection.

        Robinson paper uses statistical change-point detection, not heuristics.
        We implement a simple likelihood-ratio based approach with robust variance estimation.

        ICML NOTE: We use standard variance (not MAD) for BIC because:
        1. BIC assumes Gaussian likelihood which naturally uses variance
        2. Outliers in slopes are informative (not noise to be rejected)
        3. Robust methods would reduce sensitivity to regime changes
        """
        if len(slopes) < 5:
            return len(slopes) // 2

        # Use change-point detection with BIC/likelihood ratio
        best_idx = len(slopes) // 2
        min_bic = float('inf')

        for i in range(2, len(slopes) - 2):
            # Split data
            left = slopes[:i]
            right = slopes[i:]

            # Compute BIC for split model vs single model
            # Split model: two different means
            left_mean = np.mean(left)
            right_mean = np.mean(right)
            # ICML FIX: Use machine epsilon (standard variance is appropriate here)
            left_var = np.var(left) + np.finfo(np.float64).eps
            right_var = np.var(right) + np.finfo(np.float64).eps

            # Log-likelihood for split model
            ll_split = -0.5 * (len(left) * np.log(left_var) +
                              len(right) * np.log(right_var) +
                              np.sum((left - left_mean)**2) / left_var +
                              np.sum((right - right_mean)**2) / right_var)

            # BIC = -2*log_likelihood + k*log(n)
            # k=4 parameters (2 means, 2 variances)
            bic = -2 * ll_split + 4 * np.log(len(slopes))

            if bic < min_bic:
                min_bic = bic
                best_idx = i

        return best_idx

    def _compute_p_value(
        self,
        slopes: np.ndarray,
        slope_changes: np.ndarray,
        discontinuities: List[int]
    ) -> float:
        """
        Compute p-value for fiber bundle hypothesis test.

        Tests null hypothesis: slopes are non-increasing (proper fiber bundle)
        Alternative hypothesis: slopes are increasing (violates fiber bundle)

        Uses one-sided Mann-Kendall test for monotonic increasing trend.
        """
        if len(slopes) < 3:
            return 1.0  # Not enough data, fail to reject null

        # Use Mann-Kendall test for monotonic trend
        # This is a non-parametric test that doesn't assume normality
        try:
            from scipy.stats import kendalltau, norm

            # Compute Kendall's tau correlation between position and slope
            x = np.arange(len(slopes))
            tau, _ = kendalltau(x, slopes, method="auto")

            # Compute proper one-sided p-value using normal approximation
            n = len(slopes)
            # Variance of tau under null hypothesis
            var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1)) if n > 1 else np.inf

            if not np.isfinite(var_tau) or var_tau == 0:
                z = 0.0
            else:
                z = tau / np.sqrt(var_tau)

            # One-sided p-value for increasing trend (fiber bundle violation)
            # Higher z means more evidence of increasing trend
            p_value = 1.0 - norm.cdf(z)

        except Exception as e:
            warnings.warn(f"Mann-Kendall test failed: {e}. Using neutral p-value.")
            p_value = 1.0

        return np.clip(p_value, 0, 1)

    def _compute_cfar_threshold(self) -> float:
        """Compute CFAR detection threshold.

        FIX: Use one-sided test consistent with CFAR detector.
        """
        return -stats.norm.ppf(self.significance_level)

    def _holm_bonferroni_correction(
        self,
        p_value: float,
        n_tests: int
    ) -> float:
        """
        Apply Holm-Bonferroni correction for multiple testing.

        The paper uses this to control family-wise error rate.
        """
        # Adjust significance level
        adjusted_alpha = self.significance_level / n_tests

        # Scale p-value
        adjusted_p = min(1.0, p_value * n_tests)

        return adjusted_p

    def _compute_local_signal_dimension(
        self,
        embeddings: np.ndarray,
        point_idx: int,
        distances: np.ndarray,
        sample_indices: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute local signal dimension (semantic flexibility).

        This measures how many meaningful directions exist in the
        token's semantic neighborhood. Higher dimension = more flexibility.

        Robinson paper mentions the CONCEPT of local signal dimension
        causing output variability, but does NOT provide a computation method.
        This PCA-based implementation is our contribution.

        Args:
            embeddings: Full embedding matrix
            point_idx: Index of point to analyze
            distances: Distances to all other points

        Returns:
            Estimated local signal dimension
        """
        # Find semantic neighborhood (medium radius)
        # Use transition radius if available
        radius = np.median(distances[distances > 0])
        if radius == 0:
            radius = 1.0

        # Get neighbors within semantic radius
        neighbor_mask = (distances > 0) & (distances <= radius)

        # FIX: Handle bootstrap sampling case - indices refer to sampled subset
        if sample_indices is not None:
            # Map mask indices to actual embedding indices
            neighbor_indices_in_sample = np.where(neighbor_mask)[0]
            if len(neighbor_indices_in_sample) < 3:
                return float(embeddings.shape[1])
            neighbor_indices = sample_indices[neighbor_indices_in_sample]
        else:
            neighbor_indices = np.where(neighbor_mask)[0]
            if len(neighbor_indices) < 3:
                return float(embeddings.shape[1])

        # Get neighbor embeddings relative to center
        center = embeddings[point_idx]
        neighbors = embeddings[neighbor_indices] - center

        # Compute PCA to find effective dimension
        try:
            pca = PCA(n_components=min(len(neighbor_indices), embeddings.shape[1]))
            pca.fit(neighbors)

            # Compute explained variance ratio
            explained_var = pca.explained_variance_ratio_

            # Find effective dimension using 95% variance threshold
            cumsum = np.cumsum(explained_var)
            effective_dim = np.argmax(cumsum >= 0.95) + 1

            # Alternative: use entropy-based dimension
            # Normalize eigenvalues
            eigenvalues = pca.explained_variance_
            eigenvalues = eigenvalues / np.sum(eigenvalues)

            # Compute entropy - ICML FIX: Use machine epsilon
            entropy = -np.sum(eigenvalues * np.log(eigenvalues + np.finfo(np.float64).eps))

            # Convert entropy to effective dimension
            # Max entropy = log(n), so effective_dim = exp(entropy)
            entropy_dim = np.exp(entropy)

            # Return weighted average
            local_dim = 0.7 * effective_dim + 0.3 * entropy_dim

            return float(local_dim)

        except Exception as e:
            # Fallback to simple estimate
            return float(min(len(neighbor_indices), embeddings.shape[1] // 2))

    def _compute_signal_noise_dimensions(
        self,
        embeddings: np.ndarray,
        point_idx: int,
        distances: np.ndarray,
        transition_radius: float,
        sample_indices: Optional[np.ndarray] = None
    ) -> Tuple[float, float]:
        """
        Compute separate signal and noise dimensions (diagnostic, not from paper).

        According to the paper, fiber bundles have TWO local dimensions:
        - Noise dimension: Valid near the point (within transition radius)
        - Signal dimension: Valid far from the point (beyond transition radius)

        This separation is KEY to the fiber bundle hypothesis.

        Args:
            embeddings: Full embedding matrix
            point_idx: Index of point to analyze
            distances: Distances to all other points
            transition_radius: Radius separating regimes

        Returns:
            (signal_dimension, noise_dimension) tuple
        """
        # Ensure we have full distances for PCA neighborhoods
        # When bootstrap sampling is used, distances may be partial
        if distances is None or len(distances) != embeddings.shape[0]:
            # Recompute exact distances for this point (needed for proper PCA)
            distances = np.linalg.norm(embeddings - embeddings[point_idx], axis=1)

        # Near neighborhood (noise dimension)
        near_mask = (distances > 0) & (distances <= transition_radius)
        near_indices = np.where(near_mask)[0]

        # Far neighborhood (signal dimension)
        far_mask = (distances > transition_radius) & (distances <= transition_radius * 3)
        far_indices = np.where(far_mask)[0]

        # Compute noise dimension (near regime)
        if len(near_indices) >= 3:
            near_embeddings = embeddings[near_indices] - embeddings[point_idx]
            try:
                pca_near = PCA(n_components=min(len(near_indices), embeddings.shape[1]))
                pca_near.fit(near_embeddings)

                # Noise dimension: 95% variance threshold
                cumsum_near = np.cumsum(pca_near.explained_variance_ratio_)
                noise_dim = float(np.argmax(cumsum_near >= 0.95) + 1)
            except:
                noise_dim = float(embeddings.shape[1] // 4)  # Fallback
        else:
            noise_dim = float(embeddings.shape[1] // 4)

        # Compute signal dimension (far regime)
        if len(far_indices) >= 3:
            far_embeddings = embeddings[far_indices] - embeddings[point_idx]
            try:
                pca_far = PCA(n_components=min(len(far_indices), embeddings.shape[1]))
                pca_far.fit(far_embeddings)

                # Signal dimension: 95% variance threshold
                cumsum_far = np.cumsum(pca_far.explained_variance_ratio_)
                signal_dim = float(np.argmax(cumsum_far >= 0.95) + 1)
            except:
                signal_dim = float(embeddings.shape[1] // 2)  # Fallback
        else:
            signal_dim = float(embeddings.shape[1] // 2)

        # According to paper: signal dimension should be >= noise dimension
        # for a proper fiber bundle
        if signal_dim < noise_dim:
            # This itself is a violation indicator!
            pass

        return signal_dim, noise_dim

    def visualize_volume_growth(
        self,
        result: RobinsonTestResult,
        save_path: Optional[str] = None,
        title: Optional[str] = None
    ) -> None:
        """
        PAPER-EMPHASIZED: Visualize volume growth in log-log space.

        The paper emphasizes these plots as KEY to understanding violations.
        Shows how neighborhood volume scales with radius, revealing:
        - Piecewise linear segments (expected)
        - Slope increases (violations)
        - Regime transitions

        Args:
            result: Test result containing volume growth data
            save_path: Optional path to save figure
            title: Optional custom title
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Main log-log plot (paper's primary visualization)
        ax = axes[0, 0]
        ax.plot(result.log_radii, result.log_volumes, 'b-', linewidth=2, label='Actual')

        # Mark regime transition
        transition_idx = np.argmin(np.abs(result.radii - result.transition_radius))
        ax.axvline(result.log_radii[transition_idx], color='r', linestyle='--',
                  alpha=0.5, label='Regime transition')

        # Add theoretical lines for each regime
        # Small radius: slope = embedding dimension
        small_x = result.log_radii[:transition_idx]
        if len(small_x) > 0:
            small_y_theory = small_x * result.small_radius_slope + result.log_volumes[0]
            ax.plot(small_x, small_y_theory, 'g--', alpha=0.7, label='Small radius fit')

        # Large radius: different slope
        large_x = result.log_radii[transition_idx:]
        if len(large_x) > 0:
            large_y_theory = large_x * result.large_radius_slope + result.log_volumes[transition_idx]
            ax.plot(large_x, large_y_theory, 'r--', alpha=0.7, label='Large radius fit')

        ax.set_xlabel('log(radius)')
        ax.set_ylabel('log(volume)')
        ax.set_title('Volume Growth Pattern (KEY DIAGNOSTIC)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add violation indicator
        if result.violates_hypothesis:
            ax.text(0.05, 0.95, 'VIOLATION DETECTED', transform=ax.transAxes,
                   color='red', fontweight='bold', va='top')

        # 2. Slopes over radius
        ax = axes[0, 1]
        ax.plot(result.radii[:-1], result.slopes[:-1], 'b-', linewidth=2)

        # Mark discontinuities
        for disc_idx in result.detected_discontinuities:
            if disc_idx < len(result.radii):
                ax.axvline(result.radii[disc_idx], color='r', linestyle=':', alpha=0.5)

        ax.set_xlabel('Radius')
        ax.set_ylabel('Slope (d log(V) / d log(r))')
        ax.set_title('Slope Evolution')
        ax.grid(True, alpha=0.3)

        # Highlight increasing slopes (violations)
        if result.increasing_slopes:
            ax.fill_between(result.radii[:-1], result.slopes[:-1], alpha=0.3, color='red')
            ax.text(0.05, 0.95, 'INCREASING SLOPES', transform=ax.transAxes,
                   color='red', fontweight='bold', va='top')

        # 3. Slope changes
        ax = axes[1, 0]
        # FIX: Use proper bar plot with midpoints and widths
        if len(result.slope_changes) > 0 and len(result.radii) > 1:
            # Compute midpoints and widths for bars
            midpoints = (result.radii[:-1] + result.radii[1:]) / 2
            if len(midpoints) > len(result.slope_changes):
                midpoints = midpoints[:len(result.slope_changes)]
            widths = np.diff(result.radii)[:len(result.slope_changes)]
            ax.bar(midpoints, result.slope_changes, width=widths,
                  alpha=0.7, edgecolor='black')
        else:
            # Fallback for edge cases
            ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                   transform=ax.transAxes)
        ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Radius')
        ax.set_ylabel('Slope Change')
        ax.set_title('Slope Changes (Positive = Violation)')
        ax.grid(True, alpha=0.3)

        # Mark significant changes
        # FIX: Use consistent one-sided threshold from result
        if len(result.slope_changes) > 0:
            threshold = result.cfar_threshold * np.std(result.slope_changes)
            ax.axhline(threshold, color='r', linestyle='--', alpha=0.5, label='CFAR threshold')
        else:
            threshold = 0
        ax.legend()

        # 4. Dimension analysis
        ax = axes[1, 1]
        dimensions = [result.noise_dimension, result.signal_dimension]
        labels = ['Noise\n(Near)', 'Signal\n(Far)']
        colors = ['blue', 'green']

        bars = ax.bar(labels, dimensions, color=colors, alpha=0.7, edgecolor='black')
        ax.set_ylabel('Dimension')
        ax.set_title('Signal vs Noise Dimensions')
        ax.grid(True, alpha=0.3, axis='y')

        # Add local signal dimension for comparison
        ax.axhline(result.local_signal_dimension, color='orange', linestyle='--',
                  alpha=0.7, label=f'Combined: {result.local_signal_dimension:.1f}')
        ax.legend()

        # Annotate bars
        for bar, dim in zip(bars, dimensions):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{dim:.1f}', ha='center', va='bottom')

        # Overall title
        if title:
            fig.suptitle(title, fontsize=14, fontweight='bold')
        else:
            status = "VIOLATES" if result.violates_hypothesis else "SATISFIES"
            fig.suptitle(f'Fiber Bundle Test: {status} Hypothesis (p={result.p_value:.4f})',
                        fontsize=14, fontweight='bold')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')

        plt.show()


def analyze_embedding_space(
    embeddings: np.ndarray,
    n_samples: int = 100,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Analyze entire embedding space for fiber bundle violations.

    Args:
        embeddings: (n_points, n_dims) embedding matrix
        n_samples: Number of points to test
        seed: Random seed for sampling

    Returns:
        Summary statistics and identified problematic regions
    """
    np.random.seed(seed)
    n_points = len(embeddings)

    # Sample points to test
    if n_samples >= n_points:
        test_indices = np.arange(n_points)
    else:
        test_indices = np.random.choice(n_points, n_samples, replace=False)

    # Initialize test
    tester = RobinsonFiberBundleTest()

    # Precompute distance matrix if feasible
    if n_points < 10000:
        print("Precomputing distance matrix...")
        distances = cdist(embeddings, embeddings)
    else:
        distances = None

    # Run tests
    results = []
    violations = 0

    print(f"Testing {len(test_indices)} points...")
    for i, idx in enumerate(test_indices):
        if i % 10 == 0:
            print(f"  Progress: {i}/{len(test_indices)}")

        result = tester.test_point(embeddings, idx, distances)
        results.append(result)

        if result.violates_hypothesis:
            violations += 1

    # Compute summary statistics
    violation_rate = violations / len(test_indices)
    avg_p_value = np.mean([r.p_value for r in results])

    # Identify worst violators
    worst_indices = sorted(
        range(len(results)),
        key=lambda i: results[i].max_slope_increase,
        reverse=True
    )[:10]

    print(f"\nResults:")
    print(f"  Violation rate: {violation_rate:.1%}")
    print(f"  Average p-value: {avg_p_value:.4f}")
    print(f"  Points with increasing slopes: {sum(r.increasing_slopes for r in results)}")

    return {
        'violation_rate': violation_rate,
        'avg_p_value': avg_p_value,
        'n_increasing_slopes': sum(r.increasing_slopes for r in results),
        'worst_violators': [test_indices[i] for i in worst_indices],
        'results': results
    }


if __name__ == "__main__":
    # Example usage
    print("Robinson Fiber Bundle Test - Example")
    print("=" * 50)

    # Generate synthetic data
    np.random.seed(42)

    # Case 1: Well-behaved manifold-like data
    print("\nTest 1: Manifold-like data (should pass)")
    n_points = 1000
    n_dims = 50

    # Generate data on low-dimensional manifold
    latent_dim = 5
    latent = np.random.randn(n_points, latent_dim)
    projection = np.random.randn(latent_dim, n_dims)
    data_good = latent @ projection + 0.1 * np.random.randn(n_points, n_dims)

    tester = RobinsonFiberBundleTest()
    result = tester.test_point(data_good, 0)
    print(f"  Violates hypothesis: {result.violates_hypothesis}")
    print(f"  p-value: {result.p_value:.4f}")
    print(f"  Increasing slopes: {result.increasing_slopes}")

    # Case 2: Irregular data (should fail)
    print("\nTest 2: Irregular data with discontinuities (should fail)")
    data_bad = np.random.randn(n_points, n_dims)
    # Add irregular structure
    data_bad[n_points//2:] *= 3.0
    data_bad[n_points//4:n_points//2] += np.random.randn(n_points//4, n_dims) * 2

    result = tester.test_point(data_bad, 0)
    print(f"  Violates hypothesis: {result.violates_hypothesis}")
    print(f"  p-value: {result.p_value:.4f}")
    print(f"  Increasing slopes: {result.increasing_slopes}")
    print(f"  Max slope increase: {result.max_slope_increase:.3f}")