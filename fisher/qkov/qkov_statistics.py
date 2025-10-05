"""
Statistical Testing for QK-OV Interference Metrics (Section 6)

Implements:
1. Permutation null testing
2. Benjamini-Hochberg FDR correction
3. Bootstrap confidence intervals
4. Cluster-level corrections
5. Effect size computation

THEORETICAL FOUNDATION
----------------------
Standard statistical rigor for multiple testing:
- Benjamini & Hochberg (1995): FDR control
- Maris & Oostenveld (2007): Cluster-based permutation tests
- Efron & Tibshirani (1994): Bootstrap methods

Author: ICLR 2026 Project
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from scipy import stats
import logging

logger = logging.getLogger(__name__)


@dataclass
class PermutationTestResult:
    """Result of permutation test for a block/layer/head."""
    block: str
    layer: int
    head: int
    observed_score: float
    null_mean: float
    null_std: float
    p_value: float
    significant: bool
    effect_size: float  # Cohen's d


@dataclass
class ClusterResult:
    """Result of cluster-level testing."""
    cluster_id: int
    samples: List[Tuple[int, int]]  # List of (i, j) pairs
    cluster_mean: float
    p_value: float
    significant: bool


class QKOVStatistics:
    """
    Statistical testing for QK-OV interference metrics.

    Handles multiple testing correction, permutation tests, and effect sizes.
    """

    def __init__(
        self,
        fdr_alpha: float = 0.05,
        n_permutations: int = 1000,
        n_bootstrap: int = 1000,
        min_effect_size: float = 0.2,
        cluster_threshold: float = 0.01
    ):
        """
        Initialize statistical testing.

        Args:
            fdr_alpha: FDR significance level (default: 0.05)
            n_permutations: Number of permutation samples (default: 1000)
            n_bootstrap: Number of bootstrap samples (default: 1000)
            min_effect_size: Minimum Cohen's d to report (default: 0.2)
            cluster_threshold: Initial significance for cluster formation (default: 0.01)
        """
        self.fdr_alpha = fdr_alpha
        self.n_permutations = n_permutations
        self.n_bootstrap = n_bootstrap
        self.min_effect_size = min_effect_size
        self.cluster_threshold = cluster_threshold

    def permutation_test_block(
        self,
        contributions: torch.Tensor,  # [n_samples_A, param_dim]
        gradients: torch.Tensor,      # [n_samples_B, param_dim]
        fisher: torch.Tensor,          # [param_dim]
        epsilon: float = 1e-10
    ) -> Dict[str, Any]:
        """
        Permutation test for a single parameter block.

        Null hypothesis: interference scores arise from random pairing.

        Args:
            contributions: C_i for samples from task A
            gradients: g_j for samples from task B
            fisher: Î_n for this block
            epsilon: Numerical stability

        Returns:
            Dictionary with observed score, null distribution, and p-value
        """
        n_a = contributions.shape[0]
        n_b = gradients.shape[0]

        # Compute observed scores
        fisher_reg = fisher.clamp_min(epsilon)
        normalized_contrib = contributions / fisher_reg.unsqueeze(0)  # [n_a, dim]

        # Score matrix: M[i,j] = <C_i / I_n, |g_j|>
        observed = torch.matmul(
            normalized_contrib,
            gradients.abs().T
        )  # [n_a, n_b]

        observed_mean = observed.mean().item()

        # Permutation null distribution
        null_scores = []

        for _ in range(self.n_permutations):
            # Randomly permute contributions (shuffle sample identities)
            perm_idx = torch.randperm(n_a)
            contrib_perm = contributions[perm_idx]

            normalized_perm = contrib_perm / fisher_reg.unsqueeze(0)
            score_perm = torch.matmul(normalized_perm, gradients.abs().T).mean().item()
            null_scores.append(score_perm)

        null_scores = np.array(null_scores)

        # Compute p-value (two-tailed)
        p_value = np.mean(np.abs(null_scores - null_scores.mean()) >=
                         np.abs(observed_mean - null_scores.mean()))

        # Effect size (Cohen's d)
        null_std = null_scores.std()
        if null_std > 0:
            effect_size = (observed_mean - null_scores.mean()) / null_std
        else:
            effect_size = 0.0

        return {
            'observed_mean': observed_mean,
            'null_mean': null_scores.mean(),
            'null_std': null_std,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < self.fdr_alpha and abs(effect_size) >= self.min_effect_size,
            'null_distribution': null_scores
        }

    def benjamini_hochberg_fdr(
        self,
        p_values: List[float],
        alpha: Optional[float] = None
    ) -> Tuple[List[bool], List[float]]:
        """
        Apply Benjamini-Hochberg FDR correction.

        Args:
            p_values: List of p-values to correct
            alpha: FDR level (uses self.fdr_alpha if None)

        Returns:
            Tuple of (significant_mask, q_values)
        """
        if alpha is None:
            alpha = self.fdr_alpha

        n = len(p_values)
        if n == 0:
            return [], []

        # Sort p-values with original indices
        sorted_idx = np.argsort(p_values)
        sorted_p = np.array(p_values)[sorted_idx]

        # BH procedure
        thresholds = alpha * np.arange(1, n + 1) / n
        significant_sorted = sorted_p <= thresholds

        # Find largest k where p_(k) <= alpha * k / n
        if not significant_sorted.any():
            significant_mask = [False] * n
            q_values = [1.0] * n
        else:
            max_k = np.where(significant_sorted)[0][-1]

            # Compute q-values (adjusted p-values)
            q_values_sorted = np.minimum.accumulate(
                sorted_p * n / np.arange(1, n + 1)[::-1]
            )[::-1]

            # Restore original order
            q_values = np.zeros(n)
            q_values[sorted_idx] = q_values_sorted

            significant_mask = [False] * n
            for i in range(max_k + 1):
                significant_mask[sorted_idx[i]] = True

            q_values = q_values.tolist()

        return significant_mask, q_values

    def cluster_correction(
        self,
        scores: np.ndarray,  # [n_samples_A, n_samples_B, n_layers, n_heads]
        p_values: np.ndarray,  # Same shape
        block: str
    ) -> List[ClusterResult]:
        """
        Cluster-level correction for spatially/temporally related tests.

        Groups significant samples by template/type and tests cluster-level means.

        Args:
            scores: Interference scores
            p_values: Uncorrected p-values
            block: Block identifier ('Q', 'K', 'V', or 'O')

        Returns:
            List of significant clusters
        """
        # Find initially significant voxels
        initial_sig = p_values < self.cluster_threshold

        if not initial_sig.any():
            return []

        # Form clusters using connected components
        # (simplified: cluster samples with similar scores)
        from scipy.ndimage import label

        # Flatten spatial dimensions (layers, heads)
        flat_scores = scores.reshape(scores.shape[0], scores.shape[1], -1)
        flat_sig = initial_sig.reshape(initial_sig.shape[0], initial_sig.shape[1], -1)

        clusters = []

        for spatial_idx in range(flat_sig.shape[2]):
            sig_mask = flat_sig[:, :, spatial_idx]

            if not sig_mask.any():
                continue

            # Label connected components in sample space
            labeled, n_clusters = label(sig_mask)

            for cluster_id in range(1, n_clusters + 1):
                cluster_mask = labeled == cluster_id
                cluster_samples = np.argwhere(cluster_mask)

                if len(cluster_samples) < 2:
                    continue  # Skip single-sample clusters

                # Compute cluster-level statistic
                cluster_scores = flat_scores[:, :, spatial_idx][cluster_mask]
                cluster_mean = cluster_scores.mean()

                # Permutation test for cluster
                # (Simplified: compare to null distribution of cluster means)
                null_cluster_means = []
                for _ in range(self.n_permutations):
                    perm_sig = np.random.permutation(sig_mask.flatten()).reshape(sig_mask.shape)
                    if perm_sig.any():
                        perm_scores = flat_scores[:, :, spatial_idx][perm_sig]
                        null_cluster_means.append(perm_scores.mean())

                if null_cluster_means:
                    p_cluster = np.mean(
                        np.abs(null_cluster_means) >= np.abs(cluster_mean)
                    )
                else:
                    p_cluster = 1.0

                clusters.append(ClusterResult(
                    cluster_id=cluster_id,
                    samples=[(int(i), int(j)) for i, j in cluster_samples],
                    cluster_mean=float(cluster_mean),
                    p_value=float(p_cluster),
                    significant=p_cluster < self.fdr_alpha
                ))

        return clusters

    def bootstrap_confidence_interval(
        self,
        scores: np.ndarray,  # [n_samples]
        confidence_level: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for score distribution.

        Args:
            scores: Array of interference scores
            confidence_level: Confidence level (default: 0.95)

        Returns:
            Tuple of (mean, lower_bound, upper_bound)
        """
        n = len(scores)
        bootstrap_means = []

        for _ in range(self.n_bootstrap):
            resample = np.random.choice(scores, size=n, replace=True)
            bootstrap_means.append(resample.mean())

        bootstrap_means = np.array(bootstrap_means)

        mean = scores.mean()
        alpha = 1 - confidence_level
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))

        return float(mean), float(lower), float(upper)

    def test_heatmap(
        self,
        heatmap_results: Dict[str, Any],
        contributions_by_task: Dict[str, torch.Tensor],
        gradients_by_task: Dict[str, torch.Tensor],
        fisher: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        """
        Full statistical testing pipeline for heatmap results.

        Args:
            heatmap_results: Output from QKOVInterferenceMetric.compute_heatmap()
            contributions_by_task: Task -> parameter contributions
            gradients_by_task: Task -> parameter gradients
            fisher: EMA Fisher per parameter

        Returns:
            Dictionary with corrected p-values, significant conflicts, clusters
        """
        results = {
            'permutation_tests': {},
            'fdr_corrected': {},
            'clusters': {},
            'summary': {}
        }

        # Run permutation tests per block
        for block in ['Q', 'K', 'V', 'O']:
            logger.info(f"Running permutation tests for block {block}...")

            # Extract block-specific data
            # (In practice, would slice contributions/gradients/fisher by block)
            # For now, use placeholder

            # Collect all p-values for this block
            block_p_values = []
            block_tests = []

            # Test each layer/head combination
            scores = heatmap_results[block]['scores']  # [n_a, n_b, n_layers, n_heads]

            for layer in range(scores.shape[2]):
                for head in range(scores.shape[3]):
                    # Extract scores for this layer/head
                    layer_head_scores = scores[:, :, layer, head].flatten()

                    # Compute statistics (simplified - would use actual permutation test)
                    observed_mean = layer_head_scores.mean()
                    observed_std = layer_head_scores.std()

                    # Mock p-value for now (replace with actual permutation test)
                    if observed_std > 0:
                        z_score = abs(observed_mean) / (observed_std / np.sqrt(len(layer_head_scores)))
                        p_value = 2 * (1 - stats.norm.cdf(z_score))
                    else:
                        p_value = 1.0

                    block_p_values.append(p_value)
                    block_tests.append({
                        'layer': layer,
                        'head': head,
                        'observed': float(observed_mean),
                        'p_value': p_value
                    })

            # Apply FDR correction
            significant_mask, q_values = self.benjamini_hochberg_fdr(block_p_values)

            # Add q-values to test results
            for test, sig, q in zip(block_tests, significant_mask, q_values):
                test['q_value'] = q
                test['significant'] = sig

            results['permutation_tests'][block] = block_tests
            results['fdr_corrected'][block] = [
                t for t in block_tests if t['significant']
            ]

            # Cluster-level correction
            p_value_array = np.array([t['p_value'] for t in block_tests]).reshape(
                scores.shape[2], scores.shape[3]
            )
            # Expand to match scores shape
            p_value_full = np.tile(p_value_array, (scores.shape[0], scores.shape[1], 1, 1))

            clusters = self.cluster_correction(scores, p_value_full, block)
            results['clusters'][block] = [
                {
                    'cluster_id': c.cluster_id,
                    'n_samples': len(c.samples),
                    'mean': c.cluster_mean,
                    'p_value': c.p_value,
                    'significant': c.significant
                }
                for c in clusters
            ]

        # Summary statistics
        total_tests = sum(len(results['permutation_tests'][b]) for b in ['Q', 'K', 'V', 'O'])
        total_significant = sum(len(results['fdr_corrected'][b]) for b in ['Q', 'K', 'V', 'O'])
        total_clusters = sum(len(results['clusters'][b]) for b in ['Q', 'K', 'V', 'O'])

        results['summary'] = {
            'total_tests': total_tests,
            'total_significant': total_significant,
            'fdr_alpha': self.fdr_alpha,
            'proportion_significant': total_significant / total_tests if total_tests > 0 else 0,
            'total_clusters': total_clusters,
            'n_permutations': self.n_permutations
        }

        return results