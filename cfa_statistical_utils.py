#!/usr/bin/env python3
"""
Statistical utilities for CFA-2-complete.py
Provides proper statistical methods for hypothesis testing and validation
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from scipy import stats
from scipy.stats import wasserstein_distance, bootstrap
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower
import warnings


class StatisticalUtils:
    """Statistical utilities for catastrophic forgetting analysis."""

    @staticmethod
    def determine_sample_size(
        effect_size: float = 0.5,
        power: float = 0.8,
        alpha: float = 0.05,
        test_type: str = 'two-sided'
    ) -> int:
        """
        Determine required sample size for statistical power.

        Based on Cohen (1988) and power analysis best practices.

        Args:
            effect_size: Expected effect size (Cohen's d)
                - 0.2 = small, 0.5 = medium, 0.8 = large
            power: Desired statistical power (1 - β)
            alpha: Significance level (Type I error rate)
            test_type: 'two-sided', 'larger', or 'smaller'

        Returns:
            Required sample size per group
        """
        analysis = TTestPower()
        n = analysis.solve_power(
            effect_size=effect_size,
            power=power,
            alpha=alpha,
            alternative=test_type
        )
        return int(np.ceil(n))

    @staticmethod
    def apply_multiple_testing_correction(
        p_values: Union[List[float], np.ndarray],
        method: str = 'fdr_bh',
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Apply multiple testing correction.

        Methods:
        - 'bonferroni': Bonferroni correction (most conservative)
        - 'fdr_bh': Benjamini-Hochberg FDR (recommended)
        - 'fdr_by': Benjamini-Yekutieli FDR
        - 'holm': Holm-Bonferroni method

        Args:
            p_values: Uncorrected p-values
            method: Correction method
            alpha: Family-wise error rate or FDR

        Returns:
            Dictionary with corrected p-values and rejection decisions
        """
        if len(p_values) == 0:
            return {'corrected_pvals': [], 'reject': [], 'method': method}

        reject, pvals_corrected, alphac_sidak, alphac_bonf = multipletests(
            p_values, alpha=alpha, method=method
        )

        return {
            'corrected_pvals': pvals_corrected.tolist(),
            'reject': reject.tolist(),
            'method': method,
            'alpha': alpha,
            'n_rejected': np.sum(reject),
            'sidak_alpha': alphac_sidak,
            'bonferroni_alpha': alphac_bonf
        }

    @staticmethod
    def bootstrap_confidence_interval(
        data: np.ndarray,
        statistic: callable,
        n_bootstrap: int = 10000,
        confidence_level: float = 0.95,
        method: str = 'percentile',
        random_state: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute bootstrap confidence intervals.

        Per Efron & Tibshirani (1993) "An Introduction to the Bootstrap"

        Args:
            data: Input data array
            statistic: Function to compute statistic of interest
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95% CI)
            method: 'percentile', 'basic', or 'bca'
            random_state: Random seed for reproducibility

        Returns:
            Dictionary with point estimate and confidence interval
        """
        if random_state is not None:
            rng = np.random.RandomState(random_state)
        else:
            rng = np.random

        # Compute point estimate
        point_estimate = statistic(data)

        # Bootstrap resampling
        res = bootstrap(
            (data,),
            statistic,
            n_resamples=n_bootstrap,
            confidence_level=confidence_level,
            method=method,
            random_state=rng
        )

        return {
            'estimate': float(point_estimate),
            'ci_low': float(res.confidence_interval.low),
            'ci_high': float(res.confidence_interval.high),
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap,
            'method': method
        }

    @staticmethod
    def test_normality(
        data: np.ndarray,
        alpha: float = 0.05
    ) -> Dict[str, Any]:
        """
        Test for normality using multiple methods.

        Args:
            data: Data to test
            alpha: Significance level

        Returns:
            Dictionary with test results
        """
        results = {}

        # Shapiro-Wilk test (best for small samples)
        if len(data) <= 5000:
            stat, p_val = stats.shapiro(data)
            results['shapiro_wilk'] = {
                'statistic': stat,
                'p_value': p_val,
                'is_normal': p_val > alpha
            }

        # Anderson-Darling test
        result = stats.anderson(data)
        results['anderson_darling'] = {
            'statistic': result.statistic,
            'critical_values': result.critical_values.tolist(),
            'significance_levels': result.significance_level.tolist(),
            'is_normal': result.statistic < result.critical_values[2]  # 5% level
        }

        # D'Agostino-Pearson test
        if len(data) >= 20:
            stat, p_val = stats.normaltest(data)
            results['dagostino_pearson'] = {
                'statistic': stat,
                'p_value': p_val,
                'is_normal': p_val > alpha
            }

        # Overall assessment
        normal_tests = [v.get('is_normal', False) for v in results.values()]
        results['overall_normal'] = sum(normal_tests) > len(normal_tests) / 2

        return results

    @staticmethod
    def robust_correlation(
        x: np.ndarray,
        y: np.ndarray,
        method: str = 'auto'
    ) -> Dict[str, float]:
        """
        Compute correlation with appropriate method based on data properties.

        Args:
            x, y: Data arrays
            method: 'auto', 'pearson', 'spearman', or 'kendall'

        Returns:
            Dictionary with correlation coefficient and p-value
        """
        if len(x) != len(y):
            raise ValueError("Arrays must have same length")

        if len(x) < 3:
            return {'correlation': np.nan, 'p_value': np.nan, 'method': 'insufficient_data'}

        # Remove NaN values
        mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[mask]
        y_clean = y[mask]

        if len(x_clean) < 3:
            return {'correlation': np.nan, 'p_value': np.nan, 'method': 'insufficient_data'}

        # Auto-select method based on normality
        if method == 'auto':
            x_normal = StatisticalUtils.test_normality(x_clean)['overall_normal']
            y_normal = StatisticalUtils.test_normality(y_clean)['overall_normal']

            if x_normal and y_normal:
                method = 'pearson'
            else:
                method = 'spearman'

        # Compute correlation
        if method == 'pearson':
            corr, p_val = stats.pearsonr(x_clean, y_clean)
        elif method == 'spearman':
            corr, p_val = stats.spearmanr(x_clean, y_clean)
        elif method == 'kendall':
            corr, p_val = stats.kendalltau(x_clean, y_clean)
        else:
            raise ValueError(f"Unknown correlation method: {method}")

        return {
            'correlation': float(corr),
            'p_value': float(p_val),
            'method': method,
            'n_samples': len(x_clean),
            'n_excluded': len(x) - len(x_clean)
        }

    @staticmethod
    def compute_effect_size(
        group1: np.ndarray,
        group2: np.ndarray,
        effect_type: str = 'cohen_d'
    ) -> Dict[str, float]:
        """
        Compute effect size measures.

        Args:
            group1, group2: Data for two groups
            effect_type: 'cohen_d', 'glass_delta', or 'hedges_g'

        Returns:
            Dictionary with effect size and interpretation
        """
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        n1, n2 = len(group1), len(group2)

        if effect_type == 'cohen_d':
            # Pooled standard deviation
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            effect_size = (mean1 - mean2) / pooled_std

        elif effect_type == 'glass_delta':
            # Use control group SD
            effect_size = (mean1 - mean2) / std2

        elif effect_type == 'hedges_g':
            # Cohen's d with bias correction
            pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
            cohen_d = (mean1 - mean2) / pooled_std
            # Hedges' correction
            correction = 1 - (3 / (4 * (n1 + n2) - 9))
            effect_size = cohen_d * correction
        else:
            raise ValueError(f"Unknown effect type: {effect_type}")

        # Interpretation (Cohen's conventions)
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            interpretation = 'negligible'
        elif abs_effect < 0.5:
            interpretation = 'small'
        elif abs_effect < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'

        return {
            'effect_size': float(effect_size),
            'effect_type': effect_type,
            'interpretation': interpretation,
            'mean_diff': float(mean1 - mean2),
            'group1_mean': float(mean1),
            'group2_mean': float(mean2),
            'group1_std': float(std1),
            'group2_std': float(std2)
        }

    @staticmethod
    def compute_wasserstein_distance(
        dist1: Union[np.ndarray, torch.Tensor],
        dist2: Union[np.ndarray, torch.Tensor],
        normalize: bool = True
    ) -> float:
        """
        Compute Wasserstein distance between distributions.

        Per Ji et al. for distribution shift measurement.

        Args:
            dist1, dist2: Distributions to compare
            normalize: Whether to normalize distributions

        Returns:
            Wasserstein distance
        """
        # Convert to numpy if needed
        if torch.is_tensor(dist1):
            dist1 = dist1.detach().cpu().numpy()
        if torch.is_tensor(dist2):
            dist2 = dist2.detach().cpu().numpy()

        # Flatten
        dist1 = dist1.flatten()
        dist2 = dist2.flatten()

        # Normalize if requested
        if normalize:
            dist1 = dist1 / (np.sum(np.abs(dist1)) + 1e-10)
            dist2 = dist2 / (np.sum(np.abs(dist2)) + 1e-10)

        return float(wasserstein_distance(dist1, dist2))

    @staticmethod
    def permutation_test(
        group1: np.ndarray,
        group2: np.ndarray,
        statistic: callable = None,
        n_permutations: int = 10000,
        random_state: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Perform permutation test for difference between groups.

        Non-parametric alternative to t-test.

        Args:
            group1, group2: Data for two groups
            statistic: Function to compute test statistic (default: mean difference)
            n_permutations: Number of permutations
            random_state: Random seed

        Returns:
            Dictionary with test results
        """
        if statistic is None:
            statistic = lambda x, y: np.mean(x) - np.mean(y)

        if random_state is not None:
            np.random.seed(random_state)

        # Observed statistic
        observed = statistic(group1, group2)

        # Combine groups
        combined = np.concatenate([group1, group2])
        n1 = len(group1)

        # Permutation distribution
        perm_stats = []
        for _ in range(n_permutations):
            np.random.shuffle(combined)
            perm_g1 = combined[:n1]
            perm_g2 = combined[n1:]
            perm_stats.append(statistic(perm_g1, perm_g2))

        perm_stats = np.array(perm_stats)

        # Two-tailed p-value
        p_value = np.mean(np.abs(perm_stats) >= np.abs(observed))

        return {
            'observed_statistic': float(observed),
            'p_value': float(p_value),
            'n_permutations': n_permutations,
            'perm_mean': float(np.mean(perm_stats)),
            'perm_std': float(np.std(perm_stats)),
            'significant': p_value < 0.05
        }

    @staticmethod
    def validate_fisher_performance_correlation(
        fisher_damages: np.ndarray,
        performance_drops: np.ndarray,
        use_bootstrap: bool = True
    ) -> Dict[str, Any]:
        """
        Validate that Fisher damage correlates with actual performance drop.

        Addresses theoretical issue in CFA-2-complete.py

        Args:
            fisher_damages: Fisher information damage scores
            performance_drops: Actual performance degradation
            use_bootstrap: Whether to compute bootstrap CI

        Returns:
            Validation results with correlation and significance
        """
        # Use robust correlation
        corr_result = StatisticalUtils.robust_correlation(
            fisher_damages, performance_drops, method='spearman'
        )

        result = {
            'correlation': corr_result['correlation'],
            'p_value': corr_result['p_value'],
            'method': corr_result['method'],
            'is_significant': corr_result['p_value'] < 0.05,
            'interpretation': 'Fisher damage predicts performance' if corr_result['p_value'] < 0.05
                            else 'No significant relationship'
        }

        # Add bootstrap confidence interval if requested
        if use_bootstrap and len(fisher_damages) > 20:
            def corr_stat(indices):
                indices = indices.astype(int)
                return stats.spearmanr(
                    fisher_damages[indices],
                    performance_drops[indices]
                )[0]

            boot_result = StatisticalUtils.bootstrap_confidence_interval(
                np.arange(len(fisher_damages)),
                corr_stat,
                n_bootstrap=5000
            )
            result['bootstrap_ci'] = [boot_result['ci_low'], boot_result['ci_high']]

        return result


class HypothesisValidator:
    """Validate catastrophic forgetting hypotheses with proper statistics."""

    def __init__(self, stats_utils: Optional[StatisticalUtils] = None):
        """Initialize with statistical utilities."""
        self.stats = stats_utils or StatisticalUtils()

    def validate_all_hypotheses(
        self,
        data: Dict[str, Any],
        alpha: float = 0.05
    ) -> Dict[str, Dict]:
        """
        Validate all catastrophic forgetting hypotheses.

        Returns results with multiple testing correction applied.
        """
        results = {}
        p_values = []

        # H1: Fisher damage correlates with performance
        if 'fisher_damages' in data and 'performance_drops' in data:
            h1_result = self.stats.validate_fisher_performance_correlation(
                data['fisher_damages'],
                data['performance_drops']
            )
            results['h1_fisher_performance'] = h1_result
            p_values.append(h1_result['p_value'])

        # H2: Attention entropy drops during forgetting
        if 'baseline_entropy' in data and 'crisis_entropy' in data:
            h2_result = self.stats.permutation_test(
                data['baseline_entropy'],
                data['crisis_entropy']
            )
            results['h2_attention_entropy'] = h2_result
            p_values.append(h2_result['p_value'])

        # H3: Dead neurons increase
        if 'baseline_dead' in data and 'crisis_dead' in data:
            h3_result = self.stats.permutation_test(
                data['baseline_dead'],
                data['crisis_dead']
            )
            results['h3_dead_neurons'] = h3_result
            p_values.append(h3_result['p_value'])

        # Apply multiple testing correction
        if p_values:
            correction = self.stats.apply_multiple_testing_correction(
                p_values, method='fdr_bh', alpha=alpha
            )

            # Update results with corrected p-values
            for i, key in enumerate(results.keys()):
                results[key]['p_value_corrected'] = correction['corrected_pvals'][i]
                results[key]['reject_null'] = correction['reject'][i]

        return results