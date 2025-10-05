"""
Batch processor integration utilities for proper multi-batch handling.

This module provides utilities to properly process multiple batches using batch_processor,
ensuring theoretical correctness and statistical validity.
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class MultiBatchProcessor:
    """
    Extended batch processor that handles multiple batches correctly.

    Key features:
    1. Process ALL batches, not just batches[0]
    2. Use theoretically correct accumulation methods
    3. Integrate with existing batch_processor.py
    """

    def __init__(self, batch_processor):
        """Initialize with existing batch processor."""
        self.batch_processor = batch_processor

    def process_all_batches(
        self,
        batches: List[Dict[str, torch.Tensor]],
        compute_fn: Callable,
        accumulation_method: str = 'mean',
        **kwargs
    ) -> Any:
        """
        Process ALL batches with proper accumulation.

        Args:
            batches: List of all available batches
            compute_fn: Function to compute on each batch
            accumulation_method: How to combine results:
                - 'mean': Simple average (valid for expectations)
                - 'weighted_mean': Weighted by batch size
                - 'sum': Sum (for counts)
                - 'concat': Concatenate (for raw data)
                - 'fisher': Fisher information accumulation
                - 'fisher_combine_pvalues': Combine p-values properly
                - 'max_entropy': Global entropy (not average)
                - 'gradient_correlation': Full correlation matrix

        Returns:
            Accumulated result across all batches
        """
        if not batches:
            return None

        # Process each batch
        results = []
        batch_sizes = []

        for i, batch in enumerate(batches):
            logger.debug(f"Processing batch {i+1}/{len(batches)}")

            # Use batch_processor for memory-efficient processing
            result = self.batch_processor.process_batch(
                batch=batch,
                compute_fn=compute_fn,
                reduction='none',  # We'll do our own reduction
                **kwargs
            )

            results.append(result)
            batch_sizes.append(batch['input_ids'].shape[0])

        # Apply theoretically correct accumulation
        return self._accumulate_results(results, batch_sizes, accumulation_method)

    def _accumulate_results(
        self,
        results: List[Any],
        batch_sizes: List[int],
        method: str
    ) -> Any:
        """Apply theoretically correct accumulation method."""

        if not results:
            return None

        if method == 'mean':
            # Simple average - valid for expectations
            return self._simple_mean(results)

        elif method == 'weighted_mean':
            # Weighted average by batch size
            return self._weighted_mean(results, batch_sizes)

        elif method == 'sum':
            # Sum - for counts
            return self._sum_results(results)

        elif method == 'concat':
            # Concatenate raw data
            return self._concat_results(results)

        elif method == 'fisher':
            # Fisher information accumulation
            return self._accumulate_fisher(results, batch_sizes)

        elif method == 'fisher_combine_pvalues':
            # Properly combine p-values using Fisher's method
            return self._combine_pvalues_fisher(results)

        elif method == 'max_entropy':
            # Global entropy over all data
            return self._global_entropy(results)

        elif method == 'gradient_correlation':
            # Full correlation matrix over all gradients
            return self._full_gradient_correlation(results)

        else:
            logger.warning(f"Unknown accumulation method: {method}, using mean")
            return self._simple_mean(results)

    def _simple_mean(self, results: List[Any]) -> Any:
        """Simple average - valid for expectations."""
        if isinstance(results[0], dict):
            # Average each key
            averaged = {}
            for key in results[0].keys():
                values = [r[key] for r in results if key in r]
                if values:
                    if torch.is_tensor(values[0]):
                        averaged[key] = torch.stack(values).mean(0)
                    elif isinstance(values[0], (int, float)):
                        averaged[key] = np.mean(values)
                    else:
                        averaged[key] = values[0]  # Can't average
            return averaged
        elif torch.is_tensor(results[0]):
            return torch.stack(results).mean(0)
        else:
            return np.mean(results)

    def _weighted_mean(self, results: List[Any], weights: List[int]) -> Any:
        """Weighted average by batch size."""
        total_weight = sum(weights)
        if total_weight == 0:
            return self._simple_mean(results)

        if isinstance(results[0], dict):
            weighted = {}
            for key in results[0].keys():
                values = []
                value_weights = []
                for r, w in zip(results, weights):
                    if key in r:
                        values.append(r[key])
                        value_weights.append(w)

                if values:
                    if torch.is_tensor(values[0]):
                        weighted_sum = sum(v * w for v, w in zip(values, value_weights))
                        weighted[key] = weighted_sum / sum(value_weights)
                    elif isinstance(values[0], (int, float)):
                        weighted_sum = sum(v * w for v, w in zip(values, value_weights))
                        weighted[key] = weighted_sum / sum(value_weights)
                    else:
                        weighted[key] = values[0]
            return weighted
        else:
            weighted_sum = sum(r * w for r, w in zip(results, weights))
            return weighted_sum / total_weight

    def _sum_results(self, results: List[Any]) -> Any:
        """Sum results - for counts."""
        if isinstance(results[0], dict):
            summed = {}
            for key in results[0].keys():
                values = [r[key] for r in results if key in r]
                if values:
                    if torch.is_tensor(values[0]):
                        summed[key] = sum(values)
                    elif isinstance(values[0], (int, float)):
                        summed[key] = sum(values)
                    else:
                        summed[key] = values[0]
            return summed
        else:
            return sum(results)

    def _concat_results(self, results: List[Any]) -> Any:
        """Concatenate results - for raw data."""
        if isinstance(results[0], dict):
            concatenated = {}
            for key in results[0].keys():
                values = [r[key] for r in results if key in r]
                if values:
                    if torch.is_tensor(values[0]):
                        concatenated[key] = torch.cat(values, dim=0)
                    elif isinstance(values[0], np.ndarray):
                        concatenated[key] = np.concatenate(values, axis=0)
                    elif isinstance(values[0], list):
                        concatenated[key] = sum(values, [])
                    else:
                        concatenated[key] = values
            return concatenated
        elif torch.is_tensor(results[0]):
            return torch.cat(results, dim=0)
        elif isinstance(results[0], np.ndarray):
            return np.concatenate(results, axis=0)
        else:
            return results

    def _accumulate_fisher(self, results: List[Dict], batch_sizes: List[int]) -> Dict:
        """
        Properly accumulate Fisher information matrices.

        Fisher information is additive: F_total = Σ F_i
        But we need to weight by sample sizes for proper estimation.
        """
        if not results:
            return {}

        # Initialize accumulator
        accumulated = {}
        total_samples = sum(batch_sizes)

        for key in results[0].keys():
            if 'fisher' in key.lower():
                # Fisher matrices should be summed (weighted by sample proportion)
                fisher_values = []
                for r, n in zip(results, batch_sizes):
                    if key in r:
                        # Weight by proportion of total samples
                        weight = n / total_samples
                        fisher_values.append(r[key] * weight)

                if fisher_values:
                    if torch.is_tensor(fisher_values[0]):
                        accumulated[key] = sum(fisher_values)
                    else:
                        accumulated[key] = np.sum(fisher_values, axis=0)
            else:
                # Non-Fisher values - use weighted mean
                values = [r[key] for r in results if key in r]
                weights = [batch_sizes[i] for i, r in enumerate(results) if key in r]

                if values:
                    if torch.is_tensor(values[0]):
                        weighted_sum = sum(v * w for v, w in zip(values, weights))
                        accumulated[key] = weighted_sum / sum(weights)
                    else:
                        accumulated[key] = np.average(values, weights=weights)

        return accumulated

    def _combine_pvalues_fisher(self, results: List[Dict]) -> Dict:
        """
        Combine p-values using Fisher's method.

        NEVER average p-values! Use proper combination methods:
        - Fisher's method: χ² = -2 Σ ln(p_i)
        - Stouffer's method: Z = Σ Φ⁻¹(1-p_i) / √n
        """
        combined = {}

        for key in results[0].keys():
            if 'p_value' in key.lower() or 'pvalue' in key.lower():
                # Collect p-values
                pvalues = []
                for r in results:
                    if key in r:
                        val = r[key]
                        if torch.is_tensor(val):
                            val = val.cpu().numpy()
                        if isinstance(val, np.ndarray):
                            pvalues.extend(val.flatten())
                        else:
                            pvalues.append(val)

                if pvalues:
                    # Remove invalid p-values
                    pvalues = [p for p in pvalues if 0 < p <= 1]

                    if len(pvalues) >= 2:
                        # Fisher's method
                        chi2_stat = -2 * np.sum(np.log(pvalues))
                        df = 2 * len(pvalues)
                        combined_pvalue = 1 - stats.chi2.cdf(chi2_stat, df)

                        combined[key] = combined_pvalue
                        combined[f"{key}_fisher_chi2"] = chi2_stat
                        combined[f"{key}_n_tests"] = len(pvalues)
                    elif len(pvalues) == 1:
                        combined[key] = pvalues[0]
            else:
                # Non-p-value: use simple mean
                values = [r[key] for r in results if key in r]
                if values:
                    if torch.is_tensor(values[0]):
                        combined[key] = torch.stack(values).mean(0)
                    elif isinstance(values[0], (int, float)):
                        combined[key] = np.mean(values)
                    else:
                        combined[key] = values[0]

        return combined

    def _global_entropy(self, results: List[Dict]) -> Dict:
        """
        Compute global entropy over all attention patterns.

        Entropy should be computed over the full distribution,
        not averaged across batch-wise entropies.
        """
        # Collect all attention patterns
        all_attention_patterns = []

        for r in results:
            if 'attention_weights' in r:
                patterns = r['attention_weights']
                if torch.is_tensor(patterns):
                    all_attention_patterns.append(patterns)

        if not all_attention_patterns:
            return {'error': 'No attention patterns found'}

        # Concatenate all patterns
        global_patterns = torch.cat(all_attention_patterns, dim=0)

        # Compute global entropy
        # H(X) = -Σ p(x) log p(x)
        eps = 1e-10

        # Normalize to get probabilities
        if global_patterns.dim() == 4:  # [batch, heads, seq, seq]
            # Compute entropy per head
            probs = global_patterns.softmax(dim=-1)
            entropy = -(probs * torch.log(probs + eps)).sum(dim=-1)

            # Average over sequences and batches, but not heads
            global_entropy_per_head = entropy.mean(dim=[0, 2])  # [heads]
            global_entropy = entropy.mean()

            return {
                'global_entropy': global_entropy.item(),
                'global_entropy_per_head': global_entropy_per_head.cpu().numpy(),
                'n_samples': global_patterns.shape[0],
                'theoretical_max_entropy': np.log(global_patterns.shape[-1])
            }
        else:
            return {'error': f'Unexpected attention shape: {global_patterns.shape}'}

    def _full_gradient_correlation(self, results: List[Dict]) -> Dict:
        """
        Compute gradient correlation over ALL samples.

        Gradient pathology detection requires seeing correlations
        across the full dataset, not batch-wise correlations.
        """
        # Collect all gradients
        all_gradients = []

        for r in results:
            if 'gradients' in r:
                grads = r['gradients']
                if isinstance(grads, dict):
                    # Flatten and concatenate parameter gradients
                    flat_grads = []
                    for name in sorted(grads.keys()):
                        g = grads[name]
                        if torch.is_tensor(g):
                            flat_grads.append(g.flatten())
                    if flat_grads:
                        all_gradients.append(torch.cat(flat_grads))
                elif torch.is_tensor(grads):
                    all_gradients.append(grads.flatten())

        if len(all_gradients) < 2:
            return {'error': 'Need at least 2 gradient samples for correlation'}

        # Stack all gradients
        gradient_matrix = torch.stack(all_gradients)  # [n_samples, n_params]

        # Compute correlation matrix
        # Correlation = Cov(X,Y) / (σ_X * σ_Y)
        mean_grads = gradient_matrix.mean(dim=0, keepdim=True)
        centered = gradient_matrix - mean_grads

        # Compute covariance
        n_samples = gradient_matrix.shape[0]
        cov_matrix = torch.mm(centered.T, centered) / (n_samples - 1)

        # Get standard deviations
        std_devs = centered.std(dim=0)
        std_outer = torch.outer(std_devs, std_devs)

        # Correlation matrix
        correlation = cov_matrix / (std_outer + 1e-10)

        # Analyze pathology indicators
        eigenvalues = torch.linalg.eigvalsh(correlation)
        condition_number = eigenvalues.max() / (eigenvalues.min() + 1e-10)

        # Check for high correlation (pathology indicator)
        high_correlation_threshold = 0.9
        off_diagonal = correlation - torch.eye(correlation.shape[0], device=correlation.device)
        high_corr_pairs = (off_diagonal.abs() > high_correlation_threshold).sum() // 2

        return {
            'gradient_correlation_matrix': correlation.cpu().numpy(),
            'condition_number': condition_number.item(),
            'max_correlation': off_diagonal.abs().max().item(),
            'high_correlation_pairs': high_corr_pairs.item(),
            'n_gradient_samples': n_samples,
            'gradient_rank': torch.linalg.matrix_rank(correlation).item(),
            'pathology_detected': condition_number > 1e6 or high_corr_pairs > 0
        }


def create_proper_batch_processor(batch_processor):
    """Create a multi-batch processor from existing batch processor."""
    return MultiBatchProcessor(batch_processor)


# Example usage patterns for different metrics
METRIC_ACCUMULATION_METHODS = {
    # These can be averaged
    'loss': 'weighted_mean',
    'accuracy': 'weighted_mean',
    'perplexity': 'weighted_mean',

    # These need special handling
    'fisher_information': 'fisher',
    'p_values': 'fisher_combine_pvalues',
    'attention_entropy': 'max_entropy',
    'gradient_pathology': 'gradient_correlation',

    # These should be summed
    'dead_neuron_count': 'sum',
    'total_samples': 'sum',

    # These should be concatenated
    'all_gradients': 'concat',
    'all_activations': 'concat',
    'all_hidden_states': 'concat',
}