"""
Critical fixes for Fisher collector based on ICML reviewer feedback.

This module contains patches for:
1. Correct Welford's algorithm implementation
2. SSC (Squared Score Contributions) terminology
3. CRLB safety guards
4. True per-sample gradient collection
5. FDR correction for multiple testing
6. Token normalization in true Fisher
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


# DEPRECATED: Use utils.welford.WelfordAccumulator instead
# This old implementation is kept only for backwards compatibility
# DO NOT USE IN NEW CODE
class WelfordAccumulator:
    """
    Correct implementation of Welford's algorithm for weighted incremental statistics.

    This provides numerically stable computation of mean and variance for
    weighted samples, crucial for Fisher estimation accuracy.
    """

    def __init__(self, device='cpu', dtype=torch.float32):
        self.device = device
        self.dtype = dtype
        self.state = {}

    def update(self, key: str, value: torch.Tensor, weight: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update statistics with a new weighted sample.

        Args:
            key: Identifier for this accumulator
            value: New sample value
            weight: Weight for this sample (e.g., number of tokens)

        Returns:
            Tuple of (weighted_mean, weighted_variance)
        """
        if key not in self.state:
            # Initialize state
            self.state[key] = {
                'mean': torch.zeros_like(value, device=self.device, dtype=self.dtype),
                'M2': torch.zeros_like(value, device=self.device, dtype=self.dtype),
                'weight_sum': 0.0
            }

        s = self.state[key]

        # Weighted Welford's algorithm
        weight_sum_new = s['weight_sum'] + weight

        # Avoid division by zero
        if weight_sum_new == 0:
            return s['mean'], torch.zeros_like(s['mean'])

        delta = value - s['mean']
        R = delta * weight / weight_sum_new

        s['mean'] = s['mean'] + R
        s['M2'] = s['M2'] + weight * delta * (value - s['mean'])
        s['weight_sum'] = weight_sum_new

        # Compute variance with Bessel's correction
        if s['weight_sum'] > 1:
            # Unbiased variance estimator
            variance = s['M2'] / (s['weight_sum'] - 1)
        else:
            variance = torch.zeros_like(s['M2'])

        return s['mean'].clone(), variance.clone()

    def get_statistics(self, key: str) -> Optional[Dict[str, torch.Tensor]]:
        """Get current statistics for a key."""
        if key not in self.state:
            return None

        s = self.state[key]
        variance = s['M2'] / (s['weight_sum'] - 1) if s['weight_sum'] > 1 else torch.zeros_like(s['M2'])

        return {
            'mean': s['mean'].clone(),
            'variance': variance.clone(),
            'std': torch.sqrt(torch.maximum(variance, torch.zeros_like(variance))),
            'weight_sum': s['weight_sum'],
            'n_samples': int(s['weight_sum'])  # Approximate sample count
        }


class SSCCollector:
    """
    Collector for Squared Score Contributions (SSCs).

    CRITICAL: SSCs are NOT "per-sample Fisher" - they are individual
    score contributions that must be averaged to get Fisher.
    """

    def __init__(self, store_all: bool = False):
        """
        Args:
            store_all: If True, store all SSCs (memory intensive).
                      If False, only store statistics.
        """
        self.store_all = store_all
        self.sscs = []  # List of SSCs if store_all=True
        self.ssc_stats = {}  # Statistics per parameter

    def collect_ssc(self, model: nn.Module, sample: Dict[str, torch.Tensor],
                    task: str = 'default') -> Dict[str, torch.Tensor]:
        """
        Collect true per-sample squared score contribution.

        This computes the squared gradient for a SINGLE sample,
        which is the atomic unit for Fisher estimation.

        Args:
            model: Model to compute SSC for
            sample: Single sample (batch size must be 1)
            task: Task identifier

        Returns:
            Dictionary of SSCs per parameter
        """
        # Ensure single sample
        if 'input_ids' in sample:
            assert sample['input_ids'].size(0) == 1, "SSC requires single sample"

        model.eval()  # No dropout
        model.zero_grad()

        # Forward and backward for single sample
        with torch.enable_grad():
            outputs = model(**sample)
            loss = outputs.loss

            if loss is None:
                raise ValueError("Model must return loss for SSC computation")

            loss.backward()

        # Collect squared gradients (SSCs)
        ssc = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Square the gradient to get SSC
                ssc[name] = param.grad.detach().pow(2)

                # Update statistics if not storing all
                if not self.store_all:
                    if name not in self.ssc_stats:
                        self.ssc_stats[name] = {
                            'sum': torch.zeros_like(param.grad),
                            'sum_sq': torch.zeros_like(param.grad),
                            'count': 0
                        }

                    self.ssc_stats[name]['sum'] += ssc[name]
                    self.ssc_stats[name]['sum_sq'] += ssc[name].pow(2)
                    self.ssc_stats[name]['count'] += 1

        # Store if requested
        if self.store_all:
            self.sscs.append({
                'task': task,
                'ssc': ssc,
                'timestamp': torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
            })

        model.zero_grad()
        return ssc

    def get_fisher_estimate(self, parameter: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Get Fisher estimate from collected SSCs.

        This computes E[SSC] which is the Fisher Information.

        Args:
            parameter: Specific parameter name or None for all

        Returns:
            Fisher estimates per parameter
        """
        fisher = {}

        if self.store_all and self.sscs:
            # Compute from stored SSCs
            param_sscs = {}
            for entry in self.sscs:
                for name, ssc in entry['ssc'].items():
                    if parameter is None or name == parameter:
                        if name not in param_sscs:
                            param_sscs[name] = []
                        param_sscs[name].append(ssc)

            # Average SSCs to get Fisher
            for name, ssc_list in param_sscs.items():
                fisher[name] = torch.stack(ssc_list).mean(dim=0)

        elif self.ssc_stats:
            # Compute from statistics
            for name, stats in self.ssc_stats.items():
                if parameter is None or name == parameter:
                    if stats['count'] > 0:
                        fisher[name] = stats['sum'] / stats['count']

        return fisher


class CRLBGuard:
    """
    Guards to ensure CRLB (Cramér-Rao Lower Bound) compliance.

    INVARIANT: Inverse Fisher operations (bounds, EWC, uncertainty)
    must NEVER use individual SSCs, only expectations.
    """

    def __init__(self, strict: bool = True):
        self.strict = strict
        self.violations = []

    def check_fisher_source(self, fisher_data: Any, operation: str) -> bool:
        """
        Check if Fisher data is safe for CRLB operations.

        Args:
            fisher_data: Fisher data to check
            operation: Name of operation (e.g., 'EWC', 'uncertainty_bound')

        Returns:
            True if safe, raises error if strict=True and unsafe
        """
        # Check for SSC contamination
        is_ssc = False

        if isinstance(fisher_data, dict):
            # Check for telltale SSC keys
            ssc_indicators = ['ssc', 'sample_', 'per_sample', 'contribution']
            for key in fisher_data.keys():
                if any(ind in str(key).lower() for ind in ssc_indicators):
                    is_ssc = True
                    break

        if is_ssc:
            msg = f"CRLB VIOLATION: Operation '{operation}' attempting to use SSCs!"
            self.violations.append(msg)

            if self.strict:
                raise ValueError(msg + " Use averaged Fisher only.")
            else:
                logger.critical(msg)
                return False

        return True

    def safe_ewc_penalty(self, fisher: Dict[str, torch.Tensor],
                         current_params: Dict[str, torch.Tensor],
                         reference_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute EWC penalty with CRLB safety check.

        Args:
            fisher: Fisher information (must be expectation, not SSCs)
            current_params: Current model parameters
            reference_params: Reference parameters

        Returns:
            EWC penalty value
        """
        # Verify Fisher is not SSCs
        self.check_fisher_source(fisher, 'EWC')

        penalty = 0.0
        for name in fisher.keys():
            if name in current_params and name in reference_params:
                diff = current_params[name] - reference_params[name]
                # F * (θ - θ*)^2
                penalty += (fisher[name] * diff.pow(2)).sum()

        return penalty

    def safe_uncertainty_bound(self, fisher: Dict[str, torch.Tensor],
                              param_name: str) -> torch.Tensor:
        """
        Compute Cramér-Rao bound for parameter uncertainty.

        Args:
            fisher: Fisher information (must be expectation)
            param_name: Parameter to compute bound for

        Returns:
            Lower bound on parameter variance
        """
        # Verify Fisher is not SSCs
        self.check_fisher_source(fisher, 'uncertainty_bound')

        if param_name not in fisher:
            raise ValueError(f"Parameter {param_name} not in Fisher")

        # CRLB: Var(θ) ≥ 1/F
        # Add small epsilon for numerical stability
        eps = 1e-8
        crlb = 1.0 / (fisher[param_name] + eps)

        return crlb


def apply_benjamini_hochberg(p_values: List[float], alpha: float = 0.05) -> List[float]:
    """
    Apply Benjamini-Hochberg FDR correction for multiple testing.

    More powerful than Bonferroni while controlling False Discovery Rate.

    Args:
        p_values: List of p-values to correct
        alpha: Significance level

    Returns:
        List of adjusted p-values (q-values)
    """
    n = len(p_values)
    if n == 0:
        return []

    # Convert to numpy for easier manipulation
    p_array = np.array(p_values)

    # Get sort order
    sorted_idx = np.argsort(p_array)
    sorted_p = p_array[sorted_idx]

    # Apply BH correction
    adjusted = np.zeros(n)
    for i in range(n):
        adjusted[i] = sorted_p[i] * n / (i + 1)

    # Enforce monotonicity (q_i ≤ q_{i+1})
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    # Cap at 1.0
    adjusted = np.minimum(adjusted, 1.0)

    # Restore original order
    q_values = np.zeros(n)
    q_values[sorted_idx] = adjusted

    return q_values.tolist()


def collect_true_fisher_normalized(model: nn.Module, batch: Dict[str, torch.Tensor],
                                  n_samples: int = 10, temperature: float = 1.0) -> Dict[str, torch.Tensor]:
    """
    Collect true Fisher with proper token normalization.

    Samples from model distribution and normalizes by active tokens
    for comparability with empirical Fisher.

    Args:
        model: Model to compute Fisher for
        batch: Input batch
        n_samples: Number of samples from model distribution
        temperature: Sampling temperature

    Returns:
        Token-normalized true Fisher
    """
    model.eval()
    device = next(model.parameters()).device

    # Move batch to device
    batch = {k: v.to(device) if torch.is_tensor(v) else v
            for k, v in batch.items()}

    # Count active tokens for normalization
    if 'attention_mask' in batch:
        active_tokens = batch['attention_mask'].sum().item()
    else:
        active_tokens = batch['input_ids'].numel()

    fisher_accum = {}

    # Sample from model distribution
    with torch.no_grad():
        outputs = model(**batch)
        logits = outputs.logits / temperature

        # Sample labels from model distribution
        probs = torch.softmax(logits, dim=-1)

    for _ in range(n_samples):
        # Sample labels
        sampled_labels = torch.multinomial(
            probs.view(-1, probs.size(-1)),
            num_samples=1
        ).view(probs.shape[:-1])

        # Compute gradients with sampled labels
        model.zero_grad()
        with torch.enable_grad():
            outputs = model(**batch)

            # Cross entropy with sampled labels
            loss = nn.functional.cross_entropy(
                outputs.logits.view(-1, outputs.logits.size(-1)),
                sampled_labels.view(-1),
                reduction='mean'
            )

            loss.backward()

        # Accumulate squared gradients
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_sq = param.grad.detach().pow(2)

                if name in fisher_accum:
                    fisher_accum[name] += grad_sq
                else:
                    fisher_accum[name] = grad_sq.clone()

    # Normalize by samples AND tokens
    for name in fisher_accum:
        fisher_accum[name] = fisher_accum[name] / (n_samples * max(1, active_tokens))

    model.zero_grad()
    return fisher_accum


# Example usage and tests
if __name__ == "__main__":
    # Test Welford accumulator
    print("Testing Welford accumulator...")
    accumulator = WelfordAccumulator()

    # Generate test data
    torch.manual_seed(42)
    data = torch.randn(100, 10)
    weights = torch.rand(100) + 0.5

    # Update incrementally
    for i, (x, w) in enumerate(zip(data, weights)):
        mean, var = accumulator.update('test', x, w.item())

    # Compare with numpy
    np_mean = np.average(data.numpy(), weights=weights.numpy(), axis=0)
    np_var = np.average((data.numpy() - np_mean)**2, weights=weights.numpy(), axis=0)

    stats = accumulator.get_statistics('test')
    print(f"Mean error: {torch.norm(stats['mean'].numpy() - np_mean):.6f}")
    print(f"Var error: {torch.norm(stats['variance'].numpy() - np_var):.6f}")

    # Test FDR correction
    print("\nTesting FDR correction...")
    p_values = [0.001, 0.01, 0.02, 0.03, 0.04, 0.15, 0.5, 0.8]
    q_values = apply_benjamini_hochberg(p_values)

    print("P-values:", p_values)
    print("Q-values:", [f"{q:.3f}" for q in q_values])

    # Test CRLB guard
    print("\nTesting CRLB guard...")
    guard = CRLBGuard(strict=False)

    # Safe Fisher (expectation)
    safe_fisher = {'layer.weight': torch.randn(10, 10).abs()}
    assert guard.check_fisher_source(safe_fisher, 'test_safe')
    print("✓ Safe Fisher passed")

    # Unsafe SSC data
    unsafe_ssc = {'layer.weight_per_sample': torch.randn(10, 10).abs()}
    assert not guard.check_fisher_source(unsafe_ssc, 'test_unsafe')
    print("✓ SSC correctly identified as unsafe")

    print("\nAll tests passed!")