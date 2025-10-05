"""
Unified Welford's algorithm implementation for numerically stable statistics.

This is THE ONLY Welford implementation in the codebase.
All other code should import and use this.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Union, Tuple


class WelfordAccumulator:
    """
    Unified Welford's algorithm for numerically stable mean/variance computation.

    Supports:
    - Standard (unweighted) accumulation
    - Weighted accumulation
    - Keyed accumulation (multiple accumulators in one object)
    - Both tensor and scalar inputs

    This replaces ALL other Welford implementations in the codebase.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        use_keys: bool = False,
        weighted: bool = False
    ):
        """
        Initialize Welford accumulator.

        Args:
            device: Device for tensors (default: CPU)
            dtype: Data type (default: float32)
            use_keys: If True, support multiple named accumulators
            weighted: If True, support weighted samples
        """
        self.device = device or torch.device('cpu')
        self.dtype = dtype or torch.float32
        self.use_keys = use_keys
        self.weighted = weighted

        if use_keys:
            self.states: Dict[str, Dict[str, Any]] = {}
        else:
            self.state = self._init_state()

    def _init_state(self) -> Dict[str, Any]:
        """Initialize a single accumulator state."""
        return {
            'count': 0,
            'weight_sum': 0.0 if self.weighted else None,
            'weight_sum_sq': 0.0 if self.weighted else None,  # For effective sample size
            'mean': None,
            'M2': None  # Sum of squared deviations
        }

    def update(
        self,
        value: Union[torch.Tensor, float],
        weight: float = 1.0,
        key: Optional[str] = None
    ) -> None:
        """
        Update statistics with new sample(s).

        Args:
            value: New sample value (tensor or scalar)
            weight: Weight for this sample (only used if weighted=True)
            key: Key for this accumulator (only used if use_keys=True)
        """
        # Handle keyed accumulation
        if self.use_keys:
            if key is None:
                raise ValueError("Key required when use_keys=True")
            if key not in self.states:
                self.states[key] = self._init_state()
            state = self.states[key]
        else:
            if key is not None:
                raise ValueError("Key not allowed when use_keys=False")
            state = self.state

        # Convert to tensor if needed
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, device=self.device, dtype=self.dtype)
        else:
            value = value.to(device=self.device, dtype=self.dtype)

        # Handle batch dimension
        if value.dim() > 1:
            # Process batch of samples
            batch_size = value.shape[0]
            for i in range(batch_size):
                self._update_single(state, value[i], weight)
        else:
            # Single sample
            self._update_single(state, value, weight)

    def _update_single(
        self,
        state: Dict[str, Any],
        value: torch.Tensor,
        weight: float
    ) -> None:
        """Update state with a single sample."""
        if self.weighted:
            # Weighted Welford's algorithm
            if state['mean'] is None:
                state['mean'] = value.clone()
                state['M2'] = torch.zeros_like(value)
                state['weight_sum'] = weight
                state['weight_sum_sq'] = weight * weight
                state['count'] = 1
            else:
                weight_sum_new = state['weight_sum'] + weight
                if weight_sum_new > 0:
                    delta = value - state['mean']
                    R = delta * weight / weight_sum_new
                    state['mean'] = state['mean'] + R
                    state['M2'] = state['M2'] + weight * delta * (value - state['mean'])
                    state['weight_sum'] = weight_sum_new
                    state['weight_sum_sq'] = state['weight_sum_sq'] + weight * weight
                    state['count'] += 1
        else:
            # Standard Welford's algorithm
            state['count'] += 1
            if state['mean'] is None:
                state['mean'] = value.clone()
                state['M2'] = torch.zeros_like(value)
            else:
                delta = value - state['mean']
                state['mean'] = state['mean'] + delta / state['count']
                delta2 = value - state['mean']
                state['M2'] = state['M2'] + delta * delta2

    def get_statistics(
        self,
        key: Optional[str] = None
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        Get current statistics.

        Args:
            key: Key for accumulator (only for keyed mode)

        Returns:
            Dictionary with 'mean', 'variance', 'std', 'count'/'weight_sum'
        """
        # Get appropriate state
        if self.use_keys:
            if key is None:
                raise ValueError("Key required when use_keys=True")
            if key not in self.states:
                return {'mean': None, 'variance': None, 'std': None}
            state = self.states[key]
        else:
            state = self.state

        # Calculate statistics
        if state['mean'] is None:
            return {'mean': None, 'variance': None, 'std': None}

        result = {'mean': state['mean'].clone()}

        if self.weighted:
            if state['count'] > 1 and state['weight_sum'] > 0:
                # Weighted variance with Bessel's correction using effective sample size
                # ESS = (sum of weights)^2 / sum of squared weights
                ess = (state['weight_sum'] ** 2) / state['weight_sum_sq']
                if ess > 1:
                    # Unbiased weighted variance
                    variance = state['M2'] / (state['weight_sum'] * (1 - 1/ess))
                else:
                    variance = torch.zeros_like(state['mean'])
            else:
                variance = torch.zeros_like(state['mean'])
            result['weight_sum'] = state['weight_sum']
            result['count'] = state['count']
        else:
            if state['count'] > 1:
                # Sample variance
                variance = state['M2'] / (state['count'] - 1)
            else:
                variance = torch.zeros_like(state['mean'])
            result['count'] = state['count']

        result['variance'] = variance
        result['std'] = torch.sqrt(variance + 1e-10)

        return result

    def get_mean(self, key: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get current mean estimate."""
        stats = self.get_statistics(key)
        return stats['mean']

    def get_variance(self, key: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get current variance estimate."""
        stats = self.get_statistics(key)
        return stats['variance']

    def get_std(self, key: Optional[str] = None) -> Optional[torch.Tensor]:
        """Get current standard deviation estimate."""
        stats = self.get_statistics(key)
        return stats['std']

    def batch_update(
        self,
        batch_mean: torch.Tensor,
        batch_variance: torch.Tensor,
        batch_count: int,
        key: Optional[str] = None
    ) -> None:
        """
        Update statistics with pre-computed batch statistics.

        This is useful when you have already computed mean/variance for a batch
        and want to combine it with the running statistics.

        Args:
            batch_mean: Mean of the batch
            batch_variance: Variance of the batch
            batch_count: Number of samples in the batch
            key: Key for accumulator (only for keyed mode)
        """
        # Get appropriate state
        if self.use_keys:
            if key is None:
                raise ValueError("Key required when use_keys=True")
            if key not in self.states:
                self.states[key] = self._init_state()
            state = self.states[key]
        else:
            state = self.state

        batch_mean = batch_mean.to(device=self.device, dtype=self.dtype)
        batch_variance = batch_variance.to(device=self.device, dtype=self.dtype)

        if state['mean'] is None:
            # First batch
            state['mean'] = batch_mean.clone()
            state['M2'] = batch_variance * (batch_count - 1)
            state['count'] = batch_count
        else:
            # Combine statistics using parallel algorithm
            total_count = state['count'] + batch_count
            delta = batch_mean - state['mean']

            # Update mean
            new_mean = state['mean'] + delta * batch_count / total_count

            # Update M2 (sum of squared deviations)
            new_M2 = state['M2'] + batch_variance * (batch_count - 1) + \
                     delta * delta * state['count'] * batch_count / total_count

            state['mean'] = new_mean
            state['M2'] = new_M2
            state['count'] = total_count

    def reset(self, key: Optional[str] = None) -> None:
        """Reset accumulator state."""
        if self.use_keys:
            if key is None:
                # Reset all keys
                self.states = {}
            elif key in self.states:
                self.states[key] = self._init_state()
        else:
            self.state = self._init_state()