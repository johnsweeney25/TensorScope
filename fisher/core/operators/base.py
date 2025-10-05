"""
Abstract base classes for operators with multi-batch support.

This module provides the foundation for all operator implementations,
ensuring consistent interfaces and proper separation of concerns.

Theory:
    For any operator O computing some function of the model and data,
    we want to estimate E_x[O(θ, x)] where x ~ data distribution.

    Single batch: O(θ, x_batch) - high variance
    Multi-batch: (1/n) Σ O(θ, x_i) - variance reduced by factor n

Author: Multi-batch architecture team
Date: 2024
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class OperatorConfig:
    """Configuration for operators."""
    dtype_compute: torch.dtype = torch.float32  # Computation precision
    dtype_storage: torch.dtype = torch.float32  # Storage precision
    device: Optional[torch.device] = None
    memory_efficient: bool = True  # Use memory optimizations
    numerical_stable: bool = True  # Use numerical stability tricks
    clear_cache_freq: int = 5  # Clear GPU cache every N operations


class BatchAwareOperator(ABC):
    """
    Abstract base class for operators that can work with single or multiple batches.

    This class defines the interface that all operators must implement,
    ensuring consistent behavior across different operator types.
    """

    def __init__(self, model: nn.Module, config: Optional[OperatorConfig] = None):
        """
        Initialize operator.

        Args:
            model: Neural network model
            config: Operator configuration
        """
        self.model = model
        self.config = config or OperatorConfig()
        self.device = self.config.device or next(model.parameters()).device
        self.n_calls = 0  # Track operator calls for profiling

        # Cache model info
        self.n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self.params = [p for p in model.parameters() if p.requires_grad]

    @abstractmethod
    def single_batch_operation(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
        **kwargs
    ) -> Any:
        """
        Perform operation on a single batch.

        Args:
            batch: Input batch
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Result of the operation
        """
        pass

    @abstractmethod
    def aggregate_results(self, results: List[Any]) -> Any:
        """
        Aggregate results from multiple batches.

        Args:
            results: List of results from single_batch_operation

        Returns:
            Aggregated result
        """
        pass

    def multi_batch_operation(
        self,
        batches: List[Dict[str, torch.Tensor]],
        max_batches: Optional[int] = None,
        sampling_strategy: str = 'sequential',
        *args,
        **kwargs
    ) -> Any:
        """
        Perform operation on multiple batches with memory management.

        Args:
            batches: List of input batches
            max_batches: Maximum number of batches to use
            sampling_strategy: How to select batches ('sequential', 'random', 'stratified')
            *args: Additional arguments for single_batch_operation
            **kwargs: Additional keyword arguments

        Returns:
            Aggregated result across batches
        """
        # Sample batches if needed
        selected_batches = self._sample_batches(batches, max_batches, sampling_strategy)
        logger.info(f"{self.__class__.__name__}: Processing {len(selected_batches)} batches")

        results = []
        for i, batch in enumerate(selected_batches):
            # Memory management
            if self.config.memory_efficient and i > 0:
                if i % self.config.clear_cache_freq == 0:
                    self._clear_memory()

            try:
                # Compute for single batch
                result = self.single_batch_operation(batch, *args, **kwargs)
                results.append(result)

                # Progress logging for long operations
                if (i + 1) % 10 == 0:
                    logger.debug(f"  Processed {i+1}/{len(selected_batches)} batches")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on batch {i+1}, skipping. Processed {len(results)} so far.")
                    self._clear_memory()
                    continue
                else:
                    raise

        if not results:
            raise RuntimeError("Failed to process any batches")

        # Aggregate results
        return self.aggregate_results(results)

    def _sample_batches(
        self,
        batches: List[Dict[str, torch.Tensor]],
        max_batches: Optional[int],
        strategy: str
    ) -> List[Dict[str, torch.Tensor]]:
        """Sample batches according to strategy."""
        if max_batches is None or len(batches) <= max_batches:
            return batches

        if strategy == 'random':
            import random
            return random.sample(batches, max_batches)
        elif strategy == 'stratified':
            # Take evenly spaced batches
            indices = torch.linspace(0, len(batches)-1, max_batches).long()
            return [batches[i] for i in indices]
        else:  # sequential
            return batches[:max_batches]

    def _clear_memory(self):
        """Clear GPU memory cache."""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    def __call__(
        self,
        data: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        *args,
        **kwargs
    ) -> Any:
        """
        Make operator callable with single or multiple batches.

        Args:
            data: Single batch or list of batches
            *args: Additional arguments
            **kwargs: Additional keyword arguments

        Returns:
            Operation result
        """
        self.n_calls += 1

        if isinstance(data, list):
            return self.multi_batch_operation(data, *args, **kwargs)
        else:
            return self.single_batch_operation(data, *args, **kwargs)


class VectorOperator(BatchAwareOperator):
    """
    Abstract base for operators that perform vector operations (e.g., HVP, MVP).

    These operators take a vector and return a vector of the same dimension.
    Common examples: Hessian-vector product, Fisher-vector product.
    """

    @abstractmethod
    def matvec(self, v: List[torch.Tensor], batch: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
        """
        Matrix-vector product for a single batch.

        Args:
            v: Input vector (list of tensors matching parameter shapes)
            batch: Data batch

        Returns:
            Output vector (list of tensors matching parameter shapes)
        """
        pass

    def single_batch_operation(
        self,
        batch: Dict[str, torch.Tensor],
        v: List[torch.Tensor]
    ) -> List[torch.Tensor]:
        """Wrapper for matvec to match interface."""
        return self.matvec(v, batch)

    def aggregate_results(
        self,
        results: List[List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        Average vector results across batches.

        For vector operators, we average the vectors element-wise.
        This implements: (1/n) Σ Op(batch_i, v)
        """
        n_results = len(results)

        # Initialize with zeros
        aggregated = [torch.zeros_like(r) for r in results[0]]

        # Sum all results
        for result in results:
            for i, r in enumerate(result):
                aggregated[i] = aggregated[i] + r

        # Average
        for i in range(len(aggregated)):
            aggregated[i] = aggregated[i] / n_results

        return aggregated


class ScalarOperator(BatchAwareOperator):
    """
    Abstract base for operators that return scalar values.

    Examples: Loss computation, metric calculation.
    """

    @abstractmethod
    def compute_scalar(self, batch: Dict[str, torch.Tensor]) -> float:
        """
        Compute scalar value for a single batch.

        Args:
            batch: Input batch

        Returns:
            Scalar value
        """
        pass

    def single_batch_operation(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
        **kwargs
    ) -> float:
        """Wrapper for compute_scalar to match interface."""
        return self.compute_scalar(batch)

    def aggregate_results(self, results: List[float]) -> Dict[str, float]:
        """
        Aggregate scalar results with statistics.

        Returns mean, std, min, max for analysis.
        """
        import numpy as np

        results_array = np.array(results)
        return {
            'mean': float(np.mean(results_array)),
            'std': float(np.std(results_array)),
            'min': float(np.min(results_array)),
            'max': float(np.max(results_array)),
            'n_batches': len(results)
        }


class MatrixOperator(BatchAwareOperator):
    """
    Abstract base for operators that return matrices.

    Examples: Covariance computation, correlation matrices.
    """

    @abstractmethod
    def compute_matrix(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute matrix for a single batch.

        Args:
            batch: Input batch

        Returns:
            Matrix result
        """
        pass

    def single_batch_operation(
        self,
        batch: Dict[str, torch.Tensor],
        *args,
        **kwargs
    ) -> torch.Tensor:
        """Wrapper for compute_matrix to match interface."""
        return self.compute_matrix(batch)

    def aggregate_results(self, results: List[torch.Tensor]) -> torch.Tensor:
        """
        Average matrix results across batches.

        For covariance-like matrices, this gives the expected value.
        """
        # Stack and average
        stacked = torch.stack(results)
        return torch.mean(stacked, dim=0)