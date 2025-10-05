"""
Multi-batch wrapper and integration utilities.

This module provides a unified interface for using multi-batch operators
with the existing codebase, including integration with ICLRMetrics and
unified_model_analysis.

Key Features:
- Unified interface for all multi-batch operations
- Automatic operator selection based on requirements
- Memory-aware batch scheduling
- Variance tracking and reduction verification

Author: Multi-batch architecture team
Date: 2024
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any, Union, Literal
import logging
import numpy as np
from dataclasses import dataclass

from .base import OperatorConfig
from .hessian import HessianOperator, MultiBatchHessianOperator, GaussNewtonApproxHessian
from .fisher import FisherOperator, GGNOperator, MultiBatchFisherOperator

logger = logging.getLogger(__name__)


@dataclass
class MultiBatchConfig:
    """Configuration for multi-batch operations."""
    enabled: bool = True  # Enable multi-batch processing
    max_batches: int = 20  # Maximum batches to use
    sampling_strategy: Literal['sequential', 'random', 'stratified'] = 'sequential'
    memory_limit_gb: Optional[float] = None  # GPU memory limit
    variance_target: Optional[float] = None  # Stop when variance < threshold
    use_gauss_newton: bool = False  # Use GGN instead of Hessian for memory
    track_variance: bool = True  # Track variance reduction


class MultiBatchOperatorFactory:
    """
    Factory for creating appropriate multi-batch operators.

    This class handles operator selection and configuration based on
    the specific requirements of each use case.
    """

    @staticmethod
    def create_eigenvalue_operator(
        operator_type: str,
        model: nn.Module,
        config: Optional[MultiBatchConfig] = None,
        operator_config: Optional[OperatorConfig] = None
    ) -> Any:
        """
        Create operator for eigenvalue computation.

        Args:
            operator_type: 'hessian', 'fisher', 'ggn'
            model: Neural network model
            config: Multi-batch configuration
            operator_config: Operator-specific configuration

        Returns:
            Appropriate operator instance
        """
        config = config or MultiBatchConfig()
        operator_config = operator_config or OperatorConfig()

        if operator_type == 'hessian':
            if config.use_gauss_newton:
                logger.info("Using Gauss-Newton approximation for Hessian (memory efficient)")
                return GaussNewtonApproxHessian(model, operator_config)
            else:
                return MultiBatchHessianOperator(model, None, operator_config)

        elif operator_type == 'fisher':
            return MultiBatchFisherOperator(model, empirical=True, config=operator_config)

        elif operator_type == 'ggn':
            return GGNOperator(model, mode='auto', config=operator_config)

        else:
            raise ValueError(f"Unknown operator type: {operator_type}")


class MultiBatchLanczosInterface:
    """
    Interface for using multi-batch operators with Lanczos algorithm.

    This class provides compatibility with existing Lanczos implementations
    while adding multi-batch support.
    """

    def __init__(
        self,
        operator: Any,
        batches: List[Dict[str, torch.Tensor]],
        config: Optional[MultiBatchConfig] = None
    ):
        """
        Initialize multi-batch Lanczos interface.

        Args:
            operator: Base operator (Hessian, Fisher, etc.)
            batches: List of data batches
            config: Multi-batch configuration
        """
        self.operator = operator
        self.batches = batches
        self.config = config or MultiBatchConfig()
        self.variance_history = []

    def matvec(self, v: Union[torch.Tensor, List[torch.Tensor]]) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Matrix-vector product interface for Lanczos.

        This method is called by the Lanczos algorithm and handles
        multi-batch averaging transparently.

        Args:
            v: Input vector (can be flat tensor or list of tensors)

        Returns:
            Matrix-vector product (same format as input)
        """
        # Convert flat tensor to list if needed
        if isinstance(v, torch.Tensor):
            v_list = self._unflatten_vector(v)
        else:
            v_list = v

        # Perform multi-batch operation
        if self.config.enabled and len(self.batches) > 1:
            result = self._multi_batch_matvec(v_list)
        else:
            # Single batch fallback
            result = self.operator.matvec(v_list, self.batches[0])

        # Convert back to flat tensor if needed
        if isinstance(v, torch.Tensor):
            return self._flatten_vector(result)
        else:
            return result

    def _multi_batch_matvec(self, v: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Perform multi-batch matrix-vector product with variance tracking.

        Args:
            v: Input vector (list of tensors)

        Returns:
            Averaged matrix-vector product
        """
        # Select batches
        selected_batches = self._select_batches()

        results = []
        for i, batch in enumerate(selected_batches):
            try:
                # Clear cache periodically
                if i > 0 and i % 5 == 0:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Compute MVP for this batch
                mvp = self.operator.matvec(v, batch)
                results.append(mvp)

                # Check early stopping based on variance
                if self.config.variance_target and len(results) >= 5:
                    variance = self._compute_variance(results)
                    if variance < self.config.variance_target:
                        logger.info(f"Variance target {self.config.variance_target} reached "
                                  f"after {len(results)} batches")
                        break

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on batch {i+1}, continuing with {len(results)} batches")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise

        # Average results
        averaged = self._average_results(results)

        # Track variance reduction if enabled
        if self.config.track_variance:
            self._track_variance_reduction(results)

        return averaged

    def _select_batches(self) -> List[Dict[str, torch.Tensor]]:
        """Select batches according to strategy."""
        max_batches = min(self.config.max_batches, len(self.batches))

        if self.config.sampling_strategy == 'random':
            import random
            return random.sample(self.batches, max_batches)
        elif self.config.sampling_strategy == 'stratified':
            indices = np.linspace(0, len(self.batches)-1, max_batches).astype(int)
            return [self.batches[i] for i in indices]
        else:  # sequential
            return self.batches[:max_batches]

    def _average_results(self, results: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        """Average multiple MVP results."""
        n_results = len(results)
        averaged = [torch.zeros_like(r) for r in results[0]]

        for result in results:
            for i, r in enumerate(result):
                averaged[i] = averaged[i] + r

        for i in range(len(averaged)):
            averaged[i] = averaged[i] / n_results

        return averaged

    def _compute_variance(self, results: List[List[torch.Tensor]]) -> float:
        """Compute variance across results."""
        # Flatten and stack results
        flattened = []
        for result in results:
            flat = torch.cat([r.flatten() for r in result])
            flattened.append(flat)

        stacked = torch.stack(flattened)
        variance = torch.var(stacked, dim=0).mean().item()
        return variance

    def _track_variance_reduction(self, results: List[List[torch.Tensor]]):
        """Track variance reduction for analysis."""
        if len(results) < 2:
            return

        variance = self._compute_variance(results)
        self.variance_history.append({
            'n_batches': len(results),
            'variance': variance,
            'reduction_factor': 1.0 / variance if variance > 0 else float('inf')
        })

        # Log variance reduction
        if len(results) in [5, 10, 20]:
            logger.info(f"Variance with {len(results)} batches: {variance:.6f} "
                       f"(reduction factor: {len(results):.1f}x theoretical, "
                       f"{1.0/variance if variance > 0 else 'inf':.1f}x actual)")

    def _unflatten_vector(self, v_flat: torch.Tensor) -> List[torch.Tensor]:
        """Convert flat vector to list of parameter-shaped tensors."""
        v_list = []
        offset = 0
        for param in self.operator.params:
            numel = param.numel()
            v_param = v_flat[offset:offset + numel].view_as(param)
            v_list.append(v_param)
            offset += numel
        return v_list

    def _flatten_vector(self, v_list: List[torch.Tensor]) -> torch.Tensor:
        """Convert list of tensors to flat vector."""
        return torch.cat([v.flatten() for v in v_list])


def integrate_with_iclr_metrics():
    """
    Monkey-patch ICLRMetrics to use multi-batch operators.

    This function modifies the ICLRMetrics class to automatically use
    multi-batch operators when multiple batches are provided.
    """
    try:
        import sys
        import os
        sys.path.insert(0, '/Users/john/ICLR 2026 proj/pythonProject')
        from ICLRMetrics import ICLRMetrics

        original_hessian = ICLRMetrics.compute_hessian_eigenvalues_lanczos
        original_fisher = ICLRMetrics.compute_fisher_eigenvalues_lanczos

        def compute_hessian_multi_batch(
            self,
            model,
            data_batch,
            k=5,
            max_iter=20,
            **kwargs
        ):
            """Enhanced Hessian computation with multi-batch support."""

            # Check if we received multiple batches
            if isinstance(data_batch, list) and len(data_batch) > 1:
                logger.info(f"Using multi-batch Hessian with {len(data_batch)} batches")

                # Create multi-batch config
                config = MultiBatchConfig(
                    enabled=True,
                    max_batches=min(20, len(data_batch)),
                    sampling_strategy='sequential',
                    track_variance=True
                )

                # Create operator
                operator = MultiBatchHessianOperator(model)

                # Create interface
                interface = MultiBatchLanczosInterface(operator, data_batch, config)

                # Import Lanczos
                from fisher.core.fisher_lanczos_unified import lanczos_algorithm, LanczosConfig

                # Run Lanczos with multi-batch interface
                lanczos_config = LanczosConfig(k=k, max_iters=max_iter)
                results = lanczos_algorithm(interface, lanczos_config)

                # Add variance tracking info
                results['multi_batch'] = True
                results['n_batches_used'] = len(interface.variance_history[-1]['n_batches']
                                               if interface.variance_history else data_batch)
                results['variance_reduction'] = interface.variance_history[-1]['reduction_factor'] \
                                               if interface.variance_history else 1.0

                return results
            else:
                # Single batch, use original
                if isinstance(data_batch, list):
                    data_batch = data_batch[0]
                return original_hessian(self, model, data_batch, k, max_iter, **kwargs)

        def compute_fisher_multi_batch(
            self,
            model,
            data_batch,
            k=5,
            max_iter=20,
            **kwargs
        ):
            """Enhanced Fisher computation with multi-batch support."""

            # Similar to Hessian but with Fisher operator
            if isinstance(data_batch, list) and len(data_batch) > 1:
                logger.info(f"Using multi-batch Fisher with {len(data_batch)} batches")

                config = MultiBatchConfig(
                    enabled=True,
                    max_batches=min(30, len(data_batch)),  # Can use more batches for Fisher
                    sampling_strategy='sequential',
                    track_variance=True
                )

                operator = MultiBatchFisherOperator(model)
                interface = MultiBatchLanczosInterface(operator, data_batch, config)

                from fisher.core.fisher_lanczos_unified import lanczos_algorithm, LanczosConfig

                lanczos_config = LanczosConfig(k=k, max_iters=max_iter, reorth_period=5)  # PSD
                results = lanczos_algorithm(interface, lanczos_config)

                results['multi_batch'] = True
                results['n_batches_used'] = len(interface.variance_history[-1]['n_batches']
                                               if interface.variance_history else data_batch)
                results['variance_reduction'] = interface.variance_history[-1]['reduction_factor'] \
                                               if interface.variance_history else 1.0

                return results
            else:
                if isinstance(data_batch, list):
                    data_batch = data_batch[0]
                return original_fisher(self, model, data_batch, k, max_iter, **kwargs)

        # Replace methods
        ICLRMetrics.compute_hessian_eigenvalues_lanczos = compute_hessian_multi_batch
        ICLRMetrics.compute_fisher_eigenvalues_lanczos = compute_fisher_multi_batch

        logger.info("Successfully integrated multi-batch operators with ICLRMetrics")

    except Exception as e:
        logger.warning(f"Failed to integrate with ICLRMetrics: {e}")


# Auto-integrate when module is imported
integrate_with_iclr_metrics()