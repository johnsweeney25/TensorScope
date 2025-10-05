"""
Hessian operators with multi-batch support.

This module implements Hessian-vector products with proper theoretical foundations
and numerical stability considerations.

Theory:
    The Hessian matrix H = ∇²L(θ) is the second derivative of the loss.
    For a dataset, we want H = E[∇²L(θ, x)] where x ~ data distribution.

    Key properties:
    - Symmetric (always)
    - Can have negative eigenvalues (non-convex optimization)
    - Expensive to compute (requires double backprop)

Mathematical Foundation:
    Hessian-vector product: Hv = ∇(∇L · v)

    This requires:
    1. Compute gradients g = ∇L with create_graph=True
    2. Compute g · v (dot product)
    3. Take gradient of dot product w.r.t. parameters

Memory Complexity:
    O(model_size) for parameters
    + O(model_size) for gradients with graph
    + O(batch_size × sequence_length × hidden_dim) for activations
    ≈ 3× model memory for double backprop

Author: Multi-batch architecture team
Date: 2024
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Callable, Any
import logging

from .base import VectorOperator, OperatorConfig

logger = logging.getLogger(__name__)


class HessianOperator(VectorOperator):
    """
    Hessian-vector product operator using double backpropagation.

    This operator computes Hv where H is the Hessian matrix of the loss
    with respect to model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Callable] = None,
        config: Optional[OperatorConfig] = None
    ):
        """
        Initialize Hessian operator.

        Args:
            model: Neural network model
            loss_fn: Optional loss function. If None, assumes model outputs loss
            config: Operator configuration
        """
        super().__init__(model, config)
        self.loss_fn = loss_fn
        self.is_psd = False  # Hessian can have negative eigenvalues

    def matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute Hessian-vector product for a single batch.

        Args:
            v: Input vector (list of tensors matching parameter shapes)
            batch: Data batch

        Returns:
            Hv: Hessian-vector product
        """
        # Ensure model is in eval mode for consistency
        self.model.eval()

        # Clear any existing gradients
        self.model.zero_grad(set_to_none=True)

        try:
            # Compute loss
            if self.loss_fn is not None:
                loss = self.loss_fn(self.model, batch)
            else:
                outputs = self.model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs

            # Ensure loss requires grad
            if not loss.requires_grad:
                loss = loss.requires_grad_(True)

            # First gradient computation (with graph for second derivative)
            # CRITICAL: retain_graph=True is required for theoretical correctness
            grads = torch.autograd.grad(
                loss, self.params,
                create_graph=True,   # Need graph for second derivative
                retain_graph=True,   # Must retain for HVP computation
                allow_unused=True
            )

            # Compute g^T v efficiently
            grad_dot_v = 0
            for g, p, vi in zip(grads, self.params, v):
                if g is not None:
                    # Cast to computation precision if needed
                    if g.dtype != self.config.dtype_compute:
                        g = g.to(self.config.dtype_compute)
                    if vi.dtype != self.config.dtype_compute:
                        vi = vi.to(self.config.dtype_compute)

                    grad_dot_v = grad_dot_v + (g * vi).sum()

            # Second gradient (Hessian-vector product)
            hvp = torch.autograd.grad(
                grad_dot_v, self.params,
                retain_graph=False,  # Don't need graph after this
                allow_unused=True
            )

            # Handle None gradients
            hvp_clean = []
            for h, vi in zip(hvp, v):
                if h is not None:
                    hvp_clean.append(h)
                else:
                    hvp_clean.append(torch.zeros_like(vi))

            # Clean up intermediate tensors
            del grads, grad_dot_v

            return hvp_clean

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM in Hessian computation: {e}")
                logger.warning("Consider: 1) Reducing batch size, 2) Using GGN instead")
                self._clear_memory()
                # Return zeros as fallback
                return [torch.zeros_like(vi) for vi in v]
            else:
                raise


class RegularizedHessianOperator(HessianOperator):
    """
    Hessian operator with regularization: (H + λI)v

    Regularization improves conditioning and ensures positive definiteness
    for sufficiently large λ.
    """

    def __init__(
        self,
        model: nn.Module,
        regularization: float = 1e-4,
        loss_fn: Optional[Callable] = None,
        config: Optional[OperatorConfig] = None
    ):
        """
        Initialize regularized Hessian operator.

        Args:
            model: Neural network model
            regularization: Regularization strength λ
            loss_fn: Optional loss function
            config: Operator configuration
        """
        super().__init__(model, loss_fn, config)
        self.regularization = regularization
        self.is_psd = regularization > 0  # PSD if regularized

    def matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute (H + λI)v = Hv + λv

        Args:
            v: Input vector
            batch: Data batch

        Returns:
            Regularized Hessian-vector product
        """
        # Compute Hv
        hvp = super().matvec(v, batch)

        # Add regularization: (H + λI)v = Hv + λv
        for i in range(len(hvp)):
            hvp[i] = hvp[i] + self.regularization * v[i]

        return hvp


class GaussNewtonApproxHessian(VectorOperator):
    """
    Gauss-Newton approximation to Hessian: G = J^T H_output J

    This approximation:
    - Is always positive semi-definite (PSD)
    - Uses less memory (no retain_graph needed)
    - Works well near minima
    - Ignores second-order terms in Hessian

    Theory:
        H = J^T H_output J + Σ ∇²f_i(θ) · r_i
        G = J^T H_output J  (drops second term)

    Where:
        J = Jacobian of outputs w.r.t. parameters
        H_output = Hessian of loss w.r.t. outputs
        r_i = residuals
    """

    def __init__(
        self,
        model: nn.Module,
        config: Optional[OperatorConfig] = None
    ):
        """
        Initialize Gauss-Newton approximation.

        Args:
            model: Neural network model
            config: Operator configuration
        """
        super().__init__(model, config)
        self.is_psd = True  # GGN is always PSD

    def matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute Gauss-Newton approximation Gv.

        For cross-entropy loss, this equals the Fisher Information Matrix.

        Args:
            v: Input vector
            batch: Data batch

        Returns:
            Gv: Gauss-Newton vector product
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs

        # First gradient (no graph needed for GGN)
        grads = torch.autograd.grad(
            loss, self.params,
            create_graph=False,  # Key difference from Hessian
            retain_graph=True,   # Need for second pass
            allow_unused=True
        )

        # Compute g^T v
        grad_dot_v = 0
        for g, vi in zip(grads, v):
            if g is not None:
                grad_dot_v = grad_dot_v + (g * vi).sum()

        # For GGN: multiply by gradient again (outer product approximation)
        gvp = []
        for g, vi in zip(grads, v):
            if g is not None:
                # G ≈ g ⊗ g, so Gv ≈ g(g^T v)
                gvp.append(g * grad_dot_v)
            else:
                gvp.append(torch.zeros_like(vi))

        return gvp


class MultiBatchHessianOperator(VectorOperator):
    """
    Multi-batch Hessian operator with advanced memory management.

    This operator computes the expected Hessian over multiple batches,
    providing better statistical estimates with reduced variance.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[Callable] = None,
        config: Optional[OperatorConfig] = None,
        use_gauss_newton: bool = False
    ):
        """
        Initialize multi-batch Hessian operator.

        Args:
            model: Neural network model
            loss_fn: Optional loss function
            config: Operator configuration
            use_gauss_newton: If True, use GGN approximation (saves memory)
        """
        super().__init__(model, config)
        self.loss_fn = loss_fn
        self.use_gauss_newton = use_gauss_newton

        # Create underlying operator
        if use_gauss_newton:
            self.operator = GaussNewtonApproxHessian(model, config)
            self.is_psd = True
        else:
            self.operator = HessianOperator(model, loss_fn, config)
            self.is_psd = False

    def matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Single batch HVP (delegates to underlying operator).

        Args:
            v: Input vector
            batch: Data batch

        Returns:
            Hessian-vector product
        """
        return self.operator.matvec(v, batch)

    def multi_batch_matvec(
        self,
        v: List[torch.Tensor],
        batches: List[Dict[str, torch.Tensor]],
        max_batches: int = 20,
        sampling_strategy: str = 'sequential'
    ) -> List[torch.Tensor]:
        """
        Compute HVP averaged over multiple batches.

        This implements: Hv = E[∇²L(x)]v ≈ (1/n)Σ∇²L(xᵢ)v

        Args:
            v: Input vector
            batches: List of data batches
            max_batches: Maximum number of batches to use
            sampling_strategy: How to sample batches

        Returns:
            Averaged Hessian-vector product
        """
        selected_batches = self._sample_batches(batches, max_batches, sampling_strategy)
        logger.info(f"Computing multi-batch HVP with {len(selected_batches)} batches")

        hvp_sum = None
        n_processed = 0

        for i, batch in enumerate(selected_batches):
            try:
                # Clear cache periodically
                if i > 0 and i % 5 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Compute HVP for this batch
                hvp_batch = self.matvec(v, batch)

                # Accumulate
                if hvp_sum is None:
                    hvp_sum = hvp_batch
                else:
                    for j in range(len(hvp_sum)):
                        hvp_sum[j] = hvp_sum[j] + hvp_batch[j]

                n_processed += 1

                # Progress logging
                if (i + 1) % 10 == 0:
                    logger.debug(f"  Processed {i+1}/{len(selected_batches)} batches")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on batch {i+1}, continuing with {n_processed} batches")
                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()
                    continue
                else:
                    raise

        if n_processed == 0:
            raise RuntimeError("Failed to process any batches")

        # Average
        for j in range(len(hvp_sum)):
            hvp_sum[j] = hvp_sum[j] / n_processed

        if n_processed < len(selected_batches):
            logger.warning(f"Only processed {n_processed}/{len(selected_batches)} batches")

        return hvp_sum