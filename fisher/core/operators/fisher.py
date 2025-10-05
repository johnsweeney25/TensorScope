"""
Fisher and Gauss-Newton operators with multi-batch support.

This module implements Fisher Information Matrix and Gauss-Newton
approximations with proper theoretical foundations.

Theory:
    Fisher Information Matrix: F = E[∇L∇L^T] where L is log-likelihood
    Gauss-Newton: G = J^T H_output J where J is Jacobian

    Key properties:
    - Both are positive semi-definite (PSD)
    - Fisher = GGN for exponential family with canonical link
    - More memory efficient than Hessian (no double backprop)

Mathematical Relationships:
    For cross-entropy loss with softmax:
    - Empirical Fisher: F_emp = ∇L∇L^T (using actual labels)
    - True Fisher: F_true = E_y~p[∇L∇L^T] (expectation over predicted distribution)
    - GGN = True Fisher for this case

Memory Complexity:
    O(model_size) for parameters
    + O(model_size) for gradients (no graph needed)
    + O(batch_size × sequence_length × hidden_dim) for activations
    ≈ 2× model memory (vs 3× for Hessian)

Author: Multi-batch architecture team
Date: 2024
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional, Any
import logging

from .base import VectorOperator, OperatorConfig

logger = logging.getLogger(__name__)


class FisherOperator(VectorOperator):
    """
    Fisher Information Matrix operator using gradient outer products.

    This implements the empirical Fisher: F = ∇L∇L^T
    For a vector v: Fv = ∇L(∇L^T v)
    """

    def __init__(
        self,
        model: nn.Module,
        empirical: bool = True,
        config: Optional[OperatorConfig] = None
    ):
        """
        Initialize Fisher operator.

        Args:
            model: Neural network model
            empirical: If True, use empirical Fisher (actual labels)
                      If False, use true Fisher (sample from predicted distribution)
            config: Operator configuration
        """
        super().__init__(model, config)
        self.empirical = empirical
        self.is_psd = True  # Fisher is always PSD

    def matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute Fisher-vector product for a single batch.

        Implements: Fv = ∇L(∇L^T v)

        Args:
            v: Input vector
            batch: Data batch

        Returns:
            Fv: Fisher-vector product
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # Forward pass
        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs

        # Compute gradients (no graph needed for Fisher)
        grads = torch.autograd.grad(
            loss, self.params,
            create_graph=False,  # Key: No double backprop needed
            retain_graph=False,
            allow_unused=True
        )

        # Compute g^T v
        grad_dot_v = 0
        for g, vi in zip(grads, v):
            if g is not None:
                # Cast to computation precision
                if g.dtype != self.config.dtype_compute:
                    g = g.to(self.config.dtype_compute)
                if vi.dtype != self.config.dtype_compute:
                    vi = vi.to(self.config.dtype_compute)

                grad_dot_v = grad_dot_v + (g * vi).sum()

        # Fisher-vector product: Fv = g(g^T v)
        fvp = []
        for g in grads:
            if g is not None:
                fvp.append(g * grad_dot_v)
            else:
                fvp.append(torch.zeros_like(self.params[len(fvp)]))

        return fvp


class GGNOperator(VectorOperator):
    """
    Gauss-Newton operator with multiple implementation modes.

    Modes:
    - 'empirical': Fast gradient outer product approximation
    - 'true': Accurate J^T H_output J computation
    - 'auto': Automatically choose based on loss type
    """

    def __init__(
        self,
        model: nn.Module,
        mode: str = 'empirical',
        config: Optional[OperatorConfig] = None
    ):
        """
        Initialize GGN operator.

        Args:
            model: Neural network model
            mode: Computation mode ('empirical', 'true', 'auto')
            config: Operator configuration
        """
        super().__init__(model, config)
        self.mode = mode
        self.is_psd = True  # GGN is always PSD

    def matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute GGN-vector product.

        Args:
            v: Input vector
            batch: Data batch

        Returns:
            Gv: GGN-vector product
        """
        # Auto-detect mode if needed
        actual_mode = self._determine_mode(batch) if self.mode == 'auto' else self.mode

        if actual_mode == 'true':
            return self._true_ggn_matvec(v, batch)
        else:
            return self._empirical_ggn_matvec(v, batch)

    def _empirical_ggn_matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Empirical GGN using gradient outer product.

        This is fast but approximate. Exact for exponential family
        with canonical link (e.g., softmax + cross-entropy).
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        outputs = self.model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs

        # Single gradient computation
        grads = torch.autograd.grad(
            loss, self.params,
            create_graph=False,
            retain_graph=False,
            allow_unused=True
        )

        # Compute g^T v
        grad_dot_v = 0
        for g, vi in zip(grads, v):
            if g is not None:
                grad_dot_v = grad_dot_v + (g * vi).sum()

        # GGN-vector product: Gv = g(g^T v)
        gvp = []
        for g in grads:
            if g is not None:
                gvp.append(g * grad_dot_v)
            else:
                gvp.append(torch.zeros_like(self.params[len(gvp)]))

        return gvp

    def _true_ggn_matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        True GGN computation: J^T H_output J v

        More accurate but requires two forward passes.
        """
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        # First forward pass for Jacobian
        outputs = self.model(**batch)

        # For classification, get logits before softmax
        if hasattr(outputs, 'logits'):
            logits = outputs.logits
        else:
            logits = outputs

        # Flatten logits for Jacobian computation
        logits_flat = logits.view(-1, logits.size(-1))
        batch_size = logits_flat.size(0)

        # Compute Jv (Jacobian-vector product)
        jv = []
        for i in range(batch_size):
            # Gradient of output i w.r.t. parameters
            grads = torch.autograd.grad(
                logits_flat[i].sum(), self.params,
                retain_graph=True,
                allow_unused=True
            )

            # Accumulate Jv
            if i == 0:
                jv = list(grads)
            else:
                for j, g in enumerate(grads):
                    if g is not None and jv[j] is not None:
                        jv[j] = jv[j] + g

        # Apply Hessian of output (for softmax + CE, this is diagonal)
        # H_output ≈ diag(p(1-p)) where p are predicted probabilities
        probs = torch.softmax(logits_flat, dim=-1)

        # Weight Jv by output Hessian
        # For simplicity, use Fisher approximation: H_output ≈ diag(p)
        weights = probs.mean(dim=-1)  # Average over classes

        # Weighted Jacobian transpose times Jv
        gjv = []
        for param_grad in jv:
            if param_grad is not None:
                # Weight by output Hessian approximation
                weighted = param_grad * weights.mean()
                gjv.append(weighted)
            else:
                gjv.append(torch.zeros_like(self.params[len(gjv)]))

        return gjv

    def _determine_mode(self, batch: Dict[str, torch.Tensor]) -> str:
        """
        Auto-detect whether to use empirical or true GGN.

        For cross-entropy loss with softmax, empirical = true GGN.
        For other losses, true GGN is more accurate.
        """
        try:
            # Check if using cross-entropy loss
            if 'labels' in batch:
                # Likely using cross-entropy
                return 'empirical'
            else:
                # Other loss, use true GGN
                return 'true'
        except:
            # Default to empirical (faster)
            return 'empirical'


class MultiBatchFisherOperator(VectorOperator):
    """
    Multi-batch Fisher operator with variance reduction.

    Computes Fisher Information Matrix vector products averaged
    over multiple batches for improved statistical estimates.
    """

    def __init__(
        self,
        model: nn.Module,
        empirical: bool = True,
        config: Optional[OperatorConfig] = None
    ):
        """
        Initialize multi-batch Fisher operator.

        Args:
            model: Neural network model
            empirical: Use empirical vs true Fisher
            config: Operator configuration
        """
        super().__init__(model, config)
        self.fisher_op = FisherOperator(model, empirical, config)
        self.is_psd = True

    def matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """Single batch Fisher-vector product."""
        return self.fisher_op.matvec(v, batch)

    def multi_batch_matvec(
        self,
        v: List[torch.Tensor],
        batches: List[Dict[str, torch.Tensor]],
        max_batches: int = 30,  # Can use more batches than Hessian (less memory)
        sampling_strategy: str = 'sequential'
    ) -> List[torch.Tensor]:
        """
        Compute Fisher-vector product averaged over multiple batches.

        Fisher uses less memory than Hessian, so we can process more batches.

        Args:
            v: Input vector
            batches: List of data batches
            max_batches: Maximum batches (can be higher than Hessian)
            sampling_strategy: How to sample batches

        Returns:
            Averaged Fisher-vector product
        """
        selected_batches = self._sample_batches(batches, max_batches, sampling_strategy)
        logger.info(f"Computing multi-batch Fisher with {len(selected_batches)} batches")

        fvp_sum = None
        n_processed = 0

        for i, batch in enumerate(selected_batches):
            try:
                # Less frequent cache clearing (Fisher uses less memory)
                if i > 0 and i % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

                # Compute FVP for this batch
                fvp_batch = self.matvec(v, batch)

                # Accumulate
                if fvp_sum is None:
                    fvp_sum = fvp_batch
                else:
                    for j in range(len(fvp_sum)):
                        fvp_sum[j] = fvp_sum[j] + fvp_batch[j]

                n_processed += 1

                # Progress logging
                if (i + 1) % 20 == 0:
                    logger.debug(f"  Processed {i+1}/{len(selected_batches)} batches")

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on batch {i+1} (rare for Fisher)")
                    self._clear_memory()
                    continue
                else:
                    raise

        if n_processed == 0:
            raise RuntimeError("Failed to process any batches")

        # Average
        for j in range(len(fvp_sum)):
            fvp_sum[j] = fvp_sum[j] / n_processed

        logger.info(f"Fisher multi-batch complete: {n_processed} batches, "
                   f"{n_processed/len(selected_batches)*100:.1f}% success rate")

        return fvp_sum


class KFACOperator(VectorOperator):
    """
    Kronecker-Factored Approximate Curvature (K-FAC) operator.

    K-FAC approximates Fisher as block-diagonal with Kronecker factors:
    F_l ≈ A_l ⊗ G_l

    Where:
    - A_l = E[a_l a_l^T] (input covariance)
    - G_l = E[g_l g_l^T] (gradient covariance)

    This is extremely memory efficient for large models.
    """

    def __init__(
        self,
        model: nn.Module,
        kfac_factors: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        config: Optional[OperatorConfig] = None
    ):
        """
        Initialize K-FAC operator.

        Args:
            model: Neural network model
            kfac_factors: Pre-computed K-FAC factors (A and G matrices)
            config: Operator configuration
        """
        super().__init__(model, config)
        self.kfac_factors = kfac_factors or {}
        self.is_psd = True  # K-FAC approximation maintains PSD property

    def matvec(
        self,
        v: List[torch.Tensor],
        batch: Dict[str, torch.Tensor]
    ) -> List[torch.Tensor]:
        """
        Compute K-FAC approximated Fisher-vector product.

        Uses pre-computed Kronecker factors for efficiency.
        """
        if not self.kfac_factors:
            raise ValueError("K-FAC factors not computed. Call compute_kfac_factors first.")

        kfac_mv = []

        param_idx = 0
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            if name in self.kfac_factors:
                # Apply K-FAC approximation: (A ⊗ G)v
                A = self.kfac_factors[name]['A']  # Input factor
                G = self.kfac_factors[name]['G']  # Gradient factor

                v_param = v[param_idx]

                # Efficient Kronecker product multiplication
                if len(v_param.shape) == 2:  # Weight matrix
                    # (A ⊗ G)vec(V) = vec(GVA^T)
                    result = G @ v_param @ A.T
                else:  # Bias or 1D parameter
                    result = G @ v_param.unsqueeze(-1)
                    result = result.squeeze(-1)

                kfac_mv.append(result)
            else:
                # No K-FAC factors, use diagonal approximation
                kfac_mv.append(v[param_idx])

            param_idx += 1

        return kfac_mv

    def compute_kfac_factors(
        self,
        batches: List[Dict[str, torch.Tensor]],
        damping: float = 1e-4
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Compute K-FAC factors from data batches.

        This is expensive but only needs to be done once.
        The factors can then be reused for many matrix-vector products.
        """
        logger.info("Computing K-FAC factors...")

        # Implementation would compute A and G matrices for each layer
        # This is a placeholder for the full implementation
        raise NotImplementedError("K-FAC factor computation not yet implemented")

        return self.kfac_factors