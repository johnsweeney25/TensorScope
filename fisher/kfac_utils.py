#!/usr/bin/env python3
"""
KFAC Natural Gradient Utilities
================================

Unified utilities for KFAC-based natural gradient computation across all metrics.
Provides clean, numerically robust implementations that avoid clusterfucks.

Theory:
- Natural gradient: ∇_nat = F^(-1) * ∇
- KFAC approximation: F ≈ A ⊗ G where A = activation covariance, G = gradient covariance
- Block-diagonal structure captures more interactions than diagonal Fisher
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple, Union, Any
import logging
import sys
from contextlib import nullcontext
import numpy as np
from collections import defaultdict

# Safe tqdm import with fallback (match superposition style)
try:
    from tqdm.auto import tqdm
except (ImportError, RuntimeError):
    class tqdm:
        def __init__(self, *args, **kwargs):
            pass
        def update(self, *args, **kwargs):
            pass
        def set_postfix(self, *args, **kwargs):
            pass
        def close(self):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass

# Redirect logging through tqdm to avoid broken bars (fallback to no-op)
try:
    from tqdm.contrib.logging import logging_redirect_tqdm
except Exception:
    def logging_redirect_tqdm(**kwargs):
        return nullcontext()

logger = logging.getLogger(__name__)


class KFACNaturalGradient:
    """
    Unified KFAC-based natural gradient computer with numerical robustness.

    Design principles:
    1. Single source of truth for KFAC computation
    2. Numerically stable implementations with careful damping
    3. Efficient caching and reuse of factors
    4. Clean interfaces for all use cases
    """

    def __init__(
        self,
        damping: float = 1e-4,
        damping_A: Optional[float] = None,
        damping_G: Optional[float] = None,
        ema_decay: float = 0.99,
        update_freq: int = 10,
        min_layer_size: int = 32,
        use_eigenvalue_correction: bool = True,
        max_condition_number: float = 1e6,
        use_gpu_eigh: bool = True,
        show_progress: bool = False
    ):
        """
        Initialize K-FAC natural gradient computer per Martens & Grosse (2015).
        
        This implementation follows the original K-FAC paper with theoretically justified
        defaults. All hyperparameters affecting numerical results are documented below
        with references to the literature.

        Args:
            damping: Base damping factor λ for Tikhonov regularization.
                Default: 1e-4 (standard in K-FAC literature, see Martens & Grosse 2015).
                Ensures Fisher approximation F + λI remains positive definite.
                **Scientific note**: This is a regularization hyperparameter that should
                be tuned via validation performance, similar to weight decay.
                
            damping_A: Separate damping for activation covariance A.
                Default: None (uses damping). Can be set differently if activations and
                gradients have vastly different scales.
                
            damping_G: Separate damping for gradient covariance G.
                Default: None (uses damping). Useful for models with gradient scale issues.
                
            ema_decay: Exponential moving average decay α for running estimates.
                Default: 0.99 (from Ba et al. 2017, "Distributed Second-Order Optimization").
                Fisher factors: F_t = α·F_{t-1} + (1-α)·F_batch
                **Implementation detail**: Does not affect final results, only convergence.
                
            update_freq: Update K-FAC factors every N steps.
                Default: 10 (standard practice for computational efficiency).
                **Implementation detail**: Does not affect converged results.
                
            min_layer_size: Minimum layer dimension to use K-FAC.
                Default: 32 (for layers with fewer than 32 features, use diagonal Fisher).
                **Justification**: K-FAC overhead (eigendecomposition) outweighs benefits
                for very small layers. This is a performance optimization, not a hyperparameter.
                
            use_eigenvalue_correction: Whether to apply eigenvalue clipping.
                Default: True (REQUIRED for K-FAC theory to hold).
                **Critical**: Setting this to False violates K-FAC assumptions and may
                lead to numerical instability or divergence. Always keep True for publication.
                
            max_condition_number: Maximum condition number κ for Fisher factors.
                Default: 1e6 (standard in optimization literature, see Nocedal & Wright 2006).
                Eigenvalues are clipped: λ_min' = max(λ_min, λ_max / κ_max).
                **Justification**: 
                - κ = 1e6 ensures eigenvalues span at most 6 orders of magnitude
                - Prevents numerical issues in matrix inversion (which requires ~1/condition_number precision)
                - float32 has ~7 decimal digits, so 1e6 leaves safety margin
                - Larger κ (e.g., 1e10) can cause inversion instability
                - Smaller κ (e.g., 1e4) over-regularizes and degrades natural gradient quality
                **Sensitivity**: Results should be REPORTED with this value in experiments section.
                Ablation over {1e4, 1e5, 1e6, 1e7} recommended if reviewers ask.
                
            use_gpu_eigh: Whether to prefer GPU for eigendecomposition.
                Default: True (faster when memory allows).
                **Implementation detail**: Does not affect numerical results (CPU and GPU
                eigendecomposition are deterministic). Only affects speed.
                
            show_progress: Whether to display progress bars.
                Default: False.
                **Implementation detail**: UI only, no effect on results.
                
        References:
            - Martens & Grosse (2015): "Optimizing Neural Networks with Kronecker-factored 
              Approximate Curvature" (ICML)
            - Ba et al. (2017): "Distributed Second-Order Optimization using Kronecker-Factored 
              Approximations" (ICLR)
            - Nocedal & Wright (2006): "Numerical Optimization" (2nd ed.), Chapter 3
        """
        self.damping = damping
        self.damping_A = damping_A if damping_A is not None else damping
        self.damping_G = damping_G if damping_G is not None else damping
        self.ema_decay = ema_decay
        self.update_freq = update_freq
        self.min_layer_size = min_layer_size
        self.use_eigenvalue_correction = use_eigenvalue_correction
        self.max_condition_number = max_condition_number
        self.use_gpu_eigh = use_gpu_eigh
        self.show_progress = show_progress
        # Storage
        self.kfac_factors = {}  # {layer_name: {'A_eigvecs', 'A_eigvals', 'G_eigvecs', 'G_eigvals'}}
        self.diagonal_fisher = {}  # Fallback diagonal Fisher
        self.update_count = 0
        self.inv_cache = {}  # Cache inverted factors

    def collect_kfac_factors(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        loss: Optional[torch.Tensor] = None,
        fisher_type: str = "empirical",
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Collect KFAC factors from a forward-backward pass.

        Args:
            model: Neural network model
            batch: Input batch
            loss: Optional precomputed loss
            fisher_type: Type of Fisher to compute:
                - "empirical": Use actual labels (GGN approximation)
                - "true": Sample labels from model's predictive distribution
                - "mc": Monte Carlo sampling (alias for "true")
            attention_mask: Optional mask for sequence models (shape: [batch, seq_len])

        Returns:
            Dictionary of KFAC factors per layer
        """
        # Auto-wire attention mask from batch if not provided
        if attention_mask is None and 'attention_mask' in batch:
            attention_mask = batch['attention_mask']

        self.update_count += 1

        # Only update periodically for efficiency
        if self.update_count % self.update_freq != 0:
            return self.kfac_factors

        device = next(model.parameters()).device
        activations = {}
        gradients = {}

        # Register hooks for collecting activations and gradients
        handles = []

        def save_input_hook(name):
            def hook(module, input, output):
                try:
                    # Safely extract input
                    if isinstance(input, tuple):
                        if len(input) == 0:
                            logger.warning(f"Empty input tuple for {name}")
                            return
                        input = input[0]

                    # Check for None or empty tensor
                    if input is None or input.numel() == 0:
                        logger.warning(f"Empty input for {name}")
                        return

                    act = input.detach()

                    # Flatten batch dimensions safely BEFORE adding bias term
                    if act.dim() > 2:
                        act = act.reshape(-1, act.shape[-1])
                    elif act.dim() < 2:
                        logger.warning(f"Unexpected activation shape {act.shape} for {name}")
                        return

                    # Add homogeneous coordinate for bias after flattening
                    if module.bias is not None:
                        ones = torch.ones(act.shape[0], 1, device=act.device, dtype=act.dtype)
                        act = torch.cat([act, ones], dim=-1)

                    # Apply attention mask if provided for sequence models
                    if attention_mask is not None:
                        mask_flat = attention_mask.reshape(-1)
                        if mask_flat.shape[0] == act.shape[0]:
                            mask_flat = mask_flat.to(device=act.device).bool()
                            if mask_flat.any():  # Only apply mask if at least one element is True
                                act = act[mask_flat]
                            else:
                                logger.warning(f"All mask values are False for {name}")
                                return

                    # Check for NaN/Inf
                    if not torch.isfinite(act).all():
                        logger.warning(f"Non-finite values in activation for {name}")
                        act = torch.nan_to_num(act, nan=0.0, posinf=1e6, neginf=-1e6)

                    activations[name] = act

                except Exception as e:
                    logger.error(f"Error in forward hook for {name}: {e}")

            return hook

        def save_grad_hook(name):
            def hook(module, grad_input, grad_output):
                try:
                    # Safely extract gradient
                    if isinstance(grad_output, tuple):
                        if len(grad_output) == 0:
                            logger.warning(f"Empty grad_output tuple for {name}")
                            return
                        grad_output = grad_output[0]

                    # Check for None or empty tensor
                    if grad_output is None or grad_output.numel() == 0:
                        logger.warning(f"Empty grad_output for {name}")
                        return

                    grad = grad_output.detach()

                    # Flatten batch dimensions safely for sequence tensors
                    if grad.dim() > 2:
                        grad = grad.reshape(-1, grad.shape[-1])
                    elif grad.dim() < 2:
                        logger.warning(f"Unexpected gradient shape {grad.shape} for {name}")
                        return

                    # Apply attention mask if provided for sequence models
                    if attention_mask is not None:
                        mask_flat = attention_mask.reshape(-1)
                        if mask_flat.shape[0] == grad.shape[0]:
                            mask_flat = mask_flat.to(device=grad.device).bool()
                            if mask_flat.any():  # Only apply mask if at least one element is True
                                grad = grad[mask_flat]
                            else:
                                logger.warning(f"All mask values are False for gradient in {name}")
                                return

                    # Check for NaN/Inf
                    if not torch.isfinite(grad).all():
                        logger.warning(f"Non-finite values in gradient for {name}")
                        grad = torch.nan_to_num(grad, nan=0.0, posinf=1e6, neginf=-1e6)

                    gradients[name] = grad

                except Exception as e:
                    logger.error(f"Error in backward hook for {name}: {e}")

            return hook

        # Register hooks for eligible layers and count for progress
        eligible_layers: List[str] = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Skip small layers
                if module.in_features < self.min_layer_size or module.out_features < self.min_layer_size:
                    continue
                eligible_layers.append(name)
                handles.append(module.register_forward_hook(save_input_hook(name)))
                handles.append(module.register_full_backward_hook(save_grad_hook(name)))

        # Always do a fresh forward pass to trigger hooks
        model.zero_grad()

        # Always pass full batch to preserve attention_mask, labels, etc.
        outputs = model(**batch)

        # Compute loss based on Fisher type
        if loss is None:
            if fisher_type in ["true", "mc"]:
                # True Fisher: sample labels from model's predictive distribution
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                # Sample from the model's predictive distribution
                with torch.no_grad():
                    probs = torch.softmax(logits, dim=-1)
                    # Sample labels from the distribution
                    sampled_labels = torch.multinomial(
                        probs.view(-1, probs.shape[-1]),
                        num_samples=1
                    ).squeeze(-1).view(probs.shape[:-1])

                # Compute loss with sampled labels
                if logits.dim() == 2 and sampled_labels.dim() == 1:
                    loss = nn.CrossEntropyLoss()(logits, sampled_labels)
                else:
                    # For sequence models, flatten
                    loss = nn.CrossEntropyLoss()(
                        logits.reshape(-1, logits.shape[-1]),
                        sampled_labels.reshape(-1)
                    )
            else:
                # Empirical Fisher: use actual labels
                if 'labels' in batch:
                    labels = batch['labels']
                    if hasattr(outputs, 'loss'):
                        loss = outputs.loss
                    elif hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        if logits.dim() == 2 and labels.dim() == 1:
                            loss = nn.CrossEntropyLoss()(logits, labels)
                        else:
                            loss = nn.CrossEntropyLoss()(
                                logits.reshape(-1, logits.shape[-1]),
                                labels.reshape(-1)
                            )
                    elif outputs.dim() == 2 and labels.dim() == 1:
                        loss = nn.CrossEntropyLoss()(outputs, labels)
                    else:
                        loss = outputs.mean()  # Fallback
                else:
                    loss = outputs.mean()  # Fallback
        else:
            # Direct model call
            outputs = model(**batch)
            if loss is None:
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs.mean()

        try:
            # Backward pass to trigger gradient hooks
            loss.backward()

            # Progress bar for K-FAC factor updates (avoid log spam)
            # Match superposition style: single line, throttled updates, no postfix churn
            use_bar = False
            if self.show_progress:
                try:
                    use_bar = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
                except Exception:
                    use_bar = False

            pbar = None
            if use_bar:
                pbar = tqdm(
                    total=len(eligible_layers),
                    desc="K-FAC (layers)",
                    unit="layer",
                    leave=False,
                    dynamic_ncols=True,
                    mininterval=0.5,
                    ncols=100,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
                )

            # Update KFAC factors with numerical robustness
            cm = logging_redirect_tqdm() if use_bar else nullcontext()
            with cm:
                for idx, name in enumerate(eligible_layers, start=1):
                    if name not in activations:
                        if pbar is not None:
                            pbar.update(1)
                        continue
                    if name in gradients:
                        act = activations[name]
                        grad = gradients[name]

                    # Skip if tensors are too small
                    if act.shape[0] < 2:
                        logger.warning(f"Batch size too small for {name}: {act.shape[0]}")
                        continue

                    # Compute covariances with numerical stability
                    batch_size = act.shape[0]

                    # Activation covariance A = E[a * a^T]
                    A = torch.mm(act.t(), act) / batch_size

                    # Gradient covariance G = E[g * g^T]
                    G = torch.mm(grad.t(), grad) / batch_size

                    # Check for NaN/Inf in covariances
                    if not torch.isfinite(A).all() or not torch.isfinite(G).all():
                        logger.warning(f"Non-finite values in covariances for {name}")
                        continue

                    # Apply eigenvalue correction for numerical stability
                    if self.use_eigenvalue_correction:
                        A_decomp = self._stabilize_matrix(A, name + "_A", damping=self.damping_A)
                        G_decomp = self._stabilize_matrix(G, name + "_G", damping=self.damping_G)
                        
                        # Update with EMA in eigenspace (more efficient than reconstructing)
                        if name in self.kfac_factors and 'A_eigvals' in self.kfac_factors[name]:
                            # EMA update in decomposed form
                            old_A_eigvals = self.kfac_factors[name]['A_eigvals']
                            old_G_eigvals = self.kfac_factors[name]['G_eigvals']
                            
                            # Simple EMA on eigenvalues (eigenvectors change slowly, so we update less frequently)
                            A_eigvals_updated = self.ema_decay * old_A_eigvals + (1 - self.ema_decay) * A_decomp['eigvals']
                            G_eigvals_updated = self.ema_decay * old_G_eigvals + (1 - self.ema_decay) * G_decomp['eigvals']
                            
                            # Store updated decomposition
                            self.kfac_factors[name] = {
                                'A_eigvecs': A_decomp['eigvecs'],  # Update eigenvectors each time
                                'A_eigvals': A_eigvals_updated,
                                'G_eigvecs': G_decomp['eigvecs'],
                                'G_eigvals': G_eigvals_updated
                            }
                    else:
                            # First time: store decomposition directly
                            self.kfac_factors[name] = {
                                'A_eigvecs': A_decomp['eigvecs'],
                                'A_eigvals': A_decomp['eigvals'],
                                'G_eigvecs': G_decomp['eigvecs'],
                                'G_eigvals': G_decomp['eigvals']
                            }
                    else:
                        # Simple damping without eigenvalue correction
                        A_damped = A + self.damping_A * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
                        G_damped = G + self.damping_G * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
                        
                        # Still store as decomposition for consistency (identity eigenvectors)
                        n_A, n_G = A.shape[0], G.shape[0]
                        self.kfac_factors[name] = {
                            'A_eigvecs': torch.eye(n_A, dtype=torch.float32),
                            'A_eigvals': torch.full((n_A,), self.damping_A, dtype=torch.float32),
                            'G_eigvecs': torch.eye(n_G, dtype=torch.float32),
                            'G_eigvals': torch.full((n_G,), self.damping_G, dtype=torch.float32)
                        }
                    
                    # Clear inverse cache for this layer (decomposition has changed)
                        if name in self.inv_cache:
                            del self.inv_cache[name]

                    # Free per-layer activation/gradient to reduce peak memory
                    try:
                        del activations[name]
                    except KeyError:
                        pass
                    try:
                        del gradients[name]
                    except KeyError:
                        pass
                    # Occasionally trim CUDA cache to mitigate fragmentation
                    if torch.cuda.is_available():
                        if idx % 8 == 0:
                            torch.cuda.empty_cache()

                    # Update progress bar (no postfix to avoid spammy reprints in non-TTY consoles)
                    if pbar is not None:
                        pbar.update(1)

        finally:
            # Close progress bar
            if pbar is not None:
                try:
                    pbar.close()
                except Exception:
                    pass
            # Always clean up hooks even if there's an error
            for handle in handles:
                try:
                    handle.remove()
                except:
                    pass  # Hook may have already been removed
            
            # Clear gradients to free GPU memory
            model.zero_grad(set_to_none=True)
            
            # Clear activation and gradient dicts
            activations.clear()
            gradients.clear()
            
            # Force CUDA cache cleanup to reduce fragmentation
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self.kfac_factors

    def _stabilize_matrix(self, M: torch.Tensor, name: str = "", damping: Optional[float] = None) -> Dict[str, torch.Tensor]:
        """
        Stabilise a covariance matrix by enforcing a bounded condition number.

        This follows the standard K-FAC recipe (Martens & Grosse, 2015): compute an
        eigendecomposition and clip small eigenvalues to maintain a maximum condition
        number. Instead of reconstructing the full matrix, we return the eigendecomposition
        directly, which is more memory-efficient and avoids redundant decomposition later.

        Args:
            M: Symmetric positive semidefinite matrix to stabilise.
            name: Layer name for logging context.
            damping: Minimum eigenvalue threshold (defaults to ``self.damping``).

        Returns:
            Dictionary with 'eigvecs' and 'eigvals' (both on CPU, float32).
            The stabilized matrix can be reconstructed as: eigvecs @ diag(eigvals) @ eigvecs.T
        """
        if damping is None:
            damping = self.damping

            orig_device = M.device
            orig_dtype = M.dtype

        # Work in float32 for numerical stability (common practice in K-FAC codebases).
            M_work = M.to(torch.float32)

        # Decide where to run the eigendecomposition. We prefer the original CUDA device
        # when available, but fall back to CPU if the matrix is likely to exceed memory.
            if self.use_gpu_eigh and orig_device.type == 'cuda' and torch.cuda.is_available():
                device_eigh = orig_device

            # Memory check: the cuSOLVER eigen routines need the input matrix plus ~3×
            # workspace. If we cannot secure a 50% headroom, fall back to CPU to avoid OOM.
            try:
                device_index = orig_device.index if orig_device.index is not None else torch.cuda.current_device()
                free_mem, _ = torch.cuda.mem_get_info(device_index)
                elem_bytes = M_work.element_size()
                matrix_bytes = M_work.numel() * elem_bytes
                est_workspace = matrix_bytes * 3
                if est_workspace > free_mem * 0.5:
                    logger.debug(
                        "K-FAC stabilisation for %s: expected GPU workspace %.1f GiB exceeds available %.1f GiB;"
                        " using CPU eigendecomposition",
                        name,
                        est_workspace / (1024 ** 3),
                        free_mem / (1024 ** 3),
                    )
                    device_eigh = torch.device('cpu')
            except Exception:
                # If memory info is unavailable, err on the safe side for very large matrices.
                if M_work.shape[0] > 8192:
                    device_eigh = torch.device('cpu')
            else:
                device_eigh = torch.device('cpu')

            M_work = M_work.to(device_eigh)

            try:
                eigvals, eigvecs = torch.linalg.eigh(M_work)
        except RuntimeError as err:
            if device_eigh.type == 'cuda' and 'out of memory' in str(err).lower():
                logger.warning(
                    "K-FAC stabilisation for %s: GPU eigendecomposition OOM (n=%d); retrying on CPU",
                    name,
                    M_work.shape[0],
                )
                torch.cuda.empty_cache()
                M_work = M_work.cpu()
                eigvals, eigvecs = torch.linalg.eigh(M_work)
                else:
                    raise

        try:
            max_eig = eigvals.max()
            min_allowed = torch.maximum(
                max_eig / self.max_condition_number,
                torch.tensor(damping, dtype=eigvals.dtype, device=eigvals.device),
            )
            eigvals_clipped = torch.clamp(eigvals, min=min_allowed)

            # MEMORY OPTIMIZATION: Return eigendecomposition directly instead of reconstructing.
            # This avoids the expensive Q Λ Q^T reconstruction and subsequent re-decomposition.
            # The natural gradient computation already applies the preconditioner in the eigenbasis.
            return {
                'eigvecs': eigvecs.cpu().float(),
                'eigvals': eigvals_clipped.cpu().float()
            }
        except Exception as err:
            logger.error(
                "K-FAC eigenvalue correction failed for %s (shape=%s): %s; falling back to diagonal damping",
                name,
                tuple(M.shape),
                err,
            )
            # Fallback: return identity eigenvectors with damped eigenvalues
            n = M.shape[0]
            return {
                'eigvecs': torch.eye(n, dtype=torch.float32),
                'eigvals': torch.full((n,), damping, dtype=torch.float32)
            }


    def compute_natural_gradient(
        self,
        gradients: Dict[str, torch.Tensor],
        model: nn.Module,
        scale: float = 1.0,
        use_diagonal_fallback: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Transform gradients to natural gradients using KFAC.

        Args:
            gradients: Dictionary of gradients per parameter
            model: Model (for layer structure)
            scale: Global scaling factor
            use_diagonal_fallback: Use diagonal Fisher for layers without KFAC

        Returns:
            Dictionary of natural gradients
        """
        natural_grads = {}

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            # Check if we have KFAC factors for this layer
            if name in self.kfac_factors:
                # Use KFAC natural gradient
                nat_grads = self._compute_layer_natural_gradient(
                    name, module, gradients, scale
                )
                natural_grads.update(nat_grads)

            elif use_diagonal_fallback:
                # Use diagonal Fisher fallback
                weight_name = f"{name}.weight" if name else "weight"
                bias_name = f"{name}.bias" if name else "bias"

                if weight_name in gradients:
                    grad = gradients[weight_name]
                    # Simple diagonal preconditioning
                    if weight_name in self.diagonal_fisher:
                        fisher_diag = self.diagonal_fisher[weight_name]
                        nat_grad = grad / (fisher_diag + self.damping)
                    else:
                        nat_grad = grad / self.damping
                    natural_grads[weight_name] = nat_grad * scale

                if bias_name in gradients and module.bias is not None:
                    grad = gradients[bias_name]
                    if bias_name in self.diagonal_fisher:
                        fisher_diag = self.diagonal_fisher[bias_name]
                        nat_grad = grad / (fisher_diag + self.damping)
                    else:
                        nat_grad = grad / self.damping
                    natural_grads[bias_name] = nat_grad * scale

            else:
                # Pass through original gradients
                weight_name = f"{name}.weight" if name else "weight"
                bias_name = f"{name}.bias" if name else "bias"

                if weight_name in gradients:
                    natural_grads[weight_name] = gradients[weight_name] * scale
                if bias_name in gradients and module.bias is not None:
                    natural_grads[bias_name] = gradients[bias_name] * scale

        # Handle any remaining gradients not in Linear layers
        for param_name, grad in gradients.items():
            if param_name not in natural_grads:
                natural_grads[param_name] = grad * scale

        return natural_grads

    def _compute_layer_natural_gradient(
        self,
        layer_name: str,
        module: nn.Module,
        gradients: Dict[str, torch.Tensor],
        scale: float
    ) -> Dict[str, torch.Tensor]:
        """
        Compute natural gradient for a single layer using KFAC.

        Natural gradient: G^(-1) * grad * A^(-1)
        """
        result = {}

        # Get decomposed factors (stored on CPU)
        factors = self.kfac_factors[layer_name]
        A_eigvecs = factors['A_eigvecs']
        A_eigvals = factors['A_eigvals']
        G_eigvecs = factors['G_eigvecs']
        G_eigvals = factors['G_eigvals']
        
        # Determine target device from gradients
        weight_name = f"{layer_name}.weight" if layer_name else "weight"
        if weight_name in gradients:
            target_device = gradients[weight_name].device
            # Move decomposition to gradient device
            A_eigvecs = A_eigvecs.to(target_device)
            A_eigvals = A_eigvals.to(target_device)
            G_eigvecs = G_eigvecs.to(target_device)
            G_eigvals = G_eigvals.to(target_device)
        else:
            target_device = torch.device('cpu')

        # Cache the decomposition on the target device (no need to recompute!)
        if layer_name not in self.inv_cache:
                self.inv_cache[layer_name] = {
                'eigvals_A': A_eigvals,
                'eigvecs_A': A_eigvecs,
                'eigvals_G': G_eigvals,
                'eigvecs_G': G_eigvecs,
                        'use_cholesky': False
                    }

        cache = self.inv_cache[layer_name]

        # Compute natural gradients
        weight_name = f"{layer_name}.weight" if layer_name else "weight"
        bias_name = f"{layer_name}.bias" if layer_name else "bias"

        if weight_name in gradients:
            grad = gradients[weight_name]

            # Handle bias by extending gradient matrix
            if module.bias is not None and bias_name in gradients:
                bias_grad = gradients[bias_name].unsqueeze(1)
                # Combine weight and bias gradients
                grad_combined = torch.cat([grad, bias_grad], dim=1)

                # Apply inverse using cached decomposition
                if cache.get('use_cholesky', False):
                    # Use Cholesky solve: L_G * L_G^T * X = grad -> solve for X
                    # First solve L_G * Y = grad for Y
                    Y = torch.linalg.solve_triangular(cache['L_G'], grad_combined, upper=False)
                    # Then solve L_G^T * X = Y for X (which is G^(-1) * grad)
                    G_inv_grad = torch.linalg.solve_triangular(cache['L_G'].T, Y, upper=True)

                    # Same for A: solve A * Z = G_inv_grad^T for Z^T
                    temp = G_inv_grad.T
                    Y = torch.linalg.solve_triangular(cache['L_A'], temp, upper=False)
                    nat_grad_combined = torch.linalg.solve_triangular(cache['L_A'].T, Y, upper=True).T

                elif cache.get('use_pinv', False):
                    # Use precomputed pseudo-inverse
                    nat_grad_combined = torch.mm(torch.mm(cache['G_inv'], grad_combined), cache['A_inv'])

                else:
                    # Use eigendecomposition
                    # G^(-1) * grad = V_G * diag(1/lambda_G) * V_G^T * grad
                    G_inv_grad = cache['eigvecs_G'] @ (
                        (cache['eigvecs_G'].T @ grad_combined) / cache['eigvals_G'].unsqueeze(1)
                    )
                    # grad * A^(-1) = grad * V_A * diag(1/lambda_A) * V_A^T
                    nat_grad_combined = G_inv_grad @ cache['eigvecs_A'] @ (
                        torch.diag(1.0 / cache['eigvals_A']) @ cache['eigvecs_A'].T
                    )

                # Split back into weight and bias
                nat_grad_weight = nat_grad_combined[:, :-1]
                nat_grad_bias = nat_grad_combined[:, -1]

                result[weight_name] = nat_grad_weight * scale
                result[bias_name] = nat_grad_bias * scale
            else:
                # Weight only - truncate A dimensions
                if cache.get('use_cholesky', False):
                    # Cholesky solve for weight only
                    Y = torch.linalg.solve_triangular(cache['L_G'], grad, upper=False)
                    G_inv_grad = torch.linalg.solve_triangular(cache['L_G'].T, Y, upper=True)

                    # Truncate A's Cholesky factor for weight dimensions
                    L_A_truncated = cache['L_A'][:grad.shape[1], :grad.shape[1]]
                    temp = G_inv_grad.T
                    Y = torch.linalg.solve_triangular(L_A_truncated, temp, upper=False)
                    nat_grad = torch.linalg.solve_triangular(L_A_truncated.T, Y, upper=True).T

                elif cache.get('use_pinv', False):
                    # Use precomputed pseudo-inverse
                    A_inv_truncated = cache['A_inv'][:grad.shape[1], :grad.shape[1]]
                    nat_grad = torch.mm(torch.mm(cache['G_inv'], grad), A_inv_truncated)

                else:
                    # Use eigendecomposition with truncation
                    G_inv_grad = cache['eigvecs_G'] @ (
                        (cache['eigvecs_G'].T @ grad) / cache['eigvals_G'].unsqueeze(1)
                    )
                    # Truncate A's eigenvectors
                    eigvecs_A_truncated = cache['eigvecs_A'][:grad.shape[1], :]
                    eigvals_A_inv = 1.0 / cache['eigvals_A']
                    nat_grad = G_inv_grad @ eigvecs_A_truncated @ (
                        torch.diag(eigvals_A_inv) @ eigvecs_A_truncated.T
                    )

                result[weight_name] = nat_grad * scale

        return result

    def update_diagonal_fisher(
        self,
        param_name: str,
        fisher_values: torch.Tensor,
        ema: bool = True
    ):
        """
        Update diagonal Fisher approximation for fallback.

        Args:
            param_name: Parameter name
            fisher_values: Diagonal Fisher values
            ema: Use exponential moving average
        """
        if ema and param_name in self.diagonal_fisher:
            old_fisher = self.diagonal_fisher[param_name]
            self.diagonal_fisher[param_name] = (
                self.ema_decay * old_fisher +
                (1 - self.ema_decay) * fisher_values
            )
        else:
            self.diagonal_fisher[param_name] = fisher_values

    def get_fisher_scaled_gradient(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        compute_fresh: bool = True,
        power: float = -1.0,
        fisher_type: str = "empirical"
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher-scaled gradients: (F + λI)^(power) * g

        This is the main interface for metrics that need Fisher scaling.
        Default power=-1 gives natural gradient.

        Args:
            model: Model with computed gradients
            batch: Input batch
            compute_fresh: Whether to recompute KFAC factors
            power: Power to raise Fisher to (default -1 for inverse)
            fisher_type: 'empirical' or 'true' Fisher (true requires sampling)

        Returns:
            Fisher-scaled gradients
        """
        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        # Update KFAC factors if needed - NOW PASSES fisher_type
        if compute_fresh or not self.kfac_factors:
            self.collect_kfac_factors(model, batch, fisher_type=fisher_type)

        # Apply power scaling if not standard natural gradient
        if power != -1.0:
            # This requires eigendecomposition - more expensive
            return self._compute_powered_natural_gradient(
                gradients, model, power
            )
        else:
            # Standard natural gradient
            return self.compute_natural_gradient(gradients, model)

    def _compute_powered_natural_gradient(
        self,
        gradients: Dict[str, torch.Tensor],
        model: nn.Module,
        power: float
    ) -> Dict[str, torch.Tensor]:
        """
        Compute (F + λI)^(power) * g for arbitrary power.

        Used for fractional powers in natural gradient descent variants.
        Properly handles bias via augmented matrices.
        """
        powered_grads = {}

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if name in self.kfac_factors:
                # Get decomposed factors (already stored in eigenspace)
                factors = self.kfac_factors[name]
                A_eigvecs = factors['A_eigvecs']
                A_eigvals = factors['A_eigvals']
                G_eigvecs = factors['G_eigvecs']
                G_eigvals = factors['G_eigvals']
                
                # Determine target device from gradients
                weight_name = f"{name}.weight" if name else "weight"
                if weight_name in gradients:
                    target_device = gradients[weight_name].device
                    A_eigvecs = A_eigvecs.to(target_device)
                    A_eigvals = A_eigvals.to(target_device)
                    G_eigvecs = G_eigvecs.to(target_device)
                    G_eigvals = G_eigvals.to(target_device)
                else:
                    target_device = torch.device('cpu')

                # Apply power scaling directly to eigenvalues (no need to decompose again!)
                try:
                    # Apply power with separate damping
                    eigvals_A_power = (A_eigvals + self.damping_A) ** power
                    eigvals_G_power = (G_eigvals + self.damping_G) ** power

                    # Reconstruct powered matrices
                    A_power = A_eigvecs @ torch.diag(eigvals_A_power) @ A_eigvecs.t()
                    G_power = G_eigvecs @ torch.diag(eigvals_G_power) @ G_eigvecs.t()

                    # Get gradient names
                    weight_name = f"{name}.weight" if name else "weight"
                    bias_name = f"{name}.bias" if name else "bias"

                    if weight_name in gradients:
                        grad_weight = gradients[weight_name]

                        # Handle bias if present
                        if module.bias is not None and bias_name in gradients:
                            grad_bias = gradients[bias_name].unsqueeze(1)
                            # Combine weight and bias gradients
                            grad_combined = torch.cat([grad_weight, grad_bias], dim=1)

                            # Apply powered transformation: G^power * grad * A^power
                            nat_grad_combined = torch.mm(
                                torch.mm(G_power, grad_combined), A_power
                            )

                            # Split back into weight and bias
                            powered_grads[weight_name] = nat_grad_combined[:, :-1]
                            powered_grads[bias_name] = nat_grad_combined[:, -1]
                        else:
                            # Weight only - truncate A to match dimensions
                            A_power_truncated = A_power[:grad_weight.shape[1], :grad_weight.shape[1]]
                            nat_grad = torch.mm(
                                torch.mm(G_power, grad_weight), A_power_truncated
                            )
                            powered_grads[weight_name] = nat_grad

                except Exception as e:
                    logger.warning(f"Powered scaling failed for {name}: {e}")
                    # Fallback to original gradients
                    weight_name = f"{name}.weight" if name else "weight"
                    bias_name = f"{name}.bias" if name else "bias"
                    if weight_name in gradients:
                        powered_grads[weight_name] = gradients[weight_name]
                    if bias_name in gradients:
                        powered_grads[bias_name] = gradients[bias_name]

        # Handle remaining parameters
        for param_name, grad in gradients.items():
            if param_name not in powered_grads:
                powered_grads[param_name] = grad

        return powered_grads

    def compute_fisher_vector_product(
        self,
        vector: Dict[str, torch.Tensor],
        scale: float = 1.0,
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Fisher-vector product using KFAC: F * v

        Useful for trust region methods and conjugate gradient.
        Properly handles bias via augmented matrices.

        Args:
            vector: Vector to multiply with Fisher
            scale: Scaling factor
            model: Optional model for layer structure info

        Returns:
            Fisher-vector product
        """
        fvp = {}

        for layer_name, factors in self.kfac_factors.items():
            # Get decomposed factors (stored on CPU) and reconstruct on target device
            A_eigvecs = factors['A_eigvecs']
            A_eigvals = factors['A_eigvals']
            G_eigvecs = factors['G_eigvecs']
            G_eigvals = factors['G_eigvals']
            
            # Determine target device from vector
            weight_name = f"{layer_name}.weight" if layer_name else "weight"
            if weight_name in vector:
                target_device = vector[weight_name].device
                A_eigvecs = A_eigvecs.to(target_device)
                A_eigvals = A_eigvals.to(target_device)
                G_eigvecs = G_eigvecs.to(target_device)
                G_eigvals = G_eigvals.to(target_device)
                
                # Reconstruct matrices (only when needed for FVP)
                A = A_eigvecs @ torch.diag(A_eigvals) @ A_eigvecs.t()
                G = G_eigvecs @ torch.diag(G_eigvals) @ G_eigvecs.t()
            else:
                continue

            # Get vector components
            bias_name = f"{layer_name}.bias" if layer_name else "bias"

            if weight_name in vector:
                v_weight = vector[weight_name]

                # Check if we have bias
                has_bias = bias_name in vector

                if has_bias:
                    v_bias = vector[bias_name].unsqueeze(1)
                    # Combine weight and bias vectors
                    v_combined = torch.cat([v_weight, v_bias], dim=1)

                    # Compute G * v * A for combined vector
                    fv_combined = torch.mm(torch.mm(G, v_combined), A)

                    # Split back into weight and bias
                    fvp[weight_name] = fv_combined[:, :-1] * scale
                    fvp[bias_name] = fv_combined[:, -1] * scale
                else:
                    # Weight only - truncate A to match dimensions
                    A_truncated = A[:v_weight.shape[1], :v_weight.shape[1]]

                    # Compute G * v * A (Kronecker product property)
                    fv = torch.mm(torch.mm(G, v_weight), A_truncated)
                    fvp[weight_name] = fv * scale

        # Handle remaining parameters with diagonal approximation
        for param_name, v in vector.items():
            if param_name not in fvp:
                if param_name in self.diagonal_fisher:
                    fvp[param_name] = self.diagonal_fisher[param_name] * v * scale
                else:
                    fvp[param_name] = v * scale

        return fvp

    def _apply_fisher_power(
        self,
        grad: torch.Tensor,
        fisher_factors: Dict[str, torch.Tensor],
        power: float
    ) -> torch.Tensor:
        """
        Apply (F + λI)^(power) to a gradient tensor.

        Used for natural gradient transformations in conflict detection.

        Args:
            grad: Gradient tensor
            fisher_factors: Dictionary with 'A' and 'G' KFAC factors
            power: Power to raise Fisher to (e.g., -0.5 for normalization)

        Returns:
            Transformed gradient
        """
        if 'A' not in fisher_factors or 'G' not in fisher_factors:
            return grad

        # Get factors (may be on CPU) and move to gradient device
        A = fisher_factors['A']
        G = fisher_factors['G']
        
        if grad.device != A.device:
            A = A.to(grad.device)
            G = G.to(grad.device)

        try:
            # Eigendecomposition for powered matrices
            eigvals_A, eigvecs_A = torch.linalg.eigh(A)
            eigvals_G, eigvecs_G = torch.linalg.eigh(G)

            # Apply power with separate damping
            eigvals_A = (eigvals_A + self.damping_A) ** power
            eigvals_G = (eigvals_G + self.damping_G) ** power

            # Reconstruct powered matrices
            A_power = eigvecs_A @ torch.diag(eigvals_A) @ eigvecs_A.t()
            G_power = eigvecs_G @ torch.diag(eigvals_G) @ eigvecs_G.t()

            # Reshape gradient if needed
            orig_shape = grad.shape
            if grad.dim() == 1:
                # Assume it's a flattened weight matrix
                # Try to infer shape from A and G dimensions
                out_features = G.shape[0]
                in_features = A.shape[0]
                if grad.numel() == out_features * in_features:
                    grad = grad.reshape(out_features, in_features)
                else:
                    # Can't reshape, return original
                    return grad

            # Apply transformation: G^power * grad * A^power
            result = torch.mm(torch.mm(G_power, grad), A_power)

            # Restore original shape
            if len(orig_shape) == 1:
                result = result.flatten()

            return result

        except Exception as e:
            logger.debug(f"Fisher power application failed: {e}")
            return grad

    def clear_cache(self):
        """Clear all cached inverses to free memory."""
        self.inv_cache.clear()

    def reset(self):
        """Reset all stored factors and cache."""
        self.kfac_factors.clear()
        self.diagonal_fisher.clear()
        self.inv_cache.clear()
        self.update_count = 0
