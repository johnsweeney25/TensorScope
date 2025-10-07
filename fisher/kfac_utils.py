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
import torch.distributed as dist
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
        show_progress: bool = False,
        kfac_use_woodbury: bool = True,
        kfac_policy: str = "all",
        kfac_big_threshold: int = 4096,
        kfac_true_fisher_head: bool = False,
        kfac_eps: float = 1e-6,
        kfac_auto_rho: float = 1.0,
        kfac_auto_t_max: int = 8192,
        woodbury_store_device: str = "auto",
        woodbury_dtype: str = "fp16",
        kfac_distributed_reduce: str = "gram"
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
                
            kfac_use_woodbury: Whether to use Woodbury identity for G-side computation.
                Default: True (recommended for memory efficiency and exact inverse).
                **Theory**: For empirical Fisher, G = U U^T (rank-T). Woodbury gives exact
                (G + λI)^{-1} without forming the o×o matrix, using only U and T×T algebra.
                
            kfac_policy: When to apply Woodbury ("all", "auto", "hybrid", or "small_only").
                Default: "all" (Woodbury everywhere on G-side, eigendecomp on A-side).
                - "all": Use Woodbury for all layers (cleanest, exact for empirical Fisher)
                - "auto": Choose based on compute cost (Woodbury if T ≤ ρ·o and T ≤ T_max)
                - "hybrid": Woodbury for large output layers (out_features ≥ threshold or lm_head)
                - "small_only": Never use Woodbury (legacy eigendecomp path)
                **Recommendation**: Use "all" for paper results, "auto" for production.
                
            kfac_big_threshold: Output dimension threshold for "hybrid" policy.
                Default: 4096 (layers with out_features ≥ 4096 use Woodbury in hybrid mode).
                **Note**: Only affects "hybrid" policy.
                
            kfac_true_fisher_head: Use true Fisher (with sampled labels) for lm_head G-side.
                Default: False (use empirical Fisher everywhere).
                **Theory**: True Fisher for softmax uses G = diag(p̄) - U U^T structure.
                Requires additional computation but can reduce variance.
                
            kfac_eps: Numerical stabilization epsilon for Woodbury solve.
                Default: 1e-6 (initial jitter for Cholesky with backoff).
                **Implementation detail**: Uses exponential backoff (ε, 10ε, 100ε) if needed.
                
            kfac_auto_rho: Cost ratio threshold for "auto" policy.
                Default: 1.0 (use Woodbury if T ≤ ρ·out_dim).
                **Theory**: Woodbury O(oT²+T³) vs eigendecomp O(o³); break-even at T≈ρ·o.
                
            kfac_auto_t_max: Maximum T for Woodbury in "auto" policy.
                Default: 8192 (avoid large T×T matrices even if cheaper than eigendecomp).
                **Practical**: T=8192 → S is 8k×8k×4B = 256MB (manageable).
                
            woodbury_store_device: Device for storing Woodbury factors ("auto", "cuda", "cpu").
                Default: "auto" (GPU if fits, CPU otherwise).
                **Memory**: U+S_inv moved to target device on use.
                
            woodbury_dtype: Dtype for U matrix ("fp16" or "bf16").
                Default: "fp16" (sufficient for gradient statistics).
                **Note**: S_inv always computed in fp32 for stability.
                
            kfac_distributed_reduce: DDP/FSDP reduction mode ("gram", "gather", "none").
                Default: "gram" (all-reduce U^T@U for efficiency).
                - "gram": Reduce Gram matrix U^T@U (cheap, exact for empirical Fisher)
                - "gather": All-gather U columns (more memory, supports arbitrary ops)
                - "none": No distributed reduction (single-GPU or debug)
                
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
        
        # Woodbury identity configuration
        self.kfac_use_woodbury = kfac_use_woodbury
        self.kfac_policy = kfac_policy
        self.kfac_big_threshold = kfac_big_threshold
        self.kfac_eps = kfac_eps
        self.kfac_auto_rho = kfac_auto_rho
        self.kfac_auto_t_max = kfac_auto_t_max
        self.woodbury_store_device = woodbury_store_device
        self.woodbury_dtype = woodbury_dtype
        
        # True Fisher for lm_head (experimental, not yet implemented)
        if kfac_true_fisher_head:
            logger.warning(
                "kfac_true_fisher_head=True is not yet implemented. "
                "See docs/TRUE_FISHER_LM_HEAD_THEORY.md for theory. "
                "Falling back to empirical Fisher."
            )
        self.kfac_true_fisher_head = False  # Disabled until implemented
        
        # Distributed reduction: "gram" mode is mathematically incorrect
        # (all-reducing U^T@U only works if ranks have identical token columns)
        self.kfac_distributed_reduce = kfac_distributed_reduce
        if dist.is_initialized() and self.kfac_distributed_reduce == "gram":
            logger.warning(
                "⚠️  kfac_distributed_reduce='gram' is mathematically incorrect "
                "(U^T@U from different ranks cannot be meaningfully summed). "
                "Switching to 'gather' for correctness. "
                "See KFAC_BUGFIXES_SUMMARY.md for details."
            )
            self.kfac_distributed_reduce = "gather"
        
        # Storage
        # New schema supports both eigendecomp and Woodbury representations:
        # {layer_name: {
        #     'A_eigvecs', 'A_eigvals', 'A_bias_augmented',  # A-side (always eigendecomp)
        #     'G_type': 'eig' | 'woodbury_empirical' | 'woodbury_true',
        #     # If G_type == 'eig':
        #         'G_eigvecs', 'G_eigvals'
        #     # If G_type == 'woodbury_empirical' or 'woodbury_true':
        #         'U', 'S_inv', 'lambda_G', 'T_effective',
        #         'D_lambda_inv' (only for woodbury_true)
        # }}
        self.kfac_factors = {}
        self.diagonal_fisher = {}  # Fallback diagonal Fisher
        self.update_count = 0
        self.inv_cache = {}  # Cache inverted factors
    
    def _should_use_woodbury(
        self, 
        layer_name: str, 
        out_dim: int, 
        T_effective: Optional[int] = None
    ) -> bool:
        """
        Determine if a layer should use Woodbury for G-side based on policy.
        
        Args:
            layer_name: Name of the layer
            out_dim: Output dimension (out_features)
            T_effective: Number of effective tokens (for "auto" policy)
            
        Returns:
            True if Woodbury should be used for this layer's G matrix
            
        Theory:
            Woodbury cost: O(oT² + T³)
            Eigendecomp cost: O(o³)
            Break-even when T ≈ o (approximately)
            
            For "auto" policy, use Woodbury if:
            - T ≤ ρ·o (cost favorable)
            - T ≤ T_max (avoid huge S matrices)
        """
        if not self.kfac_use_woodbury:
            return False
            
        if self.kfac_policy == "all":
            return True
            
        elif self.kfac_policy == "auto":
            # Auto policy: choose based on compute cost
            if T_effective is None:
                # Conservative: assume worst case T = min(seq_len * batch, out_dim)
                logger.debug(f"Auto policy without T_effective for {layer_name}; defaulting to hybrid")
                return layer_name.endswith("lm_head") or out_dim >= self.kfac_big_threshold
            
            # Use Woodbury if:
            # 1. T ≤ ρ·out_dim (compute favorable)
            # 2. T ≤ T_max (memory manageable)
            cost_favorable = T_effective <= self.kfac_auto_rho * out_dim
            memory_safe = T_effective <= self.kfac_auto_t_max
            
            use_woodbury = cost_favorable and memory_safe
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    f"Auto policy for {layer_name}: T={T_effective}, o={out_dim}, "
                    f"cost_favorable={cost_favorable}, memory_safe={memory_safe}, "
                    f"use_woodbury={use_woodbury}"
                )
            
            return use_woodbury
            
        elif self.kfac_policy == "hybrid":
            is_lm_head = layer_name.endswith("lm_head")
            is_big = out_dim >= self.kfac_big_threshold
            return is_lm_head or is_big
            
        elif self.kfac_policy == "small_only":
            return False
            
        else:
            logger.warning(f"Unknown kfac_policy '{self.kfac_policy}', defaulting to 'all'")
            return True

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

                    # Offload activations to CPU immediately to avoid holding large tensors on GPU.
                    if act.is_cuda:
                        act = act.to(device='cpu')
                    activations[name] = act.to(dtype=torch.float32, copy=False)

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

                    # Offload gradients to CPU immediately to keep GPU memory bounded.
                    if grad.is_cuda:
                        grad = grad.to(device='cpu')
                    gradients[name] = grad.to(dtype=torch.float32, copy=False)

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
        # else: loss was already provided, no need to call model again

        # Initialize progress bar variable to avoid UnboundLocalError
        pbar = None
        
        try:
            # Backward pass to trigger gradient hooks
            # Check if backward has already been called by checking if gradients exist
            has_gradients = any(p.grad is not None for p in model.parameters())
            if has_gradients:
                logger.warning("Model already has gradients - backward may have already been called")
            else:
                loss.backward()
                logger.info("K-FAC backward pass completed; assembling per-layer covariances")

            # Progress bar for K-FAC factor updates (avoid log spam)
            # Match superposition style: single line, throttled updates, no postfix churn
            use_bar = False
            if self.show_progress:
                try:
                    use_bar = hasattr(sys.stderr, "isatty") and sys.stderr.isatty()
                except Exception:
                    use_bar = False
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
                    if name not in gradients:
                        if pbar is not None:
                            pbar.update(1)
                        continue
                    
                    act = activations[name]
                    grad = gradients[name]

                    # Skip if tensors are too small
                    if act.shape[0] < 2:
                        logger.warning(f"Batch size too small for {name}: {act.shape[0]}")
                        continue

                    logger.info(f"  ↳ Computing covariances for {name}")

                    # Get dimensions for memory planning
                    batch_size = act.shape[0]
                    in_dim = act.shape[-1]
                    out_dim = grad.shape[-1]
                    T_effective = batch_size  # Will be updated based on actual tokens
                    
                    # Determine if this layer uses Woodbury for G-side
                    # Note: T_effective passed for "auto" policy cost heuristic
                    use_woodbury = self._should_use_woodbury(name, out_dim, T_effective)
                    
                    # ===== A-SIDE (activation covariance): Always use eigendecomp =====
                    # Stage to GPU for fast computation
                    act_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                    
                    # Compute A (activation covariance)
                    try:
                        act_work = act.to(act_device, non_blocking=True)
                        A = torch.mm(act_work.t(), act_work) / batch_size
                        if act_device.type == 'cuda':
                            del act_work
                    except RuntimeError as err:
                        if 'out of memory' in str(err).lower():
                            logger.warning(
                                "    A-covariance GPU OOM for %s (in=%d, out=%d); retrying on CPU",
                                name, in_dim, out_dim
                            )
                            torch.cuda.empty_cache()
                            A = torch.mm(act.t(), act) / batch_size
                        else:
                            raise
                    
                    # Check for NaN/Inf in A
                    if not torch.isfinite(A).all():
                        logger.warning(f"Non-finite values in A covariance for {name}")
                        continue

                    # Apply eigenvalue correction for A-side
                    if self.use_eigenvalue_correction:
                        logger.info(f"  ↳ Stabilizing A eigen-decomposition for {name}")
                        A_decomp = self._stabilize_matrix(A, name + "_A", damping=self.damping_A)
                        
                        # Log clipped eigenvalues for reproducibility
                        if logger.isEnabledFor(logging.DEBUG) and self.max_condition_number:
                            min_allowed = A_decomp['eigvals'].max() / self.max_condition_number
                            n_clipped_A = (A_decomp['eigvals'] <= min_allowed).sum().item()
                            if n_clipped_A > 0:
                                logger.debug(
                                    f"    A-side clipped {n_clipped_A}/{len(A_decomp['eigvals'])} "
                                    f"eigenvalues for {name}"
                                )
                    else:
                        # Simple damping without eigenvalue correction
                        A_damped = A + self.damping_A * torch.eye(A.shape[0], device=A.device, dtype=A.dtype)
                        A_eigvals, A_eigvecs = torch.linalg.eigh(A_damped.cpu().float())
                        A_decomp = {'eigvecs': A_eigvecs, 'eigvals': A_eigvals}
                    
                    # Track whether bias is augmented (for consistency checks)
                    module = dict(model.named_modules())[name]
                    has_bias = module.bias is not None
                    
                    # ===== G-SIDE: Woodbury or eigendecomp based on policy =====
                    if use_woodbury:
                        # Build Woodbury representation: G = U U^T (empirical Fisher)
                        # U must be [out_dim, T] so that G = U @ U.T is [out_dim, out_dim]
                        logger.info(f"  ↳ Building Woodbury factors for {name} (out_dim={out_dim})")
                        
                        # Flatten gradient to token-level
                        G_tokens = grad  # already shape [T, out_dim] after masking
                        T_effective = G_tokens.shape[0]
                        
                        if T_effective == 0:
                            logger.warning(f"No tokens for {name} after masking; skipping")
                            continue
                        
                        # Build U = G_tokens.T / sqrt(T) with shape [out_dim, T]
                        sqrt_T = float(T_effective) ** 0.5
                        
                        # Determine dtype and storage device
                        dtype = torch.float16 if self.woodbury_dtype == "fp16" else torch.bfloat16
                        store_device_raw = (
                            torch.device("cuda") if (
                                self.woodbury_store_device == "cuda" or
                                (self.woodbury_store_device == "auto" and torch.cuda.is_available())
                            ) else torch.device("cpu")
                        )
                        
                        try:
                            # Transpose to [out_dim, T] and scale
                            U = (G_tokens.t().contiguous() / sqrt_T).to(
                                device=store_device_raw, dtype=dtype, non_blocking=True
                            )
                        except RuntimeError:
                            # GPU OOM, fall back to CPU (rare)
                            logger.warning(f"    Woodbury U GPU OOM for {name}; using CPU")
                            torch.cuda.empty_cache()
                            U = (G_tokens.t().contiguous() / sqrt_T).to(device='cpu', dtype=dtype)
                        
                        # Check for NaN/Inf in U
                        if not torch.isfinite(U.float()).all():
                            logger.warning(f"Non-finite values in U for {name}")
                            continue
                        
                        # Sanity check: U must be [out_dim, T]
                        assert U.shape == (out_dim, T_effective), \
                            f"U shape mismatch: expected [{out_dim}, {T_effective}], got {U.shape}"
                        
                        # Build S = I_T + (1/λ_G) * U^T @ U in FP32
                        # With U = [out_dim, T], we have U^T @ U = [T, out_dim] @ [out_dim, T] = [T, T]
                        try:
                            lambda_inv = 1.0 / self.damping_G
                            
                            # === DDP/FSDP: Aggregate U columns across ranks ===
                            if dist.is_initialized() and self.kfac_distributed_reduce == "gather":
                                # All-gather U columns with padding for variable token counts
                                world_size = dist.get_world_size()
                                
                                # Find max T across ranks
                                T_local_tensor = torch.tensor([T_effective], device=U.device, dtype=torch.int64)
                                T_max_tensor = T_local_tensor.clone()
                                dist.all_reduce(T_max_tensor, op=dist.ReduceOp.MAX)
                                T_max = int(T_max_tensor.item())
                                
                                # Pad U to T_max along dim=1 (token dimension)
                                # U is [out_dim, T], pad to [out_dim, T_max]
                                U_pad = torch.zeros(out_dim, T_max, device=U.device, dtype=U.dtype)
                                U_pad[:, :T_effective] = U
                                
                                # All-gather padded U
                                U_list = [torch.empty_like(U_pad) for _ in range(world_size)]
                                dist.all_gather(U_list, U_pad)
                                
                                # All-gather true lengths
                                len_list = [torch.empty_like(T_local_tensor) for _ in range(world_size)]
                                dist.all_gather(len_list, T_local_tensor)
                                lens = [int(x.item()) for x in len_list]
                                
                                # Concatenate unpadded columns along dim=1
                                U_global = torch.cat([U_list[i][:, :lens[i]] for i in range(world_size)], dim=1)
                                T_effective = U_global.shape[1]
                                
                                # Enforce T_max cap from auto policy
                                if T_effective > self.kfac_auto_t_max:
                                    logger.warning(
                                        f"    DDP: T_global={T_effective} exceeds kfac_auto_t_max={self.kfac_auto_t_max}; "
                                        f"truncating for memory safety"
                                    )
                                    U_global = U_global[:, :self.kfac_auto_t_max]
                                    T_effective = self.kfac_auto_t_max
                                
                                logger.debug(
                                    f"    DDP Gather for {name}: T_local={T_local_tensor.item()}, "
                                    f"T_global={T_effective}, world_size={world_size}"
                                )
                                
                                # Update U to global version
                                U = U_global
                            
                            # Build S with (global or local) U
                            # S = I_T + (1/λ_G) * U^T @ U where U is [out_dim, T]
                            # This gives S as [T, T]
                            UT = U.t().to(torch.float32)  # [T, out_dim]
                            U32 = U.to(torch.float32)     # [out_dim, T]
                            S = (UT @ U32) / self.damping_G  # [T, T]
                            S.diagonal().add_(1.0)  # Add I_T
                            
                            # Invert S via Cholesky with robust jitter backoff
                            eps = self.kfac_eps
                            S_inv = None
                            for attempt in range(3):
                                try:
                                    L = torch.linalg.cholesky(S)
                                    S_inv = torch.cholesky_inverse(L)
                                    break
                                except RuntimeError as e:
                                    if attempt < 2:
                                        # Add jitter and retry
                                        S.diagonal().add_(eps)
                                        eps *= 10.0
                                        logger.debug(f"    Cholesky failed for {name}, retrying with jitter")
                                    else:
                                        # Final fallback: use explicit inverse
                                        logger.warning(
                                            f"    Cholesky failed 3 times for {name}; falling back to explicit inverse"
                                        )
                                        S_inv = torch.linalg.inv(S)
                            
                            # Sanity check: S_inv must be [T, T]
                            assert S_inv.shape == (T_effective, T_effective), \
                                f"S_inv shape mismatch for {name}: expected [{T_effective}, {T_effective}], got {S_inv.shape}"
                            
                            # Apply dtype policy (convert from fp16 default if needed)
                            if self.woodbury_dtype == "bf16" and torch.cuda.is_bf16_supported():
                                U = U.to(dtype=torch.bfloat16)
                                dtype_str = "bf16"
                            else:
                                # Keep as fp16 (already set)
                                dtype_str = "fp16"
                            
                            # Determine storage device based on policy
                            U_size_mb = U.numel() * 2 / 1e6  # fp16/bf16 = 2 bytes
                            S_inv_size_mb = S_inv.numel() * 4 / 1e6  # fp32 = 4 bytes
                            total_mb = U_size_mb + S_inv_size_mb
                            
                            if self.woodbury_store_device == "auto":
                                # Auto: GPU if < 500MB, CPU otherwise
                                if total_mb < 500:
                                    store_device = U.device  # Keep on current device (GPU)
                                    device_str = "GPU"
                                else:
                                    store_device = torch.device('cpu')
                                    device_str = "CPU"
                                    U = U.to(device='cpu')
                                    S_inv = S_inv.to(device='cpu')
                            elif self.woodbury_store_device == "cuda":
                                store_device = torch.device('cuda')
                                device_str = "GPU"
                                # Already on GPU
                            else:  # "cpu"
                                store_device = torch.device('cpu')
                                device_str = "CPU"
                                U = U.to(device='cpu')
                                S_inv = S_inv.to(device='cpu')
                            
                            # Store Woodbury factors
                            self.kfac_factors[name] = {
                                'A_eigvecs': A_decomp['eigvecs'],
                                'A_eigvals': A_decomp['eigvals'],
                                'A_bias_augmented': has_bias,
                                'G_type': 'woodbury_empirical',
                                'U': U.contiguous(),  # [out_dim, T], fp16/bf16, device per policy
                                'S_inv': S_inv.contiguous(),  # [T, T], fp32, device per policy
                                'lambda_G': self.damping_G,
                                'T_effective': T_effective
                            }
                            
                            # Reproducibility logging
                            if logger.isEnabledFor(logging.INFO):
                                logger.info(
                                    f"    ✓ Woodbury: out_dim={out_dim}, T={T_effective}, "
                                    f"memory={total_mb:.1f}MB, storage={device_str}, dtype={dtype_str}, "
                                    f"λ_G={self.damping_G:.2e}, policy={self.kfac_policy}"
                                )
                        
                        except Exception as e:
                            logger.error(f"Woodbury construction failed for {name}: {e}; skipping layer")
                            continue
                    
                    else:
                        # Small layer: use traditional eigendecomp for G
                        logger.info(f"  ↳ Using eigendecomp for G ({name}, out_dim={out_dim})")
                        
                        # Compute dense G
                        try:
                            grad_device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
                            grad_work = grad.to(grad_device, non_blocking=True)
                            G = torch.mm(grad_work.t(), grad_work) / batch_size
                            if grad_device.type == 'cuda':
                                del grad_work
                        except RuntimeError as err:
                            if 'out of memory' in str(err).lower():
                                logger.warning(f"    G-covariance GPU OOM for {name}; using CPU")
                                torch.cuda.empty_cache()
                                G = torch.mm(grad.t(), grad) / batch_size
                            else:
                                raise
                        
                        # Check for NaN/Inf
                        if not torch.isfinite(G).all():
                            logger.warning(f"Non-finite values in G covariance for {name}")
                            continue
                        
                        # Decompose G
                        if self.use_eigenvalue_correction:
                            G_decomp = self._stabilize_matrix(G, name + "_G", damping=self.damping_G)
                            # Guard: Ensure eigenvalues are positive before division
                            assert G_decomp['eigvals'].min() > 0, \
                                f"G eigenvalues must be positive after damping (got min={G_decomp['eigvals'].min():.2e})"
                        else:
                            G_damped = G + self.damping_G * torch.eye(G.shape[0], device=G.device, dtype=G.dtype)
                            G_eigvals, G_eigvecs = torch.linalg.eigh(G_damped.cpu().float())
                            G_decomp = {'eigvecs': G_eigvecs, 'eigvals': G_eigvals}
                            # Guard: Ensure eigenvalues are positive
                            assert G_eigvals.min() > 0, \
                                f"G eigenvalues must be positive after damping (got min={G_eigvals.min():.2e})"
                        
                        # Store eigendecomp factors
                        self.kfac_factors[name] = {
                            'A_eigvecs': A_decomp['eigvecs'],
                            'A_eigvals': A_decomp['eigvals'],
                            'A_bias_augmented': has_bias,
                            'G_type': 'eig',
                            'G_eigvecs': G_decomp['eigvecs'],
                            'G_eigvals': G_decomp['eigvals']
                        }
                        
                        # Reproducibility logging for eigendecomp path
                        if logger.isEnabledFor(logging.INFO):
                            # Compute condition numbers
                            kappa_A = float(A_decomp['eigvals'].max() / A_decomp['eigvals'].min())
                            kappa_G = float(G_decomp['eigvals'].max() / G_decomp['eigvals'].min())
                            
                            logger.info(
                                f"    ✓ Eigendecomp: in_dim={in_dim}, out_dim={out_dim}, "
                                f"κ_A={kappa_A:.2e}, κ_G={kappa_G:.2e}, "
                                f"λ_A={self.damping_A:.2e}, λ_G={self.damping_G:.2e}, "
                                f"κ_max={self.max_condition_number:.2e}"
                            )
                    
                    # Note: inv_cache is legacy and unused (factors are fresh on each update)

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
                    # More frequent cleanup for large models to prevent OOM
                    if torch.cuda.is_available():
                        if idx % 4 == 0:  # More frequent cleanup
                            torch.cuda.empty_cache()

                    # Update progress bar (no postfix to avoid spammy reprints in non-TTY consoles)
                    if pbar is not None:
                        pbar.update(1)

            # Log summary statistics for reproducibility
            n_woodbury = sum(1 for f in self.kfac_factors.values() if f.get('G_type') == 'woodbury_empirical')
            n_eig = sum(1 for f in self.kfac_factors.values() if f.get('G_type') == 'eig')
            
            logger.info(
                f"  ✓ K-FAC factor accumulation complete: "
                f"{len(self.kfac_factors)} layers ({n_woodbury} Woodbury, {n_eig} eigendecomp), "
                f"policy={self.kfac_policy}, update_freq={self.update_freq}, "
                f"damping=(λ_A={self.damping_A:.2e}, λ_G={self.damping_G:.2e}), "
                f"κ_max={self.max_condition_number:.2e}"
            )

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

        # Work in float32 for numerical stability (common practice in K-FAC codebases).
        M_work = M.to(torch.float32)

        # Default to CPU; promote to CUDA only while solving if safe.
        device_eigh = torch.device('cpu')
        if self.use_gpu_eigh and torch.cuda.is_available():
            target_device = orig_device if orig_device.type == 'cuda' else torch.device('cuda')

            # Memory check: input + ~3× workspace; require 50% free headroom
            try:
                device_index = target_device.index if target_device.index is not None else torch.cuda.current_device()
                free_mem, _ = torch.cuda.mem_get_info(device_index)
                elem_bytes = M_work.element_size()
                matrix_bytes = M_work.numel() * elem_bytes
                est_workspace = matrix_bytes * 3
                if est_workspace <= free_mem * 0.5:
                    device_eigh = target_device
                else:
                    logger.warning(
                        "K-FAC stabilisation for %s: expected GPU workspace %.1f GiB exceeds available %.1f GiB;"
                        " falling back to CPU eigendecomposition",
                        name,
                        est_workspace / (1024 ** 3),
                        free_mem / (1024 ** 3),
                    )
            except Exception:
                # If memory info is unavailable, err on the safe side for very large matrices.
                if M_work.shape[0] > 8192:
                    logger.warning(
                        "K-FAC stabilisation for %s: matrix dim=%d, memory probes unavailable; using CPU eigendecomposition",
                        name,
                        M_work.shape[0],
                    )
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
            # Add Tikhonov damping λI in eigenspace (true regularization)
            eigvals = eigvals + float(damping)
            
            # Optionally clamp to enforce maximum condition number
            if self.max_condition_number is not None and self.max_condition_number > 0:
                max_eig = eigvals.max()
                min_allowed = max_eig / self.max_condition_number
                eigvals_clipped = torch.clamp(eigvals, min=min_allowed)
            else:
                eigvals_clipped = eigvals

            # MEMORY OPTIMIZATION: Return eigendecomposition directly instead of reconstructing.
            # This avoids the expensive Q Λ Q^T reconstruction and subsequent re-decomposition.
            # The natural gradient computation already applies the preconditioner in the eigenbasis.
            eigvecs_cpu = eigvecs.cpu().float()
            eigvals_cpu = eigvals_clipped.cpu().float()
            return {
                'eigvecs': eigvecs_cpu,
                'eigvals': eigvals_cpu
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

        Natural gradient: (G+λI)^(-1) * grad * A^(-1)
        
        Supports both eigendecomp and Woodbury representations for G-side.
        """
        result = {}

        # Get factors
        factors = self.kfac_factors[layer_name]
        A_eigvecs = factors['A_eigvecs']  # CPU, float32
        A_eigvals = factors['A_eigvals']  # CPU, float32
        G_type = factors['G_type']
        
        # Bias-consistency assertion (prevent dimension bugs)
        has_bias = module.bias is not None
        A_bias_augmented = factors.get('A_bias_augmented', False)
        if A_bias_augmented != has_bias:
            raise RuntimeError(
                f"A-side augmentation mismatch for layer {layer_name}: "
                f"factors indicate bias_augmented={A_bias_augmented}, "
                f"but module.bias is {'not None' if has_bias else 'None'}. "
                f"This may indicate loading factors from a different checkpoint."
            )
        
        # Determine target device from gradients
        weight_name = f"{layer_name}.weight" if layer_name else "weight"
        if weight_name not in gradients:
            return result
            
        grad = gradients[weight_name]
        target_device = grad.device
        
        # Move A-side to target device
        A_eigvecs = A_eigvecs.to(target_device)
        A_eigvals = A_eigvals.to(target_device)
        bias_name = f"{layer_name}.bias" if layer_name else "bias"
        
        # Prepare gradient (possibly augmented with bias)
        if has_bias:
            # Get bias gradient (use zeros if missing to stay consistent with augmented A)
            if bias_name in gradients:
                bias_grad = gradients[bias_name].unsqueeze(1)
            else:
                # Zero-bias fallback: bias exists but grad is missing
                bias_grad = torch.zeros(
                    grad.shape[0], 1,
                    device=grad.device,
                    dtype=grad.dtype
                )
            # Combine weight and bias gradients
            Y = torch.cat([grad, bias_grad], dim=1)  # [out, in+1]
        else:
            Y = grad  # [out, in] 
        
        # ===== APPLY G-SIDE INVERSE: (G+λI)^{-1} @ Y =====
        if G_type == 'woodbury_empirical':
            # Woodbury: (G+λI)^{-1} = (1/λ)I - (1/λ)U S^{-1} U^T (1/λ)
            # Result: Y_G = (1/λ) Y - (1/λ) U @ (S^{-1} @ (U^T @ ((1/λ) Y)))
            U = factors['U'].to(device=target_device, non_blocking=True)  # [out, T], fp16/bf16
            S_inv = factors['S_inv'].to(device=target_device, non_blocking=True)  # [T, T], fp32
            lambda_G = factors['lambda_G']
            
            lambda_inv = 1.0 / lambda_G
            Y0 = (lambda_inv * Y).float()  # [out, in(+1)], promote to fp32
            
            # Woodbury correction
            Z = U.t().float() @ Y0  # [T, in(+1)]
            W = S_inv @ Z  # [T, in(+1)]
            Y_G = Y0 - lambda_inv * (U.float() @ W)  # [out, in(+1)]
            
        elif G_type == 'eig':
            # Traditional eigendecomp: (G+λI)^{-1} = V diag(1/λ) V^T
            G_eigvecs = factors['G_eigvecs'].to(target_device)
            G_eigvals = factors['G_eigvals'].to(target_device)
            
            tmp = G_eigvecs.T @ Y  # [out, in(+1)]
            tmp = tmp / G_eigvals.unsqueeze(1)  # broadcast divide
            Y_G = G_eigvecs @ tmp  # [out, in(+1)]
        
        else:
            raise ValueError(f"Unknown G_type: {G_type}")
        
        # ===== APPLY A-SIDE INVERSE: Y_G @ A^{-1} =====
        # A^{-1} = V_A diag(1/λ_A) V_A^T (same for all G_types)
        tmp = Y_G @ A_eigvecs  # [out, dim_A]
        tmp = tmp / A_eigvals  # broadcast divide
        Y_nat = tmp @ A_eigvecs.T  # [out, in(+1)]
        
        # ===== SPLIT BACK INTO WEIGHT AND BIAS =====
        if has_bias:
            result[weight_name] = (Y_nat[:, :-1] * scale).to(grad.dtype)
            if bias_name in gradients:  # Only return bias grad if it was present
                result[bias_name] = (Y_nat[:, -1] * scale).to(grad.dtype)
        else:
            result[weight_name] = (Y_nat * scale).to(grad.dtype)

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
        
        Note: For Woodbury layers, only power ∈ {-1, 1} are supported exactly.
        Other powers raise NotImplementedError or fall back to eigendecomp if available.
        """
        # Special cases: power == -1 (natural gradient) or power == 1 (FVP)
        if power == -1.0:
            return self.compute_natural_gradient(gradients, model)
        elif power == 1.0:
            return self.compute_fisher_vector_product(gradients, scale=1.0, model=model)
        
        powered_grads = {}

        for name, module in model.named_modules():
            if not isinstance(module, nn.Linear):
                continue

            if name in self.kfac_factors:
                # Get factors
                factors = self.kfac_factors[name]
                G_type = factors['G_type']
                
                # Check if Woodbury layer with unsupported power
                if G_type == 'woodbury_empirical':
                    logger.warning(
                        f"Powered natural gradient (power={power}) not supported for Woodbury layers like {name}; "
                        "falling back to original gradient"
                    )
                    weight_name = f"{name}.weight" if name else "weight"
                    bias_name = f"{name}.bias" if name else "bias"
                    if weight_name in gradients:
                        powered_grads[weight_name] = gradients[weight_name]
                    if bias_name in gradients:
                        powered_grads[bias_name] = gradients[bias_name]
                    continue
                
                # Eigendecomp path (small layers)
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
                    # Apply power to eigenvalues (already have damping from stabilization)
                    eigvals_A_power = A_eigvals ** power
                    eigvals_G_power = G_eigvals ** power

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

                            # Apply powered transformation in eigenspace: G^power * grad * A^power
                            # G^power * grad = V_G * diag(λ_G^power) * V_G^T * grad
                            tmp = G_eigvecs.T @ grad_combined  # [dim_G, in+1]
                            tmp = tmp * eigvals_G_power.unsqueeze(1)  # broadcast multiply
                            tmp = G_eigvecs @ tmp  # [out, in+1]
                            
                            # tmp * A^power = tmp * V_A * diag(λ_A^power) * V_A^T
                            tmp2 = tmp @ A_eigvecs  # [out, dim_A]
                            tmp2 = tmp2 * eigvals_A_power  # broadcast multiply
                            nat_grad_combined = tmp2 @ A_eigvecs.T  # [out, in+1]

                            # Split back into weight and bias
                            powered_grads[weight_name] = nat_grad_combined[:, :-1]
                            powered_grads[bias_name] = nat_grad_combined[:, -1]
                        else:
                            # Weight only - use only weight dimensions of A
                            in_features = grad_weight.shape[1]
                            
                            # Apply powered transformation in eigenspace
                            # G^power * grad
                            tmp = G_eigvecs.T @ grad_weight
                            tmp = tmp * eigvals_G_power.unsqueeze(1)
                            tmp = G_eigvecs @ tmp
                            
                            # tmp * A^power (truncate to weight dimensions)
                            A_eigvecs_trunc = A_eigvecs[:in_features, :]
                            tmp2 = tmp @ A_eigvecs_trunc
                            tmp2 = tmp2 * eigvals_A_power
                            nat_grad = tmp2 @ A_eigvecs_trunc.T
                            
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
        Compute Fisher-vector product using KFAC: (G + λI) * v * (A + λI)
        
        Note: This includes damping (for trust regions, etc.). For undamped Fisher,
        subtract λ*v from the result.

        Args:
            vector: Vector to multiply with Fisher
            scale: Scaling factor
            model: Optional model for layer structure info

        Returns:
            Fisher-vector product
        """
        fvp = {}

        for layer_name, factors in self.kfac_factors.items():
            # Get A-side factors
            A_eigvecs = factors['A_eigvecs']  # CPU, float32
            A_eigvals = factors['A_eigvals']  # CPU, float32 (already has damping)
            G_type = factors['G_type']
            
            # Determine target device from vector
            weight_name = f"{layer_name}.weight" if layer_name else "weight"
            if weight_name not in vector:
                continue
                
            v_weight = vector[weight_name]
            target_device = v_weight.device
            
            # Move A-side to target device
            A_eigvecs = A_eigvecs.to(target_device)
            A_eigvals = A_eigvals.to(target_device)
            
            # Prepare vector (possibly augmented with bias)
            bias_name = f"{layer_name}.bias" if layer_name else "bias"
            has_bias = bias_name in vector
            
            if has_bias:
                v_bias = vector[bias_name].unsqueeze(1)
                v = torch.cat([v_weight, v_bias], dim=1)  # [out, in+1]
            else:
                v = v_weight  # [out, in]
            
            # ===== APPLY G-SIDE: (G + λI) @ v =====
            if G_type == 'woodbury_empirical':
                # Woodbury forward: (G + λI) v = λv + U U^T v
                U = factors['U'].to(device=target_device, non_blocking=True)  # [out, T], fp16/bf16
                lambda_G = factors['lambda_G']
                
                v_G = (lambda_G * v).float()  # [out, in(+1)]
                v_G = v_G + U.float() @ (U.t().float() @ v.float())  # Add U U^T v
                
            elif G_type == 'eig':
                # Traditional eigendecomp: (G + λI) v = V diag(λ) V^T v
                G_eigvecs = factors['G_eigvecs'].to(target_device)
                G_eigvals = factors['G_eigvals'].to(target_device)
                
                tmp = G_eigvecs.T @ v  # [out, in(+1)]
                tmp = tmp * G_eigvals.unsqueeze(1)  # broadcast multiply
                v_G = G_eigvecs @ tmp  # [out, in(+1)]
            
            else:
                raise ValueError(f"Unknown G_type: {G_type}")
            
            # ===== APPLY A-SIDE: v_G @ (A + λI) =====
            # (A + λI) uses eigenvalues that already include damping
            tmp = v_G @ A_eigvecs  # [out, dim_A]
            tmp = tmp * A_eigvals  # broadcast multiply
            fv = tmp @ A_eigvecs.T  # [out, in(+1)]
            
            # ===== SPLIT BACK INTO WEIGHT AND BIAS =====
            if has_bias:
                fvp[weight_name] = (fv[:, :-1] * scale).to(v_weight.dtype)
                fvp[bias_name] = (fv[:, -1] * scale).to(v_weight.dtype)
            else:
                fvp[weight_name] = (fv * scale).to(v_weight.dtype)

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
        
        **LEGACY METHOD**: This method expects old-style dense A/G matrices.
        For new Woodbury-based factors, use get_fisher_scaled_gradient instead.

        Args:
            grad: Gradient tensor
            fisher_factors: Dictionary with 'A' and 'G' dense covariance matrices
            power: Power to raise Fisher to (e.g., -0.5 for normalization)

        Returns:
            Transformed gradient (or original if factors unavailable)
        """
        if 'A' not in fisher_factors or 'G' not in fisher_factors:
            logger.debug("Legacy _apply_fisher_power: A or G missing; returning original gradient")
            return grad

        # Get factors (may be on CPU) and move to gradient device
        A = fisher_factors['A']
        G = fisher_factors['G']
        
        if grad.device != A.device:
            A = A.to(grad.device)
            G = G.to(grad.device)

        try:
            # Eigendecomposition for powered matrices (stay in eigenspace to avoid torch.diag)
            eigvals_A, eigvecs_A = torch.linalg.eigh(A)
            eigvals_G, eigvecs_G = torch.linalg.eigh(G)

            # Apply power (eigenvalues should already include damping if from recent code)
            eigvals_A_power = eigvals_A ** power
            eigvals_G_power = eigvals_G ** power

            # Reshape gradient if needed
            orig_shape = grad.shape
            if grad.dim() == 1:
                # Assume it's a flattened weight matrix
                out_features = G.shape[0]
                in_features = A.shape[0]
                if grad.numel() == out_features * in_features:
                    grad = grad.reshape(out_features, in_features)
                else:
                    # Can't reshape, return original
                    return grad

            # Apply transformation in eigenspace: G^power * grad * A^power
            # G^power * grad
            tmp = eigvecs_G.T @ grad
            tmp = tmp * eigvals_G_power.unsqueeze(1)
            tmp = eigvecs_G @ tmp
            
            # tmp * A^power
            tmp2 = tmp @ eigvecs_A
            tmp2 = tmp2 * eigvals_A_power
            result = tmp2 @ eigvecs_A.T

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
