#!/usr/bin/env python3
"""
Advanced Fisher Collector with True Fisher, K-FAC, and Capacity Metrics.

This module extends the basic FisherCollector with:
1. True Fisher sampling from model distribution
2. K-FAC block-diagonal approximation
3. Eigenvalue-based capacity metrics
4. Natural gradient approximation

Fisher Information Types:
- Empirical Fisher: F = E[∇log p(y_data|x,θ) * ∇log p(y_data|x,θ)^T]
  Uses ground-truth labels from the training data.
  This is what the basic FisherCollector computes.

- True Fisher: F = E[∇log p(y|x,θ) * ∇log p(y|x,θ)^T] where y ~ p(y|x,θ)
  Samples labels from the model's predictive distribution.
  More theoretically principled but computationally expensive.
  This is available via collect_true_fisher() when use_true_fisher=True.

Note: Both are approximations. True Fisher with model sampling is closer
to the actual Fisher Information Matrix but requires multiple forward passes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple, Union, Any
from collections import defaultdict
from .fisher_collector import FisherCollector

logger = logging.getLogger(__name__)


class AdvancedFisherCollector(FisherCollector):
    """
    Advanced Fisher Collector with theoretical improvements.

    Features:
    - True Fisher via sampling from model distribution
    - K-FAC block-diagonal approximation
    - Eigenvalue-based capacity metrics
    - Natural gradient computation
    """

    def __init__(
        self,
        reduction: str = 'group',
        storage: str = 'cpu_fp16',
        ema_decay: float = 0.99,
        use_ewc: bool = False,
        debug: bool = False,
        use_true_fisher: bool = False,
        use_kfac: bool = False,
        kfac_update_freq: int = 10,
        damping: float = 1e-4,
        kfac_show_progress: bool = False
    ):
        """
        Initialize Advanced Fisher Collector.

        Args:
            reduction: Reduction type ('param', 'group')
            storage: Storage strategy ('gpu', 'cpu', 'cpu_fp16')
            ema_decay: EMA decay rate
            use_ewc: Whether to use EWC-style Fisher
            debug: Debug mode
            use_true_fisher: Use true Fisher (sample from model)
            use_kfac: Use K-FAC approximation
            kfac_update_freq: How often to update K-FAC factors
            damping: Damping factor for natural gradient
        """
        super().__init__(reduction, storage, ema_decay, use_ewc, debug)

        self.use_true_fisher = use_true_fisher
        self.use_kfac = use_kfac
        self.kfac_update_freq = kfac_update_freq
        self.damping = damping

        # K-FAC handler - use centralized implementation with task support
        if use_kfac:
            from fisher.kfac_utils import KFACNaturalGradient
            self.kfac_handler = KFACNaturalGradient(
                damping=damping,
                ema_decay=ema_decay,
                update_freq=kfac_update_freq,
                use_gpu_eigh=True,
                show_progress=kfac_show_progress
            )
            # Task-specific KFAC factors: {task_name: {layer_name: factors}}
            self.kfac_factors_by_task = {}
            # Keep backward compatibility with single task
            self.kfac_factors = self.kfac_handler.kfac_factors
            self.kfac_update_count = 0
            self.kfac_inv_cache = self.kfac_handler.inv_cache
        else:
            self.kfac_handler = None
            self.kfac_factors_by_task = {}
            self.kfac_factors = {}
            self.kfac_update_count = 0
            self.kfac_inv_cache = {}

        # Capacity metrics storage
        self.capacity_metrics = {}

        logger.info(f"AdvancedFisherCollector initialized: true_fisher={use_true_fisher}, kfac={use_kfac}")

    # ============= TRUE FISHER IMPLEMENTATION =============

    def collect_true_fisher(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        task: str = 'default',
        n_samples: int = 1,
        temperature: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Collect true Fisher by sampling from model distribution.

        True Fisher: F = E[∇log p(y|x,θ) * ∇log p(y|x,θ)^T]
        where y ~ p(y|x,θ) (sampled from model)

        Args:
            model: Model to compute Fisher for
            batch: Input batch
            task: Task name for storage
            n_samples: Number of samples per input
            temperature: Temperature for sampling

        Returns:
            Dictionary of Fisher values
        """
        model.eval()  # Important: use eval mode for sampling (no dropout)

        # Move batch to model device
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # Get model predictions
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits

            # Apply temperature
            logits = logits / temperature

            # Get vocabulary size
            vocab_size = logits.shape[-1]
            batch_size, seq_len = logits.shape[:2]

        # Count active tokens for normalization (FIX: was missing)
        if 'attention_mask' in batch:
            active_tokens = batch['attention_mask'].sum().item()
        else:
            active_tokens = batch['input_ids'].numel()

        # Collect Fisher over multiple samples
        fisher_accum = defaultdict(lambda: 0)

        for _ in range(n_samples):
            # Sample from model distribution
            with torch.no_grad():
                # Reshape for sampling
                logits_flat = logits.view(-1, vocab_size)
                probs = F.softmax(logits_flat, dim=-1)

                # Sample from categorical distribution
                sampled = torch.multinomial(probs, num_samples=1)
                sampled_labels = sampled.view(batch_size, seq_len)

                # Apply attention mask to sampled labels (ignore padding)
                if 'attention_mask' in batch:
                    sampled_labels = sampled_labels.masked_fill(
                        batch['attention_mask'] == 0, -100
                    )

            # Create batch with sampled labels
            sampled_batch = batch.copy()
            sampled_batch['labels'] = sampled_labels

            # Compute gradients with sampled labels
            model.zero_grad()
            # Keep model in eval mode (no dropout during Fisher computation)
            # Gradients still flow in eval mode

            outputs = model(**sampled_batch)
            loss = outputs.loss
            loss.backward()

            # Accumulate squared gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Get group-reduced Fisher
                    group_fisher, group_type, num_groups = self._reduce_to_groups(
                        name, param.grad, param.shape, model
                    )

                    # Create key
                    key = self._make_key(task, name, group_type)

                    # FIX: Normalize by both n_samples AND active tokens for comparability
                    fisher_accum[key] += group_fisher / (n_samples * max(1, active_tokens))

        # Store in appropriate location
        if self.use_kfac:
            # Update K-FAC factors if using K-FAC
            self._update_kfac_factors(model, batch, fisher_accum)

        # Store Fisher
        for key, value in fisher_accum.items():
            if key in self.fisher_ema:
                # Update EMA
                self.fisher_ema[key] = (
                    self.ema_decay * self.fisher_ema[key] +
                    (1 - self.ema_decay) * value
                )
            else:
                self.fisher_ema[key] = value

        return dict(fisher_accum)

    # ============= K-FAC IMPLEMENTATION =============

    def _update_kfac_factors(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        fisher_grads: Optional[Dict] = None,
        task_name: Optional[str] = None
    ):
        """
        Update K-FAC factors using centralized implementation.

        K-FAC approximates Fisher as: F ≈ A ⊗ G
        where A = E[a*a^T] (input activation covariance)
              G = E[g*g^T] (pre-activation gradient covariance)
              
        Args:
            model: Neural network model
            batch: Input batch
            fisher_grads: Optional precomputed gradients
            task_name: Optional task identifier for task-specific factors
        """
        if self.kfac_handler is None:
            logger.warning("K-FAC not enabled. Set use_kfac=True in constructor.")
            return

        # Use centralized KFAC implementation
        self.kfac_handler.collect_kfac_factors(model, batch)

        # Store task-specific factors if task_name provided
        if task_name is not None:
            self.kfac_factors_by_task[task_name] = self.kfac_handler.kfac_factors.copy()
            logger.debug(f"Stored KFAC factors for task: {task_name}")

        # Update references for backward compatibility
        self.kfac_factors = self.kfac_handler.kfac_factors
        self.kfac_inv_cache = self.kfac_handler.inv_cache
        self.kfac_update_count += 1

    def get_kfac_natural_gradient(
        self,
        model: nn.Module,
        compute_grad: bool = True,
        task_name: Optional[str] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute natural gradient using centralized K-FAC implementation.

        Natural gradient: ∇_nat = F^(-1) * ∇
        With K-FAC: F^(-1) ≈ (A^(-1) ⊗ G^(-1))

        Args:
            model: Model with computed gradients
            compute_grad: If True, compute gradients first
            task_name: Optional task identifier for task-specific factors

        Returns:
            Dictionary of natural gradients per parameter
        """
        if self.kfac_handler is None:
            raise ValueError("K-FAC not enabled. Set use_kfac=True in constructor.")

        # Collect gradients from model
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        # Use task-specific factors if available
        if task_name is not None and task_name in self.kfac_factors_by_task:
            # Temporarily switch to task-specific factors
            original_factors = self.kfac_handler.kfac_factors
            self.kfac_handler.kfac_factors = self.kfac_factors_by_task[task_name]
            try:
                result = self.kfac_handler.compute_natural_gradient(gradients, model)
            finally:
                # Restore original factors
                self.kfac_handler.kfac_factors = original_factors
            return result
        else:
            # Use current factors (backward compatibility)
            return self.kfac_handler.compute_natural_gradient(gradients, model)

    # ============= LANCZOS SPECTRUM METHODS =============

    def lanczos_spectrum(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        operator: str = 'ggn',
        k: int = 10,
        max_iters: int = 30,
        seed: int = 42,
        regularization: float = 1e-8,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Compute eigenspectrum using unified Lanczos system.

        Args:
            model: Neural network model
            batch: Input batch
            operator: Operator type ('hessian', 'ggn', 'empirical_fisher', 'kfac')
            k: Number of top eigenvalues to compute
            max_iters: Maximum Lanczos iterations
            seed: Random seed for reproducibility
            regularization: Diagonal regularization for PSD operators
            verbose: Whether to print progress

        Returns:
            Dictionary with eigenvalues and spectral metrics
        """
        # Import unified system
        try:
            from .fisher_lanczos_unified import (
                compute_spectrum, LanczosConfig, create_operator
            )
        except ImportError:
            logger.error("fisher_lanczos_unified not found. Please ensure it's in the same directory.")
            return {'error': 'Unified Lanczos system not available'}

        # Prepare configuration
        config = LanczosConfig(
            k=k,
            max_iters=max_iters,
            seed=seed,
            regularization=regularization,
            dtype_compute=torch.float32,
            dtype_tridiag=torch.float64
        )

        # Prepare batch (ensure proper device)
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()}

        # Special handling for K-FAC operator
        kfac_factors = None
        if operator == 'kfac' or operator == 'kfac_fisher':
            if not self.use_kfac or not self.kfac_factors:
                # Fall back to empirical Fisher
                logger.warning("K-FAC factors not available, using empirical Fisher instead")
                operator = 'empirical_fisher'
            else:
                kfac_factors = self.kfac_factors

        # Compute spectrum
        try:
            results = compute_spectrum(
                model=model,
                batch=batch,
                operator_type=operator,
                config=config,
                kfac_factors=kfac_factors,
                verbose=verbose
            )

            # Add metadata
            results['batch_size'] = batch['input_ids'].shape[0] if 'input_ids' in batch else None
            results['model_params'] = sum(p.numel() for p in model.parameters())

            # Store for later use
            self.last_spectrum = results

            return results

        except Exception as e:
            logger.error(f"Spectrum computation failed: {e}")
            return {
                'error': str(e),
                'operator': operator,
                'k': k
            }

    def compute_hessian_spectrum(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        k: int = 10,
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience method for Hessian spectrum."""
        return self.lanczos_spectrum(model, batch, operator='hessian', k=k, **kwargs)

    def compute_fisher_spectrum(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        k: int = 10,
        use_ggn: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """Convenience method for Fisher spectrum."""
        operator = 'ggn' if use_ggn else 'empirical_fisher'
        return self.lanczos_spectrum(model, batch, operator=operator, k=k, **kwargs)

    # ============= CAPACITY METRICS =============

    def compute_capacity_metrics(
        self,
        task: str = 'default',
        use_kfac: Optional[bool] = None,
        use_spectrum: bool = False,
        model: Optional[nn.Module] = None,
        batch: Optional[Dict[str, torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Compute eigenvalue-based capacity metrics.

        Metrics include:
        - Trace (sum of eigenvalues)
        - Log-determinant (log product of eigenvalues)
        - Effective rank
        - Condition number
        - PAC-Bayes complexity

        Args:
            task: Task name
            use_kfac: Whether to use K-FAC factors (None = auto)
            use_spectrum: Whether to use Lanczos spectrum (requires model and batch)
            model: Model for spectrum computation
            batch: Batch for spectrum computation

        Returns:
            Dictionary of capacity metrics
        """
        metrics = {}

        # Option 1: Use Lanczos spectrum for more accurate metrics
        if use_spectrum and model is not None and batch is not None:
            # Compute spectrum using GGN (PSD) operator for clean metrics
            spectrum = self.compute_fisher_spectrum(model, batch, k=20, use_ggn=True)

            if 'eigenvalues' in spectrum and len(spectrum['eigenvalues']) > 0:
                metrics['method'] = 'lanczos_spectrum'
                metrics['operator'] = spectrum.get('operator', 'ggn')
                metrics['top_eigenvalues'] = spectrum['eigenvalues']
                metrics['max_eigenvalue'] = spectrum.get('max_eigenvalue', 0)
                metrics['spectral_gap'] = spectrum.get('spectral_gap', 0)
                metrics['condition_number'] = spectrum.get('condition_number', float('inf'))
                metrics['effective_rank'] = spectrum.get('effective_rank', 1)

                # Approximate trace and log-det from top eigenvalues
                eigs = np.array(spectrum['eigenvalues'])
                metrics['trace_estimate'] = float(np.sum(eigs))
                metrics['log_det_estimate'] = float(np.sum(np.log(eigs + 1e-12)))
                # PAC-Bayes complexity ~ sqrt(trace(F) / n)
                n_samples = 10000  # TODO: Make this a parameter
                metrics['pac_bayes_complexity'] = float(np.sqrt(metrics['trace_estimate'] / max(n_samples, 1)))

                return metrics

        # Option 2: Use existing K-FAC or diagonal Fisher methods
        if use_kfac is None:
            use_kfac = self.use_kfac and bool(self.kfac_factors)

        if use_kfac and self.kfac_factors:
            # Compute metrics from K-FAC factors
            layer_metrics = {}

            for layer_name, factors in self.kfac_factors.items():
                A = factors['A']
                G = factors['G']

                # Use closed-form formulas for Kronecker product metrics
                # Avoids O(n²) memory for eigenvalue outer product
                try:
                    # Run eigens on CPU float32 to avoid GPU BF16 limitations and large VRAM spikes
                    # Move to CPU first, then convert dtype to avoid temporary GPU memory spike
                    eigvals_A = torch.linalg.eigvalsh(A.cpu().float())
                    eigvals_G = torch.linalg.eigvalsh(G.cpu().float())

                    # Filter positive eigenvalues
                    eigvals_A = eigvals_A[eigvals_A > 1e-8]
                    eigvals_G = eigvals_G[eigvals_G > 1e-8]

                    if len(eigvals_A) > 0 and len(eigvals_G) > 0:
                        # Closed-form metrics for Kronecker product F = G ⊗ A
                        # trace(F) = trace(G) * trace(A)
                        trace = eigvals_G.sum().item() * eigvals_A.sum().item()

                        # log det(F) = dim(A) * log det(G) + dim(G) * log det(A)
                        log_det = (len(eigvals_A) * torch.log(eigvals_G).sum().item() +
                                   len(eigvals_G) * torch.log(eigvals_A).sum().item())

                        # Effective rank: (Σλ)² / Σ(λ²)
                        # For Kronecker: sum_eig = sum(eigvals_G) * sum(eigvals_A)
                        #               sum_sq = sum(eigvals_G²) * sum(eigvals_A²)
                        sum_eig = eigvals_G.sum() * eigvals_A.sum()
                        sum_sq_eig = (eigvals_G ** 2).sum() * (eigvals_A ** 2).sum()
                        eff_rank = (sum_eig ** 2) / (sum_sq_eig + 1e-8)

                        # Condition number: max(F) / min(F) = (max_G * max_A) / (min_G * min_A)
                        cond = (eigvals_G.max() * eigvals_A.max()) / (eigvals_G.min() * eigvals_A.min() + 1e-8)

                        layer_metrics[layer_name] = {
                            'trace': float(trace),
                            'log_det': float(log_det),
                            'effective_rank': float(eff_rank),
                            'condition_number': float(cond),
                            'n_positive_eigenvalues': len(eigvals_A) * len(eigvals_G)
                        }
                except Exception as e:
                    logger.warning(f"Failed to compute eigenvalues for {layer_name}: {e}")
                    continue

            if layer_metrics:
                # Aggregate metrics
                metrics['per_layer'] = layer_metrics
                metrics['total_trace'] = sum(m['trace'] for m in layer_metrics.values())
                metrics['total_log_det'] = sum(m['log_det'] for m in layer_metrics.values())
                metrics['avg_effective_rank'] = np.mean([m['effective_rank'] for m in layer_metrics.values()])
                metrics['max_condition_number'] = max(m['condition_number'] for m in layer_metrics.values())

                # PAC-Bayes complexity bound (McAllester)
                # Complexity ~ sqrt(trace(F) / n) where n is the effective sample size
                # Using a default of 10000 samples - should be passed as parameter
                n_samples = 10000  # TODO: Make this a parameter
                metrics['pac_bayes_complexity'] = np.sqrt(metrics['total_trace'] / max(n_samples, 1))

        else:
            # Compute from diagonal Fisher
            fisher_values = self.get_group_fisher(task, bias_corrected=True)

            if fisher_values:
                all_values = []
                for value in fisher_values.values():
                    if torch.is_tensor(value):
                        all_values.extend(value.cpu().numpy().flatten())

                all_values = np.array(all_values)
                all_values = all_values[all_values > 1e-8]  # Filter near-zero

                if len(all_values) > 0:
                    metrics['trace'] = float(np.sum(all_values))
                    metrics['log_det'] = float(np.sum(np.log(all_values + 1e-8)))

                    # Effective rank for diagonal
                    sum_fisher = np.sum(all_values)
                    sum_sq_fisher = np.sum(all_values ** 2)
                    metrics['effective_rank'] = float((sum_fisher ** 2) / (sum_sq_fisher + 1e-8))

                    metrics['condition_number'] = float(np.max(all_values) / (np.min(all_values) + 1e-8))
                    metrics['n_parameters'] = len(all_values)

                    # PAC-Bayes complexity
                    # Complexity ~ sqrt(trace(F) / n) where n is the effective sample size
                    n_samples = 10000  # TODO: Make this a parameter
                    metrics['pac_bayes_complexity'] = float(np.sqrt(metrics['trace'] / max(n_samples, 1)))

        # Store for later retrieval
        self.capacity_metrics[task] = metrics

        return metrics

    def compute_model_capacity_score(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        task: str = 'capacity_test'
    ) -> float:
        """
        Compute a single capacity score for the model.

        This score correlates with model's ability to fit complex functions
        and generalization capability.

        Args:
            model: Model to evaluate
            batch: Sample batch for Fisher computation
            task: Task name

        Returns:
            Capacity score (higher = more capacity)
        """
        # Collect Fisher (true or empirical based on settings)
        if self.use_true_fisher:
            self.collect_true_fisher(model, batch, task)
        else:
            self.collect_fisher(model, batch, task, mode='oneshot')

        # Compute capacity metrics
        metrics = self.compute_capacity_metrics(task)

        if not metrics:
            return 0.0

        # Combine metrics into single score
        # Use normalized effective rank as primary indicator
        if 'avg_effective_rank' in metrics:
            eff_rank = metrics['avg_effective_rank']
        elif 'effective_rank' in metrics:
            eff_rank = metrics['effective_rank']
        else:
            return 0.0

        # Normalize by model size for fair comparison
        n_params = sum(p.numel() for p in model.parameters())
        normalized_rank = eff_rank / np.sqrt(n_params)

        # Penalize high condition number (indicates poor conditioning)
        if 'condition_number' in metrics or 'max_condition_number' in metrics:
            cond = metrics.get('condition_number', metrics.get('max_condition_number', 1.0))
            conditioning_penalty = 1.0 / (1.0 + np.log10(cond + 1))
        else:
            conditioning_penalty = 1.0

        # Final capacity score
        capacity_score = normalized_rank * conditioning_penalty

        return float(capacity_score)

    # ============= FLATNESS/SHARPNESS METRICS =============

    def compute_loss_landscape_curvature(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        epsilon: float = 0.01,
        n_samples: int = 10
    ) -> Dict[str, float]:
        """
        Estimate loss landscape curvature using Fisher and random perturbations.

        This gives a measure of flatness/sharpness that correlates with generalization.

        Args:
            model: Model to evaluate
            batch: Evaluation batch
            epsilon: Perturbation magnitude
            n_samples: Number of random directions to sample

        Returns:
            Dictionary with curvature metrics
        """
        model.eval()

        # Get original loss
        with torch.no_grad():
            outputs = model(**batch)
            loss_orig = outputs.loss.item()

        # Collect Fisher for importance weighting
        if self.use_true_fisher:
            fisher_dict = self.collect_true_fisher(model, batch, '_temp_curvature')
        else:
            self.compute_oneshot_fisher(model, batch, '_temp_curvature')
            fisher_dict = self.get_group_fisher('_temp_curvature', bias_corrected=False)

        # Sample random directions weighted by Fisher
        losses_perturbed = []

        for _ in range(n_samples):
            # Generate random direction
            direction = {}
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Random Gaussian direction
                    d = torch.randn_like(param)

                    # Weight by Fisher importance (if available)
                    # Try to find matching key in fisher_dict
                    fisher_matched = False
                    for group_type in ['channel', 'head', 'row', 'param', 'bias']:
                        key = self._make_key('_temp_curvature', name, group_type)
                        if key in fisher_dict:
                            fisher_values = fisher_dict[key]
                            # For group-reduced Fisher, broadcast to parameter shape
                            if fisher_values.numel() < d.numel():
                                # Broadcast group values to full parameter
                                if 'weight' in name and len(param.shape) >= 2:
                                    # Assume first dimension is output channels/heads
                                    fisher_values = fisher_values.view(-1, *([1]*(len(param.shape)-1)))
                                    fisher_values = fisher_values.expand_as(d)
                            if fisher_values.shape == d.shape:
                                # Element-wise weighting
                                d = d * torch.sqrt(fisher_values + 1e-8)
                                fisher_matched = True
                                break

                    # Normalize direction
                    d = d / (d.norm() + 1e-8)
                    direction[name] = d

            # Compute directional norm
            dir_norm = sum(d.norm() ** 2 for d in direction.values()) ** 0.5

            # Apply perturbation
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in direction:
                        param.add_(direction[name], alpha=epsilon / dir_norm)

                # Evaluate perturbed loss
                outputs_pert = model(**batch)
                loss_pert = outputs_pert.loss.item()
                losses_perturbed.append(loss_pert)

                # Restore parameters
                for name, param in model.named_parameters():
                    if name in direction:
                        param.add_(direction[name], alpha=-epsilon / dir_norm)

        # Clean up temporary Fisher
        self.clear_fisher('_temp_curvature')

        # Compute curvature metrics
        losses_perturbed = np.array(losses_perturbed)

        # Average increase in loss (sharpness)
        avg_increase = np.mean(losses_perturbed - loss_orig)

        # Maximum increase (worst-case sharpness)
        max_increase = np.max(losses_perturbed - loss_orig)

        # Variance of perturbations (landscape roughness)
        variance = np.var(losses_perturbed)

        # Effective curvature (second-order approximation)
        # L(θ + εd) ≈ L(θ) + ε²/2 * d^T H d
        # So curvature ≈ 2 * ΔL / ε²
        effective_curvature = 2 * avg_increase / (epsilon ** 2)

        return {
            'average_sharpness': float(avg_increase),
            'max_sharpness': float(max_increase),
            'landscape_variance': float(variance),
            'effective_curvature': float(effective_curvature),
            'original_loss': float(loss_orig),
            'epsilon': epsilon,
            'n_samples': n_samples
        }

    # ============= OVERRIDE PARENT METHODS =============

    def collect_fisher(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        task: str = 'default',
        mode: str = 'ema',
        **kwargs
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Override parent to use true Fisher if enabled.
        """
        if self.use_true_fisher and mode == 'ema':
            # Use true Fisher for EMA updates
            return self.collect_true_fisher(model, batch, task, **kwargs)
        else:
            # Use parent's empirical Fisher
            return super().collect_fisher(model, batch, task, mode, **kwargs)

    def update_fisher_ema(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        task: str = 'default'
    ):
        """
        Override to use true Fisher if enabled.
        """
        if self.use_true_fisher:
            self.collect_true_fisher(model, batch, task)
        else:
            super().update_fisher_ema(model, batch, task)

    # ============= ANALYSIS METHODS =============

    def analyze_fisher_spectrum(
        self,
        task: str = 'default'
    ) -> Dict[str, Any]:
        """
        Analyze the spectrum of Fisher information.

        Returns statistics about eigenvalue distribution which
        indicates learning dynamics and capacity utilization.
        """
        if self.use_kfac and self.kfac_factors:
            # Analyze K-FAC spectrum
            spectrum_stats = {}

            for layer_name, factors in self.kfac_factors.items():
                A = factors['A']
                G = factors['G']

                try:
                    # Get eigenvalues
                    eigvals_A = torch.linalg.eigvalsh(A).cpu().numpy()
                    eigvals_G = torch.linalg.eigvalsh(G).cpu().numpy()

                    # Analyze spectrum
                    for name, eigvals in [('activation', eigvals_A), ('gradient', eigvals_G)]:
                        eigvals = eigvals[eigvals > 1e-8]  # Filter near-zero

                        if len(eigvals) > 0:
                            spectrum_stats[f"{layer_name}_{name}"] = {
                                'max_eigenvalue': float(np.max(eigvals)),
                                'min_eigenvalue': float(np.min(eigvals)),
                                'mean_eigenvalue': float(np.mean(eigvals)),
                                'spectral_norm': float(np.max(np.abs(eigvals))),
                                'nuclear_norm': float(np.sum(eigvals)),
                                'log_determinant': float(np.sum(np.log(eigvals + 1e-8))),
                                'n_significant': int(np.sum(eigvals > np.max(eigvals) * 0.01))
                            }
                except Exception as e:
                    logger.warning(f"Failed to analyze spectrum for {layer_name}: {e}")

            return spectrum_stats

        else:
            # Analyze diagonal Fisher spectrum
            fisher_values = self.get_group_fisher(task, bias_corrected=True)

            if not fisher_values:
                return {}

            all_values = []
            for value in fisher_values.values():
                if torch.is_tensor(value):
                    all_values.extend(value.cpu().numpy().flatten())

            all_values = np.array(all_values)
            all_values = all_values[all_values > 1e-8]

            if len(all_values) == 0:
                return {}

            # Compute spectrum statistics
            return {
                'diagonal_fisher': {
                    'max_value': float(np.max(all_values)),
                    'min_value': float(np.min(all_values)),
                    'mean_value': float(np.mean(all_values)),
                    'median_value': float(np.median(all_values)),
                    'std_value': float(np.std(all_values)),
                    'skewness': float(self._compute_skewness(all_values)),
                    'kurtosis': float(self._compute_kurtosis(all_values)),
                    'n_significant': int(np.sum(all_values > np.max(all_values) * 0.01)),
                    'sparsity': float(np.mean(all_values < np.mean(all_values) * 0.1))
                }
            }

    def _compute_skewness(self, values: np.ndarray) -> float:
        """Compute skewness of distribution."""
        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-8:
            return 0.0
        return np.mean(((values - mean) / std) ** 3)

    def _compute_kurtosis(self, values: np.ndarray) -> float:
        """Compute kurtosis of distribution."""
        mean = np.mean(values)
        std = np.std(values)
        if std < 1e-8:
            return 0.0
        return np.mean(((values - mean) / std) ** 4) - 3.0


# ============= TESTING =============

class TestLMModel(nn.Module):
    """Test model that accepts LM-style inputs."""

    def __init__(self, vocab_size=10, hidden_dim=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, vocab_size)
        self.vocab_size = vocab_size

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Embed and pool
        x = self.embedding(input_ids)  # (batch, seq, hidden)
        x = x.mean(dim=1)  # Pool over sequence

        # MLP
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)

        # Expand logits to match sequence length
        seq_len = input_ids.shape[1]
        logits = logits.unsqueeze(1).expand(-1, seq_len, -1)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, self.vocab_size),
                labels.reshape(-1),
                ignore_index=-100
            )

        # Return LM-style output
        class Output:
            pass
        output = Output()
        output.loss = loss
        output.logits = logits
        return output


def test_advanced_fisher():
    """Test advanced Fisher functionality."""
    import torch

    # Create LM-style test model
    model = TestLMModel(vocab_size=10, hidden_dim=20)

    # Create collector
    collector = AdvancedFisherCollector(
        use_true_fisher=True,
        use_kfac=True,
        kfac_update_freq=1
    )

    # Create dummy batch
    batch = {
        'input_ids': torch.randint(0, 10, (4, 8)),
        'attention_mask': torch.ones(4, 8),
        'labels': torch.randint(0, 10, (4, 8))
    }

    # Test true Fisher collection
    print("Testing true Fisher collection...")
    fisher = collector.collect_true_fisher(model, batch, 'test', n_samples=3)
    print(f"Collected {len(fisher)} Fisher groups")

    # Test capacity metrics
    print("\nTesting capacity metrics...")
    metrics = collector.compute_capacity_metrics('test')
    print(f"Capacity metrics: {metrics}")

    # Test curvature estimation
    print("\nTesting loss landscape curvature...")
    curvature = collector.compute_loss_landscape_curvature(model, batch)
    print(f"Curvature: {curvature}")

    # Test spectrum analysis
    print("\nTesting Fisher spectrum analysis...")
    spectrum = collector.analyze_fisher_spectrum('test')
    print(f"Spectrum stats: {spectrum}")

    print("\n✅ All tests passed!")


if __name__ == "__main__":
    test_advanced_fisher()
