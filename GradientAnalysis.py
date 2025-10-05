#!/usr/bin/env python3
"""
Consolidated Gradient Analysis Module
======================================
All gradient diagnostic and analysis functions in one place.
Organized by use case for easy discovery and usage.
AUDITED
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import inspect
import sys
from typing import Dict, Any, Optional, List, Tuple, Union
from collections import defaultdict
import warnings
import logging
from scipy import stats
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

# Import GPU memory tracker
try:
    from gpu_memory_tracker import get_tracker, log_memory_state
except ImportError:
    # Fallback if tracker not available
    def get_tracker():
        return None
    def log_memory_state(msg=""):
        pass

logger = logging.getLogger(__name__)


class GradientAnalysis:
    """
    Consolidated gradient diagnostics and analysis tools.

    IMPORTANT: Why GradientAnalysis gradients are NOT redundant:
    =============================================================
    While fisher_collector.py computes gradients for Fisher Information Matrix
    estimation (squared gradients averaged over many samples), GradientAnalysis
    computes gradients for DIFFERENT diagnostic purposes:

    1. Fisher gradients (fisher_collector.py):
       - Purpose: Estimate parameter importance via E[∇L(θ)²]
       - Uses: EWC regularization, pruning, uncertainty estimation
       - Samples: 768+ samples for statistical validity

    2. GradientAnalysis gradients (THIS module):
       - Purpose: Diagnose training dynamics and task conflicts
       - Uses: Detect vanishing/exploding gradients, measure task interference
       - Samples: Often just 2-4 samples for quick diagnostics
       - Key difference: Needs RAW gradients (not squared) for:
         * Cosine similarity between task gradients
         * Gradient flow analysis through layers
         * Stability checks (different batch pairs)
         * Dispersion analysis (gradient variance)

    These are fundamentally different computational needs that require
    separate gradient computations with different batch configurations.

    Complete Function Overview (8 Public Functions):
    ================================================
    
    1. DIAGNOSTIC ANALYSIS (Training Health):
    -----------------------------------------
    • compute_gradient_pathology: Detect vanishing/exploding gradients
      - Analyzes gradient magnitudes across layers
      - Returns percentile-based health metrics
      - Use when: Training stuck, loss not decreasing
      - Output: Dict with vanishing/exploding layer counts
    
    2. CONFLICT MEASUREMENT (Task Interference - 5 functions):
    ----------------------------------------------------------
    • compute_gradient_conflict_pcgrad: PCGrad-based layer-wise conflict
      - Based on Yu et al. 2020 PCGrad paper
      - Returns both global and per-layer conflicts
      - Use when: Need layer-by-layer conflict breakdown
      - Output: Dict with global_conflict and layer_conflicts
    
    • compute_raw_gradient_conflict: Statistical global conflict
      - Resamples gradients for variance estimate
      - Memory-efficient layerwise processing option
      - Use when: Need robust conflict measure with uncertainty
      - Output: Scalar mean ± std deviation
    
    • compute_gradient_conflict_pair: Fine-grained parameter similarities
      - Per-parameter cosine similarities (was conflict_matrix)
      - Highest memory usage but most detailed
      - Use when: Need parameter-level conflict analysis
      - Output: Dict with per-parameter similarities

    • compute_gradient_conflict_matrix_multi: Future placeholder
      - Will compute true NxN conflict matrix for N tasks
      - Not yet implemented
    
    3. ALIGNMENT ANALYSIS (Spatial & Temporal - 2 functions):
    ---------------------------------------------------------
    • compute_layer_gradient_alignment: WHERE conflicts occur (spatial)
      - Identifies which layers have gradient conflicts
      - Single snapshot analysis
      - Use when: Deciding which layers to freeze/adapt
      - Output: Dict with per-layer alignments + worst layer
    
    • compute_gradient_alignment_trajectory: WHEN conflicts occur (temporal)
      - Tracks alignment evolution over training
      - Supports multiple checkpoints/batches
      - Use when: Monitoring training stability/convergence
      - Output: Time-series with trends and stability metrics
    
    KEY DIFFERENCES SUMMARY:
    -----------------------
    Granularity:
    - Global: raw_gradient_conflict, gradient_pathology
    - Per-layer: layer_gradient_alignment, gradient_conflict_pcgrad
    - Per-parameter: gradient_conflict_pair
    - Time-series: gradient_alignment_trajectory
    
    Memory Usage (Low to High):
    - raw_gradient_conflict (with layerwise=True)
    - gradient_pathology
    - layer_gradient_alignment
    - gradient_conflict_pcgrad
    - gradient_alignment_trajectory
    - gradient_conflict_pair
    
    Output Types:
    - Scalars: raw_gradient_conflict
    - Layer maps: layer_gradient_alignment, gradient_conflict_pcgrad
    - Parameter maps: gradient_conflict_pair
    - Time series: gradient_alignment_trajectory
    - Health metrics: gradient_pathology
    
    PRACTICAL USAGE GUIDE:
    ---------------------
    For debugging training issues:
    → Start with compute_gradient_pathology
    
    For multi-task learning conflicts:
    → Use compute_raw_gradient_conflict for quick check
    → Use compute_layer_gradient_alignment to identify problem layers
    → Use compute_gradient_conflict_pair for detailed parameter analysis
    
    For monitoring training dynamics:
    → Use compute_gradient_alignment_trajectory
    
    They're complementary - combine multiple functions for complete analysis!
    """
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu', computation_dtype=None):
        self.device = device
        self.computation_dtype = computation_dtype  # For numerical stability

    def _get_optimal_batch_configs(self, force_conservative: bool = False) -> List[Tuple]:
        """
        Return STATIC batch configurations for reproducibility.
        NO dynamic GPU detection - always use same configs for conference submission.

        Args:
            force_conservative: Ignored (kept for backward compatibility)

        Returns:
            List of (batch_size, seq_len, weight, name) tuples - ALWAYS THE SAME
        """
        # STATIC CONFIGURATION - no GPU detection, no dynamic adjustment
        # These conservative configs work on most GPUs and ensure reproducibility

        STATIC_CONFIGS = [
            (32, 256, 0.30, "tokens_0_to_256"),    # Short sequences, fast - increased for better GPU utilization
            (8, 1024, 0.20, "tokens_0_to_1024"),  # Standard sequences
            (3, 2048, 0.15, "tokens_0_to_2048")     # Long sequences - reduced to 3 to avoid OOM
        ]
        return STATIC_CONFIGS

    def _get_optimal_subsample_ratio(self, batch_size: int, min_samples: int = 32) -> float:
        """
        Calculate optimal subsample ratio based on batch size and statistical requirements.

        Args:
            batch_size: Current batch size
            min_samples: Minimum samples needed for stable gradient estimates (default 32)

        Returns:
            Optimal subsample_ratio between 0.0 and 1.0
        """
        if batch_size >= min_samples * 2:
            # Large batch: can afford to subsample for variance reduction
            return 0.5
        elif batch_size >= min_samples:
            # Medium batch: use most of it
            return 0.75
        else:
            # Small batch: use all samples
            return 1.0

    # ============= HELPER METHODS =============

    def apply_fdr_correction(self, p_values: Union[List[float], np.ndarray],
                            alpha: float = 0.05,
                            method: str = 'benjamini-hochberg') -> Dict[str, Any]:
        """
        Apply False Discovery Rate (FDR) correction for multiple comparisons.
        Critical for ICLR 2026 publication statistical validity.

        Args:
            p_values: List or array of p-values from multiple tests
            alpha: Desired false discovery rate (default 0.05)
            method: FDR method ('benjamini-hochberg' or 'bonferroni')

        Returns:
            Dict containing:
            - corrected_p_values: FDR-adjusted p-values
            - significant: Boolean mask of significant results
            - threshold: Significance threshold used
            - n_significant: Number of significant results after correction

        Note:
            Benjamini-Hochberg controls the expected proportion of false discoveries
            among rejected hypotheses, more powerful than Bonferroni for many tests.
        """
        p_values = np.array(p_values)
        n_tests = len(p_values)

        if n_tests == 0:
            return {
                'corrected_p_values': np.array([]),
                'significant': np.array([]),
                'threshold': alpha,
                'n_significant': 0
            }

        if method == 'benjamini-hochberg':
            # Sort p-values and keep track of original order
            sorted_idx = np.argsort(p_values)
            sorted_p = p_values[sorted_idx]

            # Calculate BH thresholds
            bh_thresholds = alpha * np.arange(1, n_tests + 1) / n_tests

            # Find largest i where P(i) <= threshold
            significant_sorted = sorted_p <= bh_thresholds
            if np.any(significant_sorted):
                max_significant_idx = np.max(np.where(significant_sorted)[0])
                threshold = bh_thresholds[max_significant_idx]
            else:
                threshold = 0.0

            # Apply correction
            significant = p_values <= threshold

            # Adjust p-values (BH adjustment)
            corrected_p = np.minimum(1, p_values * n_tests / np.arange(1, n_tests + 1))

        elif method == 'bonferroni':
            # Simple Bonferroni correction
            threshold = alpha / n_tests
            significant = p_values < threshold
            corrected_p = np.minimum(1, p_values * n_tests)
        else:
            raise ValueError(f"Unknown FDR method: {method}")

        return {
            'corrected_p_values': corrected_p,
            'significant': significant,
            'threshold': threshold,
            'n_significant': np.sum(significant),
            'method': method,
            'n_tests': n_tests,
            'alpha': alpha
        }
    
    def _group_parameters_by_layer(self, model) -> Dict[str, List[str]]:
        """
        Groups model parameter names by layer for consistent analysis.
        Centralizes layer parsing logic to avoid duplication.
        
        Supports multiple model architectures:
        - Transformer models: layers.N, layer.N
        - GPT-style: h.N (transformer blocks)
        - BERT-style: encoder.layer.N, decoder.layer.N
        """
        layer_groups = defaultdict(list)
        for name, _ in model.named_parameters():
            layer_found = False

            # BERT-style: encoder.layer.N, decoder.layer.N (check first for specificity)
            if '.layer.' in name and ('encoder' in name or 'decoder' in name):
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layer' and i + 1 < len(parts) and parts[i+1].isdigit():
                        layer_idx = int(parts[i+1])
                        prefix = 'encoder' if 'encoder' in name else 'decoder'
                        layer_groups[f'{prefix}_layer_{layer_idx}'].append(name)
                        layer_found = True
                        break

            # Check for general layer patterns
            elif 'layers.' in name or 'layer.' in name:
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part in ['layers', 'layer'] and i + 1 < len(parts) and parts[i+1].isdigit():
                        layer_idx = int(parts[i+1])
                        layer_groups[f'layer_{layer_idx}'].append(name)
                        layer_found = True
                        break

            # GPT-style: h.0, h.1, etc.
            elif '.h.' in name or name.startswith('h.'):
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'h' and i + 1 < len(parts) and parts[i+1].isdigit():
                        layer_idx = int(parts[i+1])
                        layer_groups[f'layer_{layer_idx}'].append(name)
                        layer_found = True
                        break
            
            # If no layer pattern found, categorize by type
            if not layer_found:
                # Use lowercase for case-insensitive matching
                name_lower = name.lower()

                # Check for embedding patterns
                embed_patterns = ['embed', 'wte', 'wpe', 'embed_tokens', 'tok_embeddings',
                                'word_embeddings', 'position_embeddings']
                if any(pattern in name_lower for pattern in embed_patterns):
                    layer_groups['embedding'].append(name)
                # Check for output/head patterns
                elif any(pattern in name_lower for pattern in ['lm_head', 'classifier', 'fc_out',
                                                               'score', 'language_model_head']):
                    layer_groups['output'].append(name)
                # Check for normalization patterns
                elif any(pattern in name_lower for pattern in ['ln_f', 'layer_norm', 'layernorm',
                                                               'final_layernorm', 'final_norm',
                                                               'input_layernorm', 'post_attention_layernorm',
                                                               'rms_norm', 'rmsnorm']):
                    layer_groups['final_norm'].append(name)
                else:
                    layer_groups['other'].append(name)
        
        return dict(layer_groups)
    
    def _transform_to_natural_gradient(
        self,
        grad: torch.Tensor,
        param_name: str,
        kfac_factors: Optional[Dict] = None
    ) -> torch.Tensor:
        """
        Transform gradient to natural gradient space using KFAC or diagonal Fisher.

        Args:
            grad: Raw gradient
            param_name: Parameter name for finding corresponding KFAC factors
            kfac_factors: Dict of KFAC factors by layer

        Returns:
            Natural gradient (or original if transformation fails)
        """
        if kfac_factors is None:
            return grad

        # Try to find KFAC factors for this parameter's layer
        layer_name = '.'.join(param_name.split('.')[:-1]) if '.' in param_name else param_name

        if layer_name in kfac_factors:
            try:
                from fisher.kfac_utils import KFACNaturalGradient

                # Create temporary KFAC handler
                kfac = KFACNaturalGradient()

                # Apply natural gradient transformation
                nat_grad = kfac._apply_fisher_power(
                    grad,
                    kfac_factors[layer_name],
                    power=-1.0  # F^(-1) for natural gradient
                )

                return nat_grad
            except Exception as e:
                logger.debug(f"Natural gradient transformation failed for {param_name}: {e}")

        # Check if we have diagonal Fisher from parent
        if hasattr(self, 'parent') and hasattr(self.parent, 'get_group_fisher'):
            try:
                fisher_dict = self.parent.get_group_fisher(task='default', bias_corrected=True)
                if param_name in fisher_dict:
                    fisher_diag = fisher_dict[param_name]
                    if fisher_diag.shape == grad.shape:
                        # Apply diagonal natural gradient
                        nat_grad = grad / (fisher_diag + 1e-8)
                        return nat_grad
            except Exception as e:
                logger.debug(f"Diagonal Fisher transformation failed for {param_name}: {e}")

        return grad

    def _to_model_device(self, model, batch):
        """Move batch to model's device."""
        try:
            device = next(model.parameters()).device
        except StopIteration:
            # Handle empty models
            device = torch.device('cpu')
        return {k: v.to(device) if torch.is_tensor(v) else v
                for k, v in batch.items()}
    
    def _with_labels(self, batch):
        """Add labels to batch if not present, properly masking padding tokens."""
        if 'labels' not in batch and 'input_ids' in batch:
            batch = dict(batch)
            # Clone input_ids as labels
            labels = batch['input_ids'].clone()

            # Mask padding positions with -100 (ignored in loss calculation)
            # This prevents NaN loss when padding tokens are included
            if 'attention_mask' in batch:
                # Set labels to -100 where attention_mask is 0 (padding positions)
                labels[batch['attention_mask'] == 0] = -100
            else:
                # If no attention mask, try to detect padding tokens
                # Most models use pad_token_id for padding
                pad_token_id = getattr(self, '_pad_token_id', None)
                if pad_token_id is not None:
                    labels[batch['input_ids'] == pad_token_id] = -100
                else:
                    # Log warning if we can't identify padding
                    logger.debug("No attention_mask and no pad_token_id available - labels may include padding")

            batch['labels'] = labels
        return batch
    
    def _take(self, batch, n):
        """Take first n examples from batch."""
        # Create indices on same device as input_ids to avoid device mismatch
        device = batch['input_ids'].device
        indices = torch.randperm(batch['input_ids'].size(0), device=device)[:n]
        return self._slice_batch(batch, indices)
    
    def _slice_batch(self, batch, indices):
        """Slice batch by indices."""
        result = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                # Ensure indices are on same device as tensor
                idx = indices.to(v.device) if indices.device != v.device else indices
                result[k] = v[idx]
            elif isinstance(v, (list, tuple)) and len(v) == len(batch['input_ids']):
                # Handle list/tuple fields with batch dimension
                result[k] = [v[i] for i in indices.cpu().tolist()]
            else:
                # Pass through other values unchanged
                result[k] = v
        return result
    
    def _shuffle_all(self, batch):
        """Shuffle all examples in batch."""
        n = batch['input_ids'].size(0)
        idx = torch.randperm(n, device=batch['input_ids'].device)
        return self._slice_batch(batch, idx)

    def _validate_gradient_computation(self, model, sample_batch=None, max_test_params=100):
        """
        Validate that gradients can be computed properly for a model.

        This is a critical diagnostic function that tests gradient flow through the model
        to catch issues early before running expensive gradient analysis functions.

        Args:
            model: The model to validate
            sample_batch: Optional test batch (will create a small one if not provided)
            max_test_params: Maximum number of parameters to check (for efficiency)

        Returns:
            Dict with validation results:
                - success: bool indicating if gradients flow properly
                - coverage: float percentage of parameters with gradients
                - num_with_grads: int count of parameters with gradients
                - total_params: int total trainable parameters
                - error: str error message if validation failed
                - problematic_layers: list of layer names without gradients
        """
        was_training = model.training
        model.eval()  # Use eval mode for deterministic validation

        try:
            # Count trainable parameters
            total_params = sum(1 for p in model.parameters() if p.requires_grad)
            if total_params == 0:
                return {
                    'success': False,
                    'coverage': 0.0,
                    'num_with_grads': 0,
                    'total_params': 0,
                    'error': 'Model has no trainable parameters (all frozen)'
                }

            # Create a minimal test batch if not provided
            if sample_batch is None:
                device = next(model.parameters()).device
                sample_batch = {
                    'input_ids': torch.randint(0, 1000, (2, 8), device=device),  # Small batch
                    'attention_mask': torch.ones(2, 8, device=device, dtype=torch.long)
                }

            # Ensure batch is on model device
            sample_batch = self._to_model_device(model, sample_batch)
            sample_batch = self._with_labels(sample_batch)

            # Clear any existing gradients
            model.zero_grad(set_to_none=True)

            # Forward pass
            try:
                outputs = model(**sample_batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else None

                if loss is None:
                    # Try to compute loss manually
                    if hasattr(outputs, 'logits'):
                        logits = outputs.logits
                        labels = sample_batch.get('labels', sample_batch['input_ids'])
                        loss_fct = torch.nn.CrossEntropyLoss()

                        if logits.dim() == 3:  # Language model
                            shift_logits = logits[..., :-1, :].contiguous()
                            shift_labels = labels[..., 1:].contiguous()
                            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                        else:
                            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))
                    else:
                        return {
                            'success': False,
                            'coverage': 0.0,
                            'num_with_grads': 0,
                            'total_params': total_params,
                            'error': 'Model returns no loss and no logits for loss computation'
                        }

                # Check loss is valid
                if not loss.requires_grad:
                    return {
                        'success': False,
                        'coverage': 0.0,
                        'num_with_grads': 0,
                        'total_params': total_params,
                        'error': f'Loss does not require gradients (value={loss.item():.4f})'
                    }

                if torch.isnan(loss) or torch.isinf(loss):
                    return {
                        'success': False,
                        'coverage': 0.0,
                        'num_with_grads': 0,
                        'total_params': total_params,
                        'error': f'Loss is invalid (NaN or Inf): {loss.item()}'
                    }

                # Backward pass with FP32 precision guarantee
                with torch.cuda.amp.autocast(enabled=False):
                    if loss.dtype not in [torch.float32, torch.float64]:
                        loss = loss.float()
                    loss.backward()

            except Exception as e:
                return {
                    'success': False,
                    'coverage': 0.0,
                    'num_with_grads': 0,
                    'total_params': total_params,
                    'error': f'Forward/backward pass failed: {str(e)}'
                }

            # Check gradient coverage
            params_with_grads = 0
            params_checked = 0
            problematic_layers = []

            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue

                params_checked += 1
                if param.grad is not None and not torch.all(param.grad == 0):
                    params_with_grads += 1
                else:
                    # Track layers without gradients (only first few to avoid spam)
                    if len(problematic_layers) < 10:
                        layer_name = name.split('.')[0] if '.' in name else name
                        if layer_name not in problematic_layers:
                            problematic_layers.append(layer_name)

                # Stop checking after max_test_params for efficiency
                if params_checked >= max_test_params:
                    # Extrapolate to full model
                    params_with_grads = int(params_with_grads * total_params / params_checked)
                    break

            coverage = (params_with_grads / total_params * 100) if total_params > 0 else 0

            # Determine success
            # We expect at least 90% coverage for healthy gradient flow
            # The 338 parameter bug gives ~0.00002% coverage
            success = coverage >= 90.0

            result = {
                'success': success,
                'coverage': coverage,
                'num_with_grads': params_with_grads,
                'total_params': total_params,
            }

            if not success:
                if coverage < 1.0:
                    result['error'] = f'Critical: Only {coverage:.4f}% parameters have gradients (likely gradient checkpointing bug)'
                elif coverage < 50.0:
                    result['error'] = f'Poor gradient flow: Only {coverage:.1f}% parameters have gradients'
                else:
                    result['error'] = f'Partial gradient flow: {coverage:.1f}% parameters have gradients'

                if problematic_layers:
                    result['problematic_layers'] = problematic_layers

                # Add diagnostic hints
                if coverage < 1.0 and hasattr(model, 'is_gradient_checkpointing'):
                    if getattr(model, 'is_gradient_checkpointing', False):
                        result['hint'] = 'Gradient checkpointing is enabled - try model.gradient_checkpointing_disable()'

            return result

        finally:
            model.train(was_training)
            # Clean up test gradients
            model.zero_grad(set_to_none=True)

    # ============= CORE DIAGNOSTICS (audited) =============

    def compute_gradient_pathology(
        self,
        model,
        batch: Union[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]],
        learning_rate: float = 1e-3,
        n_samples: int = 5,
        compute_snr: bool = True,
        compute_flow_score: bool = True,
        absolute_vanishing_threshold: float = 1e-7,
        absolute_exploding_threshold: float = 100.0,
        memory_efficient: bool = False,
        use_natural_gradient: bool = False,
        kfac_factors: Optional[Dict] = None,
        show_progress: bool = True
    ) -> Dict[str, Any]:
        """
        Improved gradient pathology detection with theoretically sound metrics.

        This version fixes the theoretical issues in the original compute_gradient_pathology:
        1. Uses absolute thresholds instead of percentiles (no circular logic)
        2. Samples multiple batches for statistical robustness
        3. Computes signal-to-noise ratio for gradient reliability
        4. Measures gradient flow through network depth
        5. Considers learning rate in threshold calculations
        6. NEW: Optionally analyzes pathologies in natural gradient space

        Natural gradient analysis reveals true optimization pathologies:
        - Raw gradient vanishing might be fine if Fisher is small (unimportant parameters)
        - Natural gradient explosion indicates true optimization instability
        - Natural gradient SNR shows actual signal quality for optimization

        Args:
            model: Model to analyze
            batch: Single batch or list of batches for multi-sample analysis
            learning_rate: Current learning rate (for threshold scaling)
            n_samples: Number of gradient samples if single batch provided
            compute_snr: Whether to compute signal-to-noise ratio
            compute_flow_score: Whether to compute gradient flow score
            absolute_vanishing_threshold: Base threshold for vanishing gradients
            absolute_exploding_threshold: Base threshold for exploding gradients
            memory_efficient: If True, use Welford's algorithm for SNR (saves memory for large models)
            use_natural_gradient: If True, analyze pathologies in natural gradient space
            kfac_factors: Pre-computed KFAC factors for natural gradient (optional)

        Returns:
            Dict with improved gradient pathology metrics:
                - gradient_statistics: Mean, std, min, max across samples
                - vanishing_parameters: List of parameters with vanishing gradients
                - exploding_parameters: List of parameters with exploding gradients
                - signal_to_noise_ratio: SNR for each parameter (if computed)
                - gradient_flow_score: Score indicating gradient flow health
                - layer_gradient_decay: Gradient norm decay through layers
                - optimization_health_score: Overall health score (0-1)
        """
        was_training = model.training

        # Check current gradient status and warn if most params are frozen
        params_with_grad = sum(1 for p in model.parameters() if p.requires_grad)
        total_params = sum(1 for p in model.parameters())
        grad_coverage = params_with_grad / total_params if total_params > 0 else 0

        if grad_coverage < 0.9:
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_count = sum(p.numel() for p in model.parameters())
            logger.warning(
                f"⚠️ Gradient Warning: Only {params_with_grad}/{total_params} parameters "
                f"({grad_coverage*100:.2f}%) have requires_grad=True. "
                f"That's {param_count:,}/{total_count:,} values. "
                f"Enabling gradients for ALL parameters to ensure proper analysis."
            )

        # CRITICAL: Enable gradients for ALL parameters (pretrained models load with requires_grad=False)
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = True

        model.eval()  # Use eval for deterministic analysis (gradients still work!)

        # Freeze BatchNorm statistics to avoid side effects during diagnostics
        # We want gradients but not running stat updates
        bn_layers = []
        original_momentum = []
        for module in model.modules():
            if isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d,
                                 torch.nn.LayerNorm, torch.nn.GroupNorm)):
                bn_layers.append(module)
                original_momentum.append(module.momentum if hasattr(module, 'momentum') else None)
                if hasattr(module, 'momentum'):
                    module.momentum = 0  # Freeze running stats

        # Validate gradient computation first
        validation = self._validate_gradient_computation(model)
        if not validation['success']:
            # Restore BN momentum and gradient states before returning
            for module, momentum in zip(bn_layers, original_momentum):
                if hasattr(module, 'momentum') and momentum is not None:
                    module.momentum = momentum

            # Restore original gradient states
            for name, param in model.named_parameters():
                param.requires_grad = original_requires_grad[name]

            model.train(was_training)

            return {
                'error': validation['error'],
                'gradient_coverage': validation['coverage'],
                'validation_failed': True
            }

        # Convert single batch to list
        if isinstance(batch, dict):
            # If single batch provided, split into non-overlapping microbatches
            # CRITICAL FIX: Shuffling produces identical gradients for permutation-invariant losses!
            base_batch = batch
            batches = []

            if 'input_ids' in base_batch and torch.is_tensor(base_batch['input_ids']):
                batch_size = base_batch['input_ids'].size(0)

                # Warn if batch too small
                if batch_size < n_samples * 2:
                    logger.warning(
                        f"Batch size {batch_size} is small for {n_samples} samples. "
                        f"SNR computation may be unreliable. Consider providing multiple distinct batches."
                    )

                # Split into microbatches
                chunk_size = max(1, batch_size // n_samples)

                for i in range(n_samples):
                    start_idx = i * chunk_size
                    end_idx = min((i + 1) * chunk_size, batch_size) if i < n_samples - 1 else batch_size

                    if start_idx < batch_size:
                        micro_batch = {}
                        for k, v in base_batch.items():
                            if torch.is_tensor(v) and v.size(0) == batch_size:
                                # Ensure contiguous tensors to avoid view issues
                                micro_batch[k] = v[start_idx:end_idx].contiguous()
                            else:
                                micro_batch[k] = v
                        batches.append(micro_batch)

                # Fallback if we couldn't create enough microbatches
                if len(batches) < n_samples:
                    logger.warning(f"Only created {len(batches)} microbatches from batch size {batch_size}")
                    # Pad with the full batch for remaining samples
                    while len(batches) < n_samples:
                        batches.append(base_batch)
            else:
                # Fallback for non-standard batch format
                logger.warning("Cannot split batch into microbatches, using full batch (SNR will be unreliable)")
                batches = [base_batch] * n_samples
        else:
            # Already a list of batches
            batches = batch

        # Collect gradients from multiple batches
        all_gradients = defaultdict(list) if not memory_efficient else None
        all_gradient_norms = defaultdict(list)  # param_name -> list of gradient norms

        # For memory-efficient SNR computation using unified Welford's algorithm
        if memory_efficient:
            from utils.welford import WelfordAccumulator
            gradient_accumulator = WelfordAccumulator(
                device='cpu',  # Keep on CPU to save GPU memory
                dtype=torch.float32,
                use_keys=True,  # One accumulator per parameter
                weighted=False
            )
        else:
            gradient_accumulator = None

        # Add progress bar for batch processing
        from contextlib import nullcontext
        cm = logging_redirect_tqdm() if show_progress else nullcontext()
        with cm:
            for batch_idx, current_batch in enumerate(tqdm(batches,
                                                          desc="Computing gradient pathology",
                                                          leave=False,
                                                          file=sys.stderr,
                                                          disable=not show_progress)):
                # Move batch to device
                current_batch = self._to_model_device(model, current_batch)
                current_batch = self._with_labels(current_batch)

                # Forward and backward
                model.zero_grad(set_to_none=True)

                try:
                    outputs = model(**current_batch)
                    loss = outputs.loss

                    if loss is None or not loss.requires_grad:
                        continue

                    # Ensure FP32 gradient computation
                    with torch.cuda.amp.autocast(enabled=False):
                        if loss.dtype not in [torch.float32, torch.float64]:
                            loss = loss.float()
                        loss.backward()

                    # Collect gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grad = param.grad.detach()

                            # Transform to natural gradient if requested
                            if use_natural_gradient:
                                grad = self._transform_to_natural_gradient(
                                    grad, name, kfac_factors
                                )

                            # Compute norm in float32 for stability
                            grad_norm = grad.float().norm().item()
                            all_gradient_norms[name].append(grad_norm)

                            if memory_efficient and compute_snr:
                                # Use unified Welford accumulator for memory efficiency
                                grad_cpu = grad.cpu().float()
                                gradient_accumulator.update(grad_cpu, key=name)
                            elif not memory_efficient:
                                # Store full gradients for standard SNR computation
                                grad_cpu = grad.cpu()
                                all_gradients[name].append(grad_cpu)

                except Exception as e:
                    logger.warning(f"Failed to compute gradients for batch {batch_idx}: {e}")
                    continue

        if not all_gradient_norms:
            return {'error': 'Failed to compute gradients for any batch'}

        # Compute statistics across batches
        results = {
            'num_batches_analyzed': len(batches),
            'gradient_statistics': {},
            'vanishing_parameters': [],
            'exploding_parameters': [],
            'pathological_layers': [],
            'natural_gradient_used': use_natural_gradient,
            'kfac_enabled': kfac_factors is not None and len(kfac_factors) > 0,
            'multi_batch_stats': {
                'n_batches_requested': n_samples,
                'n_batches_processed': len(batches),
                'memory_efficient_mode': memory_efficient,
                'welford_algorithm_used': memory_efficient and compute_snr
            }
        }

        # Scaled thresholds based on learning rate
        vanishing_threshold = absolute_vanishing_threshold * learning_rate
        exploding_threshold = absolute_exploding_threshold / learning_rate

        # Analyze each parameter (with progress bar for many parameters)
        param_iterator = all_gradient_norms.items()
        if len(all_gradient_norms) > 100:
            param_iterator = tqdm(param_iterator,
                                  desc="Analyzing parameters",
                                  total=len(all_gradient_norms),
                                  leave=False,
                                  file=sys.stderr,
                                  disable=not show_progress)
        cm2 = logging_redirect_tqdm() if show_progress else nullcontext()
        with cm2:
            for param_name, grad_norms in param_iterator:
                mean_norm = np.mean(grad_norms)
                std_norm = np.std(grad_norms) if len(grad_norms) > 1 else 0.0

                stats = {
                    'mean_norm': mean_norm,
                    'std_norm': std_norm,
                    'min_norm': min(grad_norms),
                    'max_norm': max(grad_norms),
                    'cv': std_norm / (mean_norm + 1e-10),  # Coefficient of variation
                    'n_samples': len(grad_norms),
                    'variance_reduction': 1.0 / len(grad_norms) if len(grad_norms) > 1 else 1.0,
                    'confidence_95': 1.96 * std_norm / np.sqrt(len(grad_norms)) if len(grad_norms) > 1 else float('inf')
                }

                # Check for pathological gradients using absolute thresholds
                if mean_norm < vanishing_threshold:
                    stats['is_vanishing'] = True
                    results['vanishing_parameters'].append(param_name)
                else:
                    stats['is_vanishing'] = False

                if mean_norm > exploding_threshold:
                    stats['is_exploding'] = True
                    results['exploding_parameters'].append(param_name)
                else:
                    stats['is_exploding'] = False

                results['gradient_statistics'][param_name] = stats

        # Populate pathological_layers with actual layer names
        pathological_by_layer = defaultdict(set)
        for param_name in results['vanishing_parameters']:
            # Extract layer name (everything before .weight or .bias)
            layer = '.'.join(param_name.split('.')[:-1]) if '.' in param_name else param_name
            pathological_by_layer[layer].add('vanishing')

        for param_name in results['exploding_parameters']:
            # Extract layer name
            layer = '.'.join(param_name.split('.')[:-1]) if '.' in param_name else param_name
            pathological_by_layer[layer].add('exploding')

        results['pathological_layers'] = sorted(list(pathological_by_layer.keys()))
        results['layer_pathology_details'] = {k: sorted(list(v)) for k, v in pathological_by_layer.items()}

        # Compute Signal-to-Noise Ratio if requested
        if compute_snr and len(batches) > 1:
            snr_results = {}

            if memory_efficient and gradient_accumulator:
                # Compute SNR from unified Welford accumulator
                # Get list of parameter names that have been accumulated
                param_names = list(all_gradient_norms.keys())

                snr_iterator = param_names
                if len(param_names) > 100:
                    snr_iterator = tqdm(snr_iterator, desc="Computing SNR", total=len(param_names), leave=False, file=sys.stderr)

                with logging_redirect_tqdm():
                    for param_name in snr_iterator:
                        stats = gradient_accumulator.get_statistics(key=param_name)
                        if stats['mean'] is not None and stats.get('count', 0) > 1:
                            mean_grad = stats['mean']
                            std_grad = stats['std']

                            # SNR = ||E[grad]|| / mean(std[grad])
                            signal = mean_grad.norm().item()
                            noise = std_grad.mean().item()
                            snr = signal / (noise + 1e-10)

                            snr_results[param_name] = {
                                'snr': snr,
                                'signal': signal,
                                'noise': noise,
                                'reliable': snr > 3.0
                            }
            else:
                # Standard SNR computation from stored gradients (with progress for many params)
                snr_iterator = all_gradients.items()
                if len(all_gradients) > 100:
                    snr_iterator = tqdm(snr_iterator, desc="Computing SNR", total=len(all_gradients), leave=False, file=sys.stderr)
                with logging_redirect_tqdm():
                    for param_name, gradients in snr_iterator:
                        if len(gradients) > 1:
                            # Stack gradients and compute mean/std
                            stacked = torch.stack(gradients)
                            mean_grad = stacked.mean(dim=0)
                            std_grad = stacked.std(dim=0)

                            # SNR = ||E[grad]|| / mean(std[grad])
                            signal = mean_grad.norm().item()
                            noise = std_grad.mean().item()
                            snr = signal / (noise + 1e-10)

                            snr_results[param_name] = {
                                'snr': snr,
                                'signal': signal,
                                'noise': noise,
                                'reliable': snr > 3.0  # SNR > 3 is generally considered reliable
                            }

            results['signal_to_noise_ratio'] = snr_results

        # Compute Gradient Flow Score if requested
        if compute_flow_score:
            flow_results = self._compute_gradient_flow_score(model, results['gradient_statistics'])
            results.update(flow_results)

        # Calculate overall optimization health score
        num_params = len(results['gradient_statistics'])
        num_vanishing = len(results['vanishing_parameters'])
        num_exploding = len(results['exploding_parameters'])

        # Health score components
        vanishing_penalty = (num_vanishing / num_params) if num_params > 0 else 0
        exploding_penalty = (num_exploding / num_params) if num_params > 0 else 0

        # Average SNR score if computed
        snr_score = 1.0
        if 'signal_to_noise_ratio' in results:
            snr_values = [v['snr'] for v in results['signal_to_noise_ratio'].values()]
            if snr_values:
                # Sigmoid-like scoring for SNR (3.0 = 0.5, higher is better)
                avg_snr = np.mean(snr_values)
                snr_score = 1.0 / (1.0 + np.exp(-0.5 * (avg_snr - 3.0)))

        # Flow score if computed
        flow_score = results.get('gradient_flow_health', 1.0)

        # Combined health score (0 = very unhealthy, 1 = perfectly healthy)
        health_score = (1.0 - vanishing_penalty) * (1.0 - exploding_penalty) * snr_score * flow_score

        results['optimization_health_score'] = float(health_score)
        results['health_interpretation'] = self._interpret_health_score(health_score)

        # Add diagnostic summary
        results['summary'] = {
            'total_parameters': num_params,
            'vanishing_count': num_vanishing,
            'exploding_count': num_exploding,
            'vanishing_percentage': vanishing_penalty * 100,
            'exploding_percentage': exploding_penalty * 100,
            'learning_rate': learning_rate,
            'vanishing_threshold_used': vanishing_threshold,
            'exploding_threshold_used': exploding_threshold
        }

        # Clean up
        model.zero_grad(set_to_none=True)

        # Restore BatchNorm momentum values
        for module, momentum in zip(bn_layers, original_momentum):
            if hasattr(module, 'momentum') and momentum is not None:
                module.momentum = momentum

        # Restore original gradient states
        for name, param in model.named_parameters():
            param.requires_grad = original_requires_grad[name]

        model.train(was_training)

        return results

    def _compute_gradient_flow_score(self, model, gradient_statistics: Dict) -> Dict[str, Any]:
        """
        Compute gradient flow score through network depth.

        Measures how well gradients propagate from output to input layers.
        A healthy network should maintain gradient magnitudes without
        exponential decay (vanishing) or growth (exploding).
        """
        # Group parameters by depth/layer index
        layer_groups = self._group_parameters_by_layer(model)

        # Calculate mean gradient norm per layer, normalized by parameter count
        # This ensures fair comparison across layers with different sizes
        layer_gradient_means = {}
        for layer_name, param_names in layer_groups.items():
            total_gradient_magnitude = 0.0
            total_params = 0

            for param_name in param_names:
                if param_name in gradient_statistics:
                    grad_norm = gradient_statistics[param_name]['mean_norm']
                    # Get parameter count for proper weighting
                    for name, param in model.named_parameters():
                        if name == param_name:
                            param_count = param.numel()
                            # Accumulate total gradient magnitude
                            total_gradient_magnitude += grad_norm * param_count
                            total_params += param_count
                            break

            if total_params > 0:
                # Compute per-parameter gradient magnitude for fair comparison
                # This normalizes large layers (like embeddings) appropriately
                layer_gradient_means[layer_name] = total_gradient_magnitude / total_params

        if len(layer_gradient_means) < 2:
            return {
                'gradient_flow_score': float('nan'),
                'gradient_flow_health': 1.0,
                'layer_gradient_decay': {}
            }

        # Sort layers by depth (heuristic: later in name = deeper)
        # This is model-dependent and may need adjustment
        sorted_layers = sorted(layer_gradient_means.items(),
                             key=lambda x: self._estimate_layer_depth(x[0]))

        # Compute gradient decay rate
        layer_norms = [norm for _, norm in sorted_layers]
        layer_names = [name for name, _ in sorted_layers]

        # Fit exponential decay/growth: gradient_norm = a * exp(b * depth)
        # log(gradient_norm) = log(a) + b * depth
        depths = np.arange(len(layer_norms))
        log_norms = np.log(np.array(layer_norms) + 1e-10)

        # Linear regression in log space
        decay_rate = np.polyfit(depths, log_norms, 1)[0] if len(depths) > 1 else 0

        # Ideal decay rate is close to 0 (no decay or explosion)
        # Convert to health score: exp(-|decay_rate|)
        flow_health = np.exp(-abs(decay_rate))

        return {
            'gradient_flow_score': float(decay_rate),
            'gradient_flow_health': float(flow_health),
            'layer_gradient_decay': dict(sorted_layers),
            'gradient_decay_rate': float(decay_rate),
            'interpretation': self._interpret_flow_score(decay_rate)
        }

    def _estimate_layer_depth(self, layer_name: str) -> int:
        """
        Estimate layer depth from name (heuristic).

        Common patterns:
        - layers.0, layers.1, ... -> extract number
        - encoder.layer.0, encoder.layer.1, ... -> extract number
        - blocks.0, blocks.1, ... -> extract number
        """
        import re

        # Try to extract layer number
        patterns = [
            r'layer[s]?\.(\d+)',
            r'block[s]?\.(\d+)',
            r'\.(\d+)\.',
        ]

        for pattern in patterns:
            match = re.search(pattern, layer_name)
            if match:
                return int(match.group(1))

        # Fallback: use position in name
        return len(layer_name.split('.'))

    def _interpret_flow_score(self, decay_rate: float) -> str:
        """Interpret gradient flow decay rate."""
        if abs(decay_rate) < 0.1:
            return "Excellent: Gradients flow well through network"
        elif decay_rate < -0.5:
            return "Vanishing: Gradients decay too quickly"
        elif decay_rate > 0.5:
            return "Exploding: Gradients grow through network"
        elif decay_rate < -0.1:
            return "Mild vanishing: Some gradient decay"
        else:
            return "Mild exploding: Some gradient growth"

    def _interpret_health_score(self, score: float) -> str:
        """Interpret overall optimization health score."""
        if score > 0.8:
            return "Excellent: Network is well-conditioned for optimization"
        elif score > 0.6:
            return "Good: Minor gradient issues but training should proceed"
        elif score > 0.4:
            return "Fair: Some gradient pathology, consider adjustments"
        elif score > 0.2:
            return "Poor: Significant gradient issues, training may struggle"
        else:
            return "Critical: Severe gradient pathology, training likely to fail"

    def compute_gradient_conflict_pcgrad(
        self,
        model,
        batch1: Dict[str, torch.Tensor],
        batch2: Dict[str, torch.Tensor],
        per_layer: bool = True,
        eval_mode: bool = True,  # Use eval mode for deterministic gradient computation
        use_layerwise: bool = False  # Added for memory efficiency
    ) -> Dict[str, Any]:
        """
        Measure gradient conflict using the conflict metric from PCGrad.
        Note: This measures conflict but does NOT perform PCGrad's projection step.
        Based on: "Gradient Surgery for Multi-Task Learning" (Yu et al., 2020)

        Args:
            model: Model to analyze
            batch1: First task batch
            batch2: Second task batch
            per_layer: If True, compute layer-wise conflicts (not just per-parameter)
            eval_mode: If True, use eval mode for deterministic measurement (no dropout)
            use_layerwise: If True, compute gradients layer-by-layer to save memory

        Use this when: Multi-task learning, instruction tuning, RLHF performance issues
        Red flags: conflict_score > 0.5 means tasks are pulling in opposite directions
        """
        was_training = model.training

        # Set mode based on eval_mode parameter
        if eval_mode:
            model.eval()  # Deterministic: no dropout
        else:
            model.train()  # Include dropout effects

        # Use memory-efficient layerwise computation if requested
        if use_layerwise:
            # Use the memory-efficient implementation
            cosine_sim = self._compute_layerwise_gradient_conflict(model, batch1, batch2)

            # Build basic results
            results = {
                'gradient_cosine_similarity': float(cosine_sim) if not np.isnan(cosine_sim) else float('nan'),
                'gradient_dot_product': float('nan'),  # Not computed in layerwise mode
                'gradient_conflict': float(max(0, -cosine_sim)) if not np.isnan(cosine_sim) else 0.0,
                'conflict_score': float(max(0, -cosine_sim)) if not np.isnan(cosine_sim) else 0.0,
                'gradient_magnitude_ratio': float('nan'),  # Not computed in layerwise mode
                'use_layerwise': True
            }

            model.train(was_training)
            return results
        
        # Move first batch to device
        device = next(model.parameters()).device
        batch1 = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch1.items()}

        # Add labels if needed
        batch1 = self._with_labels(batch1)

        # Compute gradients for first batch
        model.zero_grad(set_to_none=True)
        # Check if model supports use_cache parameter
        supports_use_cache = 'use_cache' in inspect.signature(model.forward).parameters
        outputs1 = model(**batch1, use_cache=False) if supports_use_cache else model(**batch1)
        loss1 = outputs1.loss

        # Check if loss requires grad
        if not loss1.requires_grad:
            # Handle models with no trainable parameters
            model.train(was_training)  # Restore original training mode
            return {
                'gradient_cosine_similarity': float('nan'),
                'gradient_dot_product': 0.0,
                'gradient_conflict': 0.0,
                'conflict_score': 0.0,
                'gradient_magnitude_ratio': float('nan'),
                'error': 'Loss does not require gradients'
            }

        # Ensure FP32 gradient computation
        with torch.cuda.amp.autocast(enabled=False):
            if loss1.dtype not in [torch.float32, torch.float64]:
                loss1 = loss1.float()
            loss1.backward()

        grads1 = []
        grad_norms1 = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.clone().detach().cpu()  # Move to CPU to save GPU memory
                grads1.append((name, g))
                grad_norms1.append(g.norm().item())
                param.grad = None  # Free GPU memory (more compatible than del)

        # Save loss value before cleanup
        loss1_value = float(loss1.item())

        # Free first batch and computation graph before loading second batch
        del outputs1, loss1, batch1
        model.zero_grad(set_to_none=True)

        # Force GPU memory cleanup before loading second batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # NOW load second batch to device
        batch2 = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch2.items()}
        batch2 = self._with_labels(batch2)

        # Compute gradients for second batch
        outputs2 = model(**batch2, use_cache=False) if supports_use_cache else model(**batch2)
        loss2 = outputs2.loss
        # Ensure FP32 gradient computation
        with torch.cuda.amp.autocast(enabled=False):
            if loss2.dtype not in [torch.float32, torch.float64]:
                loss2 = loss2.float()
            loss2.backward()

        grads2 = []
        grad_norms2 = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                g = param.grad.clone().detach().cpu()  # Move to CPU to save GPU memory
                grads2.append((name, g))
                grad_norms2.append(g.norm().item())
                param.grad = None  # Free GPU memory (more compatible than del)

        # Save loss value before cleanup
        loss2_value = float(loss2.item())

        # Free second batch and computation graph
        del outputs2, loss2, batch2
        model.zero_grad(set_to_none=True)

        # Clear GPU cache after gradient computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Compute conflict metrics
        results = {
            'loss1': loss1_value,
            'loss2': loss2_value,
            'loss_difference': float(abs(loss1_value - loss2_value))
        }
        
        # Build gradient dicts for consistent ordering
        g1 = {n: g.flatten() for n, g in grads1}
        g2 = {n: g.flatten() for n, g in grads2}
        common = sorted(g1.keys() & g2.keys())
        
        if len(common) == 0:
            model.train(was_training)  # Restore original training mode
            return {'error': 'No overlapping parameters with gradients'}
        
        # Compute global cosine on CPU (streamed), avoids giant GPU tensors
        dot = 0.0
        n1 = 0.0
        n2 = 0.0
        for n in common:
            a = g1[n].to(torch.float32)  # CPU, cast to float32
            b = g2[n].to(torch.float32)  # CPU, cast to float32
            dot += float((a * b).sum())
            n1 += float(a.pow(2).sum())
            n2 += float(b.pow(2).sum())

        if n1 < 1e-12 or n2 < 1e-12:
            cosine_sim = float('nan')
            dot_product = 0.0
        else:
            cosine_sim = dot / (np.sqrt(n1) * np.sqrt(n2))
            dot_product = dot

        # Compute gradient norms for ratio
        norm1 = np.sqrt(n1)
        norm2 = np.sqrt(n2)

        results.update({
            'gradient_cosine_similarity': float(cosine_sim),
            'gradient_dot_product': float(dot_product),
            'gradient_conflict': float(max(0, -cosine_sim)),  # PCGrad-style: only negative cosine is conflict
            'conflict_score': float(max(0, -cosine_sim)),  # Alias for clarity
            'gradient_magnitude_ratio': float(norm1 / (norm2 + 1e-10)) if not np.isnan(norm1) and not np.isnan(norm2) else float('nan')
        })
        
        if per_layer:
            # Group parameters by actual layers using the helper
            layer_groups = self._group_parameters_by_layer(model)
            
            # Build gradient dictionaries for easy lookup
            grads1_dict = {name: g for name, g in grads1}
            grads2_dict = {name: g for name, g in grads2}
            
            layer_conflicts = {}
            layer_similarities = {}
            
            # Compute conflict per actual layer (not per parameter)
            for layer_name, param_names in layer_groups.items():
                # Collect all gradients for this layer
                layer_grads1 = []
                layer_grads2 = []
                
                for param_name in param_names:
                    if param_name in grads1_dict and param_name in grads2_dict:
                        layer_grads1.append(grads1_dict[param_name].flatten())
                        layer_grads2.append(grads2_dict[param_name].flatten())
                
                if layer_grads1 and layer_grads2:
                    # Concatenate all gradients for this layer, cast to float32 for stability
                    layer_grad1 = torch.cat(layer_grads1).to(torch.float32)
                    layer_grad2 = torch.cat(layer_grads2).to(torch.float32)
                    
                    # Compute layer-level similarity - handle "no signal" explicitly
                    if layer_grad1.norm() <= 1e-12 or layer_grad2.norm() <= 1e-12:
                        layer_similarities[layer_name] = float('nan')  # Mark as no signal
                        layer_conflicts[layer_name] = float('nan')  # Exclude from statistics
                        continue
                    
                    layer_sim = F.cosine_similarity(
                        layer_grad1.unsqueeze(0),
                        layer_grad2.unsqueeze(0)
                    ).item()
                    
                    layer_similarities[layer_name] = layer_sim
                    # PCGrad-style conflict: only negative cosine indicates conflict
                    layer_conflicts[layer_name] = max(0, -layer_sim)
            
            # Compute summary statistics
            if layer_conflicts:
                # Track zero-gradient layers
                zero_grad_layers = [layer for layer, sim in layer_similarities.items() 
                                  if np.isnan(sim)]
                
                # Filter out NaN values for statistics
                conflict_values = [v for v in layer_conflicts.values() if not np.isnan(v)]
                sim_values = [v for v in layer_similarities.values() if not np.isnan(v)]
                
                results.update({
                    'layer_conflicts': layer_conflicts,  # Dict of layer_name -> conflict
                    'layer_similarities': layer_similarities,  # Dict of layer_name -> similarity
                    'mean_layer_conflict': float(np.mean(conflict_values)) if conflict_values else float('nan'),
                    'max_layer_conflict': float(np.max(conflict_values)) if conflict_values else float('nan'),
                    'min_layer_conflict': float(np.min(conflict_values)) if conflict_values else float('nan'),
                    'conflicting_layers_count': int(sum(1 for c in conflict_values if c > 0)),
                    'layer_conflict_std': float(np.std(conflict_values)) if conflict_values else float('nan'),
                    'zero_grad_layers': zero_grad_layers,
                    'zero_grad_layers_count': len(zero_grad_layers)
                })
                
                # Find most conflicting layers
                sorted_conflicts = sorted(layer_conflicts.items(), key=lambda x: x[1], reverse=True)
                most_conflicting = [name for name, _ in sorted_conflicts[:5]]
                results['most_conflicting_layers'] = most_conflicting

        # Clean up gradient dicts to free memory
        del grads1, grads2, g1, g2

        model.train(was_training)
        return results
    
    # ============= DETAILED ANALYSIS (Audited) =============
    
    def compute_gradient_alignment_trajectory(
        self,
        models: Union[Any, List[Any]],
        batches: List[Dict[str, torch.Tensor]],
        reference_batch: Optional[Dict[str, torch.Tensor]] = None,
        eval_mode: bool = True,  # Use eval mode for deterministic gradient computation
        memory_efficient: bool = False,
        max_batch_size: Optional[int] = None,  # For validation only
        max_seq_length: Optional[int] = None   # For validation only
    ) -> Dict[str, Any]:
        """
        Track how gradient alignment changes across batches and checkpoints.

        IMPORTANT: Batch Size Configuration
        ------------------------------------
        This function validates batch sizes but does NOT automatically adjust them.
        You must provide properly sized batches.

        If max_batch_size or max_seq_length are provided, the function will:
        - Validate that batches don't exceed these limits
        - Raise an error if limits are exceeded (no automatic adjustment)

        To control batch size:
        1. Configure batch_size in your config (default: 256)
        2. The function will validate against max_batch_size if provided
        3. Ensure your batches are within the configured limits

        Args:
            models: Single model or list of models (checkpoints)
            batches: List of batches to compute alignment with
            reference_batch: Reference batch for alignment (defaults to first batch)
            eval_mode: If True, use eval mode for deterministic measurements (no dropout)
            memory_efficient: If True, compute alignment incrementally to save memory

        Use this when: Analyzing training stability, comparing checkpoints
        Red flags: High variance in alignment = unstable training
        """
        import gc

        # Clear GPU memory at the very beginning to start fresh
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        try:
            from gpu_memory_tracker import log_gpu_memory
        except ImportError:
            # Create a simple fallback if module not available
            def log_gpu_memory(msg):
                if torch.cuda.is_available():
                    logger.info(f"{msg} - GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB allocated")

        # Early check for empty inputs
        if not batches:
            return {'error': 'No batches provided'}

        # Upfront validation: Check memory requirements before starting
        if torch.cuda.is_available() and batches and 'input_ids' in batches[0]:
            # Get actual batch dimensions - use them as-is, no adjustment
            actual_batch_size = batches[0]['input_ids'].shape[0]
            actual_seq_length = batches[0]['input_ids'].shape[1]

            # Estimate memory requirements (rough estimate)
            # Model params (gradients need 2x for grad+momentum), activations, reference gradients
            if isinstance(models, list) and models:
                model = models[0]
            else:
                model = models

            model_params = sum(p.numel() for p in model.parameters()) / 1e9  # In billions
            estimated_memory_gb = (
                model_params * 4 +  # Model weights (fp32)
                model_params * 4 * 2 +  # Gradients (need to store reference + current)
                actual_batch_size * actual_seq_length * 4096 * 4 / 1e9 * 2  # Activations (rough estimate)
            )

            # Get available GPU memory
            free_memory, total_memory = torch.cuda.mem_get_info()
            free_gb = free_memory / 1e9

            logger.info(f"[GRADIENT TRAJECTORY] Processing batches of size {actual_batch_size}x{actual_seq_length}")
            logger.info(f"[GRADIENT TRAJECTORY] Memory estimate: {estimated_memory_gb:.1f}GB required, {free_gb:.1f}GB available")

            # Fail fast if memory requirements exceed available memory
            if estimated_memory_gb > free_gb * 0.8:  # Leave 20% buffer for safety
                error_msg = (
                    f"Memory requirements ({estimated_memory_gb:.1f}GB) exceed available GPU memory ({free_gb:.1f}GB). "
                    f"Current batch size: {actual_batch_size}x{actual_seq_length}. "
                    f"Please reduce gradient_trajectory_batch_size or gradient_trajectory_seq_length in your config."
                )
                logger.error(f"[GRADIENT TRAJECTORY] ❌ {error_msg}")
                return {'error': error_msg, 'oom': True}

        # Force memory_efficient mode for this multi-batch function to prevent OOM
        # This is the only gradient function that processes multiple batches in a loop
        memory_efficient = True

        # Handle single model vs multiple models
        if not isinstance(models, list):
            models = [models]

        if not models:
            return {'error': 'No models provided'}

        if reference_batch is None:
            reference_batch = batches[0]

        all_results = {}
        training_states = []  # Track original training states

        # Log initial memory state
        if torch.cuda.is_available():
            log_gpu_memory("[GRADIENT TRAJECTORY] Starting computation")
        
        for model_idx, model in enumerate(models):
            training_states.append(model.training)  # Store original state
            
            # Set mode based on eval_mode parameter
            if eval_mode:
                model.eval()  # Deterministic: no dropout
            else:
                model.train()  # Include dropout effects
            
            # Get reference gradient with proper parameter tracking
            ref_batch = reference_batch.copy() if isinstance(reference_batch, dict) else reference_batch

            # Log reference batch dimensions for debugging
            if 'input_ids' in ref_batch:
                ref_batch_size = ref_batch['input_ids'].shape[0]
                ref_seq_length = ref_batch['input_ids'].shape[1]
                logger.debug(f"[GRADIENT TRAJECTORY] Reference batch: size={ref_batch_size}, seq_len={ref_seq_length}")

            ref_batch = self._to_model_device(model, ref_batch)
            ref_batch = self._with_labels(ref_batch)

            # Try to compute reference gradients with OOM protection
            try:
                model.zero_grad(set_to_none=True)
                ref_outputs = model(**ref_batch)
                ref_outputs.loss.backward()

                # Build reference gradient dict with parameter names
                ref_grads_dict = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        # Move to CPU immediately to free GPU memory
                        ref_grads_dict[name] = param.grad.clone().detach().cpu()
                        param.grad = None  # Free GPU memory immediately

                ref_param_count = len(ref_grads_dict)  # Track parameter count

                # Free reference batch and outputs to reduce memory pressure
                del ref_outputs, ref_batch
                model.zero_grad(set_to_none=True)

                # CRITICAL: Clear all CUDA caches after reference gradient computation
                # This is essential because reference gradients are kept in memory throughout
                # the entire batch loop, creating a baseline memory footprint
                if torch.cuda.is_available():
                    torch.cuda.synchronize()  # Ensure all GPU ops complete
                    torch.cuda.empty_cache()
                    # Reset the memory allocator stats to avoid fragmentation
                    torch.cuda.reset_peak_memory_stats()
                gc.collect()  # Force CPU memory cleanup

            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"[GRADIENT TRAJECTORY] OOM computing reference gradients: {str(e)}")
                model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                return {'error': f'OOM computing reference gradients for model {model_idx}', 'oom': True}
            
            # Track alignment with each batch
            alignments = []
            conflicts = []
            nan_count = 0
            common_params_counts = []  # Track common params per batch
            
            for batch_idx, batch in enumerate(batches):
                # Log memory before processing batch
                if torch.cuda.is_available() and batch_idx % 10 == 0:
                    log_gpu_memory(f"[GRADIENT TRAJECTORY] Processing batch {batch_idx}/{len(batches)}")

                try:
                    # Log batch dimensions for debugging
                    if 'input_ids' in batch:
                        actual_batch_size = batch['input_ids'].shape[0]
                        actual_seq_length = batch['input_ids'].shape[1]
                        logger.debug(f"[GRADIENT TRAJECTORY] Processing batch {batch_idx}: size={actual_batch_size}, seq_len={actual_seq_length}")

                    batch = self._to_model_device(model, batch)
                    batch = self._with_labels(batch)

                    # Clear GPU memory before forward pass
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                    model.zero_grad(set_to_none=True)

                    # Use gradient checkpointing if available to reduce activation memory
                    if hasattr(model, 'gradient_checkpointing_enable'):
                        try:
                            # Use non-reentrant mode to preserve gradient flow
                            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                        except TypeError:
                            # Fallback for older transformers versions
                            model.gradient_checkpointing_enable()

                        # Enable input gradients for proper flow
                        if hasattr(model, 'enable_input_require_grads'):
                            model.enable_input_require_grads()

                        outputs = model(**batch)
                        outputs.loss.backward()
                        model.gradient_checkpointing_disable()
                    else:
                        outputs = model(**batch)
                        outputs.loss.backward()

                    # Build batch gradient dict with immediate CPU transfer
                    # Use streaming to avoid large memory spikes
                    batch_grads_dict = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            # Move to CPU immediately in chunks to avoid memory spike
                            grad_cpu = param.grad.detach().cpu()
                            batch_grads_dict[name] = grad_cpu
                            param.grad = None  # Free GPU memory immediately

                            # Force synchronization every 10 parameters to prevent accumulation
                            if len(batch_grads_dict) % 10 == 0 and torch.cuda.is_available():
                                torch.cuda.synchronize()

                    # Free outputs and batch to reduce memory accumulation
                    del outputs, batch
                    model.zero_grad(set_to_none=True)

                    # ALWAYS synchronize and clear cache for this multi-batch function
                    # This is critical to prevent memory fragmentation across batch iterations
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                        torch.cuda.empty_cache()
                        # Extra cleanup for trajectory function to prevent accumulation
                        if batch_idx % 5 == 0:
                            torch.cuda.reset_peak_memory_stats()

                except torch.cuda.OutOfMemoryError as e:
                    logger.warning(f"[GRADIENT TRAJECTORY] OOM on batch {batch_idx}, skipping: {str(e)}")
                    model.zero_grad(set_to_none=True)
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    # Add NaN for this batch and continue
                    alignments.append(float('nan'))
                    conflicts.append(float('nan'))
                    nan_count += 1
                    continue
                
                # Find common parameters with gradients
                common_params = sorted(ref_grads_dict.keys() & batch_grads_dict.keys())
                common_params_counts.append(len(common_params))  # Track for metadata
                
                if len(common_params) == 0:
                    cosine_sim = float('nan')
                    nan_count += 1
                elif memory_efficient:
                    # Memory-efficient computation: process on CPU to avoid GPU memory
                    # Use streaming computation for very large models
                    dot_product = 0.0
                    ref_norm_sq = 0.0
                    batch_norm_sq = 0.0

                    # Process in smaller chunks to avoid memory spikes even on CPU
                    chunk_size = 10  # Process 10 parameters at a time
                    param_chunks = [common_params[i:i+chunk_size] for i in range(0, len(common_params), chunk_size)]

                    for chunk in param_chunks:
                        for param_name in chunk:
                            # Keep everything on CPU in memory_efficient mode
                            ref_g = ref_grads_dict[param_name].view(-1).float()
                            batch_g = batch_grads_dict[param_name].view(-1).float()

                            # Compute on CPU to avoid GPU memory usage
                            # Use item() to convert to Python float immediately
                            dot_product += float((ref_g * batch_g).sum().item())
                            ref_norm_sq += float(ref_g.pow(2).sum().item())
                            batch_norm_sq += float(batch_g.pow(2).sum().item())

                            # Delete intermediate tensors immediately
                            del ref_g, batch_g

                        # Force garbage collection after each chunk
                        gc.collect()

                    # No sync needed since we computed on CPU
                    if ref_norm_sq < 1e-12 or batch_norm_sq < 1e-12:
                        cosine_sim = float('nan')
                        nan_count += 1
                    else:
                        cosine_sim = dot_product / (np.sqrt(ref_norm_sq) * np.sqrt(batch_norm_sq))

                else:
                    # Standard computation: concatenate gradients with fixed ordering
                    # Cast to float32 for consistency with memory_efficient path
                    ref_grad_vec = torch.cat([ref_grads_dict[n].flatten().float() for n in common_params])
                    batch_grad_vec = torch.cat([batch_grads_dict[n].flatten().float() for n in common_params])

                    # Compute alignment with zero-norm guard
                    if ref_grad_vec.norm() < 1e-12 or batch_grad_vec.norm() < 1e-12:
                        cosine_sim = float('nan')
                        nan_count += 1
                    else:
                        cosine_sim = F.cosine_similarity(
                            ref_grad_vec.unsqueeze(0),
                            batch_grad_vec.unsqueeze(0)
                        ).item()

                    # Clean up concatenated gradient vectors to prevent memory accumulation
                    del ref_grad_vec, batch_grad_vec

                alignments.append(cosine_sim)
                # Use PCGrad-style conflict: only negative cosine is conflict
                conflicts.append(max(0, -cosine_sim) if not np.isnan(cosine_sim) else float('nan'))

                # Clean up batch gradients dict after use (for all paths)
                del batch_grads_dict

                # Periodic aggressive cleanup every 5 batches (more frequent for trajectory)
                if batch_idx % 5 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
            
            # Store results for this model
            model_key = f'model_{model_idx}' if len(models) > 1 else 'single_model'
            
            # Filter out NaN values for statistics
            valid_alignments = [a for a in alignments if not np.isnan(a)]
            valid_conflicts = [c for c in conflicts if not np.isnan(c)]
            
            # Compute trend only on valid alignment points
            if len(valid_alignments) >= 2:
                # Only use indices where alignment is valid for trend computation
                valid_indices = [i for i, a in enumerate(alignments) if not np.isnan(a)]
                alignment_trend = float(np.polyfit(valid_indices, 
                                                  [alignments[i] for i in valid_indices], 1)[0])
            else:
                alignment_trend = float('nan')
            
            all_results[model_key] = {
                'mean_alignment': float(np.mean(valid_alignments)) if valid_alignments else float('nan'),
                'alignment_variance': float(np.var(valid_alignments)) if valid_alignments else float('nan'),
                'max_conflict': float(np.max(valid_conflicts)) if valid_conflicts else float('nan'),
                'alignment_trend': alignment_trend,
                'stable_alignment': bool(np.std(valid_alignments) < 0.1) if valid_alignments else False,
                'alignments': alignments,
                'conflicts': conflicts,
                'nan_alignments_count': nan_count,
                'valid_alignments_count': len(valid_alignments),
                'eval_mode': eval_mode,
                'memory_efficient': memory_efficient,
                'ref_param_count': ref_param_count,
                'common_params_per_batch': common_params_counts,
                'min_common_params': min(common_params_counts) if common_params_counts else 0,
                'max_common_params': max(common_params_counts) if common_params_counts else 0
            }

            # CRITICAL: Clean up reference gradients dict after processing all batches for this model
            # This is essential to prevent memory accumulation across models
            del ref_grads_dict
            # Force Python garbage collection
            gc.collect()

            # Aggressive GPU cleanup
            if torch.cuda.is_available():
                torch.cuda.synchronize()  # Wait for all GPU operations to complete
                torch.cuda.empty_cache()   # Release cached memory
                torch.cuda.reset_peak_memory_stats()  # Reset memory tracking

            # Additional CPU memory cleanup
            gc.collect()  # Run again after GPU cleanup

            # Log memory after processing model
            if torch.cuda.is_available():
                log_gpu_memory(f"[GRADIENT TRAJECTORY] Completed model {model_idx}")
        
        # If multiple models, compute cross-checkpoint statistics
        if len(models) > 1:
            # Extract alignment trajectories across checkpoints
            checkpoint_alignments = []
            for i in range(len(batches)):
                batch_alignments = [all_results[f'model_{j}']['alignments'][i] 
                                  for j in range(len(models))]
                checkpoint_alignments.append(batch_alignments)
            
            all_results['cross_checkpoint'] = {
                'mean_alignment_per_batch': [float(np.nanmean(batch_aligns)) if batch_aligns else float('nan')
                                            for batch_aligns in checkpoint_alignments],
                'alignment_variance_per_batch': [float(np.nanvar(batch_aligns)) if batch_aligns else float('nan')
                                                for batch_aligns in checkpoint_alignments],
                'checkpoint_consistency': float(np.nanmean([np.nanvar(batch_aligns) 
                                                        for batch_aligns in checkpoint_alignments])),
                'num_checkpoints': len(models),
                'num_batches': len(batches)
            }
            
            # Compute overall summary with NaN-safe operations
            all_alignments = []
            all_conflicts = []
            for key in all_results:
                if key.startswith('model_'):
                    all_alignments.extend(all_results[key]['alignments'])
                    all_conflicts.extend(all_results[key]['conflicts'])
            
            # Filter valid values for consistent statistics
            valid_alignments_all = [a for a in all_alignments if not np.isnan(a)]
            valid_conflicts_all = [c for c in all_conflicts if not np.isnan(c)]
            
            # Compute all stats from valid filtered lists for consistency
            all_results['summary'] = {
                'overall_mean_alignment': float(np.mean(valid_alignments_all)) if valid_alignments_all else float('nan'),
                'overall_alignment_variance': float(np.var(valid_alignments_all)) if valid_alignments_all else float('nan'),
                'overall_max_conflict': float(np.max(valid_conflicts_all)) if valid_conflicts_all else float('nan'),
                'stable_alignment': bool(np.std(valid_alignments_all) < 0.1) if valid_alignments_all else False,
                'total_nan_count': sum(1 for a in all_alignments if np.isnan(a)),
                'total_valid_count': len(valid_alignments_all)
            }
        
        # Restore training state for all models
        for model, was_training in zip(models, training_states):
            model.train(was_training)

        # Clear GPU cache after trajectory computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return all_results
    
    def compute_layer_gradient_alignment(
        self,
        model,
        math_batch: Dict[str, torch.Tensor],
        general_batch: Dict[str, torch.Tensor],
        eval_mode: bool = True,  # Use eval mode for deterministic gradient computation
        use_layerwise: bool = False,  # Added for memory efficiency
        apply_fdr: bool = True  # Apply FDR correction for multiple layer tests (ICLR 2026)
    ) -> Dict[str, Any]:
        """
        Compute gradient alignment between tasks PER LAYER (WHERE conflicts occur).

        Returns per-layer cosine similarities and identifies which specific layers
        have gradient conflicts. Single measurement, no resampling.

        Key differences from other functions:
        - Unlike compute_raw_gradient_conflict: Returns PER-LAYER breakdown (not global scalar)
        - Unlike compute_gradient_conflict_matrix_multi: Actually groups by layer (not per-parameter)
        - Unlike compute_gradient_alignment_trajectory: Single snapshot (not time series)

        Args:
            model: Model to analyze
            math_batch: First task batch
            general_batch: Second task batch
            eval_mode: If True, use eval mode for deterministic measurement (no dropout)
            use_layerwise: If True, compute gradients layer-by-layer to save memory

        Use this when: Need to identify WHICH layers conflict for targeted interventions
        Output: Dict with per-layer cosines, conflicts, and most conflicting layer

        See also:
        - compute_raw_gradient_conflict: For robust global conflict with variance
        - compute_gradient_conflict_matrix: For per-parameter (not recommended)
        """
        # Set mode based on eval_mode parameter
        was_training = model.training
        if eval_mode:
            model.eval()  # Deterministic: no dropout
        else:
            model.train()  # Include dropout effects

        # Use memory-efficient layerwise computation if requested
        if use_layerwise:
            # Use the memory-efficient implementation and build per-layer results
            global_cosine_sim = self._compute_layerwise_gradient_conflict(model, math_batch, general_batch)

            # For layerwise mode, we can't provide per-layer breakdown (would defeat the memory savings)
            # Return global metrics with a note about the limitation
            results = {
                'global_cosine_similarity': float(global_cosine_sim) if not np.isnan(global_cosine_sim) else float('nan'),
                'global_conflict': float(max(0, -global_cosine_sim)) if not np.isnan(global_cosine_sim) else 0.0,
                'use_layerwise': True,
                'note': 'Per-layer breakdown not available in layerwise mode (memory optimization)'
            }

            model.train(was_training)
            return results
        
        math_batch = self._to_model_device(model, math_batch)
        math_batch = self._with_labels(math_batch)

        # Use centralized layer grouping for consistency
        layer_groups = self._group_parameters_by_layer(model)

        # Build reverse mapping: param_name -> layer_key
        param_to_layer = {}
        for layer_key, param_names in layer_groups.items():
            for param_name in param_names:
                param_to_layer[param_name] = layer_key

        # Organize gradients by layer
        layer_math_grads = defaultdict(list)
        layer_general_grads = defaultdict(list)

        # Get math gradients
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            math_outputs = model(**math_batch)
            # Ensure FP32 gradient computation
            with torch.cuda.amp.autocast(enabled=False):
                math_loss = math_outputs.loss
                if math_loss.dtype not in [torch.float32, torch.float64]:
                    math_loss = math_loss.float()
                math_loss.backward()

        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_key = param_to_layer.get(name, 'other')
                # Use reshape for non-contiguous tensors, detach before cpu
                layer_math_grads[layer_key].append(param.grad.detach().reshape(-1).cpu())
                param.grad = None  # Free GPU memory (more idiomatic than del)

        # Free first batch and its computation graph before loading second batch
        del math_outputs, math_batch
        model.zero_grad(set_to_none=True)

        # Force GPU memory cleanup before loading second batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # NOW load the general batch to GPU
        general_batch = self._to_model_device(model, general_batch)
        general_batch = self._with_labels(general_batch)

        # Get general gradients
        model.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(True):
            general_outputs = model(**general_batch)
            # Ensure FP32 gradient computation
            with torch.cuda.amp.autocast(enabled=False):
                general_loss = general_outputs.loss
                if general_loss.dtype not in [torch.float32, torch.float64]:
                    general_loss = general_loss.float()
                general_loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_key = param_to_layer.get(name, 'other')
                # Use reshape for non-contiguous tensors, detach before cpu
                layer_general_grads[layer_key].append(param.grad.detach().reshape(-1).cpu())
                param.grad = None  # Free GPU memory (more idiomatic than del)

        # Free second batch and its computation graph
        del general_outputs, general_batch
        model.zero_grad(set_to_none=True)

        # Clear GPU cache after gradient computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Calculate alignment per layer (concatenated gradients)
        alignment_scores = {}
        per_layer = []  # For compatibility
        
        # Compute on CPU to avoid moving gradients back to GPU
        for layer_key in layer_math_grads:
            if layer_key in layer_general_grads:
                # Concatenate all gradients for this layer, cast to float32 for stability
                math_grad_concat = torch.cat(layer_math_grads[layer_key]).float()  # Cast to float32
                general_grad_concat = torch.cat(layer_general_grads[layer_key]).float()  # Cast to float32

                # Cosine similarity for entire layer with zero-norm guard (compute on CPU)
                if math_grad_concat.norm() < 1e-12 or general_grad_concat.norm() < 1e-12:
                    cos_sim = float('nan')
                    conflict = float('nan')  # Keep NaN conflicts out of statistics
                else:
                    cos_sim = F.cosine_similarity(
                        math_grad_concat.unsqueeze(0),
                        general_grad_concat.unsqueeze(0)
                    ).item()
                    # Gradient conflict (negative cosine = conflict)
                    conflict = -cos_sim if cos_sim < 0 else 0
                
                # Calculate actual parameter count
                num_param_tensors = len(layer_math_grads[layer_key])
                num_parameters = sum(g.numel() for g in layer_math_grads[layer_key])
                
                alignment_scores[layer_key] = {
                    'cosine_similarity': cos_sim,
                    'conflict_score': conflict,
                    'is_conflicting': cos_sim < -0.1,
                    'num_param_tensors': num_param_tensors,  # Number of parameter tensors
                    'num_parameters': num_parameters,  # Actual parameter count
                    'num_params': num_param_tensors  # Kept for backward compatibility
                }
                
                # Add to per_layer list for compatibility
                if layer_key.startswith('layer_'):
                    per_layer.append(cos_sim)
        
        # Summary statistics with NaN-safe operations
        cos_sims = [v['cosine_similarity'] for v in alignment_scores.values()]
        conflicts = [v['conflict_score'] for v in alignment_scores.values()]

        # Filter out NaN values for statistics
        valid_cos_sims = [c for c in cos_sims if not np.isnan(c)]
        valid_conflicts = [c for c in conflicts if not np.isnan(c)]
        nan_layers = [k for k, v in alignment_scores.items() if np.isnan(v['cosine_similarity'])]

        # Apply FDR correction for multiple layer comparisons (ICLR 2026)
        fdr_results = None
        if apply_fdr and valid_cos_sims:
            # Convert cosine similarities to p-values for conflict detection
            # Using Fisher transformation: test if correlation is significantly negative
            # H0: cosine similarity >= 0 (no conflict)
            # H1: cosine similarity < 0 (conflict)
            p_values = []
            layer_keys_for_fdr = []

            for layer_key, scores in alignment_scores.items():
                cos_sim = scores['cosine_similarity']
                if not np.isnan(cos_sim):
                    # One-tailed test for negative correlation
                    # Using normal approximation (conservative)
                    # Standard error assuming n≈30 samples (typical batch size)
                    n_approx = 30  # Conservative estimate of effective sample size
                    se = 1.0 / np.sqrt(n_approx - 3)  # Fisher transform standard error
                    z_score = 0.5 * np.log((1 + cos_sim) / (1 - cos_sim + 1e-10))  # Fisher z
                    # One-tailed p-value for negative correlation
                    p_value = stats.norm.cdf(z_score / se)
                    p_values.append(p_value)
                    layer_keys_for_fdr.append(layer_key)

            # Apply FDR correction
            fdr_results = self.apply_fdr_correction(p_values, alpha=0.05)

            # Update is_conflicting based on FDR-corrected results
            for i, layer_key in enumerate(layer_keys_for_fdr):
                alignment_scores[layer_key]['p_value'] = p_values[i]
                alignment_scores[layer_key]['is_conflicting'] = fdr_results['significant'][i]
                alignment_scores[layer_key]['fdr_corrected'] = True
        
        # Find max conflict layer (from valid conflicts only)
        max_conflict_layer = None
        if valid_conflicts:
            # Find the layer with max conflict among non-NaN values
            max_conflict_value = max(valid_conflicts)
            for layer_key, scores in alignment_scores.items():
                if scores['conflict_score'] == max_conflict_value:
                    max_conflict_layer = layer_key
                    break
        
        model.train(was_training)
        
        # Count conflicting layers using FDR-corrected results if available
        num_conflicting = sum(1 for v in alignment_scores.values()
                             if v['is_conflicting'] and not np.isnan(v['cosine_similarity']))

        result = {
            'per_layer_alignment': alignment_scores,
            'per_layer': per_layer,  # For compatibility
            'mean_alignment': float(np.mean(valid_cos_sims)) if valid_cos_sims else float('nan'),
            'min_alignment': float(np.min(valid_cos_sims)) if valid_cos_sims else float('nan'),
            'std_alignment': float(np.std(valid_cos_sims)) if valid_cos_sims else float('nan'),
            'mean_conflict': float(np.mean(valid_conflicts)) if valid_conflicts else float('nan'),
            'num_conflicting_layers': num_conflicting,
            'max_conflict': float(max(valid_conflicts)) if valid_conflicts else float('nan'),
            'max_conflict_layer': max_conflict_layer,
            'nan_layers': nan_layers,
            'nan_layer_count': len(nan_layers),
            'valid_layer_count': len(valid_cos_sims)
        }

        # Add FDR correction results if applied
        if fdr_results:
            result['fdr_correction'] = {
                'applied': True,
                'method': fdr_results['method'],
                'n_tests': fdr_results['n_tests'],
                'n_significant': fdr_results['n_significant'],
                'alpha': fdr_results['alpha'],
                'threshold': fdr_results['threshold'],
                'note': 'Layer conflicts determined using FDR-corrected p-values'
            }
        else:
            result['fdr_correction'] = {
                'applied': False,
                'note': 'Using simple threshold-based conflict detection'
            }

        return result
    
    # ============= MULTI-TASK SPECIFIC (Audited) =============
    
    def compute_gradient_conflict_pair(
        self,
        model,
        task1_batch: Dict[str, torch.Tensor],
        task2_batch: Dict[str, torch.Tensor],
        eval_mode: bool = True,  # Use eval mode for deterministic gradient computation
        use_layerwise: bool = False  # Added for memory efficiency
    ) -> Dict[str, Any]:
        """
        Measures gradient conflict between TWO tasks (pairwise comparison).
        Returns global conflict and per-parameter similarities.

        Key differences from other functions:
        - Unlike compute_layer_gradient_alignment: Per-PARAMETER (not per-layer)
        - Unlike compute_raw_gradient_conflict: Single measurement (no resampling)
        - Returns parameter-level granularity for detailed analysis

        Args:
            model: Model to analyze
            task1_batch: First task batch
            task2_batch: Second task batch
            eval_mode: If True, use eval mode for deterministic measurement (no dropout)
            use_layerwise: If True, compute gradients layer-by-layer to save memory

        Returns:
            Dict with global conflict and per-parameter similarities

        See also:
        - compute_layer_gradient_alignment: For per-layer analysis
        - compute_raw_gradient_conflict: For robust estimates with variance
        """
        was_training = model.training
        if eval_mode:
            model.eval()  # Deterministic: no dropout
        else:
            model.train()  # Include dropout effects

        # Use memory-efficient layerwise computation if requested
        if use_layerwise:
            # Use the memory-efficient implementation
            cosine_sim = self._compute_layerwise_gradient_conflict(model, task1_batch, task2_batch)

            # Build results with limited information (no per-parameter details in layerwise mode)
            results = {
                'overall_cosine_similarity': float(cosine_sim) if not np.isnan(cosine_sim) else float('nan'),
                'overall_conflict': float(max(0, -cosine_sim)) if not np.isnan(cosine_sim) else 0.0,
                'use_layerwise': True,
                'note': 'Per-parameter similarities not available in layerwise mode (memory optimization)'
            }

            model.train(was_training)
            return results
        
        task1_batch = self._to_model_device(model, task1_batch)
        task1_batch = self._with_labels(task1_batch)

        # Get gradients for task 1 with parameter names
        model.zero_grad(set_to_none=True)
        outputs1 = model(**task1_batch, use_cache=False) if 'use_cache' in inspect.signature(model.forward).parameters else model(**task1_batch)
        outputs1.loss.backward()
        grads1_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads1_dict[name] = param.grad.clone().flatten().cpu()
                param.grad = None  # Free GPU memory immediately

        # Free first batch and computation graph before loading second batch
        del outputs1, task1_batch
        model.zero_grad(set_to_none=True)

        # Force GPU memory cleanup before loading second batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # NOW load task2_batch to GPU
        task2_batch = self._to_model_device(model, task2_batch)
        task2_batch = self._with_labels(task2_batch)

        # Get gradients for task 2 with parameter names
        model.zero_grad(set_to_none=True)
        outputs2 = model(**task2_batch, use_cache=False) if 'use_cache' in inspect.signature(model.forward).parameters else model(**task2_batch)
        outputs2.loss.backward()
        grads2_dict = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grads2_dict[name] = param.grad.clone().flatten().cpu()
                param.grad = None  # Free GPU memory immediately

        # Free second batch and computation graph
        del outputs2, task2_batch
        model.zero_grad(set_to_none=True)

        # Clear GPU cache after gradient computation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Find common parameters with gradients (fixes alignment bug)
        common_params = sorted(grads1_dict.keys() & grads2_dict.keys())
        
        if len(common_params) == 0:
            model.train(was_training)
            return {
                'error': 'No overlapping parameters with gradients',
                'overall_gradient_conflict': float('nan'),
                'gradient_cosine_similarity': float('nan')
            }
        
        # Compute global cosine on CPU (streamed), avoids giant GPU tensors
        dot = 0.0
        n1 = 0.0
        n2 = 0.0
        for name in common_params:
            g1 = grads1_dict[name].to(torch.float32)
            g2 = grads2_dict[name].to(torch.float32)
            dot += float((g1 * g2).sum())
            n1 += float(g1.pow(2).sum())
            n2 += float(g2.pow(2).sum())

        if n1 < 1e-12 or n2 < 1e-12:
            cosine_sim = float('nan')
        else:
            cosine_sim = dot / (np.sqrt(n1) * np.sqrt(n2))

        # Compute gradient norms for ratio (using already computed values)
        norm1 = torch.tensor(np.sqrt(n1))
        norm2 = torch.tensor(np.sqrt(n2))
        
        # Compute per-parameter conflicts (aligned by name)
        param_conflicts = []
        param_similarities = {}
        
        for param_name in common_params:
            # Cast to float32 for numerical stability on CPU
            g1 = grads1_dict[param_name].to(torch.float32)
            g2 = grads2_dict[param_name].to(torch.float32)

            # Zero-norm guard for per-parameter similarity
            if g1.norm() < 1e-12 or g2.norm() < 1e-12:
                param_sim = float('nan')
            else:
                param_sim = torch.nn.functional.cosine_similarity(
                    g1.unsqueeze(0),
                    g2.unsqueeze(0)
                ).item()
            
            param_similarities[param_name] = param_sim
            # PCGrad-style conflict: only negative cosine indicates conflict
            if not np.isnan(param_sim):
                param_conflicts.append(max(0, -param_sim))
        
        # Filter valid similarities for statistics
        valid_param_sims = [v for v in param_similarities.values() if not np.isnan(v)]
        
        result = {
            'overall_gradient_conflict': max(0, -cosine_sim) if not np.isnan(cosine_sim) else float('nan'),
            'mean_param_conflict': float(np.mean(param_conflicts)) if param_conflicts else float('nan'),
            'max_param_conflict': float(max(param_conflicts)) if param_conflicts else float('nan'),
            'conflicting_params': sum(1 for c in param_conflicts if c > 0),
            'gradient_magnitude_ratio': norm1.item() / (norm2.item() + 1e-10),
            'gradient_cosine_similarity': cosine_sim,
            'param_similarities': param_similarities,  # Dict of param_name -> similarity
            'num_common_params': len(common_params),
            'num_valid_similarities': len(valid_param_sims)
        }

        # Free gradient dictionaries to release memory
        del grads1_dict, grads2_dict

        model.train(was_training)
        return result
    
    def compute_gradient_conflict_matrix_multi(
        self,
        model,
        task_batches: Dict[str, Dict[str, torch.Tensor]],
        eval_mode: bool = True,  # Use eval mode for deterministic gradient computation
        return_type: str = 'matrix'  # 'matrix' or 'dict'
    ) -> Union[np.ndarray, Dict[str, Dict[str, float]]]:
        """
        Compute TRUE gradient conflict matrix for multiple tasks.
        Returns NxN matrix where N is the number of tasks.

        Args:
            model: Model to analyze
            task_batches: Dict mapping task names to batches
                         e.g., {'math': batch1, 'coding': batch2, 'writing': batch3}
            eval_mode: If True, use eval mode for deterministic measurement
            return_type: 'matrix' for numpy array, 'dict' for nested dict

        Returns:
            If return_type='matrix': NxN numpy array where element [i,j] is
                                     the conflict between task i and task j
            If return_type='dict': Nested dict task_name -> task_name -> conflict

        Example:
            >>> tasks = {'math': math_batch, 'code': code_batch, 'text': text_batch}
            >>> matrix = compute_gradient_conflict_matrix_multi(model, tasks)
            >>> # matrix[0,1] = conflict between math and code
        """
        import logging
        logger = logging.getLogger(__name__)

        # Get task names in sorted order for consistent indexing
        task_names = sorted(task_batches.keys())
        n_tasks = len(task_names)

        # Initialize conflict matrix
        conflict_matrix = np.zeros((n_tasks, n_tasks))
        conflict_dict = {}

        # Compute pairwise conflicts
        logger.info(f"Computing gradient conflict matrix for {n_tasks} tasks")

        for i, task1 in enumerate(task_names):
            conflict_dict[task1] = {}

            for j, task2 in enumerate(task_names):
                if i == j:
                    # Self-conflict is 0 (perfect alignment)
                    conflict_matrix[i, j] = 0.0
                    conflict_dict[task1][task2] = 0.0
                elif j > i:
                    # Compute conflict for upper triangle
                    try:
                        result = self.compute_gradient_conflict_pair(
                            model=model,
                            task1_batch=task_batches[task1],
                            task2_batch=task_batches[task2],
                            eval_mode=eval_mode,
                            use_layerwise=True  # Memory efficient
                        )

                        # Extract global conflict value
                        conflict_value = result.get('global_conflict', 0.0)

                        # Store in matrix (symmetric)
                        conflict_matrix[i, j] = conflict_value
                        conflict_matrix[j, i] = conflict_value

                        # Store in dict
                        conflict_dict[task1][task2] = conflict_value
                        if task2 not in conflict_dict:
                            conflict_dict[task2] = {}
                        conflict_dict[task2][task1] = conflict_value

                        logger.debug(f"Conflict {task1} <-> {task2}: {conflict_value:.4f}")

                    except Exception as e:
                        logger.warning(f"Failed to compute conflict between {task1} and {task2}: {e}")
                        # Set NaN for failed computations
                        conflict_matrix[i, j] = np.nan
                        conflict_matrix[j, i] = np.nan
                        conflict_dict[task1][task2] = np.nan
                        if task2 not in conflict_dict:
                            conflict_dict[task2] = {}
                        conflict_dict[task2][task1] = np.nan

        # Return based on requested type
        if return_type == 'matrix':
            return conflict_matrix
        else:
            return conflict_dict
    
    def compute_raw_gradient_conflict(
        self,
        model,
        math_batch: Dict[str, torch.Tensor],
        general_batch: Dict[str, torch.Tensor],
        n_samples: int = 20,  # Increased to min of 20 for statistical validity
        eval_mode: bool = True,  # Use eval mode for deterministic gradient computation
        subsample_ratio: Optional[float] = None,  # Auto-detect based on batch size
        use_layerwise: bool = True,  # Changed to True by default to prevent OOM
        auto_optimize_subsample: bool = True,  # Auto-optimize subsample_ratio based on batch size
        process_all_samples: bool = True  # Process ALL samples in chunks, not just sampling
    ) -> Dict[str, Any]:
        """
        Compute ROBUST GLOBAL gradient conflict (HOW MUCH conflict, with uncertainty).

        Returns mean and std of overall gradient conflict across n_samples with
        optional subsampling for variance estimation.

        Key differences from other functions:
        - Unlike compute_layer_gradient_alignment: Returns GLOBAL scalar (not per-layer)
        - Unlike compute_gradient_conflict_matrix: Multiple samples for ROBUST estimate
        - Unlike compute_gradient_alignment_trajectory: Variance across samples (not time)
        - use_layerwise=True: Memory optimization only, still returns GLOBAL result

        Args:
            model: Model to analyze
            math_batch: First task batch
            general_batch: Second task batch
            n_samples: Number of resampling rounds for variance estimate (default=20 for
                      statistical validity, provides ~95% CI with ±2*std)
            eval_mode: If True, use eval mode for deterministic measurement
            subsample_ratio: Fraction of batch to use per sample (reduce noise)
            use_layerwise: If True, compute gradients layer-by-layer to save memory
                          (still aggregates to one overall cosine, NOT per-layer)

        Statistical Note (ICLR 2026):
            n_samples=20 provides robust estimates with standard error ~0.22σ
            n_samples=5 (old default) had standard error ~0.45σ (too high)
            For publication-quality results, consider n_samples=30-50

        Use this when: Need stable scalar conflict measure with confidence interval
        Output: Mean cosine, std, and PCGrad conflict score across resamples

        See also:
        - compute_layer_gradient_alignment: For WHERE conflicts occur (per-layer)
        - compute_gradient_alignment_trajectory: For WHEN conflicts occur (over time)
        """
        was_training = model.training

        # Always use eval mode for deterministic gradient computation
        model.eval()

        # Auto-optimize subsample_ratio if not explicitly provided
        if subsample_ratio is None and auto_optimize_subsample:
            batch_size = math_batch['input_ids'].size(0)
            subsample_ratio = self._get_optimal_subsample_ratio(batch_size)
            logger.debug(f"Auto-selected subsample_ratio={subsample_ratio:.2f} for batch_size={batch_size}")
        elif subsample_ratio is None:
            subsample_ratio = 0.5  # Default fallback

        cosine_sims = []

        # Process ALL samples if requested (new default behavior)
        if process_all_samples:
            batch_size = math_batch['input_ids'].size(0)
            # Determine chunk size based on memory constraints
            if use_layerwise:
                # Layerwise is more memory efficient, can use larger chunks
                chunk_size = min(32, batch_size)  # Process up to 32 at a time
            else:
                # Full gradient computation needs smaller chunks
                chunk_size = min(16, batch_size)  # Process up to 16 at a time

            logger.info(f"Processing ALL {batch_size} samples in chunks of {chunk_size} (process_all_samples=True)")

            # Define process function for chunking
            def process_chunk(model, chunk1, chunk2):
                if use_layerwise:
                    return self._compute_layerwise_gradient_conflict(model, chunk1, chunk2)
                else:
                    # Original full gradient computation
                    chunk1 = self._to_model_device(model, chunk1)
                    chunk2 = self._to_model_device(model, chunk2)
                    chunk1 = self._with_labels(chunk1)
                    chunk2 = self._with_labels(chunk2)

                    # Compute gradients for chunk1
                    model.zero_grad(set_to_none=True)
                    outputs1 = model(**chunk1, use_cache=False) if 'use_cache' in inspect.signature(model.forward).parameters else model(**chunk1)
                    outputs1.loss.backward()
                    grads1 = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

                    # Compute gradients for chunk2
                    model.zero_grad(set_to_none=True)
                    outputs2 = model(**chunk2, use_cache=False) if 'use_cache' in inspect.signature(model.forward).parameters else model(**chunk2)
                    outputs2.loss.backward()
                    grads2 = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])

                    # Compute cosine similarity
                    cosine_sim = torch.nn.functional.cosine_similarity(grads1.unsqueeze(0), grads2.unsqueeze(0)).item()
                    model.zero_grad(set_to_none=True)
                    return cosine_sim

            # Process all samples and get aggregated result
            result = self._process_all_samples_in_chunks(
                model, math_batch, general_batch,
                chunk_size=chunk_size,
                process_fn=process_chunk,
                aggregate_method='mean'
            )

            if result is not None:
                # For compatibility, wrap single result as if from multiple samples
                mean_cosine = float(result) if not isinstance(result, dict) else float(result.get('cosine_similarity', result))
                # Since we processed all samples, std is 0 (no sampling variance)
                model.train(was_training)
                return {
                    'raw_conflict_score': float(max(0, -mean_cosine)),
                    'raw_cosine_similarity': mean_cosine,
                    'raw_cosine_similarity_std': 0.0,  # No sampling variance when using all data
                    'n_samples': 1,  # Treated as one complete pass
                    'total_samples_processed': batch_size,
                    'chunk_size_used': chunk_size,
                    'eval_mode': eval_mode,
                    'process_all_samples': True,
                    'use_layerwise': use_layerwise
                }
            else:
                logger.error("Failed to process any chunks")
                model.train(was_training)
                return {
                    'raw_conflict_score': np.nan,
                    'raw_cosine_similarity': np.nan,
                    'raw_cosine_similarity_std': np.nan,
                    'error': 'Failed to process any chunks'
                }

        # Original sampling-based approach (backward compatibility)
        # Wrap in try-finally to ensure cleanup even on errors
        try:
            for sample_idx in range(n_samples):
                if subsample_ratio < 1.0:
                    sub_size = max(2, int(math_batch['input_ids'].size(0) * subsample_ratio))
                    mb = self._take(math_batch, sub_size)
                    gb = self._take(general_batch, sub_size)
                else:
                    mb = self._shuffle_all(math_batch)
                    gb = self._shuffle_all(general_batch)

                if use_layerwise:
                    # Don't move batches to device here - let layerwise function handle it
                    # for proper memory staging (one batch at a time)
                    tracker = get_tracker()
                    if tracker:
                        tracker.take_snapshot('before_layerwise_conflict')
                    cosine_sim = self._compute_layerwise_gradient_conflict(
                        model, mb, gb
                    )
                    if tracker:
                        tracker.compare_snapshots('before_layerwise_conflict', 'after_layerwise_conflict')
                else:
                    # Original full gradient computation
                    tracker = get_tracker()

                    # Track batch movement to GPU
                    if tracker:
                        log_memory_state("Before moving math batch to GPU")
                    mb = self._to_model_device(model, mb)
                    if tracker:
                        log_memory_state("After moving math batch to GPU")

                    model.zero_grad(set_to_none=True)

                    # Track forward pass
                    if tracker:
                        # Log detailed info before forward pass
                        logger.info(f"[GRADIENT DEBUG] Math batch info:")
                        logger.info(f"  - Input IDs shape: {mb['input_ids'].shape}")
                        logger.info(f"  - Device: {mb['input_ids'].device}")
                        logger.info(f"  - Dtype: {mb['input_ids'].dtype}")
                        log_memory_state("Before math forward pass")

                        with tracker.track_context('math_forward_pass'):
                            # Log exact allocation point
                            tracker.log_allocation(0, "math_forward_start", {"batch_shape": str(mb['input_ids'].shape)})
                            math_outputs = model(**self._with_labels(mb))
                            math_loss = math_outputs.loss
                            tracker.log_allocation(0, "math_forward_end", {"loss": float(math_loss.item())})

                        log_memory_state("After math forward pass")
                    else:
                        math_outputs = model(**self._with_labels(mb))
                        math_loss = math_outputs.loss

                    # Track backward pass
                    if tracker:
                        with tracker.track_context('math_backward_pass'):
                            math_loss.backward()
                    else:
                        math_loss.backward()

                    # Offload math gradients to CPU to minimize GPU memory usage
                    if tracker:
                        log_memory_state("Before offloading math gradients to CPU")
                    math_grads = {}
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            # Move to CPU in original dtype to minimize memory
                            math_grads[name] = param.grad.detach().to("cpu", non_blocking=True)
                            param.grad = None  # Free GPU memory immediately
                    if tracker:
                        log_memory_state("After offloading math gradients to CPU")

                    if len(math_grads) == 0:
                        del math_grads  # Free the dictionary to prevent memory leak
                        continue

                    # Free first-pass references before loading second batch
                    del math_outputs, math_loss, mb
                    model.zero_grad(set_to_none=True)

                    # Force GPU memory cleanup before loading second batch
                    if torch.cuda.is_available():
                        if tracker:
                            log_memory_state("Before empty_cache")
                        torch.cuda.empty_cache()
                        if tracker:
                            log_memory_state("After empty_cache")

                    # NOW load the second batch to GPU (after cleanup)
                    if tracker:
                        log_memory_state("Before moving general batch to GPU")
                    gb = self._to_model_device(model, gb)
                    if tracker:
                        log_memory_state("After moving general batch to GPU")

                    # Compute general gradients (GPU memory now free from math grads)
                    model.zero_grad(set_to_none=True)

                    # Track general forward pass
                    if tracker:
                        with tracker.track_context('general_forward_pass'):
                            general_outputs = model(**self._with_labels(gb))
                            general_loss = general_outputs.loss
                    else:
                        general_outputs = model(**self._with_labels(gb))
                        general_loss = general_outputs.loss

                    # Track general backward pass
                    if tracker:
                        with tracker.track_context('general_backward_pass'):
                            general_loss.backward()
                    else:
                        general_loss.backward()

                    # Accumulate on CPU in float32 for numerical stability
                    dot_product = 0.0
                    math_norm_sq = 0.0
                    general_norm_sq = 0.0

                    for name, param in model.named_parameters():
                        if param.grad is not None and name in math_grads:
                            # Move general grad to CPU and compute in float32
                            math_g = math_grads[name].reshape(-1).float()  # CPU, upcast to float32
                            general_g = param.grad.detach().to("cpu").reshape(-1).float()  # CPU, upcast to float32

                            # Accumulate on CPU (no GPU sync needed)
                            dot_product += float((math_g * general_g).sum())
                            math_norm_sq += float(math_g.pow(2).sum())
                            general_norm_sq += float(general_g.pow(2).sum())

                            # Free the stored gradient immediately after use
                            del math_grads[name]

                    # Clear any remaining math_grads entries (defensive cleanup)
                    math_grads.clear()

                    # Clean up second pass references
                    del general_outputs, general_loss, gb
                    model.zero_grad(set_to_none=True)  # Free general grads from GPU

                    # No GPU sync needed - already accumulated as Python floats

                    # Compute cosine similarity from accumulated values
                    if math_norm_sq > 0 and general_norm_sq > 0:
                        cosine_sim = dot_product / (np.sqrt(math_norm_sq) * np.sqrt(general_norm_sq))
                    else:
                        continue

                if cosine_sim is not None and not np.isnan(cosine_sim):
                    cosine_sims.append(cosine_sim)

                model.zero_grad(set_to_none=True)

        finally:
            # Ensure cleanup happens even if an error occurs
            model.zero_grad(set_to_none=True)

        model.train(was_training)

        if len(cosine_sims) == 0:
            return {
                'raw_conflict_score': np.nan,
                'raw_cosine_similarity': np.nan,
                'raw_cosine_similarity_std': np.nan,
                'eval_mode': eval_mode
            }

        mean_cosine = float(np.mean(cosine_sims))
        
        return {
            'raw_conflict_score': float(max(0, -mean_cosine)),  # PCGrad-style: clamp to [0, inf)
            'raw_cosine_similarity': mean_cosine,  # Raw value for analysis
            'raw_cosine_similarity_std': float(np.std(cosine_sims)),
            'n_samples': len(cosine_sims),
            'eval_mode': eval_mode,
            'subsample_ratio': subsample_ratio,
            'use_layerwise': use_layerwise
        }
    
    def _compute_layerwise_gradient_conflict(self, model, math_batch, general_batch,
                                            target_batch_size=32, target_seq_len=None):
        """
        Compute gradient conflict with memory-efficient single forward pass.
        Computes gradients once and processes them layer-by-layer for memory efficiency.

        NOTE: Returns GLOBAL conflict (not per-layer) - the layerwise processing
        is purely for memory efficiency.

        Args:
            model: The model to analyze
            math_batch: First task batch
            general_batch: Second task batch
            target_batch_size: Target batch size for forward pass
            target_seq_len: Target sequence length (None = use original)
        """
        # This implementation uses a SINGLE forward pass per batch
        # and processes the resulting gradients layer-by-layer

        tracker = get_tracker()
        if tracker:
            log_memory_state("Starting layerwise gradient conflict computation")

        # Check if model supports use_cache parameter for memory optimization
        supports_use_cache = 'use_cache' in inspect.signature(model.forward).parameters

        # Adjust batch size and sequence length as needed
        actual_batch_size = math_batch['input_ids'].shape[0] if 'input_ids' in math_batch else 0
        actual_seq_len = math_batch['input_ids'].shape[1] if 'input_ids' in math_batch else 0

        # Adjust batch size
        if actual_batch_size != target_batch_size:
            if actual_batch_size > target_batch_size:
                # Reduce batch size
                math_batch = {k: v[:target_batch_size] if torch.is_tensor(v) else v
                             for k, v in math_batch.items()}
                general_batch = {k: v[:target_batch_size] if torch.is_tensor(v) else v
                               for k, v in general_batch.items()}
            else:
                # Replicate to reach target batch size
                replications = (target_batch_size + actual_batch_size - 1) // actual_batch_size
                math_batch = {k: v.repeat(replications, *([1] * (len(v.shape) - 1)))[:target_batch_size]
                             if torch.is_tensor(v) else v for k, v in math_batch.items()}
                general_batch = {k: v.repeat(replications, *([1] * (len(v.shape) - 1)))[:target_batch_size]
                               if torch.is_tensor(v) else v for k, v in general_batch.items()}
            actual_batch_size = target_batch_size

        # Adjust sequence length if specified
        if target_seq_len is not None and actual_seq_len != target_seq_len:
            if actual_seq_len > target_seq_len:
                # Truncate sequences
                math_batch = {k: v[:, :target_seq_len] if torch.is_tensor(v) and len(v.shape) > 1 else v
                             for k, v in math_batch.items()}
                general_batch = {k: v[:, :target_seq_len] if torch.is_tensor(v) and len(v.shape) > 1 else v
                               for k, v in general_batch.items()}
            else:
                # Pad sequences (with padding token or zeros)
                # This is more complex and depends on the model's tokenizer
                logger.warning(f"Sequence padding from {actual_seq_len} to {target_seq_len} not implemented")
            actual_seq_len = min(actual_seq_len, target_seq_len)

        logger.info(f"[GRADIENT DEBUG] Layerwise conflict - batch config: {actual_batch_size} × {actual_seq_len}")

        math_batch = self._to_model_device(model, math_batch)
        math_batch = self._with_labels(math_batch)

        model.zero_grad(set_to_none=True)

        # Check memory before forward pass
        if torch.cuda.is_available():
            mem_before = torch.cuda.memory_allocated() / 1e9
            logger.info(f"[GRADIENT DEBUG] Memory before math forward: {mem_before:.2f}GB")

        with torch.enable_grad():
            # Single forward pass for math batch
            if supports_use_cache:
                math_outputs = model(**math_batch, use_cache=False)
            else:
                math_outputs = model(**math_batch)
            math_loss = math_outputs.loss
            math_loss.backward()

        # Store all math gradients on CPU immediately
        math_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Move to CPU immediately to free GPU memory
                math_grads[name] = param.grad.detach().to("cpu", non_blocking=True)
                param.grad = None  # Free GPU memory

        # Free math batch and computation graph
        del math_batch, math_outputs, math_loss
        model.zero_grad(set_to_none=True)

        # Force GPU memory cleanup before loading second batch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Second pass: compute general gradients and accumulate
        general_batch = self._to_model_device(model, general_batch)
        general_batch = self._with_labels(general_batch)

        model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            # Single forward pass for general batch
            if supports_use_cache:
                general_outputs = model(**general_batch, use_cache=False)
            else:
                general_outputs = model(**general_batch)
            general_loss = general_outputs.loss
            general_loss.backward()

        # Accumulate cosine similarity on CPU layer-by-layer
        # This reduces peak memory by processing gradients incrementally
        total_dot_product = 0.0
        total_math_norm_sq = 0.0
        total_general_norm_sq = 0.0

        # Process all parameters (no need for layer grouping since we do single pass)
        for name, param in model.named_parameters():
            if param.grad is not None and name in math_grads:
                # Move general gradient to CPU and compute similarity
                math_g = math_grads[name].reshape(-1).float()
                general_g = param.grad.detach().to("cpu").reshape(-1).float()

                # Accumulate on CPU
                total_dot_product += float((math_g * general_g).sum())
                total_math_norm_sq += float(math_g.pow(2).sum())
                total_general_norm_sq += float(general_g.pow(2).sum())

                # Free memory immediately
                del math_grads[name]
                param.grad = None

        # Clean up
        del general_batch, general_outputs, general_loss
        model.zero_grad(set_to_none=True)

        # Compute overall cosine similarity
        if total_math_norm_sq > 0 and total_general_norm_sq > 0:
            cosine_sim = total_dot_product / (np.sqrt(total_math_norm_sq) * np.sqrt(total_general_norm_sq))
            return cosine_sim
        else:
            return None

    # ============= MULTI-SCALE GRADIENT ANALYSIS =============

    def compute_multiscale_raw_gradient_conflict(
        self,
        model,
        math_batch: Dict[str, torch.Tensor],
        general_batch: Dict[str, torch.Tensor],
        eval_mode: bool = True,  # Use eval mode for deterministic gradient computation
        batch_size_configs: Optional[List[Tuple]] = None,
        auto_detect_gpu: bool = True,
        force_conservative: bool = False
    ) -> Dict[str, Any]:
        """
        Compute gradient conflict at multiple scales for comprehensive analysis.
        Automatically detects GPU and uses optimal batch sizes unless overridden.

        Args:
            model: Model to analyze
            math_batch: Math task batch
            general_batch: General task batch
            eval_mode: Whether to use eval mode
            batch_size_configs: Optional custom batch configurations
            auto_detect_gpu: If True, automatically detect GPU and use optimal configs
            force_conservative: If True, use conservative configs regardless of GPU

        Returns:
            Combined score and individual results for diagnostic purposes.
        """
        import gc

        # Get optimal configurations based on GPU or use provided configs
        if batch_size_configs is not None:
            configs = batch_size_configs
            logger.info("Using custom batch size configurations")
        elif auto_detect_gpu:
            configs = self._get_optimal_batch_configs(force_conservative=force_conservative)
        else:
            # Use static configs (no dynamic detection)
            configs = self._get_optimal_batch_configs()
            logger.info("Using static batch configurations for reproducibility")

        results = {}
        weighted_scores = []

        for batch_size, seq_len, weight, name in configs:
            # More realistic memory estimate (multiply by 4x for actual usage)
            base_estimate = (batch_size * 24 * 16 * seq_len * seq_len * 4 / 1e9 +
                           batch_size * seq_len * 1024 * 24 * 4 * 2 / 1e9) * 1.3
            realistic_estimate = base_estimate * 4  # Actual usage is ~4x higher

            logger.info(f"[{name}] Processing {batch_size} math + {batch_size} general samples, {seq_len} tokens each")
            logger.info(f"[{name}] Estimated memory: ~{realistic_estimate:.1f} GB (base: {base_estimate:.1f} GB)")

            # Log current memory state
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / 1e9
                free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
                logger.info(f"[{name}] Current GPU: {allocated:.1f}GB allocated, {free:.1f}GB free")

            # Clear GPU cache before each configuration
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Prepare batches for this configuration
            # Note: Input batches should be created at moderate length (e.g., 512 tokens)
            # This function will truncate for shorter configs or pad for longer ones
            config_math_batch = self._prepare_batch_for_config(math_batch, batch_size, seq_len,
                                                               sampling_seed=42,  # For reproducibility
                                                               allow_duplication=True)  # Allow but warn
            config_general_batch = self._prepare_batch_for_config(general_batch, batch_size, seq_len,
                                                                 sampling_seed=42,
                                                                 allow_duplication=True)

            # Use layerwise computation for memory efficiency
            cosine_sim = self._compute_layerwise_gradient_conflict(
                model, config_math_batch, config_general_batch,
                target_batch_size=batch_size, target_seq_len=seq_len
            )

            if cosine_sim is not None and not np.isnan(cosine_sim):
                conflict_score = float(max(0, -cosine_sim))
                results[name] = {
                    'cosine_similarity': float(cosine_sim),
                    'conflict_score': conflict_score,
                    'batch_size': batch_size,
                    'seq_len': seq_len,
                    'weight': weight
                }
                weighted_scores.append((conflict_score, weight))
            else:
                results[name] = {
                    'error': 'Failed to compute',
                    'batch_size': batch_size,
                    'seq_len': seq_len
                }

        # Compute combined score
        if weighted_scores:
            combined_score = sum(score * w for score, w in weighted_scores) / sum(w for _, w in weighted_scores)
        else:
            combined_score = float('nan')

        # Compute agreement between scales
        if len(results) >= 2:
            scores = [r['conflict_score'] for r in results.values() if 'conflict_score' in r]
            if len(scores) >= 2:
                agreement = 1.0 - (np.std(scores) / (np.mean(scores) + 1e-10))
            else:
                agreement = float('nan')
        else:
            agreement = float('nan')

        return {
            'combined_conflict_score': combined_score,
            'agreement_score': agreement,
            'high_confidence': agreement > 0.8,
            'scales': results,
            'interpretation': self._interpret_multiscale_results(results)
        }

    def compute_multiscale_layer_gradient_alignment(
        self,
        model,
        math_batch: Dict[str, torch.Tensor],
        general_batch: Dict[str, torch.Tensor],
        eval_mode: bool = False  # CRITICAL FIX: Changed from True - must use train mode for gradient computation
    ) -> Dict[str, Any]:
        """
        Compute per-layer gradient alignment at multiple scales.
        Note: In layerwise mode, we get global scores rather than per-layer breakdowns,
        but multiple scales still provide valuable diagnostic information.
        """
        import gc

        configs = [
            (32, 256, 0.30, "tokens_0_to_256"),     # Short sequences, fast
            (8, 1024, 0.20, "tokens_0_to_1024"),   # Standard sequences
            (3, 2048, 0.15, "tokens_0_to_2048")      # Long sequences - reduced to 3 to avoid OOM
        ]

        results = {}

        for batch_size, seq_len, weight, name in configs:
            logger.info(f"[MULTISCALE] Computing layer alignment for {name}: {batch_size} samples × {seq_len} tokens")

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Prepare batches
            config_math_batch = self._prepare_batch_for_config(math_batch, batch_size, seq_len,
                                                               sampling_seed=42,  # For reproducibility
                                                               allow_duplication=True)  # Allow but warn
            config_general_batch = self._prepare_batch_for_config(general_batch, batch_size, seq_len,
                                                                 sampling_seed=42,
                                                                 allow_duplication=True)

            # Compute with layerwise efficiency
            cosine_sim = self._compute_layerwise_gradient_conflict(
                model, config_math_batch, config_general_batch,
                target_batch_size=batch_size, target_seq_len=seq_len
            )

            if cosine_sim is not None and not np.isnan(cosine_sim):
                results[name] = {
                    'global_cosine_similarity': float(cosine_sim),
                    'global_conflict': float(max(0, -cosine_sim)),
                    'config': f"{batch_size}×{seq_len}"
                }
            else:
                results[name] = {'error': 'Computation failed'}

        return {
            'multiscale_analysis': results,
            'position_dependent_conflict': self._check_position_dependency(results)
        }

    def compute_multiscale_gradient_conflict_pair(
        self,
        model,
        task1_batch: Dict[str, torch.Tensor],
        task2_batch: Dict[str, torch.Tensor],
        eval_mode: bool = False  # CRITICAL FIX: Changed from True - must use train mode for gradient computation
    ) -> Dict[str, Any]:
        """
        Compute pairwise gradient conflicts at multiple scales.
        """
        import gc


        configs = [
            (32, 256, 0.30, "tokens_0_to_256"),     # Short sequences, fast
            (8, 1024, 0.20, "tokens_0_to_1024"),   # Standard sequences
            (3, 2048, 0.15, "tokens_0_to_2048")      # Long sequences - reduced to 3 to avoid OOM
        ]
        results = {}

        for batch_size, seq_len, weight, name in configs:
            logger.info(f"[MULTISCALE] Computing pair conflict for {name}: {batch_size} samples × {seq_len} tokens")

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            # Prepare batches
            config_task1_batch = self._prepare_batch_for_config(task1_batch, batch_size, seq_len,
                                                               sampling_seed=42,
                                                               allow_duplication=True)
            config_task2_batch = self._prepare_batch_for_config(task2_batch, batch_size, seq_len,
                                                               sampling_seed=42,
                                                               allow_duplication=True)

            # Compute conflict
            cosine_sim = self._compute_layerwise_gradient_conflict(
                model, config_task1_batch, config_task2_batch,
                target_batch_size=batch_size, target_seq_len=seq_len
            )

            if cosine_sim is not None and not np.isnan(cosine_sim):
                results[name] = {
                    'overall_cosine_similarity': float(cosine_sim),
                    'overall_conflict': float(max(0, -cosine_sim)),
                    'config': f"{batch_size}×{seq_len}",
                    'weight': weight
                }
            else:
                results[name] = {'error': 'Computation failed'}

        return {
            'multiscale_pair_analysis': results,
            'consensus_conflict': self._compute_consensus_score(results)
        }

    def compute_multiscale_gradient_conflict_pcgrad(
        self,
        model,
        batch1: Dict[str, torch.Tensor],
        batch2: Dict[str, torch.Tensor],
        eval_mode: bool = True,  # Use eval mode for deterministic gradient computation
        process_all_samples: bool = True  # Process ALL samples at each scale
    ) -> Dict[str, Any]:
        """
        Compute PCGrad-style gradient conflicts at multiple scales.
        Now processes ALL samples in chunks instead of just sampling.
        """
        import gc

        # Conservative chunk sizes for memory safety

        configs = [
            (32, 256, 0.30, "tokens_0_to_256"),     # Short sequences, fast
            (8, 1024, 0.20, "tokens_0_to_1024"),   # Standard sequences
            (3, 2048, 0.15, "tokens_0_to_2048")      # Long sequences - reduced to 3 to avoid OOM
        ]

        results = {}
        pcgrad_scores = []
        total_samples = batch1['input_ids'].shape[0]

        for chunk_size, seq_len, weight, name in configs:
            logger.info(f"[MULTISCALE] Computing PCGrad conflict for {name}: processing {total_samples} samples in chunks of {chunk_size} × {seq_len} tokens")

            if process_all_samples:
                # Process ALL samples in chunks
                all_conflicts = []
                num_chunks = (total_samples + chunk_size - 1) // chunk_size

                for chunk_idx in range(num_chunks):
                    start_idx = chunk_idx * chunk_size
                    end_idx = min(start_idx + chunk_size, total_samples)
                    actual_chunk_size = end_idx - start_idx

                    if chunk_idx % 10 == 0:
                        logger.debug(f"  Processing chunk {chunk_idx+1}/{num_chunks} (samples {start_idx}-{end_idx-1})")

                    # Extract chunk with proper sequence length
                    chunk1 = self._prepare_batch_for_config(
                        {k: v[start_idx:end_idx] if torch.is_tensor(v) else v
                         for k, v in batch1.items()},
                        actual_chunk_size, seq_len,
                        allow_duplication=False  # No duplication needed within chunks
                    )
                    chunk2 = self._prepare_batch_for_config(
                        {k: v[start_idx:end_idx] if torch.is_tensor(v) else v
                         for k, v in batch2.items()},
                        actual_chunk_size, seq_len,
                        allow_duplication=False
                    )

                    # Compute conflict for this chunk
                    cosine_sim = self._compute_layerwise_gradient_conflict(
                        model, chunk1, chunk2,
                        target_batch_size=actual_chunk_size, target_seq_len=seq_len
                    )

                    if cosine_sim is not None and not np.isnan(cosine_sim):
                        all_conflicts.append(float(max(0, -cosine_sim)))

                    # Clear memory between chunks
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

                # Aggregate results from all chunks
                if all_conflicts:
                    avg_conflict = sum(all_conflicts) / len(all_conflicts)
                    avg_cosine = -(sum(all_conflicts) / len(all_conflicts))  # Reverse to get cosine
                    results[name] = {
                        'gradient_cosine_similarity': float(avg_cosine),
                        'pcgrad_conflict_score': avg_conflict,
                        'conflict_detected': avg_conflict > 0,
                        'config': f"{chunk_size}×{seq_len}",
                        'weight': weight,
                        'total_samples_processed': total_samples,
                        'num_chunks': num_chunks
                    }
                    pcgrad_scores.append((avg_conflict, weight))
                else:
                    results[name] = {'error': 'All chunks failed'}

            else:
                # Original sampling approach (backward compatibility)
                config_batch1 = self._prepare_batch_for_config(batch1, chunk_size, seq_len,
                                                               sampling_seed=42,
                                                               allow_duplication=True)
                config_batch2 = self._prepare_batch_for_config(batch2, chunk_size, seq_len,
                                                               sampling_seed=42,
                                                               allow_duplication=True)

                cosine_sim = self._compute_layerwise_gradient_conflict(
                    model, config_batch1, config_batch2,
                    target_batch_size=chunk_size, target_seq_len=seq_len
                )

                if cosine_sim is not None and not np.isnan(cosine_sim):
                    pcgrad_conflict = float(max(0, -cosine_sim))
                    results[name] = {
                        'gradient_cosine_similarity': float(cosine_sim),
                        'pcgrad_conflict_score': pcgrad_conflict,
                        'conflict_detected': pcgrad_conflict > 0,
                        'config': f"{chunk_size}×{seq_len}",
                        'weight': weight
                    }
                    pcgrad_scores.append((pcgrad_conflict, weight))
                else:
                    results[name] = {'error': 'Computation failed'}

        # Compute weighted PCGrad score
        if pcgrad_scores:
            weighted_pcgrad = sum(s * w for s, w in pcgrad_scores) / sum(w for _, w in pcgrad_scores)
        else:
            weighted_pcgrad = float('nan')

        return {
            'multiscale_pcgrad_conflict': weighted_pcgrad,
            'scales': results,
            'pcgrad_intervention_recommended': weighted_pcgrad > 0.3
        }

    # Helper methods for multi-scale analysis

    def _prepare_batch_for_config(self, batch: Dict[str, torch.Tensor],
                                  target_batch_size: int, target_seq_len: int,
                                  pad_token_id: Optional[int] = None,
                                  padding_side: str = 'right',
                                  sampling_seed: Optional[int] = None,
                                  allow_duplication: bool = False) -> Dict[str, torch.Tensor]:
        """
        Prepare batch for specific configuration by adjusting batch size and sequence length.

        Args:
            batch: Input batch dictionary
            target_batch_size: Target batch size
            target_seq_len: Target sequence length
            pad_token_id: Token ID to use for padding (defaults to 0)
            padding_side: 'left' or 'right' padding (default: 'right')
            sampling_seed: Random seed for reproducible sampling
            allow_duplication: If True, allows sample duplication when batch is too small.
                             If False, raises error. Duplication biases gradient statistics!

        Returns:
            Prepared batch with adjusted dimensions

        Raises:
            ValueError: If inputs are invalid or batch size adjustment would cause bias
        """
        # Input validation
        if target_batch_size <= 0 or target_seq_len <= 0:
            raise ValueError(f"Target dimensions must be positive: batch_size={target_batch_size}, seq_len={target_seq_len}")

        if not isinstance(target_batch_size, int) or not isinstance(target_seq_len, int):
            raise ValueError(f"Target dimensions must be integers: batch_size={type(target_batch_size)}, seq_len={type(target_seq_len)}")

        if not batch or 'input_ids' not in batch:
            logger.warning("Batch missing or has no input_ids, returning as-is")
            return batch

        # Set default pad token
        if pad_token_id is None:
            pad_token_id = getattr(self, '_pad_token_id', 0)
            if pad_token_id == 0:
                logger.debug("Using default pad_token_id=0. Consider setting explicitly for your model.")

        prepared_batch = {}
        current_batch_size = batch['input_ids'].shape[0]
        current_seq_len = batch['input_ids'].shape[1] if len(batch['input_ids'].shape) > 1 else 1

        # Get reference device from first tensor
        device = None
        for value in batch.values():
            if torch.is_tensor(value):
                device = value.device
                break

        # Set random seed for reproducibility
        if sampling_seed is not None:
            torch.manual_seed(sampling_seed)

        # Prepare indices for batch size adjustment
        indices = None
        if current_batch_size != target_batch_size:
            if current_batch_size > target_batch_size:
                # Sample subset deterministically
                if sampling_seed is None:
                    logger.warning(f"Sampling {target_batch_size} from {current_batch_size} samples without seed - results non-reproducible")
                indices = torch.randperm(current_batch_size, device=device)[:target_batch_size]
            else:
                # Handle insufficient samples
                if not allow_duplication:
                    raise ValueError(
                        f"Batch size {current_batch_size} < target {target_batch_size}. "
                        f"Sample duplication would bias gradients. Set allow_duplication=True to override."
                    )
                logger.warning(
                    f"Duplicating samples from {current_batch_size} to {target_batch_size}. "
                    f"This will create correlated gradients and bias statistics!"
                )
                # Cycle through samples to minimize correlation
                reps = target_batch_size // current_batch_size
                remainder = target_batch_size % current_batch_size
                indices = torch.cat([
                    torch.arange(current_batch_size, device=device).repeat(reps),
                    torch.arange(remainder, device=device)
                ])

        for key, value in batch.items():
            if not torch.is_tensor(value):
                prepared_batch[key] = value
                continue

            # Ensure same device
            if value.device != device:
                logger.warning(f"Tensor {key} on different device {value.device} vs {device}")
                value = value.to(device)

            # Get shape info for this specific tensor
            value_shape = value.shape
            value_seq_len = value_shape[1] if len(value_shape) > 1 else None

            # Adjust batch size if needed
            if indices is not None:
                value = value[indices]

            # Adjust sequence length for 2D+ tensors
            if value_seq_len is not None:
                if value_seq_len > target_seq_len:
                    # Truncate
                    value = value[:, :target_seq_len]
                elif value_seq_len < target_seq_len:
                    # Pad based on key type and padding side
                    pad_length = target_seq_len - value_seq_len
                    batch_size = value.shape[0]

                    if key == 'input_ids':
                        # Use specified pad token
                        padding = torch.full((batch_size, pad_length),
                                           pad_token_id, dtype=value.dtype, device=device)
                    elif key == 'attention_mask':
                        # Pad with 0s (masked positions)
                        padding = torch.zeros((batch_size, pad_length),
                                            dtype=value.dtype, device=device)
                    elif key == 'labels':
                        # Pad with -100 (ignored in loss)
                        padding = torch.full((batch_size, pad_length),
                                           -100, dtype=value.dtype, device=device)
                    elif key == 'position_ids':
                        # Continue position sequence
                        start_pos = value[:, -1:] + 1 if value_seq_len > 0 else torch.zeros((batch_size, 1), dtype=value.dtype, device=device)
                        padding = start_pos + torch.arange(pad_length, dtype=value.dtype, device=device).unsqueeze(0)
                    elif key == 'token_type_ids':
                        # Pad with same token type as last token
                        last_type = value[:, -1:] if value_seq_len > 0 else torch.zeros((batch_size, 1), dtype=value.dtype, device=device)
                        padding = last_type.expand(batch_size, pad_length)
                    else:
                        # Default: pad with zeros, preserving dtype
                        padding_shape = list(value.shape)
                        padding_shape[1] = pad_length
                        padding = torch.zeros(padding_shape, dtype=value.dtype, device=device)

                    # Apply padding on specified side
                    if padding_side == 'left':
                        value = torch.cat([padding, value], dim=1)
                    else:
                        value = torch.cat([value, padding], dim=1)

            prepared_batch[key] = value

        return prepared_batch

    def _interpret_multiscale_results(self, results: Dict[str, Any]) -> str:
        """
        Interpret multi-scale results to provide actionable insights.
        """
        # Configuration constants
        CONFLICT_DIFFERENCE_THRESHOLD = 0.2  # Minimum difference to be considered significant
        HIGH_CONFLICT_THRESHOLD = 0.5        # Threshold for "high" conflict
        LOW_CONFLICT_THRESHOLD = 0.2         # Threshold for "low" conflict
        EPSILON = 1e-8                        # For numerical comparisons

        if not results:
            return "No results to interpret"

        # Dynamically collect all available token ranges with valid conflict scores
        token_configs = []
        for key, value in results.items():
            if key.startswith('tokens_0_to_') and isinstance(value, dict):
                conflict_score = value.get('conflict_score')
                if conflict_score is not None and not np.isnan(conflict_score):
                    try:
                        # Extract token count from key (e.g., 'tokens_0_to_256' -> 256)
                        token_count = int(key.split('_')[-1])
                        token_configs.append((token_count, conflict_score))
                    except (ValueError, IndexError):
                        continue

        # Sort by token count to analyze progression
        token_configs.sort(key=lambda x: x[0])

        if not token_configs:
            return "Insufficient valid data for interpretation"

        if len(token_configs) == 1:
            token_count, score = token_configs[0]
            if score > HIGH_CONFLICT_THRESHOLD:
                return f"High conflict at {token_count} tokens: {score:.3f}"
            elif score < LOW_CONFLICT_THRESHOLD:
                return f"Low conflict at {token_count} tokens: {score:.3f}"
            else:
                return f"Moderate conflict at {token_count} tokens: {score:.3f}"

        # Extract key metrics for analysis
        first_tokens, first_score = token_configs[0]
        last_tokens, last_score = token_configs[-1]

        # Calculate overall statistics first
        mean_score = np.mean([score for _, score in token_configs])
        std_score = np.std([score for _, score in token_configs])

        # Check if all values are essentially minimal (prioritize this check)
        if mean_score < LOW_CONFLICT_THRESHOLD:
            # Even if there's some variation, if mean is low, it's minimal conflict
            return (f"Minimal conflict across all lengths: "
                   f"{first_tokens}T ({first_score:.3f}) to {last_tokens}T ({last_score:.3f})")

        # Check if all values are essentially the same (numerical precision)
        if std_score < EPSILON * 10:  # Very small standard deviation
            return (f"Moderate consistent conflict (μ={mean_score:.3f}): "
                   f"{first_tokens}T ({first_score:.3f}) to {last_tokens}T ({last_score:.3f})")

        # Calculate trend if we have enough data points
        if len(token_configs) >= 3:
            token_counts = [tc for tc, _ in token_configs]
            scores = [score for _, score in token_configs]

            # Calculate the actual range of scores
            actual_range = max(scores) - min(scores)

            # Calculate correlation to detect monotonic patterns
            if len(set(scores)) > 1:  # Avoid division by zero in correlation
                correlation, p_value = stats.spearmanr(token_counts, scores)

                # Detect trend with statistical significance
                # Also check that the actual difference is meaningful (not just numerical noise)
                if p_value < 0.05 and actual_range > 0.05:  # Significant AND meaningful
                    if correlation > 0.7:
                        # Strong increasing trend
                        return (f"Conflict increases with sequence length (ρ={correlation:.2f}, p={p_value:.3f}): "
                               f"{first_tokens}T ({first_score:.3f}) → {last_tokens}T ({last_score:.3f})")
                    elif correlation < -0.7:
                        # Strong decreasing trend
                        return (f"Conflict decreases with sequence length (ρ={correlation:.2f}, p={p_value:.3f}): "
                               f"{first_tokens}T ({first_score:.3f}) → {last_tokens}T ({last_score:.3f})")

            # Check for non-monotonic patterns only if variation is meaningful
            if actual_range > 0.05:  # Only report peaks if they're meaningful
                max_score = max(scores)
                max_idx = scores.index(max_score)

                if max_idx not in [0, len(scores)-1]:
                    # Peak in the middle
                    peak_tokens = token_counts[max_idx]
                    # But only report as peak if it's significantly higher
                    min_score = min(scores)
                    if (max_score - min_score) > 0.05:
                        return (f"Conflict peaks at {peak_tokens} tokens ({max_score:.3f}), "
                               f"lower at boundaries: {first_tokens}T ({first_score:.3f}), "
                               f"{last_tokens}T ({last_score:.3f})")

        # Fallback to simple difference analysis
        score_diff = abs(first_score - last_score)

        if score_diff > CONFLICT_DIFFERENCE_THRESHOLD:
            if first_score > last_score:
                return (f"Conflict concentrated early: {first_tokens} tokens ({first_score:.3f}) "
                       f"vs {last_tokens} tokens ({last_score:.3f})")
            else:
                # Build progression string for available intermediate points
                progression = f"{first_tokens}T ({first_score:.3f})"
                if len(token_configs) > 2:
                    # Add one intermediate point for clarity
                    mid_idx = len(token_configs) // 2
                    mid_tokens, mid_score = token_configs[mid_idx]
                    progression += f" → {mid_tokens}T ({mid_score:.3f})"
                progression += f" → {last_tokens}T ({last_score:.3f})"
                return f"Conflict increases with length: {progression}"

        # Check overall conflict level
        mean_score = np.mean([score for _, score in token_configs])

        if mean_score > HIGH_CONFLICT_THRESHOLD:
            score_list = ", ".join([f"{tc}T: {s:.3f}" for tc, s in token_configs[:3]])
            if len(token_configs) > 3:
                score_list += f", ..., {token_configs[-1][0]}T: {token_configs[-1][1]:.3f}"
            return f"Strong conflict throughout: {score_list}"
        elif mean_score < LOW_CONFLICT_THRESHOLD:
            return (f"Minimal conflict across all lengths: "
                   f"{first_tokens}T ({first_score:.3f}) to {last_tokens}T ({last_score:.3f})")
        else:
            return (f"Moderate consistent conflict (μ={mean_score:.3f}): "
                   f"{first_tokens}T ({first_score:.3f}) to {last_tokens}T ({last_score:.3f})")

    def _check_position_dependency(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if conflicts are position-dependent based on multi-scale results.

        Theory: Position dependency means conflict behavior changes with sequence length.
        We analyze this through variance, coefficient of variation, and correlation.
        """
        import re

        scores = []
        token_counts = []
        token_ranges = []

        # Dynamically discover available token ranges
        available_configs = []
        for config_name in results.keys():
            if config_name.startswith('tokens_0_to_'):
                available_configs.append(config_name)

        # Sort by token count for consistent ordering
        available_configs.sort(key=lambda x: int(x.split('_')[-1]) if x.split('_')[-1].isdigit() else 0)

        for config_name in available_configs:
            if config_name in results:
                score = results[config_name].get('global_conflict',
                       results[config_name].get('conflict_score', None))

                # Input validation - skip invalid scores
                if score is None or np.isnan(score) or np.isinf(score):
                    continue

                # Validate score is in expected range [0, 2] from cosine similarity
                if score < 0 or score > 2.1:  # Allow small tolerance
                    warnings.warn(f"Unexpected conflict score {score} outside [0,2] range")
                    continue

                scores.append(float(score))
                token_ranges.append(config_name)

                # Robust token count extraction
                match = re.search(r'tokens_\d+_to_(\d+)', config_name)
                if match:
                    token_counts.append(int(match.group(1)))
                else:
                    # Fallback: try to extract any number from the end
                    try:
                        token_counts.append(int(config_name.split('_')[-1]))
                    except (ValueError, IndexError):
                        token_counts.append(len(scores) * 256)  # Default progression

        if len(scores) < 2:
            return {
                'position_dependent': False,
                'confidence': 0.0,
                'reason': 'Insufficient data points'
            }

        # Convert to numpy arrays for vectorized operations
        scores_arr = np.array(scores)
        positions_arr = np.array(token_counts)

        # Compute basic statistics
        mean_score = np.mean(scores_arr)
        std_score = np.std(scores_arr)
        variance = np.var(scores_arr)

        # Coefficient of Variation (scale-invariant measure)
        cv = std_score / mean_score if mean_score > 1e-6 else 0.0

        # Check for weak monotonicity (allowing equality)
        eps = 1e-6  # Tolerance for floating point comparison
        is_increasing = all(scores[i] <= scores[i+1] + eps for i in range(len(scores)-1))
        is_decreasing = all(scores[i] + eps >= scores[i+1] for i in range(len(scores)-1))

        # Correlation analysis with sequence length
        # Check if scores are constant (no variance) to avoid scipy warning
        scores_are_constant = np.std(scores_arr) < 1e-10

        if len(scores) >= 3 and not scores_are_constant:  # Need at least 3 points and variance for meaningful correlation
            corr_coef, p_value = stats.pearsonr(positions_arr, scores_arr)
            spearman_corr, spearman_p = stats.spearmanr(positions_arr, scores_arr)
        elif scores_are_constant:
            # Scores are constant - correlation undefined
            corr_coef = p_value = spearman_corr = spearman_p = np.nan
        else:
            corr_coef = p_value = spearman_corr = spearman_p = np.nan

        # Determine position dependency based on multiple criteria
        # If scores are constant, there's definitively NO position dependency
        if scores_are_constant:
            position_dependent = False  # Constant scores = no dependency on position
        else:
            # 1. Significant variance (CV > 0.1 suggests >10% variation)
            # 2. Significant correlation with position
            # 3. Clear monotonic trend
            position_dependent = (
                cv > 0.1 or  # >10% coefficient of variation
                (not np.isnan(p_value) and p_value < 0.05 and abs(corr_coef) > 0.5) or  # Significant correlation
                (is_increasing or is_decreasing)  # Clear monotonic trend
            )

        # Calculate confidence based on multiple factors
        # Higher variance/CV = higher confidence in position dependency
        # Significant correlation = higher confidence
        confidence_factors = []
        if cv > 0:
            confidence_factors.append(min(1.0, cv * 5))  # Scale CV to [0,1]
        if not np.isnan(p_value) and p_value < 0.1:
            confidence_factors.append(1.0 - p_value)  # Lower p-value = higher confidence
        if is_increasing or is_decreasing:
            confidence_factors.append(0.8)  # Monotonic pattern adds confidence

        confidence = np.mean(confidence_factors) if confidence_factors else 0.0

        # Determine pattern type
        if scores_are_constant:
            pattern = 'constant'  # All scores identical across positions
        elif is_increasing:
            pattern = 'increasing'
        elif is_decreasing:
            pattern = 'decreasing'
        elif not np.isnan(corr_coef):
            if abs(corr_coef) > 0.3:
                pattern = 'correlated' if corr_coef > 0 else 'anti-correlated'
            else:
                pattern = 'mixed'
        else:
            pattern = 'mixed'

        result = {
            'position_dependent': position_dependent,
            'pattern': pattern,
            'variance': float(variance),
            'coefficient_of_variation': float(cv),
            'mean_conflict': float(mean_score),
            'std_conflict': float(std_score),
            'confidence': float(confidence),
            'correlation': {
                'pearson_r': float(corr_coef) if not np.isnan(corr_coef) else None,
                'pearson_p': float(p_value) if not np.isnan(p_value) else None,
                'spearman_r': float(spearman_corr) if not np.isnan(spearman_corr) else None,
                'spearman_p': float(spearman_p) if not np.isnan(spearman_p) else None
            },
            'num_data_points': len(scores)
        }

        # Build comprehensive interpretation
        interpretations = []

        # Show all data points (or intelligently sample if too many)
        if len(scores) <= 6:
            # Show all points
            data_str = " → ".join([f"{tc}T:{s:.3f}" for tc, s in zip(token_counts, scores)])
        else:
            # Sample: first 2, middle, last 2
            indices = [0, 1, len(scores)//2, -2, -1]
            sampled = [(token_counts[i], scores[i]) for i in indices if 0 <= i < len(scores)]
            data_str = " → ".join([f"{tc}T:{s:.3f}" for tc, s in sampled[:3]]) + \
                       " ... " + " → ".join([f"{tc}T:{s:.3f}" for tc, s in sampled[3:]])

        interpretations.append(f"Conflict progression: {data_str}")

        # Add statistical interpretation
        if scores_are_constant:
            interpretations.append(f"Conflict is constant across all sequence lengths (value={mean_score:.3f})")
            interpretations.append("No position dependency - scale-invariant gradient conflict")
        elif position_dependent:
            if pattern == 'increasing':
                interpretations.append(f"Conflict increases with sequence length (r={corr_coef:.3f})")
            elif pattern == 'decreasing':
                interpretations.append(f"Conflict decreases with sequence length (r={corr_coef:.3f})")
            elif pattern in ['correlated', 'anti-correlated']:
                interpretations.append(f"Conflict {pattern} with length (r={corr_coef:.3f}, p={p_value:.3f})")
            else:
                interpretations.append(f"Variable conflict pattern (CV={cv:.2f}, mean={mean_score:.3f}±{std_score:.3f})")
        else:
            interpretations.append(f"Position-independent conflict (CV={cv:.2f}, mean={mean_score:.3f})")

        # Add confidence interpretation
        if confidence > 0.7:
            interpretations.append(f"High confidence ({confidence:.2f})")
        elif confidence > 0.3:
            interpretations.append(f"Moderate confidence ({confidence:.2f})")
        else:
            interpretations.append(f"Low confidence ({confidence:.2f})")

        result['interpretation'] = " | ".join(interpretations)

        return result

    def _compute_consensus_score(self, results: Dict[str, Any]) -> float:
        """
        Compute robust consensus conflict score from multi-scale results.

        This function computes a weighted average of conflict scores across different
        batch size and sequence length configurations. It includes validation,
        error handling, and numerical stability improvements.

        Args:
            results: Dictionary with config results containing 'weight' and 'overall_conflict'

        Returns:
            Weighted consensus score in [0, 1] range, or NaN if computation fails
        """
        valid_scores = []
        invalid_configs = []

        for config_name, config_results in results.items():
            # Skip error entries
            if isinstance(config_results, dict) and 'error' in config_results:
                logger.debug(f"Skipping {config_name} due to error: {config_results['error']}")
                continue

            # Check for required fields
            if isinstance(config_results, dict) and 'weight' in config_results and 'overall_conflict' in config_results:
                weight = config_results['weight']
                score = config_results['overall_conflict']

                # Validate inputs
                if not np.isfinite(weight) or weight <= 0:
                    invalid_configs.append(f"{config_name}: invalid weight={weight}")
                    continue

                if not np.isfinite(score):
                    invalid_configs.append(f"{config_name}: non-finite score={score}")
                    continue

                # Check score is in expected range [0, 1] with small tolerance for numerical errors
                if score < -0.01 or score > 1.01:
                    logger.warning(f"Score out of expected range in {config_name}: {score:.4f}")
                    # Clamp to valid range
                    score = np.clip(score, 0.0, 1.0)

                valid_scores.append((float(score), float(weight)))

        # Log any invalid configurations
        if invalid_configs:
            logger.debug(f"Invalid configurations: {', '.join(invalid_configs)}")

        # Check if we have any valid scores
        if not valid_scores:
            logger.warning("No valid scores for consensus computation - all configurations failed or had invalid data")
            return float('nan')

        # Compute total weight with numerical safety check
        total_weight = sum(w for _, w in valid_scores)
        if total_weight <= 1e-10:  # Numerical safety threshold
            logger.error(f"Total weight too small for safe division: {total_weight}")
            return float('nan')

        # Compute weighted average
        weighted_sum = sum(score * weight for score, weight in valid_scores)
        consensus = weighted_sum / total_weight

        # Final validation and clamping
        consensus = np.clip(consensus, 0.0, 1.0)

        # Log statistics for debugging
        if len(valid_scores) > 1:
            scores_only = [s for s, _ in valid_scores]
            weights_only = [w for _, w in valid_scores]
            normalized_weights = [w/total_weight for w in weights_only]
            logger.debug(
                f"Consensus from {len(valid_scores)}/{len(results)} configs: {consensus:.4f} "
                f"(score range: [{min(scores_only):.4f}, {max(scores_only):.4f}], "
                f"weight range: [{min(normalized_weights):.3f}, {max(normalized_weights):.3f}])"
            )
        else:
            logger.debug(f"Consensus from single valid config: {consensus:.4f}")

        return float(consensus)

    def compute_gradient_dispersion(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        checkpoint_paths: Optional[List[str]] = None,
        sample_every_n: int = 50,
        metric: str = 'gini'
    ) -> Dict[str, Any]:
        """
        Compute gradient dispersion to measure if training affects parameters evenly.

        Dispersion metrics measure how uniformly gradients are distributed across parameters.
        Low dispersion = gradients affect many parameters evenly (democratic updates)
        High dispersion = gradients concentrate on few parameters (sparse updates)

        Args:
            model: Model to analyze
            batch: Input batch for gradient computation
            checkpoint_paths: Optional list of checkpoint paths to analyze over time
            sample_every_n: If using checkpoints, sample every N checkpoints (default 50)
            metric: 'gini' (inequality), 'entropy' (information), or 'cv' (coefficient of variation)

        Returns:
            Dict with:
                - dispersion_score: Overall dispersion metric (0=uniform, 1=concentrated for Gini)
                - per_layer_dispersion: Dispersion by layer
                - top_k_concentration: Fraction of gradient norm in top k% parameters
                - temporal_trend: If checkpoints provided, dispersion over training
                - interpretation: Human-readable interpretation
        """
        results = {
            'metric': metric,
            'dispersion_score': None,
            'per_layer_dispersion': {},
            'top_k_concentration': {},
            'interpretation': ''
        }

        # Helper function to compute dispersion metrics
        def compute_dispersion_metric(grad_magnitudes, metric_type='gini'):
            """Compute dispersion metric for gradient magnitudes."""
            if len(grad_magnitudes) == 0:
                return float('nan')

            # Ensure we're working with absolute values
            grad_magnitudes = grad_magnitudes.abs()

            # Check for all zeros case
            if grad_magnitudes.sum() == 0:
                return float('nan')

            if metric_type == 'gini':
                # Gini coefficient (0=perfect equality, 1=perfect inequality)
                sorted_grads = grad_magnitudes.sort()[0]
                n = len(sorted_grads)
                # Use int64 to prevent overflow for large parameter counts
                index = torch.arange(1, n + 1, dtype=torch.int64, device=sorted_grads.device).float()

                # Add numerical stability
                grad_sum = sorted_grads.sum()
                if grad_sum < 1e-10:  # Near-zero gradients
                    return 0.0  # Perfect equality when all gradients are near-zero

                gini = ((2 * index - n - 1) * sorted_grads).sum() / (n * grad_sum)
                return gini.item()

            elif metric_type == 'entropy':
                # Shannon entropy (higher=more uniform)
                grad_sum = grad_magnitudes.sum()
                if grad_sum < 1e-10:  # Near-zero gradients
                    return 0.0

                probs = grad_magnitudes / grad_sum
                # Add small epsilon to avoid log(0), don't filter zeros
                probs = probs + 1e-10
                entropy = -(probs * probs.log()).sum()

                # Use correct normalization: max entropy for n outcomes is log(n)
                # But only count non-negligible components
                n_effective = (grad_magnitudes > 1e-10).sum().float()
                if n_effective <= 1:
                    return 0.0  # No entropy if only one effective component

                max_entropy = n_effective.log()
                return (entropy / max_entropy).item() if max_entropy > 0 else 0.0

            elif metric_type == 'cv':
                # Coefficient of variation (std/mean)
                mean = grad_magnitudes.mean()
                std = grad_magnitudes.std()
                # Use appropriate epsilon for float32 precision
                return (std / (mean + 1e-6)).item()

            else:
                raise ValueError(f"Unknown metric: {metric_type}")

        # Single model analysis
        if checkpoint_paths is None:
            # Use training mode for accurate gradient computation
            model.train()

            # Clear any existing gradients
            model.zero_grad()

            # Compute gradients
            batch = self._to_model_device(model, batch)
            outputs = model(**batch)
            loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

            # Check for valid loss
            if torch.isnan(loss) or torch.isinf(loss):
                results['interpretation'] = 'Invalid loss (NaN or Inf) - cannot compute gradients'
                return results

            loss.backward()

            # Collect all gradient magnitudes
            all_grads = []
            layer_grads = {}

            for name, param in model.named_parameters():
                if param.grad is not None:
                    # Take absolute values consistently
                    grad_mag = param.grad.flatten().abs()
                    all_grads.append(grad_mag)

                    # Group by layer
                    layer_name = name.split('.')[0]
                    if layer_name not in layer_grads:
                        layer_grads[layer_name] = []
                    layer_grads[layer_name].append(grad_mag)

            # Compute overall dispersion
            if all_grads:
                all_grads = torch.cat(all_grads)
                results['dispersion_score'] = compute_dispersion_metric(all_grads, metric)

                # Compute top-k concentration
                sorted_grads = all_grads.abs().sort(descending=True)[0]
                total_norm = sorted_grads.sum()
                for k in [1, 5, 10, 25]:
                    top_k_idx = int(len(sorted_grads) * k / 100)
                    top_k_norm = sorted_grads[:top_k_idx].sum()
                    results['top_k_concentration'][f'top_{k}_percent'] = (top_k_norm / total_norm).item()

                # Per-layer dispersion
                for layer_name, grads in layer_grads.items():
                    layer_grad = torch.cat(grads)
                    results['per_layer_dispersion'][layer_name] = compute_dispersion_metric(layer_grad, metric)

            # Clean up gradients
            model.zero_grad()

        else:
            # Analyze dispersion across checkpoints
            sampled_checkpoints = checkpoint_paths[::sample_every_n]
            temporal_dispersion = []
            temporal_steps = []

            for i, ckpt_path in enumerate(sampled_checkpoints):
                # Load checkpoint
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)

                # Use training mode for accurate gradient computation
                model.train()

                # Clear any existing gradients
                model.zero_grad()

                # Compute gradients
                batch = self._to_model_device(model, batch)
                outputs = model(**batch)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                # Check for valid loss
                if torch.isnan(loss) or torch.isinf(loss):
                    continue  # Skip this checkpoint

                loss.backward()

                # Collect gradient magnitudes
                all_grads = []
                for param in model.parameters():
                    if param.grad is not None:
                        # Take absolute values consistently
                        all_grads.append(param.grad.flatten().abs())

                if all_grads:
                    all_grads = torch.cat(all_grads)
                    dispersion = compute_dispersion_metric(all_grads, metric)
                    temporal_dispersion.append(dispersion)
                    temporal_steps.append(i * sample_every_n)

                # Clean up
                model.zero_grad()

                # Properly clean up memory
                if 'model_state_dict' in checkpoint:
                    del checkpoint['model_state_dict']
                del checkpoint

                # Force garbage collection for large models
                import gc
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            results['temporal_trend'] = {
                'steps': temporal_steps,
                'dispersion': temporal_dispersion,
                'trend': 'increasing' if temporal_dispersion[-1] > temporal_dispersion[0] else 'decreasing'
            }
            results['dispersion_score'] = temporal_dispersion[-1] if temporal_dispersion else None

        # Generate interpretation
        if results['dispersion_score'] is not None:
            score = results['dispersion_score']
            if metric == 'gini':
                if score < 0.3:
                    interp = f"Low dispersion (Gini={score:.3f}): Gradients affect parameters evenly"
                elif score < 0.6:
                    interp = f"Moderate dispersion (Gini={score:.3f}): Some parameter concentration"
                else:
                    interp = f"High dispersion (Gini={score:.3f}): Gradients concentrate on few parameters"
            elif metric == 'entropy':
                if score > 0.7:
                    interp = f"High entropy ({score:.3f}): Uniform gradient distribution"
                elif score > 0.4:
                    interp = f"Moderate entropy ({score:.3f}): Semi-uniform distribution"
                else:
                    interp = f"Low entropy ({score:.3f}): Concentrated gradients"
            else:  # cv
                if score < 1.0:
                    interp = f"Low CV ({score:.3f}): Relatively uniform gradients"
                elif score < 2.0:
                    interp = f"Moderate CV ({score:.3f}): Some variation in gradients"
                else:
                    interp = f"High CV ({score:.3f}): Large gradient variations"

            # Add top-k concentration info
            if 'top_k_concentration' in results and results['top_k_concentration']:
                top_1 = results['top_k_concentration'].get('top_1_percent', 0)
                top_10 = results['top_k_concentration'].get('top_10_percent', 0)
                interp += f"\nTop 1% params: {top_1:.1%} of gradient norm"
                interp += f"\nTop 10% params: {top_10:.1%} of gradient norm"

            # Add temporal trend if available
            if 'temporal_trend' in results:
                trend = results['temporal_trend']['trend']
                interp += f"\nTrend over training: {trend} dispersion"

            results['interpretation'] = interp

        return results

    def _process_all_samples_in_chunks(self, model, batch1, batch2, chunk_size, process_fn,
                                      aggregate_method='mean', clear_memory=True):
        """
        Process all samples in chunks and aggregate results.

        Args:
            model: Model to compute gradients for
            batch1: First batch (e.g., math batch) with all samples
            batch2: Second batch (e.g., general batch) with all samples
            chunk_size: Size of chunks to process at once
            process_fn: Function to process each chunk pair, should return a scalar or dict
            aggregate_method: How to aggregate results ('mean', 'weighted_mean', 'all')
            clear_memory: Whether to clear CUDA cache between chunks

        Returns:
            Aggregated result (scalar for 'mean', dict for 'weighted_mean', list for 'all')
        """
        all_results = []
        all_weights = []
        num_samples = batch1['input_ids'].shape[0]
        total_chunks = (num_samples + chunk_size - 1) // chunk_size

        logger.info(f"Processing {num_samples} samples in {total_chunks} chunks of size {chunk_size}")

        for chunk_idx, i in enumerate(range(0, num_samples, chunk_size)):
            end_idx = min(i + chunk_size, num_samples)
            actual_chunk_size = end_idx - i

            # Extract chunks
            chunk1 = {k: v[i:end_idx] if torch.is_tensor(v) else v
                     for k, v in batch1.items()}
            chunk2 = {k: v[i:end_idx] if torch.is_tensor(v) else v
                     for k, v in batch2.items()}

            # Log progress periodically
            if chunk_idx % 10 == 0 or chunk_idx == total_chunks - 1:
                logger.debug(f"Processing chunk {chunk_idx+1}/{total_chunks} (samples {i}-{end_idx-1})")

            try:
                # Process this chunk
                result = process_fn(model, chunk1, chunk2)
                all_results.append(result)
                all_weights.append(actual_chunk_size)  # Weight by number of samples

            except torch.cuda.OutOfMemoryError as e:
                logger.warning(f"OOM on chunk {chunk_idx}, reducing chunk size and retrying")
                if clear_memory:
                    torch.cuda.empty_cache()
                    import gc
                    gc.collect()

                # Try with half the chunk size
                if actual_chunk_size > 1:
                    half_size = actual_chunk_size // 2
                    # Process first half
                    chunk1_half = {k: v[i:i+half_size] if torch.is_tensor(v) else v
                                  for k, v in batch1.items()}
                    chunk2_half = {k: v[i:i+half_size] if torch.is_tensor(v) else v
                                  for k, v in batch2.items()}
                    result1 = process_fn(model, chunk1_half, chunk2_half)

                    # Process second half
                    chunk1_half = {k: v[i+half_size:end_idx] if torch.is_tensor(v) else v
                                  for k, v in batch1.items()}
                    chunk2_half = {k: v[i+half_size:end_idx] if torch.is_tensor(v) else v
                                  for k, v in batch2.items()}
                    result2 = process_fn(model, chunk1_half, chunk2_half)

                    # Average the two halves
                    if isinstance(result1, dict):
                        result = {k: (result1[k] + result2[k]) / 2 for k in result1}
                    else:
                        result = (result1 + result2) / 2

                    all_results.append(result)
                    all_weights.append(actual_chunk_size)
                else:
                    logger.error(f"Cannot process even single sample, skipping chunk {chunk_idx}")
                    continue

            # Clear memory between chunks if requested
            if clear_memory and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Aggregate results based on method
        if not all_results:
            logger.warning("No chunks successfully processed")
            return None

        if aggregate_method == 'all':
            return all_results
        elif aggregate_method == 'mean':
            if isinstance(all_results[0], dict):
                # Average each key across all results
                aggregated = {}
                for key in all_results[0]:
                    values = [r[key] for r in all_results if key in r]
                    aggregated[key] = sum(values) / len(values) if values else 0
                return aggregated
            else:
                # Simple average for scalar results
                return sum(all_results) / len(all_results)
        elif aggregate_method == 'weighted_mean':
            total_weight = sum(all_weights)
            if isinstance(all_results[0], dict):
                # Weighted average for each key
                aggregated = {}
                for key in all_results[0]:
                    weighted_sum = sum(r[key] * w for r, w in zip(all_results, all_weights)
                                     if key in r)
                    aggregated[key] = weighted_sum / total_weight if total_weight > 0 else 0
                return aggregated
            else:
                # Weighted average for scalar results
                weighted_sum = sum(r * w for r, w in zip(all_results, all_weights))
                return weighted_sum / total_weight if total_weight > 0 else 0
        else:
            raise ValueError(f"Unknown aggregate_method: {aggregate_method}")
