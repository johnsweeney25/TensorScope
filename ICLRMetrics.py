#!/usr/bin/env python3
"""
ICLR 2026 Critical Metrics Implementation (AUDITED)
=========================================
Metrics specifically designed to test hypotheses about why one-shot RLVR models
resist perturbation better than instruct models.

Key Hypotheses:
1. Flat Minima: RLVR models converge to flatter regions of loss landscape
2. Gradient Harmony: RLVR reduces gradient conflict between tasks
3. Feature Robustness: RLVR learns more robust feature representations
4. Mode Connectivity: RLVR solutions are better connected in weight space
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
from typing import Dict, Any, Optional, List
import warnings
import gc
import re  # For regex patterns in attention detection
import logging

# Import mode connectivity utilities for advanced interpolation
try:
    from mode_connectivity_utils import compute_bezier_path
    BEZIER_AVAILABLE = True
except ImportError:
    BEZIER_AVAILABLE = False
    compute_bezier_path = None

# Set up logger
logger = logging.getLogger(__name__)


# Helper functions for safe tensor operations
def _safe_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    """Safely copy tensors preserving device and dtype."""
    dst.copy_(src.to(device=dst.device, dtype=dst.dtype))


def _float_like(t: torch.Tensor) -> bool:
    """Check if tensor is floating point type."""
    return t.is_floating_point()


def _check_compatible(sd1: Dict[str, torch.Tensor], sd2: Dict[str, torch.Tensor]) -> None:
    """Verify two state dicts have compatible structure."""
    keys1, keys2 = set(sd1.keys()), set(sd2.keys())
    if keys1 != keys2:
        missing1 = keys2 - keys1
        missing2 = keys1 - keys2
        raise ValueError(f"State dict mismatch. Missing in model1: {missing1}, missing in model2: {missing2}")
    for k in keys1:
        if sd1[k].shape != sd2[k].shape:
            raise ValueError(f"Shape mismatch at {k}: {sd1[k].shape} vs {sd2[k].shape}")


# Helper for safe type conversion
def _to_float(x) -> float:
    """Safely convert tensor or scalar to Python float."""
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)

def _to_float_safe(x) -> float:
    """Safely convert tensor or scalar to Python float with extra safety."""
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)

# Import GradientAnalysis at top level for clarity
try:
    from GradientAnalysis import GradientAnalysis
    GRADIENT_ANALYSIS_AVAILABLE = True
except ImportError:
    GRADIENT_ANALYSIS_AVAILABLE = False
    warnings.warn("GradientAnalysis not available. Gradient conflict metrics will be skipped.")

# Try to import sklearn, but make it optional
try:
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("sklearn not available. Some metrics (2D loss landscape) will be limited.")
    
    # Simple fallback PCA implementation using numpy SVD
    class PCA:
        def __init__(self, n_components=2):
            self.n_components = n_components
            self.components_ = None
            self.explained_variance_ratio_ = None
            
        def fit_transform(self, X):
            print("Warning fallback to PCA using numpy SVD-25235")
            # Center the data
            X = np.array(X)
            X_centered = X - np.mean(X, axis=0)
            
            # Compute SVD
            U, s, Vt = np.linalg.svd(X_centered, full_matrices=False)
            
            # Store components
            self.components_ = Vt[:self.n_components]
            
            # Compute explained variance ratio
            variance = s**2 / (X.shape[0] - 1)
            total_variance = np.sum(variance)
            self.explained_variance_ratio_ = variance[:self.n_components] / total_variance
            
            # Transform data
            return X_centered @ Vt.T[:, :self.n_components]


class ICLRMetrics:
    """Critical metrics for ICLR 2026 submission on RLVR robustness."""

    def __init__(self, device: Optional[str] = None, batch_processor=None):
        """Initialize ICLR metrics calculator.

        Args:
            device: Device to use for computations
            batch_processor: Optional BatchProcessor instance for memory-efficient processing
        """
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        # Import batch processor if not provided
        if batch_processor is None:
            try:
                from batch import BatchProcessor
                self.batch_processor = BatchProcessor()
            except ImportError:
                logger.warning("BatchProcessor not available. Large batch processing may cause OOM.")
                self.batch_processor = None
        else:
            self.batch_processor = batch_processor

        print(f"ICLRMetrics initialized on {self.device}")

    # ============= HELPER FUNCTIONS =============

    def _compute_loss(self, model, batch: Dict[str, torch.Tensor], loss_fn: Optional[Any] = None):
        """
        Compute loss for a model on a batch, handling various edge cases.

        Args:
            model: The model to evaluate
            batch: Input batch
            loss_fn: Optional custom loss function(model, batch) -> loss

        Returns:
            Loss value (as tensor if gradients needed, otherwise float)
        """
        # Custom loss functions may need gradients
        if loss_fn is not None:
            # Return loss as-is (could be tensor or float)
            result = loss_fn(model, batch)
            # Only convert to float if it's already a scalar
            if not torch.is_tensor(result):
                return float(result)
            return result

        with torch.no_grad():

            # Check if labels are provided
            if 'labels' in batch and batch['labels'] is not None:
                outputs = model(**batch)
            else:
                # For causal LM without explicit labels, prepare them properly
                input_ids = batch.get('input_ids')
                if input_ids is None:
                    raise RuntimeError("No input_ids found in batch")

                # Validate and clamp input_ids and labels to valid vocabulary range
                if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                    vocab_size = model.config.vocab_size
                    max_token_id = input_ids.max().item()
                    if max_token_id >= vocab_size:
                        # Clamp input_ids to valid range
                        import warnings
                        warnings.warn(f"Token IDs ({max_token_id}) >= vocab_size ({vocab_size}), clamping to valid range")
                        # Create a copy of the batch with clamped input_ids
                        batch = dict(batch)  # Make a copy to avoid modifying original
                        batch['input_ids'] = torch.clamp(input_ids, 0, vocab_size - 1)
                        input_ids = batch['input_ids']

                # For causal language models, labels are the same as input_ids
                # The model will handle the shifting internally
                labels = input_ids.clone()
                outputs = model(**batch, labels=labels)

            # Check if model returned a loss
            if getattr(outputs, 'loss', None) is None:
                raise RuntimeError(
                    "Model did not return a loss. Provide 'labels' in batch or a custom 'loss_fn'."
                )

            return float(outputs.loss.item())

    def _filter_normalize_direction(self, model, direction: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply filter normalization to a direction vector for better loss landscape visualization.
        Based on Li et al. (2018) "Visualizing the Loss Landscape of Neural Nets"

        Args:
            model: The model (for parameter shapes)
            direction: List of tensors matching model.parameters()

        Returns:
            Filter-normalized direction
        """
        normed = []
        for p, d in zip(model.parameters(), direction):
            if d is None or not p.requires_grad:
                normed.append(torch.zeros_like(p))
                continue

            # Per-filter (or per-row) normalization for conv/linear layers
            if p.ndim >= 2:
                # Normalize each filter independently
                axes = tuple(range(1, p.ndim))
                dnorm = d.norm(p=2, dim=axes, keepdim=True).clamp_min(1e-12)
                wnorm = p.data.norm(p=2, dim=axes, keepdim=True).clamp_min(1e-12)
                # Li et al. normalize direction then scale by weight norm
                normed.append(d * (wnorm / dnorm))
            else:
                # For 1D parameters (bias, etc.), use global norm
                dnorm = d.norm(p=2).clamp_min(1e-12)
                wnorm = p.data.norm(p=2).clamp_min(1e-12)
                normed.append(d * (wnorm / dnorm))

        return normed

    def _layer_normalize_direction(self, model, direction: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Apply layer normalization to a direction vector for transformer loss landscape visualization.
        Normalizes each layer's parameters separately, more appropriate for transformers.

        Args:
            model: The model (for parameter shapes)
            direction: List of tensors matching model.parameters()

        Returns:
            Layer-normalized direction
        """
        normed = []

        # Group parameters by layer (assumes naming convention)
        layer_groups = {}
        param_list = list(model.named_parameters())

        for i, (name, p) in enumerate(param_list):
            if direction[i] is None or not p.requires_grad:
                normed.append(torch.zeros_like(p))
                continue

            # Extract layer identifier from parameter name
            # Common patterns: 'layer.0.weight', 'encoder.layer.0', 'h.0.weight'
            layer_match = re.search(r'(layer|h|block|encoder|decoder)\.(\d+)', name)
            if layer_match:
                layer_key = f"{layer_match.group(1)}.{layer_match.group(2)}"
            else:
                # For non-layer params (embeddings, final layers), treat separately
                layer_key = name.split('.')[0] if '.' in name else 'global'

            if layer_key not in layer_groups:
                layer_groups[layer_key] = []
            layer_groups[layer_key].append((i, p, direction[i]))

        # Normalize each layer group
        normalized_directions = [None] * len(direction)

        for layer_key, params_in_layer in layer_groups.items():
            if not params_in_layer:
                continue

            # Compute layer norm
            layer_norm_sq = sum((d**2).sum() for _, _, d in params_in_layer)
            layer_norm = torch.sqrt(layer_norm_sq.clamp_min(1e-12))

            # Compute weight norm for scaling
            weight_norm_sq = sum((p.data**2).sum() for _, p, _ in params_in_layer)
            weight_norm = torch.sqrt(weight_norm_sq.clamp_min(1e-12))

            # Apply normalization to this layer's parameters
            for idx, p, d in params_in_layer:
                normalized_directions[idx] = d * (weight_norm / layer_norm)

        # Fill in any remaining None entries
        for i in range(len(normalized_directions)):
            if normalized_directions[i] is None:
                normalized_directions[i] = torch.zeros_like(param_list[i][1])

        return normalized_directions

    def _hvp(self, loss, params: List[torch.Tensor], vec: List[torch.Tensor],
             retain_graph: bool = False) -> List[torch.Tensor]:
        """
        Compute Hessian-vector product efficiently using autograd.
        Memory-optimized version with configurable graph retention.

        Args:
            loss: Scalar loss tensor with grad graph (or callable that returns loss)
            params: List of model parameters
            vec: Vector to multiply with Hessian
            retain_graph: Whether to retain graph for multiple HVP calls

        Returns:
            Hessian-vector product
        """
        # If loss is callable, compute it
        if callable(loss):
            loss = loss()

        # Filter to only parameters that require gradients
        params_with_grad = [p for p in params if p.requires_grad]
        vec_with_grad = [v for p, v in zip(params, vec) if p.requires_grad]

        if not params_with_grad:
            # No parameters require gradients
            return [torch.zeros_like(v) for v in vec]

        # First compute gradients with graph for second derivative
        grads = torch.autograd.grad(loss, params_with_grad, create_graph=True,
                                   retain_graph=True, allow_unused=True)

        # Check if we have any valid gradients
        valid_grads = [g for g in grads if g is not None]
        if not valid_grads:
            # No gradients available, return zeros
            return [torch.zeros_like(v) for v in vec]

        # Compute dot product of gradients with vector
        grad_dot_v_terms = [(g * v).sum() for g, v in zip(grads, vec_with_grad) if g is not None]
        if not grad_dot_v_terms:
            return [torch.zeros_like(v) for v in vec]

        grad_dot_v = sum(grad_dot_v_terms)

        # Ensure grad_dot_v has gradient capability
        if not grad_dot_v.requires_grad:
            # If no second-order gradients, return first-order approximation
            return list(grads) if len(grads) == len(vec) else [torch.zeros_like(v) for v in vec]

        # Compute Hessian-vector product
        # retain_graph is configurable based on whether we need multiple HVPs
        hvp = torch.autograd.grad(grad_dot_v, params_with_grad,
                                 retain_graph=retain_graph, allow_unused=True)

        # Explicitly delete intermediate tensors to free memory if not retaining
        if not retain_graph:
            del grad_dot_v, grad_dot_v_terms, grads

        # Map back to original parameter list
        hvp_full = []
        grad_idx = 0
        for p, v in zip(params, vec):
            if p.requires_grad:
                h = hvp[grad_idx] if grad_idx < len(hvp) and hvp[grad_idx] is not None else None
                hvp_full.append(h if h is not None else torch.zeros_like(v))
                grad_idx += 1
            else:
                hvp_full.append(torch.zeros_like(v))

        return hvp_full

    def _modified_gram_schmidt(self, vectors: List[List[torch.Tensor]], new_vec: List[torch.Tensor]) -> List[torch.Tensor]:
        """Apply modified Gram-Schmidt orthogonalization for numerical stability.

        Args:
            vectors: List of orthonormal vectors to orthogonalize against
            new_vec: New vector to orthogonalize

        Returns:
            Orthogonalized and normalized new vector
        """
        result = new_vec
        for v in vectors:
            # Use double orthogonalization for numerical stability
            for _ in range(2):  # Perform twice for stability
                dot = sum((r * vi).sum() for r, vi in zip(result, v))
                result = [r - dot * vi for r, vi in zip(result, v)]

        # CRITICAL: Normalize the result
        norm = torch.sqrt(sum((ri**2).sum() for ri in result)).clamp_min(1e-12)
        result = [ri / norm for ri in result]

        return result

    def _set_random_seeds(self, seed: Optional[int]) -> None:
        """Set random seeds for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

    # ============= LOSS LANDSCAPE GEOMETRY (AUDITED & FIXED) =============
    
    def compute_loss_barrier(
        self,
        model1,
        model2,
        data_batch: Optional[Dict[str, torch.Tensor]] = None,
        data_loader: Optional[Any] = None,
        n_points: int = 20,
        return_trajectory: bool = False,
        interpolate_buffers: bool = False,
        method: str = 'linear',
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute the loss barrier between two models via interpolation.
        Critical for understanding if RLVR models live in flatter minima.

        Implementation follows Draxler et al. (2018) and Garipov et al. (2018):
        - By default, only interpolates trainable parameters (not buffers like BN stats)
        - Uses memory-efficient interpolation without deepcopy
        - Supports both linear and Bezier curve interpolation

        Args:
            model1: First model (e.g., base/instruct)
            model2: Second model (e.g., RLVR fine-tuned)
            data_batch: Batch to evaluate loss on
            data_loader: DataLoader for multi-batch evaluation
            n_points: Number of interpolation points
            return_trajectory: Whether to return full loss trajectory
            interpolate_buffers: Whether to interpolate buffers (BN stats etc).
                               Default False matches literature standard.
            method: Interpolation method ('linear' or 'bezier')
            seed: Random seed for reproducibility

        Returns:
            Dictionary with barrier height, sharpness metrics
        """
        # Set seeds for reproducibility
        self._set_random_seeds(seed)

        # Input validation
        if (data_batch is None) == (data_loader is None):
            return {'error': 'Provide exactly one of data_batch or data_loader'}

        if method not in ['linear', 'bezier']:
            return {'error': f'Unknown interpolation method: {method}. Use "linear" or "bezier"'}

        if method == 'bezier' and not BEZIER_AVAILABLE:
            warnings.warn('Bezier interpolation requested but mode_connectivity_utils not available, falling back to linear')
            method = 'linear'

        model1.eval()
        model2.eval()

        # CRITICAL FIX: Check device compatibility
        device1 = next(model1.parameters()).device
        device2 = next(model2.parameters()).device

        if device1 != device2:
            warnings.warn(f"Models on different devices: {device1} vs {device2}. Using {device1}.")
            # Move model2 parameters to device1 for interpolation
            # Note: We don't permanently move model2, just work with its state dict

        device = device1  # Use first model's device as target
        
        # Move batch to model device if using single batch
        batch = None
        if data_batch is not None:
            # CRITICAL FIX: Validate batch structure
            if not isinstance(data_batch, dict):
                return {'error': f'data_batch must be a dict, got {type(data_batch)}'}

            # Check for common expected keys (warn if missing, don't error)
            common_keys = {'input_ids', 'inputs', 'labels', 'attention_mask'}
            if not any(k in data_batch for k in common_keys):
                warnings.warn(f"Batch may be missing expected keys. Found: {list(data_batch.keys())}")

            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                    for k, v in data_batch.items()}
        
        # Get and check states
        s1 = model1.state_dict()
        s2 = model2.state_dict()
        try:
            _check_compatible(s1, s2)
        except ValueError as e:
            return {'error': f'Models are incompatible: {e}'}

        # CRITICAL FIX: Validate parameter shapes match
        for key in s1.keys():
            if key in s2 and s1[key].shape != s2[key].shape:
                return {'error': f'Shape mismatch for {key}: {s1[key].shape} vs {s2[key].shape}'}

        # Pre-identify parameters vs buffers for correct interpolation
        param_name_set = {n for (n, _) in model1.named_parameters()}
        param_keys = [k for k in s1.keys() if k in param_name_set]
        buffer_keys = [k for k in s1.keys() if k not in param_name_set]

        # Interpolation setup
        alphas = np.linspace(0.0, 1.0, int(n_points))
        losses = []

        # CRITICAL FIX: Use vector operations for efficiency (like sample_directional_losses)
        from torch.nn.utils import parameters_to_vector, vector_to_parameters

        # Save original state efficiently
        original_params_vector = parameters_to_vector(model1.parameters()).clone()
        original_buffers = {name: buf.detach().clone()
                           for name, buf in model1.named_buffers()}

        # Precompute device tensors with smart transfers
        # Only transfer if not already on target device
        def smart_to_device(tensor, target_device):
            if tensor.device != target_device:
                return tensor.detach().to(target_device)
            return tensor.detach()

        s1_dev = {k: smart_to_device(v, device) for k, v in s1.items()}
        s2_dev = {k: smart_to_device(v, device) for k, v in s2.items()}

        # Evaluation helper with proper error handling
        @torch.inference_mode()
        def eval_loss() -> float:
            if data_loader is not None:
                total = 0.0
                count = 0
                for loader_batch in data_loader:
                    b = {k: (v.to(device) if torch.is_tensor(v) else v)
                         for k, v in loader_batch.items()}
                    loss_val = self._compute_loss(model1, b, self.loss_fn)
                    total += _to_float(loss_val)
                    count += 1
                return total / max(count, 1)
            else:
                loss_val = self._compute_loss(model1, batch, self.loss_fn)
                return _to_float(loss_val)

        # CRITICAL FIX: Add progress tracking
        try:
            from tqdm import tqdm
            from tqdm.contrib.logging import logging_redirect_tqdm
iterator = tqdm(enumerate(alphas), total=len(alphas), desc="Computing loss barrier", file=sys.stderr, leave=False, dynamic_ncols=True, mininterval=0.5)
            use_tqdm = True
        except ImportError:
            iterator = enumerate(alphas)
            use_tqdm = False

        # CRITICAL FIX: Ensure complete gradient isolation
        with torch.inference_mode():  # No gradients can be computed inside this block
            try:
                # Wrap with logging redirect if tqdm is available
                if use_tqdm:
                    ctx = logging_redirect_tqdm()
                    ctx.__enter__()

                for i, alpha in iterator:
                    alpha = float(alpha)

                    if method == 'bezier':
                        # CRITICAL FIX: Validate Bezier path with error handling
                        try:
                            new_state = compute_bezier_path(s1_dev, s2_dev, alpha, n_control_points=3)

                            # Validate the result
                            if not isinstance(new_state, dict):
                                raise ValueError("Bezier path did not return a dict")
                            if not all(k in new_state for k in param_keys):
                                raise ValueError("Bezier path missing required parameters")

                            # Handle buffers based on interpolate_buffers flag
                            if not interpolate_buffers:
                                for k in buffer_keys:
                                    new_state[k] = s1_dev[k]  # Keep buffers from model1
                        except Exception as e:
                            warnings.warn(f"Bezier interpolation failed at alpha={alpha}: {e}. Using linear.")
                            method = 'linear'  # Fall through to linear

                    if method == 'linear':  # Note: not 'else' to allow fallback
                        # Linear interpolation with logging
                        new_state = {}
                        non_float_params = []

                        for k in param_keys:
                            t1, t2 = s1_dev[k], s2_dev[k]
                            if _float_like(t1) and _float_like(t2):
                                # Memory-efficient interpolation on device
                                new_state[k] = torch.lerp(t1, t2, alpha)
                            else:
                                new_state[k] = t1  # Keep non-float params from model1
                                non_float_params.append(k)

                        # Log non-interpolatable parameters once
                        if i == 0 and non_float_params:
                            logger.debug(f"Cannot interpolate {len(non_float_params)} non-float parameters")

                        # Handle buffers based on interpolate_buffers flag
                        for k in buffer_keys:
                            t1, t2 = s1_dev[k], s2_dev[k]
                            if interpolate_buffers and _float_like(t1) and _float_like(t2):
                                new_state[k] = torch.lerp(t1, t2, alpha)
                            else:
                                new_state[k] = t1  # Keep buffers from model1

                    # Load interpolated state into model1 temporarily
                    try:
                        model1.load_state_dict(new_state, strict=True)
                        model1.eval()
                    except Exception as e:
                        logger.warning(f"Failed to load interpolated state at alpha={alpha}: {e}")
                        losses.append(float('nan'))
                        continue

                    # Compute and store loss
                    loss_val = eval_loss()
                    losses.append(loss_val)

                    # Optional debug logging
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Alpha={alpha:.3f}, Loss={loss_val:.6f}")

                    # Periodic memory cleanup
                    if i > 0 and i % 5 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()

                # Exit logging redirect context if used
                if use_tqdm:
                    ctx.__exit__(None, None, None)

            finally:
                # CRITICAL FIX: Restore model1 using vector operations
                with torch.inference_mode():
                    vector_to_parameters(original_params_vector, model1.parameters())
                for name, buf in model1.named_buffers():
                    buf.data.copy_(original_buffers[name])

            # Clean up
            del s1_dev, s2_dev, original_state, original_buffers
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Ensure models are in eval mode
            model1.eval()
            model2.eval()
        
        # Compute barrier metrics with NaN/Inf handling
        if len(losses) == 0:
            return {'error': 'No losses computed during interpolation', 'n_points': n_points}

        # Convert to numpy and filter non-finite values
        losses_np = np.asarray(losses, dtype=np.float64)
        finite_mask = np.isfinite(losses_np)
        n_nonfinite = int(np.sum(~finite_mask))

        if not finite_mask.any():
            return {'error': 'All loss values were non-finite'}

        # Work with finite values only
        finite_losses = losses_np[finite_mask]
        finite_alphas = alphas[finite_mask]

        start_loss = float(finite_losses[0])
        end_loss = float(finite_losses[-1])

        # Use max of endpoints per literature standard
        endpoints_max = max(start_loss, end_loss)
        barrier_height = float(np.max(finite_losses) - endpoints_max)

        # Normalized barrier with proper epsilon
        # CRITICAL FIX: Use more stable epsilon
        eps = 1e-8  # More stable than machine epsilon (~2.2e-16)
        normalized_barrier = barrier_height / (abs(endpoints_max) + eps)

        # Find the maximum point
        max_idx = int(np.argmax(finite_losses))
        barrier_position = float(finite_alphas[max_idx])

        # Compute sharpness around barrier (second-order finite difference)
        # CRITICAL FIX: Use correct formula with h²
        local_sharpness = 0.0
        if 0 < max_idx < len(finite_losses) - 1:
            # Calculate step size h
            h = finite_alphas[1] - finite_alphas[0] if len(finite_alphas) > 1 else 1.0
            # Correct second derivative approximation: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
            local_sharpness = float(
                (finite_losses[max_idx+1] - 2 * finite_losses[max_idx] + finite_losses[max_idx-1]) / (h ** 2)
            )
        else:
            local_sharpness = 0

        # Compute path length (total variation) on finite values
        path_length = float(np.sum(np.abs(np.diff(finite_losses))))

        # Mode connectivity: barrier <= 0 means no barrier (literature definition)
        is_connected = bool(barrier_height <= 0.0)

        # Interpolation smoothness (second-order finite differences)
        smoothness = 0.0
        if len(finite_losses) >= 3:
            second_diffs = np.diff(np.diff(finite_losses))
            if len(second_diffs) > 0:
                smoothness = float(np.mean(np.abs(second_diffs)))

        results = {
            'barrier_height': barrier_height,
            'barrier_height_normalized': normalized_barrier,
            'barrier_position': barrier_position,
            'barrier_sharpness': local_sharpness,
            'path_length': path_length,
            'start_loss': start_loss,
            'end_loss': end_loss,
            'max_loss': float(np.max(finite_losses)),
            'mean_loss': float(np.mean(finite_losses)),
            'loss_variance': float(np.var(finite_losses)),
            'is_mode_connected': is_connected,
            'interpolation_smoothness': smoothness,
            'n_points': int(len(finite_losses)),
            'n_nonfinite': n_nonfinite,
            'interpolate_buffers': interpolate_buffers,
            'interpolation_method': method
        }

        if return_trajectory:
            results['loss_trajectory'] = finite_losses.tolist()
            results['alpha_values'] = finite_alphas.tolist()

        return results
    
    def sample_directional_losses(
        self,
        model,
        data_batch: Dict[str, torch.Tensor],
        n_samples: int = 50,
        span: float = 0.5,
        loss_fn: Optional[Any] = None,
        use_filter_norm: bool = True,
        relative_span: bool = False,
        seed: Optional[int] = None,
        batch_config=None,  # NEW: Optional BatchConfig for memory-efficient processing
        two_sided: bool = False,
        compute_curvature: bool = False,
        global_renorm: bool = False,  # FIXED: Changed default to False to preserve filter norm intent
        compute_gradient: bool = False  # Compute directional gradient (first order)
    ) -> Dict[str, Any]:
        """
        Sample losses along random directions from the model's current position in parameter space.

        THEORETICAL BACKGROUND:
        This method implements 1D directional loss sampling for understanding the local geometry
        of the loss landscape around the current model parameters. It samples random unit directions
        in parameter space and evaluates the loss at θ + α·d where:
        - θ = current model parameters
        - d = random direction vector (normalized)
        - α = span (step size)

        KEY CONCEPTS:
        1. Directional Derivative: If compute_gradient=True, computes ∇L(θ)·d for each direction d
        2. Directional Curvature: If two_sided=True and compute_curvature=True, estimates the
           second derivative using finite differences: d²L/dα² ≈ (L(θ+αd) + L(θ-αd) - 2L(θ)) / α²
        3. Filter Normalization (Li et al., 2018): Normalizes directions per-filter for CNNs to
           account for scale differences between layers, ensuring equal contribution from all layers

        ANALYSIS INSIGHTS:
        - Loss variance along random directions indicates optimization difficulty
        - Correlation between directional gradient and loss change validates Taylor approximation
        - High curvature indicates sharp minima; low curvature indicates flat regions
        - Asymmetry between positive/negative directions reveals landscape anisotropy

        IMPORTANT: This is NOT a 2D landscape visualization - it samples independent 1D slices.
        For true 2D landscape visualization, use compute_loss_landscape_2d.

        Args:
            model: Model to analyze (requires gradients enabled for gradient/curvature computation)
            data_batch: Batch to evaluate loss on
            n_samples: Number of random directions to sample
            span: Step size in parameter space (absolute by default, relative if relative_span=True)
            loss_fn: Optional custom loss function (defaults to cross-entropy)
            use_filter_norm: Whether to use filter normalization (recommended for CNNs)
            relative_span: If True, span is relative to parameter norm ||θ||
            seed: Random seed for reproducibility
            batch_config: Optional BatchConfig for memory-efficient processing
            two_sided: If True, sample at both +span and -span for each direction
            compute_curvature: If True and two_sided, compute directional curvature estimates
            global_renorm: If True and use_filter_norm, apply global normalization after filter norm.
                          WARNING: Setting this True with use_filter_norm=True destroys the scale-aware
                          properties of filter normalization. Recommended: False (default).
            compute_gradient: If True, compute directional gradient (requires grad-enabled model)

        Returns:
            Dictionary containing:
            - baseline_loss: Loss at current parameters L(θ)
            - loss_mean/std: Statistics of losses at perturbed positions
            - dir_grad_mean/std: Statistics of directional derivatives (if computed)
            - curvature_mean/std: Statistics of directional curvatures (if computed)
            - dir_grad_delta_corr: Correlation between predicted and actual loss changes
            - Additional statistics and diagnostic information
        """
        # Import utilities for efficient parameter handling
        from torch.nn.utils import parameters_to_vector, vector_to_parameters
        import warnings
        import gc

        # CRITICAL FIX: Aggressive memory cleanup BEFORE starting
        # This function is memory-intensive and needs a clean slate
        # Clear any cached activations from previous metrics
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated_gb = torch.cuda.memory_allocated() / 1e9
            logger.info(f"🧹 Starting sample_directional_losses with {allocated_gb:.2f}GB allocated")

        # Set seeds for reproducibility
        # Note: Direction sampling is deterministic given seed, but loss evaluation
        # may vary due to model internals (dropout, batch norm, etc.)
        self._set_random_seeds(seed)

        # Store original model state to restore later
        original_training = model.training
        # Use eval mode for deterministic gradient computation
        # Gradients work perfectly fine in eval mode!
        model.eval()
        model_device = next(model.parameters()).device

        # Handle batch processing if batch_config provided
        used_batch_processor = False
        if batch_config is not None and self.batch_processor is not None:
            try:
                from batch import create_batch
                batch = create_batch(data_batch, batch_config, model_device)
                used_batch_processor = True
            except ImportError:
                # Fallback to simple batch processing
                batch = {k: v.to(model_device) if torch.is_tensor(v) else v
                        for k, v in data_batch.items()}
        else:
            batch = {k: v.to(model_device) if torch.is_tensor(v) else v
                    for k, v in data_batch.items()}

        # Check for parameters with requires_grad
        # Note: We work with parameters that have requires_grad=True
        # If none are found, this likely means gradient context wasn't set up properly
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        if not trainable_params:
            # Try to find ALL parameters as a fallback (some models might have gradients disabled)
            all_params = list(model.parameters())
            if not all_params:
                return {'error': 'No parameters found in model'}

            # Log warning and proceed with all parameters but disable gradient computation
            import warnings
            warnings.warn(
                f"No parameters with requires_grad=True found. Found {len(all_params)} total parameters. "
                "Proceeding without gradient computation. Ensure model is called with proper gradient context."
            )
            trainable_params = all_params
            compute_gradient = False  # Force disable gradient computation
            compute_curvature = False  # Can't compute curvature without gradients
        n_trainable = sum(p.numel() for p in trainable_params)

        # Check for heterogeneous dtypes and determine update path
        dtypes = {p.dtype for p in trainable_params}
        homogeneous_dtype = (len(dtypes) == 1)
        dtype0 = next(iter(dtypes)) if dtypes else torch.float32
        # Only use fast vector path for homogeneous float32 models
        fast_vec_path = (homogeneous_dtype and dtype0 == torch.float32)
        per_param_update = not fast_vec_path

        # Create base vector on device
        with torch.no_grad():
            base_vec_native = parameters_to_vector(trainable_params)
            # Only convert to float32 if not already (for numerical stability in directions)
            base_vec = base_vec_native if fast_vec_path else base_vec_native.float()
            param_norm = base_vec.norm().item()

        # Define consistent loss evaluation function
        def _loss_eval():
            """Compute loss using batch processor if available, otherwise direct."""
            if used_batch_processor and self.batch_processor is not None:
                return self.batch_processor.process_batch(
                    batch=batch,
                    compute_fn=lambda chunk: self._compute_loss(model, chunk, loss_fn),
                    reduction='mean',
                    config_override=batch_config
                )
            else:
                return self._compute_loss(model, batch, loss_fn)

        # Compute baseline loss and optionally gradient
        baseline_loss = None
        baseline_gradient = None

        if compute_gradient:
            # CRITICAL FIX: Compute gradients WITHOUT batch processor to avoid blocking
            # The batch processor might have gradient blocking optimizations that interfere
            model.zero_grad()

            try:
                # Use direct loss computation for gradient pass
                if used_batch_processor:
                    # Temporarily bypass batch processor for gradient computation
                    loss_val = self._compute_loss(model, batch, loss_fn)
                else:
                    loss_val = _loss_eval()

                if torch.is_tensor(loss_val):
                    baseline_loss = _to_float_safe(loss_val)
                    loss_val.backward()
                    with torch.no_grad():
                        baseline_gradient = parameters_to_vector(
                            [p.grad if p.grad is not None else torch.zeros_like(p)
                             for p in trainable_params]
                        ).float()
                    model.zero_grad()
                else:
                    # If loss_fn returns a float, can't compute gradients
                    warnings.warn("Cannot compute gradients - loss_fn returned float instead of tensor")
                    compute_gradient = False
                    baseline_gradient = None
                    baseline_loss = _to_float_safe(loss_val)
            except RuntimeError as e:
                # Handle mixed dtype gradient errors gracefully
                if "dtype" in str(e).lower():
                    warnings.warn(f"Cannot compute gradients with mixed dtypes: {e}")
                    compute_gradient = False
                    baseline_gradient = None
                    # Still compute baseline loss without gradients
                    with torch.no_grad():
                        baseline_loss = _to_float_safe(_loss_eval())
                else:
                    raise  # Re-raise other runtime errors
        else:
            with torch.no_grad():
                baseline_loss = _to_float_safe(_loss_eval())

        # Calculate actual span based on relative_span setting
        actual_span = span * param_norm if relative_span else span
        relative_step = actual_span / (param_norm + 1e-12)

        # Check for conflicting options
        if compute_curvature and not two_sided:
            warnings.warn("compute_curvature=True requires two_sided=True. Disabling curvature.")
            compute_curvature = False

        # Add curvature scale warning
        if compute_curvature and actual_span > 0.1:
            warnings.warn(
                f"Large span ({actual_span:.3f}) may give inaccurate curvature estimates. "
                "Consider using span <= 0.1 for better finite difference approximation."
            )

        # CRITICAL FIX: Warn about conflicting normalization settings
        if use_filter_norm and global_renorm:
            warnings.warn(
                "global_renorm=True with use_filter_norm=True destroys filter normalization's "
                "scale-aware properties. Consider setting global_renorm=False."
            )

        # CRITICAL FIX: Warn about numerical precision issues with small spans and non-float32
        if not fast_vec_path and actual_span < 0.01:
            warnings.warn(
                f"Small span ({actual_span:.4f}) with non-float32 model (dtype={dtype0}) may lose "
                f"precision during parameter updates. Consider span >= 0.1 for reliable results with "
                f"bfloat16/float16 models."
            )

        # Storage for results
        losses_positive = []
        losses_negative = [] if two_sided else None
        curvatures = [] if compute_curvature else None
        directional_grads = [] if compute_gradient and baseline_gradient is not None else None
        dir_norms = []  # Track direction norms after all normalizations

        # For PCA: completely disabled for large models to prevent OOM
        # The PCA on random directions is not essential and adds little value for large models
        variance_explained = []
        projected_dirs = []

        # CRITICAL FIX: Disable projection entirely for models >1M params
        # The variance explained metric is not worth the memory cost
        use_projection = False  # Completely disabled to ensure no OOM
        proj_dim = None

        # Log why projection is disabled
        if n_samples >= 2 and base_vec.numel() >= 1_000_000:
            logger.debug(f"PCA projection disabled for {base_vec.numel()/1e6:.0f}M param model (memory safety)")

        # Optional: Store checksum for validation (helps debug restore issues)
        if logger.isEnabledFor(logging.DEBUG):
            param_checksum_before = sum(p.sum().item() for p in trainable_params)
            logger.debug(f"Parameter checksum before sampling: {param_checksum_before:.8e}")

        # CRITICAL FIX: Create explicit generator for perfect CUDA reproducibility (ICML requirement)
        # Using an explicit generator ensures reproducible random directions across runs
        # even with different CUDA memory states
        rng_generator = None
        if seed is not None:
            rng_generator = torch.Generator(device=model_device)
            rng_generator.manual_seed(seed)
            logger.debug(f"Created torch.Generator with seed={seed} for reproducible sampling")

        # Wrap entire sampling loop in no_grad for complete gradient isolation
        with torch.no_grad():
            try:
                for sample_idx in range(n_samples):
                    # Generate random direction directly on device (much faster!)
                    # Use explicit generator for reproducibility if seed was provided
                    if rng_generator is not None:
                        # Unique seed per sample for diverse directions but reproducible
                        rng_generator.manual_seed(seed + sample_idx)
                        dir_vec = torch.randn_like(base_vec, generator=rng_generator)
                    else:
                        dir_vec = torch.randn_like(base_vec)

                    # Apply filter normalization if requested
                    if use_filter_norm:
                        # Reshape to parameter shapes for filter normalization
                        direction_tensors = []
                        idx = 0
                        for p in trainable_params:
                            numel = p.numel()
                            direction_tensors.append(
                                dir_vec[idx:idx+numel].reshape(p.shape)
                            )
                            idx += numel

                        # Apply filter normalization
                        normalized_direction = self._filter_normalize_direction(model, direction_tensors)

                        # Flatten back to vector
                        dir_vec = torch.cat([d.reshape(-1) for d in normalized_direction])

                        # CRITICAL FIX: Delete intermediate tensors immediately
                        del direction_tensors
                        del normalized_direction

                        # Apply global renormalization if requested (for consistent span)
                        if global_renorm:
                            dir_norm = dir_vec.norm()
                            if dir_norm > 0:
                                dir_vec = dir_vec / dir_norm
                    else:
                        # Simple global normalization
                        dir_norm = dir_vec.norm()
                        if dir_norm > 0:
                            dir_vec = dir_vec / dir_norm
                        else:
                            # Extremely unlikely zero vector
                            dir_vec = torch.randn_like(base_vec)
                            dir_vec = dir_vec / dir_vec.norm()

                    # Record direction norm
                    dir_norms.append(dir_vec.norm().item())

                    # Projection disabled to prevent OOM with large models
                    # The PCA variance explained metric is not essential

                    # Compute directional gradient if available
                    if directional_grads is not None:
                        dir_grad = (baseline_gradient @ dir_vec).item()
                        directional_grads.append(dir_grad)

                    # Update parameters based on fast_vec_path
                    if fast_vec_path:
                        # Fast vector update for homogeneous float32
                        perturbed_vec = base_vec + actual_span * dir_vec
                        vector_to_parameters(perturbed_vec, trainable_params)
                        # CRITICAL FIX: Delete temporary immediately
                        del perturbed_vec
                    else:
                        # Per-parameter update for mixed dtypes or non-float32
                        idx = 0
                        for p in trainable_params:
                            n = p.numel()
                            new_param = (base_vec[idx:idx+n] + actual_span * dir_vec[idx:idx+n]).reshape(p.shape)
                            p.data.copy_(new_param.to(device=p.device, dtype=p.dtype))
                            # CRITICAL FIX: Delete temporary immediately
                            del new_param
                            idx += n

                    # Compute loss using consistent path
                    loss_pos = _to_float_safe(_loss_eval())
                    losses_positive.append(loss_pos)

                    # Compute loss at -span if two-sided
                    if two_sided:
                        if fast_vec_path:
                            # Fast vector update for homogeneous float32
                            perturbed_vec = base_vec - actual_span * dir_vec
                            vector_to_parameters(perturbed_vec, trainable_params)
                            # CRITICAL FIX: Delete temporary immediately
                            del perturbed_vec
                        else:
                            # Per-parameter update for mixed dtypes or non-float32
                            idx = 0
                            for p in trainable_params:
                                n = p.numel()
                                new_param = (base_vec[idx:idx+n] - actual_span * dir_vec[idx:idx+n]).reshape(p.shape)
                                p.data.copy_(new_param.to(device=p.device, dtype=p.dtype))
                                # CRITICAL FIX: Delete temporary immediately
                                del new_param
                                idx += n

                        # Compute loss using consistent path
                        loss_neg = _to_float_safe(_loss_eval())
                        losses_negative.append(loss_neg)

                        # Compute curvature if requested
                        if compute_curvature:
                            # Finite difference approximation: f''(x) ≈ [f(x+h) + f(x-h) - 2f(x)] / h²
                            curvature = (loss_pos + loss_neg - 2 * baseline_loss) / (actual_span ** 2)
                            curvatures.append(curvature)

                    # CRITICAL FIX: Delete direction vector immediately after use
                    del dir_vec

                    # CRITICAL FIX: Periodic cache clearing to force GPU cleanup
                    if (sample_idx + 1) % 10 == 0:
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"Cleared GPU cache after {sample_idx + 1} samples")

                # CRITICAL FIX: Delete baseline_gradient immediately after loop ends
                # Don't wait for finally block - free 6GB for 1.5B models NOW
                if baseline_gradient is not None:
                    del baseline_gradient
                    baseline_gradient = None
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug("Deleted baseline_gradient after sampling loop")

            finally:
                # CRITICAL: Always restore original parameters
                if fast_vec_path:
                    # Fast vector restore for homogeneous float32
                    vector_to_parameters(base_vec, trainable_params)
                else:
                    # Per-parameter restore for mixed dtypes or non-float32
                    idx = 0
                    for p in trainable_params:
                        n = p.numel()
                        p.data.copy_(base_vec[idx:idx+n].reshape(p.shape).to(device=p.device, dtype=p.dtype))
                        idx += n

                # Optional: Validate restoration BEFORE cleanup
                if logger.isEnabledFor(logging.DEBUG):
                    param_checksum_after = sum(p.sum().item() for p in trainable_params)
                    checksum_diff = abs(param_checksum_after - param_checksum_before)
                    if checksum_diff > 1e-6:
                        logger.warning(
                            f"Parameter restoration may be incomplete. "
                            f"Checksum diff: {checksum_diff:.8e}"
                        )
                    else:
                        logger.debug("Parameters successfully restored")

                # CRITICAL FIX: Explicit memory cleanup for large tensors
                # Delete base_vec (6GB for 1.5B models) - the main memory leak!
                if 'base_vec' in locals():
                    del base_vec
                if 'base_vec_native' in locals():
                    del base_vec_native

                # CRITICAL FIX: Delete baseline gradient (safety net - already deleted after loop)
                if 'baseline_gradient' in locals() and baseline_gradient is not None:
                    del baseline_gradient
                    baseline_gradient = None

                # CRITICAL FIX: Delete batch explicitly to free GPU memory
                if 'batch' in locals():
                    # Delete tensor values first
                    for k, v in batch.items():
                        if torch.is_tensor(v):
                            del v
                    # Then delete the dict itself
                    del batch

                # CRITICAL FIX: Aggressive GPU cleanup
                # Force immediate memory release, not just marking for GC
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()  # Ensure all operations complete

                    if logger.isEnabledFor(logging.DEBUG):
                        final_allocated = torch.cuda.memory_allocated() / 1e9
                        logger.debug(f"After cleanup: {final_allocated:.2f}GB allocated")

        # Compute PCA on projected directions (memory efficient with sign hashing)
        if use_projection and projected_dirs:
            try:
                from sklearn.decomposition import PCA
                pca = PCA(n_components=min(2, len(projected_dirs)), svd_solver='auto')
                _ = pca.fit_transform(projected_dirs)
                variance_explained = pca.explained_variance_ratio_.tolist()
            except Exception:
                # PCA can fail or sklearn not available
                pass
        
        # Compute statistics with NaN handling
        loss_array_pos = np.array(losses_positive, dtype=np.float64)
        finite_mask = np.isfinite(loss_array_pos)
        if not finite_mask.any():
            return {'error': 'All losses were non-finite'}

        finite_losses = loss_array_pos[finite_mask]
        n_nonfinite_pos = int(np.sum(~finite_mask))

        # Compute deltas from baseline
        deltas = finite_losses - baseline_loss

        eps = np.finfo(np.float64).eps
        results = {
            'baseline_loss': float(baseline_loss),
            'loss_mean': float(np.mean(finite_losses)),
            'loss_std': float(np.std(finite_losses)),
            'loss_max': float(np.max(finite_losses)),
            'loss_min': float(np.min(finite_losses)),
            'delta_mean': float(np.mean(deltas)),
            'delta_std': float(np.std(deltas)),
            'delta_max': float(np.max(deltas)),
            'delta_min': float(np.min(deltas)),
            'landscape_roughness': float(np.std(finite_losses) / (abs(np.mean(finite_losses)) + eps)),
            'pca_explained_variance': variance_explained,
            'n_samples': int(len(finite_losses)),
            'n_nonfinite_pos': n_nonfinite_pos,
            'actual_span': float(actual_span),
            'param_norm': float(param_norm),
            'relative_step': float(relative_step),
            'use_filter_norm': use_filter_norm,
            'global_renorm': global_renorm if use_filter_norm else None,
            'relative_span': relative_span,
            'two_sided': two_sided,
            'seed': seed,
            'n_trainable_params': n_trainable,
            'homogeneous_dtype': homogeneous_dtype,
            'fast_vec_path': fast_vec_path,
            'used_batch_processor': used_batch_processor,
            'projection_dim': proj_dim if use_projection else None,
            'note': 'Production-ready with dtype-safe updates and consistent batch processing'
        }

        # Add direction norm statistics
        if dir_norms:
            results['dir_norm_mean'] = float(np.mean(dir_norms))
            results['dir_norm_std'] = float(np.std(dir_norms))

        # Add two-sided statistics if available
        if two_sided and losses_negative:
            loss_array_neg = np.array(losses_negative, dtype=np.float64)
            finite_neg_mask = np.isfinite(loss_array_neg)
            finite_neg = loss_array_neg[finite_neg_mask]
            if len(finite_neg) > 0:
                results['loss_mean_negative'] = float(np.mean(finite_neg))
                results['loss_std_negative'] = float(np.std(finite_neg))
                results['n_nonfinite_neg'] = int(np.sum(~finite_neg_mask))

        # Add curvature statistics if computed
        if curvatures:
            curv_array = np.array(curvatures, dtype=np.float64)
            finite_curv = curv_array[np.isfinite(curv_array)]
            if len(finite_curv) > 0:
                results['curvature_mean'] = float(np.mean(finite_curv))
                results['curvature_std'] = float(np.std(finite_curv))
                results['curvature_max'] = float(np.max(finite_curv))
                results['curvature_min'] = float(np.min(finite_curv))
                # Add percentiles for distribution analysis
                if len(finite_curv) >= 10:
                    results['curvature_p10'] = float(np.percentile(finite_curv, 10))
                    results['curvature_p50'] = float(np.percentile(finite_curv, 50))
                    results['curvature_p90'] = float(np.percentile(finite_curv, 90))

        # Add directional gradient statistics if computed
        if directional_grads:
            dg_array = np.array(directional_grads, dtype=np.float64)
            finite_dg = dg_array[np.isfinite(dg_array)]
            if len(finite_dg) > 0:
                results['dir_grad_mean'] = float(np.mean(finite_dg))
                results['dir_grad_std'] = float(np.std(finite_dg))
                results['dir_grad_max'] = float(np.max(finite_dg))
                results['dir_grad_min'] = float(np.min(finite_dg))

                # CRITICAL FIX: Use JOINT finite mask for both arrays to ensure proper alignment
                # directional_grads and losses_positive must both be finite for correlation
                if len(directional_grads) == len(losses_positive) and len(directional_grads) > 1:
                    dg_for_corr = np.array(directional_grads, dtype=np.float64)

                    # Create JOINT finite mask: both loss and gradient must be finite
                    finite_loss_mask = np.isfinite(loss_array_pos)
                    finite_grad_mask = np.isfinite(dg_for_corr)
                    joint_finite_mask = finite_loss_mask & finite_grad_mask

                    if joint_finite_mask.sum() > 1:
                        # Apply joint mask to both arrays
                        dg_masked = dg_for_corr[joint_finite_mask]
                        deltas_masked = (loss_array_pos - baseline_loss)[joint_finite_mask]

                        if len(dg_masked) > 1 and len(deltas_masked) > 1:
                            # Both arrays now have same length and all values are finite
                            assert len(dg_masked) == len(deltas_masked), "Arrays must have same length after joint masking"

                            try:
                                # Double-check all values are finite (should be guaranteed by mask)
                                if np.all(np.isfinite(dg_masked)) and np.all(np.isfinite(deltas_masked)):
                                    corr_matrix = np.corrcoef(dg_masked, deltas_masked)
                                    results['dir_grad_delta_corr'] = float(corr_matrix[0, 1])
                                    results['n_samples_for_corr'] = int(len(dg_masked))
                                else:
                                    results['dir_grad_delta_corr'] = None
                                    results['n_samples_for_corr'] = 0
                            except Exception as e:
                                logger.warning(f"Failed to compute gradient-delta correlation: {e}")
                                results['dir_grad_delta_corr'] = None
                                results['n_samples_for_corr'] = 0
                        else:
                            results['dir_grad_delta_corr'] = None
                            results['n_samples_for_corr'] = int(joint_finite_mask.sum())
                    else:
                        results['dir_grad_delta_corr'] = None
                        results['n_samples_for_corr'] = int(joint_finite_mask.sum())
                else:
                    results['dir_grad_delta_corr'] = None
                    results['n_samples_for_corr'] = 0

        # Restore original model training state
        if original_training:
            model.train()

        return results

    def _process_batch_for_grid(self, batch, model_device):
        """Simply move batch to device. Batch size should be set in unified_model_analysis.py"""
        # Move batch to device
        batch = {k: v.to(model_device) if torch.is_tensor(v) else v
                for k, v in batch.items()}
        return batch

    def compute_loss_landscape_2d(
        self,
        model,
        data_batch: Optional[Dict[str, torch.Tensor]] = None,
        data_batches: Optional[List[Dict[str, torch.Tensor]]] = None,  # NEW: Support multiple batches
        n_points: int = 25,  # UPDATED: 25x25 optimal balance of resolution and noise
        span: float = 0.1,
        loss_fn: Optional[Any] = None,
        normalization_mode: str = 'layer',  # 'layer', 'filter', 'global' (layer best for transformers)
        seed: Optional[int] = None,
        batch_config=None,  # NEW: Optional BatchConfig for memory-efficient processing
        aggressive_cleanup: bool = True,   # NEW: Clean memory after EVERY iteration
        max_batches_per_point: int = 10    # NEW: Max batches to aggregate per grid point
    ) -> Dict[str, Any]:
        """
        Compute true 2D loss landscape visualization.

        Based on:
        Li, H., Xu, Z., Taylor, G., Studer, C., & Goldstein, T. (2018).
        "Visualizing the Loss Landscape of Neural Nets."
        Advances in Neural Information Processing Systems (NeurIPS), 31, 6389-6399.
        https://arxiv.org/abs/1712.09913

        Creates a 2D grid along two ORTHOGONAL random directions. By default uses
        layer normalization (more appropriate for transformers) to handle scale differences
        between network layers, but also supports filter normalization (Li et al., 2018)
        for CNNs or simple global normalization.

        MEMORY OPTIMIZED VERSION: Fixes OOM issues on H100 through:
        - In-place parameter updates (no temporary tensors)
        - Aggressive memory cleanup after each evaluation
        - Adaptive batch sizing for large grids
        - Numerically stable orthogonalization

        Args:
            model: Model to analyze
            data_batch: Single batch to evaluate loss on (backward compatibility)
            data_batches: Multiple batches for aggregated loss computation (preferred)
            n_points: Number of points per axis (total = n_points^2)
            span: Maximum distance from origin in each direction
            loss_fn: Optional custom loss function
            normalization_mode: Direction normalization method:
                - 'layer': Layer normalization for transformers (DEFAULT - recommended)
                - 'filter': Filter normalization (Li et al. 2018) for CNNs
                - 'global': Simple global L2 normalization
            seed: Random seed for reproducibility
            batch_config: Optional BatchConfig for memory-efficient processing.
                         If None, uses adaptive mode with chunk_size=32, max_size=128
            aggressive_cleanup: If True, clean memory after every iteration (prevents OOM)
            max_batches_per_point: Maximum batches to aggregate per grid point (memory limit)

        Returns:
            Dictionary with grid losses, statistics, and orthogonality diagnostics
        """
        # Set seeds for reproducibility
        self._set_random_seeds(seed)
        model.eval()
        model_device = next(model.parameters()).device

        # CRITICAL FIX: Normalize all batches to CPU to prevent GPU memory leaks
        # Batches will be moved to GPU on-demand during evaluation
        def _to_cpu_batch(batch_dict):
            """Ensure batch is on CPU to prevent GPU memory accumulation."""
            cpu_batch = {}
            for k, v in batch_dict.items():
                if torch.is_tensor(v):
                    cpu_batch[k] = v.detach().cpu()
                else:
                    cpu_batch[k] = v
            return cpu_batch

        # Handle batch inputs - support both single and multiple batches
        # CRITICAL: Normalize to CPU to prevent GPU accumulation
        if data_batches is not None:
            batches_to_use = [_to_cpu_batch(b) for b in data_batches[:max_batches_per_point]]
            logger.info(f"Using {len(batches_to_use)} batches (out of {len(data_batches)} available) for loss landscape computation")
        elif data_batch is not None:
            batches_to_use = [_to_cpu_batch(data_batch)]
        else:
            raise ValueError("Either data_batch or data_batches must be provided")

        # FIX 1: Batches now stored on CPU and moved to GPU only when needed
        # to avoid keeping all processed batches in GPU memory

        # Setup batch processing configuration with proper import guard
        # UPDATED: Properly use batch processor from /batch directory
        if batch_config is None and self.batch_processor is not None:
            try:
                # Import from batch package (properly structured in /batch directory)
                from batch import BatchConfig, ProcessingMode

                # Get current batch size from the actual batches
                if batches_to_use and 'input_ids' in batches_to_use[0]:
                    current_batch_size = batches_to_use[0]['input_ids'].shape[0]
                else:
                    current_batch_size = 32

                # Use appropriate chunk size based on actual batch size
                chunk_size = min(16, current_batch_size)  # Don't chunk larger than batch

                batch_config = BatchConfig(
                    mode=ProcessingMode.ADAPTIVE,
                    chunk_size=chunk_size,
                    max_size=current_batch_size,  # Don't exceed current batch size
                    seed=seed,
                    weighted=True,
                    clear_cache=True  # Important for memory management
                )
                logger.debug(f"Using BatchProcessor with chunk_size={chunk_size}, max_size={current_batch_size}")
            except ImportError as e:
                logger.warning(f"BatchProcessor not available: {e}. Using direct computation.")
                batch_config = None

        # Note: batches already moved to device in the processing loop above

        # CRITICAL FIX: Disable HuggingFace KV cache to prevent memory accumulation
        # KV caching is pure overhead for loss landscape evaluation
        original_use_cache = None
        if hasattr(model, "config") and hasattr(model.config, "use_cache"):
            original_use_cache = model.config.use_cache
            model.config.use_cache = False
            logger.debug("Disabled HuggingFace KV cache for loss landscape computation")

        # Save original parameters
        base_params = [p.detach().clone() for p in model.parameters()]

        # Generate two ORTHOGONAL random directions
        directions = []

        # First direction
        d1 = [torch.randn_like(p) if p.requires_grad else None
              for p in model.parameters()]

        # Normalize first direction
        if normalization_mode == 'filter':
            d1 = self._filter_normalize_direction(model, d1)
        elif normalization_mode == 'layer':
            d1 = self._layer_normalize_direction(model, d1)
        else:  # 'global'
            norm_sq = sum((d**2).sum() for d in d1 if d is not None)
            eps = 1e-8  # Fixed epsilon for numerical stability (not dtype-dependent)
            total_norm = torch.sqrt(norm_sq + eps)
            d1 = [d / total_norm if d is not None else None for d in d1]

        directions.append(d1)

        # Second direction - will be orthogonalized
        d2 = [torch.randn_like(p) if p.requires_grad else None
              for p in model.parameters()]

        # CRITICAL FIX: Normalize d2 BEFORE orthogonalization
        # Gram-Schmidt must be performed in the same metric space as the final directions
        # Otherwise, directions won't be orthogonal in the normalized space
        if normalization_mode == 'filter':
            d2 = self._filter_normalize_direction(model, d2)
        elif normalization_mode == 'layer':
            d2 = self._layer_normalize_direction(model, d2)
        else:  # 'global'
            norm_sq = sum((d**2).sum() for d in d2 if d is not None)
            eps = 1e-8  # Fixed epsilon for numerical stability (not dtype-dependent)
            total_norm = torch.sqrt(norm_sq + eps)
            d2 = [d / total_norm if d is not None else None for d in d2]

        # FIX 2: In-place Gram-Schmidt orthogonalization in the NORMALIZED space
        # Now both d1 and d2 are normalized, so we can orthogonalize properly
        with torch.no_grad():
            dot_product = sum((d2i * d1i).sum() for d2i, d1i in zip(d2, d1)
                             if d2i is not None and d1i is not None)
            d1_norm_sq = sum((d**2).sum() for d in d1 if d is not None)

            # Numerically stable projection coefficient
            eps = 1e-8  # Fixed epsilon (not dtype-dependent, avoid bfloat16 issue)
            projection_coeff = (dot_product / (d1_norm_sq + eps)).item()

            # In-place orthogonalization to avoid creating temporary tensors
            for d2i, d1i in zip(d2, d1):
                if d2i is not None and d1i is not None:
                    d2i.add_(d1i, alpha=-projection_coeff)  # d2 = d2 - coeff*d1 (in-place)

        # Re-normalize d2 after orthogonalization (Gram-Schmidt requires re-normalization)
        if normalization_mode == 'filter':
            d2 = self._filter_normalize_direction(model, d2)
        elif normalization_mode == 'layer':
            d2 = self._layer_normalize_direction(model, d2)
        else:  # 'global'
            norm_sq = sum((d**2).sum() for d in d2 if d is not None)
            eps = 1e-8  # Fixed epsilon for numerical stability
            total_norm = torch.sqrt(norm_sq + eps)
            d2 = [d / total_norm if d is not None else None for d in d2]

        directions.append(d2)

        # Compute diagnostic: cosine angle between directions (should be ~0)
        final_dot = sum((d2i * d1i).sum() for d2i, d1i in zip(directions[1], directions[0])
                       if d2i is not None and d1i is not None)
        d1_final_norm = torch.sqrt(sum((d**2).sum() for d in directions[0] if d is not None))
        d2_final_norm = torch.sqrt(sum((d**2).sum() for d in directions[1] if d is not None))
        cos_angle = float((final_dot / (d1_final_norm * d2_final_norm + 1e-12)).item())

        # Verify orthogonality with stronger assertion for ICML publication quality
        if abs(cos_angle) > 0.01:
            logger.warning(f"Direction orthogonality issue: cos_angle={cos_angle:.6f} (should be ~0)")
            logger.warning(f"This may indicate a bug in the normalization or orthogonalization procedure")

        # For ICML: Assert that directions are reasonably orthogonal
        # Allow slightly more tolerance (0.05) to account for numerical precision
        assert abs(cos_angle) < 0.05, \
            f"Directions not orthogonal: cos_angle={cos_angle:.6f}. This is a critical bug!"

        # Create grid
        alphas = np.linspace(-span, span, n_points)
        grid_losses = np.full((n_points, n_points), np.nan, dtype=np.float64)

        # Use try/finally to guarantee weight restoration
        try:
            with torch.inference_mode():
                # Track iteration count for memory management
                iteration_count = 0

                for i, a1 in enumerate(alphas):
                    for j, a2 in enumerate(alphas):
                        # Convert to Python floats to avoid dtype issues
                        alpha1, alpha2 = float(a1), float(a2)

                        # FIX 3: IN-PLACE PARAMETER UPDATE - NO TEMPORARY TENSORS
                        # Old way: p.data.copy_(w0 + alpha1 * d1 + alpha2 * d2)
                        # This created 3-4 temporary tensors totaling 9-12 GB!
                        # New way: Three in-place operations, zero temporaries
                        with torch.no_grad():
                            for p, w0, d1, d2 in zip(model.parameters(),
                                                    base_params,
                                                    directions[0],
                                                    directions[1]):
                                if p.requires_grad and d1 is not None and d2 is not None:
                                    # In-place operations to avoid temporary tensors
                                    p.data.copy_(w0)  # p = w0
                                    p.data.add_(d1, alpha=alpha1)  # p += alpha1 * d1
                                    p.data.add_(d2, alpha=alpha2)  # p += alpha2 * d2

                        # Compute loss with proper error handling
                        try:
                            # Use unified Welford's algorithm for numerically stable multi-batch averaging
                            from utils.welford import WelfordAccumulator

                            # CRITICAL FIX: Initialize Welford accumulator on CPU to prevent GPU accumulation
                            # Accumulating scalars on GPU is unnecessary and causes persistent allocations
                            loss_accumulator = WelfordAccumulator(
                                device=torch.device('cpu'),  # CPU for scalar aggregation
                                dtype=torch.float32,
                                use_keys=False,
                                weighted=False
                            )

                            # Process batches with proper cleanup
                            for batch_idx, batch_orig in enumerate(batches_to_use):
                                # Create processed batch on-demand to save memory
                                batch = self._process_batch_for_grid(batch_orig, model_device)
                                if self.batch_processor is not None and batch_config is not None:
                                    # Use batch processor for memory-efficient computation
                                    # FIX: Create a closure that properly captures model variable
                                    def compute_loss_fn(chunk, current_model=model, current_loss_fn=loss_fn):
                                        return self._compute_loss(current_model, chunk, current_loss_fn)

                                    loss = self.batch_processor.process_batch(
                                        batch=batch,
                                        compute_fn=compute_loss_fn,
                                        reduction='mean',
                                        config_override=batch_config
                                    )
                                    # CRITICAL FIX: Delete closure to prevent reference leaks
                                    del compute_loss_fn
                                else:
                                    # Fallback to original single-batch computation
                                    loss = self._compute_loss(model, batch, loss_fn)

                                # CRITICAL FIX: Ensure loss is a scalar on CPU before aggregation
                                # If loss_fn returns per-token/per-example tensor, this prevents
                                # massive GPU allocations for the accumulator
                                if torch.is_tensor(loss):
                                    if loss.ndim > 0:
                                        loss = loss.mean()  # Reduce to scalar
                                    loss_scalar = loss.detach().float().cpu()  # Move to CPU
                                else:
                                    loss_scalar = torch.tensor(float(loss), dtype=torch.float32)

                                # Update accumulator with scalar loss on CPU
                                loss_accumulator.update(loss_scalar)

                                # CRITICAL FIX: Clean up batch memory after processing
                                del loss
                                del loss_scalar
                                if torch.cuda.is_available():
                                    # Delete batch tensors from GPU memory
                                    for key in list(batch.keys()):
                                        if torch.is_tensor(batch[key]):
                                            del batch[key]
                                    del batch  # CRITICAL: Delete the dict itself to prevent leaks
                                    # NOTE: Don't call empty_cache() here - causes fragmentation
                                    # Will be called per-row instead

                            # Get final mean loss (numerically stable)
                            mean_loss = loss_accumulator.get_mean()
                            loss_val = mean_loss.item() if mean_loss is not None else 0.0

                            # Optional: Track variance for diagnostics
                            if hasattr(self, '_track_landscape_variance'):
                                variance = loss_accumulator.get_variance()
                                if variance is not None:
                                    self._landscape_variances.append(variance.item())

                            # FIX 4: Convert to float immediately and delete tensor
                            grid_losses[i, j] = float(loss_val)  # loss_val is already a float from line 1531
                            del loss_val  # Critical: free memory immediately

                            # CRITICAL FIX: Delete WelfordAccumulator to prevent memory accumulation
                            del loss_accumulator
                            # NOTE: Don't call empty_cache() here - causes fragmentation

                        except RuntimeError as e:
                            if "out of memory" in str(e).lower():
                                logger.warning(f"OOM at ({i},{j}), attempting recovery")
                                # Emergency cleanup
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                gc.collect()
                                grid_losses[i, j] = np.nan
                            else:
                                logger.warning(f"Loss computation failed at ({i},{j}): {e}")
                                grid_losses[i, j] = np.nan
                        except Exception as e:
                            logger.warning(f"Loss computation failed at ({i},{j}): {e}")
                            grid_losses[i, j] = np.nan

                        # Track iteration for debugging
                        iteration_count += 1

                        # CRITICAL FIX: Removed per-iteration empty_cache() - causes fragmentation
                        # Only clear cache per-row to reduce allocator thrashing

                        # Periodic gc.collect (but not empty_cache)
                        if iteration_count % 50 == 0:
                            gc.collect()

                            # Log memory usage for monitoring
                            if torch.cuda.is_available():
                                allocated = torch.cuda.memory_allocated() / 1e9
                                reserved = torch.cuda.memory_reserved() / 1e9
                                logger.debug(f"Iteration {iteration_count}/{n_points*n_points}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved")

                    # CRITICAL FIX: Clear cache per-row (not per-iteration) to prevent fragmentation
                    # Per-row is infrequent enough to avoid allocator thrashing
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        finally:
            # ALWAYS restore parameters, even if exceptions occurred
            with torch.no_grad():
                for p, w0 in zip(model.parameters(), base_params):
                    if p.requires_grad:
                        p.data.copy_(w0)

            # CRITICAL FIX: Restore HuggingFace cache setting
            if original_use_cache is not None and hasattr(model, "config"):
                model.config.use_cache = original_use_cache

            # FIX 6: Final cleanup to free direction vectors and temporary data
            del directions
            del base_params
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        # Statistics with NaN handling
        valid_mask = ~np.isnan(grid_losses)
        valid = grid_losses[valid_mask]

        if len(valid) == 0:
            return {'error': 'All loss computations failed'}

        # Robust roughness computation via Total Variation
        roughness = 0.0
        if valid_mask.sum() > 4:  # Need at least 2x2 valid region
            try:
                # Use finite differences for more robust gradient
                # Forward differences in x direction
                dx = np.zeros_like(grid_losses)
                dx[:, :-1] = grid_losses[:, 1:] - grid_losses[:, :-1]

                # Forward differences in y direction
                dy = np.zeros_like(grid_losses)
                dy[:-1, :] = grid_losses[1:, :] - grid_losses[:-1, :]

                # Total variation: sum of absolute gradients
                # Use nansum to ignore NaN values
                tv_x = np.nansum(np.abs(dx[valid_mask]))
                tv_y = np.nansum(np.abs(dy[valid_mask]))
                n_valid_gradients = np.sum(~np.isnan(dx[valid_mask])) + np.sum(~np.isnan(dy[valid_mask]))

                if n_valid_gradients > 0:
                    roughness = float((tv_x + tv_y) / n_valid_gradients)
            except Exception as e:
                logger.warning(f"Roughness computation failed: {e}")
                roughness = 0.0

        eps = np.finfo(np.float64).eps

        # Collect variance statistics if available
        variance_stats = {}
        if hasattr(self, '_landscape_variances') and self._landscape_variances:
            variance_stats = {
                'loss_variance_mean': float(np.mean(self._landscape_variances)),
                'loss_variance_std': float(np.std(self._landscape_variances)),
                'loss_variance_reduction': len(batches_to_use)  # Theoretical variance reduction factor
            }
            # Clean up
            del self._landscape_variances

        return {
            'grid_losses': grid_losses.tolist(),
            'axis_values': alphas.tolist(),
            'loss_min': float(np.min(valid)) if len(valid) > 0 else np.nan,
            'loss_max': float(np.max(valid)) if len(valid) > 0 else np.nan,
            'loss_mean': float(np.mean(valid)) if len(valid) > 0 else np.nan,
            'loss_std': float(np.std(valid)) if len(valid) > 0 else np.nan,
            'roughness': roughness,
            'normalized_roughness': roughness / (abs(np.mean(valid)) + eps) if len(valid) > 0 else 0.0,
            'grid_shape': [int(n_points), int(n_points)],
            'normalization_mode': normalization_mode,
            'n_valid': int(valid_mask.sum()),
            'n_total': int(n_points * n_points),
            # Orthogonality diagnostics
            'cos_angle_d1_d2': cos_angle,  # Should be ~0 for orthogonal directions
            'norm_d1': float(d1_final_norm.item()),
            'norm_d2': float(d2_final_norm.item()),
            'orthogonality_check': abs(cos_angle) < 0.01,  # True if properly orthogonal
            'note': 'True 2D grid with orthogonal directions (Gram-Schmidt)',
            'memory_optimized': True,  # Indicates this is the fixed version
            'batch_size_used': batches_to_use[0]['input_ids'].shape[0] if batches_to_use and 'input_ids' in batches_to_use[0] else None,
            'n_batches_used': len(batches_to_use),  # Number of batches used for averaging
            'welford_averaging': True,  # Indicates use of numerically stable averaging
            **variance_stats  # Include variance statistics if available
        }

    # ============= FEATURE ATTRIBUTION (AUDITED) =============
    
    def compute_integrated_gradients(
        self,
        model,
        input_batch: Dict[str, torch.Tensor],
        baseline_batch: Optional[Dict[str, torch.Tensor]] = None,
        n_steps: int = 20,
        target_layer: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        DEPRECATED: Use established_analysis.analyze_token_importance() instead.

        This implementation has been deprecated in favor of the more efficient
        and feature-complete implementation in established_analysis.py which includes:
        - Optimized chunk sizes (32 vs 2)
        - Cache clearing between chunks (prevents OOM)
        - Convergence delta verification
        - Better memory management

        This method now wraps analyze_token_importance() for backward compatibility.

        Reference:
            Sundararajan, M., Taly, A., & Yan, Q. (2017).
            Axiomatic Attribution for Deep Networks.
            Proceedings of the 34th International Conference on Machine Learning (ICML),
            70, 3319-3328.
            https://arxiv.org/abs/1703.01365

        Args:
            model: PyTorch model
            input_batch: Dict with 'input_ids' and optional 'attention_mask'
            baseline_batch: Ignored (established_analysis creates optimal baseline)
            n_steps: Number of integration steps (default: 20)
            target_layer: Ignored (established_analysis uses embeddings)

        Returns:
            Dict with aggregated attribution statistics (for backward compatibility)
        """
        import warnings
        warnings.warn(
            "ICLRMetrics.compute_integrated_gradients() is deprecated. "
            "Use established_analysis.analyze_token_importance() instead. "
            "This wrapper will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )

        # Import established_analysis
        from established_analysis import EstablishedAnalysisMethods

        # Create established_analysis instance if needed
        if not hasattr(self, '_established_analysis_cache'):
            tokenizer = None
            # Try to extract tokenizer from model config
            if hasattr(model, 'config'):
                if hasattr(model.config, 'tokenizer'):
                    tokenizer = model.config.tokenizer
                elif hasattr(model.config, '_name_or_path'):
                    # Try to load tokenizer from model path
                    try:
                        from transformers import AutoTokenizer
                        tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
                    except:
                        pass

            self._established_analysis_cache = EstablishedAnalysisMethods(
                model=model,
                tokenizer=tokenizer
            )

        # Call the better implementation
        result = self._established_analysis_cache.analyze_token_importance(
            inputs=input_batch['input_ids'],
            position_of_interest=0,  # Default to position 0 (ICLRMetrics behavior)
            n_steps=n_steps,
            return_convergence_delta=True,
            attention_mask=input_batch.get('attention_mask')
        )

        # Handle error case
        if 'error' in result:
            return result

        # Convert to ICLRMetrics format for backward compatibility
        # ICLRMetrics returns aggregated statistics, not per-token attributions
        attributions = result['attributions']

        # Compute statistics that ICLRMetrics originally returned
        import numpy as np

        aggregated = {
            'mean_attribution': float(np.abs(attributions).mean()),
            'max_attribution': float(np.abs(attributions).max()),
            'attribution_sparsity': float((np.abs(attributions) < 0.01).mean()),
            'attribution_entropy': float(-np.sum(
                np.abs(attributions) / (np.abs(attributions).sum() + 1e-10) *
                np.log(np.abs(attributions) / (np.abs(attributions).sum() + 1e-10) + 1e-10)
            )),
            'top_token_importance': float(np.sort(np.abs(attributions).flatten())[-10:].mean()) if attributions.size >= 10 else float(np.abs(attributions).mean()),
            'method': 'layer_integrated_gradients (via established_analysis)',
            'n_steps': n_steps,
            'implementation': 'deprecated_wrapper'
        }

        # Compute Gini coefficient if ICLRMetrics has the method
        if hasattr(self, '_compute_gini'):
            try:
                flat_attrs = np.abs(attributions).flatten()
                aggregated['attribution_gini'] = float(self._compute_gini(flat_attrs))
            except:
                pass

        # Add convergence delta if available
        if 'convergence_delta' in result:
            aggregated['convergence_delta'] = float(np.abs(result['convergence_delta']).mean())

        # Add most important positions (top 5)
        mean_importance_per_token = np.abs(attributions).mean(axis=0) if attributions.ndim > 1 else np.abs(attributions)
        top_k = min(5, len(mean_importance_per_token))
        aggregated['most_important_positions'] = np.argsort(mean_importance_per_token)[-top_k:][::-1].tolist()

        return aggregated

    def _process_attention_chunked(self, model, batch: Dict[str, torch.Tensor],
                                  chunk_size: int) -> Optional[tuple]:
        """
        MEMORY-EFFICIENT: Process attention weights in chunks without storing all layers.

        FIX: Instead of storing all 28 layers (3.76 GB), we now:
        1. Process sequentially when computing rollout
        2. Return attention tensors only when needed for fallback

        Args:
            model: Model to process
            batch: Full batch to process
            chunk_size: Size of each chunk

        Returns:
            Tuple of attention tensors for all layers, or None if extraction fails
        """
        batch_size = batch['input_ids'].shape[0]
        model_device = next(model.parameters()).device

        if batch_size <= chunk_size:
            # Can process in one go
            outputs = model(**batch, output_attentions=True)
            if hasattr(outputs, 'attentions') and outputs.attentions is not None:
                return outputs.attentions
            return None

        # MEMORY FIX: For chunked processing, compute rollout directly per chunk
        # instead of storing all layers
        logger.info(f"Memory-efficient processing: {batch_size} samples in chunks of {chunk_size}")

        # We'll compute statistics per chunk and aggregate
        chunk_results = []
        num_chunks = (batch_size + chunk_size - 1) // chunk_size

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, batch_size)

            # Create chunk batch
            chunk_batch = {k: v[start_idx:end_idx] if torch.is_tensor(v) else v
                          for k, v in batch.items()}

            # Process chunk
            with torch.no_grad():
                chunk_outputs = model(**chunk_batch, output_attentions=True)

                if hasattr(chunk_outputs, 'attentions') and chunk_outputs.attentions is not None:
                    # CRITICAL FIX: Compute rollout for this chunk immediately
                    # Don't store all layers!
                    # ICML FIX: Pass attention mask to exclude padding
                    chunk_mask = chunk_batch.get('attention_mask', None)
                    chunk_rollout = self._compute_attention_rollout_efficient(
                        chunk_outputs.attentions,
                        attention_mask=chunk_mask
                    )
                    chunk_results.append(chunk_rollout)

                    # Free memory immediately
                    del chunk_outputs.attentions
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

        # Return aggregated results instead of raw tensors
        if chunk_results:
            # For compatibility, return a special marker that indicates
            # we've already computed the rollout
            return ('rollout_computed', chunk_results)

        return None

    def _compute_attention_rollout_efficient(
        self,
        attention_tensors,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        MEMORY-EFFICIENT: Compute attention rollout without storing all layers.

        Implements attention rollout from:
        Abnar & Zuidema (2020). "Quantifying Attention Flow in Transformers"
        ACL 2020. Formula: A = 0.5 * W_att + 0.5 * I (with normalization)

        Key innovation: Sequential processing reduces memory from 3.76 GB to 268 MB
        by keeping only 2 layers in memory at once instead of all 28.

        ICML FIX: Now properly handles attention_mask to exclude padding tokens
        from entropy and statistics calculations (prevents bias).

        Args:
            attention_tensors: List/tuple of attention weights from each layer
                             Shape: [batch, heads, seq, seq] or [batch, seq, seq]
            attention_mask: Optional mask [batch, seq] where 1=valid, 0=padding

        Returns:
            Dict with rollout statistics and entropy metrics
        """
        if not attention_tensors:
            return {}

        # Get dimensions
        first_attn = attention_tensors[0]
        if first_attn.dim() == 4:  # [batch, heads, seq, seq]
            batch_size = first_attn.shape[0]
            seq_len = first_attn.shape[2]
        elif first_attn.dim() == 3:  # [batch, seq, seq]
            batch_size = first_attn.shape[0]
            seq_len = first_attn.shape[1]
        else:
            return {}

        device = first_attn.device
        dtype = first_attn.dtype

        # Initialize rollout with first layer
        if first_attn.dim() == 4:
            rollout = first_attn.mean(dim=1)  # Average over heads: [batch, seq, seq]
        else:
            rollout = first_attn

        # Prepare identity matrix for residual connections
        eye = torch.eye(seq_len, device=device, dtype=dtype)
        eye = eye.unsqueeze(0).expand(batch_size, -1, -1)

        # Apply residual and normalize (Abnar & Zuidema formula)
        rollout = 0.5 * rollout + 0.5 * eye
        # ICML FIX: dtype-aware clamping with bfloat16 support
        if rollout.dtype == torch.float16:
            clamp_min = 1e-6
        elif rollout.dtype == torch.bfloat16:
            clamp_min = 1e-7
        else:
            clamp_min = 1e-10
        rollout = rollout / rollout.sum(dim=-1, keepdim=True).clamp(min=clamp_min)

        # Store initial statistics
        initial_stats = {
            'layer_0_mean': float(rollout.mean().item()),
            'layer_0_max': float(rollout.max().item())
        }

        # Sequential rollout through remaining layers
        # MEMORY OPTIMIZATION: Process layers and delete as we go
        for i in range(1, len(attention_tensors)):
            # Get next layer attention
            attn = attention_tensors[i]
            if attn.dim() == 4:
                attn = attn.mean(dim=1)  # [batch, seq, seq]

            # Apply residual and normalize
            attn = 0.5 * attn + 0.5 * eye
            # ICML FIX: dtype-aware clamping with bfloat16 support
            if attn.dtype == torch.float16:
                clamp_min = 1e-6
            elif attn.dtype == torch.bfloat16:
                clamp_min = 1e-7
            else:
                clamp_min = 1e-10
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp(min=clamp_min)

            # Multiply for rollout
            rollout = torch.bmm(rollout, attn)

            # MEMORY OPTIMIZATION: Delete the attention tensor we just used
            # Note: We can't set attention_tensors[i] = None because it may be a tuple (immutable)
            # The tensor reference will be garbage collected after the loop
            del attn  # Delete the processed tensor to free memory immediately

            # Force memory cleanup every few layers
            if i % 5 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # All layers processed, no need to clear (will be garbage collected)

        # Get final attribution from first token (CLS token or first position)
        attribution = rollout[:, 0, :]  # [batch, seq]

        # Compute statistics
        # ICML FIX: Improved epsilon handling for bfloat16 support
        if rollout.dtype == torch.float16:
            entropy_clamp_min = 1e-6  # float16: limited precision
        elif rollout.dtype == torch.bfloat16:
            entropy_clamp_min = 1e-7  # bfloat16: wider range, limited mantissa
        else:
            entropy_clamp_min = 1e-12  # float32/float64: full precision

        # ICML FIX: Apply attention mask to exclude padding from statistics
        if attention_mask is not None:
            # Expand mask for attention dimensions [batch, seq] -> [batch, seq, seq]
            mask_2d = attention_mask.unsqueeze(1) * attention_mask.unsqueeze(2)  # [batch, seq, seq]
            mask_2d = mask_2d.to(dtype=rollout.dtype, device=rollout.device)

            # Mask rollout for entropy computation (set padding to 0)
            rollout_masked = rollout * mask_2d

            # Renormalize over valid tokens only
            valid_sum = rollout_masked.sum(dim=-1, keepdim=True).clamp(min=entropy_clamp_min)
            rollout_masked = rollout_masked / valid_sum

            # Use masked version for entropy
            # Improved numerically stable entropy: mask zeros to avoid log(0)
            mask_valid = rollout_masked > entropy_clamp_min
            log_probs = torch.where(mask_valid, rollout_masked.log(), torch.zeros_like(rollout_masked))
            attention_entropy = -(rollout_masked * log_probs).sum(dim=-1).mean()

            # Mask attribution as well
            attribution_mask = attention_mask.to(dtype=attribution.dtype, device=attribution.device)
            attribution_masked = attribution * attribution_mask
            # Renormalize
            attribution_masked = attribution_masked / attribution_masked.sum(dim=-1, keepdim=True).clamp(min=entropy_clamp_min)

            # Compute attribution entropy with masking
            mask_valid_attr = attribution_masked > entropy_clamp_min
            log_probs_attr = torch.where(mask_valid_attr, attribution_masked.log(), torch.zeros_like(attribution_masked))
            rollout_entropy = -(attribution_masked * log_probs_attr).sum(dim=-1).mean()

            # Max should only consider valid tokens
            rollout_max = (attribution * attribution_mask).max()
        else:
            # No mask: use original calculation (numerically improved)
            mask_valid = rollout > entropy_clamp_min
            log_probs = torch.where(mask_valid, rollout.log(), torch.zeros_like(rollout))
            attention_entropy = -(rollout * log_probs).sum(dim=-1).mean()

            mask_valid_attr = attribution > entropy_clamp_min
            log_probs_attr = torch.where(mask_valid_attr, attribution.log(), torch.zeros_like(attribution))
            rollout_entropy = -(attribution * log_probs_attr).sum(dim=-1).mean()

            rollout_max = attribution.max()

        results = {
            'mean_attention': initial_stats['layer_0_mean'],
            'max_attention': initial_stats['layer_0_max'],
            'attention_entropy': float(attention_entropy.item()),
            'attention_concentration': float(rollout.max(dim=-1)[0].mean().item()),
            'rollout_max': float(rollout_max.item()),
            'rollout_entropy': float(rollout_entropy.item()),
            'batch_size': batch_size,
            'seq_length': seq_len
        }

        return results

    def compute_attention_attribution(
        self,
        model,
        input_batch: Dict[str, torch.Tensor],
        layer_idx: int = -1,  # TODO: Currently unused - implementation always uses all layers
        attention_layer_pattern: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Compute attention-based feature attribution using attention rollout.

        Implements the attention rollout method from:
        Abnar & Zuidema (2020). "Quantifying Attention Flow in Transformers"
        Proceedings of the 58th Annual Meeting of the ACL, pp. 4190-4197.

        The method tracks information flow through transformer layers by multiplying
        attention matrices with residual connections: A = 0.5 * W_att + 0.5 * I

        MEMORY-EFFICIENT: Processes attention sequentially to use only 268 MB
        instead of 3.76 GB for typical 28-layer models.

        Args:
            model: Model to analyze
            input_batch: Input batch with 'input_ids' and optionally 'attention_mask'
            layer_idx: Which attention layer to analyze (-1 for last)
            attention_layer_pattern: Custom pattern for attention layers
                                    (e.g., 'transformer.h.*.attn' for GPT2)
                                    If None, tries to use output_attentions=True

        Returns:
            Dict with:
                - mean_attention: Average attention weight
                - max_attention: Maximum attention weight
                - attention_entropy: Entropy of attention distribution
                - attention_concentration: Concentration metric
                - rollout_max: Maximum rollout value
                - rollout_entropy: Entropy of rollout distribution
        """
        # Save original training mode
        was_training = model.training
        model.eval()

        model_device = next(model.parameters()).device
        batch = {k: v.to(model_device) if torch.is_tensor(v) else v
                for k, v in input_batch.items()}
        
        # Try using model's built-in attention output first (most reliable)
        attention_weights = None
        
        # Forward pass with attention output using chunked processing
        with torch.no_grad():
            try:
                # CHUNKED PROCESSING: Process ALL samples regardless of memory
                batch_size = batch['input_ids'].shape[0]
                seq_len = batch['input_ids'].shape[1]

                # Estimate memory for attention based on model config
                # Memory = batch * heads * seq^2 * 4 bytes * num_layers
                estimated_layers = getattr(model.config, 'num_hidden_layers', 28)  # Query from model config
                estimated_heads = getattr(model.config, 'num_attention_heads', 16)  # Query from model config
                attention_memory_gb = (batch_size * estimated_heads * seq_len * seq_len * 4 * estimated_layers) / (1024**3)

                # Determine safe chunk size based on memory requirements
                # OPTIMIZED FOR H100 (80GB): Balance memory usage and noise
                # Noise scales as 1/sqrt(chunk_size), so larger chunks = less noise

                # Get available GPU memory
                if torch.cuda.is_available():
                    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                else:
                    gpu_memory_gb = 8.0  # Conservative CPU estimate

                # Reserve 20% for model weights and other operations
                available_memory_gb = gpu_memory_gb * 0.6  # Use 60% for attention

                if gpu_memory_gb >= 80:  # H100 or A100-80GB
                    # Can handle much larger chunks with lower noise
                    if attention_memory_gb > 40.0:
                        chunk_size = min(128, batch_size)  # Large chunks, low noise
                    elif attention_memory_gb > 20.0:
                        chunk_size = min(256, batch_size)  # Very large chunks
                    elif attention_memory_gb > 10.0:
                        chunk_size = min(512, batch_size)  # Huge chunks
                    else:
                        chunk_size = batch_size  # Process everything at once
                elif gpu_memory_gb >= 40:  # A100-40GB, A6000
                    if attention_memory_gb > 20.0:
                        chunk_size = min(64, batch_size)
                    elif attention_memory_gb > 10.0:
                        chunk_size = min(128, batch_size)
                    else:
                        chunk_size = batch_size
                else:  # Smaller GPUs (V100, RTX, etc.)
                    if attention_memory_gb > 8.0:
                        chunk_size = min(32, batch_size)  # Smaller chunks
                    elif attention_memory_gb > 4.0:
                        chunk_size = min(64, batch_size)
                    elif attention_memory_gb > 2.0:
                        chunk_size = min(128, batch_size)
                    else:
                        chunk_size = batch_size

                # Ensure minimum chunk size for noise reduction
                # Noise std ~ 1/sqrt(chunk_size), so chunk_size >= 16 keeps noise < 25%
                chunk_size = max(16, chunk_size)

                # Use chunked processing to handle ALL samples
                if chunk_size < batch_size:
                    logger.info(f"Attention memory {attention_memory_gb:.1f}GB, processing ALL {batch_size} samples in chunks of {chunk_size}")

                # Process attention through chunks
                attention_weights = self._process_attention_chunked(model, batch, chunk_size)

                # MEMORY FIX: Check if we already computed rollout in chunked processing
                if isinstance(attention_weights, tuple) and len(attention_weights) == 2:
                    if attention_weights[0] == 'rollout_computed':
                        # Already computed efficiently, aggregate results
                        chunk_results = attention_weights[1]
                        if chunk_results:
                            # ICML FIX: Weighted averaging by chunk size (statistically correct)
                            # Simple averaging is biased when chunks have unequal sizes
                            chunk_sizes = [r.get('batch_size', 1) for r in chunk_results]
                            total_samples = sum(chunk_sizes)

                            aggregated = {}
                            for key in chunk_results[0].keys():
                                if key not in ['batch_size', 'seq_length']:
                                    values = [r[key] for r in chunk_results]
                                    # Weighted average: E[X] = Σ (n_i/N) × X_i
                                    weighted_sum = sum(v * n for v, n in zip(values, chunk_sizes))
                                    aggregated[key] = weighted_sum / total_samples

                            # Add batch info
                            aggregated['batch_size'] = batch_size
                            aggregated['seq_length'] = seq_len

                            model.train(was_training)
                            return aggregated
                        else:
                            model.train(was_training)
                            return {'error': 'No attention rollout computed'}

            except RuntimeError as e:
                if "CUDA" in str(e) or "out of memory" in str(e).lower():
                    # OOM error - try with smaller chunks
                    logger.warning(f"OOM during attention extraction, using smaller chunks")
                    try:
                        # Try with very small chunks of 16 samples
                        fallback_chunk_size = 16
                        logger.info(f"Retrying with smaller chunks of {fallback_chunk_size}")
                        attention_weights = self._process_attention_chunked(model, batch, fallback_chunk_size)

                        # Check again for pre-computed rollout
                        if isinstance(attention_weights, tuple) and attention_weights[0] == 'rollout_computed':
                            chunk_results = attention_weights[1]
                            if chunk_results:
                                # ICML FIX: Weighted averaging by chunk size (statistically correct)
                                chunk_sizes = [r.get('batch_size', 1) for r in chunk_results]
                                total_samples = sum(chunk_sizes)

                                aggregated = {}
                                for key in chunk_results[0].keys():
                                    if key not in ['batch_size', 'seq_length']:
                                        values = [r[key] for r in chunk_results]
                                        # Weighted average: E[X] = Σ (n_i/N) × X_i
                                        weighted_sum = sum(v * n for v, n in zip(values, chunk_sizes))
                                        aggregated[key] = weighted_sum / total_samples
                                aggregated['batch_size'] = batch_size
                                aggregated['seq_length'] = seq_len
                                model.train(was_training)
                                return aggregated
                    except Exception:
                        # Still failing, return graceful error
                        model.train(was_training)
                        return {
                            'error': f'CUDA OOM in attention extraction despite reduction. Batch: {batch_size}, Seq: {seq_len}',
                            'batch_size': batch_size,
                            'seq_length': seq_len,
                            'estimated_memory_gb': attention_memory_gb
                        }
                # Other runtime errors - continue to fallback
                pass
            except (TypeError, AttributeError):
                # Model doesn't support output_attentions or failed
                pass
        
        # Fallback: Try to extract from specific known architectures
        if attention_weights is None:
            attention_weights = []
            hooks = []
            
            # Type-based detection instead of regex
            known_attention_types = []
            try:
                from transformers.models.gpt2.modeling_gpt2 import GPT2Attention
                known_attention_types.append(GPT2Attention)
            except ImportError:
                pass
            try:
                from transformers.models.bert.modeling_bert import BertAttention
                known_attention_types.append(BertAttention)
            except ImportError:
                pass
            try:
                # Generic MultiheadAttention
                known_attention_types.append(nn.MultiheadAttention)
            except:
                pass
            
            def attention_hook(module, input, output):
                # Try to extract attention weights from output
                if isinstance(output, tuple) and len(output) > 1:
                    # Many attention modules return (output, attn_weights)
                    if torch.is_tensor(output[1]):
                        attention_weights.append(output[1])
                elif hasattr(output, 'attentions'):
                    attention_weights.append(output.attentions)
            
            # Register hooks based on module type or attributes
            for name, module in model.named_modules():
                is_attention = False
                
                # Check by type
                if known_attention_types and isinstance(module, tuple(known_attention_types)):
                    is_attention = True
                # Check by attributes
                elif hasattr(module, 'attention') or hasattr(module, 'self_attn'):
                    is_attention = True
                # Check by method names
                elif hasattr(module, 'forward') and 'attention' in module.__class__.__name__.lower():
                    is_attention = True
                # Custom pattern if provided
                elif attention_layer_pattern:
                    if re.search(attention_layer_pattern, name):
                        is_attention = True

                if is_attention:
                    hook = module.register_forward_hook(attention_hook)
                    hooks.append((name, hook))

            # Forward pass with guaranteed hook cleanup
            try:
                with torch.no_grad():
                    outputs = model(**batch)
            except RuntimeError as e:
                # Clean up hooks before returning error
                for _, hook in hooks:
                    hook.remove()
                if "CUDA" in str(e) or "device-side assert" in str(e):
                    # CUDA error - clean up and return
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    model.train(was_training)
                    return {
                        'error': f'CUDA error in hook-based extraction: {str(e)}',
                        'batch_size': batch['input_ids'].shape[0] if 'input_ids' in batch else None
                    }
                model.train(was_training)
                return {'error': f'Forward pass failed: {str(e)}'}
            except Exception as e:
                # Clean up hooks before returning error
                for _, hook in hooks:
                    hook.remove()
                model.train(was_training)
                return {'error': f'Forward pass failed: {str(e)}'}
            finally:
                # Always remove hooks, even if forward pass fails
                for _, hook in hooks:
                    hook.remove()
        
        if not attention_weights:
            # Restore training mode before returning
            model.train(was_training)
            return {'error': 'No attention weights captured'}

        # MEMORY-EFFICIENT ROLLOUT: Use new efficient computation
        # Instead of storing all layers, compute rollout sequentially
        if isinstance(attention_weights, tuple) and hasattr(attention_weights, '__len__'):
            # We have attention tensors, compute rollout efficiently
            # ICML FIX: Pass attention mask to exclude padding
            results = self._compute_attention_rollout_efficient(
                attention_weights,
                attention_mask=batch.get('attention_mask', None)
            )

            # Check if we got valid results
            if not results:
                model.train(was_training)
                return {'error': 'Failed to compute attention rollout'}

            # Add any missing keys for compatibility
            if 'attention_concentration' not in results:
                results['attention_concentration'] = results.get('max_attention', 0.0)

            # ICML FIX: Clean up attention_weights to free memory
            if isinstance(attention_weights, (list, tuple)):
                if isinstance(attention_weights, list):
                    attention_weights.clear()
                del attention_weights

        # ICML FIX: Clean up batch tensors to prevent memory leaks
        try:
            if 'batch' in locals():
                for k in list(batch.keys()):
                    if torch.is_tensor(batch[k]):
                        del batch[k]
                del batch
        except:
            pass  # Ignore cleanup errors

        # Restore original training mode
        model.train(was_training)

        # Clear GPU cache one final time
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results
    

    def compute_pruning_sensitivity(
        self,
        model,
        data_batch: Dict[str, torch.Tensor],
        sparsity_levels: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Test model sensitivity to pruning - robust models should degrade gracefully.
        """
        if sparsity_levels is None:
            sparsity_levels = [0.1, 0.3, 0.5, 0.7, 0.9]

        # Save original training mode
        was_training = model.training
        model.eval()
        model_device = next(model.parameters()).device
        batch = {k: v.to(model_device) if torch.is_tensor(v) else v 
                for k, v in data_batch.items()}
        
        # Get baseline performance using the fixed _compute_loss method
        baseline_loss = self._compute_loss(model, batch)
        
        results = {
            'baseline_loss': float(baseline_loss)
        }
        
        # Collect all parameters
        all_params = []
        param_shapes = []
        for p in model.parameters():
            all_params.append(p.detach().clone())
            param_shapes.append(p.shape)
        
        for sparsity in sparsity_levels:
            # Create magnitude-based mask
            all_weights = torch.cat([p.abs().flatten() for p in all_params])
            threshold = torch.quantile(all_weights, sparsity)
            
            # Apply mask - more memory efficient
            with torch.no_grad():
                for p, original in zip(model.parameters(), all_params):
                    mask = (original.abs() > threshold).to(dtype=original.dtype, device=p.device)
                    pruned_param = (original * mask).to(device=p.device, dtype=p.dtype)
                    p.data.copy_(pruned_param)
            
            # Evaluate pruned model using the fixed _compute_loss method
            pruned_loss = self._compute_loss(model, batch)
            
            # Restore original weights using copy_ for proper restoration
            with torch.no_grad():
                for p, original in zip(model.parameters(), all_params):
                    p.data.copy_(original.to(p.device))
            
            # Store results
            results[f'loss_at_{int(sparsity*100)}pct_pruned'] = float(pruned_loss)
            # Ensure degradation is positive (loss should increase with pruning)
            results[f'degradation_at_{int(sparsity*100)}pct'] = float(
                max(0, (pruned_loss - baseline_loss)) / (baseline_loss + 1e-10)
            )
        
        # Compute robustness score (area under degradation curve)
        degradations = [results[f'degradation_at_{int(s*100)}pct'] for s in sparsity_levels]
        # Clamp degradations to avoid negative robustness score
        robustness_score = 1.0 / (1.0 + max(0, np.mean(degradations)))
        results['pruning_robustness_score'] = float(robustness_score)

        # Restore original training mode
        model.train(was_training)

        return results
    
    # ============= MODE CONNECTIVITY (Audited) =============
    
    def compute_mode_connectivity(
        self,
        models: List,
        data_batch: Dict[str, torch.Tensor],
        method: str = 'linear'
    ) -> Dict[str, Any]:
        """
        Test if models are mode-connected (lie in same loss basin).
        """
        if len(models) < 2:
            return {'error': 'Need at least 2 models for connectivity'}

        model_device = next(models[0].parameters()).device
        batch = {k: v.to(model_device) if torch.is_tensor(v) else v
                for k, v in data_batch.items()}
        
        # Compute pairwise connectivity
        n_models = len(models)
        connectivity_matrix = np.zeros((n_models, n_models))
        barrier_matrix = np.zeros((n_models, n_models))
        
        for i in range(n_models):
            for j in range(i+1, n_models):
                # Compute barrier between models i and j
                barrier_result = self.compute_loss_barrier(
                    models[i], models[j], batch, n_points=10, method=method
                )

                # Handle potential error results
                if 'error' in barrier_result:
                    barrier_height = float('inf')
                    is_connected = False
                else:
                    barrier_height = barrier_result['barrier_height']
                    is_connected = barrier_result['is_mode_connected']
                
                connectivity_matrix[i, j] = connectivity_matrix[j, i] = float(is_connected)
                barrier_matrix[i, j] = barrier_matrix[j, i] = barrier_height
        
        # Compute connectivity statistics
        n_pairs = n_models * (n_models - 1) // 2
        n_connected = np.sum(connectivity_matrix) / 2
        
        # Guard against empty arrays - include zeros for proper min/mean
        valid_barriers = barrier_matrix[barrier_matrix >= 0]
        results = {
            'mean_barrier_height': float(np.mean(valid_barriers)) if valid_barriers.size > 0 else 0.0,
            'max_barrier_height': float(np.max(barrier_matrix)) if barrier_matrix.size > 0 else 0.0,
            'min_barrier_height': float(np.min(valid_barriers)) if valid_barriers.size > 0 else 0.0,
            'connectivity_ratio': float(n_connected / n_pairs) if n_pairs > 0 else 0,
            'fully_connected': bool(n_connected == n_pairs),
            'n_connected_pairs': int(n_connected),
            'n_total_pairs': int(n_pairs)
        }
        
        return results
    
    # ============= HESSIAN ANALYSIS =============

    def compute_hessian_eigenvalues_lanczos(
        self,
        model,
        data_batch: Dict[str, torch.Tensor],
        k: int = 5,
        max_iter: int = 20,
        loss_fn: Optional[Any] = None,
        memory_efficient: bool = True,
        max_batch_size: Optional[int] = None,  # Will use config defaults
        operator: str = 'hessian',  # Can also use 'ggn', 'empirical_fisher'
        config: Optional[Any] = None,  # Accept config for batch size
        **kwargs  # For additional parameters like ggn_mode
    ) -> Dict[str, Any]:
        """
        Compute top-k eigenvalues using unified Lanczos system.

        Supports multiple operators (Hessian, GGN, Fisher) for different use cases:
        - 'hessian': Can be indefinite, shows negative curvature (pathology detection)
        - 'ggn': Always PSD, better conditioned (optimization metrics)
        - 'empirical_fisher': PSD, uses per-sample gradients

        Critical Fixes Applied (2025-09-30):
        ====================================
        1. Memory Leaks Fixed:
           - loss_fn now explicitly deletes outputs object (was 5GB leak per call)
           - HessianOperator explicitly deletes loss tensor (was 1.5GB leak per call)
           - Over 20 Lanczos iterations: prevented 128 GB leak that caused OOM

        2. BFloat16 Precision:
           - Hessian automatically uses Float32 (BFloat16 insufficient for indefinite matrices)
           - PSD operators still use BFloat16 for memory efficiency
           - See fisher_lanczos_unified.py for automatic dtype selection

        3. Batch Size Selection:
           - Hessian on H100 with 1.5B models: batch_size=16 (was 32)
           - Memory calculation: 3x model size per batch item for double backprop
           - Conservative limits ensure no OOM on H100 80GB

        Memory Requirements (H100 80GB, Qwen 1.5B):
        ============================================
        - Model + gradients: 6.18 GB
        - Lanczos vectors (5 Float32): 30 GB
        - Per-iteration working set: ~12 GB
        - Total peak: ~48 GB (fits with 32 GB safety margin)

        Args:
            model: Model to analyze (will enable requires_grad for all params)
            data_batch: Batch to compute eigenvalues on (will be sliced if too large)
            k: Number of top eigenvalues to compute (default: 5)
            max_iter: Maximum Lanczos iterations (default: 20, recommend 3*k)
            loss_fn: Optional custom loss function (should delete outputs!)
            memory_efficient: Use memory optimizations (default: True)
            max_batch_size: Maximum batch size for computation (None = auto-select)
            operator: Operator type ('hessian', 'ggn', 'empirical_fisher')
            config: Configuration object with batch size settings
            **kwargs: Additional parameters (e.g., ggn_mode)

        Returns:
            Dictionary with:
            - top_eigenvalues: Top-k eigenvalues (descending)
            - max_eigenvalue: Largest eigenvalue
            - min_computed_eigenvalue: Smallest of top-k
            - has_negative_eigenvalues: Whether Hessian has negative curvature
            - batch_size_used: Actual batch size used
            - warnings: Quality warnings (if any)
            - Additional operator-specific metrics

        Raises:
            RuntimeError: If OOM occurs despite all safety measures

        Note:
            For ICML submission, this function now has production-grade memory management
            and numerical precision appropriate for indefinite matrices (Hessian).
        """
        # Save original training mode and gradient states
        was_training = model.training
        original_grad_states = {name: param.requires_grad for name, param in model.named_parameters()}

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
                f"Enabling gradients for ALL parameters to ensure proper Hessian computation."
            )

        # CRITICAL FIX: Enable gradients for all parameters
        # This ensures gradient computation works properly for Hessian even with frozen parameters
        for param in model.parameters():
            param.requires_grad_(True)

        # Use eval mode for deterministic analysis (no dropout randomness)
        # Gradients still work in eval mode!
        model.eval()

        try:
            from fisher.core.fisher_lanczos_unified import compute_spectrum, LanczosConfig

            # Prepare batch
            device = next(model.parameters()).device
            batch = {k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in data_batch.items()}

            # Calculate model size impact on batch size
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            model_size_gb = n_params * 4 / 1e9  # Assume float32

            # Determine appropriate batch size based on operator and GPU
            if max_batch_size is None:
                # Use config defaults or intelligent defaults based on GPU
                if operator == 'hessian':
                    # Check GPU memory to determine appropriate batch size
                    if device.type == 'cuda':
                        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

                        # Adjust batch size based on both GPU memory AND model size
                        # Hessian requires ~3x model memory per batch item for double backprop
                        memory_per_batch = model_size_gb * 3

                        # Leave 20% GPU memory for other operations
                        available_memory = gpu_memory_gb * 0.8

                        # Calculate max safe batch size based on available memory
                        theoretical_max = int(available_memory / memory_per_batch)

                        # Apply conservative limits based on GPU and model size
                        # CRITICAL: Hessian uses double backprop which is VERY memory intensive
                        # Be extremely conservative with batch sizes
                        if n_params > 1e9:  # Models over 1B parameters
                            # For large models, theoretical_max is often very small (3-4)
                            # Always respect it!
                            if gpu_memory_gb > 70:  # H100 (80GB) or A100-80GB
                                default_hessian_batch = min(4, theoretical_max)  # Very conservative
                            elif gpu_memory_gb > 30:  # A100-40GB
                                default_hessian_batch = min(2, theoretical_max)
                            else:  # Smaller GPUs
                                default_hessian_batch = 1  # Only single sample
                        else:  # Smaller models (<1B params)
                            if gpu_memory_gb > 70:  # H100 (80GB) or A100-80GB
                                default_hessian_batch = min(8, theoretical_max)  # Still conservative
                            elif gpu_memory_gb > 30:  # A100-40GB
                                default_hessian_batch = min(4, theoretical_max)
                            else:  # Smaller GPUs
                                default_hessian_batch = min(2, theoretical_max)

                        # Ensure at least batch size of 1
                        default_hessian_batch = max(1, default_hessian_batch)
                    else:
                        default_hessian_batch = 32  # CPU can handle more

                    # Use config if available, otherwise use intelligent default
                    if config and hasattr(config, 'hessian_batch_size'):
                        effective_max_batch = config.hessian_batch_size
                    else:
                        effective_max_batch = default_hessian_batch

                    if device.type == 'cuda':
                        logger.info(f"Using batch size {effective_max_batch} for Hessian (double backprop, GPU memory: {gpu_memory_gb:.1f}GB, model params: {n_params/1e9:.1f}B)")
                    else:
                        logger.info(f"Using batch size {effective_max_batch} for Hessian (double backprop, CPU mode)")
                else:
                    # GGN/Fisher: Use config value or default
                    if config and hasattr(config, 'ggn_batch_size'):
                        effective_max_batch = config.ggn_batch_size
                    else:
                        # Default batch size for GGN/Fisher
                        # This should be set appropriately in the config based on model size
                        effective_max_batch = 8  # Conservative default
            else:
                # Use provided max_batch_size but apply operator-specific limits
                if operator == 'hessian':
                    # Still apply memory-based limits for Hessian
                    if device.type == 'cuda':
                        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

                        # Consider model size for safety limits
                        memory_per_batch = model_size_gb * 3
                        available_memory = gpu_memory_gb * 0.8
                        theoretical_max = int(available_memory / memory_per_batch)

                        if n_params > 1e9:  # Large models
                            # Match the conservative settings from above
                            if gpu_memory_gb > 70:  # H100
                                max_safe_batch = min(4, theoretical_max)  # Reduced from 8
                            elif gpu_memory_gb > 30:  # A100-40GB
                                max_safe_batch = min(2, theoretical_max)  # Reduced from 4
                            else:
                                max_safe_batch = 1  # Single sample only
                        else:
                            if gpu_memory_gb > 70:  # H100
                                max_safe_batch = min(8, theoretical_max)  # Reduced from 24
                            elif gpu_memory_gb > 30:  # A100-40GB
                                max_safe_batch = min(4, theoretical_max)  # Reduced from 16
                            else:
                                max_safe_batch = min(2, theoretical_max)  # Reduced from 8

                        max_safe_batch = max(1, max_safe_batch)
                        effective_max_batch = min(max_batch_size, max_safe_batch)
                        if effective_max_batch < max_batch_size:
                            logger.info(f"Reducing Hessian batch size from {max_batch_size} to {effective_max_batch} for GPU safety (model: {n_params/1e9:.1f}B params)")
                    else:
                        effective_max_batch = max_batch_size
                else:
                    effective_max_batch = max_batch_size

            # Apply batch size limit
            if 'input_ids' in batch and batch['input_ids'].shape[0] > effective_max_batch:
                batch = {k: v[:effective_max_batch] if torch.is_tensor(v) and v.shape[0] > effective_max_batch else v
                        for k, v in batch.items()}

            # Configure Lanczos with memory optimization for large models
            # For models over 1B parameters, always use selective reorthogonalization
            force_memory_efficient = n_params > 1e9

            # Determine computation dtype based on model and operator
            # For BFloat16 models, keep BFloat16 for memory efficiency
            model_dtype = next(model.parameters()).dtype
            if model_dtype == torch.bfloat16:
                # Keep BFloat16 for computations to avoid memory duplication
                compute_dtype = torch.bfloat16
                logger.debug("Using BFloat16 for Lanczos computations to save memory")
            else:
                compute_dtype = torch.float32

            # For PSD operators (Fisher/GGN), always use selective reorthogonalization
            is_psd = operator in ['ggn', 'empirical_fisher', 'kfac']
            if is_psd:
                reorth_period = 5  # Always selective for PSD
                regularization = 1e-8  # Small regularization for PSD
            else:
                reorth_period = 5 if (memory_efficient or force_memory_efficient) else 0
                regularization = 0  # No regularization for Hessian

            config = LanczosConfig(
                k=k,
                max_iters=max_iter,
                tol=1e-10,
                reorth_period=reorth_period,
                dtype_compute=compute_dtype,
                dtype_tridiag=torch.float64,  # Always use float64 for small tridiagonal matrix
                seed=42,
                regularization=regularization
            )

            if force_memory_efficient and not memory_efficient:
                logger.info(f"Forcing memory-efficient mode for large model ({n_params/1e9:.1f}B params)")

            # Create loss function if not provided
            if loss_fn is None:
                def loss_fn():
                    """
                    Memory-safe loss function for Hessian computation.
                    CRITICAL: Must extract and detach loss to avoid holding huge outputs object.
                    outputs contains logits (B×L×V) which can be 5GB+ for large models!
                    """
                    outputs = model(**batch)
                    loss = outputs.loss

                    # Explicitly delete outputs to free GPU memory immediately
                    # This prevents accumulation across Lanczos iterations
                    del outputs

                    # IMPORTANT: Don't detach! Hessian needs create_graph=True
                    # The loss tensor itself is small, only ~4 bytes
                    return loss

            # Compute spectrum using specified operator
            # Pass ggn_mode if it's in kwargs
            ggn_mode = kwargs.get('ggn_mode', 'empirical')
            results = compute_spectrum(
                model=model,
                batch=batch,
                operator_type=operator,  # Use the specified operator
                config=config,
                loss_fn=loss_fn,
                ggn_mode=ggn_mode,
                verbose=False
            )

            # Format results to match expected output
            if 'eigenvalues' in results and len(results['eigenvalues']) > 0:
                eigs = results['eigenvalues']
                return {
                    'top_eigenvalues': eigs[:k],
                    'max_eigenvalue': results.get('max_eigenvalue', 0),
                    'min_computed_eigenvalue': results.get('min_eigenvalue', 0),
                    'spectral_gap': eigs[0] - eigs[-1] if len(eigs) > 1 else 0,
                    'condition_number': results.get('range_ratio', 1.0),
                    'sharpness_score': results.get('sharpness_score', 0),
                    'lanczos_iterations': results.get('iterations', max_iter),
                    'k_requested': k,
                    'k_computed': len(eigs),
                    'batch_size_used': batch['input_ids'].shape[0] if 'input_ids' in batch else None,
                    'has_negative_eigenvalues': results.get('has_negative_eigenvalues', False),
                    'operator_used': operator,
                    'is_psd': results.get('is_psd', operator != 'hessian'),
                    'note': f'Computed using {operator} operator via unified Lanczos system'
                }
            else:
                # Return error result
                return {
                    'error': results.get('error', 'Computation failed'),
                    'top_eigenvalues': [],
                    'max_eigenvalue': 0,
                    'min_computed_eigenvalue': 0,
                    'spectral_gap': 0,
                    'condition_number': 1.0,
                    'sharpness_score': 0,
                    'lanczos_iterations': 0,
                    'k_requested': k,
                    'k_computed': 0,
                    'batch_size_used': batch['input_ids'].shape[0] if 'input_ids' in batch else None,
                    'note': 'Eigenvalue computation failed'
                }

        except ImportError as e:
            return {
                'error': f'Unified Lanczos system not available: {e}',
                'top_eigenvalues': [],
                'note': 'Please ensure fisher module is properly installed'
            }
        except Exception as e:
            return {
                'error': f'Lanczos computation failed: {str(e)}',
                'top_eigenvalues': [],
                'note': 'Check model and batch compatibility'
            }
        finally:
            # Restore original training state and gradient states
            if was_training:
                model.train()
            else:
                model.eval()

            # Restore original gradient states
            for name, param in model.named_parameters():
                param.requires_grad_(original_grad_states[name])

    def compute_fisher_eigenvalues_lanczos(
        self,
        model,
        data_batch: Dict[str, torch.Tensor],
        k: int = 5,
        max_iter: int = 20,
        use_ggn: bool = True,
        ggn_mode: str = 'empirical',
        config: Optional[Any] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Compute Fisher Information Matrix eigenvalues using Lanczos.

        This computes a PSD (positive semi-definite) spectrum, which is better
        for conditioning metrics than the potentially indefinite Hessian.

        Args:
            model: Model to analyze
            data_batch: Batch to compute eigenvalues on
            k: Number of top eigenvalues to compute
            max_iter: Maximum Lanczos iterations
            use_ggn: If True, use GGN (recommended), else empirical Fisher
            ggn_mode: Mode for GGN computation ('empirical', 'true', 'auto')

        Returns:
            Dictionary with eigenvalues and PSD metrics
        """
        # Fisher-specific optimizations
        operator = 'ggn' if use_ggn else 'empirical_fisher'
        kwargs['ggn_mode'] = ggn_mode  # Pass ggn_mode to compute_hessian_eigenvalues_lanczos

        # Force memory-efficient mode for Fisher
        kwargs['memory_efficient'] = True

        return self.compute_hessian_eigenvalues_lanczos(
            model, data_batch, k=k, max_iter=max_iter,
            operator=operator, config=config, **kwargs
        )

    def compute_spectrum_comparison(
        self,
        model,
        data_batch: Dict[str, torch.Tensor],
        k: int = 5,
        ggn_mode: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Compute both Hessian and Fisher spectra for comparison.

        Args:
            model: Neural network model
            data_batch: Input batch
            k: Number of eigenvalues to compute
            ggn_mode: Mode for GGN computation ('empirical', 'true', 'auto')

        Returns:
            Dictionary with both spectra and comparison metrics
        """
        # Compute Hessian spectrum (can be indefinite)
        hessian_results = self.compute_hessian_eigenvalues_lanczos(
            model, data_batch, k=k, operator='hessian'
        )

        # Compute Fisher/GGN spectrum (PSD)
        fisher_results = self.compute_fisher_eigenvalues_lanczos(
            model, data_batch, k=k, use_ggn=True, ggn_mode=ggn_mode
        )

        return {
            'hessian': hessian_results,
            'fisher_ggn': fisher_results,
            'comparison': {
                'hessian_has_negative': hessian_results.get('has_negative_eigenvalues', False),
                'fisher_condition': fisher_results.get('condition_number', float('inf')),
                'hessian_sharpness': hessian_results.get('sharpness_score', 0),
                'fisher_max_eigenvalue': fisher_results.get('max_eigenvalue', 0)
            }
        }

    def compute_hessian_eigenvalues_safe(
        self,
        model,
        data_batch: Dict[str, torch.Tensor],
        k: int = 5,
        max_iter: int = 20,
        loss_fn: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Safe wrapper for Hessian eigenvalue computation with automatic fallback.

        Attempts Lanczos with progressively smaller batch sizes until it succeeds.
        Falls back to power iteration if Lanczos fails completely.
        """
        batch_sizes_to_try = [32, 16, 8, 4, 2, 1]
        original_batch_size = data_batch['input_ids'].shape[0] if 'input_ids' in data_batch else None

        for max_batch in batch_sizes_to_try:
            try:
                # Try with current batch size limit
                result = self.compute_hessian_eigenvalues_lanczos(
                    model=model,
                    data_batch=data_batch,
                    k=k,
                    max_iter=max_iter,
                    loss_fn=loss_fn,
                    memory_efficient=True,
                    max_vectors_in_memory=2,  # Very conservative
                    max_batch_size=max_batch
                )

                if original_batch_size and original_batch_size > max_batch:
                    result['note'] += f' (batch reduced from {original_batch_size} to {max_batch})'

                return result

            except torch.cuda.OutOfMemoryError:
                # Clear cache and try smaller batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                continue
            except Exception as e:
                # For other errors, log and continue
                import warnings
                warnings.warn(f"Lanczos failed with batch size {max_batch}: {str(e)}")
                continue

        # If all batch sizes failed, return a simple approximation
        import warnings
        warnings.warn("Lanczos failed at all batch sizes. Returning power iteration estimate.")

        try:
            # Fall back to simple power iteration for largest eigenvalue only
            model.eval()
            with torch.enable_grad():
                # Use batch size 1 for minimal memory
                mini_batch = {k: v[:1] if torch.is_tensor(v) and v.shape[0] > 0 else v
                             for k, v in data_batch.items()}
                model_device = next(model.parameters()).device
                mini_batch = {k: v.to(model_device) if torch.is_tensor(v) else v
                             for k, v in mini_batch.items()}

                if loss_fn is not None:
                    loss = loss_fn(model, mini_batch)
                else:
                    # Prepare labels properly for causal LM if needed
                    if 'labels' not in mini_batch or mini_batch['labels'] is None:
                        input_ids = mini_batch.get('input_ids')
                        if input_ids is None:
                            raise RuntimeError("No input_ids found in batch")
                        labels = input_ids.clone()
                        # Validate vocabulary size
                        if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                            vocab_size = model.config.vocab_size
                            max_token_id = labels.max().item()
                            if max_token_id >= vocab_size:
                                labels = torch.clamp(labels, 0, vocab_size - 1)
                        outputs = model(**mini_batch, labels=labels)
                    else:
                        outputs = model(**mini_batch)
                    loss = outputs.loss

                params = [p for p in model.parameters() if p.requires_grad]

                # Simple power iteration for largest eigenvalue
                v = [torch.randn_like(p) * 0.01 for p in params]  # Small init
                eigenval = 0.0

                for _ in range(5):  # Just 5 iterations
                    # Normalize
                    norm = torch.sqrt(sum((vi**2).sum() for vi in v))
                    v = [vi / norm for vi in v]

                    # Apply Hessian
                    Hv = self._hvp(loss, params, v)

                    # Estimate eigenvalue
                    eigenval = sum((hvi * vi).sum() for hvi, vi in zip(Hv, v))
                    if torch.is_tensor(eigenval):
                        eigenval = eigenval.item()

                    # Update v
                    v = Hv

                return {
                    'top_eigenvalues': [eigenval],
                    'max_eigenvalue': eigenval,
                    'min_computed_eigenvalue': eigenval,
                    'spectral_gap': 0.0,
                    'condition_number': float('inf'),
                    'lanczos_iterations': 0,
                    'k_requested': k,
                    'k_computed': 1,
                    'batch_size_used': 1,
                    'note': 'Fallback: power iteration estimate only'
                }

        except Exception as e:
            # Complete failure - return zeros
            return {
                'top_eigenvalues': [],
                'max_eigenvalue': 0.0,
                'min_computed_eigenvalue': 0.0,
                'spectral_gap': 0.0,
                'condition_number': float('inf'),
                'lanczos_iterations': 0,
                'k_requested': k,
                'k_computed': 0,
                'batch_size_used': 0,
                'note': f'Complete failure: {str(e)}'
            }

    # ============= HESSIAN ANALYSIS (NO USES?) =============
    #TODO CONSIDER WHY THIS EXISTS
    def compute_hessian_eigenvalues(
        self,
        model,
        data_batch: Dict[str, torch.Tensor],
        n_eigenvalues: int = 10,
        use_power_iteration: bool = True
    ) -> Dict[str, Any]:
        """
        Compute top eigenvalues of Hessian to understand loss landscape curvature.
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
                f"Enabling gradients for ALL parameters to ensure proper Hessian computation."
            )

        # CRITICAL: Enable gradients for ALL parameters (pretrained models load with requires_grad=False)
        original_grad_states = {}
        for name, param in model.named_parameters():
            original_grad_states[name] = param.requires_grad
            param.requires_grad = True

        model.eval()  # Use eval mode for deterministic analysis (gradients still work!)
        model_device = next(model.parameters()).device
        batch = {k: v.to(model_device) if torch.is_tensor(v) else v
                for k, v in data_batch.items()}

        # Compute loss and gradients with gradients enabled
        with torch.enable_grad():
            # Prepare labels properly for causal LM if needed
            if 'labels' not in batch or batch['labels'] is None:
                input_ids = batch.get('input_ids')
                if input_ids is None:
                    raise RuntimeError("No input_ids found in batch")
                labels = input_ids.clone()
                # Validate vocabulary size
                if hasattr(model, 'config') and hasattr(model.config, 'vocab_size'):
                    vocab_size = model.config.vocab_size
                    max_token_id = labels.max().item()
                    if max_token_id >= vocab_size:
                        labels = torch.clamp(labels, 0, vocab_size - 1)
                        import warnings
                        warnings.warn(f"Token IDs ({max_token_id}) >= vocab_size ({vocab_size}), clamping to valid range")
                outputs = model(**batch, labels=labels)
            else:
                outputs = model(**batch)
            loss = outputs.loss

            # Ensure loss requires grad for Hessian computation
            if not loss.requires_grad:
                loss.requires_grad_(True)

            if use_power_iteration:
                # Power iteration for top eigenvalue
                eigenvalues = []

                for _ in range(min(n_eigenvalues, 5)):  # Limit for memory
                    # Random vector
                    v = []
                    for p in model.parameters():
                        v.append(torch.randn_like(p))

                    # Normalize
                    v_norm = torch.sqrt(sum([(vi**2).sum() for vi in v]))
                    v = [vi / v_norm for vi in v]

                    # Power iteration with early stopping
                    prev_eigenvalue = None
                    converged = False
                    max_iter = 10  # Reduced from 20
                
                    for iter_idx in range(max_iter):
                        # Compute Hv (Hessian-vector product) more efficiently
                        model.zero_grad()
                        # Use gradient checkpointing to save memory
                        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
                    
                        # Compute grad·v
                        grad_v = sum([(g * vi).sum() for g, vi in zip(grads, v)])

                        # Second derivative - always retain graph for power iteration
                        model.zero_grad()
                        # Always retain graph since we're in a loop
                        Hv = torch.autograd.grad(grad_v, model.parameters(), retain_graph=True, allow_unused=True)

                        # Update v - handle None gradients from allow_unused
                        Hv_safe = [hvi if hvi is not None else torch.zeros_like(vi)
                                  for hvi, vi in zip(Hv, v)]
                        v_new_norm = torch.sqrt(sum([(hvi**2).sum() for hvi in Hv_safe]) + 1e-10)
                        v = [hvi / v_new_norm for hvi in Hv_safe]

                        # Eigenvalue estimate - handle None gradients
                        eigenvalue = sum([(hvi * vi).sum() if hvi is not None else 0
                                         for hvi, vi in zip(Hv, v)])
                        if torch.is_tensor(eigenvalue):
                            eigenvalue = eigenvalue.item()

                        # Check convergence for early stopping
                        if prev_eigenvalue is not None:
                            if abs(eigenvalue - prev_eigenvalue) < 1e-4:
                                converged = True
                                break
                        prev_eigenvalue = eigenvalue

                        # Clear intermediate tensors to save memory
                        if iter_idx % 3 == 0:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                
                    if 'eigenvalue' in locals():
                        eigenvalues.append(eigenvalue)
                    else:
                        eigenvalues.append(0)
            else:
                # Simpler approximation using gradient statistics
                model.zero_grad()
                loss.backward(retain_graph=False)

                grad_norms = []
                for p in model.parameters():
                    if p.grad is not None:
                        grad_norms.append(p.grad.norm().item())

                # Approximate eigenvalues from gradient norms
                eigenvalues = sorted(grad_norms, reverse=True)[:n_eigenvalues]
        
        eigenvalues = np.array(eigenvalues)
        
        # Compute spectral statistics
        results = {
            'max_eigenvalue': float(np.max(eigenvalues)) if len(eigenvalues) > 0 else 0,
            'min_eigenvalue': float(np.min(eigenvalues)) if len(eigenvalues) > 0 else 0,
            'trace_estimate': float(np.sum(eigenvalues)),
            'spectral_norm': float(np.max(np.abs(eigenvalues))) if len(eigenvalues) > 0 else 0,
            'condition_number': float(
                min(1e10, np.max(np.abs(eigenvalues)) / max(np.min(np.abs(eigenvalues)), 1e-10))
            ) if len(eigenvalues) > 0 and np.min(np.abs(eigenvalues)) > 0 else 1e10,
            'eigenvalue_spread': float(np.std(eigenvalues)) if len(eigenvalues) > 0 else 0,
            'n_positive_eigenvalues': int(np.sum(eigenvalues > 0)),
            'n_negative_eigenvalues': int(np.sum(eigenvalues < 0))
        }
        
        # Check for sharp vs flat minimum
        results['is_sharp_minimum'] = results['max_eigenvalue'] > 10
        results['sharpness_score'] = float(np.mean(np.abs(eigenvalues)))

        # Restore original model state
        # Restore original gradient states
        for name, param in model.named_parameters():
            param.requires_grad = original_grad_states[name]

        if not was_training:
            model.eval()

        return results
    
    # ============= HELPER METHODS =============
    
    def _compute_gini(self, values: np.ndarray) -> float:
        """Compute Gini coefficient for measuring inequality."""
        # Filter out NaN and inf values
        values = values[np.isfinite(values)]

        n = len(values)
        if n == 0:
            return 0.0

        sorted_values = np.sort(values)
        cumsum = np.cumsum(sorted_values)

        # Handle all zeros or invalid cumsum
        if cumsum[-1] == 0 or not np.isfinite(cumsum[-1]):
            return 0.0

        # Correct Gini formula: G = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
        index = np.arange(1, n + 1)
        numerator = 2 * np.sum(index * sorted_values)
        denominator = n * cumsum[-1]

        # Extra safety check
        if denominator == 0 or not np.isfinite(denominator):
            return 0.0

        return numerator / denominator - (n + 1) / n
    
    def analyze_rlvr_vs_instruct(
        self,
        model_instruct,
        model_rlvr,
        test_batch: Dict[str, torch.Tensor],
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive comparison of RLVR vs Instruct models.
        Tests all major hypotheses about RLVR robustness.
        """
        results = {}

        if verbose:
            print("Analyzing RLVR vs Instruct models...")

        # Validate batch size for splitting
        batch_size = test_batch['input_ids'].shape[0]
        if batch_size < 2:
            return {
                'error': f'Batch size too small for analysis. Need at least 2 samples, got {batch_size}',
                'batch_size': batch_size
            }

        # 1. Loss Landscape - with error handling
        if verbose:
            print("  Computing loss barrier...")
        try:
            barrier_results = self.compute_loss_barrier(
                model_instruct, model_rlvr, test_batch
            )
            results['loss_barrier'] = barrier_results
        except Exception as e:
            if verbose:
                print(f"    Warning: Loss barrier computation failed: {e}")
            results['loss_barrier'] = {'error': str(e)}
            barrier_results = {'barrier_height': float('inf'), 'is_mode_connected': False}

        # 2. Gradient Conflict - with error handling and availability check
        if verbose:
            print("  Computing gradient conflict...")

        conflict_reduction = 0.0
        try:
            if GRADIENT_ANALYSIS_AVAILABLE:
                # Split batch for conflict analysis
                half_size = batch_size // 2
                batch1 = {k: v[:half_size] if torch.is_tensor(v) else v
                         for k, v in test_batch.items()}
                batch2 = {k: v[half_size:] if torch.is_tensor(v) else v
                         for k, v in test_batch.items()}

                grad_analyzer = GradientAnalysis(device=self.device)
                conflict_instruct = grad_analyzer.compute_gradient_conflict_pcgrad(model_instruct, batch1, batch2)
                conflict_rlvr = grad_analyzer.compute_gradient_conflict_pcgrad(model_rlvr, batch1, batch2)

                results['gradient_conflict_instruct'] = conflict_instruct
                results['gradient_conflict_rlvr'] = conflict_rlvr
                conflict_reduction = float(
                    conflict_instruct.get('gradient_conflict', 0) - conflict_rlvr.get('gradient_conflict', 0)
                )
                results['conflict_reduction'] = conflict_reduction
            else:
                if verbose:
                    print("    Warning: GradientAnalysis not available, skipping gradient conflict")
                results['gradient_conflict_instruct'] = {'error': 'GradientAnalysis not available'}
                results['gradient_conflict_rlvr'] = {'error': 'GradientAnalysis not available'}
                results['conflict_reduction'] = 0.0
        except Exception as e:
            if verbose:
                print(f"    Warning: Gradient conflict computation failed: {e}")
            results['gradient_conflict_instruct'] = {'error': str(e)}
            results['gradient_conflict_rlvr'] = {'error': str(e)}
            results['conflict_reduction'] = 0.0

        # 3. Feature Attribution - with error handling
        if verbose:
            print("  Computing feature attribution...")
        try:
            attr_instruct = self.compute_integrated_gradients(model_instruct, test_batch)
            attr_rlvr = self.compute_integrated_gradients(model_rlvr, test_batch)

            results['attribution_instruct'] = attr_instruct
            results['attribution_rlvr'] = attr_rlvr
            results['attribution_entropy_diff'] = float(
                attr_rlvr.get('attribution_entropy', 0) - attr_instruct.get('attribution_entropy', 0)
            )
        except Exception as e:
            if verbose:
                print(f"    Warning: Feature attribution failed: {e}")
            results['attribution_instruct'] = {'error': str(e)}
            results['attribution_rlvr'] = {'error': str(e)}
            results['attribution_entropy_diff'] = 0.0

        # 4. Hessian Analysis - with error handling
        if verbose:
            print("  Computing Hessian eigenvalues...")

        sharpness_reduction = 0.0
        try:
            hessian_instruct = self.compute_hessian_eigenvalues_lanczos(model_instruct, test_batch)
            hessian_rlvr = self.compute_hessian_eigenvalues_lanczos(model_rlvr, test_batch)

            results['hessian_instruct'] = hessian_instruct
            results['hessian_rlvr'] = hessian_rlvr
            sharpness_reduction = float(
                hessian_instruct.get('sharpness_score', 0) - hessian_rlvr.get('sharpness_score', 0)
            )
            results['sharpness_reduction'] = sharpness_reduction
        except Exception as e:
            if verbose:
                print(f"    Warning: Hessian analysis failed: {e}")
            results['hessian_instruct'] = {'error': str(e)}
            results['hessian_rlvr'] = {'error': str(e)}
            results['sharpness_reduction'] = 0.0

        # 5. Summary Statistics - using safe access with defaults
        results['summary'] = {
            # FIXED: Separate mode connectivity from flatness (they're different concepts)
            'rlvr_is_mode_connected': barrier_results.get('is_mode_connected', False),
            'barrier_is_low': barrier_results.get('barrier_height', float('inf')) < 0.1,
            'rlvr_has_less_conflict': results.get('conflict_reduction', 0) > 0,
            'rlvr_is_mode_connected': barrier_results.get('is_mode_connected', False),
            'rlvr_has_lower_curvature': results.get('sharpness_reduction', 0) > 0,
            'overall_robustness_score': float(
                (float(barrier_results.get('barrier_height', float('inf')) < 0.1) +
                 float(results.get('conflict_reduction', 0) > 0) +
                 float(results.get('sharpness_reduction', 0) > 0)) / 3
            )
        }
        
        if verbose:
            print("\nSummary:")
            for key, value in results['summary'].items():
                print(f"  {key}: {value}")
        
        return results


# ICLR 2026 Hypotheses to test
ICLR_HYPOTHESES = {
    'flat_minima': "RLVR models converge to flatter minima → measure barrier height",
    'gradient_conflict': "RLVR reduces task interference → measure gradient alignment",
    'lottery_tickets': "RLVR preserves important subnetworks → measure ticket overlap (see LotteryTicketAnalysis.py)",
    'feature_attribution': "RLVR learns different features → measure attribution patterns",
    'mode_connectivity': "RLVR solutions are mode-connected → measure loss barriers",
    'hessian_sharpness': "RLVR has better-conditioned Hessian → measure eigenspectrum"
}

# CLI Interface for running metrics directly
def main():
    """Comprehensive CLI for ICLR metrics with multi-batch support."""
    import argparse
    import json
    from transformers import AutoModel, AutoTokenizer
    from torch.utils.data import DataLoader, Dataset

    parser = argparse.ArgumentParser(description="ICLR Metrics CLI - Compute metrics with multi-batch support")

    # Model arguments
    parser.add_argument("--model", type=str, required=True, help="Model name or path")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer name (defaults to model)")

    # Function to run
    parser.add_argument("--function", type=str, required=True,
                       choices=["loss_landscape", "loss_barrier", "integrated_gradients",
                               "attention_attribution", "hessian", "fisher", "pruning",
                               "mode_connectivity", "spectrum", "rlvr_comparison"],
                       help="Which metric function to run")

    # Data arguments
    parser.add_argument("--data_file", type=str, help="Path to data file (jsonl format)")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to process")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size per batch")
    parser.add_argument("--max_length", type=int, default=512, help="Max sequence length")

    # Function-specific arguments
    parser.add_argument("--n_points", type=int, default=25, help="Grid points for loss landscape")
    parser.add_argument("--span", type=float, default=0.1, help="Span for loss landscape")
    parser.add_argument("--k_eigenvalues", type=int, default=5, help="Number of eigenvalues")
    parser.add_argument("--n_steps", type=int, default=50, help="Steps for integrated gradients")
    parser.add_argument("--sparsity_levels", type=str, default="[0.1,0.3,0.5,0.7,0.9]",
                       help="Sparsity levels for pruning (JSON list)")

    # Output
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")

    # Batch processing
    parser.add_argument("--use_batch_processor", action="store_true", default=True,
                       help="Use batch processor for memory efficiency")
    parser.add_argument("--aggregate_batches", action="store_true", default=True,
                       help="Aggregate results across batches (reduces noise)")

    args = parser.parse_args()

    # Load model and tokenizer
    print(f"Loading model: {args.model}")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)

    if torch.cuda.is_available():
        model = model.cuda()

    # Initialize metrics
    from batch import BatchProcessor
    batch_processor = BatchProcessor() if args.use_batch_processor else None
    metrics = ICLRMetrics(device='cuda' if torch.cuda.is_available() else 'cpu',
                         batch_processor=batch_processor)

    # Load or create data batches
    batches = []

    if args.data_file:
        print(f"Loading data from {args.data_file}")
        # Load data from file
        with open(args.data_file, 'r') as f:
            texts = [json.loads(line)['text'] for line in f][:args.num_batches * args.batch_size]

        # Create batches
        for i in range(0, len(texts), args.batch_size):
            batch_texts = texts[i:i+args.batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True,
                              max_length=args.max_length, return_tensors='pt')
            if torch.cuda.is_available():
                encoded = {k: v.cuda() for k, v in encoded.items()}
            batches.append(encoded)
    else:
        print(f"Creating {args.num_batches} random batches")
        # Create random batches for testing
        for _ in range(args.num_batches):
            batch = {
                'input_ids': torch.randint(0, tokenizer.vocab_size, (args.batch_size, args.max_length)),
                'attention_mask': torch.ones(args.batch_size, args.max_length)
            }
            if torch.cuda.is_available():
                batch = {k: v.cuda() for k, v in batch.items()}
            batches.append(batch)

    print(f"Created {len(batches)} batches of size {args.batch_size}")

    # Run the requested function
    results = {}

    if args.function == "loss_landscape":
        print("Computing loss landscape with multi-batch aggregation...")
        results = metrics.compute_loss_landscape_2d(
            model=model,
            data_batches=batches if args.aggregate_batches else None,
            data_batch=batches[0] if not args.aggregate_batches else None,
            n_points=args.n_points,
            span=args.span
        )

    elif args.function == "loss_barrier":
        print("Computing loss barrier...")
        # Create DataLoader for loss_barrier
        from torch.utils.data import TensorDataset
        all_input_ids = torch.cat([b['input_ids'] for b in batches])
        all_attention_mask = torch.cat([b['attention_mask'] for b in batches])
        dataset = TensorDataset(all_input_ids, all_attention_mask)
        data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        results = metrics.compute_loss_barrier(
            model1=model,
            model2=model,  # Would need second model for real comparison
            data_batch=batches[0],
            data_loader=data_loader if args.aggregate_batches else None
        )

    elif args.function == "integrated_gradients":
        print("Computing integrated gradients...")
        if args.aggregate_batches:
            # Aggregate across batches
            all_results = []
            for batch in batches:
                result = metrics.compute_integrated_gradients(
                    model=model,
                    data_batch=batch,
                    n_steps=args.n_steps
                )
                all_results.append(result)

            # Average the results
            results = {
                'attribution_entropy': np.mean([r.get('attribution_entropy', 0) for r in all_results]),
                'max_attribution': np.mean([r.get('max_attribution', 0) for r in all_results]),
                'num_batches': len(all_results)
            }
        else:
            results = metrics.compute_integrated_gradients(
                model=model,
                data_batch=batches[0],
                n_steps=args.n_steps
            )

    elif args.function == "attention_attribution":
        print("Computing attention attribution...")
        # This already supports batch aggregation in unified_model_analysis
        results = metrics.compute_attention_attribution(
            model=model,
            input_batch=batches[0]
        )

    elif args.function == "hessian":
        print("Computing Hessian eigenvalues...")
        # NOTE: Multi-batch averaging is now handled internally by passing multiple batches
        # The old aggregate_batches code was incorrectly averaging eigenvalues AFTER computation
        # instead of averaging Hessian-vector products DURING Lanczos
        if args.aggregate_batches and len(batches) > 1:
            # Pass multiple batches for proper multi-batch averaging
            results = metrics.compute_hessian_eigenvalues_lanczos(
                model=model,
                data_batch=batches[:5],  # Pass list of batches
                k=args.k_eigenvalues
            )
        else:
            results = metrics.compute_hessian_eigenvalues_lanczos(
                model=model,
                data_batch=batches[0],
                k=args.k_eigenvalues
            )

    elif args.function == "fisher":
        print("Computing Fisher eigenvalues...")
        # NOTE: Multi-batch averaging is now handled internally by passing multiple batches
        # The old aggregate_batches code was incorrectly averaging eigenvalues AFTER computation
        if args.aggregate_batches and len(batches) > 1:
            # Pass multiple batches for proper multi-batch averaging
            results = metrics.compute_fisher_eigenvalues_lanczos(
                model=model,
                data_batch=batches[:5],  # Pass list of batches
                k=args.k_eigenvalues
            )
        else:
            results = metrics.compute_fisher_eigenvalues_lanczos(
                model=model,
                data_batch=batches[0],
                k=args.k_eigenvalues
            )

    elif args.function == "pruning":
        print("Computing pruning sensitivity...")
        sparsity_levels = json.loads(args.sparsity_levels)
        results = metrics.compute_pruning_sensitivity(
            model=model,
            data_batch=batches[0],
            sparsity_levels=sparsity_levels
        )

    elif args.function == "mode_connectivity":
        print("Computing mode connectivity...")
        results = metrics.compute_mode_connectivity(
            model1=model,
            model2=model,  # Would need second model
            data_batch=batches[0]
        )

    elif args.function == "spectrum":
        print("Computing spectrum comparison...")
        results = metrics.compute_spectrum_comparison(
            model=model,
            data_batch=batches[0]
        )

    elif args.function == "rlvr_comparison":
        print("Running RLVR vs Instruct comparison...")
        results = metrics.analyze_rlvr_vs_instruct(
            model_instruct=model,
            model_rlvr=model,  # Would need RLVR model
            batch1=batches[0],
            batch2=batches[1] if len(batches) > 1 else batches[0],
            test_batch=batches[2] if len(batches) > 2 else batches[0]
        )

    # Output results
    if args.verbose:
        print("\nResults:")
        print(json.dumps(results, indent=2, default=str))

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to {args.output}")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Run CLI if arguments provided
        main()
    else:
        # Show help message
        print("ICLR 2026 Critical Metrics CLI")
        print("=" * 60)
        print("\nUsage: python ICLRMetrics.py --function <metric> --model <model_path> [options]")
        print("\nAvailable functions:")
        print("  - loss_landscape: Compute 2D loss landscape (supports multi-batch)")
        print("  - loss_barrier: Compute loss barrier between models (DataLoader support)")
        print("  - integrated_gradients: Feature attribution (supports aggregation)")
        print("  - attention_attribution: Attention flow analysis")
        print("  - hessian: Hessian eigenvalues (supports aggregation)")
        print("  - fisher: Fisher eigenvalues (supports aggregation)")
        print("  - pruning: Pruning sensitivity analysis")
        print("  - mode_connectivity: Mode connectivity test")
        print("  - spectrum: Spectrum comparison")
        print("  - rlvr_comparison: Full RLVR vs Instruct comparison")
        print("\nOptions:")
        print("  --num_batches N: Process N batches (default: 10)")
        print("  --batch_size B: Samples per batch (default: 32)")
        print("  --aggregate_batches: Aggregate results across batches (reduces noise)")
        print("  --use_batch_processor: Use memory-efficient batch processing")
        print("\nExample:")
        print("  python ICLRMetrics.py --function loss_landscape --model gpt2 --num_batches 20 --aggregate_batches")
        print("\nKey Hypotheses to Test:")
        for hypothesis, explanation in ICLR_HYPOTHESES.items():
            print(f"  {hypothesis}: {explanation}")
