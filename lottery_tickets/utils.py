"""
Utility Functions for Lottery Tickets
======================================
Common utilities used across modules.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Any
import random
import numpy as np
import os


def ensure_deterministic_pruning(seed: int = 42) -> None:
    """
    Ensure deterministic behavior for reproducible results.

    Critical for ICML submission - ensures reproducibility.

    Args:
        seed: Random seed to use
    """
    # Python random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic algorithms
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For newer PyTorch versions
    if hasattr(torch, 'use_deterministic_algorithms'):
        try:
            torch.use_deterministic_algorithms(True)
        except RuntimeError:
            # Some operations don't have deterministic implementations
            pass

    # Set environment variables
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    print(f"Deterministic mode enabled with seed {seed}")


def apply_mask(
    model: nn.Module,
    mask: Dict[str, torch.Tensor],
    clone_weights: bool = False
) -> Optional[Dict[str, torch.Tensor]]:
    """
    Apply pruning mask to model parameters.

    Args:
        model: Model to apply mask to
        mask: Dictionary of masks per parameter
        clone_weights: If True, return original weights

    Returns:
        Original weights if clone_weights=True, else None
    """
    original_weights = {} if clone_weights else None

    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                if clone_weights:
                    original_weights[name] = param.data.clone()

                # Apply mask
                mask_tensor = mask[name].to(param.device)
                param.data.mul_(mask_tensor)

    return original_weights


def remove_mask(
    model: nn.Module,
    mask: Dict[str, torch.Tensor],
    original_weights: Optional[Dict[str, torch.Tensor]] = None,
    noise_scale: float = 0.01
) -> None:
    """
    Remove mask by restoring original weights or adding noise.

    Args:
        model: Model to restore
        mask: Mask that was applied
        original_weights: Optional original weights to restore
        noise_scale: Scale of random noise if no original weights
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                if original_weights and name in original_weights:
                    # Restore original weights
                    param.data.copy_(original_weights[name])
                else:
                    # Add small noise to pruned weights
                    mask_tensor = mask[name].to(param.device)
                    inv_mask = ~mask_tensor.bool()
                    noise = torch.randn_like(param) * noise_scale
                    param.data.add_(noise * inv_mask.float())


def compute_sparsity(
    mask: Union[Dict[str, torch.Tensor], torch.Tensor]
) -> float:
    """
    Compute overall sparsity from mask(s).

    Args:
        mask: Single mask or dictionary of masks

    Returns:
        Sparsity level (fraction of zeros)
    """
    if isinstance(mask, dict):
        total_params = 0
        total_zeros = 0

        for mask_tensor in mask.values():
            total_params += mask_tensor.numel()
            total_zeros += (mask_tensor == 0).sum().item()

        return total_zeros / max(total_params, 1)
    else:
        return (mask == 0).float().mean().item()


def compute_histogram_quantile(
    tensor: torch.Tensor,
    q: float,
    bins: int = 1000
) -> float:
    """
    Memory-efficient quantile computation using histogram.

    Args:
        tensor: Tensor to compute quantile for
        q: Quantile value (0-1)
        bins: Number of histogram bins

    Returns:
        Approximate quantile value
    """
    if tensor.numel() == 0:
        return 0.0

    min_val = tensor.min().item()
    max_val = tensor.max().item()

    if max_val <= min_val:
        return min_val

    # Compute histogram
    hist = torch.histc(tensor, bins=bins, min=min_val, max=max_val)

    # Find quantile from histogram
    total = tensor.numel()
    target_count = int(q * total)
    cumsum = 0
    bin_width = (max_val - min_val) / bins

    for i, count in enumerate(hist):
        cumsum += count.item()
        if cumsum >= target_count:
            # Return bin center
            return min_val + (i + 0.5) * bin_width

    return max_val


def get_model_size(model: nn.Module) -> Dict[str, Any]:
    """
    Get model size statistics.

    Args:
        model: Model to analyze

    Returns:
        Size statistics
    """
    total_params = 0
    trainable_params = 0
    param_details = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        if param.requires_grad:
            trainable_params += num_params

        param_details[name] = {
            'shape': list(param.shape),
            'numel': num_params,
            'dtype': str(param.dtype),
            'requires_grad': param.requires_grad
        }

    # Calculate memory usage
    bytes_per_param = 4  # Assuming float32
    total_memory_mb = (total_params * bytes_per_param) / (1024 * 1024)
    trainable_memory_mb = (trainable_params * bytes_per_param) / (1024 * 1024)

    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'total_memory_mb': total_memory_mb,
        'trainable_memory_mb': trainable_memory_mb,
        'param_details': param_details
    }


def create_model_wrapper(model: nn.Module) -> nn.Module:
    """
    Create a wrapper for models with different interfaces.

    Handles:
    - Transformer models (input_ids, labels)
    - Simple models (Linear, CNN)
    - Custom models
    """
    from types import SimpleNamespace

    class ModelWrapper(nn.Module):
        def __init__(self, base_model):
            super().__init__()
            self.model = base_model

        def forward(self, *args, **kwargs):
            # Handle batch dict with input_ids
            if not args and 'input_ids' in kwargs:
                input_tensor = kwargs['input_ids']
                labels = kwargs.get('labels')

                # Check if model expects dict or tensor
                try:
                    # Try transformer-style first
                    outputs = self.model(**kwargs)
                except (TypeError, RuntimeError):
                    # Fall back to tensor input
                    try:
                        outputs = self.model(input_tensor)
                    except Exception:
                        # Last resort - try without labels
                        outputs = self.model(input_tensor)

                # Ensure we have proper output format
                if torch.is_tensor(outputs):
                    # Convert tensor output to expected format
                    result = SimpleNamespace(logits=outputs)

                    # Add loss if we have labels
                    if labels is not None:
                        if outputs.dim() == 2 and labels.dim() == 1:
                            # Classification loss
                            loss_fn = nn.CrossEntropyLoss()
                            result.loss = loss_fn(outputs, labels)
                        else:
                            # Default loss
                            result.loss = outputs.mean()
                    else:
                        result.loss = outputs.mean()

                    return result
                elif hasattr(outputs, 'loss'):
                    return outputs
                else:
                    # Wrap in SimpleNamespace
                    if hasattr(outputs, 'logits'):
                        return outputs
                    else:
                        return SimpleNamespace(
                            logits=outputs,
                            loss=outputs.mean() if torch.is_tensor(outputs) else 0.0
                        )

            # Handle direct tensor input
            elif args:
                outputs = self.model(*args)
            else:
                # Extract tensor from kwargs
                x = kwargs.get('x', kwargs.get('input', kwargs.get('inputs')))
                if x is not None:
                    outputs = self.model(x)
                else:
                    outputs = self.model(**kwargs)

            # Ensure consistent output format
            if torch.is_tensor(outputs):
                return SimpleNamespace(logits=outputs, loss=outputs.mean())
            elif not hasattr(outputs, 'loss'):
                return SimpleNamespace(
                    logits=outputs if torch.is_tensor(outputs) else outputs.logits,
                    loss=outputs.mean() if torch.is_tensor(outputs) else 0.0
                )
            else:
                return outputs

        def named_parameters(self):
            return self.model.named_parameters()

        def parameters(self):
            return self.model.parameters()

        def train(self, mode=True):
            return self.model.train(mode)

        def eval(self):
            return self.model.eval()

        def to(self, *args, **kwargs):
            self.model = self.model.to(*args, **kwargs)
            return self

    return ModelWrapper(model)


def get_device(model: nn.Module) -> torch.device:
    """
    Get device of model parameters.

    Args:
        model: Model to check

    Returns:
        Device where model is located
    """
    devices = {p.device for p in model.parameters()}

    if len(devices) > 1:
        raise ValueError(f"Model spans multiple devices: {devices}")

    return next(iter(devices)) if devices else torch.device('cpu')


def move_to_device(
    data: Union[Dict, list, torch.Tensor],
    device: torch.device
) -> Union[Dict, list, torch.Tensor]:
    """
    Move data to device recursively.

    Args:
        data: Data to move (dict, list, or tensor)
        device: Target device

    Returns:
        Data on target device
    """
    if torch.is_tensor(data):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) if torch.is_tensor(v) else v
                for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) if torch.is_tensor(item) else item
                for item in data]
    else:
        return data