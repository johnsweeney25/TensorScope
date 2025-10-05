#!/usr/bin/env python3
"""
Numerical Safety Utilities for ICML

Critical safety functions to prevent numerical instabilities,
division by zero, NaN/Inf propagation, and other issues that
will cause paper rejection.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
import logging

logger = logging.getLogger(__name__)

# Dtype-aware epsilon values
EPSILON_MAP = {
    torch.float16: 1e-4,
    torch.bfloat16: 1e-4,
    torch.float32: 1e-6,
    torch.float64: 1e-8,
}

# Context-specific epsilon values
CONTEXT_EPSILON = {
    'default': 1e-6,
    'small': 1e-8,
    'tiny': 1e-10,
    'large': 1e-4,
    'half': 1e-4,
    'division': 1e-8,
    'log': 1e-10,
    'sqrt': 1e-10,
    'norm': 1e-8,
    'std': 1e-8,
    'eigenvalue': 1e-10,
}


def get_epsilon(
    dtype: Optional[torch.dtype] = None,
    context: str = 'default',
    device: Optional[Union[str, torch.device]] = None
) -> float:
    """
    Get appropriate epsilon for dtype and context.

    Args:
        dtype: Tensor dtype (uses dtype-specific epsilon)
        context: Context for epsilon selection
        device: Device (may affect epsilon in future)

    Returns:
        Safe epsilon value
    """
    # Dtype-specific epsilon takes precedence
    if dtype is not None and dtype in EPSILON_MAP:
        return EPSILON_MAP[dtype]

    # Otherwise use context-specific epsilon
    return CONTEXT_EPSILON.get(context, CONTEXT_EPSILON['default'])


def safe_divide(
    numerator: torch.Tensor,
    denominator: torch.Tensor,
    eps: Optional[float] = None,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    Safely divide tensors with automatic epsilon and NaN handling.

    Args:
        numerator: Numerator tensor
        denominator: Denominator tensor
        eps: Epsilon value (auto-determined if None)
        fill_value: Value to use when denominator is zero

    Returns:
        Safe division result
    """
    if eps is None:
        eps = get_epsilon(denominator.dtype, context='division')

    # Handle zero denominator
    is_zero = torch.abs(denominator) < eps
    safe_denom = torch.where(is_zero, torch.ones_like(denominator), denominator + eps)

    result = numerator / safe_denom

    # Fill zero denominators with fill_value
    result = torch.where(is_zero, torch.full_like(result, fill_value), result)

    # Check for NaN/Inf
    if not torch.isfinite(result).all():
        logger.warning(f"safe_divide produced NaN/Inf values, replacing with {fill_value}")
        result = torch.where(torch.isfinite(result), result, torch.full_like(result, fill_value))

    return result


def safe_log(
    x: torch.Tensor,
    eps: Optional[float] = None
) -> torch.Tensor:
    """
    Safe logarithm that clamps input to avoid NaN.

    Args:
        x: Input tensor
        eps: Minimum value before log (auto-determined if None)

    Returns:
        Safe log result
    """
    if eps is None:
        eps = get_epsilon(x.dtype, context='log')

    return torch.log(torch.clamp(x, min=eps))


def safe_sqrt(
    x: torch.Tensor,
    eps: Optional[float] = None
) -> torch.Tensor:
    """
    Safe square root that clamps input to avoid NaN.

    Args:
        x: Input tensor
        eps: Minimum value before sqrt (auto-determined if None)

    Returns:
        Safe sqrt result
    """
    if eps is None:
        eps = get_epsilon(x.dtype, context='sqrt')

    return torch.sqrt(torch.clamp(x, min=eps))


def safe_norm(
    x: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    keepdim: bool = False,
    eps: Optional[float] = None
) -> torch.Tensor:
    """
    Safe norm computation that avoids zero norms.

    Args:
        x: Input tensor
        dim: Dimension(s) to compute norm over
        keepdim: Keep reduced dimensions
        eps: Minimum norm value (auto-determined if None)

    Returns:
        Safe norm result
    """
    if eps is None:
        eps = get_epsilon(x.dtype, context='norm')

    norm = torch.norm(x, dim=dim, keepdim=keepdim)
    return torch.clamp(norm, min=eps)


def safe_std(
    x: torch.Tensor,
    dim: Optional[Union[int, Tuple[int, ...]]] = None,
    unbiased: bool = True,
    keepdim: bool = False,
    eps: Optional[float] = None
) -> torch.Tensor:
    """
    Safe standard deviation that avoids zero.

    Args:
        x: Input tensor
        dim: Dimension(s) to compute std over
        unbiased: Use unbiased estimator
        keepdim: Keep reduced dimensions
        eps: Minimum std value (auto-determined if None)

    Returns:
        Safe std result
    """
    if eps is None:
        eps = get_epsilon(x.dtype, context='std')

    # Check if we have enough samples
    if dim is not None:
        sample_size = x.shape[dim] if isinstance(dim, int) else min(x.shape[d] for d in dim)
        if unbiased and sample_size <= 1:
            logger.warning(f"Computing std with sample size {sample_size}, returning {eps}")
            shape = list(x.shape)
            if not keepdim:
                if isinstance(dim, int):
                    shape.pop(dim)
                else:
                    for d in sorted(dim, reverse=True):
                        shape.pop(d)
            return torch.full(shape, eps, device=x.device, dtype=x.dtype)

    std = torch.std(x, dim=dim, unbiased=unbiased, keepdim=keepdim)
    return torch.clamp(std, min=eps)


def safe_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: Optional[float] = None
) -> torch.Tensor:
    """
    Safe normalization that handles zero vectors.

    Args:
        x: Input tensor
        dim: Dimension to normalize over
        eps: Minimum norm for normalization (auto-determined if None)

    Returns:
        Safely normalized tensor
    """
    if eps is None:
        eps = get_epsilon(x.dtype, context='norm')

    norm = safe_norm(x, dim=dim, keepdim=True, eps=eps)
    return x / norm


def check_finite(
    tensor: torch.Tensor,
    name: str = "tensor",
    replace: bool = False,
    fill_value: float = 0.0
) -> torch.Tensor:
    """
    Check for NaN/Inf values and optionally replace them.

    Args:
        tensor: Tensor to check
        name: Name for error messages
        replace: Replace NaN/Inf instead of raising error
        fill_value: Value to use for replacement

    Returns:
        Original or cleaned tensor

    Raises:
        ValueError: If tensor contains NaN/Inf and replace=False
    """
    if torch.isfinite(tensor).all():
        return tensor

    nan_count = torch.isnan(tensor).sum().item()
    inf_count = torch.isinf(tensor).sum().item()

    msg = f"{name} contains {nan_count} NaN and {inf_count} Inf values"

    if replace:
        logger.warning(f"{msg}, replacing with {fill_value}")
        return torch.where(torch.isfinite(tensor), tensor, torch.full_like(tensor, fill_value))
    else:
        raise ValueError(msg)


def safe_entropy(
    probs: torch.Tensor,
    dim: int = -1,
    keepdim: bool = False,
    eps: Optional[float] = None
) -> torch.Tensor:
    """
    Safe entropy computation for probability distributions.

    Args:
        probs: Probability tensor (should sum to 1 along dim)
        dim: Dimension of probability distribution
        keepdim: Keep reduced dimension
        eps: Epsilon for numerical stability

    Returns:
        Safe entropy values
    """
    if eps is None:
        eps = get_epsilon(probs.dtype, context='log')

    # Ensure probabilities are valid
    probs = torch.clamp(probs, min=eps, max=1.0 - eps)

    # Normalize to ensure sum to 1
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp(min=eps)

    # Compute entropy
    log_probs = safe_log(probs, eps=eps)
    entropy = -(probs * log_probs).sum(dim=dim, keepdim=keepdim)

    return check_finite(entropy, name="entropy", replace=True, fill_value=0.0)


def safe_kl_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    dim: int = -1,
    eps: Optional[float] = None
) -> torch.Tensor:
    """
    Safe KL divergence computation.

    Args:
        p: Source distribution
        q: Target distribution
        dim: Distribution dimension
        eps: Epsilon for numerical stability

    Returns:
        Safe KL divergence
    """
    if eps is None:
        eps = get_epsilon(p.dtype, context='log')

    # Ensure valid probabilities
    p = torch.clamp(p, min=eps, max=1.0 - eps)
    q = torch.clamp(q, min=eps, max=1.0 - eps)

    # Normalize
    p = p / p.sum(dim=dim, keepdim=True).clamp(min=eps)
    q = q / q.sum(dim=dim, keepdim=True).clamp(min=eps)

    # Compute KL
    log_p = safe_log(p, eps=eps)
    log_q = safe_log(q, eps=eps)
    kl = (p * (log_p - log_q)).sum(dim=dim)

    return check_finite(kl, name="kl_divergence", replace=True, fill_value=0.0)


def safe_cosine_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    dim: int = -1,
    eps: Optional[float] = None
) -> torch.Tensor:
    """
    Safe cosine similarity that handles zero vectors.

    Args:
        x: First tensor
        y: Second tensor
        dim: Dimension to compute similarity over
        eps: Epsilon for normalization

    Returns:
        Safe cosine similarity
    """
    if eps is None:
        eps = get_epsilon(x.dtype, context='norm')

    x_norm = safe_norm(x, dim=dim, keepdim=True, eps=eps)
    y_norm = safe_norm(y, dim=dim, keepdim=True, eps=eps)

    dot_product = (x * y).sum(dim=dim)
    norm_product = (x_norm * y_norm).squeeze(dim)

    similarity = safe_divide(dot_product, norm_product, eps=eps, fill_value=0.0)

    # Clamp to valid range
    return torch.clamp(similarity, min=-1.0, max=1.0)


def safe_eigenvalues(
    matrix: torch.Tensor,
    eps: Optional[float] = None
) -> torch.Tensor:
    """
    Safe eigenvalue computation with condition number check.

    Args:
        matrix: Square matrix
        eps: Minimum eigenvalue magnitude

    Returns:
        Safe eigenvalues
    """
    if eps is None:
        eps = get_epsilon(matrix.dtype, context='eigenvalue')

    try:
        eigenvalues = torch.linalg.eigvals(matrix)

        # Handle complex eigenvalues
        if eigenvalues.is_complex():
            eigenvalues = eigenvalues.abs()

        # Ensure minimum magnitude
        eigenvalues = torch.where(
            torch.abs(eigenvalues) < eps,
            torch.full_like(eigenvalues, eps),
            eigenvalues
        )

        return eigenvalues

    except Exception as e:
        logger.error(f"Eigenvalue computation failed: {e}")
        # Return identity-like eigenvalues as fallback
        n = matrix.shape[-1]
        return torch.full((n,), 1.0, device=matrix.device, dtype=matrix.real.dtype)


def safe_condition_number(
    matrix: torch.Tensor,
    eps: Optional[float] = None
) -> float:
    """
    Safe condition number computation.

    Args:
        matrix: Square matrix
        eps: Minimum eigenvalue magnitude

    Returns:
        Condition number (clamped to reasonable range)
    """
    eigenvalues = safe_eigenvalues(matrix, eps=eps)

    max_eig = eigenvalues.abs().max()
    min_eig = eigenvalues.abs().min()

    if min_eig < 1e-10:
        return 1e10  # Return large but finite condition number

    condition = max_eig / min_eig

    # Clamp to reasonable range
    return min(condition.item(), 1e10)


# Convenience function for batch safety checks
def make_safe(
    tensor: torch.Tensor,
    name: str = "tensor",
    check_finite_vals: bool = True,
    check_zeros: bool = False,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None
) -> torch.Tensor:
    """
    Comprehensive safety check and correction.

    Args:
        tensor: Tensor to make safe
        name: Name for logging
        check_finite_vals: Check for NaN/Inf
        check_zeros: Ensure no exact zeros
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Safe tensor
    """
    if check_finite_vals:
        tensor = check_finite(tensor, name=name, replace=True)

    if check_zeros:
        eps = get_epsilon(tensor.dtype)
        tensor = torch.where(tensor == 0, torch.full_like(tensor, eps), tensor)

    if min_val is not None:
        tensor = torch.clamp(tensor, min=min_val)

    if max_val is not None:
        tensor = torch.clamp(tensor, max=max_val)

    return tensor