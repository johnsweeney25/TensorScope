#!/usr/bin/env python3
"""
Mode Connectivity Utilities for ICLR 2026
==========================================
Helper functions for improved mode connectivity analysis including:
- Weight permutation alignment (Git Re-Basin style)
- Bezier curve interpolation for smoother paths
- Advanced connectivity metrics
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
import math


def compute_bezier_curve(
    t: float,
    control_points: List[torch.Tensor]
) -> torch.Tensor:
    """
    Compute point on Bezier curve at parameter t.

    Args:
        t: Parameter in [0, 1]
        control_points: List of control point tensors

    Returns:
        Interpolated tensor at parameter t
    """
    n = len(control_points) - 1
    result = torch.zeros_like(control_points[0])

    for i, point in enumerate(control_points):
        # Bernstein polynomial coefficient
        coeff = math.comb(n, i) * (t ** i) * ((1 - t) ** (n - i))
        result += coeff * point

    return result


def compute_bezier_path(
    state1: Dict[str, torch.Tensor],
    state2: Dict[str, torch.Tensor],
    alpha: float,
    n_control_points: int = 3
) -> Dict[str, torch.Tensor]:
    """
    Compute model state along a Bezier curve path.

    Args:
        state1: Starting model state dict
        state2: Ending model state dict
        alpha: Parameter in [0, 1] along the path
        n_control_points: Number of control points (3 = quadratic Bezier)

    Returns:
        Interpolated state dict at alpha
    """
    if n_control_points < 2:
        raise ValueError("Need at least 2 control points for Bezier curve")

    interpolated = {}

    for key in state1.keys():
        if key not in state2:
            continue

        t1 = state1[key]
        t2 = state2[key]

        # Only interpolate floating point tensors
        if not t1.is_floating_point():
            interpolated[key] = t1
            continue

        if n_control_points == 2:
            # Linear interpolation (degenerate case)
            interpolated[key] = (1 - alpha) * t1 + alpha * t2
        elif n_control_points == 3:
            # Quadratic Bezier with midpoint
            midpoint = 0.5 * (t1 + t2)
            # Optionally add noise to midpoint to escape saddle
            # midpoint += 0.01 * torch.randn_like(midpoint)
            control_points = [t1, midpoint, t2]
            interpolated[key] = compute_bezier_curve(alpha, control_points)
        else:
            # Higher order Bezier
            control_points = [t1]
            for i in range(1, n_control_points - 1):
                t = i / (n_control_points - 1)
                cp = (1 - t) * t1 + t * t2
                # Add small perturbation to intermediate control points
                cp += 0.01 * torch.randn_like(cp)
                control_points.append(cp)
            control_points.append(t2)
            interpolated[key] = compute_bezier_curve(alpha, control_points)

    return interpolated


def align_neural_networks_permutation(
    state1: Dict[str, torch.Tensor],
    state2: Dict[str, torch.Tensor],
    model_type: str = 'mlp'
) -> Dict[str, torch.Tensor]:
    """
    Align two neural networks by finding optimal permutation (simplified version).
    Based on "Git Re-Basin" (Ainsworth et al., 2023).

    This is a placeholder - full implementation would include:
    - Weight matching via Hungarian algorithm
    - Activation matching for better alignment
    - Support for different architectures (CNN, Transformer)

    Args:
        state1: Reference model state dict
        state2: Model to align (will be permuted)
        model_type: Type of model ('mlp', 'cnn', 'transformer')

    Returns:
        Aligned version of state2
    """
    warnings.warn("Using placeholder weight alignment. Full Git Re-Basin implementation needed for production.")

    # For now, just return state2 unchanged
    # Full implementation would:
    # 1. Identify permutation symmetries in architecture
    # 2. Compute correlation/similarity matrices
    # 3. Solve assignment problem with Hungarian algorithm
    # 4. Apply optimal permutations

    return state2.copy()


def compute_path_length(
    losses: np.ndarray,
    alphas: np.ndarray
) -> float:
    """
    Compute the length of the loss path (total variation).

    Args:
        losses: Array of loss values along path
        alphas: Array of alpha values (path parameters)

    Returns:
        Path length metric
    """
    if len(losses) < 2:
        return 0.0

    # Compute distances in (alpha, loss) space
    path_length = 0.0
    for i in range(1, len(losses)):
        dalpha = alphas[i] - alphas[i-1]
        dloss = losses[i] - losses[i-1]
        # Euclidean distance in normalized space
        path_length += np.sqrt(dalpha**2 + (dloss/np.mean(losses))**2)

    return float(path_length)


def compute_barrier_sharpness(
    losses: np.ndarray,
    barrier_idx: int
) -> float:
    """
    Compute sharpness of barrier using second-order finite differences.

    Args:
        losses: Array of losses along path
        barrier_idx: Index of maximum (barrier) point

    Returns:
        Sharpness metric (second derivative approximation)
    """
    if barrier_idx <= 0 or barrier_idx >= len(losses) - 1:
        return 0.0

    # Second-order finite difference
    sharpness = losses[barrier_idx] - 0.5 * (losses[barrier_idx-1] + losses[barrier_idx+1])

    # Normalize by local scale
    local_scale = 0.5 * (abs(losses[barrier_idx-1]) + abs(losses[barrier_idx+1])) + 1e-10

    return float(sharpness / local_scale)


def analyze_connectivity_spectrum(
    models: List[torch.nn.Module],
    compute_barrier_fn: Any,
    data_batch: Dict[str, torch.Tensor]
) -> Dict[str, Any]:
    """
    Analyze the full connectivity spectrum of a set of models.

    Args:
        models: List of models to analyze
        compute_barrier_fn: Function to compute barrier between two models
        data_batch: Data to evaluate on

    Returns:
        Dictionary with connectivity graph metrics
    """
    n = len(models)
    if n < 2:
        return {'error': 'Need at least 2 models'}

    # Compute all pairwise barriers
    barrier_matrix = np.full((n, n), np.nan)

    for i in range(n):
        for j in range(i+1, n):
            result = compute_barrier_fn(models[i], models[j], data_batch)
            if 'error' not in result:
                barrier = result.get('barrier_height', np.nan)
                barrier_matrix[i, j] = barrier_matrix[j, i] = barrier

    # Analyze connectivity graph
    finite_barriers = barrier_matrix[np.isfinite(barrier_matrix)]

    if len(finite_barriers) == 0:
        return {'error': 'No valid barriers computed'}

    # Compute clustering coefficient (how connected are neighbors)
    threshold = np.percentile(finite_barriers, 25)  # Connected if in bottom quartile
    adjacency = (barrier_matrix <= threshold).astype(float)
    np.fill_diagonal(adjacency, 0)

    # Simple clustering coefficient
    clustering = 0.0
    for i in range(n):
        neighbors = np.where(adjacency[i] > 0)[0]
        if len(neighbors) >= 2:
            # Check connections between neighbors
            neighbor_connections = 0
            possible_connections = 0
            for j1 in range(len(neighbors)):
                for j2 in range(j1+1, len(neighbors)):
                    possible_connections += 1
                    if adjacency[neighbors[j1], neighbors[j2]] > 0:
                        neighbor_connections += 1
            if possible_connections > 0:
                clustering += neighbor_connections / possible_connections

    clustering /= n

    return {
        'mean_barrier': float(np.mean(finite_barriers)),
        'median_barrier': float(np.median(finite_barriers)),
        'barrier_spread': float(np.std(finite_barriers)),
        'clustering_coefficient': float(clustering),
        'connectivity_threshold': float(threshold),
        'n_models': n,
        'n_pairs_computed': len(finite_barriers) // 2
    }


def estimate_loss_curvature(
    model: torch.nn.Module,
    data_batch: Dict[str, torch.Tensor],
    epsilon: float = 1e-3
) -> float:
    """
    Estimate local loss curvature using finite differences.
    Cheaper alternative to Hessian eigenvalues.

    Args:
        model: Model to analyze
        data_batch: Data to evaluate on
        epsilon: Perturbation size

    Returns:
        Curvature estimate
    """
    model.eval()
    device = next(model.parameters()).device

    # Get base loss
    with torch.no_grad():
        outputs = model(**data_batch)
        base_loss = outputs.loss.item()

    # Sample random directions and measure curvature
    curvatures = []

    for _ in range(5):  # Sample a few directions
        # Random direction
        direction = []
        for p in model.parameters():
            direction.append(torch.randn_like(p))

        # Normalize direction
        norm = torch.sqrt(sum((d**2).sum() for d in direction))
        direction = [d / norm for d in direction]

        # Measure loss at +/- epsilon
        with torch.no_grad():
            # +epsilon
            for p, d in zip(model.parameters(), direction):
                p.data.add_(epsilon * d)

            outputs = model(**data_batch)
            loss_plus = outputs.loss.item()

            # -2*epsilon (to get to -epsilon from +epsilon)
            for p, d in zip(model.parameters(), direction):
                p.data.add_(-2 * epsilon * d)

            outputs = model(**data_batch)
            loss_minus = outputs.loss.item()

            # Restore
            for p, d in zip(model.parameters(), direction):
                p.data.add_(epsilon * d)

        # Second derivative approximation
        curvature = (loss_plus - 2*base_loss + loss_minus) / (epsilon**2)
        curvatures.append(curvature)

    return float(np.mean(curvatures))


# Configuration for different scenarios
CONNECTIVITY_CONFIGS = {
    'quick': {
        'n_points': 11,
        'eps_rel': 0.05,
        'eps_abs': 1e-3,
    },
    'standard': {
        'n_points': 21,
        'eps_rel': 0.02,
        'eps_abs': 1e-4,
    },
    'thorough': {
        'n_points': 51,
        'eps_rel': 0.01,
        'eps_abs': 1e-5,
    }
}


if __name__ == "__main__":
    print("Mode Connectivity Utilities")
    print("=" * 50)
    print("\nKey functions:")
    print("- compute_bezier_path: Smoother interpolation paths")
    print("- align_neural_networks_permutation: Weight matching (placeholder)")
    print("- analyze_connectivity_spectrum: Multi-model analysis")
    print("- estimate_loss_curvature: Quick sharpness estimate")
    print("\nUse CONNECTIVITY_CONFIGS for preset analysis configurations")