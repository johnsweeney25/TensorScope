"""
Gradient Management System for UnifiedModelAnalysis

This module provides intelligent gradient state management for metrics computation,
optimizing memory usage and preventing gradient-related errors.

Author: ICLR 2026 Team
Date: 2025
"""

import torch
import torch.nn as nn
from contextlib import contextmanager
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass
import logging
from enum import Enum

logger = logging.getLogger(__name__)

class GradientScope(Enum):
    """Defines the scope of gradient requirements for a metric."""
    NONE = "none"           # No gradients needed
    MODEL = "model"         # Only model parameters need gradients
    INPUTS = "inputs"       # Only inputs need gradients
    BOTH = "both"          # Both model and inputs need gradients

@dataclass
class GradientState:
    """Stores the gradient state of model parameters."""
    param_states: Dict[str, bool]
    training_mode: bool
    gradient_enabled: bool
    device: torch.device  # Track the device state

    def __repr__(self):
        enabled_count = sum(1 for v in self.param_states.values() if v)
        return (f"GradientState(params={enabled_count}/{len(self.param_states)} enabled, "
                f"training={self.training_mode}, gradients={self.gradient_enabled}, device={self.device})")

class GradientComputationManager:
    """
    Manages gradient computation state for models based on metric requirements.

    NOTE: This is different from GradientMemoryManager in fisher/core/ which STORES gradients.
    This class controls whether gradients are COMPUTED at all.

    This class provides:
    - Context managers for temporary gradient state changes
    - Memory-efficient gradient management by disabling computation
    - Proper state restoration after metric computation
    - Support for different gradient scopes
    """

    def __init__(self, enable_logging: bool = True):
        """
        Initialize the GradientManager.

        Args:
            enable_logging: Whether to log gradient state changes
        """
        self.enable_logging = enable_logging
        self._state_stack: List[GradientState] = []
        self._gradient_cache: Dict[int, bool] = {}  # Cache gradient requirements by model id

    def get_current_state(self, model: nn.Module) -> GradientState:
        """
        Get the current gradient state of a model.

        Args:
            model: The PyTorch model

        Returns:
            GradientState containing parameter gradient flags, training mode, and device
        """
        param_states = {}
        for name, param in model.named_parameters():
            param_states[name] = param.requires_grad

        # Get the device of the model
        device = next(model.parameters()).device if sum(1 for _ in model.parameters()) > 0 else torch.device('cpu')

        return GradientState(
            param_states=param_states,
            training_mode=model.training,
            gradient_enabled=torch.is_grad_enabled(),
            device=device
        )

    def restore_state(self, model: nn.Module, state: GradientState) -> nn.Module:
        """
        Restore a model to a previous gradient state.

        Args:
            model: The PyTorch model
            state: The state to restore

        Returns:
            The model with restored state
        """
        # Don't move device - keep model where it is to preserve gradient graph
        # model = model.to(state.device)  # Commented out - moving breaks gradients

        # Restore parameter gradient states
        for name, param in model.named_parameters():
            if name in state.param_states:
                param.requires_grad_(state.param_states[name])

        # Restore training mode
        if state.training_mode:
            model.train()
        else:
            model.eval()

        if self.enable_logging:
            logger.debug(f"Restored gradient state: {state}")

        return model

    def disable_gradients_recursive(self, model: nn.Module):
        """
        Recursively disable gradients for all parameters in a model.

        This is more reliable than the wrapper approach in the proposal.

        Args:
            model: The PyTorch model
        """
        for param in model.parameters():
            param.requires_grad_(False)

        if self.enable_logging:
            logger.debug(f"Disabled gradients for {sum(1 for _ in model.parameters())} parameters")

    def enable_gradients_recursive(self, model: nn.Module):
        """
        Recursively enable gradients for all parameters in a model.

        Args:
            model: The PyTorch model
        """
        for param in model.parameters():
            param.requires_grad_(True)

        if self.enable_logging:
            logger.debug(f"Enabled gradients for {sum(1 for _ in model.parameters())} parameters")

    @contextmanager
    def gradient_context(self,
                        model: nn.Module,
                        requires_grad: bool,
                        gradient_scope: GradientScope = GradientScope.BOTH,
                        eval_mode: bool = True):  # FIXED: Default to eval mode for deterministic gradients
        """
        Context manager for temporary gradient state and device changes.

        This properly handles:
        - Saving and restoring gradient states
        - Device migration (CPU for non-gradient, GPU for gradient)
        - Training vs eval mode
        - Nested contexts
        - Error recovery

        Args:
            model: The PyTorch model
            requires_grad: Whether gradients are needed
            gradient_scope: The scope of gradient requirements
            eval_mode: If True, use eval mode even when gradients are needed
                      (for consistent dropout/batch norm behavior)

        Yields:
            The model with appropriate gradient configuration and device placement
        """
        # Save current state including device
        original_state = self.get_current_state(model)
        self._state_stack.append(original_state)

        try:
            if not requires_grad or gradient_scope == GradientScope.NONE:
                # Keep model on original device to preserve gradient computation ability
                # Moving between devices breaks the gradient graph
                # Just disable gradients without moving
                self.disable_gradients_recursive(model)
                model.eval()

                # Use torch.no_grad() context for additional safety
                with torch.no_grad():
                    yield model
            else:
                # Keep model on its current device for gradient computation
                # Configure gradients based on scope
                if gradient_scope in [GradientScope.MODEL, GradientScope.BOTH]:
                    self.enable_gradients_recursive(model)
                else:
                    # For INPUTS only, disable model gradients
                    self.disable_gradients_recursive(model)

                # CRITICAL FIX: Always use eval mode for deterministic gradient computation
                # Gradients work perfectly fine in eval mode!
                model.eval()

                # Enable gradient computation
                with torch.enable_grad():
                    yield model

        finally:
            # Always restore original state and device, even if an error occurred
            if self._state_stack:
                state_to_restore = self._state_stack.pop()
                model = self.restore_state(model, state_to_restore)

            # Clear any cached computations
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def prepare_batch_for_gradient_computation(self,
                                              batch: Dict[str, torch.Tensor],
                                              requires_grad: bool,
                                              gradient_scope: GradientScope) -> Dict[str, torch.Tensor]:
        """
        Prepare a batch for gradient computation based on requirements.

        Args:
            batch: Input batch dictionary
            requires_grad: Whether gradients are needed
            gradient_scope: The scope of gradient requirements

        Returns:
            Batch with appropriate gradient settings
        """
        if not requires_grad or gradient_scope in [GradientScope.NONE, GradientScope.MODEL]:
            # Detach all inputs to prevent gradient computation
            return {k: v.detach() if torch.is_tensor(v) else v
                   for k, v in batch.items()}
        else:
            # Keep gradients for inputs
            return batch

    def check_memory_available(self, required_gb: float = 1.0) -> bool:
        """
        Check if sufficient GPU memory is available.

        Args:
            required_gb: Required memory in GB

        Returns:
            True if sufficient memory is available
        """
        if not torch.cuda.is_available():
            return True  # CPU always has system memory available

        # Get current memory usage
        allocated = torch.cuda.memory_allocated() / 1e9
        cached = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        available = total - allocated

        if self.enable_logging:
            logger.debug(f"GPU Memory: {allocated:.2f}GB allocated, {available:.2f}GB available, "
                        f"{required_gb:.2f}GB required")

        return available >= required_gb

    def optimize_for_memory(self, model: nn.Module, aggressive: bool = False):
        """
        Optimize model for memory efficiency.

        Args:
            model: The PyTorch model
            aggressive: If True, apply more aggressive optimizations
        """
        # Clear gradient buffers for parameters that don't need gradients
        for param in model.parameters():
            if not param.requires_grad and param.grad is not None:
                param.grad = None

        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()

        if self.enable_logging:
            logger.debug("Applied memory optimizations")

    def should_enable_gradients_for_batch(self,
                                         metric_names: List[str],
                                         metric_registry: Any) -> Tuple[bool, List[str], List[str]]:
        """
        Decide whether to enable gradients for a batch of metrics.

        Args:
            metric_names: List of metric names to compute
            metric_registry: Registry containing metric information

        Returns:
            Tuple of (should_enable, gradient_metrics, no_gradient_metrics)
        """
        gradient_metrics = []
        no_gradient_metrics = []

        for name in metric_names:
            if name in metric_registry.metrics:
                metric_info = metric_registry.metrics[name]
                if metric_info.get('requires_gradients', False):
                    gradient_metrics.append(name)
                else:
                    no_gradient_metrics.append(name)

        # Enable gradients if more than 30% of metrics need them
        # This is more conservative than the 50% in the proposal
        gradient_ratio = len(gradient_metrics) / len(metric_names) if metric_names else 0
        should_enable = gradient_ratio > 0.3

        if self.enable_logging:
            logger.info(f"Gradient decision: {len(gradient_metrics)}/{len(metric_names)} metrics "
                       f"need gradients ({gradient_ratio:.1%}), enabling={should_enable}")

        return should_enable, gradient_metrics, no_gradient_metrics


class MemoryOptimizedBatchProcessor:
    """
    Processes metrics in batches with memory optimization.

    This class handles the efficient computation of multiple metrics
    by grouping them by gradient requirements and processing them
    in an order that minimizes memory usage.
    """

    def __init__(self, gradient_manager: 'GradientComputationManager', memory_limit_gb: float = 8.0):
        """
        Initialize the batch processor.

        Args:
            gradient_manager: The gradient manager instance
            memory_limit_gb: Memory limit in GB
        """
        self.gradient_manager = gradient_manager
        self.memory_limit_gb = memory_limit_gb

    def process_metrics(self,
                       model: nn.Module,
                       batch: Dict[str, torch.Tensor],
                       metrics_to_compute: List[Tuple[str, Dict[str, Any]]],
                       compute_fn) -> Dict[str, Any]:
        """
        Process multiple metrics with optimal memory management.

        Args:
            model: The PyTorch model
            batch: Input batch
            metrics_to_compute: List of (metric_name, metric_info) tuples
            compute_fn: Function to compute individual metrics

        Returns:
            Dictionary of metric results
        """
        # Sort metrics by gradient requirements
        # Process gradient-free metrics first (more memory efficient)
        gradient_free = []
        gradient_required = []

        for name, info in metrics_to_compute:
            if info.get('requires_gradients', False):
                gradient_required.append((name, info))
            else:
                gradient_free.append((name, info))

        results = {}

        # Process gradient-free metrics in batch
        if gradient_free:
            logger.info(f"Processing {len(gradient_free)} gradient-free metrics")

            with self.gradient_manager.gradient_context(
                model,
                requires_grad=False,
                gradient_scope=GradientScope.NONE
            ):
                # Detach batch for gradient-free computation
                detached_batch = self.gradient_manager.prepare_batch_for_gradient_computation(
                    batch, False, GradientScope.NONE
                )

                for name, info in gradient_free:
                    try:
                        # Check memory before expensive metrics
                        if info.get('expensive', False):
                            if not self.gradient_manager.check_memory_available(2.0):
                                logger.warning(f"Skipping {name} due to memory constraints")
                                results[name] = {'error': 'Insufficient memory'}
                                continue

                        results[name] = compute_fn(name, model, detached_batch, info)

                    except Exception as e:
                        logger.error(f"Error computing {name}: {e}")
                        results[name] = {'error': str(e)}

                # Clean up after gradient-free batch
                self.gradient_manager.optimize_for_memory(model)

        # Process gradient-requiring metrics individually
        if gradient_required:
            logger.info(f"Processing {len(gradient_required)} gradient-requiring metrics")

            for name, info in gradient_required:
                try:
                    # Check memory before gradient computation
                    if not self.gradient_manager.check_memory_available(3.0):
                        logger.warning(f"Skipping {name} due to memory constraints")
                        results[name] = {'error': 'Insufficient memory for gradient computation'}
                        continue

                    # Clear gradients from previous metric
                    model.zero_grad()

                    # Determine gradient scope and eval mode
                    gradient_scope = GradientScope(info.get('gradient_scope', 'both'))
                    eval_mode = info.get('eval_mode', False)

                    with self.gradient_manager.gradient_context(
                        model,
                        requires_grad=True,
                        gradient_scope=gradient_scope,
                        eval_mode=eval_mode
                    ):
                        results[name] = compute_fn(name, model, batch, info)

                    # Clean up after each gradient metric
                    model.zero_grad()
                    self.gradient_manager.optimize_for_memory(model, aggressive=True)

                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"OOM error computing {name}: {e}")
                    results[name] = {'error': 'CUDA OOM'}
                    # Try to recover
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        model.zero_grad()

                except Exception as e:
                    logger.error(f"Error computing {name}: {e}")
                    results[name] = {'error': str(e)}

        return results


# Utility functions for metric classification
def classify_metric_gradient_requirements(metric_name: str) -> Tuple[bool, GradientScope, bool]:
    """
    Classify a metric's gradient requirements based on its name.

    This is a heuristic-based classification that can be overridden
    by explicit configuration.

    Args:
        metric_name: Name of the metric

    Returns:
        Tuple of (requires_gradients, gradient_scope, eval_mode)
    """
    metric_lower = metric_name.lower()

    # Gradient-requiring metrics
    if any(keyword in metric_lower for keyword in [
        'gradient', 'fisher', 'integrated_gradient', 'saliency',
        'tracin', 'influence', 'attribution', 'importance'
    ]):
        # Most gradient metrics need both model and input gradients
        # Fisher metrics might need eval mode for consistent results
        eval_mode = 'fisher' in metric_lower
        return True, GradientScope.BOTH, eval_mode

    # Gradient-free metrics
    if any(keyword in metric_lower for keyword in [
        'superposition', 'attention', 'entropy', 'dead_neuron',
        'sparsity', 'information', 'mode_connectivity',
        'representation', 'drift', 'similarity', 'manifold'
    ]):
        return False, GradientScope.NONE, False

    # Default: assume gradient-free for safety and efficiency
    logger.debug(f"Unknown metric '{metric_name}', defaulting to gradient-free")
    return False, GradientScope.NONE, False


# Compatibility alias for backward compatibility
# TODO: Remove this after updating all references
GradientManager = GradientComputationManager