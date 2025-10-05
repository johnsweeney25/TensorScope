"""
Fisher Compatibility Layer: Bridges group-level FisherCollector with per-parameter legacy code.

This module provides utilities to:
1. Expand group-level Fisher to per-parameter when needed
2. Map between old and new key schemas
3. Provide backward compatibility for existing code
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Union, Tuple
import re


class FisherCompatibilityMixin:
    """
    Mixin class that provides compatibility between FisherCollector's group-level
    Fisher and legacy per-parameter Fisher methods.
    """

    def expand_group_to_param_fisher(
        self,
        group_fisher: Dict[str, torch.Tensor],
        model: nn.Module
    ) -> Dict[str, torch.Tensor]:
        """
        Expand group-level Fisher back to per-parameter for backward compatibility.

        Args:
            group_fisher: Group-level Fisher from FisherCollector
            model: Model to get parameter shapes

        Returns:
            Per-parameter Fisher dictionary
        """
        param_fisher = {}

        for key, group_values in group_fisher.items():
            # Parse key: "task|param_name|group_type"
            parts = key.split('|')
            if len(parts) != 3:
                continue

            task, param_name, group_type = parts

            # Get the actual parameter
            param = None
            for name, p in model.named_parameters():
                if name == param_name:
                    param = p
                    break

            if param is None:
                continue

            # Expand based on group type
            if group_type == 'channel':
                # Expand channels to full parameter
                param_shape = param.shape
                if len(param_shape) >= 2:
                    # Broadcast group values to full shape
                    # group_values shape: (out_channels,)
                    # param shape: (out_channels, in_channels, ...)
                    expanded = group_values.view(-1, *([1] * (len(param_shape) - 1)))
                    expanded = expanded.expand(param_shape)
                    param_fisher[param_name] = expanded
                else:
                    param_fisher[param_name] = group_values

            elif group_type == 'head':
                # Expand heads to full parameter
                # This is more complex - need to know head structure
                param_fisher[param_name] = self._expand_heads_to_param(
                    group_values, param, param_name
                )

            elif group_type in ['row', 'token', 'bias', 'param']:
                # These are already close to per-parameter
                param_fisher[param_name] = group_values

            else:
                # Unknown group type, use as-is
                param_fisher[param_name] = group_values

        return param_fisher

    def _expand_heads_to_param(
        self,
        head_values: torch.Tensor,
        param: torch.Tensor,
        param_name: str
    ) -> torch.Tensor:
        """
        Expand per-head values to full parameter shape.

        Args:
            head_values: Per-head Fisher values
            param: Parameter tensor
            param_name: Parameter name

        Returns:
            Expanded Fisher values
        """
        num_heads = head_values.shape[0]
        param_shape = param.shape

        # Determine head dimension based on parameter name
        if any(x in param_name for x in ['q_proj', 'k_proj', 'v_proj']):
            # Output is heads * head_dim
            hidden_size = param_shape[0]
            head_dim = hidden_size // num_heads

            # Expand head values to (num_heads * head_dim, in_features)
            expanded = head_values.repeat_interleave(head_dim)
            expanded = expanded.view(-1, 1).expand(hidden_size, param_shape[1])
            return expanded

        elif 'o_proj' in param_name:
            # Input is heads * head_dim
            hidden_size = param_shape[1]
            head_dim = hidden_size // num_heads

            # Expand head values to (out_features, num_heads * head_dim)
            expanded = head_values.repeat_interleave(head_dim)
            expanded = expanded.view(1, -1).expand(param_shape[0], hidden_size)
            return expanded

        else:
            # Default: broadcast to parameter shape
            return head_values.view(-1, 1).expand(param_shape)

    def get_param_fisher_from_group(
        self,
        task: str,
        param_name: str,
        model: nn.Module,
        bias_corrected: bool = True
    ) -> Optional[torch.Tensor]:
        """
        Get per-parameter Fisher from group-level storage.

        Args:
            task: Task name
            param_name: Parameter name
            model: Model
            bias_corrected: Whether to apply bias correction

        Returns:
            Per-parameter Fisher tensor or None
        """
        # Get group Fisher
        group_fisher = self.get_group_fisher(
            task=task,
            param_name=param_name,
            bias_corrected=bias_corrected
        )

        if not group_fisher:
            return None

        # If it's a single tensor, we have exact match
        if isinstance(group_fisher, torch.Tensor):
            # Need to expand to parameter shape
            for name, param in model.named_parameters():
                if name == param_name:
                    # Determine expansion strategy
                    key = f"{task}|{param_name}|"
                    for k in self.fisher_ema.keys():
                        if k.startswith(key):
                            group_type = k.split('|')[-1]
                            if group_type == 'channel':
                                return self._expand_channels_to_param(group_fisher, param)
                            elif group_type == 'head':
                                return self._expand_heads_to_param(group_fisher, param, param_name)
                            else:
                                return group_fisher

        return None

    def _expand_channels_to_param(
        self,
        channel_values: torch.Tensor,
        param: torch.Tensor
    ) -> torch.Tensor:
        """
        Expand per-channel values to full parameter shape.

        Args:
            channel_values: Per-channel Fisher values
            param: Parameter tensor

        Returns:
            Expanded Fisher values
        """
        param_shape = param.shape
        if len(param_shape) >= 2:
            # channel_values shape: (out_channels,)
            # param shape: (out_channels, in_channels, ...)
            expanded = channel_values.view(-1, *([1] * (len(param_shape) - 1)))
            return expanded.expand(param_shape)
        return channel_values

    def migrate_old_fisher_keys(self, old_fisher_ema: Dict[str, torch.Tensor]):
        """
        Migrate old Fisher EMA keys to new format.

        Old format: "task_param_name"
        New format: "task|param_name|group_type"

        Args:
            old_fisher_ema: Old Fisher EMA dictionary
        """
        for old_key, value in old_fisher_ema.items():
            # Skip reference parameters
            if "_ref_" in old_key:
                continue

            # Parse old key
            task_match = re.match(r'^([^_]+)_(.+)$', old_key)
            if not task_match:
                continue

            task = task_match.group(1)
            param_name = task_match.group(2)

            # Determine group type based on value shape
            if value.dim() == 1:
                if 'bias' in param_name:
                    group_type = 'bias'
                elif 'norm' in param_name or 'ln' in param_name:
                    group_type = 'row'
                else:
                    group_type = 'channel'  # Likely already reduced
            else:
                group_type = 'param'  # Full parameter

            # Create new key
            new_key = f"{task}|{param_name}|{group_type}"

            # Store in new format
            self.fisher_ema[new_key] = value

    def get_task_fisher_dict(
        self,
        task: str,
        as_param_level: bool = True,
        model: Optional[nn.Module] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Get all Fisher values for a task.

        Args:
            task: Task name
            as_param_level: If True, expand to per-parameter level
            model: Model (required if as_param_level=True)

        Returns:
            Dictionary of Fisher values
        """
        # Get group-level Fisher
        group_fisher = self.get_group_fisher(task, bias_corrected=True)

        if not as_param_level:
            # Return group-level directly
            # But clean up keys to just param names
            clean_fisher = {}
            for key, value in group_fisher.items():
                parts = key.split('|')
                if len(parts) == 3:
                    param_name = parts[1]
                    clean_fisher[param_name] = value
            return clean_fisher

        if model is None:
            raise ValueError("Model required for per-parameter expansion")

        # Expand to per-parameter
        return self.expand_group_to_param_fisher(group_fisher, model)

    def compatibility_update_fisher_ema(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        task: str = 'task1'
    ):
        """
        Backward-compatible update_fisher_ema that maintains old API.

        Args:
            model: Model
            batch: Input batch
            task: Task name
        """
        # Call parent's collect_fisher (from FisherCollector)
        self.collect_fisher(model, batch, task, mode='ema')

        # For compatibility, also maintain a simple task -> param mapping
        # This allows old code that directly accesses self.fisher_ema to work
        if not hasattr(self, '_compat_fisher_cache'):
            self._compat_fisher_cache = {}

        # Cache the expanded version for direct access
        self._compat_fisher_cache[task] = self.get_task_fisher_dict(
            task, as_param_level=False, model=None
        )

    def compatibility_estimate_fisher_diagonal(
        self,
        model: nn.Module,
        data_batch: Dict[str, torch.Tensor],
        n_samples: int = 8,
        layers_prefix: Optional[list] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Backward-compatible _estimate_fisher_diagonal.

        Args:
            model: Model
            data_batch: Input batch
            n_samples: Number of samples
            layers_prefix: Layer prefixes to include

        Returns:
            Per-parameter Fisher dictionary
        """
        # Use one-shot mode
        temp_task = '_temp_oneshot'
        group_fisher = self.collect_fisher(
            model, data_batch, temp_task, mode='oneshot'
        )

        # Filter by layer prefix if specified
        if layers_prefix:
            filtered = {}
            for key, value in group_fisher.items():
                param_name = key.split('|')[1]
                if any(param_name.startswith(prefix) for prefix in layers_prefix):
                    filtered[key] = value
            group_fisher = filtered

        # Expand to per-parameter
        param_fisher = self.expand_group_to_param_fisher(group_fisher, model)

        # Clean up temporary task
        self.clear_fisher(temp_task)

        return param_fisher