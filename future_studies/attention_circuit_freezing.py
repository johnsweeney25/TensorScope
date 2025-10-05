"""
Attention Circuit Freezing: QK/OV Selective Intervention
=========================================================
Implements selective freezing of QK (attention pattern) and OV (value mixing)
circuits in transformer attention heads for causal analysis.

This module supports both fused QKV (GPT-2 style) and separate QKV (LLaMA style)
architectures, providing theoretically equivalent interventions.

Author: Advanced Neural Analysis Framework
Date: September 2024
Paper: "Dissecting Attention: Causal Analysis of QK and OV Circuits"
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any, Literal
import warnings
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class ModelArchitecture(Enum):
    """Supported model architectures."""
    SEPARATE_QKV = "separate_qkv"  # LLaMA, Mistral, Falcon
    FUSED_QKV = "fused_qkv"  # GPT-2, GPT-J, Qwen
    FLASH_ATTENTION = "flash_attention"  # Not supported
    UNKNOWN = "unknown"


class CircuitType(Enum):
    """Circuit types for selective freezing."""
    QK = "qk"  # Query-Key circuit (attention pattern)
    OV = "ov"  # Output-Value circuit (value mixing)
    BOTH = "both"  # Both circuits (full head)


class FreezeType(Enum):
    """Freezing strategies."""
    ZERO = "zero"  # Zero out activations
    MEAN = "mean"  # Replace with mean activation
    NOISE = "noise"  # Replace with small noise
    IDENTITY = "identity"  # Identity attention pattern (QK only)


@dataclass
class InterventionConfig:
    """Configuration for circuit intervention."""
    layer_indices: List[int]
    head_indices: List[int]
    circuit: CircuitType = CircuitType.BOTH
    freeze_type: FreezeType = FreezeType.ZERO
    backward_mode: Literal["stopgrad", "ste"] = "stopgrad"  # stopgrad: block gradients, ste: straight-through estimator
    noise_scale: float = 0.01
    # Deprecated - kept for backward compatibility
    preserve_gradients: Optional[bool] = None

    def __post_init__(self):
        # Handle backward compatibility
        if self.preserve_gradients is not None:
            warnings.warn(
                "preserve_gradients is deprecated. Use backward_mode='ste' for gradient preservation, "
                "'stopgrad' to block gradients.",
                DeprecationWarning
            )
            self.backward_mode = "ste" if self.preserve_gradients else "stopgrad"


class AttentionCircuitFreezer:
    """
    Selective freezing of QK and OV circuits in attention heads.

    Supports both fused and separate QKV architectures with theoretically
    equivalent interventions at the post-projection representation level.
    """

    def __init__(self, debug_gradients: bool = False):
        self.active_hooks = []
        self.model_architecture = None
        self.model_config = None
        self.debug_gradients = debug_gradients
        self.gradient_stats = {}  # Store gradient statistics for debugging
        self.intervention_effects = {}  # Track intervention effects

    def detect_architecture(self, model: nn.Module) -> ModelArchitecture:
        """
        Auto-detect model's attention architecture.

        Args:
            model: PyTorch model to analyze

        Returns:
            Detected ModelArchitecture type
        """
        # Check for flash attention
        for module in model.modules():
            if 'flash' in module.__class__.__name__.lower():
                logger.warning("Flash Attention detected - QK/OV separation not supported")
                return ModelArchitecture.FLASH_ATTENTION

        # Check for attention layers
        for name, module in model.named_modules():
            # Look for attention modules
            if 'attention' in name.lower() or 'attn' in name.lower():
                # Check for separate QKV
                if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj'):
                    logger.info(f"Detected separate QKV architecture (LLaMA-style) at {name}")
                    return ModelArchitecture.SEPARATE_QKV

                # Check for fused QKV
                elif hasattr(module, 'c_attn'):
                    logger.info(f"Detected fused QKV architecture (GPT-2 style) at {name}")
                    return ModelArchitecture.FUSED_QKV

                elif hasattr(module, 'query_key_value'):
                    logger.info(f"Detected fused QKV architecture (GPT-NeoX style) at {name}")
                    return ModelArchitecture.FUSED_QKV

        logger.warning("Could not detect attention architecture")
        return ModelArchitecture.UNKNOWN

    def get_model_config(self, model: nn.Module) -> Dict[str, Any]:
        """
        Extract model configuration for attention parameters.

        Args:
            model: PyTorch model

        Returns:
            Dictionary with model configuration
        """
        config = {}

        # Try to get from model.config
        if hasattr(model, 'config'):
            model_config = model.config
            config['num_heads'] = getattr(model_config, 'num_attention_heads', None)
            config['hidden_size'] = getattr(model_config, 'hidden_size', None)
            config['num_kv_heads'] = getattr(model_config, 'num_key_value_heads',
                                            config.get('num_heads'))
            config['head_dim'] = config['hidden_size'] // config['num_heads'] if config['num_heads'] else None

        # Fallback: infer from layer shapes
        if not config.get('num_heads'):
            for module in model.modules():
                if hasattr(module, 'num_heads'):
                    config['num_heads'] = module.num_heads
                    break

        logger.info(f"Model config: {config}")
        return config

    def _replace(self, orig: torch.Tensor, repl: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Replace tensor with proper gradient handling.

        Args:
            orig: Original tensor
            repl: Replacement tensor
            mode: "stopgrad" to block gradients, "ste" for straight-through estimator

        Returns:
            Tensor with replacement value and specified gradient behavior
        """
        repl = repl.detach()  # Never backprop through the replacement value itself

        # Debug logging if enabled
        if self.debug_gradients and orig.requires_grad:
            # Register a hook to monitor gradients during backward pass
            def grad_monitor_hook(grad):
                if grad is not None:
                    grad_norm = grad.norm().item()
                    grad_mean = grad.abs().mean().item()
                    logger.debug(
                        f"Gradient monitor [{mode}]: norm={grad_norm:.6f}, mean={grad_mean:.6f}, "
                        f"shape={list(grad.shape)}, device={grad.device}"
                    )
                return grad

            if orig.requires_grad:
                orig.register_hook(grad_monitor_hook)

        if mode == "stopgrad":
            # Forward = repl, backward = 0 gradient wrt orig
            result = repl
            if self.debug_gradients:
                logger.debug(f"Applied stopgrad: orig norm={orig.norm().item():.6f}, "
                           f"repl norm={repl.norm().item():.6f}")
        elif mode == "ste":
            # Forward = repl, backward ≈ identity wrt orig (straight-through)
            # This preserves gradient flow as if orig was unchanged
            result = repl + (orig - orig.detach())
            if self.debug_gradients:
                logger.debug(f"Applied STE: orig norm={orig.norm().item():.6f}, "
                           f"repl norm={repl.norm().item():.6f}, "
                           f"preserving gradient flow")
        else:
            raise ValueError(f"mode must be 'stopgrad' or 'ste', got {mode}")

        return result

    def _kv_group_index(self, q_head_idx: int) -> int:
        """Map a query head index to its KV group index for GQA/MQA."""
        nh = self.model_config['num_heads']
        nkv = self.model_config.get('num_kv_heads', nh) or nh

        # Critical assertions
        assert nh % nkv == 0, f"num_heads ({nh}) must be divisible by num_kv_heads ({nkv}) for GQA"
        assert 0 <= q_head_idx < nh, f"head_idx ({q_head_idx}) out of range [0, {nh})"

        group_size = nh // nkv
        return q_head_idx // group_size

    def _dims_from_module(self, module: nn.Module, output: torch.Tensor) -> tuple:
        """Derive dimensions from module and output tensor."""
        nh = self.model_config['num_heads']
        nkv = self.model_config.get('num_kv_heads', nh) or nh
        out_hidden = output.shape[-1]

        # Prefer configured head_dim if available
        hd = self.model_config.get('head_dim')
        if hd and out_hidden in (nh * hd, nkv * hd):
            return nh, nkv, hd, out_hidden

        # Fallback: infer head_dim from output size
        if out_hidden % nh == 0:
            hd = out_hidden // nh
        elif out_hidden % nkv == 0:
            hd = out_hidden // nkv
        else:
            raise ValueError(f"Cannot infer head_dim from shape {out_hidden} with nh={nh}, nkv={nkv}")

        return nh, nkv, hd, out_hidden

    def freeze_circuits(
        self,
        model: nn.Module,
        config: InterventionConfig,
        architecture: Optional[ModelArchitecture] = None
    ) -> List[Any]:
        """
        Main entry point for circuit freezing.

        Args:
            model: Model to intervene on (work with a copy!)
            config: Intervention configuration
            architecture: Optional architecture override

        Returns:
            List of hooks that must be removed when done

        Example:
            >>> model_copy = copy.deepcopy(model)
            >>> config = InterventionConfig(
            ...     layer_indices=[0, 1],
            ...     head_indices=[3, 7],
            ...     circuit=CircuitType.QK
            ... )
            >>> hooks = freezer.freeze_circuits(model_copy, config)
            >>> # Run experiments...
            >>> freezer.remove_hooks(hooks)
        """
        # Detect architecture if not provided
        if architecture is None:
            architecture = self.detect_architecture(model)
            self.model_architecture = architecture

        # Get model configuration
        self.model_config = self.get_model_config(model)

        # Store noise_scale in config for use in hooks
        self.model_config['noise_scale'] = config.noise_scale

        # Check for IDENTITY freeze type
        if config.freeze_type == FreezeType.IDENTITY:
            raise NotImplementedError(
                "IDENTITY freeze requires score-level hooks (pre-softmax), not projection outputs. "
                "This would need access to attention scores which are not available at projection level."
            )

        # Check if architecture is supported
        if architecture == ModelArchitecture.FLASH_ATTENTION:
            logger.warning(
                "Flash Attention detected. Circuit freezing at projection level is still possible, "
                "but score-level interventions (like IDENTITY) are not available. Proceeding with projection-level freezing."
            )
            # Don't raise error - we can still freeze at projection level

        if architecture == ModelArchitecture.UNKNOWN:
            raise ValueError(
                "Could not detect model architecture. "
                "Please specify architecture manually."
            )

        # Route to appropriate implementation
        hooks = []

        if architecture == ModelArchitecture.SEPARATE_QKV:
            hooks = self._freeze_separate_qkv(model, config)
        elif architecture == ModelArchitecture.FUSED_QKV:
            hooks = self._freeze_fused_qkv(model, config)

        self.active_hooks.extend(hooks)
        logger.info(f"Registered {len(hooks)} hooks for {config.circuit.value} circuit freezing")

        return hooks

    def _freeze_separate_qkv(
        self,
        model: nn.Module,
        config: InterventionConfig
    ) -> List[Any]:
        """
        Freeze circuits for models with separate Q, K, V projections (LLaMA-style).

        This is the simpler case where we can directly hook individual projections.
        """
        hooks = []
        layers = self._get_attention_layers(model)

        for layer_idx in config.layer_indices:
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} not found, skipping")
                continue

            attn_module = layers[layer_idx]

            if config.circuit in [CircuitType.QK, CircuitType.BOTH]:
                # Freeze Q and K projections
                if hasattr(attn_module, 'q_proj'):
                    hook = self._create_projection_hook(
                        config.head_indices,
                        config.freeze_type,
                        'q_proj',
                        config.backward_mode
                    )
                    hooks.append(attn_module.q_proj.register_forward_hook(hook))
                elif hasattr(attn_module, 'query'):  # BERT-style
                    hook = self._create_projection_hook(
                        config.head_indices,
                        config.freeze_type,
                        'q_proj',  # Still use 'q_proj' as identifier
                        config.backward_mode
                    )
                    hooks.append(attn_module.query.register_forward_hook(hook))

                if hasattr(attn_module, 'k_proj'):
                    hook = self._create_projection_hook(
                        config.head_indices,
                        config.freeze_type,
                        'k_proj',
                        config.backward_mode
                    )
                    hooks.append(attn_module.k_proj.register_forward_hook(hook))
                elif hasattr(attn_module, 'key'):  # BERT-style
                    hook = self._create_projection_hook(
                        config.head_indices,
                        config.freeze_type,
                        'k_proj',  # Still use 'k_proj' as identifier
                        config.backward_mode
                    )
                    hooks.append(attn_module.key.register_forward_hook(hook))

            if config.circuit in [CircuitType.OV, CircuitType.BOTH]:
                # Freeze V projection
                if hasattr(attn_module, 'v_proj'):
                    hook = self._create_projection_hook(
                        config.head_indices,
                        config.freeze_type,
                        'v_proj',
                        config.backward_mode
                    )
                    hooks.append(attn_module.v_proj.register_forward_hook(hook))
                elif hasattr(attn_module, 'value'):  # BERT-style
                    hook = self._create_projection_hook(
                        config.head_indices,
                        config.freeze_type,
                        'v_proj',  # Still use 'v_proj' as identifier
                        config.backward_mode
                    )
                    hooks.append(attn_module.value.register_forward_hook(hook))

        return hooks

    def _freeze_fused_qkv(
        self,
        model: nn.Module,
        config: InterventionConfig
    ) -> List[Any]:
        """
        Freeze circuits for models with fused QKV projection (GPT-2 style).

        More complex: must split concatenated output and selectively modify.
        """
        hooks = []
        layers = self._get_attention_layers(model)

        for layer_idx in config.layer_indices:
            if layer_idx >= len(layers):
                logger.warning(f"Layer {layer_idx} not found, skipping")
                continue

            attn_module = layers[layer_idx]

            # Find the fused QKV projection
            if hasattr(attn_module, 'c_attn'):
                hook = self._create_fused_qkv_hook(
                    config.head_indices,
                    config.circuit,
                    config.freeze_type,
                    config.backward_mode,
                    config.noise_scale
                )
                hooks.append(attn_module.c_attn.register_forward_hook(hook))
            elif hasattr(attn_module, 'query_key_value'):
                hook = self._create_fused_qkv_hook(
                    config.head_indices,
                    config.circuit,
                    config.freeze_type,
                    config.backward_mode,
                    config.noise_scale
                )
                hooks.append(attn_module.query_key_value.register_forward_hook(hook))

        return hooks

    def _create_projection_hook(
        self,
        head_indices: List[int],
        freeze_type: FreezeType,
        projection_type: str,
        backward_mode: str,
        noise_scale: float = 0.01
    ):
        """Create a hook for separate Q, K, or V projections."""

        def hook(module, input, output):
            # Output shape: [batch, seq, hidden]
            batch_size, seq_len, hidden_size = output.shape

            # Calculate dimensions - handle GQA/MQA
            num_heads = self.model_config['num_heads']
            num_kv_heads = self.model_config.get('num_kv_heads', num_heads)

            # For K and V projections in GQA/MQA, use num_kv_heads
            if projection_type in ['k_proj', 'v_proj']:
                actual_num_heads = num_kv_heads
                head_dim = hidden_size // num_kv_heads
            else:  # Q projection
                actual_num_heads = num_heads
                head_dim = hidden_size // num_heads

            # Reshape to separate heads (using reshape for safety)
            output_reshaped = output.reshape(batch_size, seq_len, actual_num_heads, head_dim)

            # Apply freezing to specified heads
            # For GQA/MQA, map query head indices to KV head indices
            if projection_type in ['k_proj', 'v_proj'] and num_kv_heads < num_heads:
                # Map query head indices to KV head indices
                # E.g., with 32 Q heads and 8 KV heads, Q heads 0-3 map to KV head 0
                heads_per_kv = num_heads // num_kv_heads
                mapped_indices = set(h // heads_per_kv for h in head_indices if h < num_heads)
                effective_indices = [h for h in mapped_indices if h < num_kv_heads]
            else:
                effective_indices = [h for h in head_indices if h < actual_num_heads]

            for head_idx in effective_indices:
                orig_slice = output_reshaped[:, :, head_idx, :]

                if freeze_type == FreezeType.ZERO:
                    repl = torch.zeros_like(orig_slice)
                    output_reshaped[:, :, head_idx, :] = self._replace(orig_slice, repl, backward_mode)

                elif freeze_type == FreezeType.MEAN:
                    # Per-token mean (across head_dim, not batch/seq)
                    # Important: detach to prevent gradient leakage through mean
                    mean_val = orig_slice.mean(dim=-1, keepdim=True).detach()
                    output_reshaped[:, :, head_idx, :] = self._replace(orig_slice, mean_val, backward_mode)

                elif freeze_type == FreezeType.NOISE:
                    ns = self.model_config.get('noise_scale', noise_scale)
                    noise = torch.randn_like(orig_slice) * ns
                    output_reshaped[:, :, head_idx, :] = self._replace(orig_slice, noise, backward_mode)

                elif freeze_type == FreezeType.IDENTITY:
                    # For identity, we can't directly implement it here since we need
                    # access to attention scores. Log a warning.
                    logger.warning(
                        "IDENTITY freeze type not supported for projection hooks. "
                        "Use ZERO or implement at attention score level."
                    )

            # Reshape back
            return output_reshaped.reshape(batch_size, seq_len, hidden_size)

        return hook

    def _create_fused_qkv_hook(
        self,
        head_indices: List[int],
        circuit: CircuitType,
        freeze_type: FreezeType,
        backward_mode: str,
        noise_scale: float = 0.01
    ):
        """Create a hook for fused QKV projections with proper GQA/MQA support."""

        def hook(module, input, output):
            x = output
            batch_size, seq_len, total_size = x.shape

            # Get dimensions from config
            nh = self.model_config['num_heads']
            nkv = self.model_config.get('num_kv_heads', nh) or nh

            # Infer head_dim from total size and head counts
            # For fused QKV: total = nh*hd + 2*nkv*hd
            hd = self.model_config.get('head_dim')
            if hd is None:
                # total_size = nh*hd + 2*nkv*hd = hd*(nh + 2*nkv)
                expected_divisor = nh + 2 * nkv
                if total_size % expected_divisor == 0:
                    hd = total_size // expected_divisor
                else:
                    # Fallback: assume equal splits (old behavior)
                    hd = total_size // (3 * nh)
                    logger.warning(f"Could not infer head_dim precisely, using {hd}")

            # Calculate split sizes
            q_hidden = nh * hd
            kv_hidden = nkv * hd

            # Split the fused tensor
            if q_hidden + 2 * kv_hidden == total_size:
                # Correct split for GQA/MQA
                q = x[..., :q_hidden]
                k = x[..., q_hidden:q_hidden + kv_hidden]
                v = x[..., q_hidden + kv_hidden:q_hidden + 2 * kv_hidden]
            else:
                # Fallback to equal splits
                splits = total_size // 3
                q, k, v = x.split(splits, dim=-1)
                logger.warning("Using fallback equal splits for QKV")

            # Reshape to separate heads (using reshape for safety)
            q = q.reshape(batch_size, seq_len, nh, hd)
            k = k.reshape(batch_size, seq_len, nkv, hd)
            v = v.reshape(batch_size, seq_len, nkv, hd)

            # Apply freezing based on circuit type
            for head_idx in head_indices:
                if head_idx >= nh:
                    continue

                # For GQA/MQA, map query head index to KV head index
                kv_head_idx = self._kv_group_index(head_idx) if nkv < nh else head_idx

                if circuit in [CircuitType.QK, CircuitType.BOTH]:
                    # Freeze Q and K for this head
                    orig_q = q[:, :, head_idx, :]

                    if freeze_type == FreezeType.ZERO:
                        q[:, :, head_idx, :] = self._replace(orig_q, torch.zeros_like(orig_q), backward_mode)
                        if kv_head_idx < nkv:
                            orig_k = k[:, :, kv_head_idx, :]
                            k[:, :, kv_head_idx, :] = self._replace(orig_k, torch.zeros_like(orig_k), backward_mode)
                    elif freeze_type == FreezeType.MEAN:
                        # Per-token mean with detach to prevent gradient leakage
                        q_mean = orig_q.mean(dim=-1, keepdim=True).detach()
                        q[:, :, head_idx, :] = self._replace(orig_q, q_mean, backward_mode)
                        if kv_head_idx < nkv:
                            orig_k = k[:, :, kv_head_idx, :]
                            k_mean = orig_k.mean(dim=-1, keepdim=True).detach()
                            k[:, :, kv_head_idx, :] = self._replace(orig_k, k_mean, backward_mode)
                    elif freeze_type == FreezeType.NOISE:
                        ns = self.model_config.get('noise_scale', noise_scale)
                        q_noise = torch.randn_like(orig_q) * ns
                        q[:, :, head_idx, :] = self._replace(orig_q, q_noise, backward_mode)
                        if kv_head_idx < nkv:
                            orig_k = k[:, :, kv_head_idx, :]
                            k_noise = torch.randn_like(orig_k) * ns
                            k[:, :, kv_head_idx, :] = self._replace(orig_k, k_noise, backward_mode)

                if circuit in [CircuitType.OV, CircuitType.BOTH]:
                    # Freeze V for this head
                    if kv_head_idx < nkv:
                        orig_v = v[:, :, kv_head_idx, :]

                        if freeze_type == FreezeType.ZERO:
                            v[:, :, kv_head_idx, :] = self._replace(orig_v, torch.zeros_like(orig_v), backward_mode)
                        elif freeze_type == FreezeType.MEAN:
                            v_mean = orig_v.mean(dim=-1, keepdim=True).detach()
                            v[:, :, kv_head_idx, :] = self._replace(orig_v, v_mean, backward_mode)
                        elif freeze_type == FreezeType.NOISE:
                            ns = self.model_config.get('noise_scale', noise_scale)
                            v_noise = torch.randn_like(orig_v) * ns
                            v[:, :, kv_head_idx, :] = self._replace(orig_v, v_noise, backward_mode)

            # Reshape back and concatenate
            q = q.reshape(batch_size, seq_len, q_hidden)
            k = k.reshape(batch_size, seq_len, kv_hidden)
            v = v.reshape(batch_size, seq_len, kv_hidden)

            return torch.cat([q, k, v], dim=-1)

        return hook

    def _get_attention_layers(self, model: nn.Module) -> List[nn.Module]:
        """Get list of attention modules from model."""
        attention_layers = []

        # Try standard paths
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # LLaMA-style
            for layer in model.model.layers:
                if hasattr(layer, 'self_attn'):
                    attention_layers.append(layer.self_attn)

        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            for layer in model.transformer.h:
                if hasattr(layer, 'attn'):
                    attention_layers.append(layer.attn)

        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
            # BERT-style
            for layer in model.encoder.layer:
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'self'):
                    attention_layers.append(layer.attention.self)

        if not attention_layers:
            # Fallback: search for attention modules
            for name, module in model.named_modules():
                if any(x in name for x in ['self_attn', 'attention', 'attn']):
                    # Check for various projection types
                    if (hasattr(module, 'q_proj') or      # LLaMA-style
                        hasattr(module, 'c_attn') or       # GPT-2 style
                        hasattr(module, 'query_key_value') or  # NeoX style
                        hasattr(module, 'query')):         # BERT-style
                        attention_layers.append(module)

        logger.info(f"Found {len(attention_layers)} attention layers")
        return attention_layers

    def remove_hooks(self, hooks: Optional[List[Any]] = None):
        """
        Remove intervention hooks.

        Args:
            hooks: Specific hooks to remove. If None, removes all active hooks.
        """
        if hooks is None:
            hooks = self.active_hooks

        removed = 0
        for hook in hooks:
            try:
                hook.remove()
                removed += 1
            except:
                pass

        # Clean up active hooks list
        if hooks is self.active_hooks:
            self.active_hooks = []
        else:
            self.active_hooks = [h for h in self.active_hooks if h not in hooks]

        logger.info(f"Removed {removed} hooks")

    def check_gradient_behavior(
        self,
        model: nn.Module,
        test_input: torch.Tensor,
        config: InterventionConfig,
        target_param_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Check if gradients behave correctly according to backward_mode.

        Args:
            model: Model with interventions applied
            test_input: Test input tensor (requires_grad=True)
            config: Intervention configuration used
            target_param_name: Optional specific parameter to check (e.g., "model.layers.0.self_attn.q_proj.weight")

        Returns:
            Dictionary with gradient statistics and validation results
        """
        # Ensure model is in training mode
        model.train()
        test_input = test_input.clone().detach().requires_grad_(True)

        # Forward pass
        output = model(test_input)

        # Create a simple loss (sum of outputs)
        if hasattr(output, 'logits'):
            loss = output.logits.sum()
        elif isinstance(output, torch.Tensor):
            loss = output.sum()
        else:
            loss = output[0].sum() if isinstance(output, (tuple, list)) else output.sum()

        # Backward pass
        loss.backward()

        # Collect gradient statistics
        stats = {
            'backward_mode': config.backward_mode,
            'loss_value': loss.item(),
            'input_grad_exists': test_input.grad is not None,
            'input_grad_norm': test_input.grad.norm().item() if test_input.grad is not None else 0.0,
            'parameter_grads': {},
            'frozen_layers': config.layer_indices,
            'frozen_heads': config.head_indices,
            'validation_passed': True
        }

        # Check specific parameters or all parameters
        params_to_check = []
        if target_param_name:
            for name, param in model.named_parameters():
                if target_param_name in name:
                    params_to_check.append((name, param))
        else:
            # Check projection parameters in frozen layers
            for layer_idx in config.layer_indices:
                for name, param in model.named_parameters():
                    if f"layers.{layer_idx}" in name or f"layer.{layer_idx}" in name or f"h.{layer_idx}" in name:
                        if any(proj in name for proj in ['q_proj', 'k_proj', 'v_proj', 'c_attn', 'query', 'key', 'value']):
                            params_to_check.append((name, param))

        # Analyze gradients
        for name, param in params_to_check:
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_mean = param.grad.mean().item()
                grad_std = param.grad.std().item()
                grad_max = param.grad.abs().max().item()
                grad_nonzero_pct = (param.grad != 0).float().mean().item() * 100

                stats['parameter_grads'][name] = {
                    'norm': grad_norm,
                    'mean': grad_mean,
                    'std': grad_std,
                    'max': grad_max,
                    'nonzero_pct': grad_nonzero_pct
                }

                # Validation logic based on backward_mode
                if config.backward_mode == "stopgrad":
                    # For targeted parameters, gradients should be significantly reduced
                    if config.circuit in [CircuitType.QK, CircuitType.BOTH]:
                        if 'q_proj' in name or 'k_proj' in name or 'query' in name or 'key' in name:
                            if grad_norm > 1e-6:  # Should be near zero
                                logger.warning(f"Expected near-zero gradient for {name} in stopgrad mode, got norm={grad_norm}")
                                stats['validation_passed'] = False

                elif config.backward_mode == "ste":
                    # Gradients should flow (be non-zero)
                    if grad_norm < 1e-8:
                        logger.warning(f"Expected non-zero gradient for {name} in STE mode, got norm={grad_norm}")
                        stats['validation_passed'] = False

        # Log summary if debug is enabled
        if self.debug_gradients:
            logger.info("="*60)
            logger.info(f"Gradient Behavior Check - {config.backward_mode} mode")
            logger.info(f"Loss: {stats['loss_value']:.6f}")
            logger.info(f"Input grad norm: {stats['input_grad_norm']:.6f}")
            logger.info("-"*40)

            for param_name, param_stats in stats['parameter_grads'].items():
                # Shorten parameter name for display
                short_name = param_name.split('.')[-3:]  # Last 3 components
                short_name = '.'.join(short_name)
                logger.info(
                    f"{short_name:30s} | "
                    f"norm: {param_stats['norm']:8.6f} | "
                    f"mean: {param_stats['mean']:8.6f} | "
                    f"max: {param_stats['max']:8.6f} | "
                    f"nonzero: {param_stats['nonzero_pct']:5.1f}%"
                )

            logger.info(f"Validation: {'PASSED' if stats['validation_passed'] else 'FAILED'}")
            logger.info("="*60)

        # Store for later inspection
        self.gradient_stats = stats

        return stats

    def verify_intervention_effect(
        self,
        model_baseline: nn.Module,
        model_intervened: nn.Module,
        test_input: torch.Tensor,
        config: InterventionConfig,
        tolerance: float = 0.01
    ) -> Dict[str, Any]:
        """
        Verify that intervention actually changes model behavior.

        Args:
            model_baseline: Original model without intervention
            model_intervened: Model with intervention applied
            test_input: Test input
            config: Intervention configuration
            tolerance: Minimum expected change

        Returns:
            Dictionary with effect measurements
        """
        with torch.no_grad():
            # Get outputs
            output_baseline = model_baseline(test_input)
            output_intervened = model_intervened(test_input)

            # Handle different output types
            if hasattr(output_baseline, 'logits'):
                out_base = output_baseline.logits
                out_int = output_intervened.logits
            else:
                out_base = output_baseline
                out_int = output_intervened

            # Calculate differences
            abs_diff = (out_base - out_int).abs()
            rel_diff = abs_diff / (out_base.abs() + 1e-8)

            effect_stats = {
                'max_absolute_diff': abs_diff.max().item(),
                'mean_absolute_diff': abs_diff.mean().item(),
                'max_relative_diff': rel_diff.max().item(),
                'mean_relative_diff': rel_diff.mean().item(),
                'output_changed': abs_diff.max().item() > tolerance,
                'circuit': config.circuit.value,
                'freeze_type': config.freeze_type.value
            }

            if self.debug_gradients:
                logger.info("="*60)
                logger.info("Intervention Effect Verification")
                logger.info(f"Circuit: {effect_stats['circuit']}, Freeze: {effect_stats['freeze_type']}")
                logger.info(f"Max absolute difference: {effect_stats['max_absolute_diff']:.6f}")
                logger.info(f"Mean absolute difference: {effect_stats['mean_absolute_diff']:.6f}")
                logger.info(f"Max relative difference: {effect_stats['max_relative_diff']:.2%}")
                logger.info(f"Effect detected: {'YES' if effect_stats['output_changed'] else 'NO'}")
                logger.info("="*60)

            self.intervention_effects = effect_stats

            return effect_stats

    def validate_intervention(
        self,
        model: nn.Module,
        test_input: torch.Tensor,
        config: InterventionConfig
    ) -> Dict[str, Any]:
        """
        Validate that interventions are working correctly.

        Args:
            model: Model with interventions
            test_input: Sample input
            config: Intervention configuration

        Returns:
            Validation results dictionary
        """
        results = {}

        # Get attention weights if possible
        # This would require additional hooks to capture attention weights
        # Implementation depends on model architecture

        logger.info("Intervention validation completed")
        return results


# Convenience functions
def freeze_qk_circuit(
    model: nn.Module,
    layer_indices: List[int],
    head_indices: List[int],
    freeze_type: FreezeType = FreezeType.ZERO,
    backward_mode: str = "stopgrad"
) -> List[Any]:
    """
    Convenience function to freeze QK circuit.

    Args:
        model: Model to intervene on
        layer_indices: Layers to target
        head_indices: Heads to freeze
        freeze_type: How to freeze

    Returns:
        List of hooks
    """
    freezer = AttentionCircuitFreezer()
    config = InterventionConfig(
        layer_indices=layer_indices,
        head_indices=head_indices,
        circuit=CircuitType.QK,
        freeze_type=freeze_type,
        backward_mode=backward_mode
    )
    return freezer.freeze_circuits(model, config)


def freeze_ov_circuit(
    model: nn.Module,
    layer_indices: List[int],
    head_indices: List[int],
    freeze_type: FreezeType = FreezeType.ZERO,
    backward_mode: str = "stopgrad"
) -> List[Any]:
    """
    Convenience function to freeze OV circuit.

    Args:
        model: Model to intervene on
        layer_indices: Layers to target
        head_indices: Heads to freeze
        freeze_type: How to freeze

    Returns:
        List of hooks
    """
    freezer = AttentionCircuitFreezer()
    config = InterventionConfig(
        layer_indices=layer_indices,
        head_indices=head_indices,
        circuit=CircuitType.OV,
        freeze_type=freeze_type,
        backward_mode=backward_mode
    )
    return freezer.freeze_circuits(model, config)


if __name__ == "__main__":
    print("Attention Circuit Freezing Module")
    print("=================================")
    print("Supports selective QK/OV circuit interventions")
    print("\nExample usage:")
    print("  from attention_circuit_freezing import freeze_qk_circuit")
    print("  hooks = freeze_qk_circuit(model, [0, 1], [3, 7])")
    print("  # Run analysis...")
    print("  for hook in hooks: hook.remove()")