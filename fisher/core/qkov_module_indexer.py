"""
Architecture-Agnostic QK-OV Module Indexer

Replaces regex-based parameter matching with direct module traversal.
This works for ANY transformer architecture that follows standard attention patterns.

Key advantages:
1. No regex patterns to maintain
2. Works with custom/research architectures
3. Fails fast with clear errors
4. Handles parameter name variations automatically

Author: ICLR 2026 Project (Intern recommendation)
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class AttentionModuleInfo:
    """Information about an attention module's structure."""
    layer_idx: int
    q_module: Optional[nn.Module] = None
    k_module: Optional[nn.Module] = None
    v_module: Optional[nn.Module] = None
    o_module: Optional[nn.Module] = None
    fused_qkv_module: Optional[nn.Module] = None
    q_param_name: Optional[str] = None
    k_param_name: Optional[str] = None
    v_param_name: Optional[str] = None
    o_param_name: Optional[str] = None
    fused_qkv_param_name: Optional[str] = None


class ModuleBasedQKOVIndexer:
    """
    Build QK-OV index directly from model's module structure.

    This is architecture-agnostic: it works by walking the model's
    actual modules rather than matching regex patterns.

    Usage:
        indexer = ModuleBasedQKOVIndexer.from_model(model)
        param_name = indexer.get_param_name(layer=3, block='Q')
    """

    def __init__(self, attention_modules: List[AttentionModuleInfo]):
        """
        Initialize with pre-built attention module information.

        Args:
            attention_modules: List of AttentionModuleInfo for each layer
        """
        self.attention_modules = attention_modules
        self._build_lookup_tables()

    def _build_lookup_tables(self):
        """Build fast lookup tables for parameter names."""
        self.param_names = {}  # (layer, block) -> param_name

        for attn_info in self.attention_modules:
            layer_idx = attn_info.layer_idx

            # Handle fused QKV
            if attn_info.fused_qkv_param_name is not None:
                self.param_names[(layer_idx, 'Q')] = attn_info.fused_qkv_param_name
                self.param_names[(layer_idx, 'K')] = attn_info.fused_qkv_param_name
                self.param_names[(layer_idx, 'V')] = attn_info.fused_qkv_param_name
            else:
                # Split projections
                if attn_info.q_param_name:
                    self.param_names[(layer_idx, 'Q')] = attn_info.q_param_name
                if attn_info.k_param_name:
                    self.param_names[(layer_idx, 'K')] = attn_info.k_param_name
                if attn_info.v_param_name:
                    self.param_names[(layer_idx, 'V')] = attn_info.v_param_name

            # O projection (always separate)
            if attn_info.o_param_name:
                self.param_names[(layer_idx, 'O')] = attn_info.o_param_name

    @classmethod
    def from_model(cls, model: nn.Module, verbose: bool = False) -> 'ModuleBasedQKOVIndexer':
        """
        Build index by traversing model's module structure.

        This uses heuristics to identify attention modules:
        - Look for modules named 'attn', 'attention', 'self_attn', etc.
        - Within those, find q/k/v/o projections or fused qkv
        - Track parameter names for each

        Args:
            model: PyTorch model to index
            verbose: Print discovered modules

        Returns:
            ModuleBasedQKOVIndexer instance
        """
        attention_modules = []

        # Strategy 1: Look for common layer list attributes
        layer_lists = []
        for attr in ['layers', 'h', 'blocks', 'encoder', 'decoder']:
            if hasattr(model, attr):
                candidate = getattr(model, attr)
                if isinstance(candidate, nn.ModuleList):
                    layer_lists.append((attr, candidate))

        # If model wraps layers (e.g., model.transformer.h)
        if hasattr(model, 'transformer'):
            for attr in ['layers', 'h', 'blocks']:
                if hasattr(model.transformer, attr):
                    candidate = getattr(model.transformer, attr)
                    if isinstance(candidate, nn.ModuleList):
                        layer_lists.append((f'transformer.{attr}', candidate))

        if hasattr(model, 'model'):
            for attr in ['layers', 'h', 'blocks']:
                if hasattr(model.model, attr):
                    candidate = getattr(model.model, attr)
                    if isinstance(candidate, nn.ModuleList):
                        layer_lists.append((f'model.{attr}', candidate))

        if not layer_lists:
            raise ValueError(
                "Could not find layer list in model. "
                "Expected attributes like 'layers', 'h', 'blocks', etc."
            )

        # Use the first found layer list
        layer_attr_name, layers = layer_lists[0]

        if verbose:
            logger.info(f"Found {len(layers)} layers in model.{layer_attr_name}")

        # Traverse each layer
        for layer_idx, layer_module in enumerate(layers):
            attn_info = cls._extract_attention_info(
                layer_module,
                layer_idx,
                layer_attr_name,
                verbose=verbose
            )
            if attn_info:
                attention_modules.append(attn_info)

        if not attention_modules:
            raise ValueError(
                "Could not find any attention modules in model layers. "
                "Expected modules named 'attn', 'attention', 'self_attn', etc."
            )

        if verbose:
            logger.info(f"Successfully indexed {len(attention_modules)} attention layers")

        return cls(attention_modules)

    @staticmethod
    def _extract_attention_info(
        layer_module: nn.Module,
        layer_idx: int,
        layer_path: str,
        verbose: bool = False
    ) -> Optional[AttentionModuleInfo]:
        """
        Extract attention module information from a single layer.

        Args:
            layer_module: The layer module to inspect
            layer_idx: Layer index
            layer_path: Path to this layer in model (for param names)
            verbose: Print discovered modules

        Returns:
            AttentionModuleInfo or None if not found
        """
        # Find attention module within layer
        attn_module = None
        attn_attr_name = None

        for attr in ['self_attn', 'attn', 'attention', 'self_attention']:
            if hasattr(layer_module, attr):
                attn_module = getattr(layer_module, attr)
                attn_attr_name = attr
                break

        if attn_module is None:
            logger.warning(f"Layer {layer_idx}: No attention module found")
            return None

        info = AttentionModuleInfo(layer_idx=layer_idx)

        # Check for fused QKV (GPT-2 style: c_attn, ViT style: qkv)
        for fused_name in ['c_attn', 'qkv', 'in_proj']:
            if hasattr(attn_module, fused_name):
                fused_module = getattr(attn_module, fused_name)
                if hasattr(fused_module, 'weight'):
                    info.fused_qkv_module = fused_module
                    info.fused_qkv_param_name = f"{layer_path}.{layer_idx}.{attn_attr_name}.{fused_name}.weight"

                    if verbose:
                        logger.info(f"Layer {layer_idx}: Found fused QKV '{fused_name}'")

                    # Still need O projection
                    for o_name in ['c_proj', 'out_proj', 'o_proj', 'wo']:
                        if hasattr(attn_module, o_name):
                            o_module = getattr(attn_module, o_name)
                            if hasattr(o_module, 'weight'):
                                info.o_module = o_module
                                info.o_param_name = f"{layer_path}.{layer_idx}.{attn_attr_name}.{o_name}.weight"
                                break

                    return info

        # Check for split Q/K/V projections
        # Try different naming conventions
        q_names = ['q_proj', 'query', 'q', 'wq', 'to_q']
        k_names = ['k_proj', 'key', 'k', 'wk', 'to_k']
        v_names = ['v_proj', 'value', 'v', 'wv', 'to_v']
        o_names = ['o_proj', 'out_proj', 'out', 'o', 'wo', 'to_out']

        for q_name in q_names:
            if hasattr(attn_module, q_name):
                q_module = getattr(attn_module, q_name)
                if hasattr(q_module, 'weight'):
                    info.q_module = q_module
                    info.q_param_name = f"{layer_path}.{layer_idx}.{attn_attr_name}.{q_name}.weight"
                    break

        for k_name in k_names:
            if hasattr(attn_module, k_name):
                k_module = getattr(attn_module, k_name)
                if hasattr(k_module, 'weight'):
                    info.k_module = k_module
                    info.k_param_name = f"{layer_path}.{layer_idx}.{attn_attr_name}.{k_name}.weight"
                    break

        for v_name in v_names:
            if hasattr(attn_module, v_name):
                v_module = getattr(attn_module, v_name)
                if hasattr(v_module, 'weight'):
                    info.v_module = v_module
                    info.v_param_name = f"{layer_path}.{layer_idx}.{attn_attr_name}.{v_name}.weight"
                    break

        for o_name in o_names:
            if hasattr(attn_module, o_name):
                o_module = getattr(attn_module, o_name)
                if hasattr(o_module, 'weight'):
                    info.o_module = o_module
                    info.o_param_name = f"{layer_path}.{layer_idx}.{attn_attr_name}.{o_name}.weight"
                    break

        # Validate we found at least some projections
        if not any([info.q_module, info.k_module, info.v_module, info.o_module]):
            logger.warning(f"Layer {layer_idx}: No Q/K/V/O projections found")
            return None

        if verbose:
            found = []
            if info.q_param_name: found.append('Q')
            if info.k_param_name: found.append('K')
            if info.v_param_name: found.append('V')
            if info.o_param_name: found.append('O')
            logger.info(f"Layer {layer_idx}: Found split projections {found}")

        return info

    def get_param_name(self, layer: int, block: str) -> Optional[str]:
        """
        Get parameter name for a specific layer and block.

        Args:
            layer: Layer index
            block: 'Q', 'K', 'V', or 'O'

        Returns:
            Parameter name string or None if not found
        """
        return self.param_names.get((layer, block))

    def validate(self, model_params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Validate that all indexed parameter names exist in model.

        Args:
            model_params: Dict of parameter names -> tensors from model

        Returns:
            Validation report dict
        """
        missing = []
        found = []

        for (layer, block), param_name in self.param_names.items():
            if param_name in model_params:
                found.append((layer, block, param_name))
            else:
                missing.append((layer, block, param_name))

        return {
            'valid': len(missing) == 0,
            'found': found,
            'missing': missing,
            'coverage': len(found) / (len(found) + len(missing)) if (len(found) + len(missing)) > 0 else 0
        }


# Example usage / test
if __name__ == '__main__':
    # Test with a mock model
    class MockAttention(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.o_proj = nn.Linear(hidden_size, hidden_size)

    class MockLayer(nn.Module):
        def __init__(self, hidden_size):
            super().__init__()
            self.self_attn = MockAttention(hidden_size)

    class MockModel(nn.Module):
        def __init__(self, num_layers=4, hidden_size=64):
            super().__init__()
            self.layers = nn.ModuleList([MockLayer(hidden_size) for _ in range(num_layers)])

    model = MockModel()

    # Build index
    indexer = ModuleBasedQKOVIndexer.from_model(model, verbose=True)

    # Test lookup
    print(f"\nTesting parameter name lookup:")
    for layer in range(4):
        for block in ['Q', 'K', 'V', 'O']:
            param_name = indexer.get_param_name(layer, block)
            print(f"  Layer {layer}, {block}: {param_name}")

    # Validate against actual model parameters
    model_params = dict(model.named_parameters())
    validation = indexer.validate(model_params)

    print(f"\nValidation:")
    print(f"  Valid: {validation['valid']}")
    print(f"  Coverage: {validation['coverage']:.1%}")
    if validation['missing']:
        print(f"  Missing: {validation['missing']}")
