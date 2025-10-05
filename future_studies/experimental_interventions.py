"""
FutureStudies: Experimental Beta Methods for Advanced Model Analysis
====================================================================

⚠️ WARNING: EXPERIMENTAL/ALPHA CODE ⚠️

This module contains experimental methods for advanced neural network analysis
including causal intervention, head freezing, and other invasive techniques.
These methods are in beta and may:
- Have unexpected behavior
- Change significantly in future versions  
- Require careful understanding before use
- Potentially damage model performance if misused

Use at your own risk and always work with model copies, not originals.

Categories:
- Head Causality & Intervention: Surgical modifications to attention heads
- Intervention Analysis: Finding corrective vectors between models
- Advanced Mechanistic Interpretability: Experimental probing techniques

Dependencies:
- torch
- numpy
- typing
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
import warnings

# Issue experimental warning on import
warnings.warn(
    "FutureStudies contains EXPERIMENTAL methods that may have unexpected behavior. "
    "These are beta features for research purposes. Use with caution.",
    category=UserWarning,
    stacklevel=2
)


class ExperimentalInterventions:
    """
    Experimental intervention methods for causal analysis and model surgery.
    
    ⚠️ ALPHA: These methods modify model behavior and should be used carefully.
    """
    
    def __init__(self):
        """Initialize experimental intervention tools."""
        self.active_hooks = []
        warnings.warn(
            "ExperimentalInterventions initialized. Remember to remove hooks after use!",
            category=UserWarning
        )
    
    # ============= HEAD CAUSALITY & INTERVENTION =============
    
    def freeze_attention_heads(
        self,
        model,
        head_indices: Dict[int, List[int]],
        freeze_type: str = 'zero'
    ) -> List:
        """
        🧪 EXPERIMENTAL: Freeze or zero specific attention heads for causality analysis.
        
        This method surgically disables specific attention heads to study their
        causal role in model behavior. Use with extreme caution as it modifies
        model computation.
        
        Args:
            model: The model to intervene on (work with a copy!)
            head_indices: {layer_idx: [head_indices_to_freeze]}
            freeze_type: 
                - 'zero': Zero out head outputs completely
                - 'identity': Make heads pass-through (not implemented)
                - 'soft': Reduce head influence by 90%
        
        Returns:
            List of hooks (MUST call remove() on each when done!)
            
        Example:
            >>> model_copy = copy.deepcopy(model)
            >>> hooks = freeze_attention_heads(model_copy, {0: [3, 7], 2: [1]})
            >>> # Run experiments...
            >>> for hook in hooks:
            ...     hook.remove()
        
        ⚠️ WARNING: Hooks persist until manually removed!
        """
        hooks = []
        
        def make_head_intervention_hook(layer_idx, heads_to_freeze, freeze_mode):
            def hook(module, input_tuple):
                # input_tuple[0] is the concatenated head outputs [B, S, H*D_head]
                if len(input_tuple) > 0:
                    hidden_states = input_tuple[0]
                    
                    # Get dimensions from parent attention module
                    # Traverse up to get self_attn module
                    parent_attn = module.parent if hasattr(module, 'parent') else None
                    
                    # Fallback: estimate from hidden dim (assuming standard config)
                    batch_size, seq_len, hidden_dim = hidden_states.shape
                    # Common head counts for models
                    if hidden_dim % 32 == 0:
                        num_heads = 32
                    elif hidden_dim % 16 == 0:
                        num_heads = 16
                    elif hidden_dim % 12 == 0:
                        num_heads = 12
                    else:
                        num_heads = 8  # Default fallback
                    
                    head_dim = hidden_dim // num_heads
                    
                    # Reshape to separate heads: [batch, seq, num_heads, head_dim]
                    hidden_states = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
                    
                    # Apply intervention to specified heads
                    for head_idx in heads_to_freeze:
                        if head_idx < num_heads:
                            if freeze_mode == 'zero':
                                hidden_states[:, :, head_idx, :] = 0
                            elif freeze_mode == 'soft':
                                hidden_states[:, :, head_idx, :] *= 0.1
                            elif freeze_mode == 'identity':
                                warnings.warn("Identity mode not yet implemented", UserWarning)
                    
                    # Reshape back
                    hidden_states = hidden_states.view(batch_size, seq_len, hidden_dim)
                    
                    # Return modified tuple
                    return (hidden_states,) + input_tuple[1:]
                
                return input_tuple
            
            return hook
        
        # Register hooks on o_proj (output projection) for robustness
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            for layer_idx, heads in head_indices.items():
                if layer_idx < len(model.model.layers):
                    layer = model.model.layers[layer_idx]
                    # Hook the o_proj (output projection) instead of self_attn
                    if hasattr(layer, 'self_attn') and hasattr(layer.self_attn, 'o_proj'):
                        # Use forward_pre_hook to modify input before projection
                        hook = layer.self_attn.o_proj.register_forward_pre_hook(
                            make_head_intervention_hook(layer_idx, heads, freeze_type)
                        )
                        hooks.append(hook)
                        self.active_hooks.append(hook)
        
        if not hooks:
            warnings.warn(
                "No hooks were registered. Check model architecture compatibility.",
                UserWarning
            )
        
        return hooks
    
    def remove_all_hooks(self):
        """Remove all active hooks from this instance."""
        for hook in self.active_hooks:
            try:
                hook.remove()
            except:
                pass
        self.active_hooks = []
        print(f"Removed {len(self.active_hooks)} hooks")
    
    # ============= INTERVENTION ANALYSIS =============
    
    def find_intervention_vectors(
        self,
        model_broken,
        model_healthy,
        probe_batch: Optional[Dict[str, torch.Tensor]] = None,
        return_per_layer: bool = True
    ) -> Dict[str, Any]:
        """
        🧪 EXPERIMENTAL: Find weight directions that could restore performance.
        
        This method computes the difference between a "healthy" and "broken" model
        to identify intervention vectors that might restore functionality.
        With single (healthy-broken) diff, this is rank-1 per parameter.
        
        Args:
            model_broken: The degraded/broken model
            model_healthy: The reference healthy model
            probe_batch: Optional batch for testing (not used in basic version)
            return_per_layer: Whether to compute per-layer intervention vectors
        
        Returns:
            Dictionary containing:
            - intervention_vector: The normalized difference vector (rank-1)
            - intervention_magnitude: Strength of intervention needed
            - per_layer_vectors: Dict of per-layer intervention directions
            - weight_diff_norms: Norms of differences per parameter
            
        ⚠️ WARNING: This is a simplified intervention that may not work for
        complex degradation patterns. Use for research exploration only.
        """
        # Get weight difference
        weight_diff = {}
        layer_diffs = defaultdict(list)
        
        for (name_b, param_b), (name_h, param_h) in zip(
            model_broken.named_parameters(),
            model_healthy.named_parameters()
        ):
            if name_b == name_h:
                diff = (param_h - param_b).detach().cpu()
                weight_diff[name_b] = diff
                
                # Group by layer for per-layer analysis
                if 'layers.' in name_b or 'layer.' in name_b:
                    parts = name_b.split('.')
                    for i, part in enumerate(parts):
                        if part in ['layers', 'layer'] and i+1 < len(parts) and parts[i+1].isdigit():
                            layer_idx = int(parts[i+1])
                            layer_key = f'layer_{layer_idx}'
                            layer_diffs[layer_key].append((name_b, diff))
                            break
        
        # The intervention vector IS the difference (rank-1 by definition)
        # Normalize to get direction
        all_diffs = []
        for name, diff in weight_diff.items():
            all_diffs.append(diff.view(-1))
        
        full_diff = torch.cat(all_diffs)
        diff_norm = full_diff.norm().item()
        
        if diff_norm > 1e-8:
            intervention_direction = full_diff / diff_norm
        else:
            intervention_direction = full_diff
            warnings.warn("Intervention vector has near-zero norm", UserWarning)
        
        results = {
            'intervention_vector': intervention_direction,  # Single rank-1 direction
            'intervention_magnitude': diff_norm,  # How strong the intervention is
            'weight_diff_norms': {name: diff.norm().item() for name, diff in weight_diff.items()},
            'total_diff_norm': diff_norm,
            'num_parameters_changed': sum(1 for d in weight_diff.values() if d.norm() > 1e-6)
        }
        
        # Per-layer intervention vectors (each is rank-1 for that layer)
        if return_per_layer:
            per_layer_vectors = {}
            per_layer_magnitudes = {}
            
            for layer_key, layer_params in layer_diffs.items():
                layer_diffs_flat = []
                for param_name, diff in layer_params:
                    layer_diffs_flat.append(diff.view(-1))
                
                if layer_diffs_flat:
                    layer_diff = torch.cat(layer_diffs_flat)
                    layer_norm = layer_diff.norm().item()
                    
                    if layer_norm > 1e-8:
                        per_layer_vectors[layer_key] = layer_diff / layer_norm
                        per_layer_magnitudes[layer_key] = layer_norm
            
            results['per_layer_vectors'] = per_layer_vectors
            results['per_layer_magnitudes'] = per_layer_magnitudes
        
        return results
    
    def apply_intervention(
        self,
        model,
        intervention_vector: torch.Tensor,
        scale: float = 1.0,
        parameter_mapping: Optional[Dict[str, slice]] = None
    ):
        """
        🧪 EXPERIMENTAL: Apply an intervention vector to a model.
        
        ⚠️ DANGER: This directly modifies model weights! Use on copies only!
        
        Args:
            model: Model to modify (use a copy!)
            intervention_vector: The intervention direction
            scale: How much of the intervention to apply
            parameter_mapping: Optional mapping of parameter names to vector slices
            
        WARNING: This is a destructive operation that permanently changes weights.
        """
        warnings.warn(
            "apply_intervention WILL MODIFY MODEL WEIGHTS. Ensure you're using a copy!",
            category=UserWarning
        )
        
        # This would need proper implementation with parameter mapping
        raise NotImplementedError(
            "apply_intervention is not yet implemented. "
            "This is a placeholder for future intervention experiments."
        )


# ============= EXPERIMENTAL UTILITIES =============

def validate_model_compatibility(model) -> Dict[str, bool]:
    """
    Check if a model is compatible with experimental interventions.
    
    Returns:
        Dictionary of compatibility flags for different intervention types
    """
    compatibility = {
        'has_layers': hasattr(model, 'model') and hasattr(model.model, 'layers'),
        'has_attention': False,
        'has_o_proj': False,
        'architecture': 'unknown'
    }
    
    if compatibility['has_layers']:
        try:
            layer = model.model.layers[0]
            compatibility['has_attention'] = hasattr(layer, 'self_attn')
            if compatibility['has_attention']:
                compatibility['has_o_proj'] = hasattr(layer.self_attn, 'o_proj')
        except:
            pass
    
    # Detect architecture
    if hasattr(model, 'config'):
        if hasattr(model.config, 'model_type'):
            compatibility['architecture'] = model.config.model_type
    
    return compatibility


# ============= SAFETY WARNINGS =============

def experimental_warning_decorator(func):
    """Decorator that adds experimental warning to functions."""
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"Calling experimental method: {func.__name__}. "
            "This is beta functionality that may change or have unexpected behavior.",
            category=UserWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)
    wrapper.__doc__ = f"🧪 EXPERIMENTAL: {func.__doc__}" if func.__doc__ else "🧪 EXPERIMENTAL METHOD"
    return wrapper


# Auto-warn on class instantiation
original_init = ExperimentalInterventions.__init__
@experimental_warning_decorator
def warned_init(self):
    original_init(self)
ExperimentalInterventions.__init__ = warned_init


if __name__ == "__main__":
    print("=" * 70)
    print("FUTURESTUDIES: Experimental Beta Methods")
    print("=" * 70)
    print("\n⚠️  WARNING: This module contains experimental features!")
    print("These methods are in beta and may:")
    print("  - Have unexpected behavior")
    print("  - Change significantly in future versions")
    print("  - Require careful understanding before use")
    print("  - Potentially damage model performance if misused")
    print("\nAlways work with model copies, never originals!")
    print("\nAvailable experimental classes:")
    print("  - ExperimentalInterventions: Head freezing and intervention vectors")
    print("\nExample usage:")
    print("  from FutureStudies import ExperimentalInterventions")
    print("  exp = ExperimentalInterventions()  # Will show warning")
    print("  hooks = exp.freeze_attention_heads(model_copy, {0: [1, 2]})")
    print("  # ... run experiments ...")
    print("  exp.remove_all_hooks()  # Critical cleanup!")
    print("=" * 70)