#!/usr/bin/env python3
"""
Metric Compatibility Checker
============================

This module checks model capabilities to determine which metrics can be computed.
It helps avoid wasting computation time on metrics that will fail.

Author: Claude
Date: 2024
"""

import torch
import logging
from typing import Dict, List, Any, Optional
from transformers import PreTrainedModel

logger = logging.getLogger(__name__)


class MetricCompatibilityChecker:
    """
    Check which metrics are compatible with a given model.

    This helps identify upfront which metrics will work with a model
    to avoid runtime failures and ensure statistical validity.
    """

    def __init__(self):
        self.compatibility_cache = {}

    def check_model_capabilities(self, model: PreTrainedModel) -> Dict[str, bool]:
        """
        Check what capabilities a model has.

        Args:
            model: The model to check

        Returns:
            Dictionary of capability names to boolean values
        """
        capabilities = {}

        # Check attention output capability
        capabilities['attention_output'] = self._check_attention_output(model)

        # Check gradient capability
        capabilities['gradients'] = self._check_gradient_support(model)

        # Check if model has specific attributes
        capabilities['has_lm_head'] = hasattr(model, 'lm_head')
        capabilities['has_embeddings'] = hasattr(model, 'get_input_embeddings')

        # Check attention implementation
        capabilities['attention_impl'] = self._get_attention_implementation(model)

        # Check if model supports different precisions
        capabilities['supports_fp16'] = self._check_fp16_support(model)
        capabilities['supports_bf16'] = self._check_bf16_support(model)

        # Check model size (for memory-intensive metrics)
        capabilities['model_size_gb'] = self._get_model_size_gb(model)
        capabilities['is_large_model'] = capabilities['model_size_gb'] > 10  # > 10GB

        return capabilities

    def _check_attention_output(self, model: PreTrainedModel) -> bool:
        """
        Check if model can output attention weights.

        This is critical for attention-based metrics like attention_entropy.
        """
        try:
            # Create a dummy input
            dummy_input = torch.randint(0, 1000, (1, 10))

            # Try to get attention with eager mode
            original_impl = None
            if hasattr(model, 'config'):
                # Store original implementation
                if hasattr(model.config, 'attn_implementation'):
                    original_impl = model.config.attn_implementation
                    model.config.attn_implementation = 'eager'
                elif hasattr(model.config, '_attn_implementation'):
                    original_impl = model.config._attn_implementation
                    model.config._attn_implementation = 'eager'

            # Try forward pass with attention output
            with torch.no_grad():
                try:
                    outputs = model(dummy_input, output_attentions=True, return_dict=True)
                    has_attention = hasattr(outputs, 'attentions') and outputs.attentions is not None
                except Exception:
                    has_attention = False

            # Restore original implementation
            if original_impl is not None:
                if hasattr(model.config, 'attn_implementation'):
                    model.config.attn_implementation = original_impl
                elif hasattr(model.config, '_attn_implementation'):
                    model.config._attn_implementation = original_impl

            return has_attention

        except Exception as e:
            logger.debug(f"Could not check attention output: {e}")
            return False

    def _check_gradient_support(self, model: PreTrainedModel) -> bool:
        """Check if model supports gradient computation."""
        try:
            # Check if any parameters require grad
            return any(p.requires_grad for p in model.parameters())
        except Exception:
            return False

    def _get_attention_implementation(self, model: PreTrainedModel) -> str:
        """Get the attention implementation type."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'attn_implementation'):
                return model.config.attn_implementation
            elif hasattr(model.config, '_attn_implementation'):
                return model.config._attn_implementation
        return 'unknown'

    def _check_fp16_support(self, model: PreTrainedModel) -> bool:
        """Check if model supports FP16."""
        try:
            # Check if model has any FP16 parameters
            return any(p.dtype == torch.float16 for p in model.parameters())
        except Exception:
            return False

    def _check_bf16_support(self, model: PreTrainedModel) -> bool:
        """Check if model supports BF16."""
        try:
            # Check if model has any BF16 parameters
            return any(p.dtype == torch.bfloat16 for p in model.parameters())
        except Exception:
            return False

    def _get_model_size_gb(self, model: PreTrainedModel) -> float:
        """Get model size in GB."""
        try:
            total_params = sum(p.numel() * p.element_size() for p in model.parameters())
            return total_params / (1024 ** 3)
        except Exception:
            return 0.0

    def get_compatible_metrics(self, model: PreTrainedModel) -> Dict[str, Dict[str, Any]]:
        """
        Get list of metrics compatible with the model.

        Returns:
            Dictionary mapping metric names to compatibility info
        """
        capabilities = self.check_model_capabilities(model)

        compatible_metrics = {
            # Attention metrics
            'compute_attention_entropy': {
                'compatible': capabilities['attention_output'],
                'reason': 'Requires attention weights output (eager mode)',
                'category': 'attention'
            },
            'compute_attention_drift': {
                'compatible': capabilities['attention_output'],
                'reason': 'Requires attention weights output (eager mode)',
                'category': 'attention'
            },

            # Gradient metrics
            'compute_gradient_conflict': {
                'compatible': capabilities['gradients'],
                'reason': 'Requires gradient computation',
                'category': 'gradient',
                'memory_intensive': True
            },
            'compute_gradient_snr': {
                'compatible': capabilities['gradients'],
                'reason': 'Requires gradient computation',
                'category': 'gradient'
            },

            # Basic metrics (should work for all models)
            'compute_activation_pattern_stability': {
                'compatible': True,
                'reason': 'Basic activation analysis',
                'category': 'activation'
            },
            'compute_weight_statistics': {
                'compatible': True,
                'reason': 'Basic weight analysis',
                'category': 'weight'
            },

            # Memory-intensive metrics
            'compute_hessian_eigenvalues': {
                'compatible': not capabilities['is_large_model'],
                'reason': 'Very memory intensive, not suitable for large models',
                'category': 'hessian',
                'memory_intensive': True
            }
        }

        return compatible_metrics

    def print_compatibility_report(self, model: PreTrainedModel):
        """
        Print a detailed compatibility report for the model.
        """
        print("\n" + "="*60)
        print("MODEL COMPATIBILITY REPORT")
        print("="*60)

        capabilities = self.check_model_capabilities(model)

        print("\nModel Capabilities:")
        print("-"*40)
        for cap, value in capabilities.items():
            if isinstance(value, bool):
                status = "✓" if value else "✗"
                print(f"  {status} {cap}: {value}")
            else:
                print(f"  • {cap}: {value}")

        metrics = self.get_compatible_metrics(model)

        print("\nMetric Compatibility:")
        print("-"*40)

        # Group by category
        categories = {}
        for metric, info in metrics.items():
            cat = info['category']
            if cat not in categories:
                categories[cat] = []
            categories[cat].append((metric, info))

        for category, metric_list in categories.items():
            print(f"\n{category.upper()} Metrics:")
            for metric, info in metric_list:
                status = "✓" if info['compatible'] else "✗"
                memory_flag = " [MEMORY INTENSIVE]" if info.get('memory_intensive') else ""
                print(f"  {status} {metric}{memory_flag}")
                if not info['compatible']:
                    print(f"     Reason: {info['reason']}")

        # Summary
        total = len(metrics)
        compatible = sum(1 for m in metrics.values() if m['compatible'])
        print("\n" + "="*60)
        print(f"SUMMARY: {compatible}/{total} metrics compatible")
        print("="*60 + "\n")


def check_model_before_analysis(model: PreTrainedModel,
                               required_metrics: Optional[List[str]] = None) -> bool:
    """
    Quick check before running analysis.

    Args:
        model: Model to check
        required_metrics: List of metrics that must be compatible

    Returns:
        True if all required metrics are compatible
    """
    checker = MetricCompatibilityChecker()

    if required_metrics is None:
        # Just print report
        checker.print_compatibility_report(model)
        return True

    # Check specific metrics
    compatible = checker.get_compatible_metrics(model)

    incompatible = []
    for metric in required_metrics:
        if metric in compatible and not compatible[metric]['compatible']:
            incompatible.append(metric)

    if incompatible:
        logger.error(f"The following required metrics are not compatible: {incompatible}")
        checker.print_compatibility_report(model)
        return False

    return True


if __name__ == "__main__":
    # Example usage
    from transformers import AutoModelForCausalLM

    print("Loading model for compatibility check...")
    model_name = "gpt2"  # Use small model for testing
    model = AutoModelForCausalLM.from_pretrained(model_name)

    checker = MetricCompatibilityChecker()
    checker.print_compatibility_report(model)

    # Check specific metrics
    print("\nChecking specific metrics...")
    required = ['compute_attention_entropy', 'compute_gradient_conflict']
    if check_model_before_analysis(model, required):
        print("✓ All required metrics are compatible!")
    else:
        print("✗ Some required metrics are not compatible.")