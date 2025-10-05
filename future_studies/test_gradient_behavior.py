#!/usr/bin/env python3
"""
Test script to verify gradient behavior in attention circuit freezing.
Demonstrates proper usage of gradient debugging utilities.
"""

import torch
import torch.nn as nn
import copy
from transformers import AutoModel, AutoTokenizer
import logging

# Set up logging to see debug output
logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

# Import our module
from attention_circuit_freezing import (
    AttentionCircuitFreezer,
    InterventionConfig,
    CircuitType,
    FreezeType
)


def test_gradient_modes(model_name="gpt2", text="The capital of France is"):
    """
    Test both stopgrad and STE modes to verify gradient behavior.
    """
    print(f"\n{'='*80}")
    print(f"Testing gradient behavior with {model_name}")
    print(f"{'='*80}\n")

    # Load model and tokenizer
    print(f"Loading model {model_name}...")
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize input
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids

    # Test configurations
    test_configs = [
        ("stopgrad", "Gradients should be blocked"),
        ("ste", "Gradients should flow through (STE)")
    ]

    results = {}

    for backward_mode, description in test_configs:
        print(f"\n{'-'*60}")
        print(f"Testing {backward_mode} mode: {description}")
        print(f"{'-'*60}\n")

        # Create model copy
        model_copy = copy.deepcopy(model)
        model_copy.train()  # Ensure in training mode

        # Create freezer with debug enabled
        freezer = AttentionCircuitFreezer(debug_gradients=True)

        # Create intervention config
        config = InterventionConfig(
            layer_indices=[0, 1],  # First two layers
            head_indices=[0, 1, 2],  # First three heads
            circuit=CircuitType.QK,
            freeze_type=FreezeType.ZERO,
            backward_mode=backward_mode
        )

        # Apply intervention
        print(f"Applying {config.circuit.value} circuit freezing...")
        hooks = freezer.freeze_circuits(model_copy, config)

        # Create test input that requires gradients
        test_input = input_ids.clone().detach()

        # Check gradient behavior
        print("\nChecking gradient behavior...")
        grad_stats = freezer.check_gradient_behavior(
            model_copy,
            test_input,
            config,
            target_param_name="layers.0.self_attn"  # Check first layer attention
        )

        # Store results
        results[backward_mode] = grad_stats

        # Clean up hooks
        freezer.remove_hooks(hooks)

        # Print summary
        print(f"\nSummary for {backward_mode}:")
        print(f"  - Validation: {'PASSED' if grad_stats['validation_passed'] else 'FAILED'}")
        print(f"  - Loss value: {grad_stats['loss_value']:.6f}")
        print(f"  - Input grad norm: {grad_stats['input_grad_norm']:.6f}")

        if grad_stats['parameter_grads']:
            print(f"\n  Parameter gradients:")
            for param_name, param_stats in list(grad_stats['parameter_grads'].items())[:3]:
                short_name = '.'.join(param_name.split('.')[-2:])
                print(f"    {short_name}: norm={param_stats['norm']:.6f}, "
                      f"nonzero={param_stats['nonzero_pct']:.1f}%")

    # Compare results
    print(f"\n{'='*60}")
    print("COMPARISON BETWEEN MODES")
    print(f"{'='*60}\n")

    if 'stopgrad' in results and 'ste' in results:
        # Compare key metrics
        sg_stats = results['stopgrad']
        ste_stats = results['ste']

        print("Gradient Norms Comparison:")
        params_compared = set(sg_stats['parameter_grads'].keys()) & set(ste_stats['parameter_grads'].keys())

        for param_name in list(params_compared)[:5]:
            short_name = '.'.join(param_name.split('.')[-2:])
            sg_norm = sg_stats['parameter_grads'][param_name]['norm']
            ste_norm = ste_stats['parameter_grads'][param_name]['norm']
            ratio = ste_norm / (sg_norm + 1e-10)

            print(f"  {short_name:20s}: "
                  f"stopgrad={sg_norm:.6f}, "
                  f"ste={ste_norm:.6f}, "
                  f"ratio={ratio:.2f}x")

        print(f"\nExpected behavior:")
        print(f"  - stopgrad should have near-zero gradients for frozen parameters")
        print(f"  - ste should have non-zero gradients (preserved flow)")
        print(f"  - Ratio should be >> 1 for affected parameters")

    return results


def test_intervention_effects(model_name="gpt2", text="The capital of France is"):
    """
    Test that interventions actually change model outputs.
    """
    print(f"\n{'='*80}")
    print(f"Testing intervention effects with {model_name}")
    print(f"{'='*80}\n")

    # Load model
    model_baseline = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    inputs = tokenizer(text, return_tensors="pt", padding=True)
    input_ids = inputs.input_ids

    # Test different freeze types
    freeze_types = [FreezeType.ZERO, FreezeType.MEAN, FreezeType.NOISE]

    for freeze_type in freeze_types:
        print(f"\n{'-'*40}")
        print(f"Testing {freeze_type.value} freezing")
        print(f"{'-'*40}\n")

        # Create model copy for intervention
        model_intervened = copy.deepcopy(model_baseline)

        # Create freezer
        freezer = AttentionCircuitFreezer(debug_gradients=True)

        # Apply intervention
        config = InterventionConfig(
            layer_indices=[0, 1, 2],  # First three layers
            head_indices=list(range(6)),  # First 6 heads
            circuit=CircuitType.BOTH,  # Freeze both QK and OV
            freeze_type=freeze_type,
            backward_mode="stopgrad"
        )

        hooks = freezer.freeze_circuits(model_intervened, config)

        # Verify intervention effect
        effect_stats = freezer.verify_intervention_effect(
            model_baseline,
            model_intervened,
            input_ids,
            config,
            tolerance=0.001
        )

        # Clean up
        freezer.remove_hooks(hooks)

        print(f"\nResults for {freeze_type.value}:")
        print(f"  - Max absolute diff: {effect_stats['max_absolute_diff']:.6f}")
        print(f"  - Mean absolute diff: {effect_stats['mean_absolute_diff']:.6f}")
        print(f"  - Effect detected: {'YES' if effect_stats['output_changed'] else 'NO'}")


def main():
    """
    Run all gradient behavior tests.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Test gradient behavior in attention circuit freezing')
    parser.add_argument('--model', default='gpt2', help='Model to test (default: gpt2)')
    parser.add_argument('--text', default='The capital of France is', help='Test input text')
    parser.add_argument('--test', choices=['gradients', 'effects', 'both'], default='both',
                        help='Which tests to run')

    args = parser.parse_args()

    if args.test in ['gradients', 'both']:
        test_gradient_modes(args.model, args.text)

    if args.test in ['effects', 'both']:
        test_intervention_effects(args.model, args.text)

    print(f"\n{'='*80}")
    print("All tests completed!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()