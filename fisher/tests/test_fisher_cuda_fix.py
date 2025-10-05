#!/usr/bin/env python3
"""
Test script to verify Fisher weighted damage CUDA error fixes.
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from ModularityMetrics import ExtendedModularityMetrics
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fisher_damage():
    """Test Fisher weighted damage functions with token ID validation."""

    print("=" * 60)
    print("Testing Fisher Weighted Damage CUDA Error Fixes")
    print("=" * 60)

    # Load a small model for testing
    model_name = "Qwen/Qwen2.5-0.5B"  # Small model for testing
    print(f"\nLoading model: {model_name}")

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(f"Vocabulary size: {model.config.vocab_size}")

    # Create test batches
    print("\nCreating test batches...")

    # Math-like batch
    math_texts = [
        "The derivative of x^2 is 2x",
        "The integral of 1/x is ln(x)",
        "The solution to x^2 - 4 = 0 is x = ±2",
        "The limit of (1 + 1/n)^n as n approaches infinity is e"
    ]

    # General text batch
    general_texts = [
        "The weather today is sunny and warm",
        "I enjoy reading books in my free time",
        "The cat sat on the mat and purred",
        "Technology has changed our daily lives"
    ]

    # Tokenize
    math_batch = tokenizer(
        math_texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    general_batch = tokenizer(
        general_texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors="pt"
    )

    # Move to device
    math_batch = {k: v.to(device) for k, v in math_batch.items()}
    general_batch = {k: v.to(device) for k, v in general_batch.items()}

    print(f"Math batch shape: {math_batch['input_ids'].shape}")
    print(f"General batch shape: {general_batch['input_ids'].shape}")

    # Check for any tokens that might exceed vocab size
    max_token_math = math_batch['input_ids'].max().item()
    max_token_general = general_batch['input_ids'].max().item()
    print(f"Max token ID in math batch: {max_token_math}")
    print(f"Max token ID in general batch: {max_token_general}")

    if max_token_math >= model.config.vocab_size or max_token_general >= model.config.vocab_size:
        print("WARNING: Token IDs exceed vocabulary size! This would cause CUDA errors.")

    # Initialize metrics calculator
    metrics = ExtendedModularityMetrics(seed=42)

    # Test 1: compute_fisher_weighted_damage
    print("\n" + "=" * 40)
    print("Test 1: compute_fisher_weighted_damage")
    print("=" * 40)

    try:
        result = metrics.compute_fisher_weighted_damage(
            model,
            math_batch,
            general_batch,
            n_fisher_samples=2  # Small number for testing
        )

        if "error" in result:
            print(f"ERROR: {result['error']}")
        else:
            print(f"✓ SUCCESS: Damage from general to math = {result['damage_A_from_B']:.6f}")
            print(f"  Total raw damage: {result['total_damage_raw']:.6f}")
            print(f"  Parameter count: {result['param_count']}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 2: compute_fisher_damage_with_asymmetry
    print("\n" + "=" * 40)
    print("Test 2: compute_fisher_damage_with_asymmetry")
    print("=" * 40)

    try:
        result = metrics.compute_fisher_damage_with_asymmetry(
            model,
            math_batch,
            general_batch,
            n_fisher_samples=2  # Small number for testing
        )

        if isinstance(result, dict) and 'damage_math_from_general' in result:
            print(f"✓ SUCCESS:")
            print(f"  Damage math from general: {result['damage_math_from_general']:.6f}")
            print(f"  Damage general from math: {result['damage_general_from_math']:.6f}")
            print(f"  Asymmetry: {result['damage_asymmetry']:.6f}")
        else:
            print(f"ERROR: Unexpected result format: {result}")
    except Exception as e:
        print(f"✗ FAILED: {e}")

    # Test 3: Test with deliberately bad token IDs (should be clamped)
    print("\n" + "=" * 40)
    print("Test 3: Testing with invalid token IDs")
    print("=" * 40)

    # Create a batch with invalid token IDs
    bad_batch = {
        'input_ids': torch.tensor([[999999, 1000000, 1000001, 100]], device=device),
        'attention_mask': torch.ones(1, 4, device=device)
    }

    print(f"Created batch with token IDs: {bad_batch['input_ids'][0].tolist()}")
    print(f"These exceed vocab_size ({model.config.vocab_size})")

    try:
        result = metrics.compute_fisher_weighted_damage(
            model,
            math_batch,
            bad_batch,
            n_fisher_samples=1
        )

        if "error" in result:
            print(f"Handled gracefully with error: {result['error']}")
        else:
            print(f"✓ SUCCESS: Function handled invalid tokens (clamped them)")
            print(f"  Damage computed: {result['damage_A_from_B']}")
    except Exception as e:
        print(f"✗ FAILED with exception: {e}")

    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)

if __name__ == "__main__":
    # Set CUDA_LAUNCH_BLOCKING for better error messages
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

    test_fisher_damage()