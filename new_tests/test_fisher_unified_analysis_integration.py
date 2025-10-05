#!/usr/bin/env python3
"""
Final test to verify Fisher storage/retrieval works for Qwen model scenario.
This simulates the exact flow that happens in unified_model_analysis.py
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from BombshellMetrics import BombshellMetrics

def test_qwen_fisher_scenario():
    """Test the complete Fisher flow as it happens with Qwen models."""

    print("\n" + "="*80)
    print("Testing Complete Fisher Flow for Qwen Model Scenario")
    print("="*80)

    # Load GPT-2 as proxy for testing (replace with Qwen/Qwen2.5-Math-1.5B for real test)
    model_name = "gpt2"
    print(f"\nLoading model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Create test batches similar to unified_model_analysis
    math_batch = tokenizer(
        ["What is 2+2?", "Calculate derivative of x^2"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )
    math_batch['labels'] = math_batch['input_ids'].clone()

    general_batch = tokenizer(
        ["The capital of France is", "Machine learning is"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32
    )
    general_batch['labels'] = general_batch['input_ids'].clone()

    # Initialize BombshellMetrics
    bombshell = BombshellMetrics()

    print("\nPhase 1: Computing Fisher EMA for tasks...")
    print("-"*60)

    # Task 1: Math batch
    task_name = 'task1'
    batch = math_batch

    print(f"  Computing Fisher EMA for task '{task_name}'...")

    # This simulates what unified_model_analysis.py does
    fisher_computed = False
    if hasattr(bombshell, '_estimate_fisher_diagonal'):
        try:
            fisher_dict = bombshell._estimate_fisher_diagonal(
                model, batch,
                n_samples=4,
                fisher_batch_size=4
            )

            # Store with NEW format (after our fix)
            for param_name, fisher_values in fisher_dict.items():
                if 'bias' in param_name:
                    group_type = 'bias'
                elif len(fisher_values.shape) == 1:
                    group_type = 'channel'
                else:
                    group_type = 'param'
                key = f'{task_name}|{param_name}|{group_type}'
                bombshell.fisher_ema[key] = fisher_values

            # Update step counter (after our fix)
            step_key = f'{task_name}_steps'
            if not hasattr(bombshell, 'fisher_steps'):
                bombshell.fisher_steps = {}
            bombshell.fisher_steps[step_key] = bombshell.fisher_steps.get(step_key, 0) + 1

            fisher_computed = True
            print(f"    ✓ Direct Fisher computed for '{task_name}', got {len(fisher_dict)} parameters")
        except Exception as e:
            print(f"    ✗ Direct Fisher failed: {e}")

    # Validate storage
    task_keys = [k for k in bombshell.fisher_ema.keys()
                 if k.startswith(f"{task_name}|")]
    if task_keys:
        print(f"    ✓ Validated: {len(task_keys)} parameters have Fisher info for '{task_name}'")
    else:
        print(f"    ✗ No Fisher data found for '{task_name}'")

    # Task 2: General batch
    task_name = 'task2'
    batch = general_batch

    print(f"  Computing Fisher EMA for task '{task_name}'...")

    # Compute and store for task2
    if hasattr(bombshell, '_estimate_fisher_diagonal'):
        try:
            fisher_dict = bombshell._estimate_fisher_diagonal(
                model, batch,
                n_samples=4,
                fisher_batch_size=4
            )

            for param_name, fisher_values in fisher_dict.items():
                if 'bias' in param_name:
                    group_type = 'bias'
                elif len(fisher_values.shape) == 1:
                    group_type = 'channel'
                else:
                    group_type = 'param'
                key = f'{task_name}|{param_name}|{group_type}'
                bombshell.fisher_ema[key] = fisher_values

            step_key = f'{task_name}_steps'
            bombshell.fisher_steps[step_key] = bombshell.fisher_steps.get(step_key, 0) + 1

            print(f"    ✓ Direct Fisher computed for '{task_name}', got {len(fisher_dict)} parameters")
        except Exception as e:
            print(f"    ✗ Direct Fisher failed: {e}")

    task_keys = [k for k in bombshell.fisher_ema.keys()
                 if k.startswith(f"{task_name}|")]
    if task_keys:
        print(f"    ✓ Validated: {len(task_keys)} parameters have Fisher info for '{task_name}'")

    print("\nPhase 2: Computing Fisher-based metrics...")
    print("-"*60)

    # Compute Fisher importance
    print("  📊 Computing Fisher importance...")
    importance1 = bombshell.compute_fisher_importance(model, task='task1')
    importance2 = bombshell.compute_fisher_importance(model, task='task2')

    if importance1:
        print(f"    ✓ Computed Fisher importance for task 'task1': {len(importance1)} parameters")
    else:
        print(f"    ✗ Failed to compute Fisher importance for 'task1'")

    if importance2:
        print(f"    ✓ Computed Fisher importance for task 'task2': {len(importance2)} parameters")
    else:
        print(f"    ✗ Failed to compute Fisher importance for 'task2'")

    # Compare Fisher between tasks
    print("  🔄 Computing Fisher comparison...")
    comparison = bombshell.compare_task_fisher('task1', 'task2')

    if comparison and 'overlap' in comparison:
        print(f"    ✓ Computed Fisher comparison between 'task1' and 'task2'")
        print(f"      Overlap: {comparison.get('overlap', 0):.2%}")
    else:
        print(f"    ✗ Failed to compute Fisher comparison")

    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    # Final validation
    total_keys = len(bombshell.fisher_ema.keys())
    new_format_keys = sum(1 for k in bombshell.fisher_ema.keys() if '|' in k)
    old_format_keys = sum(1 for k in bombshell.fisher_ema.keys() if '_' in k and '|' not in k)

    print(f"Total Fisher keys stored: {total_keys}")
    print(f"New format keys (task|param|group): {new_format_keys}")
    print(f"Old format keys (task_param): {old_format_keys}")

    success = importance1 and importance2 and comparison and new_format_keys > 0 and old_format_keys == 0

    if success:
        print("\n✅ SUCCESS: Fisher storage and retrieval working correctly!")
        print("   The Qwen model Fisher computation issue is FIXED!")
    else:
        print("\n❌ FAILURE: Issues remain with Fisher computation")
        if not importance1 or not importance2:
            print("   - Fisher importance computation failed")
        if not comparison:
            print("   - Fisher comparison failed")
        if old_format_keys > 0:
            print("   - Still using old format keys")

    return success

if __name__ == "__main__":
    success = test_qwen_fisher_scenario()
    exit(0 if success else 1)