#!/usr/bin/env python3
"""Debug scale_by_fisher gradient computation issue."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_gradient_computation():
    """Test gradient computation with a real model."""

    print("Loading model...")
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    print("\nCreating batch...")
    text = "The quick brown fox"
    batch = tokenizer(text, return_tensors='pt', padding=True)

    # Add labels if not present
    if 'labels' not in batch:
        batch['labels'] = batch['input_ids'].clone()

    print(f"Batch keys: {batch.keys()}")
    print(f"Input shape: {batch['input_ids'].shape}")

    print("\nTesting gradient computation...")

    # Method 1: Simple gradient computation
    print("\n1. Simple gradient test:")
    model.train()
    model.zero_grad()

    try:
        outputs = model(**batch)
        print(f"   Loss: {outputs.loss}")

        if outputs.loss is not None:
            outputs.loss.backward()

            grads_found = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grads_found += 1

            print(f"   Gradients found: {grads_found}/{len(list(model.named_parameters()))}")
        else:
            print("   ERROR: Loss is None!")
    except Exception as e:
        print(f"   ERROR: {e}")

    model.zero_grad()

    # Method 2: With requires_grad explicitly set
    print("\n2. With explicit requires_grad:")

    # Check initial state
    params_requiring_grad = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"   Initial params requiring grad: {params_requiring_grad}")

    # Enable all gradients
    for param in model.parameters():
        param.requires_grad_(True)

    params_requiring_grad = sum(1 for p in model.parameters() if p.requires_grad)
    print(f"   After enabling: {params_requiring_grad}")

    model.train()
    model.zero_grad()

    try:
        with torch.enable_grad():
            outputs = model(**batch)
            print(f"   Loss: {outputs.loss}")

            if outputs.loss is not None:
                outputs.loss.backward()

                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.clone().detach()

                print(f"   Gradients computed: {len(gradients)}/{len(list(model.named_parameters()))}")

                if gradients:
                    # Show a few gradient norms
                    for i, (name, grad) in enumerate(list(gradients.items())[:3]):
                        print(f"     {name}: norm={grad.norm().item():.6f}")
                else:
                    print("   ERROR: No gradients despite loss.backward()!")
            else:
                print("   ERROR: Loss is None!")
    except Exception as e:
        print(f"   ERROR: {e}")
        import traceback
        traceback.print_exc()

    model.eval()
    model.zero_grad()

    print("\n" + "="*50)
    print("Debug complete!")

if __name__ == "__main__":
    test_gradient_computation()