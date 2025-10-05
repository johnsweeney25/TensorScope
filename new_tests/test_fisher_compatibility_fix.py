#!/usr/bin/env python3
"""
Test to verify Fisher storage/retrieval fix in BombshellMetrics.
Tests that Fisher information computed is correctly retrievable.
"""

import torch
import torch.nn as nn
from BombshellMetrics import BombshellMetrics

def test_fisher_storage_retrieval():
    """Test that Fisher information can be stored and retrieved correctly."""

    print("\n" + "="*80)
    print("Testing Fisher Storage/Retrieval Fix")
    print("="*80)

    # Create a simple model that accepts batch format
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(100, 10)
            self.layer1 = nn.Linear(10, 10)
            self.layer2 = nn.Linear(10, 10)

        def forward(self, input_ids, labels=None, **kwargs):
            x = self.embed(input_ids)
            x = x.mean(dim=1)  # Simple pooling
            logits = self.layer2(self.layer1(x))

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(logits, labels)

            # Return in expected format
            return type('Output', (), {'loss': loss, 'logits': logits})()

    model = SimpleModel()

    # Create test batch
    batch = {
        'input_ids': torch.randint(0, 10, (2, 5)),
        'labels': torch.randint(0, 10, (2,))
    }

    # Initialize BombshellMetrics
    bombshell = BombshellMetrics()

    print("\nTest 1: Store Fisher information")
    print("-"*60)

    # Update Fisher EMA for task1 (will compute gradients internally)
    task_name = 'task1'
    bombshell.update_fisher_ema(model, batch, task=task_name)

    # Check what was stored
    fisher_keys = list(bombshell.fisher_ema.keys())
    print(f"Stored {len(fisher_keys)} Fisher keys")
    if fisher_keys:
        print(f"Sample keys: {fisher_keys[:3]}")

    print("\nTest 2: Retrieve Fisher information")
    print("-"*60)

    # Try to get Fisher for task1
    group_fisher = bombshell.get_group_fisher('task1', bias_corrected=False)

    if group_fisher:
        print(f"✅ Successfully retrieved Fisher for 'task1': {len(group_fisher)} entries")
        print(f"Sample retrieved keys: {list(group_fisher.keys())[:3]}")
    else:
        print(f"❌ Failed to retrieve Fisher for 'task1'")
        print(f"Debug: Looking for keys starting with 'task1|'")
        matching = [k for k in bombshell.fisher_ema.keys() if k.startswith('task1')]
        print(f"Debug: Found {len(matching)} keys starting with 'task1'")
        if matching:
            print(f"Debug: Sample matching keys: {matching[:3]}")

    print("\nTest 3: Use compute_fisher_importance")
    print("-"*60)

    # This internally uses get_group_fisher
    importance = bombshell.compute_fisher_importance(model, task='task1')

    if importance:
        print(f"✅ compute_fisher_importance succeeded: {len(importance)} parameters")
    else:
        print(f"❌ compute_fisher_importance returned empty dict")

    print("\nTest 4: Store and retrieve task2")
    print("-"*60)

    # Update Fisher for task2
    bombshell.update_fisher_ema(model, batch, task='task2')

    # Check if task2 Fisher can be retrieved
    fisher2 = bombshell.get_group_fisher('task2', bias_corrected=False)

    if fisher2:
        print(f"✅ Successfully retrieved Fisher for 'task2': {len(fisher2)} entries")
    else:
        print(f"❌ Failed to retrieve Fisher for 'task2'")

    print("\n" + "="*80)
    print("Summary")
    print("="*80)

    # Final validation
    all_passed = True

    # Check if Fisher is stored with correct format
    new_format_keys = [k for k in bombshell.fisher_ema.keys() if '|' in k]
    if new_format_keys:
        print(f"✅ Fisher stored with new format keys (task|param|group): {len(new_format_keys)} keys")
    else:
        print(f"❌ No new format keys found")
        all_passed = False

    # Check if retrieval works for both tasks
    if group_fisher and fisher2:
        print(f"✅ Fisher retrieval works correctly for both tasks")
    else:
        print(f"❌ Fisher retrieval still has issues")
        all_passed = False

    return all_passed

if __name__ == "__main__":
    success = test_fisher_storage_retrieval()
    if success:
        print("\n🎉 Fisher storage/retrieval fix verified!")
    else:
        print("\n❌ Fisher storage/retrieval still has issues")