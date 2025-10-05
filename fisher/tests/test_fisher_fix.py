#!/usr/bin/env python3
"""
Test script to verify Fisher metrics fixes
"""
import torch
import torch.nn as nn
from BombshellMetrics import BombshellMetrics

# Create a simple model for testing that mimics LLM interface
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 8)  # Vocab size 100, embed dim 8
        self.fc1 = nn.Linear(8, 5)
        self.fc2 = nn.Linear(5, 10)  # Output vocab size

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # Simple forward pass that returns loss
        batch_size, seq_len = input_ids.shape
        x = self.embed(input_ids)  # (batch_size, seq_len, embed_dim)
        x = x.view(batch_size * seq_len, -1)  # Flatten for linear layer
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        logits = logits.view(batch_size, seq_len, -1)  # Reshape back

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        # Return object similar to transformers model output
        class Output:
            pass

        output = Output()
        output.loss = loss
        output.logits = logits
        return output

def test_fisher_metrics():
    """Test that Fisher metrics work with correct task names"""
    print("Testing Fisher metrics fixes...")

    # Initialize metrics
    bombshell = BombshellMetrics()

    # Create model and batches
    model = SimpleModel()

    # Create mock batches with proper structure
    batch1 = {
        'input_ids': torch.randint(0, 100, (4, 10)),
        'attention_mask': torch.ones(4, 10),
        'labels': torch.randint(0, 10, (4, 10))  # Match vocab size
    }

    batch2 = {
        'input_ids': torch.randint(0, 100, (4, 10)),
        'attention_mask': torch.ones(4, 10),
        'labels': torch.randint(0, 10, (4, 10))  # Match vocab size
    }

    # Test 1: update_fisher_ema with task names
    print("\n1. Testing update_fisher_ema...")
    try:
        # Compute Fisher EMA for 'task1'
        bombshell.update_fisher_ema(model, batch1, task='task1')
        print("   ✓ Fisher EMA computed for 'task1'")

        # Compute Fisher EMA for 'task2'
        bombshell.update_fisher_ema(model, batch2, task='task2')
        print("   ✓ Fisher EMA computed for 'task2'")

        # Check that data was stored
        task1_keys = [k for k in bombshell.fisher_ema.keys() if k.startswith('task1_')]
        task2_keys = [k for k in bombshell.fisher_ema.keys() if k.startswith('task2_')]

        print(f"   ✓ Stored {len(task1_keys)} parameters for 'task1'")
        print(f"   ✓ Stored {len(task2_keys)} parameters for 'task2'")

    except Exception as e:
        print(f"   ✗ Error in update_fisher_ema: {e}")
        return False

    # Test 2: compare_task_fisher
    print("\n2. Testing compare_task_fisher...")
    try:
        result = bombshell.compare_task_fisher('task1', 'task2')

        if 'error' in result:
            print(f"   ✗ Error in compare_task_fisher: {result['error']}")
            return False

        print(f"   ✓ Task comparison successful:")
        print(f"     - Divergence: {result['divergence']:.4f}")
        print(f"     - Correlation: {result['correlation']:.4f}")
        print(f"     - Magnitude ratio: {result['magnitude_ratio']:.4f}")

    except Exception as e:
        print(f"   ✗ Error in compare_task_fisher: {e}")
        return False

    # Test 3: get_fisher_pruning_masks
    print("\n3. Testing get_fisher_pruning_masks...")
    try:
        masks_task1 = bombshell.get_fisher_pruning_masks(task='task1', sparsity=0.5)
        masks_task2 = bombshell.get_fisher_pruning_masks(task='task2', sparsity=0.5)

        if 'error' in masks_task1:
            print(f"   ✗ Error in get_fisher_pruning_masks for 'task1': {masks_task1['error']}")
            return False

        if 'error' in masks_task2:
            print(f"   ✗ Error in get_fisher_pruning_masks for 'task2': {masks_task2['error']}")
            return False

        print(f"   ✓ Generated pruning masks for 'task1': {len(masks_task1)} parameters")
        print(f"   ✓ Generated pruning masks for 'task2': {len(masks_task2)} parameters")

    except Exception as e:
        print(f"   ✗ Error in get_fisher_pruning_masks: {e}")
        return False

    # Test 4: compute_fisher_overlap (this had the torch import issue)
    print("\n4. Testing compute_fisher_overlap...")
    try:
        overlap = bombshell.compute_fisher_overlap(masks_task1, masks_task2)

        if 'error' in overlap:
            print(f"   ✗ Error in compute_fisher_overlap: {overlap['error']}")
            return False

        print(f"   ✓ Overlap computation successful:")
        print(f"     - Overlap percentage: {overlap['overlap_percentage']:.2f}%")
        print(f"     - Per-layer overlaps: {len(overlap['per_layer_overlap'])} layers")

    except Exception as e:
        print(f"   ✗ Error in compute_fisher_overlap: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ All Fisher metrics tests passed!")
    return True

if __name__ == "__main__":
    success = test_fisher_metrics()
    exit(0 if success else 1)
