#!/usr/bin/env python3
"""Quick test for Fisher metric fixes."""

import torch
from BombshellMetrics import BombshellMetrics

# Create a simple test case with fake Fisher data
metrics = BombshellMetrics()

# Simulate Fisher EMA data with various dtypes to test dtype handling
print("Creating test Fisher EMA data...")
metrics.fisher_ema = {
    'math_test_param': torch.randn(1000, 768).abs(),  # Default dtype (float32)
    'math_test_param2': torch.randn(100, 100).abs().to(torch.int32),  # Int dtype (should be converted)
    'general_test_param': torch.randn(1000, 768).abs(),
    'general_test_param2': torch.randn(100, 100).abs(),
}

print("\n1. Testing get_fisher_pruning_masks...")
try:
    masks = metrics.get_fisher_pruning_masks(task='math', sparsity=0.5)
    if 'error' in masks:
        print(f"   ❌ Error: {masks['error']}")
    else:
        print(f"   ✅ Success! Generated {len(masks)} masks")
        for name, mask in masks.items():
            print(f"      {name}: dtype={mask.dtype}, shape={mask.shape}")
except Exception as e:
    print(f"   ❌ Failed with: {e}")

print("\n2. Testing get_top_fisher_directions...")
try:
    top_coords = metrics.get_top_fisher_directions(
        task='math',
        fisher_type='ema',
        top_k_per_param=100,
        percentile=95.0
    )
    print(f"   ✅ Success! Got {len(top_coords)} parameter masks")
    for name, mask in top_coords.items():
        print(f"      {name}: selected {mask.sum().item()} coordinates")
except Exception as e:
    print(f"   ❌ Failed with: {e}")

print("\n3. Testing compute_fisher_overlap...")
try:
    masks1 = metrics.get_fisher_pruning_masks(task='math', sparsity=0.5)
    masks2 = metrics.get_fisher_pruning_masks(task='general', sparsity=0.5)

    if 'error' not in masks1 and 'error' not in masks2:
        result = metrics.compute_fisher_overlap(masks1, masks2)
        print(f"   ✅ Success! Overlap: {result['overlap_percentage']:.2f}%")
    else:
        print(f"   ❌ Could not generate masks")
except Exception as e:
    print(f"   ❌ Failed with: {e}")

print("\n4. Testing scale_by_fisher...")
try:
    # Create fake gradients
    gradients = {
        'test_param': torch.randn(1000, 768),
        'test_param2': torch.randn(100, 100)
    }

    scaled = metrics.scale_by_fisher(gradients, task='math', scaling_type='inverse')
    print(f"   ✅ Success! Scaled {len(scaled)} gradients")
except Exception as e:
    print(f"   ❌ Failed with: {e}")

print("\n" + "="*50)
print("Quick test complete!")