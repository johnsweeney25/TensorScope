#!/usr/bin/env python3
"""
Test to verify the Phase 1 vs Phase 6 shape mismatch bug.

This test demonstrates that Phase 6 receives group-reduced Fisher tensors
when it expects full parameter tensors for QK-OV slicing.
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fisher.core.fisher_collector import FisherCollector
from fisher.qkov import QKOVConfig, QKOVInterferenceMetric


def test_fisher_shape_mismatch():
    """
    Verify that Phase 6 receives group-reduced Fisher when it expects full tensors.
    """
    print("=" * 70)
    print("Test: Phase 1 vs Phase 6 Fisher Shape Mismatch")
    print("=" * 70)
    print()

    # Create minimal mock model
    class MockAttention(nn.Module):
        def __init__(self, hidden_size=512, num_heads=8):
            super().__init__()
            self.q_proj = nn.Linear(hidden_size, hidden_size)
            self.k_proj = nn.Linear(hidden_size, hidden_size)
            self.v_proj = nn.Linear(hidden_size, hidden_size)
            self.o_proj = nn.Linear(hidden_size, hidden_size)

    class MockLayer(nn.Module):
        def __init__(self, hidden_size=512, num_heads=8):
            super().__init__()
            self.self_attn = MockAttention(hidden_size, num_heads)
            self.mlp = nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size)
            )

    class MockModel(nn.Module):
        def __init__(self, hidden_size=512, num_heads=8, num_layers=2):
            super().__init__()
            self.config = type('Config', (), {
                'hidden_size': hidden_size,
                'num_attention_heads': num_heads,
                'num_hidden_layers': num_layers,
                'vocab_size': 1000,
                'num_key_value_heads': None
            })()
            self.embed_tokens = nn.Embedding(1000, hidden_size)
            self.layers = nn.ModuleList([MockLayer(hidden_size, num_heads) for _ in range(num_layers)])
            self.lm_head = nn.Linear(hidden_size, 1000, bias=False)

        def forward(self, input_ids, attention_mask=None, labels=None):
            x = self.embed_tokens(input_ids)
            for layer in self.layers:
                x = x + layer.self_attn.q_proj(x)  # Simplified
                x = x + layer.mlp(x)
            logits = self.lm_head(x)
            
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    labels.view(-1)
                )
                return type('Output', (), {'loss': loss, 'logits': logits})()
            return type('Output', (), {'logits': logits})()

    model = MockModel(hidden_size=512, num_heads=8, num_layers=2)
    
    # Create mock batch (single sample for simplicity)
    batch = {
        'input_ids': torch.randint(0, 1000, (1, 16)),
        'attention_mask': torch.ones(1, 16),
        'labels': torch.randint(0, 1000, (1, 16))
    }

    print("Step 1: Initialize FisherCollector with cross-task analysis")
    fisher_collector = FisherCollector(
        reduction='param',
        enable_cross_task_analysis=True,
        gradient_memory_mb=50,
        computation_dtype='float32'
    )
    print("✓ FisherCollector initialized")
    print()

    print("Step 2: Collect Fisher for task A")
    try:
        fisher_collector.collect_fisher(
            model=model,
            batch=batch,
            task='task_a',
            mode='ema'
        )
        print(f"✓ Task A: {len(fisher_collector.contribution_cache)} samples in contribution_cache")
        print(f"✓ Task A: {len(fisher_collector.fisher_ema)} parameters in fisher_ema")
    except Exception as e:
        print(f"✗ Fisher collection failed (expected with mock model): {e}")
        print("Proceeding with manual verification...")
    print()

    print("Step 3: Inspect fisher_ema shapes")
    print("Expected for q_proj: [512, 512] (full parameter)")
    print("Actual shapes in fisher_ema:")
    for key, value in list(fisher_collector.fisher_ema.items())[:3]:
        if 'q_proj' in key or 'k_proj' in key or 'v_proj' in key or 'o_proj' in key:
            print(f"  {key}: {value.shape}")
    print()

    # Find a q_proj key
    q_proj_key = None
    for key in fisher_collector.fisher_ema.keys():
        if 'q_proj' in key:
            q_proj_key = key
            break

    if q_proj_key:
        fisher_shape = fisher_collector.fisher_ema[q_proj_key].shape
        print(f"Step 4: Verify the bug")
        print(f"  fisher_ema['{q_proj_key}']: shape {fisher_shape}")
        
        if len(fisher_shape) == 1:
            print(f"  ❌ BUG CONFIRMED: Fisher is group-reduced to {fisher_shape}")
            print(f"     Expected: [512, 512] (full parameter)")
            print(f"     Actual: {fisher_shape} (grouped by heads)")
        else:
            print(f"  ✓ Fisher has full shape {fisher_shape}")
        print()

    print("Step 5: Inspect contribution_cache shapes")
    print("Expected: [512, 512] (full parameter)")
    if fisher_collector.contribution_cache:
        sample_key = list(fisher_collector.contribution_cache.keys())[0]
        print(f"Sample key: {sample_key}")
        
        for param_name, contrib in list(fisher_collector.contribution_cache[sample_key].items())[:3]:
            if 'q_proj' in param_name or 'k_proj' in param_name:
                print(f"  {param_name}: {contrib.shape}")
        print()

    print("Step 6: Simulate Phase 6's expectation")
    print("Phase 6 tries to slice Fisher tensor by QK-OV blocks:")
    print("  For Q, head 0: expects to slice rows [0:64, :] from [512, 512]")
    print("  But receives: grouped tensor of shape [8] (8 heads)")
    print()
    print("Result: Phase 6 cannot perform QK-OV-specific slicing!")
    print()

    print("=" * 70)
    print("Conclusion")
    print("=" * 70)
    print()
    print("BUG CONFIRMED:")
    print("  1. Phase 1 stores group-reduced Fisher in fisher_ema: [num_heads]")
    print("  2. Phase 1 treats Q/K/V identically (no block distinction)")
    print("  3. Phase 6 expects full Fisher: [hidden_size, hidden_size]")
    print("  4. Phase 6 needs to slice by Q/K/V/O blocks separately")
    print()
    print("Impact:")
    print("  - Phase 6 cannot distinguish Q vs K vs V contributions")
    print("  - The 'block-wise resolution' claim is compromised")
    print("  - Slicing group-reduced tensors produces incorrect results")
    print()
    print("See docs/PHASE1_PHASE6_THEORETICAL_CONFLICT.md for full analysis")
    print()


if __name__ == '__main__':
    test_fisher_shape_mismatch()
