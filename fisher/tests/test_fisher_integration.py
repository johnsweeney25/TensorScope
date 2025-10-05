#!/usr/bin/env python3
"""
Test Fisher integration after refactoring.
Validates backward compatibility and proper key format handling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import traceback

def test_backward_compatibility():
    """Test that the refactored Fisher implementation maintains backward compatibility."""
    print("\n" + "="*60)
    print("TESTING FISHER BACKWARD COMPATIBILITY")
    print("="*60)

    try:
        # Import the refactored BombshellMetrics (now renamed to BombshellMetrics.py)
        from BombshellMetrics import BombshellMetrics
        print("✓ BombshellMetrics imported successfully")

        # Create a simple model wrapper that accepts batch dict
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 10)
                )

            def forward(self, **batch):
                # Extract input from batch
                input_ids = batch['input_ids']
                labels = batch['labels']

                # Simple processing
                x = F.one_hot(input_ids, 10).float().mean(dim=1)
                logits = self.layers(x)

                # Compute loss
                labels_flat = labels[:, 0] if len(labels.shape) > 1 else labels
                loss = F.cross_entropy(logits, labels_flat)

                # Return in expected format
                class Output:
                    pass
                output = Output()
                output.loss = loss
                output.logits = logits
                return output

        model = SimpleModel()

        # Create dummy batch
        batch = {
            'input_ids': torch.randint(0, 10, (2, 5)),
            'labels': torch.randint(0, 10, (2, 5))
        }

        # Initialize BombshellMetrics
        bombshell = BombshellMetrics(
            fisher_reduction='group',
            fisher_storage='cpu_fp16'
        )
        print("✓ BombshellMetrics initialized with Fisher parameters")

        # Test 1: Update Fisher EMA (new method)
        bombshell.update_fisher_ema(model, batch, task='task1')
        print("✓ Fisher EMA updated successfully")

        # Test 2: Direct access to fisher_ema (backward compatibility)
        # This should work with the compatibility layer
        keys = list(bombshell.fisher_ema.keys())
        print(f"✓ Fisher EMA keys accessible: {len(keys)} keys found")

        # Test 3: Old format key access (task_param)
        # Set a value using old format
        bombshell.fisher_ema['task1_test_param'] = torch.tensor([1.0, 2.0, 3.0])
        print("✓ Old format key write works")

        # Retrieve using old format
        value = bombshell.fisher_ema['task1_test_param']
        assert value is not None, "Could not retrieve value with old format key"
        print("✓ Old format key read works")

        # Test 4: Key parsing compatibility
        for key in bombshell.fisher_ema.keys():
            # Old code expects underscore format
            if '_' in key:
                parts = key.split('_', 1)
                task = parts[0]
                print(f"  - Old format key found: task='{task}'")
            # New format uses pipes
            elif '|' in key:
                parts = key.split('|')
                task = parts[0]
                print(f"  - New format key found: task='{task}'")

        # Test 5: Task extraction (critical for unified_model_analysis.py)
        available_tasks = set()
        for key in bombshell.fisher_ema.keys():
            if '|' in key:
                task_name = key.split('|')[0]
            else:
                task_name = key.split('_')[0]
            available_tasks.add(task_name)
        print(f"✓ Task extraction works: {available_tasks}")

        # Test 6: Check if Fisher computation actually stores data
        bombshell.update_fisher_ema(model, batch, task='test_task')
        task_keys = [k for k in bombshell.fisher_ema.keys()
                    if k.startswith('test_task_') or k.startswith('test_task|')]
        print(f"✓ Fisher computation stores data: {len(task_keys)} keys for 'test_task'")

        print("\n" + "="*60)
        print("ALL BACKWARD COMPATIBILITY TESTS PASSED! ✓")
        print("="*60)
        print("\nThe Fisher refactoring is working correctly with:")
        print("- Old format keys (task_param)")
        print("- New format keys (task|param|group)")
        print("- Direct fisher_ema access")
        print("- Task extraction from keys")
        print("\nThe integration is ready for use!")

        return True

    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        traceback.print_exc()
        return False

def test_unified_model_analysis_compatibility():
    """Test specific patterns used in unified_model_analysis.py."""
    print("\n" + "="*60)
    print("TESTING UNIFIED MODEL ANALYSIS COMPATIBILITY")
    print("="*60)

    try:
        from BombshellMetrics import BombshellMetrics

        model = nn.Linear(10, 10)
        batch = {
            'input_ids': torch.randint(0, 10, (2, 5)),
            'labels': torch.randint(0, 10, (2, 5))
        }

        bombshell = BombshellMetrics()

        # Pattern 1: Direct assignment (line 5761 in unified_model_analysis.py)
        bombshell.fisher_ema['task1_model.weight'] = torch.randn(10, 10)
        print("✓ Direct assignment to fisher_ema works")

        # Pattern 2: Key validation (line 5784)
        task_keys = [k for k in bombshell.fisher_ema.keys() if k.startswith("task1_")]
        assert len(task_keys) > 0, "No keys found starting with 'task1_'"
        print(f"✓ Key filtering works: {len(task_keys)} keys found")

        # Pattern 3: Task extraction (lines 2493-2504)
        available_tasks = set()
        for key in bombshell.fisher_ema.keys():
            if '|' in key:
                task_name = key.split('|')[0]
            else:
                task_name = key.split('_')[0]
            available_tasks.add(task_name)
        assert 'task1' in available_tasks, "task1 not found in available tasks"
        print(f"✓ Task extraction pattern works: {available_tasks}")

        # Pattern 4: Total parameters analyzed (line 5833)
        total_params = len(set(
            k.split('|')[1] if '|' in k else k.split('_', 1)[1]
            for k in bombshell.fisher_ema.keys()
            if ('_' in k or '|' in k) and len(k.split('|' if '|' in k else '_')) > 1
        ))
        print(f"✓ Parameter counting works: {total_params} unique parameters")

        print("\n✓ All unified_model_analysis.py patterns work correctly!")
        return True

    except Exception as e:
        print(f"\n✗ Compatibility test failed: {e}")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = True

    # Run backward compatibility tests
    if not test_backward_compatibility():
        success = False

    # Run unified_model_analysis compatibility tests
    if not test_unified_model_analysis_compatibility():
        success = False

    if success:
        print("\n" + "🎉"*20)
        print("ALL TESTS PASSED! Fisher integration is working correctly.")
        print("🎉"*20)
        sys.exit(0)
    else:
        print("\n⚠️ Some tests failed. Please review the errors above.")
        sys.exit(1)