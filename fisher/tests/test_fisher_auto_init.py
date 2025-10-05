#!/usr/bin/env python3
"""
Test script to verify Fisher metrics auto-initialization works correctly.
This tests that compute_fisher_importance and get_fisher_pruning_masks
can auto-initialize Fisher EMA when needed.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from BombshellMetrics import BombshellMetrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 10)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids=None, labels=None, **kwargs):
        # Simple forward pass
        x = torch.randn(1, 10)  # Create dummy input
        x = self.layer1(x)
        x = torch.relu(x)
        output = self.layer2(x)

        loss = None
        if labels is not None:
            # Create dummy target to compute loss
            target = torch.randn_like(output)
            loss = self.loss_fn(output, target)

        # Return format expected by BombshellMetrics
        class Output:
            def __init__(self, loss):
                self.loss = loss

        return Output(loss)

def test_fisher_auto_init():
    """Test that Fisher metrics can auto-initialize."""
    logger.info("=" * 60)
    logger.info("TESTING FISHER AUTO-INITIALIZATION")
    logger.info("=" * 60)

    # Create simple model and metrics instance
    model = SimpleModel()
    metrics = BombshellMetrics(seed=42)

    # Create dummy batch
    batch = {
        'input_ids': torch.randint(0, 100, (4, 16)),
        'attention_mask': torch.ones(4, 16),
        'labels': torch.randint(0, 100, (4, 16))
    }

    # Test 1: compute_fisher_importance without pre-initialized EMA
    logger.info("\n1. Testing compute_fisher_importance (no pre-init)...")
    result = metrics.compute_fisher_importance(
        task='test_task',
        normalize=True,
        model=model,
        batch=batch
    )

    if 'error' in result:
        logger.error(f"   ❌ Error: {result['error']}")
    else:
        logger.info(f"   ✅ Success! Found {len(result.get('parameter_importance', {}))} parameters")
        logger.info(f"   ✅ Layer importance: {result.get('layer_importance', {}).keys()}")

    # Test 2: get_fisher_pruning_masks without pre-initialized EMA
    logger.info("\n2. Testing get_fisher_pruning_masks (no pre-init)...")

    # Clear Fisher EMA to test fresh initialization
    metrics.fisher_ema.clear()

    masks = metrics.get_fisher_pruning_masks(
        task='test_task2',
        sparsity=0.5,
        model=model,
        batch=batch
    )

    if isinstance(masks, dict) and 'error' in masks:
        logger.error(f"   ❌ Error: {masks['error']}")
    else:
        logger.info(f"   ✅ Success! Generated {len(masks)} masks")
        for name, mask in list(masks.items())[:2]:  # Show first 2 masks
            logger.info(f"      {name}: shape={mask.shape}, sparsity={(~mask).float().mean():.2%}")

    # Test 3: Verify auto-init actually populated Fisher EMA
    logger.info("\n3. Verifying Fisher EMA was populated...")
    ema_keys = [k for k in metrics.fisher_ema.keys() if 'test_task' in k]
    if ema_keys:
        logger.info(f"   ✅ Fisher EMA has {len(ema_keys)} entries")
        logger.info(f"   ✅ Sample keys: {ema_keys[:3]}")
    else:
        logger.error("   ❌ Fisher EMA was not populated")

    # Test 4: Test without model/batch (should fail gracefully)
    logger.info("\n4. Testing without model/batch (should fail)...")
    metrics.fisher_ema.clear()  # Clear EMA

    result = metrics.compute_fisher_importance(
        task='no_init_task',
        normalize=True
        # No model or batch provided
    )

    if 'error' in result:
        logger.info(f"   ✅ Correctly returned error: {result['error'][:100]}...")
    else:
        logger.error("   ❌ Should have returned an error without model/batch")

    logger.info("\n" + "=" * 60)
    logger.info("TEST COMPLETE")
    logger.info("=" * 60)

def test_with_real_model():
    """Test with a real transformer model (optional, requires GPU)."""
    if not torch.cuda.is_available():
        logger.info("Skipping real model test (no GPU available)")
        return

    logger.info("\n" + "=" * 60)
    logger.info("TESTING WITH REAL MODEL")
    logger.info("=" * 60)

    try:
        # Load a small model
        model_name = "gpt2"
        logger.info(f"Loading {model_name}...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Create metrics instance
        metrics = BombshellMetrics(seed=42)

        # Create batch
        texts = ["Hello world", "Test input", "Fisher analysis", "Auto initialization"]
        inputs = tokenizer(texts, padding=True, truncation=True, max_length=32, return_tensors="pt")
        batch = {
            'input_ids': inputs['input_ids'].to(model.device),
            'attention_mask': inputs['attention_mask'].to(model.device),
            'labels': inputs['input_ids'].to(model.device)
        }

        # Test Fisher importance with auto-init
        logger.info("\nTesting Fisher importance with real model...")
        result = metrics.compute_fisher_importance(
            task='gpt2_task',
            normalize=True,
            model=model,
            batch=batch
        )

        if 'error' in result:
            logger.error(f"   ❌ Error: {result['error']}")
        else:
            logger.info(f"   ✅ Success! Analyzed {len(result.get('parameter_importance', {}))} parameters")

            # Show top 5 important parameters
            if 'parameter_importance' in result:
                sorted_params = sorted(
                    result['parameter_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:5]
                logger.info("   Top 5 important parameters:")
                for name, importance in sorted_params:
                    logger.info(f"      {name}: {importance:.6f}")

    except Exception as e:
        logger.error(f"Real model test failed: {e}")

if __name__ == "__main__":
    # Run basic tests
    test_fisher_auto_init()

    # Optionally test with real model
    # test_with_real_model()