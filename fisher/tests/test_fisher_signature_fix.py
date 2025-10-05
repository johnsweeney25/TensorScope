#!/usr/bin/env python3
"""
Test script to verify Fisher signature fixes in unified_model_analysis.py
"""

import torch
import torch.nn as nn
from BombshellMetrics import BombshellMetrics
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Simple test model
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 10)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        x = torch.randn(4, 10)  # Batch size 4, input dim 10
        x = self.fc1(x)
        x = torch.relu(x)
        logits = self.fc2(x)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        class Output:
            pass
        output = Output()
        output.loss = loss
        output.logits = logits
        return output

def test_fisher_signatures():
    """Test that Fisher methods work with corrected signatures."""

    logger.info("Testing Fisher method signatures...")

    # Initialize metrics
    bombshell = BombshellMetrics()
    model = TestModel()

    # Enable gradients for model
    for param in model.parameters():
        param.requires_grad_(True)

    # Create test batch
    batch = {
        'input_ids': torch.randint(0, 10, (4, 10)),
        'attention_mask': torch.ones(4, 10),
        'labels': torch.randint(0, 10, (4,))
    }

    # Test 1: Update Fisher EMA (prerequisite)
    logger.info("\n1. Testing update_fisher_ema...")
    try:
        bombshell.update_fisher_ema(model, batch, task='task1')
        logger.info("   ✓ update_fisher_ema succeeded for task1")

        bombshell.update_fisher_ema(model, batch, task='task2')
        logger.info("   ✓ update_fisher_ema succeeded for task2")
    except Exception as e:
        logger.error(f"   ✗ update_fisher_ema failed: {e}")
        return False

    # Test 2: compute_fisher_importance with correct signature
    logger.info("\n2. Testing compute_fisher_importance...")
    try:
        # Should NOT pass 'batch' parameter
        result = bombshell.compute_fisher_importance(
            model=model,
            task='task1',
            normalize=True,
            return_per_layer=False
        )
        if isinstance(result, dict) and 'error' not in result:
            logger.info(f"   ✓ compute_fisher_importance succeeded")
        else:
            logger.error(f"   ✗ compute_fisher_importance returned error: {result}")
    except TypeError as e:
        logger.error(f"   ✗ compute_fisher_importance failed with TypeError: {e}")
        return False

    # Test 3: get_fisher_pruning_masks with correct signature
    logger.info("\n3. Testing get_fisher_pruning_masks...")
    try:
        # Should NOT pass 'model' parameter
        masks = bombshell.get_fisher_pruning_masks(
            task='task1',
            sparsity=0.5,
            structured=False
        )
        if isinstance(masks, dict) and 'error' not in masks:
            logger.info(f"   ✓ get_fisher_pruning_masks succeeded, got {len(masks)} masks")
        else:
            logger.error(f"   ✗ get_fisher_pruning_masks returned error: {masks}")
    except TypeError as e:
        logger.error(f"   ✗ get_fisher_pruning_masks failed with TypeError: {e}")
        return False

    # Test 4: scale_by_fisher with temperature (not scaling_type)
    logger.info("\n4. Testing scale_by_fisher...")
    try:
        # Create some gradients
        gradients = {}
        model.zero_grad()
        outputs = model(**batch)
        outputs.loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        # Should use 'temperature' not 'scaling_type'
        result = bombshell.scale_by_fisher(
            gradients=gradients,
            task='task1',
            temperature=1.0  # NOT scaling_type
        )
        if isinstance(result, dict) and 'error' not in result:
            logger.info(f"   ✓ scale_by_fisher succeeded")
        else:
            logger.error(f"   ✗ scale_by_fisher returned error: {result}")
    except TypeError as e:
        logger.error(f"   ✗ scale_by_fisher failed with TypeError: {e}")
        return False

    # Test 5: compute_fisher_overlap
    logger.info("\n5. Testing compute_fisher_overlap...")
    try:
        masks1 = bombshell.get_fisher_pruning_masks(task='task1', sparsity=0.5)
        masks2 = bombshell.get_fisher_pruning_masks(task='task2', sparsity=0.5)

        if isinstance(masks1, dict) and 'error' not in masks1 and \
           isinstance(masks2, dict) and 'error' not in masks2:
            overlap = bombshell.compute_fisher_overlap(masks1, masks2)
            if isinstance(overlap, dict) and 'error' not in overlap:
                logger.info(f"   ✓ compute_fisher_overlap succeeded")
            else:
                logger.error(f"   ✗ compute_fisher_overlap returned error: {overlap}")
        else:
            logger.error(f"   ✗ Could not generate masks for overlap test")
    except Exception as e:
        logger.error(f"   ✗ compute_fisher_overlap failed: {e}")
        return False

    logger.info("\n✅ All Fisher signature tests passed!")
    return True

if __name__ == "__main__":
    success = test_fisher_signatures()
    exit(0 if success else 1)