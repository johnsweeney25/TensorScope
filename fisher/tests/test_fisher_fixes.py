#!/usr/bin/env python3
"""Test script to verify Fisher metric fixes."""

import torch
import numpy as np
from BombshellMetrics import BombshellMetrics
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fisher_metrics():
    """Test all Fisher metrics that were failing."""

    logger.info("Initializing test model...")

    # Use a small model for testing
    model_name = "gpt2"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Initialize metrics
    metrics = BombshellMetrics()

    # Create test data
    test_text = "The quick brown fox jumps over the lazy dog"
    batch = tokenizer(test_text, return_tensors='pt', padding=True)
    batch['labels'] = batch['input_ids'].clone()

    logger.info("Updating Fisher EMA for tasks...")

    # Update Fisher EMA for different tasks
    for task in ['math', 'general']:
        logger.info(f"  Updating Fisher EMA for task: {task}")
        metrics.update_fisher_ema(model, batch, task=task)

    # Test 1: get_fisher_pruning_masks
    logger.info("\nTest 1: get_fisher_pruning_masks")
    try:
        masks = metrics.get_fisher_pruning_masks(task='math', sparsity=0.5)
        if 'error' in masks:
            logger.error(f"  ❌ Error: {masks['error']}")
        else:
            logger.info(f"  ✅ Success! Generated {len(masks)} masks")
            # Check mask properties
            for name, mask in list(masks.items())[:2]:
                sparsity_actual = 1.0 - (mask.sum().item() / mask.numel())
                logger.info(f"    {name}: shape={mask.shape}, sparsity={sparsity_actual:.2f}")
    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")

    # Test 2: get_top_fisher_directions
    logger.info("\nTest 2: get_top_fisher_directions")
    try:
        # Test with EMA Fisher (default)
        top_coords = metrics.get_top_fisher_directions(
            task='math',
            fisher_type='ema',
            top_k_per_param=100,
            percentile=95.0
        )
        logger.info(f"  ✅ Success! Got top coordinates for {len(top_coords)} parameters")
        for name, mask in list(top_coords.items())[:2]:
            selected = mask.sum().item()
            logger.info(f"    {name}: selected {selected}/{mask.numel()} coordinates")
    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")

    # Test 3: compare_task_fisher
    logger.info("\nTest 3: compare_task_fisher")
    try:
        result = metrics.compare_task_fisher(task1='math', task2='general')
        logger.info(f"  ✅ Success!")
        logger.info(f"    Divergence: {result.get('divergence', 0):.4f}")
        logger.info(f"    Correlation: {result.get('correlation', 0):.4f}")
        logger.info(f"    Magnitude ratio: {result.get('magnitude_ratio', 0):.4f}")
    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")

    # Test 4: compute_fisher_overlap
    logger.info("\nTest 4: compute_fisher_overlap")
    try:
        masks1 = metrics.get_fisher_pruning_masks(task='math', sparsity=0.5)
        masks2 = metrics.get_fisher_pruning_masks(task='general', sparsity=0.5)

        if 'error' not in masks1 and 'error' not in masks2:
            overlap_result = metrics.compute_fisher_overlap(masks1, masks2)
            logger.info(f"  ✅ Success!")
            logger.info(f"    Overlap: {overlap_result['overlap_percentage']:.2f}%")
        else:
            logger.error(f"  ❌ Could not generate masks for overlap test")
    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")

    # Test 5: scale_by_fisher
    logger.info("\nTest 5: scale_by_fisher")
    try:
        # Compute gradients
        model.zero_grad()
        model.train()
        for param in model.parameters():
            param.requires_grad_(True)

        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        model.zero_grad()
        model.eval()

        # Scale gradients by Fisher
        scaled = metrics.scale_by_fisher(gradients, task='math', scaling_type='inverse')
        logger.info(f"  ✅ Success! Scaled {len(scaled)} gradients")

        # Check scaling effect
        for name in list(scaled.keys())[:2]:
            if name in gradients:
                orig_norm = gradients[name].norm().item()
                scaled_norm = scaled[name].norm().item()
                logger.info(f"    {name}: orig_norm={orig_norm:.4f}, scaled_norm={scaled_norm:.4f}")
    except Exception as e:
        logger.error(f"  ❌ Failed: {e}")

    logger.info("\n" + "="*50)
    logger.info("Testing complete!")

if __name__ == "__main__":
    test_fisher_metrics()