#!/usr/bin/env python3
"""
Test script to verify scale_by_fisher NoneType fix
"""

import torch
import torch.nn as nn
import logging
from BombshellMetrics import BombshellMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleModel(nn.Module):
    """Simple test model."""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

def test_scale_by_fisher_with_none():
    """Test that scale_by_fisher handles None gradients gracefully."""

    metrics = BombshellMetrics()

    # Test 1: Call with None gradients
    logger.info("Test 1: scale_by_fisher with None gradients")
    result = metrics.scale_by_fisher(None, 'general')
    assert result == {}, "Should return empty dict for None gradients"
    logger.info("✅ Handles None gradients correctly")

    # Test 2: Call with empty gradients dict
    logger.info("Test 2: scale_by_fisher with empty gradients dict")
    result = metrics.scale_by_fisher({}, 'general')
    assert result == {}, "Should return empty dict for empty gradients"
    logger.info("✅ Handles empty gradients correctly")

    # Test 3: Call with valid gradients
    logger.info("Test 3: scale_by_fisher with valid gradients")
    gradients = {
        'layer1': torch.randn(10, 10),
        'layer2': torch.randn(5, 5)
    }
    result = metrics.scale_by_fisher(gradients, 'general')
    # Should return gradients (possibly scaled)
    assert isinstance(result, dict), "Should return dict for valid gradients"
    logger.info("✅ Handles valid gradients correctly")

def test_unified_analysis_gradient_computation():
    """Test that unified_model_analysis handles gradient computation errors."""

    from unified_model_analysis import MetricRegistry, MetricContext

    registry = MetricRegistry()

    # Create a model with no trainable parameters
    model = SimpleModel()
    for param in model.parameters():
        param.requires_grad = False

    batch = {
        'input_ids': torch.randint(0, 100, (8, 10)),
        'attention_mask': torch.ones(8, 10)
    }

    context = MetricContext(
        models=[model],
        batches=[batch]
    )

    # This should not crash even if gradient computation fails
    logger.info("Test 4: _compute_gradients with no trainable params")
    gradients = registry._compute_gradients(model, batch)
    if gradients is None:
        logger.info("✅ Returns None for model with no trainable params")
    else:
        logger.info(f"✅ Returns empty dict or gradients: {len(gradients)} params")

def main():
    logger.info("=" * 60)
    logger.info("Testing scale_by_fisher NoneType fixes")
    logger.info("=" * 60)

    # Run tests
    test_scale_by_fisher_with_none()
    test_unified_analysis_gradient_computation()

    logger.info("\n" + "=" * 60)
    logger.info("✅ ALL TESTS PASSED")
    logger.info("scale_by_fisher now handles None/empty gradients correctly")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()