#!/usr/bin/env python3
"""
Test that compute_fisher_overlap works correctly after fix.
This tests the fix for the error: "free variable 'torch' referenced before assignment in enclosing scope"
"""

import torch
import sys
import logging
from BombshellMetrics import BombshellMetrics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_compute_fisher_overlap():
    """Test that compute_fisher_overlap works without import errors."""

    # Create instance of BombshellMetrics
    logger.info("Creating BombshellMetrics instance...")
    metrics = BombshellMetrics(seed=42)

    # Create dummy masks for testing
    logger.info("Creating test masks...")
    masks1 = {
        'layer1.weight': torch.ones(100, dtype=torch.bool),
        'layer2.weight': torch.tensor([True, False, True, False, True]),
        'layer3.weight': torch.zeros(50, dtype=torch.bool)
    }

    masks2 = {
        'layer1.weight': torch.ones(100, dtype=torch.bool),
        'layer2.weight': torch.tensor([False, True, True, False, True]),
        'layer3.weight': torch.ones(50, dtype=torch.bool)
    }

    # Test compute_fisher_overlap
    logger.info("Testing compute_fisher_overlap...")
    try:
        result = metrics.compute_fisher_overlap(masks1, masks2)

        # Check result structure
        assert 'overlap_percentage' in result, "Missing overlap_percentage in result"
        assert 'per_layer_overlap' in result, "Missing per_layer_overlap in result"
        assert 'error' not in result, f"Error in result: {result.get('error')}"

        # Check specific values
        layer1_overlap = result['per_layer_overlap']['layer1.weight']
        layer3_overlap = result['per_layer_overlap']['layer3.weight']

        # Use approximate comparison for floating point
        assert abs(layer1_overlap - 1.0) < 0.001, f"layer1 should have 100% overlap, got {layer1_overlap}"
        assert abs(layer3_overlap - 0.0) < 0.001, f"layer3 should have 0% overlap, got {layer3_overlap}"

        logger.info(f"✅ Test passed! Overlap percentage: {result['overlap_percentage']:.2f}%")
        logger.info(f"   Per-layer overlap: {result['per_layer_overlap']}")

        return True

    except Exception as e:
        logger.error(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """Test error handling in compute_fisher_overlap."""

    metrics = BombshellMetrics(seed=42)

    # Test with invalid input types
    logger.info("Testing error handling with invalid inputs...")

    # Non-dict inputs
    result = metrics.compute_fisher_overlap("invalid", {})
    assert 'error' in result, "Should return error for non-dict input"
    logger.info(f"✅ Handled non-dict input correctly: {result['error']}")

    # Dict with non-tensor values
    masks_bad = {'layer1': "not a tensor"}
    masks_good = {'layer1': torch.ones(10, dtype=torch.bool)}
    result = metrics.compute_fisher_overlap(masks_bad, masks_good)
    assert 'error' in result, "Should return error for non-tensor values"
    logger.info(f"✅ Handled non-tensor values correctly: {result['error']}")

    return True

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Testing compute_fisher_overlap fix")
    logger.info("=" * 60)

    success = True

    # Run main test
    if not test_compute_fisher_overlap():
        success = False

    # Run error handling tests
    if not test_error_handling():
        success = False

    if success:
        logger.info("\n🎉 All tests passed! The fix works correctly.")
        sys.exit(0)
    else:
        logger.error("\n❌ Some tests failed. Please check the errors above.")
        sys.exit(1)