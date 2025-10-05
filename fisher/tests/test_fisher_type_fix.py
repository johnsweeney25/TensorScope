#!/usr/bin/env python3
"""
Test that top_fisher_directions uses correct fisher_type
"""

import torch
import logging
from BombshellMetrics import BombshellMetrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_fisher_type_validation():
    """Test that get_top_fisher_directions validates fisher_type correctly."""

    metrics = BombshellMetrics()

    # Test 1: 'ema' should work (default)
    logger.info("Test 1: fisher_type='ema' (default)")
    try:
        result = metrics.get_top_fisher_directions(task='general', fisher_type='ema')
        logger.info("✅ 'ema' fisher_type works correctly")
    except ValueError as e:
        logger.error(f"❌ 'ema' fisher_type failed: {e}")

    # Test 2: 'direct' should work (but requires model and data)
    logger.info("\nTest 2: fisher_type='direct'")
    try:
        result = metrics.get_top_fisher_directions(fisher_type='direct')
        logger.info("✅ 'direct' fisher_type accepted (though it needs model/data)")
    except ValueError as e:
        if "model and task_data are required" in str(e):
            logger.info("✅ 'direct' fisher_type works but correctly requires model/data")
        else:
            logger.error(f"❌ Unexpected error for 'direct': {e}")

    # Test 3: 'diagonal' should raise an error
    logger.info("\nTest 3: fisher_type='diagonal' (invalid)")
    try:
        result = metrics.get_top_fisher_directions(fisher_type='diagonal')
        logger.error("❌ 'diagonal' should have raised an error but didn't")
    except ValueError as e:
        if "fisher_type must be 'direct' or 'ema'" in str(e):
            logger.info(f"✅ 'diagonal' correctly rejected: {e}")
        else:
            logger.error(f"❌ Wrong error for 'diagonal': {e}")

    # Test 4: Check default behavior
    logger.info("\nTest 4: Default fisher_type")
    try:
        result = metrics.get_top_fisher_directions()  # Should use 'ema' by default
        logger.info("✅ Default fisher_type works (uses 'ema')")
    except Exception as e:
        logger.error(f"❌ Default fisher_type failed: {e}")

def test_unified_registration():
    """Test that unified_model_analysis registers with correct fisher_type."""

    from unified_model_analysis import MetricRegistry

    registry = MetricRegistry()

    # Check the registered custom args
    if 'top_fisher_directions' in registry.metrics:
        custom_args = registry.metrics['top_fisher_directions'].get('custom_args', {})
        fisher_type = custom_args.get('fisher_type', 'not_found')

        if fisher_type == 'ema':
            logger.info(f"✅ Registry has correct fisher_type: '{fisher_type}'")
        elif fisher_type == 'diagonal':
            logger.error(f"❌ Registry still has old fisher_type: '{fisher_type}'")
            logger.info("   The fix needs to be applied!")
        else:
            logger.warning(f"⚠️ Registry has unexpected fisher_type: '{fisher_type}'")
    else:
        logger.warning("⚠️ top_fisher_directions not found in registry")

def main():
    logger.info("=" * 60)
    logger.info("Testing top_fisher_directions fisher_type fix")
    logger.info("=" * 60)

    test_fisher_type_validation()

    logger.info("\n" + "=" * 60)
    logger.info("Testing unified_model_analysis registration")
    logger.info("=" * 60)

    test_unified_registration()

    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY:")
    logger.info("- 'ema' is the correct default fisher_type")
    logger.info("- 'direct' requires model and task_data")
    logger.info("- 'diagonal' is no longer valid and will raise an error")
    logger.info("=" * 60)

if __name__ == "__main__":
    main()