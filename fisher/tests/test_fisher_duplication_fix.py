#!/usr/bin/env python3
"""
Test to verify that Fisher importance metrics are not computed twice.
This script verifies that the fix prevents duplicate computation of Fisher metrics.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from unified_model_analysis import UnifiedModelAnalyzer, UnifiedConfig, ModelSpec

# Configure logging to see the messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_fisher_duplication_fix():
    """Test that Fisher importance is not computed twice."""

    print("\n" + "="*60)
    print("TESTING FISHER DUPLICATION FIX")
    print("="*60)

    # Create minimal config for testing
    config = UnifiedConfig(
        skip_expensive=True,
        skip_checkpoint_metrics=True,
        metrics_to_compute=['compute_fisher_importance', 'compute_dead_neurons'],  # Include Fisher metric
        skip_fisher_metrics=False,  # Don't skip Fisher metrics
        gradient_batch_size=4  # Small batch for testing
    )

    # Create analyzer
    analyzer = UnifiedModelAnalyzer(config)

    # Create a small model spec for testing
    model_spec = ModelSpec(
        path='Qwen/Qwen2.5-Math-1.5B',  # Small model for testing
        group='default'
    )

    print("\nExpected behavior:")
    print("1. Fisher EMA should be computed in the Fisher analysis suite")
    print("2. Fisher importance should be computed IN the suite")
    print("3. Fisher importance should be SKIPPED in individual metrics")
    print("4. Log should show: '↔️ Skipping compute_fisher_importance - already computed in Fisher analysis suite'")
    print("\n" + "-"*60)

    # Track log messages
    log_messages = []

    class LogCapture(logging.Handler):
        def emit(self, record):
            log_messages.append(record.getMessage())

    # Add handler to capture logs
    handler = LogCapture()
    logging.getLogger().addHandler(handler)

    try:
        # Run analysis
        print("\nRunning analysis...")
        results = analyzer.analyze_models([model_spec])

        # Check the logs
        print("\n" + "-"*60)
        print("VERIFICATION:")
        print("-"*60)

        # Check if Fisher importance was computed in suite
        suite_computed = any("Computing Fisher importance (as part of suite)" in msg for msg in log_messages)
        print(f"✓ Fisher importance computed in suite: {suite_computed}")

        # Check if Fisher importance was skipped in individual metrics
        skipped_individual = any("Skipping compute_fisher_importance - already computed" in msg for msg in log_messages)
        print(f"✓ Fisher importance skipped in individual metrics: {skipped_individual}")

        # Check for duplicate computation (should not see two "Starting metric 'compute_fisher_importance'" logs)
        starting_logs = [msg for msg in log_messages if "Starting metric 'compute_fisher_importance'" in msg]
        no_duplicate = len(starting_logs) == 0  # Should be 0 since it's skipped
        print(f"✓ No duplicate computation: {no_duplicate} (found {len(starting_logs)} 'Starting metric' logs)")

        # Check results
        if results and results.models:
            model_results = results.models[model_spec.id]
            if 'compute_fisher_importance' in model_results.metrics:
                print(f"✓ Fisher importance results present in final metrics")
                fisher_result = model_results.metrics['compute_fisher_importance']
                print(f"  - Compute time: {fisher_result.compute_time:.3f}s (should be 0.0 for cached)")

            if 'fisher_analysis_comprehensive' in model_results.metrics:
                print(f"✓ Comprehensive Fisher analysis results present")

        print("\n" + "="*60)
        if suite_computed and skipped_individual and no_duplicate:
            print("✅ TEST PASSED: Fisher duplication fix is working correctly!")
        else:
            print("❌ TEST FAILED: Fisher metrics are still being computed twice")
            print("\nRelevant log messages:")
            for msg in log_messages:
                if 'fisher' in msg.lower() or 'Fisher' in msg:
                    print(f"  - {msg}")
        print("="*60)

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Remove handler
        logging.getLogger().removeHandler(handler)

if __name__ == "__main__":
    test_fisher_duplication_fix()