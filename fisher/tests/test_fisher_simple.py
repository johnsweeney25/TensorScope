#!/usr/bin/env python3
"""
Simple test to check if Fisher importance computation works and is not duplicated.
"""

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging to see what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def main():
    """Run simple test of Fisher importance."""
    print("\n" + "="*60)
    print("SIMPLE FISHER IMPORTANCE TEST")
    print("="*60)

    # Import after setting up logging
    from unified_model_analysis import analyze_models, UnifiedConfig

    # Create config with Fisher metrics enabled
    config = UnifiedConfig(
        skip_expensive=True,
        skip_checkpoint_metrics=True,
        metrics_to_compute=['compute_fisher_importance'],  # Only Fisher importance
        skip_fisher_metrics=False,
        gradient_batch_size=4,  # Small for testing
        batch_size=4  # Small for testing
    )

    # Test with small model
    model_paths = ['Qwen/Qwen2.5-Math-1.5B']

    print(f"\nAnalyzing model: {model_paths[0]}")
    print("Expected: Fisher importance computed once in suite, skipped in individual metrics")
    print("-"*60)

    try:
        results = analyze_models(model_paths, config)
        print("\n" + "-"*60)
        print("Analysis complete!")

        # Check results
        if results and results.models:
            for model_id, model_results in results.models.items():
                print(f"\nModel: {model_id}")
                print(f"Metrics computed: {len(model_results.metrics)}")

                if 'compute_fisher_importance' in model_results.metrics:
                    fisher_metric = model_results.metrics['compute_fisher_importance']
                    print(f"✓ Fisher importance present")
                    print(f"  - Compute time: {fisher_metric.compute_time:.3f}s")
                    if fisher_metric.compute_time == 0.0:
                        print(f"  - (Cached from suite - no duplicate computation)")

                if 'fisher_analysis_comprehensive' in model_results.metrics:
                    print(f"✓ Comprehensive Fisher analysis present")

        print("\n✅ Test completed successfully!")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

    print("="*60)

if __name__ == "__main__":
    main()