#!/usr/bin/env python3
"""Test comprehensive Fisher analysis implementation."""

import sys
import torch
import json
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from unified_model_analysis import UnifiedModelAnalyzer, UnifiedConfig
from BombshellMetrics import BombshellMetrics

def test_fisher_comprehensive():
    """Test the complete Fisher analysis pipeline."""

    print("=" * 80)
    print("Testing Comprehensive Fisher Analysis")
    print("=" * 80)

    # Configure analysis for Fisher metrics
    config = UnifiedConfig(
        base_model="Qwen/Qwen2.5-Math-1.5B",
        model_paths=["Qwen/Qwen2.5-Math-1.5B"],
        metrics_to_compute=[
            'update_fisher_ema',
            'compute_fisher_importance',
            'compare_task_fisher',
            'get_fisher_pruning_masks',
            'compute_fisher_overlap',
            'scale_by_fisher',
            'estimate_fisher_uncertainty'
        ],
        output_dir=Path("fisher_test_output"),
        batch_size=4,
        skip_fisher_metrics=False,
        use_float16=True
    )

    # Create output directory
    config.output_dir.mkdir(exist_ok=True)

    # Initialize analyzer
    print("\n1. Initializing analyzer...")
    analyzer = UnifiedModelAnalyzer(config)

    # Create test batches
    print("\n2. Creating test batches...")
    tokenizer = analyzer.tokenizer

    # Create two different batches to simulate different tasks
    batch1 = tokenizer(
        ["Solve: 2x + 3 = 7", "Calculate: sin(π/2)"],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    batch1['labels'] = batch1['input_ids'].clone()

    batch2 = tokenizer(
        ["What is machine learning?", "Explain neural networks."],
        return_tensors='pt',
        padding=True,
        truncation=True,
        max_length=128
    )
    batch2['labels'] = batch2['input_ids'].clone()

    # Run analysis
    print("\n3. Running Fisher analysis...")
    print("-" * 40)

    try:
        # Run the analysis (this will use our new Fisher suite)
        results = analyzer.analyze()

        print("\n4. Analysis complete. Checking results...")

        # Check if Fisher analysis was included
        fisher_found = False
        fisher_results = {}

        for model_id, model_result in results.model_results.items():
            print(f"\nModel: {model_id}")

            # Look for Fisher metrics
            for metric_name, metric in model_result.metrics.items():
                if 'fisher' in metric_name.lower():
                    fisher_found = True
                    print(f"  ✓ {metric_name}: computed")

                    # Check if it's the comprehensive Fisher analysis
                    if metric_name == 'fisher_analysis_comprehensive':
                        fisher_results = metric.value

                        # Print detailed Fisher results
                        if 'fisher_ema_data' in fisher_results:
                            print(f"    - Tasks analyzed: {fisher_results['fisher_ema_data'].get('tasks', [])}")
                            print(f"    - Total parameters: {fisher_results['fisher_ema_data'].get('total_parameters', 0)}")

                        if 'fisher_metrics' in fisher_results:
                            for key, value in fisher_results['fisher_metrics'].items():
                                if isinstance(value, dict):
                                    print(f"    - {key}: {len(value)} items")
                                else:
                                    print(f"    - {key}: {value}")

                        if 'recommendations' in fisher_results:
                            recs = fisher_results['recommendations']
                            print(f"\n  Recommendations:")
                            print(f"    - Merge strategy: {recs.get('merge_strategy')}")
                            print(f"    - Pruning safe: {recs.get('pruning_safe')}")
                            print(f"    - Interference risk: {recs.get('interference_risk')}")
                            for guidance in recs.get('detailed_guidance', []):
                                print(f"    - {guidance}")

        # Save detailed results
        output_file = config.output_dir / f"fisher_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                'fisher_analysis': fisher_results,
                'fisher_found': fisher_found,
                'metrics_computed': list(model_result.metrics.keys()) if results.model_results else []
            }, f, indent=2, default=str)

        print(f"\n5. Results saved to: {output_file}")

        # Final verdict
        print("\n" + "=" * 80)
        if fisher_found and fisher_results:
            print("✅ SUCCESS: Fisher analysis pipeline is working!")
            if 'error' in str(fisher_results):
                print("⚠️  WARNING: Some Fisher metrics returned errors")
        else:
            print("❌ FAILURE: Fisher analysis did not complete properly")
        print("=" * 80)

        return fisher_found

    except Exception as e:
        print(f"\n❌ ERROR during analysis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Test direct Fisher EMA functionality first
    print("\nQuick Fisher EMA Test:")
    print("-" * 40)

    try:
        bomb = BombshellMetrics()
        print(f"Initial Fisher EMA size: {len(bomb.fisher_ema)}")

        # Try to update Fisher EMA with dummy data
        model = torch.nn.Linear(10, 10)
        batch = {
            'input_ids': torch.randint(0, 100, (2, 10)),
            'labels': torch.randint(0, 100, (2, 10)),
            'attention_mask': torch.ones(2, 10)
        }

        bomb.update_fisher_ema(model, batch, task='test')
        print(f"After update Fisher EMA size: {len(bomb.fisher_ema)}")

        # Check what keys were added
        if bomb.fisher_ema:
            print("Sample Fisher EMA keys:")
            for i, key in enumerate(list(bomb.fisher_ema.keys())[:3]):
                print(f"  - {key}")
    except Exception as e:
        print(f"Fisher EMA test failed: {e}")

    print("\n" + "=" * 80)
    print("Running Full Analysis Test")
    print("=" * 80)

    # Run the full test
    success = test_fisher_comprehensive()

    sys.exit(0 if success else 1)