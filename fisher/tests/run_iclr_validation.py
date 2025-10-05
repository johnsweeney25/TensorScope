#!/usr/bin/env python3
"""
ICLR 2026 Validation Script for Cross-Task Conflict Detection

Run this script to generate results for the ICLR paper.

Usage:
    python fisher/tests/run_iclr_validation.py \
        --model Qwen/Qwen2.5-Math-1.5B \
        --num-seeds 5 \
        --removal-pct 0.05 \
        --output results_iclr_validation.json

Expected runtime:
    - H100: ~30 minutes (5 seeds × 3 conditions)
    - A100: ~45 minutes

Output:
    - JSON file with full results
    - Console: Statistical analysis and verdict
    - Plots: Accuracy comparison (saved as PNG)
"""

import torch
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List
import matplotlib.pyplot as plt
import sys

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fisher.tests.test_cross_task_conflict_validation import (
    CrossTaskConflictValidator,
    ExperimentResult
)
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

logger = logging.getLogger(__name__)


def load_model_and_data(
    model_name: str,
    num_samples_per_task: int = 768,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Load model and prepare multi-task data.

    Args:
        model_name: HuggingFace model name
        num_samples_per_task: Number of samples per task
        device: Device to load model on

    Returns:
        Tuple of (model, tasks_data)
    """
    logger.info(f"Loading model: {model_name}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=device
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading datasets...")

    # Task 1: Mathematical reasoning (GSM8K)
    try:
        math_dataset = load_dataset("gsm8k", "main", split="train")
        math_samples = []

        for i, example in enumerate(math_dataset):
            if len(math_samples) >= num_samples_per_task:
                break

            # Tokenize
            text = f"Question: {example['question']}\nAnswer:"
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128
            )

            # Use answer for label (simplified)
            answer_text = str(example['answer'])
            answer_tokens = tokenizer(answer_text, return_tensors='pt')

            math_samples.append({
                'input_ids': inputs['input_ids'].squeeze(),
                'labels': answer_tokens['input_ids'].squeeze()[:10]  # First 10 tokens
            })

    except Exception as e:
        logger.warning(f"Could not load GSM8K: {e}")
        logger.info("Using synthetic math data instead")
        math_samples = [
            {
                'input_ids': torch.randint(0, tokenizer.vocab_size, (128,)),
                'labels': torch.randint(0, tokenizer.vocab_size, (10,))
            }
            for _ in range(num_samples_per_task)
        ]

    # Task 2: General text (C4 or synthetic)
    try:
        general_dataset = load_dataset("c4", "en", split="train", streaming=True)
        general_samples = []

        for example in general_dataset:
            if len(general_samples) >= num_samples_per_task:
                break

            text = example['text'][:500]  # First 500 chars
            inputs = tokenizer(
                text,
                return_tensors='pt',
                padding='max_length',
                truncation=True,
                max_length=128
            )

            # Next token prediction labels
            labels = inputs['input_ids'].clone()
            labels = torch.roll(labels, -1)

            general_samples.append({
                'input_ids': inputs['input_ids'].squeeze(),
                'labels': labels.squeeze()[:10]
            })

    except Exception as e:
        logger.warning(f"Could not load C4: {e}")
        logger.info("Using synthetic general data instead")
        general_samples = [
            {
                'input_ids': torch.randint(0, tokenizer.vocab_size, (128,)),
                'labels': torch.randint(0, tokenizer.vocab_size, (10,))
            }
            for _ in range(num_samples_per_task)
        ]

    tasks_data = {
        'math': math_samples[:num_samples_per_task],
        'general': general_samples[:num_samples_per_task]
    }

    logger.info(f"Loaded {len(tasks_data['math'])} math samples")
    logger.info(f"Loaded {len(tasks_data['general'])} general samples")

    return model, tasks_data


def plot_results(results_dict: Dict, output_path: str = 'validation_results.png'):
    """
    Create visualization of validation results.

    Args:
        results_dict: Results from analyze_results()
        output_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    conditions = ['Baseline', 'Random', 'Conflict']
    means = [
        results_dict['baseline']['mean'],
        results_dict['random']['mean'],
        results_dict['conflict']['mean']
    ]
    stds = [
        results_dict['baseline']['std'],
        results_dict['random']['std'],
        results_dict['conflict']['std']
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(conditions))
    width = 0.6

    bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                  color=['#3498db', '#95a5a6', '#e74c3c'],
                  alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(means, stds)):
        ax.text(i, mean + std + 0.5, f'{mean:.2f}%',
               ha='center', va='bottom', fontweight='bold')

    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Experimental Condition', fontsize=12, fontweight='bold')
    ax.set_title('Cross-Task Conflict Validation Results',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add significance annotation
    if results_dict['is_significant']:
        y_max = max(means) + max(stds) + 2
        ax.plot([1, 2], [y_max, y_max], 'k-', linewidth=1.5)
        ax.text(1.5, y_max + 0.3, f"p={results_dict['p_value_vs_random']:.4f} *",
               ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Run ICLR validation for cross-task conflict detection'
    )
    parser.add_argument('--model', type=str, default='Qwen/Qwen2.5-Math-1.5B',
                       help='HuggingFace model name')
    parser.add_argument('--num-samples', type=int, default=768,
                       help='Number of samples per task')
    parser.add_argument('--num-seeds', type=int, default=5,
                       help='Number of random seeds for statistical power')
    parser.add_argument('--removal-pct', type=float, default=0.05,
                       help='Percentage of samples to remove (0.05 = 5%%)')
    parser.add_argument('--output', type=str, default='iclr_validation_results.json',
                       help='Output JSON file path')
    parser.add_argument('--plot', type=str, default='iclr_validation_plot.png',
                       help='Output plot file path')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    logger.info("="*80)
    logger.info("ICLR 2026 VALIDATION: Cross-Task Sample Conflict Detection")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Samples per task: {args.num_samples}")
    logger.info(f"  Random seeds: {args.num_seeds}")
    logger.info(f"  Removal percentage: {args.removal_pct*100:.1f}%")
    logger.info(f"  Device: {args.device}")

    # Load model and data
    model, tasks_data = load_model_and_data(
        args.model,
        num_samples_per_task=args.num_samples,
        device=args.device
    )

    # Create validator
    validator = CrossTaskConflictValidator(
        model=model,
        tasks_data=tasks_data,
        removal_percentage=args.removal_pct,
        num_seeds=args.num_seeds,
        device=args.device
    )

    # Run experiments
    validator.run_experiment()

    # Analyze results
    analysis = validator.analyze_results()

    # Save results
    output_data = {
        'configuration': {
            'model': args.model,
            'num_samples_per_task': args.num_samples,
            'num_seeds': args.num_seeds,
            'removal_percentage': args.removal_pct,
            'device': args.device
        },
        'analysis': {
            k: (v.tolist() if hasattr(v, 'tolist') else v)
            for k, v in analysis.items()
        },
        'raw_results': {
            condition: [
                {
                    'accuracy': r.accuracy,
                    'loss': r.loss,
                    'task_accuracies': r.task_accuracies,
                    'seed': r.seed,
                    'samples_removed': r.samples_removed
                }
                for r in results
            ]
            for condition, results in validator.results.items()
        }
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {args.output}")

    # Create plot
    try:
        plot_results(analysis, args.plot)
    except Exception as e:
        logger.warning(f"Could not create plot: {e}")

    # Print summary for paper
    logger.info("\n" + "="*80)
    logger.info("SUMMARY FOR ICLR PAPER")
    logger.info("="*80)
    logger.info(f"Verdict: {analysis['verdict']}")
    logger.info(f"\nResults:")
    logger.info(f"  Baseline:          {analysis['baseline']['mean']:.2f} ± {analysis['baseline']['std']:.2f}%")
    logger.info(f"  Random filtering:  {analysis['random']['mean']:.2f} ± {analysis['random']['std']:.2f}%")
    logger.info(f"  Conflict filtering: {analysis['conflict']['mean']:.2f} ± {analysis['conflict']['std']:.2f}%")
    logger.info(f"\nImprovement over random: {analysis['improvement_vs_random']:+.2f}%")
    logger.info(f"Statistical significance: p={analysis['p_value_vs_random']:.4f}")
    logger.info(f"Effect size (Cohen's d): {analysis['cohens_d_vs_random']:.3f}")

    if analysis['is_significant']:
        logger.info("\n✓ Results are statistically significant (p < 0.05)")
        logger.info("  This supports the claim that conflict-based filtering outperforms random.")
    else:
        logger.info("\n⚠ Results are NOT statistically significant")
        logger.info("  Consider:")
        logger.info("    - Increasing --num-seeds (current: {})".format(args.num_seeds))
        logger.info("    - Using larger --num-samples")
        logger.info("    - Trying different tasks with more conflicts")


if __name__ == '__main__':
    main()
