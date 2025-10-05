#!/usr/bin/env python3
"""
Fisher Group Overlap Analysis

Provides functions to analyze overlap between Fisher groups across different tasks.
This reveals which model parameters are shared vs specialized for different tasks.
"""

import torch
from typing import Dict, Set, List, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


def extract_active_groups(
    welford_accumulators: Dict,
    threshold: float = 1e-10
) -> Dict[str, Set[str]]:
    """
    Extract groups with non-zero M2 (variance) for each task.

    Args:
        welford_accumulators: Dictionary mapping task names to Welford accumulators
        threshold: Minimum M2 value to consider a group "active"

    Returns:
        Dictionary mapping task names to sets of active group keys
    """
    task_active_groups = {}

    for task_key in welford_accumulators.keys():
        active_groups = []
        accumulator = welford_accumulators[task_key]

        if hasattr(accumulator, 'M2'):
            for param_key, m2_val in accumulator.M2.items():
                # Convert to scalar if needed
                if torch.is_tensor(m2_val):
                    m2_val = m2_val.item()

                # Check if variance is non-zero
                if m2_val > threshold:
                    # Clean the key (remove task prefix if present)
                    clean_key = param_key.split('|', 1)[-1] if '|' in param_key else param_key
                    active_groups.append(clean_key)

        task_active_groups[task_key] = set(active_groups)

    return task_active_groups


def compute_overlap_statistics(
    set1: Set[str],
    set2: Set[str],
    task1_name: str,
    task2_name: str
) -> Dict:
    """
    Compute overlap statistics between two sets of active groups.

    Args:
        set1: Active groups for task 1
        set2: Active groups for task 2
        task1_name: Name of task 1
        task2_name: Name of task 2

    Returns:
        Dictionary with overlap statistics
    """
    overlap = len(set1 & set2)
    only_1 = len(set1 - set2)
    only_2 = len(set2 - set1)
    union = len(set1 | set2)

    # Determine relationship
    if set1.issubset(set2):
        relationship = f"{task1_name}_subset_of_{task2_name}"
    elif set2.issubset(set1):
        relationship = f"{task2_name}_subset_of_{task1_name}"
    else:
        relationship = "partial_overlap"

    # Check attention head specialization
    attn_only_1 = sum(1 for g in (set1 - set2) if 'self_attn' in g or 'attn' in g)
    attn_only_2 = sum(1 for g in (set2 - set1) if 'self_attn' in g or 'attn' in g)
    different_attention = (attn_only_1 > 0 or attn_only_2 > 0)

    return {
        'task1': task1_name,
        'task2': task2_name,
        'task1_active_groups': len(set1),
        'task2_active_groups': len(set2),
        'overlap_count': overlap,
        'only_task1': only_1,
        'only_task2': only_2,
        'total_unique': union,
        'jaccard_similarity': overlap / max(1, union),
        'relationship': relationship,
        'different_attention_patterns': different_attention,
        'attention_only_task1': attn_only_1,
        'attention_only_task2': attn_only_2,
        # Store example group names
        'shared_groups': sorted(list(set1 & set2))[:10],  # First 10
        'only_task1_examples': sorted(list(set1 - set2))[:5],  # First 5
        'only_task2_examples': sorted(list(set2 - set1))[:5]   # First 5
    }


def analyze_fisher_overlap(
    welford_accumulators: Dict,
    threshold: float = 1e-10,
    log_results: bool = True
) -> Dict:
    """
    Analyze overlap between Fisher groups across multiple tasks.

    This function determines which model parameters are shared vs specialized
    for different tasks, based on which groups have non-zero variance (M2).

    Args:
        welford_accumulators: Dictionary mapping task names to Welford accumulators
        threshold: Minimum M2 value to consider a group "active" (default: 1e-10)
        log_results: Whether to log human-readable results (default: True)

    Returns:
        Dictionary with overlap analysis results:
        - 'comparisons': Dict of pairwise task comparisons
        - 'task_active_counts': Dict of active group counts per task
        - 'summary': High-level summary statistics

    Example:
        >>> from fisher.core.overlap_analysis import analyze_fisher_overlap
        >>> results = analyze_fisher_overlap(bombshell.welford_accumulators)
        >>> print(results['comparisons']['math_vs_general']['jaccard_similarity'])
        0.063
    """
    if not welford_accumulators or len(welford_accumulators) < 2:
        logger.warning("Need at least 2 tasks for overlap analysis")
        return {
            'comparisons': {},
            'task_active_counts': {},
            'summary': {'error': 'Insufficient tasks for comparison'}
        }

    if log_results:
        logger.info("\n📊 FISHER GROUP OVERLAP ANALYSIS:")

    # Extract active groups per task
    task_active_groups = extract_active_groups(welford_accumulators, threshold)

    # Store results
    comparisons = {}
    task_names = list(task_active_groups.keys())

    # Compare all pairs of tasks
    for i, task1 in enumerate(task_names):
        for task2 in task_names[i+1:]:
            set1 = task_active_groups[task1]
            set2 = task_active_groups[task2]

            # Compute statistics
            stats = compute_overlap_statistics(set1, set2, task1, task2)

            # Store with standardized key
            comparison_key = f"{task1}_vs_{task2}"
            comparisons[comparison_key] = stats

            # Log if requested
            if log_results:
                logger.info(f"\n{task1} vs {task2}:")
                logger.info(f"  • {task1}: {len(set1)} active groups")
                logger.info(f"  • {task2}: {len(set2)} active groups")
                logger.info(f"  • Overlap: {stats['overlap_count']} groups")

                if stats['relationship'].endswith('subset'):
                    logger.info(f"  ✓ {stats['relationship'].replace('_', ' ')}")
                else:
                    logger.info(f"  ✗ Partial overlap ({stats['overlap_count']}/{stats['total_unique']} = {100*stats['jaccard_similarity']:.1f}%)")

                if stats['different_attention_patterns']:
                    logger.info(f"  🔍 Different attention patterns detected!")

    # Create summary
    task_counts = {task: len(groups) for task, groups in task_active_groups.items()}

    # Compute average overlap across all pairs
    if comparisons:
        avg_jaccard = sum(c['jaccard_similarity'] for c in comparisons.values()) / len(comparisons)
        avg_overlap = sum(c['overlap_count'] for c in comparisons.values()) / len(comparisons)
    else:
        avg_jaccard = 0
        avg_overlap = 0

    summary = {
        'num_tasks': len(task_names),
        'num_comparisons': len(comparisons),
        'average_jaccard_similarity': avg_jaccard,
        'average_overlap_count': avg_overlap,
        'task_names': task_names
    }

    return {
        'comparisons': comparisons,
        'task_active_counts': task_counts,
        'summary': summary
    }


def analyze_parameter_type_overlap(
    task_active_groups: Dict[str, Set[str]]
) -> Dict[str, Dict]:
    """
    Break down overlap by parameter type (attention, MLP, norm, etc.).

    Args:
        task_active_groups: Dictionary mapping task names to sets of active groups

    Returns:
        Dictionary with overlap breakdown by parameter type
    """
    param_types = ['attention_q', 'attention_k', 'attention_v', 'attention_o',
                  'mlp_gate', 'mlp_up', 'mlp_down', 'norm', 'embedding', 'other']

    type_breakdown = defaultdict(lambda: defaultdict(int))

    task_names = list(task_active_groups.keys())
    if len(task_names) < 2:
        return {}

    task1, task2 = task_names[0], task_names[1]
    set1 = task_active_groups[task1]
    set2 = task_active_groups[task2]
    union = set1 | set2

    # Categorize each group
    for group in union:
        in_task1 = group in set1
        in_task2 = group in set2

        # Determine parameter type
        param_type = "other"
        if 'self_attn' in group or 'attn' in group:
            if 'q_proj' in group or 'query' in group:
                param_type = "attention_q"
            elif 'k_proj' in group or 'key' in group:
                param_type = "attention_k"
            elif 'v_proj' in group or 'value' in group:
                param_type = "attention_v"
            elif 'o_proj' in group or 'output' in group:
                param_type = "attention_o"
        elif 'mlp' in group:
            if 'gate' in group:
                param_type = "mlp_gate"
            elif 'up' in group:
                param_type = "mlp_up"
            elif 'down' in group:
                param_type = "mlp_down"
        elif 'norm' in group:
            param_type = "norm"
        elif 'embed' in group:
            param_type = "embedding"

        if in_task1 and in_task2:
            type_breakdown[param_type]['both'] += 1
        elif in_task1:
            type_breakdown[param_type][task1] += 1
        else:
            type_breakdown[param_type][task2] += 1

    return dict(type_breakdown)