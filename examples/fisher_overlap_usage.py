#!/usr/bin/env python3
"""
Example: Using Fisher Overlap Analysis Independently

This script demonstrates how to use the fisher.analyze_fisher_overlap function
in your own Python code to analyze parameter sharing across tasks.
"""

import torch
from fisher import analyze_fisher_overlap, extract_active_groups

# Example 1: Simple usage with Welford accumulators
def example_basic_usage():
    """
    Basic usage: Analyze overlap directly from Welford accumulators.
    """
    print("="*80)
    print("EXAMPLE 1: Basic Usage")
    print("="*80)

    # Assuming you have a BombshellMetrics instance with Fisher computed
    # from BombshellMetrics import BombshellMetrics
    # bombshell = BombshellMetrics()
    # ... compute Fisher for multiple tasks ...

    # Then analyze overlap:
    # overlap_results = analyze_fisher_overlap(bombshell.welford_accumulators)

    # The results contain:
    # - comparisons: Dict with pairwise task comparisons
    # - task_active_counts: Number of active groups per task
    # - summary: High-level statistics

    # Example output structure:
    example_results = {
        'comparisons': {
            'math_vs_general': {
                'task1': 'math',
                'task2': 'general',
                'task1_active_groups': 61,
                'task2_active_groups': 6,
                'overlap_count': 4,
                'jaccard_similarity': 0.063,
                'relationship': 'partial_overlap',
                'different_attention_patterns': True
            }
        },
        'task_active_counts': {'math': 61, 'general': 6},
        'summary': {
            'num_tasks': 2,
            'average_jaccard_similarity': 0.063
        }
    }

    print("\nExpected output structure:")
    print(f"  Jaccard similarity: {example_results['summary']['average_jaccard_similarity']:.2%}")
    print(f"  Math active groups: {example_results['task_active_counts']['math']}")
    print(f"  General active groups: {example_results['task_active_counts']['general']}")
    print(f"  Overlap: {example_results['comparisons']['math_vs_general']['overlap_count']}")


# Example 2: Extract active groups first, then analyze
def example_two_step_process():
    """
    Two-step process: Extract active groups, then analyze overlap.

    Useful when you want to inspect or modify the active groups before analysis.
    """
    print("\n" + "="*80)
    print("EXAMPLE 2: Two-Step Process")
    print("="*80)

    # Step 1: Extract active groups (those with M2 > threshold)
    # task_active_groups = extract_active_groups(
    #     bombshell.welford_accumulators,
    #     threshold=1e-10
    # )

    # This returns: {'task1': set(...), 'task2': set(...)}
    # You can inspect these sets:
    # print(f"Math active groups: {len(task_active_groups['math'])}")
    # print(f"Example groups: {list(task_active_groups['math'])[:5]}")

    # Step 2: Analyze with custom parameters
    # overlap_results = analyze_fisher_overlap(
    #     bombshell.welford_accumulators,
    #     threshold=1e-8,  # Different threshold
    #     log_results=False  # Disable logging
    # )

    print("\nUse this when you want to:")
    print("  • Inspect active groups before analysis")
    print("  • Use different thresholds")
    print("  • Disable automatic logging")


# Example 3: In your own training loop
def example_in_training_loop():
    """
    Using overlap analysis during training to monitor task interference.
    """
    print("\n" + "="*80)
    print("EXAMPLE 3: Integration in Training Loop")
    print("="*80)

    code_example = '''
# In your training script:
from fisher import analyze_fisher_overlap
from BombshellMetrics import BombshellMetrics

# Initialize
bombshell = BombshellMetrics()
model = YourModel()

# Train on multiple tasks
for epoch in range(num_epochs):
    # Task 1: Math
    for batch in math_loader:
        loss = train_step(model, batch)
        bombshell.compute_fisher_welford(
            model, batch, task_name='math'
        )

    # Task 2: General
    for batch in general_loader:
        loss = train_step(model, batch)
        bombshell.compute_fisher_welford(
            model, batch, task_name='general'
        )

    # Analyze overlap after each epoch
    if epoch % 5 == 0:
        overlap_results = analyze_fisher_overlap(
            bombshell.welford_accumulators,
            log_results=True
        )

        # Check for interference
        for comp_key, comp_data in overlap_results['comparisons'].items():
            jaccard = comp_data['jaccard_similarity']
            if jaccard > 0.6:
                print(f"⚠️  HIGH INTERFERENCE: {comp_key} ({jaccard:.1%})")
            elif comp_data['different_attention_patterns']:
                print(f"✓ Task specialization detected in {comp_key}")
    '''

    print(code_example)


# Example 4: Programmatic access to results
def example_programmatic_access():
    """
    Accessing specific results programmatically.
    """
    print("\n" + "="*80)
    print("EXAMPLE 4: Programmatic Access")
    print("="*80)

    code_example = '''
# After running analysis:
overlap_results = analyze_fisher_overlap(welford_accumulators)

# Access specific comparison
math_vs_gen = overlap_results['comparisons']['math_vs_general']

# Get key metrics
jaccard = math_vs_gen['jaccard_similarity']
overlap_count = math_vs_gen['overlap_count']
relationship = math_vs_gen['relationship']

# Check relationship type
if math_vs_gen['relationship'].endswith('subset'):
    print(f"{math_vs_gen['task2']} contains all of {math_vs_gen['task1']}")
elif math_vs_gen['relationship'] == 'partial_overlap':
    print("Tasks use partially overlapping parameters")

# Get example groups
shared = math_vs_gen['shared_groups']
math_only = math_vs_gen['only_task1_examples']
general_only = math_vs_gen['only_task2_examples']

print(f"Shared groups: {shared[:3]}")
print(f"Math-only groups: {math_only[:3]}")

# Check attention specialization
if math_vs_gen['different_attention_patterns']:
    attn_math = math_vs_gen['attention_only_task1']
    attn_gen = math_vs_gen['attention_only_task2']
    print(f"Different attention: {attn_math} math-only, {attn_gen} general-only")
    '''

    print(code_example)


# Example 5: Custom analysis function
def example_custom_analysis():
    """
    Building custom analysis on top of the overlap results.
    """
    print("\n" + "="*80)
    print("EXAMPLE 5: Custom Analysis")
    print("="*80)

    code_example = '''
from fisher import analyze_fisher_overlap, extract_active_groups
from fisher.core.overlap_analysis import analyze_parameter_type_overlap

def analyze_task_specialization(welford_accumulators):
    """Custom function to analyze how tasks specialize."""

    # Get overlap results
    overlap_results = analyze_fisher_overlap(
        welford_accumulators,
        log_results=False
    )

    # Extract active groups
    active_groups = extract_active_groups(welford_accumulators)

    # Analyze by parameter type
    type_breakdown = analyze_parameter_type_overlap(active_groups)

    # Custom metrics
    for param_type, counts in type_breakdown.items():
        both = counts.get('both', 0)
        task1 = counts.get('task1', 0)
        task2 = counts.get('task2', 0)
        total = both + task1 + task2

        if total > 0:
            specialization = (task1 + task2) / total
            print(f"{param_type}: {specialization:.1%} specialized")

    return {
        'overlap_results': overlap_results,
        'type_breakdown': type_breakdown
    }

# Use it:
results = analyze_task_specialization(bombshell.welford_accumulators)
    '''

    print(code_example)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("FISHER OVERLAP ANALYSIS - USAGE EXAMPLES")
    print("="*80)

    example_basic_usage()
    example_two_step_process()
    example_in_training_loop()
    example_programmatic_access()
    example_custom_analysis()

    print("\n" + "="*80)
    print("KEY POINTS")
    print("="*80)
    print("""
1. Import from fisher module:
   from fisher import analyze_fisher_overlap

2. Pass Welford accumulators:
   overlap_results = analyze_fisher_overlap(welford_accumulators)

3. Results include:
   - comparisons: Per-task-pair statistics
   - task_active_counts: Active groups per task
   - summary: High-level metrics

4. Configure with parameters:
   - threshold: Minimum M2 value (default: 1e-10)
   - log_results: Enable/disable logging (default: True)

5. Available from any Python script - not just unified_model_analysis!
    """)