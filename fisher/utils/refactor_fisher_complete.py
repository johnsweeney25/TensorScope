#!/usr/bin/env python3
"""
Script to complete the Fisher refactoring by:
1. Removing Fisher methods from original files
2. Copying non-Fisher methods to refactored files
"""

import re

def remove_fisher_methods_from_bombshell():
    """Remove all Fisher-related methods from BombshellMetrics.py."""

    # List of Fisher method names and their line numbers (from grep output)
    fisher_methods = [
        (135, 'cleanup_fisher_ema'),
        (5149, 'update_fisher_ema'),
        (5242, 'get_bias_corrected_fisher_ema'),
        (5269, '_estimate_fisher_diagonal'),
        (5367, '_get_top_coordinates_from_fisher'),
        (5664, 'get_top_fisher_directions'),
        (5794, '_ensure_fisher_ema_initialized'),
        (5836, 'compute_fisher_importance'),
        (5899, 'compare_task_fisher'),
        (5993, 'compute_fisher_overlap'),
        (6064, 'get_fisher_pruning_masks'),
        (6118, 'scale_by_fisher'),
        (6157, 'reset_fisher_ema'),
        (6204, 'fisher_weighted_merge'),
        (6249, 'estimate_fisher_uncertainty'),
    ]

    # Read the original file
    with open('BombshellMetrics.py', 'r') as f:
        lines = f.readlines()

    # Track which lines to keep
    keep_lines = [True] * len(lines)

    # Mark Fisher method lines for deletion
    for start_line, method_name in fisher_methods:
        # Find the method start (0-indexed)
        idx = start_line - 1

        # Mark lines for deletion from method start until next method or end of indentation
        in_method = False
        brace_count = 0

        if idx < len(lines) and 'def ' + method_name in lines[idx]:
            in_method = True
            keep_lines[idx] = False

            # Continue marking lines until we hit the next method at same indentation
            for i in range(idx + 1, len(lines)):
                line = lines[i]

                # Check if we've hit another method at the same indentation level
                if line.startswith('    def ') and not line.startswith('        '):
                    break

                # Check if we've hit a class-level comment or attribute
                if not line.startswith(' ') and line.strip() and not line.startswith('#'):
                    break

                # Mark this line for deletion
                keep_lines[i] = False

    # Also remove Fisher-related attributes from __init__
    new_lines = []
    skip_fisher_attrs = False

    for i, line in enumerate(lines):
        if keep_lines[i]:
            # Skip Fisher-related attributes in __init__
            if 'self.fisher_ema' in line or 'self.fisher_steps' in line:
                continue
            if 'self.fisher_' in line:
                continue
            new_lines.append(line)

    # Write the cleaned file
    with open('BombshellMetrics_no_fisher.py', 'w') as f:
        f.writelines(new_lines)

    print(f"Removed {len(fisher_methods)} Fisher methods from BombshellMetrics.py")
    print(f"Original lines: {len(lines)}, New lines: {len(new_lines)}")
    print(f"Removed {len(lines) - len(new_lines)} lines")

def remove_fisher_methods_from_modularity():
    """Remove all Fisher-related methods from ModularityMetrics.py."""

    # Read the original file
    with open('ModularityMetrics.py', 'r') as f:
        lines = f.readlines()

    # Find and remove Fisher methods
    new_lines = []
    skip_method = False
    current_indent = 0

    for i, line in enumerate(lines):
        # Check if this is a Fisher method
        if re.search(r'def.*[Ff]isher|def _estimate_fisher|def update_fisher_ema', line):
            skip_method = True
            # Determine the indentation level
            current_indent = len(line) - len(line.lstrip())
            continue

        # If we're skipping a method, check if we've reached the next method
        if skip_method:
            # Check if this line starts a new method at the same or lower indentation
            if line.strip().startswith('def '):
                line_indent = len(line) - len(line.lstrip())
                if line_indent <= current_indent:
                    skip_method = False
            elif not line.strip() or line.strip().startswith('#'):
                # Empty lines and comments - keep checking
                pass
            elif len(line) - len(line.lstrip()) <= current_indent and line.strip():
                # Non-empty line at same or lower indentation - method ended
                skip_method = False

        # Skip Fisher-related attributes
        if 'self.fisher_ema' in line or 'self.fisher_steps' in line:
            continue

        if not skip_method:
            new_lines.append(line)

    # Write the cleaned file
    with open('ModularityMetrics_no_fisher.py', 'w') as f:
        f.writelines(new_lines)

    print(f"\nRemoved Fisher methods from ModularityMetrics.py")
    print(f"Original lines: {len(lines)}, New lines: {len(new_lines)}")
    print(f"Removed {len(lines) - len(new_lines)} lines")

def copy_non_fisher_to_refactored():
    """Copy non-Fisher methods from cleaned files to refactored versions."""

    # For BombshellMetrics
    print("\nUpdating BombshellMetrics_refactored.py with non-Fisher methods...")

    # Read cleaned BombshellMetrics
    with open('BombshellMetrics_no_fisher.py', 'r') as f:
        bombshell_lines = f.readlines()

    # Find where class definition ends and methods begin
    class_start = -1
    for i, line in enumerate(bombshell_lines):
        if 'class BombshellMetrics' in line:
            class_start = i
            break

    # Extract methods (skip __init__ and class definition)
    methods_start = -1
    for i in range(class_start, len(bombshell_lines)):
        if 'def __init__' in bombshell_lines[i]:
            # Find end of __init__
            indent_level = len(bombshell_lines[i]) - len(bombshell_lines[i].lstrip())
            for j in range(i+1, len(bombshell_lines)):
                if bombshell_lines[j].strip().startswith('def '):
                    line_indent = len(bombshell_lines[j]) - len(bombshell_lines[j].lstrip())
                    if line_indent <= indent_level:
                        methods_start = j
                        break
            break

    if methods_start > 0:
        non_fisher_methods = bombshell_lines[methods_start:]

        # Update refactored file - insert methods before the final class closing
        with open('BombshellMetrics_refactored.py', 'r') as f:
            refactored = f.readlines()

        # Find where to insert (before end of class)
        insert_pos = len(refactored)

        # Insert the non-Fisher methods
        final_content = refactored[:insert_pos] + ['\n    # ===== NON-FISHER METHODS FROM ORIGINAL =====\n'] + non_fisher_methods

        with open('BombshellMetrics_refactored.py', 'w') as f:
            f.writelines(final_content)

        print(f"Added {len(non_fisher_methods)} lines of non-Fisher methods to BombshellMetrics_refactored.py")

    # For ModularityMetrics
    print("\nUpdating ModularityMetrics_refactored.py with non-Fisher methods...")

    # Read cleaned ModularityMetrics
    with open('ModularityMetrics_no_fisher.py', 'r') as f:
        modularity_lines = f.readlines()

    # Similar process for ModularityMetrics
    class_start = -1
    for i, line in enumerate(modularity_lines):
        if 'class ExtendedModularityMetrics' in line:
            class_start = i
            break

    methods_start = -1
    for i in range(class_start, len(modularity_lines)):
        if 'def __init__' in modularity_lines[i]:
            indent_level = len(modularity_lines[i]) - len(modularity_lines[i].lstrip())
            for j in range(i+1, len(modularity_lines)):
                if modularity_lines[j].strip().startswith('def '):
                    line_indent = len(modularity_lines[j]) - len(modularity_lines[j].lstrip())
                    if line_indent <= indent_level:
                        methods_start = j
                        break
            break

    if methods_start > 0:
        # Extract non-Fisher methods, excluding ones already in refactored file
        existing_methods = ['compute_fisher_weighted_damage', 'compute_fisher_damage_with_asymmetry',
                          'compute_subspace_distance', '_linear_cka', '_with_labels', '_to_model_device']

        non_fisher_methods = []
        current_method = []
        in_method = False

        for line in modularity_lines[methods_start:]:
            if line.strip().startswith('def '):
                # Check if this method already exists in refactored
                method_name = line.strip().split('(')[0].replace('def ', '')
                if method_name not in existing_methods:
                    if current_method:
                        non_fisher_methods.extend(current_method)
                    current_method = [line]
                    in_method = True
                else:
                    in_method = False
                    current_method = []
            elif in_method:
                current_method.append(line)

        # Don't forget the last method
        if current_method:
            non_fisher_methods.extend(current_method)

        # Update refactored file
        with open('ModularityMetrics_refactored.py', 'r') as f:
            refactored = f.readlines()

        # Find where to insert (before end of class)
        insert_pos = len(refactored)

        # Insert the non-Fisher methods
        final_content = refactored[:insert_pos] + ['\n    # ===== NON-FISHER METHODS FROM ORIGINAL =====\n'] + non_fisher_methods

        with open('ModularityMetrics_refactored.py', 'w') as f:
            f.writelines(final_content)

        print(f"Added {len(non_fisher_methods)} lines of non-Fisher methods to ModularityMetrics_refactored.py")

if __name__ == "__main__":
    print("Starting Fisher refactoring cleanup...")

    # Step 1: Remove Fisher methods from original files
    remove_fisher_methods_from_bombshell()
    remove_fisher_methods_from_modularity()

    # Step 2: Copy non-Fisher methods to refactored files
    copy_non_fisher_to_refactored()

    print("\n✅ Refactoring cleanup complete!")
    print("\nNext steps:")
    print("1. Review BombshellMetrics_no_fisher.py and ModularityMetrics_no_fisher.py")
    print("2. Rename them to replace the originals when satisfied")
    print("3. Test the refactored versions")