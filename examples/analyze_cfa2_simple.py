#!/usr/bin/env python3
"""
Simple analysis of CFA-2 to understand metric execution and result saving.
No external dependencies required.
"""

import re
import os
from pathlib import Path
from datetime import datetime


def analyze_cfa2():
    """Analyze CFA-2.py code to understand metrics and outputs."""
    
    print("="*80)
    print("CFA-2 METRIC ANALYSIS")
    print("="*80)
    
    # Read CFA-2.py
    with open('CFA-2.py', 'r') as f:
        code = f.read()
    
    # Find all metric function calls
    print("\n📊 METRICS BEING EXECUTED:\n")
    
    metric_patterns = [
        ('ExtendedModularityMetrics', r'self\.metrics\.compute_(\w+)'),
        ('BombshellMetrics', r'self\.bombshell_metrics\.compute_(\w+)'),
        ('InformationTheoryMetrics', r'self\.info_theory_metrics\.compute_(\w+)'),
        ('ICLRMetrics', r'self\.iclr_metrics\.compute_(\w+)'),
        ('ICLRCriticalMetrics', r'self\.iclr_critical_metrics\.compute_(\w+)'),
        ('EmergentPhenomenaMetrics', r'self\.emergent_metrics\.compute_(\w+)')
    ]
    
    total_metrics = 0
    metric_details = {}
    
    for module_name, pattern in metric_patterns:
        matches = re.findall(pattern, code)
        unique_matches = list(set(matches))
        
        if unique_matches:
            print(f"✓ {module_name}: {len(unique_matches)} metrics")
            for metric in sorted(unique_matches):
                print(f"  - compute_{metric}")
            metric_details[module_name] = unique_matches
            total_metrics += len(unique_matches)
        else:
            print(f"✗ {module_name}: Not used")
            metric_details[module_name] = []
    
    print(f"\n📈 Total metrics executed: {total_metrics}")
    
    # Find output files
    print("\n💾 OUTPUT FILES:\n")
    
    file_patterns = [
        (r"\.to_csv\('([^']+)'\)", "CSV"),
        (r'\.to_csv\("([^"]+)"\)', "CSV"),
        (r"json\.dump\([^,]+,\s*open\('([^']+)'", "JSON"),
        (r'with open\(["\']([^"\']+\.json)["\']', "JSON"),
        (r"plt\.savefig\('([^']+)'\)", "Plot"),
        (r'plt\.savefig\("([^"]+)"\)', "Plot")
    ]
    
    output_files = {}
    for pattern, file_type in file_patterns:
        matches = re.findall(pattern, code)
        for match in matches:
            if not match.startswith('/') and not match.startswith('http'):
                if file_type not in output_files:
                    output_files[file_type] = []
                output_files[file_type].append(match)
    
    for file_type, files in output_files.items():
        print(f"{file_type} files:")
        for f in set(files):
            print(f"  - {f}")
    
    if not output_files:
        print("⚠️  No output files found in code")
    
    # Check existing result files
    print("\n📁 EXISTING RESULT FILES:\n")
    
    extensions = ['.csv', '.json', '.png', '.pdf', '.txt']
    found_files = []
    
    for ext in extensions:
        files = list(Path('.').glob(f'*{ext}'))
        if files:
            print(f"{ext.upper()} files found: {len(files)}")
            for f in files[:5]:  # Show first 5
                size_kb = os.path.getsize(f) / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")
            if len(files) > 5:
                print(f"  ... and {len(files)-5} more")
            found_files.extend(files)
    
    if not found_files:
        print("No result files found in current directory")
    
    return metric_details, output_files


def check_available_metrics():
    """Check what metrics are available in each module."""
    
    print("\n" + "="*80)
    print("AVAILABLE METRICS IN MODULES")
    print("="*80)
    
    modules_to_check = [
        'BombshellMetrics.py',
        'InformationTheoryMetrics.py',
        'ICLRMetrics.py',
        'ModularityMetrics.py',
        'ICLR_missing_metrics.py',
        'EmergentPhenomena.py'
    ]
    
    available_metrics = {}
    
    for module_file in modules_to_check:
        if os.path.exists(module_file):
            with open(module_file, 'r') as f:
                content = f.read()
            
            # Find all compute_ functions
            pattern = r'def (compute_\w+)\('
            matches = re.findall(pattern, content)
            
            if matches:
                print(f"\n{module_file}: {len(matches)} methods available")
                available_metrics[module_file] = matches
                for method in matches[:10]:  # Show first 10
                    print(f"  - {method}")
                if len(matches) > 10:
                    print(f"  ... and {len(matches)-10} more")
        else:
            print(f"\n{module_file}: Not found")
    
    return available_metrics


def generate_recommendations():
    """Generate specific recommendations."""
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    print("""
🎯 KEY IMPROVEMENTS NEEDED:

1. COMPREHENSIVE METRIC EXECUTION
   Current: Only executing subset of available metrics
   Solution: Create metric registry and execute all systematically
   
2. ORGANIZED RESULT SAVING
   Current: Results saved to fixed filenames (overwriting previous runs)
   Solution: Use timestamped directories for each run
   
3. STRUCTURED OUTPUT
   Current: Limited CSV/JSON output
   Solution: 
   - Raw data (JSON) for complete record
   - Summary tables (CSV) for analysis
   - Visualizations (PNG) for insights
   - Reports (MD/HTML) for documentation
   
4. RESULT VISUALIZATION
   Current: No automatic visualization
   Solution: Generate plots for:
   - Metric evolution across checkpoints
   - Model comparisons
   - Correlation heatmaps
   - Distribution analysis
   
5. AUTOMATED INSIGHTS
   Current: Manual interpretation needed
   Solution: Auto-generate reports with:
   - Key findings
   - Anomaly detection
   - Trend analysis
   - Recommendations

📝 IMPLEMENTATION STEPS:

Step 1: Create enhanced wrapper script
Step 2: Build metric registry from all modules  
Step 3: Add systematic metric execution
Step 4: Implement hierarchical result storage
Step 5: Add visualization generation
Step 6: Create automated reporting

✅ QUICK START:
   Use the enhanced scripts created:
   - CFA-2-enhanced.py (comprehensive version)
   - run_cfa2_enhanced.py (monitoring wrapper)
""")


def main():
    """Main execution."""
    
    print("🔍 ANALYZING CFA-2 METRIC EXECUTION AND RESULT SAVING")
    print("="*80)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Analyze current implementation
    metrics_used, outputs = analyze_cfa2()
    
    # Check available metrics
    available = check_available_metrics()
    
    # Calculate coverage
    print("\n" + "="*80)
    print("COVERAGE ANALYSIS")
    print("="*80)
    
    for module_file, available_methods in available.items():
        module_name = module_file.replace('.py', '')
        
        # Map file names to class names used in CFA-2
        name_mapping = {
            'BombshellMetrics': 'BombshellMetrics',
            'InformationTheoryMetrics': 'InformationTheoryMetrics',
            'ICLRMetrics': 'ICLRMetrics',
            'ModularityMetrics': 'ExtendedModularityMetrics',
            'ICLR_missing_metrics': 'ICLRCriticalMetrics',
            'EmergentPhenomena': 'EmergentPhenomenaMetrics'
        }
        
        class_name = name_mapping.get(module_name, module_name)
        used_methods = metrics_used.get(class_name, [])
        
        if available_methods:
            coverage = len(used_methods) / len(available_methods) * 100
            print(f"\n{module_name}:")
            print(f"  Available: {len(available_methods)} methods")
            print(f"  Used: {len(used_methods)} methods")
            print(f"  Coverage: {coverage:.1f}%")
            
            if coverage < 100:
                unused = set([m.replace('compute_', '') for m in available_methods]) - set(used_methods)
                if unused:
                    print(f"  Unused methods: {len(unused)}")
                    for method in list(unused)[:5]:
                        print(f"    - compute_{method}")
                    if len(unused) > 5:
                        print(f"    ... and {len(unused)-5} more")
    
    # Generate recommendations
    generate_recommendations()
    
    print("\n" + "="*80)
    print("✨ Analysis complete!")
    print("="*80)


if __name__ == "__main__":
    main()