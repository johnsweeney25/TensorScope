#!/usr/bin/env python3
"""
Test Fisher metrics integration with report generation
"""
import json
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from unified_model_analysis import UnifiedModelAnalyzer, UnifiedConfig, ModelSpec, AnalysisResults, ModelResults, MetricResult

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 8)
        self.fc1 = nn.Linear(8, 5)
        self.fc2 = nn.Linear(5, 10)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        batch_size, seq_len = input_ids.shape
        x = self.embed(input_ids)
        x = x.view(batch_size * seq_len, -1)
        x = torch.relu(self.fc1(x))
        logits = self.fc2(x)
        logits = logits.view(batch_size, seq_len, -1)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

        class Output:
            pass
        output = Output()
        output.loss = loss
        output.logits = logits
        return output

def create_mock_fisher_results():
    """Create mock Fisher analysis results"""
    return {
        'summary': {
            'ema_computed': True,
            'tasks_analyzed': ['task1', 'task2'],
            'total_parameters_analyzed': 5,
            'computation_time': 0.123
        },
        'importance': {
            'task1': {
                'parameter_importance': {'fc1.weight': 0.8, 'fc2.weight': 0.6},
                'layer_importance': {'fc1': 0.8, 'fc2': 0.6},
                'top_5_critical': [('fc1.weight', 0.8), ('fc2.weight', 0.6)]
            },
            'task2': {
                'parameter_importance': {'fc1.weight': 0.7, 'fc2.weight': 0.9},
                'layer_importance': {'fc1': 0.7, 'fc2': 0.9},
                'top_5_critical': [('fc2.weight', 0.9), ('fc1.weight', 0.7)]
            }
        },
        'comparison': {
            'task1': 'task1',
            'task2': 'task2',
            'divergence': 0.234,
            'correlation': 0.567,
            'magnitude_ratio': 1.123
        },
        'overlap_analysis': {
            'overlap_percentage': 45.6,
            'high_conflict_layers': ['fc2'],
            'moderate_conflict_layers': [],
            'safe_merge_layers': ['fc1'],
            'conflict_percentage': 50.0,
            'per_layer_overlap': {'fc1': 0.2, 'fc2': 0.8},
            'tasks_compared': ['task1', 'task2']
        },
        'recommendations': {
            'merge_strategy': 'Use task-specific routing for fc2',
            'training_advice': 'Consider elastic weight consolidation',
            'risk_assessment': 'Moderate interference risk'
        }
    }

def test_report_integration():
    print("Testing Fisher metrics integration with report generation...")
    
    # Create output directory
    output_dir = Path("./test_fisher_report_output")
    output_dir.mkdir(exist_ok=True)
    
    # Create mock analysis results
    print("\n1. Creating mock analysis results with Fisher data...")
    
    # Create a minimal results object
    results = AnalysisResults(
        timestamp=datetime.now().isoformat(),
        config=UnifiedConfig(output_dir=output_dir, save_intermediate=True),
        model_results={
            'test_model': ModelResults(
                model_id='test_model',
                metrics={
                    'fisher_analysis_comprehensive': MetricResult(
                        value=create_mock_fisher_results(),
                        name='fisher_analysis_comprehensive',
                        module='fisher_analysis',
                        compute_time=0.123
                    )
                },
                compute_time=1.0
            )
        },
        group_analyses={},
        pairwise_comparisons=None,
        global_correlations=None
    )
    
    # Also set the fisher_analysis attribute
    results.model_results['test_model'].fisher_analysis = create_mock_fisher_results()
    
    print("   ✓ Created mock results with Fisher analysis")
    
    # Test JSON serialization
    print("\n2. Testing JSON serialization...")
    
    # Create analyzer just for saving
    config = UnifiedConfig(output_dir=output_dir, save_intermediate=True, generate_report=False)
    analyzer = UnifiedModelAnalyzer(config)
    
    # Save results
    analyzer._save_results(results)
    
    # Find the saved JSON file
    json_files = list(output_dir.glob("analysis_*.json"))
    if not json_files:
        print("   ✗ No JSON file created")
        return False
    
    json_path = json_files[0]
    print(f"   ✓ JSON saved to {json_path}")
    
    # Load and verify JSON contains Fisher data
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if 'fisher_analysis' not in data:
        print("   ✗ Fisher analysis not in JSON")
        return False
    
    if 'test_model' not in data['fisher_analysis'] and 'global_summary' not in data['fisher_analysis']:
        print("   ✗ Fisher data not properly structured")
        return False
    
    print("   ✓ Fisher analysis properly included in JSON")
    
    # Check structure
    fisher_data = data['fisher_analysis']
    if 'test_model' in fisher_data:
        model_fisher = fisher_data['test_model']
    else:
        model_fisher = fisher_data.get('global_summary', {})
    
    expected_sections = ['summary', 'importance', 'comparison', 'overlap_analysis', 'recommendations']
    for section in expected_sections:
        if section not in model_fisher:
            print(f"   ✗ Missing section: {section}")
            return False
    
    print(f"   ✓ All Fisher sections present: {', '.join(expected_sections)}")
    
    # Verify task names
    tasks = model_fisher['summary'].get('tasks_analyzed', [])
    if tasks != ['task1', 'task2']:
        print(f"   ✗ Wrong task names: {tasks}")
        return False
    
    print(f"   ✓ Correct task names: {tasks}")
    
    print("\n✅ Fisher metrics are properly integrated into report generation!")
    print(f"\nOutput saved in: {output_dir}")
    return True

if __name__ == "__main__":
    success = test_report_integration()
    exit(0 if success else 1)
