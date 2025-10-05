"""
Test the integration of SuperpositionMetrics with unified_model_analysis.
"""

import torch
import torch.nn as nn
from unified_model_analysis import UnifiedModelAnalyzer, UnifiedConfig


def test_superposition_integration():
    """Test that SuperpositionMetrics is properly integrated."""

    # Create analysis instance
    config = UnifiedConfig()
    analyzer = UnifiedModelAnalyzer(config)

    # Check that superposition module is loaded through registry
    assert 'superposition' in analyzer.registry.modules, "SuperpositionMetrics not loaded"

    # Check that methods are registered
    superposition_methods = [name for name in analyzer.registry.metrics.keys() if 'superposition' in name]
    print(f"Found {len(superposition_methods)} superposition methods registered")

    expected_methods = [
        'compute_vector_interference',
        'compute_feature_frequency_distribution',
        'compute_superposition_strength',
        'analyze_dimensional_scaling',
        'compute_feature_sparsity',
        'fit_scaling_law',
        'compute_representation_capacity',
        'analyze_feature_emergence'
    ]

    for method in expected_methods:
        if method in analyzer.registry.metrics:
            print(f"✓ {method} registered")
        else:
            print(f"✗ {method} NOT registered")

    # Test that we can access the module directly
    sup_module = analyzer.registry.modules['superposition']
    print(f"\nSuperposition module type: {type(sup_module)}")
    print(f"Device: {sup_module.device}")

    # Create a simple test
    weight_matrix = torch.randn(50, 32)
    result = sup_module.compute_vector_interference(weight_matrix)

    print(f"\nVector interference test:")
    print(f"  Mean overlap: {result['mean_overlap']:.4f}")
    print(f"  Effective orthogonality: {result['effective_orthogonality']:.4f}")
    print(f"  Features: {result['n_features']}, Dimensions: {result['n_dimensions']}")

    print("\n✅ Integration test successful!")


if __name__ == "__main__":
    test_superposition_integration()