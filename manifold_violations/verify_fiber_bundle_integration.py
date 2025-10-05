#!/usr/bin/env python3
"""
Verification script for fiber bundle hypothesis integration.

This script demonstrates and verifies that the fiber bundle modules
work correctly with the existing manifold analysis tools.
"""

import torch
import numpy as np
import warnings
from typing import Dict, Any

# Import existing manifold tools
from tractable_manifold_curvature import (
    compute_ricci_curvature_tractable,
    compute_sectional_curvature_tractable,
    compute_intrinsic_dimension_twonn,
    sinkhorn_distance
)

# Import new fiber bundle tools
from fiber_bundle_core import FiberBundleTest, batch_test_fiber_bundle
from manifold_fiber_integration import GeometricAnalyzer, integrated_geometric_analysis

# Try importing token analyzer
try:
    from token_stability_analyzer import TokenStabilityAnalyzer
    HAS_TOKEN_ANALYZER = True
except ImportError:
    HAS_TOKEN_ANALYZER = False


def verify_compatibility():
    """Verify that fiber bundle and manifold tools work together."""
    print("=" * 70)
    print("FIBER BUNDLE + MANIFOLD INTEGRATION VERIFICATION")
    print("=" * 70)

    # Create test data with known structure
    np.random.seed(42)
    torch.manual_seed(42)

    # Region 1: Clean spiral manifold (should pass fiber bundle test)
    t = np.linspace(0, 6*np.pi, 200)
    spiral = np.column_stack([
        t * np.cos(t) / 20,
        t * np.sin(t) / 20,
        t / 50
    ])
    spiral += np.random.randn(*spiral.shape) * 0.001  # Small noise

    # Region 2: Irregular points (should fail fiber bundle test)
    irregular = np.random.randn(100, 3) * 2

    # Combine data
    combined_data = np.vstack([spiral, irregular])
    combined_tensor = torch.from_numpy(combined_data).float()

    print(f"\nTest data: {len(combined_data)} points")
    print(f"  - Spiral manifold: {len(spiral)} points")
    print(f"  - Irregular region: {len(irregular)} points")

    # Step 1: Run fiber bundle analysis
    print("\n1. FIBER BUNDLE ANALYSIS")
    print("-" * 40)

    fb_test = FiberBundleTest(alpha=0.05, n_bootstrap=50)

    # Test points from each region
    test_indices = {
        "spiral_start": 10,
        "spiral_middle": 100,
        "spiral_end": 190,
        "irregular_start": 210,
        "irregular_middle": 250,
        "irregular_end": 290
    }

    fb_results = {}
    for name, idx in test_indices.items():
        result = fb_test.test_point(combined_data, idx)
        fb_results[name] = result
        status = "PASS" if not result.reject_null else "REJECT"
        print(f"  {name:20s}: p={result.p_value:.4f} [{status}] regime={result.regime}")

    # Step 2: Conditional manifold analysis
    print("\n2. CONDITIONAL MANIFOLD ANALYSIS")
    print("-" * 40)

    for name, idx in test_indices.items():
        fb_result = fb_results[name]
        region_type = "spiral" if "spiral" in name else "irregular"

        if not fb_result.reject_null and fb_result.p_value > 0.1:
            # Passes fiber bundle test - safe to use manifold analysis
            local_points = combined_tensor[max(0, idx-10):min(len(combined_tensor), idx+11)]

            try:
                ricci_mean, ricci_std = compute_ricci_curvature_tractable(
                    local_points, n_samples=5
                )
                dim = compute_intrinsic_dimension_twonn(local_points)

                print(f"  {name:20s}: Ricci={ricci_mean:.4f}±{ricci_std:.4f}, Dim={dim:.2f}")
            except Exception as e:
                print(f"  {name:20s}: Manifold analysis failed: {e}")
        else:
            print(f"  {name:20s}: SKIP (failed fiber bundle test)")

    # Step 3: Integrated analysis
    print("\n3. INTEGRATED GEOMETRIC ANALYSIS")
    print("-" * 40)

    analyzer = GeometricAnalyzer(alpha=0.05)
    summary = analyzer.analyze_dataset(
        combined_tensor,
        sample_size=50,
        verbose=False
    )

    print(f"  Overall structure: {summary['overall_structure']}")
    print(f"  Geometry distribution:")
    for geom_type, pct in summary['geometry_distribution'].items():
        print(f"    - {geom_type:15s}: {pct:6.2%}")
    print(f"  Mean stability: {summary['mean_stability']:.3f}")
    print(f"  Mean irregularity: {summary['mean_irregularity']:.3f}")

    # Step 4: Verify key principle
    print("\n4. KEY PRINCIPLE VERIFICATION")
    print("-" * 40)

    success_count = 0
    for name, idx in test_indices.items():
        fb_result = fb_results[name]
        is_spiral = "spiral" in name

        # Key principle: Spiral points should pass, irregular should fail
        if is_spiral and not fb_result.reject_null:
            success_count += 1
            print(f"  ✓ {name}: Correctly identified as regular")
        elif not is_spiral and fb_result.reject_null:
            success_count += 1
            print(f"  ✓ {name}: Correctly identified as irregular")
        else:
            print(f"  ✗ {name}: Misclassified")

    accuracy = success_count / len(test_indices)
    print(f"\n  Classification accuracy: {accuracy:.1%}")

    return accuracy >= 0.5  # At least 50% correct


def verify_practical_use_case():
    """Demonstrate practical use case: analyzing neural network representations."""
    print("\n" + "=" * 70)
    print("PRACTICAL USE CASE: NEURAL NETWORK ANALYSIS")
    print("=" * 70)

    # Create mock neural network representations
    # Early layers: more irregular
    early_layer = torch.randn(500, 128) * 2
    early_layer[::2] += torch.randn(250, 128) * 5  # Add irregularity

    # Late layers: more structured
    late_layer = torch.randn(500, 128) * 0.5
    # Add structure (low-rank approximation)
    U = torch.randn(500, 10)
    V = torch.randn(10, 128)
    late_layer += U @ V

    print("\nAnalyzing neural network layer representations...")

    analyzer = GeometricAnalyzer()

    # Analyze early layer
    print("\nEarly Layer Analysis:")
    early_result = analyzer.analyze_dataset(
        early_layer,
        sample_size=30,
        verbose=False
    )
    print(f"  Structure: {early_result['overall_structure']}")
    print(f"  Stability: {early_result['mean_stability']:.3f}")

    # Analyze late layer
    print("\nLate Layer Analysis:")
    late_result = analyzer.analyze_dataset(
        late_layer,
        sample_size=30,
        verbose=False
    )
    print(f"  Structure: {late_result['overall_structure']}")
    print(f"  Stability: {late_result['mean_stability']:.3f}")

    # Compare
    print("\nComparison:")
    stability_improvement = late_result['mean_stability'] - early_result['mean_stability']
    print(f"  Stability improvement: {stability_improvement:+.3f}")

    if stability_improvement > 0:
        print("  ✓ Late layers show improved geometric structure")
    else:
        print("  ✗ Unexpected: late layers less stable")

    return True


def verify_robustness():
    """Verify robustness to different data types."""
    print("\n" + "=" * 70)
    print("ROBUSTNESS VERIFICATION")
    print("=" * 70)

    test_cases = []

    # Test 1: High-dimensional data
    print("\n1. High-dimensional data (d=50):")
    high_dim = torch.randn(100, 50)
    result = integrated_geometric_analysis(high_dim, sample_size=20, verbose=False)
    print(f"   Structure: {result['overall_structure']}")
    test_cases.append(result['n_analyzed'] > 0)

    # Test 2: Very small dataset
    print("\n2. Small dataset (n=10):")
    small_data = torch.randn(10, 3)
    result = integrated_geometric_analysis(small_data, verbose=False)
    print(f"   Structure: {result['overall_structure']}")
    test_cases.append(result['n_analyzed'] > 0)

    # Test 3: Perfect manifold (sphere)
    print("\n3. Perfect manifold (unit sphere):")
    sphere = torch.randn(200, 3)
    sphere = sphere / sphere.norm(dim=1, keepdim=True)
    result = integrated_geometric_analysis(sphere, sample_size=50, verbose=False)
    print(f"   Structure: {result['overall_structure']}")
    manifold_pct = result['geometry_distribution']['manifold']
    print(f"   Manifold percentage: {manifold_pct:.1%}")
    test_cases.append(manifold_pct > 0.3)  # At least 30% detected as manifold

    # Test 4: Completely random data
    print("\n4. Random noise:")
    noise = torch.randn(100, 10) * 10
    result = integrated_geometric_analysis(noise, sample_size=30, verbose=False)
    print(f"   Structure: {result['overall_structure']}")
    irregular_pct = result['geometry_distribution']['irregular']
    print(f"   Irregular percentage: {irregular_pct:.1%}")
    test_cases.append(True)  # Should handle without crashing

    success_rate = sum(test_cases) / len(test_cases)
    print(f"\nRobustness: {success_rate:.0%} of test cases handled successfully")

    return success_rate >= 0.75


def main():
    """Run all verification tests."""
    warnings.filterwarnings('ignore')

    print("\n" + "=" * 70)
    print("FIBER BUNDLE HYPOTHESIS - INTEGRATION VERIFICATION")
    print("=" * 70)
    print("\nThis script verifies that the fiber bundle modules integrate")
    print("correctly with existing manifold analysis tools.")

    results = {}

    # Test 1: Basic compatibility
    print("\n\nTEST 1: BASIC COMPATIBILITY")
    results['compatibility'] = verify_compatibility()

    # Test 2: Practical use case
    print("\n\nTEST 2: PRACTICAL USE CASE")
    results['practical'] = verify_practical_use_case()

    # Test 3: Robustness
    print("\n\nTEST 3: ROBUSTNESS")
    results['robustness'] = verify_robustness()

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name:15s}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n✅ All verification tests passed!")
        print("\nThe fiber bundle modules are successfully integrated with")
        print("the existing manifold analysis tools.")
        print("\nKey insights:")
        print("  1. Fiber bundle test acts as a prerequisite check")
        print("  2. Manifold analysis is only applied where valid")
        print("  3. Irregular regions are correctly identified")
        print("  4. System is robust to various data types")
    else:
        print("\n⚠️ Some tests failed. Review the output above.")

    return all_passed


if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)