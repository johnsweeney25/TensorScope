"""
Test trajectory integration for superposition metrics.
"""

import unittest
import torch
import torch.nn as nn
import tempfile
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from superposition.core.analyzer import SuperpositionAnalyzer
from unified_model_analysis import UnifiedModelAnalyzer, UnifiedConfig, CheckpointSpec, SignatureType


class TestTrajectoryIntegration(unittest.TestCase):
    """Test superposition metrics in trajectory analysis."""

    def setUp(self):
        """Set up test environment."""
        self.test_dir = Path(tempfile.mkdtemp())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tearDown(self):
        """Clean up test environment."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_superposition_trajectory_method(self):
        """Test the compute_superposition_trajectory method."""
        analyzer = SuperpositionAnalyzer(device=self.device)

        # Create test model
        model = nn.Sequential(
            nn.Embedding(100, 64),
            nn.Linear(64, 128)
        ).to(self.device)

        # Test trajectory method
        result = analyzer.compute_superposition_trajectory(model)

        # Check returned metrics
        self.assertIn('phi_half', result)
        self.assertIn('phi_one', result)
        self.assertIn('mean_overlap', result)
        self.assertIn('regime_numeric', result)
        self.assertIn('welch_bound_ratio', result)

        # Check types
        self.assertIsInstance(result['phi_half'], float)
        self.assertIsInstance(result['phi_one'], float)
        self.assertIsInstance(result['regime_numeric'], int)

        # Check ranges
        self.assertGreaterEqual(result['phi_half'], 0.0)
        self.assertLessEqual(result['phi_half'], 1.0)
        self.assertGreaterEqual(result['phi_one'], 0.0)
        self.assertLessEqual(result['phi_one'], 1.0)
        self.assertIn(result['regime_numeric'], [0, 1, 2])

    def test_registration_in_unified_analyzer(self):
        """Test that superposition trajectory methods are registered."""
        config = UnifiedConfig(
            output_dir=self.test_dir,
            device=str(self.device),
            generate_report=False
        )

        uma = UnifiedModelAnalyzer(config)

        # Check registrations
        self.assertIn('compute_superposition_trajectory', uma.registry.metrics)
        self.assertIn('analyze_model_superposition', uma.registry.metrics)

        # Check signature types
        traj_info = uma.registry.metrics['compute_superposition_trajectory']
        self.assertEqual(traj_info['signature_type'], SignatureType.STANDARD)

    def test_multi_checkpoint_analysis(self):
        """Test superposition analysis across multiple checkpoints."""
        analyzer = SuperpositionAnalyzer(device=self.device)

        # Create multiple checkpoints with increasing capacity
        checkpoints = []
        for i in range(3):
            # Increase embedding dimension to simulate training progression
            dim = 32 + i * 16
            model = nn.Sequential(
                nn.Embedding(100, dim),
                nn.Linear(dim, 100)
            ).to(self.device)
            checkpoints.append(model)

        # Track metrics across checkpoints
        phi_half_evolution = []
        phi_one_evolution = []
        regime_evolution = []

        for model in checkpoints:
            result = analyzer.compute_superposition_trajectory(model)
            phi_half_evolution.append(result['phi_half'])
            phi_one_evolution.append(result['phi_one'])
            regime_evolution.append(result['regime_numeric'])

        # Verify we got metrics for all checkpoints
        self.assertEqual(len(phi_half_evolution), 3)
        self.assertEqual(len(phi_one_evolution), 3)
        self.assertEqual(len(regime_evolution), 3)

        print(f"\nTrajectory Analysis Results:")
        print(f"φ₁/₂ evolution: {phi_half_evolution}")
        print(f"φ₁ evolution: {phi_one_evolution}")
        print(f"Regime evolution: {regime_evolution}")

    def test_trajectory_json_serialization(self):
        """Test that trajectory results are JSON serializable."""
        analyzer = SuperpositionAnalyzer(device=self.device)

        model = nn.Sequential(
            nn.Embedding(50, 32),
            nn.Linear(32, 50)
        ).to(self.device)

        # Get trajectory metrics
        result = analyzer.compute_superposition_trajectory(model)

        # Should be JSON serializable
        try:
            json_str = json.dumps(result)
            self.assertIsNotNone(json_str)

            # Can round-trip
            loaded = json.loads(json_str)
            self.assertEqual(loaded['phi_half'], result['phi_half'])
        except Exception as e:
            self.fail(f"Result not JSON serializable: {e}")

    def test_default_trajectory_metrics_include_superposition(self):
        """Test that default trajectory metrics include superposition."""
        from unified_model_analysis import create_trajectory_config
        import argparse

        # Create mock args
        args = argparse.Namespace()
        args.models = []
        args.checkpoint_dir = str(self.test_dir)

        config = create_trajectory_config(args)

        # Check that superposition metrics are in trajectory_metrics
        self.assertIn('compute_superposition_trajectory', config.trajectory_metrics)

        print("\nDefault trajectory metrics include:")
        for metric in config.trajectory_metrics:
            if 'superposition' in metric.lower():
                print(f"  ✓ {metric}")

    def test_analyze_feature_emergence_integration(self):
        """Test analyze_feature_emergence for checkpoint analysis."""
        analyzer = SuperpositionAnalyzer(device=self.device)

        # Check if method exists
        self.assertTrue(hasattr(analyzer, 'analyze_feature_emergence'))

        # Note: Full test would require enhanced.py implementation
        print("\n✓ analyze_feature_emergence method available for checkpoint analysis")


def suite():
    """Create test suite."""
    return unittest.TestLoader().loadTestsFromTestCase(TestTrajectoryIntegration)


if __name__ == '__main__':
    unittest.main(verbosity=2)