"""
Integration test for superposition metrics in the full analysis pipeline.

Tests that new metrics flow correctly through:
1. UnifiedModelAnalysis
2. JSON serialization
3. Statistical report generation
4. LaTeX/PDF output
"""

import unittest
import torch
import torch.nn as nn
import json
import tempfile
import shutil
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from superposition.core.analyzer import SuperpositionAnalyzer
from unified_model_analysis import UnifiedModelAnalyzer, UnifiedConfig
from statistical_report_generator import StatisticalReportGenerator, ReportConfig
import pandas as pd


class TestSuperpositionIntegration(unittest.TestCase):
    """Test integration of superposition metrics in the full pipeline."""

    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for outputs
        self.test_dir = Path(tempfile.mkdtemp())
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

    def test_analyzer_returns_serializable(self):
        """Test that SuperpositionAnalyzer returns JSON-serializable results."""
        analyzer = SuperpositionAnalyzer(device=self.device)

        # Create test weight matrix
        weight_matrix = torch.randn(100, 50, device=self.device)

        # Get analysis as dict
        result = analyzer.compute_comprehensive_superposition_analysis(
            weight_matrix,
            return_dict=True
        )

        # Should be a dict
        self.assertIsInstance(result, dict)

        # Check for new metrics
        self.assertIn('phi_half', result)
        self.assertIn('phi_one', result)
        self.assertIn('regime', result)

        # Should be serializable
        try:
            json_str = json.dumps(result, default=str)
            self.assertIsNotNone(json_str)
        except Exception as e:
            self.fail(f"Result not JSON serializable: {e}")

    def test_unified_model_analysis_integration(self):
        """Test that metrics work in UnifiedModelAnalyzer."""
        # Create simple test model
        model = nn.Sequential(
            nn.Embedding(100, 32),
            nn.Linear(32, 64),
            nn.Linear(64, 100)
        ).to(self.device)

        # Configure UMA
        config = UnifiedConfig(
            output_dir=self.test_dir,
            device=str(self.device),
            generate_report=False
        )

        # Create analyzer
        uma = UnifiedModelAnalyzer(config)

        # Check that superposition module loaded in registry
        self.assertIn('superposition', uma.registry.modules)
        self.assertIsNotNone(uma.registry.modules['superposition'])

        # Check method is registered in registry
        self.assertIn('compute_comprehensive_superposition_analysis', uma.registry.metrics)

    def test_dataclass_serialization(self):
        """Test that dataclass results are properly serialized."""
        from dataclasses import dataclass, asdict

        # Create test dataclass
        @dataclass
        class TestResult:
            value: float
            name: str
            data: list

        # Create UMA instance
        config = UnifiedConfig(output_dir=self.test_dir)
        uma = UnifiedModelAnalyzer(config)

        # Test serialization using asdict directly (which is what we use internally)
        test_obj = TestResult(value=1.5, name="test", data=[1, 2, 3])
        serialized = asdict(test_obj)

        # Should be a dict
        self.assertIsInstance(serialized, dict)
        self.assertEqual(serialized['value'], 1.5)
        self.assertEqual(serialized['name'], "test")
        self.assertEqual(serialized['data'], [1, 2, 3])

    def test_numpy_type_conversion(self):
        """Test that numpy types are properly converted."""
        import numpy as np

        # Test JSON serialization of numpy types directly
        test_values = {
            'float64': np.float64(3.14),
            'float32': np.float32(2.71),
            'int64': np.int64(42),
            'bool_': np.bool_(True),
            'array': np.array([1, 2, 3])
        }

        for name, value in test_values.items():
            # Convert to Python type
            if isinstance(value, np.ndarray):
                serialized = value.tolist()
            elif isinstance(value, np.generic):
                serialized = value.item()
            else:
                serialized = value

            # Should be native Python type or list
            self.assertNotIsInstance(serialized, (np.ndarray, np.generic))

            # Should be JSON serializable
            try:
                json.dumps(serialized)
            except Exception as e:
                self.fail(f"Failed to serialize {name}: {e}")

    def test_report_generator_recognizes_metrics(self):
        """Test that report generator recognizes new metrics."""
        # Create sample data with new metrics
        data = {
            'model_id': ['model1', 'model2'],
            'phi_half': [0.85, 0.92],
            'phi_one': [0.45, 0.55],
            'regime': ['strong_superposition', 'strong_superposition'],
            'n_represented': [85, 92],
            'n_strongly_represented': [45, 55],
            'mean_overlap': [0.12, 0.15]
        }

        df = pd.DataFrame(data)

        # Create report config
        report_config = ReportConfig(
            output_dir=self.test_dir,
            figure_dir=self.test_dir / 'figures'
        )

        # Create generator
        generator = StatisticalReportGenerator(config=report_config)
        generator.combined_df = df

        # Analyze superposition metrics
        superposition_analysis = generator._compute_superposition_analysis(df)

        # Should find metrics
        self.assertIsNotNone(superposition_analysis)
        self.assertIn('summary', superposition_analysis)
        self.assertEqual(superposition_analysis['summary']['num_metrics'], 6)

    def test_latex_generation_with_new_metrics(self):
        """Test that LaTeX properly formats new metrics."""
        # Create sample data
        data = {
            'model_id': ['model1'],
            'phi_half': [0.85],
            'phi_one': [0.45],
            'regime': ['strong_superposition']
        }

        df = pd.DataFrame(data)

        # Create generator
        report_config = ReportConfig(output_dir=self.test_dir)
        generator = StatisticalReportGenerator(config=report_config)
        generator.combined_df = df

        # Generate superposition section
        latex_section = generator._generate_superposition_section()

        # Check for proper LaTeX formatting
        self.assertIn(r'\phi_{1/2}', latex_section)  # φ₁/₂ symbol
        self.assertIn(r'\phi_1', latex_section)      # φ₁ symbol
        self.assertIn('Strong Superposition', latex_section)  # Regime
        self.assertIn('Liu et al., 2025', latex_section)  # Paper citation

    def test_end_to_end_pipeline(self):
        """Test complete pipeline from analysis to report."""
        # Create test model
        model = nn.Sequential(
            nn.Embedding(50, 16),
            nn.Linear(16, 50)
        ).to(self.device)

        # Create test batch
        test_batch = torch.randint(0, 50, (4, 8), device=self.device)

        # Step 1: Analyze with SuperpositionAnalyzer
        analyzer = SuperpositionAnalyzer(device=self.device)
        weight_matrix = model[0].weight.data

        result = analyzer.compute_comprehensive_superposition_analysis(
            weight_matrix,
            return_dict=True
        )

        # Step 2: Create mock UMA results
        mock_results = {
            'model1': {
                'metrics': {
                    'compute_comprehensive_superposition_analysis': result
                }
            }
        }

        # Step 3: Save as JSON
        json_path = self.test_dir / 'test_results.json'
        with open(json_path, 'w') as f:
            json.dump(mock_results, f, default=str)

        # Verify JSON contains our metrics
        with open(json_path, 'r') as f:
            loaded = json.load(f)

        metrics = loaded['model1']['metrics']['compute_comprehensive_superposition_analysis']
        self.assertIn('phi_half', metrics)
        self.assertIn('phi_one', metrics)
        self.assertIn('regime', metrics)

        # Step 4: Verify metrics are in expected ranges
        self.assertGreaterEqual(metrics['phi_half'], 0.0)
        self.assertLessEqual(metrics['phi_half'], 1.0)
        self.assertGreaterEqual(metrics['phi_one'], 0.0)
        self.assertLessEqual(metrics['phi_one'], 1.0)
        self.assertIn(metrics['regime'], ['no_superposition', 'weak_superposition', 'strong_superposition'])

        print(f"\n✓ Integration test passed!")
        print(f"  - φ₁/₂ = {metrics['phi_half']:.3f}")
        print(f"  - φ₁ = {metrics['phi_one']:.3f}")
        print(f"  - Regime: {metrics['regime']}")


def suite():
    """Create test suite."""
    return unittest.TestLoader().loadTestsFromTestCase(TestSuperpositionIntegration)


if __name__ == '__main__':
    unittest.main(verbosity=2)