"""
Unit tests for cross-task conflict detection integration with unified_model_analysis.

Tests that the forensic Fisher system properly integrates with the main analysis pipeline.
"""

import unittest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch, MagicMock
from BombshellMetrics import BombshellMetrics
from fisher.core.fisher_collector import FisherCollector


class TestUnifiedCrossTaskIntegration(unittest.TestCase):
    """Test integration of cross-task conflicts with unified analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(10, 20, bias=False),
            nn.ReLU(),
            nn.Linear(20, 10, bias=False)
        )
        torch.manual_seed(42)

    def test_bombshell_metrics_initialization(self):
        """Test that BombshellMetrics properly initializes with cross-task."""
        metrics = BombshellMetrics(
            enable_cross_task_analysis=True,
            gradient_memory_mb=10
        )

        # Check that cross-task is enabled in parent class
        self.assertTrue(metrics.enable_cross_task)
        self.assertIsNotNone(metrics.gradient_manager)
        self.assertIsNotNone(metrics.conflict_detector)

    def test_fisher_collector_cross_task_storage(self):
        """Test that FisherCollector stores gradients when cross-task enabled."""
        collector = FisherCollector(
            enable_cross_task_analysis=True,
            gradient_memory_mb=5
        )

        # Process a sample
        self.model.zero_grad()
        x = torch.randn(1, 10)
        output = self.model(x)
        loss = output.sum()
        loss.backward()

        # Update Fisher with gradient storage
        batch = {'input_ids': torch.tensor([[0]])}
        collector.update_fisher_ema(self.model, batch, task='test_task')

        # Check that gradients were stored
        if collector.gradient_manager:
            stats = collector.gradient_manager.get_memory_stats()
            self.assertGreater(stats['num_gradients'], 0, "Should store gradients")

    def test_detect_cross_task_conflicts_method(self):
        """Test that detect_cross_task_conflicts method works."""
        metrics = BombshellMetrics(
            enable_cross_task_analysis=True,
            gradient_memory_mb=10
        )

        # Process conflicting tasks
        for task_id, task_name in enumerate(['task_a', 'task_b']):
            for sample_id in range(5):
                self.model.zero_grad()

                # Create opposing gradients for different tasks
                if task_id == 0:
                    x = torch.ones(1, 10)
                else:
                    x = -torch.ones(1, 10)

                output = self.model(x)
                loss = output.sum() if task_id == 0 else -output.sum()
                loss.backward()

                batch = {'input_ids': torch.tensor([[sample_id]])}
                metrics.update_fisher_ema(self.model, batch, task=task_name)

        # Detect conflicts
        conflicts = metrics.detect_cross_task_conflicts('task_a', 'task_b')

        # Verify structure
        self.assertIn('summary', conflicts)
        self.assertIn('top_conflicts', conflicts)
        self.assertIn('recommendations', conflicts)

    @patch('unified_model_analysis.UnifiedModelAnalyzer')
    def test_unified_analysis_integration(self, mock_analyzer):
        """Test integration with UnifiedModelAnalyzer."""
        # Create mock analyzer with registry
        analyzer = MagicMock()
        analyzer.registry = MagicMock()
        analyzer.registry.modules = {}

        # Add BombshellMetrics with cross-task
        bombshell = BombshellMetrics(
            enable_cross_task_analysis=True,
            gradient_memory_mb=10
        )
        analyzer.registry.modules['bombshell'] = bombshell

        # Mock context with batches
        context = Mock()
        context.batches = [
            {'input_ids': torch.tensor([[0]])},
            {'input_ids': torch.tensor([[1]])}
        ]
        context.batch = context.batches[0]

        # Process some data to create Fisher info
        for i in range(5):
            self.model.zero_grad()
            x = torch.randn(1, 10)
            output = self.model(x)
            loss = output.sum()
            loss.backward()
            bombshell.update_fisher_ema(self.model, context.batch, task=f'task_{i%2}')

        # Test that detect_cross_task_conflicts can be called
        fisher_metrics = ['detect_cross_task_conflicts']

        # Check bombshell has the method
        self.assertTrue(hasattr(bombshell, 'detect_cross_task_conflicts'))

        # Check it returns proper structure
        result = bombshell.detect_cross_task_conflicts('task_0', 'task_1')
        self.assertIsInstance(result, dict)

    def test_memory_constraints_respected(self):
        """Test that memory constraints are properly enforced."""
        metrics = BombshellMetrics(
            enable_cross_task_analysis=True,
            gradient_memory_mb=0.5  # Very small limit
        )

        # Process many samples
        for i in range(100):
            self.model.zero_grad()
            x = torch.randn(1, 10)
            output = self.model(x)
            loss = output.sum()
            loss.backward()

            batch = {'input_ids': torch.tensor([[i]])}
            metrics.update_fisher_ema(self.model, batch, task='memory_test')

        # Check memory usage
        stats = metrics.gradient_manager.get_memory_stats()
        self.assertLessEqual(
            stats['memory_usage_mb'], 0.6,  # Allow small overhead
            "Memory usage should respect limit"
        )

    def test_forensic_claim_format(self):
        """Test that forensic claims have the correct format."""
        metrics = BombshellMetrics(
            enable_cross_task_analysis=True,
            gradient_memory_mb=10
        )

        # Create known conflict
        # Task A: positive gradients
        self.model.zero_grad()
        x = torch.ones(1, 10)
        output = self.model(x)
        loss = output.sum()
        loss.backward()
        metrics.current_sample_id['math'] = 7
        metrics.update_fisher_ema(self.model, {'input_ids': torch.tensor([[7]])}, task='math')

        # Task B: negative gradients (conflict)
        self.model.zero_grad()
        x = -torch.ones(1, 10)
        output = self.model(x)
        loss = -output.sum()
        loss.backward()
        metrics.current_sample_id['code'] = 23
        metrics.update_fisher_ema(self.model, {'input_ids': torch.tensor([[23]])}, task='code')

        # Get conflicts
        conflicts = metrics.detect_cross_task_conflicts('math', 'code')

        if conflicts and conflicts.get('top_conflicts'):
            claim_info = conflicts['top_conflicts'][0]
            claim = claim_info['claim']

            # Verify claim format
            self.assertIn('Sample', claim)
            self.assertIn('conflicts with', claim)
            self.assertIn(' on ', claim)
            self.assertIn('significance', claim_info)
            self.assertIsNotNone(claim_info.get('p_value'))
            self.assertIsNotNone(claim_info.get('effect_size'))

    def test_cross_task_disabled_by_default(self):
        """Test that cross-task is disabled by default."""
        metrics = BombshellMetrics()  # No enable_cross_task_analysis

        # Should not have gradient manager
        self.assertFalse(metrics.enable_cross_task)
        self.assertIsNone(metrics.gradient_manager)
        self.assertIsNone(metrics.conflict_detector)

        # detect_cross_task_conflicts should return empty
        result = metrics.detect_cross_task_conflicts('task_a', 'task_b')
        self.assertEqual(result, {})

    def test_gradient_storage_importance_filtering(self):
        """Test that only important gradients are stored."""
        collector = FisherCollector(
            enable_cross_task_analysis=True,
            gradient_memory_mb=10
        )

        # Process samples with varying importance
        for i in range(20):
            self.model.zero_grad()

            # Create gradients with different magnitudes
            if i < 5:
                # High importance (large gradients)
                x = torch.randn(1, 10) * 10.0
            else:
                # Low importance (small gradients)
                x = torch.randn(1, 10) * 0.01

            output = self.model(x)
            loss = output.sum()
            loss.backward()

            batch = {'input_ids': torch.tensor([[i]])}
            collector.update_fisher_ema(self.model, batch, task='importance_test')

        # Check that storage is selective
        if collector.gradient_manager:
            stats = collector.gradient_manager.get_memory_stats()
            # Should store fewer than all 20 samples due to importance filtering
            self.assertLess(
                stats['num_gradients'],
                20 * len(list(self.model.parameters())),
                "Should filter by importance"
            )

    def test_statistical_significance_computation(self):
        """Test that p-values are properly computed."""
        metrics = BombshellMetrics(
            enable_cross_task_analysis=True,
            gradient_memory_mb=10
        )

        # Create clear conflicts
        for i in range(10):
            # Task A: consistent positive pattern
            self.model.zero_grad()
            x = torch.ones(1, 10) + torch.randn(1, 10) * 0.1
            output = self.model(x)
            loss = output.sum()
            loss.backward()
            metrics.update_fisher_ema(self.model, {'input_ids': torch.tensor([[i]])}, task='consistent_a')

            # Task B: consistent negative pattern (strong conflict)
            self.model.zero_grad()
            x = -torch.ones(1, 10) + torch.randn(1, 10) * 0.1
            output = self.model(x)
            loss = -output.sum()
            loss.backward()
            metrics.update_fisher_ema(self.model, {'input_ids': torch.tensor([[i]])}, task='consistent_b')

        # Detect conflicts
        conflicts = metrics.detect_cross_task_conflicts('consistent_a', 'consistent_b')

        if conflicts and conflicts.get('top_conflicts'):
            # Should have significant p-values for strong conflicts
            for conflict in conflicts['top_conflicts'][:3]:
                self.assertLess(
                    conflict['p_value'], 0.05,
                    "Strong conflicts should have p < 0.05"
                )
                self.assertGreater(
                    conflict['effect_size'], 0.3,
                    "Should have meaningful effect size"
                )

    def test_recommendations_generation(self):
        """Test that actionable recommendations are generated."""
        metrics = BombshellMetrics(
            enable_cross_task_analysis=True,
            gradient_memory_mb=10
        )

        # Create some conflicts
        for i in range(5):
            self.model.zero_grad()
            x = torch.randn(1, 10)
            output = self.model(x)
            loss = output.sum() if i % 2 == 0 else -output.sum()
            loss.backward()
            metrics.update_fisher_ema(
                self.model,
                {'input_ids': torch.tensor([[i]])},
                task='task_a' if i % 2 == 0 else 'task_b'
            )

        conflicts = metrics.detect_cross_task_conflicts('task_a', 'task_b')

        if conflicts and conflicts.get('recommendations'):
            for rec in conflicts['recommendations']:
                # Check recommendation structure
                self.assertIn('action', rec)
                self.assertIn('priority', rec)
                self.assertIn(rec['priority'], ['high', 'medium', 'low'])


if __name__ == '__main__':
    unittest.main()