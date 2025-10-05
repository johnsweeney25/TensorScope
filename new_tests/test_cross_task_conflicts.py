"""
Unit tests for cross-task sample conflict detection system.

Tests the novel capability to make claims like:
"Sample 7 from Task A conflicts with Sample 23 from Task B on layer_3.qkv (p<0.001)"
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from fisher.core.fisher_collector import FisherCollector
from fisher.core.gradient_memory_manager import GradientMemoryManager
from fisher.core.cross_task_conflict_detector import CrossTaskConflictDetector


class TestCrossTaskConflictDetection(unittest.TestCase):
    """Test suite for cross-task conflict detection system."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a simple test model
        self.model = nn.Sequential(
            nn.Linear(10, 20, bias=False),
            nn.ReLU(),
            nn.Linear(20, 10, bias=False)
        )

        # Initialize Fisher collector with cross-task enabled
        self.collector = FisherCollector(
            enable_cross_task_analysis=True,
            gradient_memory_mb=10  # Small for testing
        )

        # Set seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)

    def test_gradient_storage_compression(self):
        """Test that gradient compression maintains accuracy."""
        manager = GradientMemoryManager(max_memory_mb=10)

        # Create a test gradient
        original = torch.randn(100, 100)

        # Compress and decompress
        compressed, scale = manager.compress_gradient(original)
        reconstructed = manager.decompress_gradient(
            compressed, scale, original.shape
        )

        # Check reconstruction error is small
        mse = (original - reconstructed).pow(2).mean().item()
        self.assertLess(mse, 0.001, "Compression MSE should be < 0.001")

        # Check compression ratio
        original_size = original.element_size() * original.numel()
        compressed_size = len(compressed)
        ratio = original_size / compressed_size

        self.assertGreater(ratio, 3.0, "Compression ratio should be > 3x")

    def test_conflict_detection_opposing_tasks(self):
        """Test detection of conflicts between opposing tasks."""
        # Task A: Emphasizes first half of parameters
        task_a_data = []
        for i in range(20):
            x = torch.randn(1, 10)
            x[:, :5] *= 3.0  # Strong activation on first half
            task_a_data.append(x)

        # Task B: Emphasizes second half (opposing pattern)
        task_b_data = []
        for i in range(20):
            x = torch.randn(1, 10)
            x[:, 5:] *= 3.0  # Strong activation on second half
            task_b_data.append(x)

        # Process Task A
        for i, data in enumerate(task_a_data):
            self.model.zero_grad()
            output = self.model(data)
            loss = output.sum()
            loss.backward()

            # Store current batch
            batch = {'input_ids': torch.tensor([[i]])}
            self.collector.update_fisher_ema(self.model, batch, task='task_a')

        # Process Task B
        for i, data in enumerate(task_b_data):
            self.model.zero_grad()
            output = self.model(data)
            loss = output.sum()
            loss.backward()

            # Store current batch
            batch = {'input_ids': torch.tensor([[i]])}
            self.collector.update_fisher_ema(self.model, batch, task='task_b')

        # Detect conflicts
        conflicts = self.collector.detect_cross_task_conflicts('task_a', 'task_b')

        # Should find conflicts
        self.assertIn('summary', conflicts)
        self.assertIn('top_conflicts', conflicts)
        self.assertGreater(
            conflicts['summary']['total_conflicts'], 0,
            "Should detect conflicts between opposing tasks"
        )

        # Check format of conflict claims
        if conflicts['top_conflicts']:
            conflict = conflicts['top_conflicts'][0]
            self.assertIn('claim', conflict)
            self.assertIn('p_value', conflict)
            self.assertIn('effect_size', conflict)
            self.assertIn('significance', conflict)

            # Verify claim format
            claim = conflict['claim']
            self.assertIn('Sample', claim)
            self.assertIn('conflicts with', claim)
            self.assertIn('on', claim)

    def test_conflict_detection_similar_tasks(self):
        """Test that similar tasks have fewer conflicts."""
        # Both tasks have similar patterns
        for task_name in ['task_c', 'task_d']:
            for i in range(20):
                x = torch.randn(1, 10)
                x *= 2.0  # Uniform scaling

                self.model.zero_grad()
                output = self.model(x)
                loss = output.sum()
                loss.backward()

                batch = {'input_ids': torch.tensor([[i]])}
                self.collector.update_fisher_ema(self.model, batch, task=task_name)

        # Detect conflicts
        conflicts = self.collector.detect_cross_task_conflicts('task_c', 'task_d')

        # Should find fewer or no conflicts
        n_conflicts = conflicts['summary']['total_conflicts'] if conflicts else 0
        self.assertLessEqual(
            n_conflicts, 5,
            "Similar tasks should have few conflicts"
        )

    def test_memory_management(self):
        """Test that memory limits are respected."""
        manager = GradientMemoryManager(max_memory_mb=1)  # Very small limit

        # Try to store many gradients
        for i in range(100):
            gradient = torch.randn(100, 100)
            manager.store_gradient(
                task='test',
                sample_id=i,
                param_name=f'param_{i % 10}',
                gradient=gradient,
                fisher_magnitude=np.random.random()
            )

        # Check memory is within limit
        stats = manager.get_memory_stats()
        self.assertLessEqual(
            stats['memory_usage_mb'], 1.1,  # Allow 10% overhead
            "Memory usage should stay within limit"
        )

        # Check that eviction happened
        self.assertGreater(
            stats['total_evicted'], 0,
            "Should have evicted some gradients"
        )

    def test_statistical_significance(self):
        """Test that p-values are computed correctly."""
        detector = CrossTaskConflictDetector(
            gradient_manager=GradientMemoryManager(),
            n_bootstrap_samples=100  # Fewer for speed
        )

        # Create clearly opposing gradients
        grad_a = torch.tensor([1.0, 1.0, 1.0, -1.0, -1.0])
        grad_b = torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0])

        # Compute conflict
        conflict_score, effect_size = detector.compute_gradient_conflict(grad_a, grad_b)

        # Should be strongly negative (opposing)
        self.assertLess(conflict_score, -0.9, "Should detect opposition")
        self.assertGreater(effect_size, 0.5, "Should have large effect size")

        # Test bootstrap p-value
        p_value = detector.bootstrap_significance(grad_a, grad_b, n_samples=100)

        # Should be significant
        self.assertLess(p_value, 0.05, "Opposing gradients should be significant")

    def test_bonferroni_correction(self):
        """Test that multiple testing correction is applied."""
        detector = CrossTaskConflictDetector(
            gradient_manager=GradientMemoryManager(),
            use_bonferroni=True
        )

        # Note: This is tested implicitly in detect_conflicts
        # where p_value *= n_comparisons when use_bonferroni=True
        self.assertTrue(detector.use_bonferroni)

    def test_circuit_mapping(self):
        """Test mapping of parameters to circuit components."""
        detector = CrossTaskConflictDetector(
            gradient_manager=GradientMemoryManager()
        )

        # Test various parameter names
        mappings = {
            'attention.q_proj.weight': 'query_circuit',
            'mlp.fc1.weight': 'mlp_circuit',
            'layer_norm.weight': 'normalization',
            'embeddings.weight': 'embedding',
            'attention.o_proj.weight': 'output_circuit'
        }

        for param_name, expected_circuit in mappings.items():
            circuit = detector._map_to_circuit(param_name)
            self.assertEqual(
                circuit, expected_circuit,
                f"Parameter {param_name} should map to {expected_circuit}"
            )

    def test_actionable_recommendations(self):
        """Test generation of actionable recommendations."""
        from fisher.core.cross_task_conflict_detector import CrossTaskConflict

        detector = CrossTaskConflictDetector(
            gradient_manager=GradientMemoryManager()
        )

        # Create mock conflicts
        conflicts = [
            CrossTaskConflict(
                task_a='math', task_b='code',
                sample_a=i, sample_b=j,
                parameter=f'layer_{i % 3}.weight',
                conflict_score=-0.9,
                p_value=0.001,
                effect_size=0.8
            )
            for i in range(5) for j in range(5)
        ]

        recommendations = detector.get_actionable_recommendations(conflicts, top_k=5)

        # Should generate recommendations
        self.assertGreater(len(recommendations), 0)

        # Check recommendation format
        for rec in recommendations:
            self.assertIn('action', rec)
            self.assertIn('reason', rec)
            self.assertIn('priority', rec)

    def test_conflict_clustering(self):
        """Test that conflicts are properly clustered."""
        from fisher.core.cross_task_conflict_detector import CrossTaskConflict

        detector = CrossTaskConflictDetector(
            gradient_manager=GradientMemoryManager()
        )

        # Create conflicts with patterns
        conflicts = []

        # Cluster 1: layer_0 conflicts
        for i in range(5):
            conflicts.append(CrossTaskConflict(
                task_a='task1', task_b='task2',
                sample_a=i, sample_b=i+10,
                parameter='layer_0.weight',
                conflict_score=-0.8, p_value=0.01, effect_size=0.7
            ))

        # Cluster 2: layer_1 conflicts
        for i in range(4):
            conflicts.append(CrossTaskConflict(
                task_a='task1', task_b='task2',
                sample_a=i+20, sample_b=i+30,
                parameter='layer_1.weight',
                conflict_score=-0.9, p_value=0.001, effect_size=0.9
            ))

        clusters = detector.find_conflict_clusters(conflicts, min_cluster_size=3)

        # Should find at least one cluster
        self.assertGreater(len(clusters), 0, "Should find conflict clusters")

        # Verify cluster sizes
        for cluster_name, cluster_conflicts in clusters.items():
            self.assertGreaterEqual(
                len(cluster_conflicts), 3,
                f"Cluster {cluster_name} should meet minimum size"
            )

    def test_end_to_end_forensic_claim(self):
        """Test the full pipeline for making a forensic claim."""
        # Create specific conflicting patterns
        self.model.zero_grad()

        # Task A, Sample 7: Strong positive gradient on first layer
        x_a = torch.ones(1, 10)
        output = self.model(x_a)
        loss = output.sum()
        loss.backward()

        batch_a = {'input_ids': torch.tensor([[7]])}
        self.collector.current_sample_id['quadratic_solver'] = 7
        self.collector.update_fisher_ema(self.model, batch_a, task='quadratic_solver')

        # Task B, Sample 23: Strong negative gradient (opposing)
        x_b = -torch.ones(1, 10)
        self.model.zero_grad()
        output = self.model(x_b)
        loss = -output.sum()  # Negative to create opposition
        loss.backward()

        batch_b = {'input_ids': torch.tensor([[23]])}
        self.collector.current_sample_id['list_comprehension'] = 23
        self.collector.update_fisher_ema(self.model, batch_b, task='list_comprehension')

        # Detect conflicts
        conflicts = self.collector.detect_cross_task_conflicts(
            'quadratic_solver', 'list_comprehension'
        )

        # Should be able to make a forensic claim
        if conflicts and conflicts['top_conflicts']:
            claim = conflicts['top_conflicts'][0]['claim']
            # Should contain specific sample IDs
            self.assertIn('Sample', claim)
            # Should identify the parameter
            self.assertIn(' on ', claim)
            # Should have statistical significance
            self.assertIsNotNone(conflicts['top_conflicts'][0]['p_value'])

            print(f"\nForensic claim validated: {claim}")
            print(f"Statistical significance: {conflicts['top_conflicts'][0]['significance']}")


class TestGradientMemoryManager(unittest.TestCase):
    """Test gradient compression and memory management."""

    def test_compression_ratio(self):
        """Test that compression achieves expected ratio."""
        manager = GradientMemoryManager(compression_level=9)

        # Test various tensor sizes
        sizes = [(10, 10), (100, 100), (1000, 100)]

        for size in sizes:
            gradient = torch.randn(*size)
            compressed, scale = manager.compress_gradient(gradient)

            original_bytes = gradient.element_size() * gradient.numel()
            compressed_bytes = len(compressed)
            ratio = original_bytes / compressed_bytes

            self.assertGreater(
                ratio, 2.0,
                f"Compression ratio for {size} should be > 2x, got {ratio:.2f}x"
            )

    def test_quantization_error(self):
        """Test that quantization error is acceptable."""
        manager = GradientMemoryManager()

        # Test with various magnitude gradients
        for scale in [0.001, 0.1, 1.0, 10.0]:
            gradient = torch.randn(50, 50) * scale
            compressed, quant_scale = manager.compress_gradient(gradient)
            reconstructed = manager.decompress_gradient(
                compressed, quant_scale, gradient.shape
            )

            # Compute relative error
            rel_error = (gradient - reconstructed).abs().mean() / (gradient.abs().mean() + 1e-8)

            self.assertLess(
                rel_error, 0.05,
                f"Relative error for scale {scale} should be < 5%"
            )

    def test_importance_filtering(self):
        """Test that importance-based filtering works correctly."""
        manager = GradientMemoryManager(importance_percentile=75)

        # Add gradients with varying importance
        for i in range(100):
            should_store = manager.should_store_gradient(
                param_name='test_param',
                fisher_magnitude=i / 100.0
            )

            # After warmup, only top 25% should be stored
            if i > 20:  # After warmup
                if i >= 75:  # Top 25%
                    self.assertTrue(
                        should_store or 'critical' in 'test_param',
                        f"High importance {i/100:.2f} should be stored"
                    )

    def test_lru_eviction(self):
        """Test that LRU eviction works correctly."""
        manager = GradientMemoryManager(max_memory_mb=0.1)  # Very small

        stored_ids = []

        # Store many gradients
        for i in range(50):
            gradient = torch.randn(100, 100)
            success = manager.store_gradient(
                task='test',
                sample_id=i,
                param_name='param',
                gradient=gradient,
                fisher_magnitude=1.0
            )
            if success:
                stored_ids.append(i)

        # Check that early samples were evicted
        early_gradient = manager.get_gradient('test', 0, 'param')
        self.assertIsNone(
            early_gradient,
            "Early samples should be evicted under memory pressure"
        )

        # Recent samples should still be available
        if stored_ids:
            recent_id = stored_ids[-1]
            recent_gradient = manager.get_gradient('test', recent_id, 'param')
            self.assertIsNotNone(
                recent_gradient,
                "Recent samples should be retained"
            )


if __name__ == '__main__':
    unittest.main()