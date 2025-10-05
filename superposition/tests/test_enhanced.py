"""
Comprehensive test suite for SuperpositionMetrics v2.

Tests GPU handling, numerical precision, edge cases, and stress conditions.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict
import gc

from superposition.core.enhanced import SuperpositionMetrics, SuperpositionConfig


class TestGPUMemoryHandling(unittest.TestCase):
    """Test GPU memory management."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = SuperpositionConfig(cleanup_cuda_cache=True)
        self.metrics = SuperpositionMetrics(device=self.device, config=self.config)

    def test_large_tensor_processing(self):
        """Test processing of large tensors without OOM."""
        if self.device.type != 'cuda':
            self.skipTest("GPU not available")

        # Create large weight matrix (10K features)
        n_features = 10000
        n_dims = 512
        weight_matrix = torch.randn(n_features, n_dims, device=self.device)

        # Should process in batches without OOM
        result = self.metrics.compute_vector_interference(
            weight_matrix,
            batch_size=1000
        )

        self.assertIn('mean_overlap', result)
        self.assertEqual(result['n_features'], n_features)

        # Check memory was cleaned up
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated_after = torch.cuda.memory_allocated()
            # Memory should be reasonably low after cleanup
            self.assertLess(allocated_after / 1e9, 2.0)  # Less than 2GB

    def test_sequential_model_processing(self):
        """Test memory cleanup between model processing."""
        if self.device.type != 'cuda':
            self.skipTest("GPU not available")

        models = []
        for i in range(3):
            model = nn.Sequential(
                nn.Linear(768, 768),
                nn.ReLU(),
                nn.Linear(768, 768)
            ).to(self.device)
            models.append(model)

        test_batch = {
            'input_ids': torch.randint(0, 100, (8, 32), device=self.device)
        }

        # Process models sequentially
        for model in models:
            result = self.metrics.compute_superposition_strength(model, test_batch, n_probes=3)
            self.assertIn('superposition_ratio', result)

        # Memory should be cleaned up after each model
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            allocated_after = torch.cuda.memory_allocated()
            self.assertLess(allocated_after / 1e9, 1.0)  # Less than 1GB


class TestNumericalPrecision(unittest.TestCase):
    """Test numerical stability and precision."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = SuperpositionConfig(use_float64=False)
        self.metrics = SuperpositionMetrics(device=self.device, config=self.config)

    def test_extreme_scale_differences(self):
        """Test with extreme scale differences in data."""
        # Create weight matrix with extreme scales
        weight_matrix = torch.zeros(10, 5)
        weight_matrix[0] = 1e-10  # Very small
        weight_matrix[1] = 1e10   # Very large
        weight_matrix[2:] = torch.randn(8, 5)  # Normal scale

        result = self.metrics.compute_vector_interference(weight_matrix, normalize=True)

        # Should handle without NaN or Inf
        self.assertFalse(np.isnan(result['mean_overlap']))
        self.assertFalse(np.isinf(result['mean_overlap']))
        self.assertGreaterEqual(result['mean_overlap'], 0.0)
        self.assertLessEqual(result['mean_overlap'], 1.0)

    def test_near_zero_variance(self):
        """Test variance computation with near-identical values."""
        # Create nearly identical vectors
        base_vector = torch.randn(100)
        weight_matrix = base_vector.unsqueeze(0).repeat(50, 1)
        # Add tiny noise
        weight_matrix += torch.randn_like(weight_matrix) * 1e-8

        result = self.metrics.compute_vector_interference(weight_matrix)

        # Should compute variance without numerical errors
        self.assertGreaterEqual(result['std_overlap'], 0.0)
        self.assertFalse(np.isnan(result['std_overlap']))

    def test_power_law_fitting_edge_cases(self):
        """Test power law fitting with edge cases."""
        # Test with perfect power law
        sizes = np.array([100, 200, 400, 800, 1600])
        alpha_true = 0.75
        losses = 10.0 * sizes ** (-alpha_true)

        result = self.metrics.fit_scaling_law(sizes, losses, log_scale=True)

        self.assertIn('alpha', result)
        self.assertAlmostEqual(result['alpha'], alpha_true, places=2)
        self.assertGreater(result['r_squared'], 0.99)

        # Test with noise
        losses_noisy = losses * (1 + np.random.normal(0, 0.1, len(losses)))
        result_noisy = self.metrics.fit_scaling_law(sizes, losses_noisy, log_scale=True)

        self.assertIn('alpha', result_noisy)
        self.assertAlmostEqual(result_noisy['alpha'], alpha_true, delta=0.2)

    def test_gini_coefficient_computation(self):
        """Test Gini coefficient with known distributions."""
        # Perfect equality (all same)
        equal_values = torch.ones(100)
        result = self.metrics.compute_feature_sparsity(equal_values.unsqueeze(0))
        self.assertAlmostEqual(result['gini_coefficient'], 0.0, places=3)

        # Perfect inequality (one has everything)
        unequal_values = torch.zeros(100)
        unequal_values[0] = 100
        result = self.metrics.compute_feature_sparsity(unequal_values.unsqueeze(0))
        self.assertGreater(result['gini_coefficient'], 0.9)


class TestDeviceConsistency(unittest.TestCase):
    """Test device handling consistency."""

    def setUp(self):
        """Set up test fixtures."""
        self.cpu_metrics = SuperpositionMetrics(device=torch.device('cpu'))
        if torch.cuda.is_available():
            self.gpu_metrics = SuperpositionMetrics(device=torch.device('cuda'))
        else:
            self.gpu_metrics = None

    def test_cpu_gpu_consistency(self):
        """Test that results are consistent across devices."""
        if self.gpu_metrics is None:
            self.skipTest("GPU not available")

        # Create test data
        torch.manual_seed(42)
        weight_matrix = torch.randn(50, 30)

        # Compute on CPU
        cpu_result = self.cpu_metrics.compute_vector_interference(weight_matrix)

        # Compute on GPU
        gpu_result = self.gpu_metrics.compute_vector_interference(weight_matrix)

        # Results should be very close
        self.assertAlmostEqual(cpu_result['mean_overlap'], gpu_result['mean_overlap'], places=5)
        self.assertAlmostEqual(cpu_result['std_overlap'], gpu_result['std_overlap'], places=5)

    def test_mixed_device_inputs(self):
        """Test handling of inputs on different devices."""
        if not torch.cuda.is_available():
            self.skipTest("GPU not available")

        # CPU tensor with GPU metrics
        cpu_tensor = torch.randn(20, 10, device='cpu')
        result = self.gpu_metrics.compute_vector_interference(cpu_tensor)
        self.assertIn('mean_overlap', result)

        # GPU tensor with CPU metrics
        gpu_tensor = torch.randn(20, 10, device='cuda')
        result = self.cpu_metrics.compute_vector_interference(gpu_tensor)
        self.assertIn('mean_overlap', result)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SuperpositionConfig()
        self.metrics = SuperpositionMetrics(config=self.config)

    def test_empty_inputs(self):
        """Test handling of empty inputs."""
        # Empty weight matrix
        empty_matrix = torch.zeros(0, 10)
        result = self.metrics.compute_vector_interference(empty_matrix)
        self.assertEqual(result['n_features'], 0)
        self.assertEqual(result['mean_overlap'], 0.0)

    def test_single_feature(self):
        """Test with single feature."""
        single_feature = torch.randn(1, 10)

        # With diagonal excluded
        result = self.metrics.compute_vector_interference(single_feature, exclude_diagonal=True)
        self.assertEqual(result['mean_overlap'], 0.0)

        # With diagonal included
        result = self.metrics.compute_vector_interference(single_feature, exclude_diagonal=False)
        self.assertEqual(result['mean_overlap'], 1.0)

    def test_all_zero_activations(self):
        """Test with all-zero activations."""
        zero_activations = torch.zeros(100, 50)
        result = self.metrics.compute_feature_sparsity(zero_activations)

        self.assertEqual(result['sparsity'], 1.0)
        self.assertEqual(result['l0_norm'], 0.0)
        self.assertEqual(result['l1_l2_ratio'], 0.0)

    def test_nan_inf_inputs(self):
        """Test handling of NaN and Inf inputs."""
        # Test with NaN
        nan_tensor = torch.randn(10, 5)
        nan_tensor[0, 0] = float('nan')

        with self.assertRaises(ValueError) as context:
            self.metrics.compute_vector_interference(nan_tensor)
        self.assertIn('NaN', str(context.exception))

        # Test with Inf
        inf_tensor = torch.randn(10, 5)
        inf_tensor[0, 0] = float('inf')

        with self.assertRaises(ValueError) as context:
            self.metrics.compute_vector_interference(inf_tensor)
        self.assertIn('Inf', str(context.exception))

    def test_insufficient_data_for_scaling(self):
        """Test scaling analysis with insufficient data."""
        # Only one model
        models = {'model1': nn.Linear(10, 10)}
        test_batch = torch.randn(4, 10)

        result = self.metrics.analyze_dimensional_scaling(models, test_batch)
        self.assertIn('error', result)
        self.assertIn('Insufficient', result['error'])

    def test_svd_fallback(self):
        """Test SVD fallback mechanisms."""
        # Create a matrix that might cause SVD issues
        # Highly correlated rows
        base = torch.randn(1, 1000)
        problematic_matrix = base.repeat(100, 1) + torch.randn(100, 1000) * 0.001

        # Should use fallback methods if needed
        result = self.metrics.compute_superposition_strength(
            nn.Identity(),
            {'input_ids': problematic_matrix}
        )

        self.assertIn('effective_rank', result)
        self.assertGreater(result['effective_rank'], 0)


class TestRobustness(unittest.TestCase):
    """Test robustness under stress conditions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = SuperpositionConfig(
            svd_max_attempts=3,
            gradient_clip_norm=1.0
        )
        self.metrics = SuperpositionMetrics(config=self.config)

    def test_probe_training_robustness(self):
        """Test probe training with difficult tasks."""
        # Create challenging hidden states (nearly random)
        hidden_states = torch.randn(50, 100)

        # Add some structure to make it learnable
        hidden_states[:25] += torch.randn(1, 100) * 0.5

        accuracies = self.metrics._run_probe_experiments(hidden_states, n_probes=5)

        # Should complete without errors
        self.assertEqual(len(accuracies), 5)
        # At least some probes should do better than random
        self.assertGreater(max(accuracies), 0.55)

    def test_memory_stress(self):
        """Test under memory pressure."""
        # Allocate and free large tensors repeatedly
        for _ in range(5):
            large_tensor = torch.randn(5000, 1000)
            result = self.metrics.compute_vector_interference(
                large_tensor,
                batch_size=500
            )
            self.assertIn('mean_overlap', result)
            del large_tensor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def test_concurrent_processing(self):
        """Test processing multiple items concurrently."""
        weight_matrices = [torch.randn(100, 50) for _ in range(5)]

        results = []
        for matrix in weight_matrices:
            result = self.metrics.compute_vector_interference(matrix)
            results.append(result)

        # All should complete successfully
        self.assertEqual(len(results), 5)
        for result in results:
            self.assertIn('mean_overlap', result)


class TestConfiguration(unittest.TestCase):
    """Test configuration options."""

    def test_float64_mode(self):
        """Test computation with float64 precision."""
        config = SuperpositionConfig(use_float64=True)
        metrics = SuperpositionMetrics(config=config)

        weight_matrix = torch.randn(20, 10, dtype=torch.float64)
        result = metrics.compute_vector_interference(weight_matrix)

        self.assertIn('mean_overlap', result)
        # Results should be valid
        self.assertGreaterEqual(result['mean_overlap'], 0.0)
        self.assertLessEqual(result['mean_overlap'], 1.0)

    def test_custom_thresholds(self):
        """Test with custom configuration thresholds."""
        config = SuperpositionConfig(
            overlap_threshold=0.05,
            sparsity_relative_threshold=0.001,
            probe_accuracy_threshold=0.7
        )
        metrics = SuperpositionMetrics(config=config)

        weight_matrix = torch.randn(30, 20)
        result = metrics.compute_vector_interference(weight_matrix)

        # Should use custom threshold
        # Count of high overlaps should be affected by lower threshold
        self.assertIn('num_high_overlap_pairs', result)


class TestIntegration(unittest.TestCase):
    """Integration tests with real model-like structures."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics = SuperpositionMetrics(device=self.device)

    def test_full_pipeline(self):
        """Test full analysis pipeline."""
        # Create a simple model
        model = nn.Sequential(
            nn.Embedding(100, 64),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.Linear(64, 100)
        ).to(self.device)

        # Create test batch
        test_batch = {
            'input_ids': torch.randint(0, 100, (8, 16), device=self.device),
            'labels': torch.randint(0, 100, (8, 16), device=self.device)
        }

        # Test vector interference
        embedding_weight = model[0].weight
        interference_result = self.metrics.compute_vector_interference(embedding_weight)
        self.assertIn('mean_overlap', interference_result)

        # Test superposition strength
        superposition_result = self.metrics.compute_superposition_strength(
            model, test_batch, n_probes=3
        )
        self.assertIn('superposition_ratio', superposition_result)

        # Test feature sparsity
        with torch.no_grad():
            inputs = test_batch['input_ids']
            embeddings = model[0](inputs)
            sparsity_result = self.metrics.compute_feature_sparsity(embeddings)
        self.assertIn('sparsity', sparsity_result)

        # Test capacity estimation
        capacity_result = self.metrics.compute_representation_capacity(
            model, test_batch, n_probes=3
        )
        self.assertIn('estimated_capacity', capacity_result)

    def test_checkpoint_analysis(self):
        """Test analysis across checkpoints."""
        # Create mock checkpoints
        checkpoints = []
        for i in range(3):
            model = nn.Sequential(
                nn.Embedding(50, 32),
                nn.Linear(32, 50)
            )
            # Simulate training by adding noise
            with torch.no_grad():
                for param in model.parameters():
                    param.add_(torch.randn_like(param) * 0.1 * i)
            checkpoints.append(model)

        test_batch = {
            'input_ids': torch.randint(0, 50, (4, 8))
        }

        result = self.metrics.analyze_feature_emergence(
            checkpoints,
            test_batch,
            checkpoint_steps=[0, 100, 200]
        )

        self.assertIn('overlap_evolution', result)
        self.assertIn('effective_dimension_evolution', result)
        self.assertEqual(len(result['overlap_evolution']), 3)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)