#!/usr/bin/env python3
"""
Unit Tests for Gradient Management System

Comprehensive unit tests for the gradient management functionality integrated
into the UnifiedModelAnalysis framework.

These tests ensure proper gradient handling, memory optimization, and state
management for all metric types.
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch, MagicMock

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from gradient_manager import (
    GradientManager,
    GradientScope,
    GradientState,
    MemoryOptimizedBatchProcessor,
    classify_metric_gradient_requirements
)


class SimpleTestModel(nn.Module):
    """Simple model for testing."""
    def __init__(self, vocab_size=100, hidden_dim=64):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_size, hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_dim
        })()

    def forward(self, input_ids, output_hidden_states=False):
        x = self.embeddings(input_ids)
        logits = self.fc(x)
        hidden_states = [x] if output_hidden_states else None
        return type('Output', (), {
            'logits': logits,
            'hidden_states': hidden_states
        })()


class TestGradientManager(unittest.TestCase):
    """Test the GradientManager class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = GradientManager(enable_logging=False)
        self.model = SimpleTestModel()

    def test_get_current_state(self):
        """Test getting current gradient state."""
        state = self.manager.get_current_state(self.model)

        self.assertIsInstance(state, GradientState)
        self.assertEqual(len(state.param_states), 2)  # embeddings.weight, fc.weight, fc.bias
        self.assertTrue(state.training_mode)
        self.assertTrue(state.gradient_enabled)

    def test_disable_gradients_recursive(self):
        """Test recursive gradient disabling."""
        # Enable all gradients first
        for p in self.model.parameters():
            p.requires_grad_(True)

        self.manager.disable_gradients_recursive(self.model)

        for p in self.model.parameters():
            self.assertFalse(p.requires_grad, "All parameters should have gradients disabled")

    def test_enable_gradients_recursive(self):
        """Test recursive gradient enabling."""
        # Disable all gradients first
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.manager.enable_gradients_recursive(self.model)

        for p in self.model.parameters():
            self.assertTrue(p.requires_grad, "All parameters should have gradients enabled")

    def test_restore_state(self):
        """Test state restoration."""
        # Save original state
        original_state = self.manager.get_current_state(self.model)

        # Modify state
        self.manager.disable_gradients_recursive(self.model)
        self.model.eval()

        # Restore
        self.manager.restore_state(self.model, original_state)

        # Check restoration
        self.assertTrue(self.model.training)
        for p in self.model.parameters():
            self.assertTrue(p.requires_grad)

    def test_gradient_context_no_grad(self):
        """Test gradient context with no gradients."""
        with self.manager.gradient_context(
            self.model,
            requires_grad=False,
            gradient_scope=GradientScope.NONE
        ):
            # Check inside context
            self.assertFalse(self.model.training)
            for p in self.model.parameters():
                self.assertFalse(p.requires_grad)

        # Check restoration after context
        self.assertTrue(self.model.training)
        for p in self.model.parameters():
            self.assertTrue(p.requires_grad)

    def test_gradient_context_with_grad(self):
        """Test gradient context with gradients enabled."""
        # Start with gradients disabled
        self.manager.disable_gradients_recursive(self.model)

        with self.manager.gradient_context(
            self.model,
            requires_grad=True,
            gradient_scope=GradientScope.MODEL
        ):
            # Check inside context
            self.assertTrue(self.model.training)
            for p in self.model.parameters():
                self.assertTrue(p.requires_grad)

        # Check restoration
        for p in self.model.parameters():
            self.assertFalse(p.requires_grad)

    def test_gradient_context_eval_mode(self):
        """Test gradient context with eval mode."""
        with self.manager.gradient_context(
            self.model,
            requires_grad=True,
            gradient_scope=GradientScope.BOTH,
            eval_mode=True
        ):
            # Should have gradients but be in eval mode
            self.assertFalse(self.model.training)
            for p in self.model.parameters():
                self.assertTrue(p.requires_grad)

    def test_nested_gradient_contexts(self):
        """Test nested gradient contexts."""
        original = [p.requires_grad for p in self.model.parameters()]

        with self.manager.gradient_context(self.model, requires_grad=False):
            outer = [p.requires_grad for p in self.model.parameters()]
            self.assertTrue(all(not g for g in outer))

            with self.manager.gradient_context(self.model, requires_grad=True, gradient_scope=GradientScope.MODEL):
                inner = [p.requires_grad for p in self.model.parameters()]
                self.assertTrue(all(g for g in inner))

            # Outer context should be restored
            after_inner = [p.requires_grad for p in self.model.parameters()]
            self.assertTrue(all(not g for g in after_inner))

        # Original should be restored
        final = [p.requires_grad for p in self.model.parameters()]
        self.assertEqual(original, final)

    def test_prepare_batch_for_gradient_computation(self):
        """Test batch preparation for gradient computation."""
        batch = {
            'input_ids': torch.randn(4, 10, requires_grad=True),
            'attention_mask': torch.ones(4, 10),
            'labels': torch.randint(0, 100, (4, 10))
        }

        # Test with no gradients
        no_grad_batch = self.manager.prepare_batch_for_gradient_computation(
            batch, False, GradientScope.NONE
        )
        self.assertFalse(no_grad_batch['input_ids'].requires_grad)
        self.assertTrue(batch['input_ids'].requires_grad)  # Original unchanged

        # Test with gradients
        grad_batch = self.manager.prepare_batch_for_gradient_computation(
            batch, True, GradientScope.INPUTS
        )
        self.assertTrue(grad_batch['input_ids'].requires_grad)

    def test_check_memory_available(self):
        """Test memory availability checking."""
        # Should have at least 1MB available
        self.assertTrue(self.manager.check_memory_available(0.001))

        # Shouldn't have 10TB available
        self.assertFalse(self.manager.check_memory_available(10000))

    def test_optimize_for_memory(self):
        """Test memory optimization."""
        # Add gradient buffers
        for p in self.model.parameters():
            p.grad = torch.randn_like(p)

        # Disable gradients for some params
        params = list(self.model.parameters())
        params[0].requires_grad = False

        # Optimize
        self.manager.optimize_for_memory(self.model)

        # Check gradient cleared for non-required param
        self.assertIsNone(params[0].grad)


class TestMemoryOptimizedBatchProcessor(unittest.TestCase):
    """Test the MemoryOptimizedBatchProcessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.manager = GradientManager(enable_logging=False)
        self.processor = MemoryOptimizedBatchProcessor(self.manager, memory_limit_gb=8.0)
        self.model = SimpleTestModel()

    def test_process_metrics_separation(self):
        """Test that metrics are properly separated by gradient requirements."""
        batch = {'input_ids': torch.randint(0, 100, (4, 10))}

        metrics = [
            ('grad_metric1', {'requires_gradients': True, 'gradient_scope': 'both'}),
            ('no_grad_metric1', {'requires_gradients': False}),
            ('grad_metric2', {'requires_gradients': True, 'gradient_scope': 'model', 'eval_mode': True}),
            ('no_grad_metric2', {'requires_gradients': False}),
        ]

        call_order = []

        def mock_compute(name, model, batch, info):
            call_order.append(name)
            # Verify gradient state
            if info.get('requires_gradients'):
                if not any(p.requires_grad for p in model.parameters()):
                    return {'error': f'{name}: Expected gradients'}
                if info.get('eval_mode') and model.training:
                    return {'error': f'{name}: Should be in eval mode'}
            else:
                if any(p.requires_grad for p in model.parameters()):
                    return {'error': f'{name}: Should not have gradients'}
            return {'value': f'{name}_result'}

        results = self.processor.process_metrics(self.model, batch, metrics, mock_compute)

        # Check all succeeded
        for name, result in results.items():
            self.assertNotIn('error', result, f"Metric {name} failed: {result.get('error')}")

        # Check gradient-free processed first
        self.assertEqual(call_order[:2], ['no_grad_metric1', 'no_grad_metric2'])
        self.assertEqual(set(call_order[2:]), {'grad_metric1', 'grad_metric2'})

    def test_memory_constraints(self):
        """Test handling of memory constraints."""
        batch = {'input_ids': torch.randint(0, 100, (4, 10))}

        # Mock memory unavailable
        with patch.object(self.manager, 'check_memory_available', return_value=False):
            metrics = [
                ('expensive_metric', {'requires_gradients': True, 'expensive': True}),
            ]

            def mock_compute(name, model, batch, info):
                return {'value': 'should_not_reach'}

            results = self.processor.process_metrics(self.model, batch, metrics, mock_compute)

            self.assertIn('error', results['expensive_metric'])
            self.assertIn('memory', results['expensive_metric']['error'].lower())

    def test_gradient_cleanup_between_metrics(self):
        """Test that gradients are cleaned between metrics."""
        batch = {'input_ids': torch.randint(0, 100, (4, 10))}

        metrics = [
            ('grad_metric1', {'requires_gradients': True, 'gradient_scope': 'both'}),
            ('grad_metric2', {'requires_gradients': True, 'gradient_scope': 'both'}),
        ]

        gradient_states = []

        def mock_compute(name, model, batch, info):
            # Check no existing gradients
            has_grads = any(p.grad is not None and p.grad.abs().sum() > 0
                           for p in model.parameters())
            gradient_states.append((name, has_grads))

            # Compute some gradients
            output = model(batch['input_ids'])
            loss = output.logits.mean()
            loss.backward()

            return {'value': f'{name}_result'}

        results = self.processor.process_metrics(self.model, batch, metrics, mock_compute)

        # First metric should have no existing gradients
        self.assertFalse(gradient_states[0][1], "First metric should start with no gradients")
        # Second metric should also have no existing gradients (cleaned)
        self.assertFalse(gradient_states[1][1], "Second metric should start with cleaned gradients")


class TestMetricClassification(unittest.TestCase):
    """Test automatic metric classification."""

    def test_gradient_metric_classification(self):
        """Test classification of gradient-requiring metrics."""
        test_cases = [
            ('compute_gradient_pathology', True, GradientScope.BOTH, False),
            ('compute_fisher_importance', True, GradientScope.BOTH, True),
            ('compute_integrated_gradients', True, GradientScope.BOTH, False),
            ('compute_tracin_self_influence', True, GradientScope.BOTH, False),
        ]

        for metric_name, exp_grad, exp_scope, exp_eval in test_cases:
            with self.subTest(metric=metric_name):
                requires_grad, scope, eval_mode = classify_metric_gradient_requirements(metric_name)
                self.assertEqual(requires_grad, exp_grad)
                self.assertEqual(scope, exp_scope)
                self.assertEqual(eval_mode, exp_eval)

    def test_gradient_free_metric_classification(self):
        """Test classification of gradient-free metrics."""
        test_cases = [
            'compute_attention_entropy',
            'compute_superposition_score',
            'compute_dead_neurons',
            'compute_information_flow',
            'compute_mode_connectivity',
            'compute_representation_drift',
        ]

        for metric_name in test_cases:
            with self.subTest(metric=metric_name):
                requires_grad, scope, eval_mode = classify_metric_gradient_requirements(metric_name)
                self.assertFalse(requires_grad)
                self.assertEqual(scope, GradientScope.NONE)
                self.assertFalse(eval_mode)

    def test_unknown_metric_classification(self):
        """Test classification of unknown metrics."""
        requires_grad, scope, eval_mode = classify_metric_gradient_requirements('unknown_new_metric')
        self.assertFalse(requires_grad)  # Default to gradient-free
        self.assertEqual(scope, GradientScope.NONE)
        self.assertFalse(eval_mode)


class TestIntegrationWithUnifiedAnalysis(unittest.TestCase):
    """Test integration with UnifiedModelAnalysis (mock-based)."""

    def test_compute_with_gradient_context(self):
        """Test that compute_with_context properly uses gradient management."""
        # This would test the actual integration in unified_model_analysis.py
        # For now, we'll mock the key interactions

        manager = GradientManager(enable_logging=False)
        model = SimpleTestModel()
        batch = {'input_ids': torch.randint(0, 100, (4, 10))}

        # Test gradient metric
        with manager.gradient_context(model, requires_grad=True, gradient_scope=GradientScope.BOTH):
            output = model(batch['input_ids'])
            loss = output.logits.mean()
            loss.backward()

            # Verify gradients computed
            has_grads = all(p.grad is not None for p in model.parameters())
            self.assertTrue(has_grads)

        # Clear gradients
        model.zero_grad()

        # Test gradient-free metric
        with manager.gradient_context(model, requires_grad=False, gradient_scope=GradientScope.NONE):
            with torch.no_grad():
                output = model(batch['input_ids'])
                # Simulate superposition computation
                hidden = output.logits
                U, S, V = torch.svd(hidden.reshape(-1, hidden.shape[-1]))

            # Verify no gradients
            no_grads = all(p.grad is None or p.grad.abs().sum() == 0 for p in model.parameters())
            self.assertTrue(no_grads)


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and potential issues."""

    def test_integer_tensor_handling(self):
        """Test handling of integer tensors that can't have gradients."""
        manager = GradientManager(enable_logging=False)
        batch = {
            'input_ids': torch.randint(0, 100, (4, 10)),  # Integer tensor
            'float_input': torch.randn(4, 10)
        }

        # Prepare with no gradients - should work
        no_grad_batch = manager.prepare_batch_for_gradient_computation(
            batch, False, GradientScope.NONE
        )
        self.assertEqual(no_grad_batch['input_ids'].dtype, torch.long)

        # Prepare with gradients for inputs - integer tensors stay integer
        grad_batch = manager.prepare_batch_for_gradient_computation(
            batch, True, GradientScope.INPUTS
        )
        self.assertEqual(grad_batch['input_ids'].dtype, torch.long)
        self.assertTrue(grad_batch['float_input'].requires_grad)

    def test_cuda_memory_handling(self):
        """Test CUDA memory handling (skip if no CUDA)."""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        manager = GradientManager(enable_logging=False)
        model = SimpleTestModel().cuda()

        # Test memory check
        has_memory = manager.check_memory_available(0.001)  # 1MB
        self.assertTrue(has_memory)

        # Test memory optimization
        for p in model.parameters():
            p.grad = torch.randn_like(p)

        manager.optimize_for_memory(model, aggressive=True)

        # Verify CUDA cache cleared (no direct way to test, but shouldn't error)
        self.assertTrue(True)

    def test_exception_recovery(self):
        """Test recovery from exceptions within gradient context."""
        manager = GradientManager(enable_logging=False)
        model = SimpleTestModel()

        original_state = [p.requires_grad for p in model.parameters()]

        try:
            with manager.gradient_context(model, requires_grad=False):
                # Simulate an error
                raise ValueError("Test error")
        except ValueError:
            pass

        # Check state restored despite exception
        final_state = [p.requires_grad for p in model.parameters()]
        self.assertEqual(original_state, final_state)


def run_all_tests():
    """Run all unit tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTests(loader.loadTestsFromTestCase(TestGradientManager))
    suite.addTests(loader.loadTestsFromTestCase(TestMemoryOptimizedBatchProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestMetricClassification))
    suite.addTests(loader.loadTestsFromTestCase(TestIntegrationWithUnifiedAnalysis))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    return result.wasSuccessful()


if __name__ == "__main__":
    # Run with unittest's main for better integration
    unittest.main()