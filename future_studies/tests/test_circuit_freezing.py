"""
Test suite for attention circuit freezing
==========================================
Validates QK/OV circuit interventions across architectures.
"""

import unittest
import torch
import torch.nn as nn
import copy
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from future_studies import (
    AttentionCircuitFreezer,
    freeze_qk_circuit,
    freeze_ov_circuit,
    CircuitType,
    FreezeType,
    InterventionConfig,
    ModelArchitecture
)


class MockSeparateQKVAttention(nn.Module):
    """Mock LLaMA-style attention with separate QKV."""

    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_output)


class MockFusedQKVAttention(nn.Module):
    """Mock GPT-2 style attention with fused QKV."""

    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)  # Fused QKV
        self.c_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        # Fused QKV projection
        qkv = self.c_attn(hidden_states)
        q, k, v = qkv.split(self.hidden_size, dim=-1)

        # Reshape for attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.c_proj(attn_output)


class MockModel(nn.Module):
    """Mock transformer model for testing."""

    def __init__(self, attention_type='separate', num_layers=2):
        super().__init__()
        self.config = type('Config', (), {
            'num_attention_heads': 12,
            'hidden_size': 768,
            'num_key_value_heads': 12
        })()

        AttnClass = MockSeparateQKVAttention if attention_type == 'separate' else MockFusedQKVAttention

        # Create layers (mimicking transformer structure)
        self.model = nn.ModuleDict()
        self.model.layers = nn.ModuleList([
            type('Layer', (nn.Module,), {
                'self_attn': AttnClass(),
                'forward': lambda self, x: self.self_attn(x)
            })()
            for _ in range(num_layers)
        ])

    def forward(self, hidden_states):
        for layer in self.model.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class TestArchitectureDetection(unittest.TestCase):
    """Test model architecture detection."""

    def setUp(self):
        self.freezer = AttentionCircuitFreezer()

    def test_detect_separate_qkv(self):
        """Test detection of separate QKV architecture."""
        model = MockModel(attention_type='separate')
        architecture = self.freezer.detect_architecture(model)
        self.assertEqual(architecture, ModelArchitecture.SEPARATE_QKV)

    def test_detect_fused_qkv(self):
        """Test detection of fused QKV architecture."""
        model = MockModel(attention_type='fused')
        architecture = self.freezer.detect_architecture(model)
        self.assertEqual(architecture, ModelArchitecture.FUSED_QKV)

    def test_get_model_config(self):
        """Test extraction of model configuration."""
        model = MockModel()
        config = self.freezer.get_model_config(model)

        self.assertEqual(config['num_heads'], 12)
        self.assertEqual(config['hidden_size'], 768)
        self.assertEqual(config['head_dim'], 64)


class TestCircuitFreezing(unittest.TestCase):
    """Test circuit freezing functionality."""

    def setUp(self):
        self.freezer = AttentionCircuitFreezer()
        torch.manual_seed(42)

    def test_qk_freezing_separate(self):
        """Test QK circuit freezing on separate QKV model."""
        model = MockModel(attention_type='separate')
        model_copy = copy.deepcopy(model)

        # Create config
        config = InterventionConfig(
            layer_indices=[0],
            head_indices=[0, 1],
            circuit=CircuitType.QK,
            freeze_type=FreezeType.ZERO
        )

        # Apply freezing
        hooks = self.freezer.freeze_circuits(model_copy, config)
        self.assertGreater(len(hooks), 0)

        # Test forward pass
        input_tensor = torch.randn(2, 10, 768)
        baseline_output = model(input_tensor)
        frozen_output = model_copy(input_tensor)

        # Outputs should be different
        self.assertFalse(torch.allclose(baseline_output, frozen_output))

        # Clean up
        self.freezer.remove_hooks(hooks)

    def test_ov_freezing_separate(self):
        """Test OV circuit freezing on separate QKV model."""
        model = MockModel(attention_type='separate')
        model_copy = copy.deepcopy(model)

        config = InterventionConfig(
            layer_indices=[0],
            head_indices=[0, 1],
            circuit=CircuitType.OV,
            freeze_type=FreezeType.ZERO
        )

        hooks = self.freezer.freeze_circuits(model_copy, config)
        self.assertGreater(len(hooks), 0)

        input_tensor = torch.randn(2, 10, 768)
        baseline_output = model(input_tensor)
        frozen_output = model_copy(input_tensor)

        self.assertFalse(torch.allclose(baseline_output, frozen_output))

        self.freezer.remove_hooks(hooks)

    def test_qk_freezing_fused(self):
        """Test QK circuit freezing on fused QKV model."""
        model = MockModel(attention_type='fused')
        model_copy = copy.deepcopy(model)

        config = InterventionConfig(
            layer_indices=[0],
            head_indices=[0, 1],
            circuit=CircuitType.QK,
            freeze_type=FreezeType.ZERO
        )

        hooks = self.freezer.freeze_circuits(model_copy, config)
        self.assertGreater(len(hooks), 0)

        input_tensor = torch.randn(2, 10, 768)
        baseline_output = model(input_tensor)
        frozen_output = model_copy(input_tensor)

        self.assertFalse(torch.allclose(baseline_output, frozen_output))

        self.freezer.remove_hooks(hooks)

    def test_both_circuits_freezing(self):
        """Test freezing both QK and OV circuits (full head)."""
        model = MockModel(attention_type='separate')
        model_copy = copy.deepcopy(model)

        config = InterventionConfig(
            layer_indices=[0],
            head_indices=[0],
            circuit=CircuitType.BOTH,
            freeze_type=FreezeType.ZERO
        )

        hooks = self.freezer.freeze_circuits(model_copy, config)

        input_tensor = torch.randn(2, 10, 768)
        baseline_output = model(input_tensor)
        frozen_output = model_copy(input_tensor)

        # Should have strong effect
        diff = (baseline_output - frozen_output).abs().mean()
        self.assertGreater(diff, 0.01)

        self.freezer.remove_hooks(hooks)

    def test_gradient_preservation(self):
        """Test that gradients flow when preserve_gradients=True."""
        model = MockModel(attention_type='separate')
        model_copy = copy.deepcopy(model)
        model_copy.requires_grad_(True)

        config = InterventionConfig(
            layer_indices=[0],
            head_indices=[0],
            circuit=CircuitType.QK,
            freeze_type=FreezeType.ZERO,
            preserve_gradients=True
        )

        hooks = self.freezer.freeze_circuits(model_copy, config)

        input_tensor = torch.randn(2, 10, 768, requires_grad=True)
        output = model_copy(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        has_gradients = False
        for param in model_copy.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_gradients = True
                break

        self.assertTrue(has_gradients, "Gradients should flow through frozen circuits")

        self.freezer.remove_hooks(hooks)


class TestConvenienceFunctions(unittest.TestCase):
    """Test convenience functions."""

    def test_freeze_qk_circuit(self):
        """Test freeze_qk_circuit convenience function."""
        model = MockModel()
        model_copy = copy.deepcopy(model)

        hooks = freeze_qk_circuit(model_copy, [0], [0, 1])
        self.assertIsInstance(hooks, list)
        self.assertGreater(len(hooks), 0)

        # Clean up
        for hook in hooks:
            hook.remove()

    def test_freeze_ov_circuit(self):
        """Test freeze_ov_circuit convenience function."""
        model = MockModel()
        model_copy = copy.deepcopy(model)

        hooks = freeze_ov_circuit(model_copy, [0], [0, 1])
        self.assertIsInstance(hooks, list)
        self.assertGreater(len(hooks), 0)

        # Clean up
        for hook in hooks:
            hook.remove()


class TestGradientBehavior(unittest.TestCase):
    """Test gradient behavior in stopgrad vs STE modes."""

    def setUp(self):
        self.freezer = AttentionCircuitFreezer(debug_gradients=False)
        torch.manual_seed(42)

    def test_stopgrad_blocks_gradients(self):
        """Test that stopgrad mode blocks gradients."""
        model = MockModel(attention_type='separate', num_layers=1)
        model.requires_grad_(True)

        config = InterventionConfig(
            layer_indices=[0],
            head_indices=[0],
            circuit=CircuitType.QK,
            freeze_type=FreezeType.ZERO,
            backward_mode="stopgrad"
        )

        hooks = self.freezer.freeze_circuits(model, config)

        # Forward and backward
        input_tensor = torch.randn(1, 5, 768, requires_grad=True)
        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        # Check that Q/K projections have very small gradients
        q_proj = model.model.layers[0].self_attn.q_proj
        k_proj = model.model.layers[0].self_attn.k_proj

        if q_proj.weight.grad is not None:
            q_grad_norm = q_proj.weight.grad.norm().item()
            # In stopgrad mode, gradients should be near zero for frozen heads
            # Note: won't be exactly zero due to other heads
            self.assertLess(q_grad_norm, 1.0, "Q projection gradient should be reduced in stopgrad mode")

        self.freezer.remove_hooks(hooks)

    def test_ste_preserves_gradients(self):
        """Test that STE mode preserves gradient flow."""
        model = MockModel(attention_type='separate', num_layers=1)
        model.requires_grad_(True)

        # First get baseline gradient
        input_tensor = torch.randn(1, 5, 768, requires_grad=True)
        output_baseline = model(input_tensor)
        loss_baseline = output_baseline.sum()
        loss_baseline.backward()

        baseline_grad_norm = model.model.layers[0].self_attn.q_proj.weight.grad.norm().item()
        model.zero_grad()

        # Now test with STE
        config = InterventionConfig(
            layer_indices=[0],
            head_indices=[0],
            circuit=CircuitType.QK,
            freeze_type=FreezeType.ZERO,
            backward_mode="ste"
        )

        hooks = self.freezer.freeze_circuits(model, config)

        output = model(input_tensor)
        loss = output.sum()
        loss.backward()

        ste_grad_norm = model.model.layers[0].self_attn.q_proj.weight.grad.norm().item()

        # STE should preserve gradient magnitude (approximately)
        relative_diff = abs(ste_grad_norm - baseline_grad_norm) / baseline_grad_norm
        self.assertLess(relative_diff, 0.5, "STE should approximately preserve gradient magnitude")
        self.assertGreater(ste_grad_norm, 0.001, "STE should have non-zero gradients")

        self.freezer.remove_hooks(hooks)

    def test_gradient_monitoring(self):
        """Test that gradient monitoring works correctly."""
        model = MockModel(attention_type='separate', num_layers=1)

        # Enable debug mode
        freezer = AttentionCircuitFreezer(debug_gradients=True)

        config = InterventionConfig(
            layer_indices=[0],
            head_indices=[0],
            circuit=CircuitType.QK,
            backward_mode="stopgrad"
        )

        hooks = freezer.freeze_circuits(model, config)

        # Check gradient behavior
        input_tensor = torch.randn(1, 5, 768, requires_grad=True)
        grad_stats = freezer.check_gradient_behavior(model, input_tensor, config)

        # Verify stats are collected
        self.assertIn('backward_mode', grad_stats)
        self.assertIn('loss_value', grad_stats)
        self.assertIn('parameter_grads', grad_stats)
        self.assertEqual(grad_stats['backward_mode'], 'stopgrad')
        self.assertTrue(isinstance(grad_stats['loss_value'], float))

        freezer.remove_hooks(hooks)


class TestEquivalence(unittest.TestCase):
    """Test equivalence between fused and separate architectures."""

    def setUp(self):
        self.freezer = AttentionCircuitFreezer()
        torch.manual_seed(42)

    def test_qk_equivalence(self):
        """Test that QK freezing has similar effects on both architectures."""
        # Create matched models with same weights
        separate_model = MockModel(attention_type='separate', num_layers=1)
        fused_model = MockModel(attention_type='fused', num_layers=1)

        # Align weights (simplified - in practice would need careful mapping)
        with torch.no_grad():
            # Copy weights from separate to fused
            sep_attn = separate_model.model.layers[0].self_attn
            fused_attn = fused_model.model.layers[0].self_attn

            # Concatenate separate QKV into fused
            q_weight = sep_attn.q_proj.weight.data
            k_weight = sep_attn.k_proj.weight.data
            v_weight = sep_attn.v_proj.weight.data
            fused_attn.c_attn.weight.data = torch.cat([q_weight, k_weight, v_weight], dim=0)

        # Test input
        input_tensor = torch.randn(1, 5, 768)

        # Get baseline outputs
        sep_baseline = separate_model(input_tensor.clone())
        fused_baseline = fused_model(input_tensor.clone())

        # Apply same QK freezing to both
        config = InterventionConfig(
            layer_indices=[0],
            head_indices=[0, 1],
            circuit=CircuitType.QK,
            freeze_type=FreezeType.ZERO
        )

        sep_copy = copy.deepcopy(separate_model)
        fused_copy = copy.deepcopy(fused_model)

        sep_hooks = self.freezer.freeze_circuits(sep_copy, config)
        fused_hooks = self.freezer.freeze_circuits(fused_copy, config)

        # Get frozen outputs
        sep_frozen = sep_copy(input_tensor.clone())
        fused_frozen = fused_copy(input_tensor.clone())

        # Calculate effects
        sep_effect = (sep_baseline - sep_frozen).abs().mean()
        fused_effect = (fused_baseline - fused_frozen).abs().mean()

        # Effects should be similar (not identical due to numerical differences)
        relative_diff = abs(sep_effect - fused_effect) / (sep_effect + fused_effect)
        self.assertLess(relative_diff, 0.3, "Effects should be similar across architectures")

        # Clean up
        self.freezer.remove_hooks(sep_hooks)
        self.freezer.remove_hooks(fused_hooks)


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestArchitectureDetection))
    suite.addTest(unittest.makeSuite(TestCircuitFreezing))
    suite.addTest(unittest.makeSuite(TestConvenienceFunctions))
    suite.addTest(unittest.makeSuite(TestGradientBehavior))
    suite.addTest(unittest.makeSuite(TestEquivalence))
    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())