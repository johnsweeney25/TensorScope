#!/usr/bin/env python3
"""
Comprehensive tests for FisherCollector.

Tests:
1. Conservation: sum of per-param Fisher ≈ group Fisher
2. Head mapping: for attention matrices, number of groups == num_attention_heads
3. Token invariance: duplicate the batch → token-normalized averages stay the same
4. EMA bias correction: constant synthetic grads recover true value
5. Group reduction correctness for different layer types
6. CPU offloading and retrieval
7. Stable key generation
"""

import torch
import torch.nn as nn
import numpy as np
import unittest
from fisher_collector import FisherCollector, GroupMetadata


class SimpleAttentionModel(nn.Module):
    """Simple model with attention layers for testing."""

    def __init__(self, vocab_size=100, hidden_size=64, num_heads=4, num_layers=2):
        super().__init__()
        self.config = type('Config', (), {
            'vocab_size': vocab_size,
            'hidden_size': hidden_size,
            'num_attention_heads': num_heads
        })()

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.transformer = nn.ModuleDict({
            'h': nn.ModuleList([
                nn.ModuleDict({
                    'attn': nn.ModuleDict({
                        'q_proj': nn.Linear(hidden_size, hidden_size),
                        'k_proj': nn.Linear(hidden_size, hidden_size),
                        'v_proj': nn.Linear(hidden_size, hidden_size),
                        'o_proj': nn.Linear(hidden_size, hidden_size),
                    }),
                    'mlp': nn.ModuleDict({
                        'fc1': nn.Linear(hidden_size, hidden_size * 4),
                        'fc2': nn.Linear(hidden_size * 4, hidden_size),
                    }),
                    'ln_1': nn.LayerNorm(hidden_size),
                    'ln_2': nn.LayerNorm(hidden_size),
                })
                for _ in range(num_layers)
            ])
        })
        self.ln_f = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask=None, labels=None):
        x = self.embedding(input_ids)

        # Simple forward pass through layers
        for layer in self.transformer['h']:
            # Attention
            attn_out = layer['attn']['q_proj'](x)  # Simplified
            attn_out = layer['attn']['o_proj'](attn_out)
            x = x + attn_out
            x = layer['ln_1'](x)

            # MLP
            mlp_out = layer['mlp']['fc1'](x)
            mlp_out = torch.relu(mlp_out)
            mlp_out = layer['mlp']['fc2'](mlp_out)
            x = x + mlp_out
            x = layer['ln_2'](x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            # Simple loss: predict first token
            loss = loss_fn(logits.mean(dim=1), labels[:, 0])

        return type('Output', (), {'loss': loss, 'logits': logits})()


class TestFisherCollector(unittest.TestCase):
    """Test suite for FisherCollector."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        self.model = SimpleAttentionModel(
            vocab_size=100,
            hidden_size=64,
            num_heads=4,
            num_layers=2
        )
        self.batch = {
            'input_ids': torch.randint(0, 100, (4, 20)),
            'attention_mask': torch.ones(4, 20),
            'labels': torch.randint(0, 100, (4, 20))
        }

    def test_conservation(self):
        """Test: sum of per-param Fisher ≈ group Fisher."""
        print("\n=== Testing Conservation Property ===")

        # Collect per-parameter Fisher
        collector_param = FisherCollector(reduction='param', storage='gpu_fp32')
        fisher_param = collector_param.collect_fisher(
            self.model, self.batch, task='test', mode='oneshot'
        )

        # Collect group-level Fisher
        collector_group = FisherCollector(reduction='group', storage='gpu_fp32')
        fisher_group = collector_group.collect_fisher(
            self.model, self.batch, task='test', mode='oneshot'
        )

        # For each parameter, verify conservation
        for name, param in self.model.named_parameters():
            if 'weight' in name and any(x in name for x in ['fc', 'mlp', 'linear']):
                # Find corresponding Fisher values
                param_key = None
                group_key = None

                for key in fisher_param.keys():
                    if name in key and key.endswith('|param'):
                        param_key = key
                        break

                for key in fisher_group.keys():
                    if name in key and 'channel' in key:
                        group_key = key
                        break

                if param_key and group_key:
                    # Get values
                    param_fisher = fisher_param[param_key]
                    group_fisher = fisher_group[group_key]

                    # Sum per-param Fisher over input dimensions
                    if len(param_fisher.shape) > 1:
                        param_sum = param_fisher.sum(dim=list(range(1, len(param_fisher.shape))))
                    else:
                        param_sum = param_fisher

                    # Compare
                    if param_sum.shape == group_fisher.shape:
                        rel_error = (param_sum - group_fisher).abs() / (group_fisher.abs() + 1e-8)
                        max_error = rel_error.max().item()
                        print(f"  {name}: max relative error = {max_error:.6f}")
                        self.assertLess(max_error, 1e-3, f"Conservation violated for {name}")

        print("  ✓ Conservation test passed")

    def test_head_mapping(self):
        """Test: attention layers have correct number of head groups."""
        print("\n=== Testing Head Mapping ===")

        collector = FisherCollector(reduction='group', storage='gpu_fp32')
        fisher = collector.collect_fisher(
            self.model, self.batch, task='test', mode='oneshot'
        )

        # Check attention layers
        num_heads = self.model.config.num_attention_heads
        head_keys = [k for k in fisher.keys() if 'head' in k]

        print(f"  Model has {num_heads} attention heads")
        print(f"  Found {len(head_keys)} head-type Fisher groups")

        # For each attention projection
        for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
            proj_keys = [k for k in head_keys if proj in k]
            if proj_keys:
                # Check shape of Fisher values
                for key in proj_keys:
                    fisher_value = fisher[key]
                    self.assertEqual(
                        fisher_value.shape[0], num_heads,
                        f"Head count mismatch for {key}: expected {num_heads}, got {fisher_value.shape[0]}"
                    )
                    print(f"  ✓ {proj}: {fisher_value.shape[0]} heads")

        print("  ✓ Head mapping test passed")

    def test_token_invariance(self):
        """Test: doubling batch → normalized values stay same."""
        print("\n=== Testing Token Invariance ===")

        collector = FisherCollector(reduction='group', storage='gpu_fp32')

        # Single batch
        fisher_single = collector.collect_fisher(
            self.model, self.batch, task='test', mode='oneshot'
        )

        # Clear for fresh collection
        collector.clear_fisher('test')

        # Double batch
        double_batch = {
            'input_ids': torch.cat([self.batch['input_ids'], self.batch['input_ids']]),
            'attention_mask': torch.cat([self.batch['attention_mask'], self.batch['attention_mask']]),
            'labels': torch.cat([self.batch['labels'], self.batch['labels']])
        }
        fisher_double = collector.collect_fisher(
            self.model, double_batch, task='test', mode='oneshot'
        )

        # Compare normalized values
        for key in fisher_single.keys():
            if key in fisher_double:
                single_val = fisher_single[key]
                double_val = fisher_double[key]

                # Should be approximately equal (normalized by tokens)
                rel_diff = (single_val - double_val).abs() / (single_val.abs() + 1e-8)
                max_diff = rel_diff.max().item()

                print(f"  {key.split('|')[1].split('.')[-1]}: max relative diff = {max_diff:.6f}")
                self.assertLess(max_diff, 1e-2, f"Token invariance violated for {key}")

        print("  ✓ Token invariance test passed")

    def test_ema_bias_correction(self):
        """Test: EMA bias correction recovers true value."""
        print("\n=== Testing EMA Bias Correction ===")

        collector = FisherCollector(reduction='group', storage='gpu_fp32', ema_decay=0.9)

        # Create constant gradients (easier to verify)
        model = nn.Linear(10, 5)
        model.eval()

        # Synthetic constant gradient
        def set_constant_gradient():
            model.zero_grad()
            model.weight.grad = torch.ones_like(model.weight) * 0.1
            model.bias.grad = torch.ones_like(model.bias) * 0.05

        # Update EMA multiple times (more iterations for convergence)
        for step in range(50):  # Increased iterations
            # Manually set gradients
            set_constant_gradient()

            # Mock batch (not actually used since we set grads manually)
            mock_batch = {
                'input_ids': torch.tensor([[1, 2, 3]]),
                'attention_mask': torch.tensor([[1, 1, 1]])
            }

            # Update Fisher EMA manually
            active_tokens = 3
            task = 'bias_test'

            if step == 0:
                collector.fisher_steps[f'{task}_steps'] = 0

            collector.fisher_steps[f'{task}_steps'] += 1
            collector._apply_global_decay(task)

            # Add squared gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_sq = param.grad.pow(2)
                    group_fisher, group_type, num_groups = collector._reduce_to_groups(
                        name, param.grad, param.shape, model
                    )
                    key = collector._make_key(task, name, group_type)

                    # Normalize by tokens
                    group_fisher = group_fisher / active_tokens

                    if key in collector.fisher_ema:
                        prev = collector.fisher_ema[key]
                        collector.fisher_ema[key] = prev + (1 - collector.ema_decay) * group_fisher
                    else:
                        collector.fisher_ema[key] = group_fisher

        # Get bias-corrected values
        corrected = collector.get_bias_corrected_fisher(task)

        # For Linear layer with shape (5, 10), after group reduction to channels:
        # Each output channel gets sum of squared gradients over input dims
        # Gradient is 0.1 for all weights, so each channel gets 10 * (0.1^2) = 0.1
        # Then normalized by 3 tokens: 0.1 / 3 ≈ 0.0333
        # But with group reduction, we sum over input dimension (10 dims)
        expected_weight_per_channel = (10 * 0.1 ** 2) / 3  # Sum over 10 input dims
        expected_bias = (0.05 ** 2) / 3  # Bias has no reduction

        for key, value in corrected.items():
            if 'weight' in key and 'channel' in key:
                # Weight values after channel reduction
                mean_val = value.mean().item()
                print(f"  Weight Fisher (channel): {mean_val:.6f}, Expected: {expected_weight_per_channel:.6f}")
                # Allow some tolerance due to EMA convergence
                self.assertAlmostEqual(mean_val, expected_weight_per_channel, places=2)
            elif 'bias' in key:
                # Bias values should converge to expected
                mean_val = value.mean().item()
                print(f"  Bias Fisher: {mean_val:.6f}, Expected: {expected_bias:.6f}")
                self.assertAlmostEqual(mean_val, expected_bias, places=3)

        print("  ✓ EMA bias correction test passed")

    def test_group_reduction_linear(self):
        """Test group reduction for Linear layers."""
        print("\n=== Testing Linear Layer Group Reduction ===")

        collector = FisherCollector(reduction='group')

        # Test Linear layer
        linear = nn.Linear(20, 10)
        grad = torch.randn(10, 20)  # Output x Input

        group_fisher, group_type, num_groups = collector._reduce_linear(
            grad.pow(2), linear.weight.shape
        )

        self.assertEqual(group_type, 'channel')
        self.assertEqual(num_groups, 10)  # Number of output channels
        self.assertEqual(group_fisher.shape, (10,))

        print(f"  Linear(20, 10) → {num_groups} channels")
        print("  ✓ Linear reduction test passed")

    def test_group_reduction_attention(self):
        """Test group reduction for Attention layers."""
        print("\n=== Testing Attention Layer Group Reduction ===")

        collector = FisherCollector(reduction='group')

        # Create a mock model with attention config
        hidden_size = 64
        num_heads = 4

        # Test Q/K/V projection
        qkv_weight_shape = (hidden_size, hidden_size)
        grad = torch.randn(*qkv_weight_shape)

        group_fisher, group_type, num_groups = collector._reduce_attention(
            'transformer.h.0.attn.q_proj.weight',
            grad.pow(2),
            qkv_weight_shape,
            self.model  # Has num_attention_heads in config
        )

        self.assertEqual(group_type, 'head')
        self.assertEqual(num_groups, num_heads)
        self.assertEqual(group_fisher.shape, (num_heads,))

        print(f"  Attention Q/K/V → {num_groups} heads")
        print("  ✓ Attention reduction test passed")

    def test_cpu_offloading(self):
        """Test CPU offloading and retrieval."""
        print("\n=== Testing CPU Offloading ===")

        collector = FisherCollector(reduction='group', storage='cpu_fp16')

        # Collect Fisher (should be offloaded to CPU)
        fisher = collector.collect_fisher(
            self.model, self.batch, task='test', mode='ema'
        )

        # Check storage location and dtype
        for key, value in collector.fisher_ema.items():
            self.assertEqual(value.device.type, 'cpu', f"Value not on CPU: {key}")
            self.assertEqual(value.dtype, torch.float16, f"Value not fp16: {key}")

        print("  ✓ Values offloaded to CPU as fp16")

        # Test retrieval
        if torch.cuda.is_available():
            for key, value in collector.fisher_ema.items():
                gpu_value = collector._retrieve_from_cpu(value, 'cuda')
                self.assertEqual(gpu_value.device.type, 'cuda')
                print(f"  ✓ Retrieved {key.split('|')[2]} to GPU")

        print("  ✓ CPU offloading test passed")

    def test_stable_keys(self):
        """Test stable key generation."""
        print("\n=== Testing Stable Key Schema ===")

        collector = FisherCollector()

        # Test various parameter names
        test_cases = [
            ('transformer.h.0.attn.q_proj.weight', 'head', 'test|transformer.h.0.attn.q_proj.weight|head'),
            ('transformer.h.1.mlp.fc1.weight', 'channel', 'test|transformer.h.1.mlp.fc1.weight|channel'),
            ('embedding.weight', 'token', 'test|embedding.weight|token'),
            ('ln_f.weight', 'row', 'test|ln_f.weight|row'),
        ]

        for param_name, group_type, expected_key in test_cases:
            key = collector._make_key('test', param_name, group_type)
            self.assertEqual(
                key, expected_key,
                f"Key mismatch: {key} != {expected_key}"
            )
            print(f"  ✓ {param_name} → {key}")

        print("  ✓ Stable key test passed")

    def test_metadata_tracking(self):
        """Test metadata tracking."""
        print("\n=== Testing Metadata Tracking ===")

        collector = FisherCollector(reduction='group')

        # Collect Fisher multiple times
        for i in range(3):
            collector.collect_fisher(
                self.model, self.batch, task='test', mode='ema'
            )

        # Check metadata
        metadata = collector.get_metadata()
        self.assertGreater(len(metadata), 0, "No metadata tracked")

        for key, meta in metadata.items():
            self.assertIsInstance(meta, GroupMetadata)
            self.assertGreater(meta.tokens_seen, 0)
            self.assertGreater(meta.steps, 0)
            print(f"  {key.split('|')[1].split('.')[-1]}: {meta.tokens_seen} tokens, {meta.steps} steps")

        print("  ✓ Metadata tracking test passed")

    def test_clear_fisher(self):
        """Test clearing Fisher values."""
        print("\n=== Testing Clear Fisher ===")

        collector = FisherCollector()

        # Collect for multiple tasks
        collector.collect_fisher(self.model, self.batch, task='task1', mode='ema')
        collector.collect_fisher(self.model, self.batch, task='task2', mode='ema')

        # Clear specific task
        collector.clear_fisher('task1')
        task1_keys = [k for k in collector.fisher_ema.keys() if k.startswith('task1|')]
        self.assertEqual(len(task1_keys), 0, "Task1 not cleared")

        task2_keys = [k for k in collector.fisher_ema.keys() if k.startswith('task2|')]
        self.assertGreater(len(task2_keys), 0, "Task2 accidentally cleared")

        print("  ✓ Selective clear works")

        # Clear all
        collector.clear_fisher()
        self.assertEqual(len(collector.fisher_ema), 0, "EMA not fully cleared")
        self.assertEqual(len(collector.fisher_steps), 0, "Steps not cleared")

        print("  ✓ Clear Fisher test passed")


def run_all_tests():
    """Run all FisherCollector tests."""
    print("=" * 60)
    print("FisherCollector Test Suite")
    print("=" * 60)

    # Create test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestFisherCollector)

    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Summary
    print("\n" + "=" * 60)
    if result.wasSuccessful():
        print("✅ All FisherCollector tests passed!")
    else:
        print(f"❌ {len(result.failures)} tests failed")
        for test, traceback in result.failures:
            print(f"\nFailed: {test}")
            print(traceback)
    print("=" * 60)

    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)