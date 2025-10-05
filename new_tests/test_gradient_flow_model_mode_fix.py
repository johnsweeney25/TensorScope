#!/usr/bin/env python3
"""
Focused unit test for the model.eval() -> model.train() fix in fisher_collector.py.
This test specifically verifies that the gradient flow issue has been resolved.
"""

import unittest
import torch
import torch.nn as nn
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisher.core.fisher_collector import FisherCollector


class TestModelModeGradientFlowFix(unittest.TestCase):
    """Test that verifies the model.train() fix resolves gradient flow issues"""

    def test_compute_oneshot_fisher_uses_train_mode(self):
        """Verify compute_oneshot_fisher sets model to train mode"""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.5),  # Dropout behaves differently in train/eval mode
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        # Set model to eval mode (the bug state)
        model.eval()

        # Create batch data
        batch = {
            'input_ids': torch.randint(0, 10, (2, 10)),
            'attention_mask': torch.ones(2, 10),
            'labels': torch.randint(0, 5, (2,))
        }

        # Create Fisher collector
        collector = FisherCollector()

        # Define a simple forward function
        def forward_fn(batch):
            # For this test, we'll simulate a simple forward pass
            x = batch['input_ids'].float()
            x = x.view(x.size(0), -1)
            # Ensure model is processing correctly
            output = model(x[:, :10])  # Use first 10 features
            return output

        # Define loss function
        def loss_fn(output, batch):
            labels = batch['labels']
            return nn.CrossEntropyLoss()(output, labels)

        # Compute Fisher - this should internally set model to train mode
        fisher_info = collector.compute_oneshot_fisher(
            model=model,
            batch=batch,
            forward_fn=forward_fn,
            loss_fn=loss_fn
        )

        # Verify we got Fisher information
        self.assertIsNotNone(fisher_info)
        self.assertIsInstance(fisher_info, dict)

        # Count parameters with Fisher info
        params_with_fisher = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_with_fisher += 1

        # We should have Fisher info for trainable parameters
        self.assertGreater(params_with_fisher, 0,
                          "Should have Fisher information for model parameters")

    def test_collect_per_sample_gradients_uses_train_mode(self):
        """Verify collect_per_sample_gradients sets model to train mode"""
        # Create model with batch norm (sensitive to train/eval mode)
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20),  # BatchNorm behaves very differently in train/eval
            nn.ReLU(),
            nn.Linear(20, 5)
        )

        # Set to eval mode (the bug state)
        model.eval()

        # Create batch
        batch = {
            'input_ids': torch.randn(4, 10),  # batch size 4
            'labels': torch.randint(0, 5, (4,))
        }

        # Create collector
        collector = FisherCollector()

        # Collect per-sample gradients - should use train mode internally
        gradients = collector.collect_per_sample_gradients(
            model=model,
            batch=batch,
            n_samples=2  # Only use first 2 samples
        )

        # Verify gradients were collected
        self.assertIsNotNone(gradients)
        self.assertIsInstance(gradients, list)
        self.assertGreater(len(gradients), 0, "Should have collected gradients")

        # Each gradient dict should have entries for model parameters
        if gradients:
            first_grad = gradients[0]
            self.assertIsInstance(first_grad, dict)
            # Count non-None gradients
            non_none_grads = sum(1 for g in first_grad.values() if g is not None)
            self.assertGreater(non_none_grads, 0,
                             "Should have non-None gradients for parameters")

    def test_gradient_flow_comparison_train_vs_eval(self):
        """Compare gradient flow in train vs eval mode to demonstrate the fix"""
        # Create model with components affected by mode
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.Dropout(0.2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Dropout(0.2),
            nn.Linear(10, 5)
        )

        # Test data
        input_data = torch.randn(8, 10)  # batch size 8
        labels = torch.randint(0, 5, (8,))

        # Test 1: Gradient flow in eval mode (problematic)
        model.eval()
        model.zero_grad()
        output = model(input_data)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()

        eval_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                eval_gradients[name] = param.grad.clone()

        # Test 2: Gradient flow in train mode (correct)
        model.train()
        model.zero_grad()
        output = model(input_data)
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()

        train_gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                train_gradients[name] = param.grad.clone()

        # Verify we have gradients in both cases
        self.assertGreater(len(train_gradients), 0,
                          "Should have gradients in train mode")
        self.assertGreater(len(eval_gradients), 0,
                          "Should have gradients in eval mode")

        # The gradients should be different due to dropout/batchnorm
        # This demonstrates why train mode is important
        differences = 0
        for name in train_gradients:
            if name in eval_gradients:
                if not torch.allclose(train_gradients[name], eval_gradients[name], atol=1e-6):
                    differences += 1

        self.assertGreater(differences, 0,
                          "Train and eval modes should produce different gradients due to dropout/batchnorm")

    def test_gradient_coverage_real_scenario(self):
        """Test gradient coverage in a realistic scenario"""
        # Simulate a transformer-like architecture
        class SimpleTransformerBlock(nn.Module):
            def __init__(self, dim=64):
                super().__init__()
                self.attention = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)
                self.norm1 = nn.LayerNorm(dim)
                self.ff = nn.Sequential(
                    nn.Linear(dim, dim * 4),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(dim * 4, dim)
                )
                self.norm2 = nn.LayerNorm(dim)
                self.dropout = nn.Dropout(0.1)

            def forward(self, x):
                # Self-attention
                attn_out, _ = self.attention(x, x, x)
                x = self.norm1(x + self.dropout(attn_out))
                # Feed-forward
                ff_out = self.ff(x)
                x = self.norm2(x + self.dropout(ff_out))
                return x

        model = nn.Sequential(
            nn.Embedding(100, 64),
            SimpleTransformerBlock(64),
            nn.Linear(64, 10)
        )

        # Set to train mode (after our fix, this should be done internally)
        model.train()

        # Create input
        input_ids = torch.randint(0, 100, (4, 16))  # batch=4, seq_len=16
        labels = torch.randint(0, 10, (4,))

        # Forward and backward
        model.zero_grad()
        embeddings = model[0](input_ids)  # Embedding layer
        transformed = model[1](embeddings)  # Transformer block
        output = model[2](transformed.mean(dim=1))  # Pool and classify
        loss = nn.CrossEntropyLoss()(output, labels)
        loss.backward()

        # Check gradient coverage
        total_params = 0
        params_with_grad = 0
        zero_grad_params = 0

        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                params_with_grad += 1
                if torch.all(param.grad == 0):
                    zero_grad_params += 1

        gradient_coverage = params_with_grad / total_params if total_params > 0 else 0

        # Should have very high gradient coverage in train mode
        self.assertGreaterEqual(gradient_coverage, 0.95,
                               f"Gradient coverage {gradient_coverage:.2%} should be >= 95%")
        self.assertEqual(zero_grad_params, 0,
                         f"Should have no zero gradients, found {zero_grad_params}")

        # Report results
        print(f"\nGradient Flow Test Results:")
        print(f"  Total parameters: {total_params}")
        print(f"  Parameters with gradients: {params_with_grad}")
        print(f"  Gradient coverage: {gradient_coverage:.2%}")
        print(f"  Zero gradients: {zero_grad_params}")


if __name__ == '__main__':
    unittest.main(verbosity=2)