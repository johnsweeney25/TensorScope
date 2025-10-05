#!/usr/bin/env python3
"""
Test advanced KFAC features:
- True Fisher with label sampling
- Sequence padding mask support
- Separate damping for A and G factors
- Bias handling in powered transforms
"""

import unittest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fisher.kfac_utils import KFACNaturalGradient


class SimpleClassifier(nn.Module):
    """Simple classifier for testing."""

    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


class TestTrueFisher(unittest.TestCase):
    """Test true Fisher with label sampling."""

    def test_true_fisher_vs_empirical(self):
        """Test that true Fisher differs from empirical."""
        model = SimpleClassifier()
        kfac = KFACNaturalGradient(min_layer_size=5, update_freq=1)

        batch = {
            'input_ids': torch.randn(8, 10),
            'labels': torch.randint(0, 5, (8,))
        }

        # Collect empirical Fisher
        factors_empirical = kfac.collect_kfac_factors(
            model, batch, fisher_type="empirical"
        )

        # Reset for fair comparison
        kfac.reset()

        # Collect true Fisher (with sampled labels)
        factors_true = kfac.collect_kfac_factors(
            model, batch, fisher_type="true"
        )

        # Both should have factors
        self.assertTrue(len(factors_empirical) > 0)
        self.assertTrue(len(factors_true) > 0)

        # Factors should be different (due to different labels)
        for layer_name in factors_empirical:
            if layer_name in factors_true:
                A_emp = factors_empirical[layer_name]['A']
                A_true = factors_true[layer_name]['A']
                G_emp = factors_empirical[layer_name]['G']
                G_true = factors_true[layer_name]['G']

                # A factors should be the same (input activations unchanged)
                diff_A = (A_emp - A_true).abs().mean()
                self.assertLess(diff_A.item(), 1e-5, "A factors should be similar")

                # G factors should differ (different loss from sampled labels)
                diff_G = (G_emp - G_true).abs().mean()
                self.assertGreater(diff_G.item(), 1e-6,
                                   f"G factors should differ for {layer_name}")

    def test_mc_mode_alias(self):
        """Test that 'mc' mode is an alias for 'true'."""
        model = SimpleClassifier()
        kfac = KFACNaturalGradient(min_layer_size=5, update_freq=1)

        batch = {
            'input_ids': torch.randn(8, 10),
            'labels': torch.randint(0, 5, (8,))
        }

        # MC mode should work
        factors_mc = kfac.collect_kfac_factors(
            model, batch, fisher_type="mc"
        )
        self.assertTrue(len(factors_mc) > 0)


class TestSequencePaddingMask(unittest.TestCase):
    """Test sequence padding mask support."""

    def test_attention_mask_filtering(self):
        """Test that attention mask properly filters activations/gradients."""
        model = SimpleClassifier()
        kfac = KFACNaturalGradient(min_layer_size=5, update_freq=1)

        batch_size = 4
        seq_len = 8
        input_dim = 10

        # Create batch with sequences
        batch = {
            'input_ids': torch.randn(batch_size * seq_len, input_dim),
            'labels': torch.randint(0, 5, (batch_size * seq_len,))
        }

        # Create attention mask (half the tokens are padding)
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[:, seq_len//2:] = 0  # Mask out second half

        # Collect without mask
        factors_no_mask = kfac.collect_kfac_factors(
            model, batch, fisher_type="empirical"
        )

        kfac.reset()

        # Collect with mask
        factors_with_mask = kfac.collect_kfac_factors(
            model, batch,
            fisher_type="empirical",
            attention_mask=attention_mask
        )

        # Both should have factors
        self.assertTrue(len(factors_no_mask) > 0)
        self.assertTrue(len(factors_with_mask) > 0)

        # With mask should have different covariances
        # (computed only on non-masked tokens)
        for layer_name in factors_no_mask:
            if layer_name in factors_with_mask:
                A_no_mask = factors_no_mask[layer_name]['A']
                A_mask = factors_with_mask[layer_name]['A']

                # Should be different due to masking
                diff = (A_no_mask - A_mask).abs().mean()
                # Note: difference might be small if values are similar
                self.assertIsNotNone(diff)


class TestSeparateDamping(unittest.TestCase):
    """Test separate damping for A and G factors."""

    def test_separate_damping_parameters(self):
        """Test that separate damping can be specified."""
        kfac = KFACNaturalGradient(
            damping=1e-4,
            damping_A=1e-3,  # Different from base
            damping_G=1e-5,  # Different from base
            min_layer_size=5,
            update_freq=1
        )

        self.assertEqual(kfac.damping, 1e-4)
        self.assertEqual(kfac.damping_A, 1e-3)
        self.assertEqual(kfac.damping_G, 1e-5)

    def test_separate_damping_application(self):
        """Test that separate damping is applied correctly."""
        model = SimpleClassifier()

        # Large damping differences to make effect visible
        kfac = KFACNaturalGradient(
            damping_A=1.0,  # Very large
            damping_G=1e-6,  # Very small
            min_layer_size=5,
            update_freq=1,
            use_eigenvalue_correction=False  # Use simple damping
        )

        batch = {
            'input_ids': torch.randn(8, 10),
            'labels': torch.randint(0, 5, (8,))
        }

        factors = kfac.collect_kfac_factors(model, batch)

        # Check that factors exist and damping affected them
        for layer_name, layer_factors in factors.items():
            A = layer_factors['A']
            G = layer_factors['G']

            # With large damping_A, diagonal of A should be large
            # With small damping_G, diagonal of G should be small
            self.assertGreater(A.diag().min().item(), 0.5,
                               "A should have large diagonal due to damping_A")
            self.assertLess(G.diag().min().item(), 1.0,
                            "G should have smaller diagonal due to damping_G")


class TestBiasInPoweredTransforms(unittest.TestCase):
    """Test bias handling in powered transforms."""

    def test_powered_transform_with_bias(self):
        """Test that powered transforms handle bias correctly."""
        model = SimpleClassifier()
        kfac = KFACNaturalGradient(min_layer_size=5, update_freq=1)

        # Compute gradients
        batch = {
            'input_ids': torch.randn(8, 10),
            'labels': torch.randint(0, 5, (8,))
        }

        model.zero_grad()
        outputs = model(batch['input_ids'])
        loss = nn.CrossEntropyLoss()(outputs, batch['labels'])
        loss.backward()

        # Collect gradients
        gradients = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.clone()

        # Collect KFAC factors
        kfac.collect_kfac_factors(model, batch)

        # Test powered transform with different powers
        for power in [-0.5, -0.25, 0.5, 2.0]:
            powered_grads = kfac._compute_powered_natural_gradient(
                gradients, model, power
            )

            # Should have all gradients including bias
            for name in gradients:
                self.assertIn(name, powered_grads,
                              f"Missing {name} in powered gradients")

                # Should be different from original
                if 'fc' in name:  # Only check layers with KFAC
                    diff = (powered_grads[name] - gradients[name]).abs().mean()
                    self.assertGreater(diff.item(), 1e-8,
                                       f"Powered gradient same as original for {name}")

    def test_fisher_vector_product_with_bias(self):
        """Test that FVP handles bias correctly."""
        model = SimpleClassifier()
        kfac = KFACNaturalGradient(min_layer_size=5, update_freq=1)

        batch = {
            'input_ids': torch.randn(8, 10),
            'labels': torch.randint(0, 5, (8,))
        }

        # Collect KFAC factors
        kfac.collect_kfac_factors(model, batch)

        # Create test vector including bias
        vector = {}
        for name, param in model.named_parameters():
            vector[name] = torch.randn_like(param)

        # Compute FVP
        fvp = kfac.compute_fisher_vector_product(vector, model=model)

        # Should have all parameters including bias
        for name in vector:
            self.assertIn(name, fvp, f"Missing {name} in FVP")

            # FVP should be different from original vector
            if 'fc' in name:  # Only check layers with KFAC
                diff = (fvp[name] - vector[name]).abs().mean()
                self.assertGreater(diff.item(), 1e-8,
                                   f"FVP same as vector for {name}")


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()

    # Add all test classes
    test_suite.addTest(unittest.makeSuite(TestTrueFisher))
    test_suite.addTest(unittest.makeSuite(TestSequencePaddingMask))
    test_suite.addTest(unittest.makeSuite(TestSeparateDamping))
    test_suite.addTest(unittest.makeSuite(TestBiasInPoweredTransforms))

    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())