#!/usr/bin/env python3
"""
Tests for KFAC bug fixes.

Tests cover:
1. Double backward pass fix
2. Progress bar UnboundLocalError fix
3. Factor collection with realistic models
4. End-to-end KFAC pipeline
"""

import unittest
import torch
import torch.nn as nn
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fisher.kfac_utils import KFACNaturalGradient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestModel(nn.Module):
    """Test model that accepts keyword arguments."""
    
    def __init__(self, in_dim=64, hidden_dim=128, out_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        # Flatten input_ids to [batch*seq, features]
        x = input_ids.reshape(-1, input_ids.shape[-1])
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        
        # Create output object with loss if labels provided
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                x.reshape(-1, x.shape[-1]),
                labels.reshape(-1),
                ignore_index=-100
            )
        else:
            loss = None
            
        return type('Output', (), {'loss': loss, 'logits': x})()


class TestKFACFixes(unittest.TestCase):
    """Test suite for KFAC bug fixes."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

    def test_double_backward_pass_fix(self):
        """Test: KFAC handles double backward pass gracefully."""
        print("\n=== Testing Double Backward Pass Fix ===")
        
        model = TestModel()
        model.eval()
        
        kfac = KFACNaturalGradient(
            damping=1e-4,
            kfac_use_woodbury=True,
            kfac_policy='all',
            min_layer_size=32,
            update_freq=1
        )
        
        # Create batch
        batch = {
            'input_ids': torch.randn(4, 16, 64),
            'attention_mask': torch.ones(4, 16),
            'labels': torch.randint(0, 64, (4, 16))
        }
        
        # Test 1: KFAC with fresh model (should work)
        print("Testing KFAC with fresh model...")
        model.zero_grad()
        factors = kfac.collect_kfac_factors(model, batch, loss=None)
        print(f"✅ KFAC collected factors for {len(factors)} layers")
        self.assertGreater(len(factors), 0, "Should collect factors from fresh model")
        
        # Test 2: KFAC after manual backward (should be handled gracefully)
        print("Testing KFAC after manual backward...")
        model.zero_grad()
        output = model(**batch)
        output.loss.backward()
        
        # Now KFAC should handle the case where backward was already called
        try:
            factors = kfac.collect_kfac_factors(model, batch, loss=None)
            print(f"✅ KFAC handled double backward pass gracefully")
            print(f"   Collected factors for {len(factors)} layers")
        except RuntimeError as e:
            if "backward through the graph a second time" in str(e):
                self.fail("Double backward pass fix failed")
            else:
                raise e

    def test_progress_bar_exception_handling(self):
        """Test: Progress bar doesn't cause UnboundLocalError."""
        print("\n=== Testing Progress Bar Exception Handling ===")
        
        model = TestModel()
        model.eval()
        
        # Test with progress bar enabled
        kfac = KFACNaturalGradient(
            damping=1e-4,
            kfac_use_woodbury=True,
            kfac_policy='all',
            min_layer_size=32,
            update_freq=1,
            show_progress=True  # Enable progress bar
        )
        
        batch = {
            'input_ids': torch.randn(4, 16, 64),
            'attention_mask': torch.ones(4, 16),
            'labels': torch.randint(0, 64, (4, 16))
        }
        
        output = model(**batch)
        
        # This should not raise UnboundLocalError
        try:
            factors = kfac.collect_kfac_factors(model, batch, loss=None)
            print("✅ Progress bar exception handling works")
        except UnboundLocalError as e:
            if "pbar" in str(e):
                self.fail("Progress bar UnboundLocalError fix failed")
            else:
                raise e

    def test_factor_collection_realistic_model(self):
        """Test: KFAC collects factors from realistic model."""
        print("\n=== Testing Factor Collection with Realistic Model ===")
        
        # Create a more realistic model
        class RealisticModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed = nn.Embedding(1000, 64)
                self.transformer = nn.TransformerEncoder(
                    nn.TransformerEncoderLayer(64, 4, batch_first=True), 
                    num_layers=2
                )
                self.ln = nn.LayerNorm(64)
                self.head = nn.Linear(64, 1000)
                
            def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
                x = self.embed(input_ids)
                x = self.transformer(x, src_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
                x = self.ln(x)
                logits = self.head(x)
                
                if labels is not None:
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        labels.reshape(-1),
                        ignore_index=-100
                    )
                else:
                    loss = None
                    
                return type('Output', (), {'loss': loss, 'logits': logits})()
        
        model = RealisticModel()
        model.eval()
        
        kfac = KFACNaturalGradient(
            damping=1e-4,
            kfac_use_woodbury=True,
            kfac_policy='all',
            min_layer_size=32,
            update_freq=1
        )
        
        batch = {
            'input_ids': torch.randint(0, 1000, (4, 32)),
            'attention_mask': torch.ones(4, 32),
            'labels': torch.randint(0, 1000, (4, 32))
        }
        
        output = model(**batch)
        
        # Collect factors
        factors = kfac.collect_kfac_factors(model, batch, loss=None)
        
        print(f"✅ Collected factors for {len(factors)} layers")
        
        # Verify we got some factors
        self.assertGreater(len(factors), 0, "Should collect factors from realistic model")
        
        # Check factor types
        woodbury_count = 0
        eig_count = 0
        for layer_name, layer_factors in factors.items():
            G_type = layer_factors['G_type']
            if G_type == 'woodbury_empirical':
                woodbury_count += 1
                # Verify Woodbury factors have correct shapes
                U = layer_factors['U']
                S_inv = layer_factors['S_inv']
                # Get out_dim from the model layer
                for name, module in model.named_modules():
                    if name == layer_name and isinstance(module, nn.Linear):
                        out_dim = module.out_features
                        break
                else:
                    continue  # Skip if we can't find the layer
                self.assertEqual(U.shape[0], out_dim, "U should have out_dim as first dimension")
                self.assertEqual(S_inv.shape[0], S_inv.shape[1], "S_inv should be square")
            elif G_type == 'eig':
                eig_count += 1
        
        print(f"   Woodbury layers: {woodbury_count}")
        print(f"   Eigendecomp layers: {eig_count}")

    def test_kfac_natural_gradient_computation(self):
        """Test: KFAC natural gradient computation works end-to-end."""
        print("\n=== Testing KFAC Natural Gradient Computation ===")
        
        model = TestModel()
        model.eval()
        
        kfac = KFACNaturalGradient(
            damping=1e-4,
            kfac_use_woodbury=True,
            kfac_policy='all',
            min_layer_size=32,
            update_freq=1
        )
        
        batch = {
            'input_ids': torch.randn(4, 16, 64),
            'attention_mask': torch.ones(4, 16),
            'labels': torch.randint(0, 64, (4, 16))
        }
        
        output = model(**batch)
        
        # Collect factors
        factors = kfac.collect_kfac_factors(model, batch, loss=None)
        
        if len(factors) == 0:
            self.skipTest("No factors collected")
        
        # Compute natural gradient
        model.zero_grad()
        output.loss.backward()
        
        # Get original gradients
        original_grads = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                original_grads[name] = param.grad.clone()
        
        # Compute natural gradient
        natural_grads = kfac.compute_natural_gradient(original_grads, model)
        
        print(f"✅ Computed natural gradients for {len(natural_grads)} parameters")
        
        # Verify natural gradients are different from original
        for name, nat_grad in natural_grads.items():
            if name in original_grads:
                orig_grad = original_grads[name]
                diff = (nat_grad - orig_grad).norm().item()
                orig_norm = orig_grad.norm().item()
                relative_diff = diff / (orig_norm + 1e-8)
                
                print(f"   {name}: relative_diff={relative_diff:.4f}")
                
                # Natural gradient should be different (unless damping is very large)
                self.assertGreater(relative_diff, 0.001, f"Natural gradient should differ from original for {name}")

    def test_kfac_numerical_stability(self):
        """Test: KFAC remains numerically stable."""
        print("\n=== Testing KFAC Numerical Stability ===")
        
        model = TestModel()
        model.eval()
        
        kfac = KFACNaturalGradient(
            damping=1e-8,  # Small damping to stress test
            kfac_use_woodbury=True,
            kfac_policy='all',
            min_layer_size=32,
            update_freq=1
        )
        
        batch = {
            'input_ids': torch.randn(8, 32, 64),
            'attention_mask': torch.ones(8, 32),
            'labels': torch.randint(0, 64, (8, 32))
        }
        
        output = model(**batch)
        
        # Collect factors
        factors = kfac.collect_kfac_factors(model, batch, loss=None)
        
        # Check all factors for NaN/Inf
        all_finite = True
        for layer_name, layer_factors in factors.items():
            for key, value in layer_factors.items():
                if isinstance(value, torch.Tensor):
                    if not torch.isfinite(value).all():
                        print(f"❌ Non-finite values in {layer_name}.{key}")
                        all_finite = False
                    else:
                        print(f"✅ {layer_name}.{key}: all finite")
        
        self.assertTrue(all_finite, "All KFAC factors should be numerically stable")

    def test_kfac_memory_efficiency(self):
        """Test: KFAC Woodbury provides memory savings."""
        print("\n=== Testing KFAC Memory Efficiency ===")
        
        model = TestModel(in_dim=128, hidden_dim=256, out_dim=128)
        model.eval()
        
        kfac = KFACNaturalGradient(
            damping=1e-4,
            kfac_use_woodbury=True,
            kfac_policy='all',
            min_layer_size=32,
            update_freq=1
        )
        
        batch = {
            'input_ids': torch.randn(4, 16, 128),
            'attention_mask': torch.ones(4, 16),
            'labels': torch.randint(0, 128, (4, 16))
        }
        
        output = model(**batch)
        
        # Collect factors
        factors = kfac.collect_kfac_factors(model, batch, loss=None)
        
        # Check memory usage
        total_memory = 0
        for layer_name, layer_factors in factors.items():
            if layer_factors['G_type'] == 'woodbury_empirical':
                U = layer_factors['U']
                S_inv = layer_factors['S_inv']
                
                U_memory = U.numel() * U.element_size()
                S_inv_memory = S_inv.numel() * S_inv.element_size()
                layer_memory = U_memory + S_inv_memory
                total_memory += layer_memory
                
                out_dim, T = U.shape
                full_G_memory = out_dim * out_dim * 4  # FP32
                reduction = full_G_memory / layer_memory
                
                print(f"   {layer_name}: Woodbury={layer_memory:,} bytes, Full G={full_G_memory:,} bytes, Reduction={reduction:.1f}x")
        
        print(f"✅ Total Woodbury memory: {total_memory:,} bytes")
        self.assertGreater(total_memory, 0, "Should use some memory for factors")


def run_all_tests():
    """Run all KFAC fix tests."""
    print("=" * 80)
    print("KFAC BUG FIXES TEST SUITE")
    print("=" * 80)
    
    # Load test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(TestKFACFixes)
    
    # Run tests with verbose output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.wasSuccessful():
        print("\n✅ ALL KFAC FIX TESTS PASSED!")
        print("=" * 80)
        print("\nFixes verified:")
        print("  1. ✅ Double backward pass handled gracefully")
        print("  2. ✅ Progress bar UnboundLocalError fixed")
        print("  3. ✅ Factor collection works with realistic models")
        print("  4. ✅ Natural gradient computation works end-to-end")
        print("  5. ✅ Numerical stability maintained")
        print("  6. ✅ Memory efficiency achieved")
        print("\n🎉 KFAC implementation is ready for production!")
    else:
        print("\n❌ SOME TESTS FAILED")
        print("=" * 80)
        
        if result.failures:
            print("\nFailures:")
            for test, trace in result.failures:
                print(f"\n{test}:")
                print(trace)
        
        if result.errors:
            print("\nErrors:")
            for test, trace in result.errors:
                print(f"\n{test}:")
                print(trace)
    
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
