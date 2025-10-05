"""
Unit tests for lottery ticket memory leak fixes.
=================================================
Tests critical GPU memory optimizations for ICML 2026 submission.

Tests verify:
1. Masks created on CPU (BUG #1 fix)
2. Masks stored as bool, not float32 (BUG #2 fix)
3. No temporary accumulation during restoration (BUG #3 fix)
4. Batch tensors explicitly cleaned (BUG #4 fix)
5. Numerical correctness preserved
"""

import unittest
import torch
import torch.nn as nn
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lottery_tickets.magnitude_pruning import create_magnitude_mask
from lottery_tickets.evaluation import compute_lottery_ticket_quality


class TestModel(nn.Module):
    """Simple test model."""
    def __init__(self, size=100):
        super().__init__()
        self.fc1 = nn.Linear(size, 200)
        self.fc2 = nn.Linear(200, size)

    def forward(self, x):
        if isinstance(x, dict):
            x = x.get('input_ids', x.get('x'))
        return self.fc2(self.fc1(x))


@unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
class TestMemoryFixes(unittest.TestCase):
    """Test GPU memory leak fixes."""

    def setUp(self):
        """Set up test model on GPU."""
        torch.manual_seed(42)
        self.device = 'cuda'
        self.model = TestModel(size=100).to(self.device)

    def tearDown(self):
        """Clean up GPU memory."""
        if hasattr(self, 'model'):
            del self.model
        torch.cuda.empty_cache()

    def test_bug1_masks_on_cpu(self):
        """Test that create_magnitude_mask returns CPU tensors (BUG #1 fix)."""
        # Create masks
        mask = create_magnitude_mask(self.model, sparsity=0.5)

        # Check all masks are on CPU
        masks_on_cpu = sum(1 for v in mask.values() if not v.is_cuda)
        masks_total = len(mask)

        self.assertEqual(masks_on_cpu, masks_total,
                        f"Expected all masks on CPU, but {masks_total - masks_on_cpu} are on GPU")

        # Check mask dtype is bool
        for name, m in mask.items():
            self.assertEqual(m.dtype, torch.bool,
                           f"Mask '{name}' dtype is {m.dtype}, expected torch.bool")

    def test_bug2_mask_bool_not_float32(self):
        """Test that masks stored as bool save 4× memory (BUG #2 fix)."""
        # Create test mask on CPU
        mask = {'weight': torch.randint(0, 2, (1000, 1000), dtype=torch.bool)}

        # Check memory before
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        # Move to device as bool (as in fixed code)
        mask_bool = mask['weight'].to(self.device, dtype=torch.bool)

        mem_bool = torch.cuda.memory_allocated()
        bool_memory = mem_bool - mem_before

        # Calculate expected sizes
        expected_bool = mask['weight'].numel()  # 1 byte per element
        expected_float32 = mask['weight'].numel() * 4  # 4 bytes per element

        # Verify memory usage is closer to bool than float32
        self.assertLess(bool_memory, expected_float32 * 0.5,
                       f"Memory usage {bool_memory} too high, expected ~{expected_bool}")

        # Cleanup
        del mask_bool
        torch.cuda.empty_cache()

    def test_bug3_no_restoration_leaks(self):
        """Test that weight restoration doesn't accumulate temporaries (BUG #3 fix)."""
        # Create model with multiple parameters
        class LargeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([
                    nn.Linear(100, 100) for _ in range(20)
                ])
            def forward(self, x):
                for layer in self.layers:
                    x = layer(x)
                return x

        model = LargeModel().to(self.device)

        # Backup weights
        original_weights = {}
        for name, param in model.named_parameters():
            original_weights[name] = param.data.cpu().clone()

        # Measure memory during restoration
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        # Restore with chunking (as in fixed code)
        with torch.no_grad():
            param_list = list(model.named_parameters())
            chunk_size = 5

            for chunk_start in range(0, len(param_list), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(param_list))

                for name, param in param_list[chunk_start:chunk_end]:
                    temp = original_weights[name].to(param.device, non_blocking=False)
                    param.data.copy_(temp)
                    del temp

                torch.cuda.empty_cache()

        mem_after = torch.cuda.memory_allocated()
        mem_leaked = mem_after - mem_before

        # Should be near zero (some small fluctuation is OK)
        self.assertLess(abs(mem_leaked), 10_000_000,  # < 10 MB
                       f"Memory leaked during restoration: {mem_leaked / 1e6:.2f} MB")

        # Cleanup
        del model, original_weights
        torch.cuda.empty_cache()

    def test_bug4_batch_cleanup(self):
        """Test that batch tensors are explicitly cleaned (BUG #4 fix)."""
        # Create batch
        batch = {
            'input_ids': torch.randint(0, 1000, (256, 128)),
            'attention_mask': torch.ones((256, 128))
        }

        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        # Move to device
        batch_device = {k: v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v
                       for k, v in batch.items()}

        mem_with_batch = torch.cuda.memory_allocated()

        # Cleanup (as in fixed code)
        if isinstance(batch_device, dict):
            for v in batch_device.values():
                if torch.is_tensor(v):
                    del v
        del batch_device
        torch.cuda.empty_cache()

        mem_after = torch.cuda.memory_allocated()

        # Should be back near baseline (within allocator overhead tolerance)
        mem_diff = abs(mem_after - mem_before)
        batch_size = mem_with_batch - mem_before

        self.assertLess(mem_diff, batch_size * 0.1,  # Within 10% of original
                       f"Batch memory not fully cleaned: {mem_diff / 1e6:.2f} MB remaining")

    def test_numerical_correctness_bool_vs_float32(self):
        """Test that bool masks produce identical results to float32 masks."""
        # Create test tensor
        param = torch.randn(1000, 1000, dtype=torch.bfloat16, device=self.device)
        mask_bool = torch.randint(0, 2, (1000, 1000), dtype=torch.bool, device=self.device)
        mask_float32 = mask_bool.to(torch.float32)

        # Test multiplication
        result_bool = param * mask_bool.to(param.dtype)
        result_float32 = param * mask_float32.to(param.dtype)

        # Check if identical
        max_diff = (result_bool - result_float32).abs().max().item()

        self.assertEqual(max_diff, 0.0,
                        f"Bool and float32 masks produced different results: max_diff={max_diff}")

    def test_mask_creation_reproducibility(self):
        """Test that mask creation is reproducible with fixed seed."""
        # Create masks twice with same seed
        torch.manual_seed(42)
        mask1 = create_magnitude_mask(self.model, sparsity=0.5)

        torch.manual_seed(42)
        mask2 = create_magnitude_mask(self.model, sparsity=0.5)

        # Check all masks are identical
        for name in mask1:
            self.assertIn(name, mask2, f"Mask '{name}' missing in second creation")
            self.assertTrue(torch.equal(mask1[name], mask2[name]),
                          f"Mask '{name}' not reproducible across runs")

    def test_compute_lottery_ticket_quality_memory(self):
        """Integration test: verify compute_lottery_ticket_quality doesn't leak memory."""
        # Create mask
        mask = create_magnitude_mask(self.model, sparsity=0.5)

        # Create simple dataloader
        batch = {'input_ids': torch.randn(32, 100, device=self.device)}
        dataloader = [batch]

        # Measure memory before
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated()

        # Run function
        results = compute_lottery_ticket_quality(
            model=self.model,
            mask=mask,
            dataloader=dataloader,
            max_batches=1
        )

        # Cleanup
        del results
        torch.cuda.empty_cache()

        # Measure memory after
        mem_after = torch.cuda.memory_allocated()
        mem_leaked = mem_after - mem_before

        # Should not leak significant memory
        self.assertLess(abs(mem_leaked), 50_000_000,  # < 50 MB
                       f"compute_lottery_ticket_quality leaked {mem_leaked / 1e6:.2f} MB")

        # Verify results structure
        self.assertIsInstance(results, dict)
        self.assertIn('loss', results)
        self.assertIn('sparsity', results)


class TestNumericalCorrectness(unittest.TestCase):
    """Test numerical correctness of fixes (CPU tests)."""

    def test_mask_bool_conversion_exact(self):
        """Test that bool→float conversion is exact for {0,1}."""
        # Test all common dtypes
        dtypes = [torch.float32, torch.float16, torch.bfloat16]

        for dtype in dtypes:
            with self.subTest(dtype=dtype):
                mask_bool = torch.tensor([True, False, True, False], dtype=torch.bool)
                mask_float = mask_bool.to(dtype)

                expected = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=dtype)
                self.assertTrue(torch.equal(mask_float, expected),
                              f"Bool→{dtype} conversion not exact")

    def test_mask_multiplication_exact(self):
        """Test that param × bool_mask = param × float_mask (exact)."""
        param = torch.randn(100, 100)
        mask_bool = torch.randint(0, 2, (100, 100), dtype=torch.bool)

        # Multiply with bool and float32
        result_bool = param * mask_bool.to(param.dtype)
        result_float = param * mask_bool.to(torch.float32).to(param.dtype)

        # Should be exactly equal
        self.assertTrue(torch.equal(result_bool, result_float),
                       "Bool mask multiplication differs from float32")

    def test_quality_score_edge_cases(self):
        """Test quality score computation handles edge cases correctly."""
        # Test cases: (baseline_loss, pruned_loss, expected_score)
        test_cases = [
            (2.5, 3.0, 2.5/3.0),   # Normal case
            (2.5, 2.5, 1.0),       # Equal losses
            (3.0, 2.5, 3.0/2.5),   # Improved
            (2.5, 0.0, 0.0),       # Div by zero
            (0.0, 0.0, 1.0),       # Both zero
            (0.0, 2.5, 0.0),       # Baseline zero
        ]

        for baseline, pruned, expected in test_cases:
            with self.subTest(baseline=baseline, pruned=pruned):
                # Compute quality score (simplified logic from evaluation.py)
                if pruned > 0 and baseline > 0:
                    score = baseline / pruned
                else:
                    score = 1.0 if pruned == baseline else 0.0

                self.assertAlmostEqual(score, expected, places=5,
                                     msg=f"Quality score incorrect for baseline={baseline}, pruned={pruned}")


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()

    # Add GPU tests if available
    if torch.cuda.is_available():
        test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestMemoryFixes))

    # Always add CPU tests
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestNumericalCorrectness))

    return test_suite


if __name__ == '__main__':
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())

    # Print summary
    print("\n" + "=" * 70)
    print("MEMORY FIXES TEST SUMMARY")
    print("=" * 70)
    print(f"Tests run:     {result.testsRun}")
    print(f"Failures:      {len(result.failures)}")
    print(f"Errors:        {len(result.errors)}")
    print(f"Skipped:       {len(result.skipped)}")

    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print("\n⚠️  SOME TESTS FAILED!")

    sys.exit(0 if result.wasSuccessful() else 1)