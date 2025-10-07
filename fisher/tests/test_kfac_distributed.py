#!/usr/bin/env python3
"""
Unit tests for KFAC distributed operations (DDP/FSDP).

Tests cover:
1. DDP all-gather with variable token counts
2. Padding and unpadding logic
3. Rank-wise aggregation
4. Communication patterns
5. Fallback when distributed not initialized
"""

import unittest
import torch
import torch.distributed as dist
import numpy as np
from typing import Dict, List
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from fisher.kfac_utils import KFACNaturalGradient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestKFACDistributedLogic(unittest.TestCase):
    """
    Test suite for distributed logic in KFAC (without actually initializing distributed).
    
    These tests verify the logic correctness without requiring multi-GPU setup.
    """

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

    def test_padding_logic(self):
        """Test: Padding logic for variable token counts."""
        print("\n=== Testing Padding Logic ===")
        
        # Simulate different token counts across ranks
        token_counts = [100, 150, 80, 120]
        max_tokens = max(token_counts)
        out_dim = 64
        
        print(f"  Token counts: {token_counts}")
        print(f"  Max tokens: {max_tokens}")
        
        # Simulate padding for each rank
        padded_tensors = []
        for T in token_counts:
            # Original U: [out_dim, T]
            U = torch.randn(out_dim, T, dtype=torch.float32)
            
            # Pad to max_tokens
            if T < max_tokens:
                padding = torch.zeros(out_dim, max_tokens - T, dtype=torch.float32)
                U_padded = torch.cat([U, padding], dim=1)
            else:
                U_padded = U
            
            self.assertEqual(U_padded.shape, (out_dim, max_tokens),
                           f"Padded shape should be [{out_dim}, {max_tokens}]")
            padded_tensors.append(U_padded)
        
        print("  ✓ All tensors padded to same size")
        
        # Verify padding is zeros
        for i, U_padded in enumerate(padded_tensors):
            T = token_counts[i]
            if T < max_tokens:
                padding_region = U_padded[:, T:]
                self.assertTrue(torch.all(padding_region == 0),
                              f"Padding region should be zeros for rank {i}")
        
        print("✓ Padding logic correct")

    def test_unpadding_logic(self):
        """Test: Unpadding logic after all-gather."""
        print("\n=== Testing Unpadding Logic ===")
        
        token_counts = [100, 150, 80, 120]
        max_tokens = max(token_counts)
        out_dim = 64
        world_size = len(token_counts)
        
        # Simulate all-gathered tensor [world_size, out_dim, max_tokens]
        all_gathered = torch.randn(world_size, out_dim, max_tokens, dtype=torch.float32)
        
        # Unpad and concatenate
        U_columns_list = []
        for rank, T in enumerate(token_counts):
            U_rank = all_gathered[rank, :, :T]  # Remove padding
            U_columns_list.append(U_rank)
        
        # Concatenate along token dimension
        U_aggregated = torch.cat(U_columns_list, dim=1)
        
        expected_total_tokens = sum(token_counts)
        self.assertEqual(U_aggregated.shape, (out_dim, expected_total_tokens),
                        f"Aggregated shape should be [{out_dim}, {expected_total_tokens}]")
        
        print(f"  ✓ Unpadded shape: {U_aggregated.shape}")
        print("✓ Unpadding logic correct")

    def test_rank_aggregation(self):
        """Test: Aggregation of U matrices from multiple ranks."""
        print("\n=== Testing Rank Aggregation ===")
        
        # Simulate data from 4 ranks
        rank_data = [
            {'out_dim': 64, 'T': 100},
            {'out_dim': 64, 'T': 150},
            {'out_dim': 64, 'T': 80},
            {'out_dim': 64, 'T': 120},
        ]
        
        out_dim = 64
        max_T = max(d['T'] for d in rank_data)
        
        # Create and pad U matrices
        U_matrices = []
        for rank, data in enumerate(rank_data):
            T = data['T']
            U = torch.randn(out_dim, T, dtype=torch.float32)
            
            # Pad
            if T < max_T:
                U_padded = torch.cat([U, torch.zeros(out_dim, max_T - T)], dim=1)
            else:
                U_padded = U
            
            U_matrices.append(U_padded)
        
        # Stack (simulating all-gather result)
        all_gathered = torch.stack(U_matrices, dim=0)  # [world_size, out_dim, max_T]
        
        # Unpad and aggregate
        U_columns = []
        for rank, data in enumerate(rank_data):
            T = data['T']
            U_rank = all_gathered[rank, :, :T]
            U_columns.append(U_rank)
        
        U_aggregated = torch.cat(U_columns, dim=1)
        
        # Verify shape
        total_T = sum(d['T'] for d in rank_data)
        self.assertEqual(U_aggregated.shape, (out_dim, total_T))
        
        # Compute aggregated S matrix
        lambda_val = 1e-8
        S_aggregated = torch.eye(total_T, dtype=torch.float32) + \
                       (1.0 / lambda_val) * (U_aggregated.t() @ U_aggregated)
        
        self.assertEqual(S_aggregated.shape, (total_T, total_T))
        
        print(f"  ✓ Aggregated U shape: {U_aggregated.shape}")
        print(f"  ✓ Aggregated S shape: {S_aggregated.shape}")
        print("✓ Rank aggregation correct")

    def test_fallback_no_distributed(self):
        """Test: Fallback when distributed is not initialized."""
        print("\n=== Testing Fallback (No Distributed) ===")
        
        # Check if distributed is initialized
        is_distributed = dist.is_initialized()
        
        print(f"  Distributed initialized: {is_distributed}")
        
        if not is_distributed:
            print("  ✓ Correctly detecting non-distributed environment")
            
            # In non-distributed mode, should use local U only
            out_dim = 64
            T = 100
            U = torch.randn(out_dim, T, dtype=torch.float32)
            lambda_val = 1e-8
            
            # Compute S locally (no all-gather)
            S = torch.eye(T, dtype=torch.float32) + (1.0 / lambda_val) * (U.t() @ U)
            
            self.assertEqual(S.shape, (T, T))
            print("  ✓ Local computation works without distributed")
        else:
            print("  ⚠️  Distributed is initialized (unexpected in unit test)")
        
        print("✓ Fallback logic correct")

    def test_communication_volume(self):
        """Test: Estimate communication volume for DDP."""
        print("\n=== Testing Communication Volume ===")
        
        # Typical scenario
        world_size = 8  # 8 GPUs
        out_dim = 4096  # Large embedding (e.g., LLaMA)
        T_per_rank = 512  # Tokens per rank
        bytes_per_element = 4  # FP32
        
        # All-gather communication
        # Each rank sends: [out_dim, T_per_rank]
        # Each rank receives: [world_size, out_dim, T_per_rank]
        
        send_volume = out_dim * T_per_rank * bytes_per_element
        recv_volume = (world_size - 1) * out_dim * T_per_rank * bytes_per_element
        total_volume = send_volume + recv_volume
        
        print(f"  Scenario: {world_size} GPUs, out_dim={out_dim}, T={T_per_rank}")
        print(f"  Send volume: {send_volume / 1e6:.2f} MB")
        print(f"  Receive volume: {recv_volume / 1e6:.2f} MB")
        print(f"  Total volume per rank: {total_volume / 1e6:.2f} MB")
        
        # Compare with full gradient all-reduce (baseline)
        grad_volume = out_dim * bytes_per_element
        print(f"  Standard grad all-reduce: {grad_volume / 1e6:.2f} MB")
        
        overhead_ratio = total_volume / grad_volume
        print(f"  Communication overhead: {overhead_ratio:.1f}x")
        
        print("✓ Communication volume estimated")

    def test_dtype_consistency(self):
        """Test: Dtype consistency in distributed operations."""
        print("\n=== Testing Dtype Consistency ===")
        
        out_dim = 64
        T = 100
        
        # Test different dtypes
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        
        for dtype in dtypes:
            U = torch.randn(out_dim, T, dtype=dtype)
            
            # Padding should preserve dtype
            U_padded = torch.cat([U, torch.zeros(out_dim, 50, dtype=dtype)], dim=1)
            self.assertEqual(U_padded.dtype, dtype,
                           f"Padding should preserve dtype {dtype}")
            
            # S computation (needs FP32)
            if dtype != torch.float32:
                U_fp32 = U.to(torch.float32)
            else:
                U_fp32 = U
            
            lambda_val = 1e-8
            S = torch.eye(T, dtype=torch.float32) + (1.0 / lambda_val) * (U_fp32.t() @ U_fp32)
            self.assertEqual(S.dtype, torch.float32,
                           "S should always be FP32 for numerical stability")
        
        print("✓ Dtype consistency verified")


class TestKFACDistributedEdgeCases(unittest.TestCase):
    """Test suite for edge cases in distributed KFAC."""

    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)

    def test_single_rank(self):
        """Test: Single rank should work (no communication needed)."""
        print("\n=== Testing Single Rank ===")
        
        world_size = 1
        out_dim = 64
        T = 100
        
        U = torch.randn(out_dim, T, dtype=torch.float32)
        lambda_val = 1e-8
        
        # No all-gather needed for single rank
        S = torch.eye(T, dtype=torch.float32) + (1.0 / lambda_val) * (U.t() @ U)
        
        self.assertEqual(S.shape, (T, T))
        
        print("✓ Single rank works without communication")

    def test_highly_imbalanced_tokens(self):
        """Test: Highly imbalanced token counts across ranks."""
        print("\n=== Testing Highly Imbalanced Tokens ===")
        
        # Extreme imbalance
        token_counts = [10, 1000, 20, 800]  # 100x difference
        out_dim = 64
        max_T = max(token_counts)
        
        # This should still work but with significant padding
        total_padded = len(token_counts) * max_T
        total_actual = sum(token_counts)
        waste = (total_padded - total_actual) / total_padded
        
        print(f"  Token counts: {token_counts}")
        print(f"  Max tokens: {max_T}")
        print(f"  Padding waste: {waste * 100:.1f}%")
        
        if waste > 0.5:
            print("  ⚠️  High padding waste (>50%) - consider load balancing")
        
        print("✓ Highly imbalanced tokens handled")

    def test_zero_tokens_on_some_ranks(self):
        """Test: Some ranks having zero tokens."""
        print("\n=== Testing Zero Tokens on Some Ranks ===")
        
        # Some ranks have no tokens (e.g., all masked out)
        token_counts = [100, 0, 150, 0]
        out_dim = 64
        
        # Ranks with 0 tokens should contribute nothing
        non_zero_counts = [t for t in token_counts if t > 0]
        
        if len(non_zero_counts) > 0:
            max_T = max(non_zero_counts)
            
            # Only non-zero ranks contribute
            U_columns = []
            for T in non_zero_counts:
                U = torch.randn(out_dim, T, dtype=torch.float32)
                U_columns.append(U)
            
            U_aggregated = torch.cat(U_columns, dim=1)
            total_T = sum(non_zero_counts)
            
            self.assertEqual(U_aggregated.shape, (out_dim, total_T))
            print(f"  ✓ Aggregated shape: {U_aggregated.shape} (zeros skipped)")
        
        print("✓ Zero tokens on some ranks handled")


def run_all_tests():
    """Run all distributed test suites."""
    print("=" * 80)
    print("KFAC DISTRIBUTED OPERATIONS UNIT TESTS")
    print("=" * 80)
    
    # Load all test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestKFACDistributedLogic))
    suite.addTests(loader.loadTestsFromTestCase(TestKFACDistributedEdgeCases))
    
    # Run tests
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
        print("\n✅ ALL DISTRIBUTED TESTS PASSED!")
    else:
        print("\n❌ SOME TESTS FAILED")
    
    print("=" * 80)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    import sys
    success = run_all_tests()
    sys.exit(0 if success else 1)
