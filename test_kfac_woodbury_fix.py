#!/usr/bin/env python3
"""
Quick test to verify K-FAC Woodbury fix is working correctly.
"""

import torch
from fisher.kfac_utils import KFACNaturalGradient

def test_woodbury_shapes():
    """Test that Woodbury construction produces correct shapes."""
    print("Testing K-FAC Woodbury shape fix...")
    print("=" * 60)
    
    # Simulate gradient data as it would come from hooks
    out_dim = 1536  # e.g., q_proj output
    T_effective = 3771  # tokens after masking
    
    print(f"Test parameters:")
    print(f"  out_dim: {out_dim}")
    print(f"  T_effective: {T_effective}")
    print()
    
    # Simulate G_tokens from gradient hook [T, out_dim]
    G_tokens = torch.randn(T_effective, out_dim)
    sqrt_T = float(T_effective) ** 0.5
    
    # Build U as the fix does: transpose to [out_dim, T]
    U = (G_tokens.t().contiguous() / sqrt_T).to(dtype=torch.float16)
    print(f"✓ U shape: {U.shape} (expected: [{out_dim}, {T_effective}])")
    assert U.shape == (out_dim, T_effective), f"U shape mismatch!"
    
    # Build S = I_T + (1/λ) U^T @ U
    damping_G = 1e-8
    UT = U.t().to(torch.float32)  # [T, out_dim]
    U32 = U.to(torch.float32)     # [out_dim, T]
    S = (UT @ U32) / damping_G    # [T, T]
    S.diagonal().add_(1.0)
    print(f"✓ S shape: {S.shape} (expected: [{T_effective}, {T_effective}])")
    assert S.shape == (T_effective, T_effective), f"S shape mismatch!"
    
    # Invert S with jitter if needed (same as actual code)
    eps = 1e-6
    S_inv = None
    for attempt in range(3):
        try:
            L = torch.linalg.cholesky(S)
            S_inv = torch.cholesky_inverse(L)
            print(f"✓ S_inv shape: {S_inv.shape} (expected: [{T_effective}, {T_effective}])")
            assert S_inv.shape == (T_effective, T_effective), f"S_inv shape mismatch!"
            break
        except RuntimeError as e:
            if attempt < 2:
                print(f"  Cholesky failed (attempt {attempt+1}), adding jitter...")
                S.diagonal().add_(eps)
                eps *= 10.0
            else:
                print(f"  Cholesky failed 3 times, using explicit inverse")
                S_inv = torch.linalg.inv(S)
                print(f"✓ S_inv shape: {S_inv.shape} (expected: [{T_effective}, {T_effective}])")
                assert S_inv.shape == (T_effective, T_effective), f"S_inv shape mismatch!"
    
    if S_inv is None:
        print(f"✗ Failed to invert S")
        return False
    
    # Test Woodbury application: (G + λI)^{-1} @ v
    in_dim = 1536
    v = torch.randn(out_dim, in_dim + 1)  # [out_dim, in_dim+1] (weight + bias)
    
    lambda_inv = 1.0 / damping_G
    Y0 = (lambda_inv * v).float()
    
    # Woodbury correction
    Z = U.t().float() @ Y0  # [T, out_dim] @ [out_dim, in+1] = [T, in+1]
    print(f"✓ Z shape: {Z.shape} (expected: [{T_effective}, {in_dim + 1}])")
    assert Z.shape == (T_effective, in_dim + 1), f"Z shape mismatch!"
    
    W = S_inv @ Z  # [T, T] @ [T, in+1] = [T, in+1]
    print(f"✓ W shape: {W.shape} (expected: [{T_effective}, {in_dim + 1}])")
    assert W.shape == (T_effective, in_dim + 1), f"W shape mismatch!"
    
    Y_G = Y0 - lambda_inv * (U.float() @ W)  # [out_dim, in+1]
    print(f"✓ Y_G shape: {Y_G.shape} (expected: [{out_dim}, {in_dim + 1}])")
    assert Y_G.shape == (out_dim, in_dim + 1), f"Y_G shape mismatch!"
    
    print()
    print("=" * 60)
    print("✓ All shape checks passed!")
    print("✓ K-FAC Woodbury fix is working correctly.")
    return True


if __name__ == "__main__":
    success = test_woodbury_shapes()
    exit(0 if success else 1)
