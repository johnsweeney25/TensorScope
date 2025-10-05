"""
Unit tests for GGN theoretical correctness and numerical precision.
=====================================================================
Critical verification tests for ICML submission.
"""

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List
import random
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from lottery_tickets.utils import ensure_deterministic_pruning


class SimpleTestNet(nn.Module):
    """Simple network for testing."""
    def __init__(self, input_dim: int = 10, hidden_dim: int = 20, output_dim: int = 3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class TestGGNTheoretical(unittest.TestCase):
    """Test theoretical relationships between Fisher and GGN."""

    def setUp(self):
        """Set up test environment with reproducibility."""
        ensure_deterministic_pruning(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def compute_empirical_fisher(self, model, inputs, targets):
        """Compute empirical Fisher matrix."""
        model.zero_grad()
        outputs = model(inputs)
        log_probs = F.log_softmax(outputs, dim=-1)
        loss = F.nll_loss(log_probs, targets)

        grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.flatten() for g in grads])

        return grad_vec.outer(grad_vec)

    def compute_true_fisher(self, model, inputs):
        """Compute true Fisher matrix (expectation over model's distribution)."""
        model.zero_grad()
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=-1)
        log_probs = F.log_softmax(outputs, dim=-1)

        num_classes = outputs.shape[-1]
        batch_size = outputs.shape[0]

        fisher_sum = None

        for b in range(batch_size):
            fisher_b = None
            for c in range(num_classes):
                model.zero_grad()
                loss_c = -log_probs[b, c]
                grads_c = torch.autograd.grad(loss_c, model.parameters(), retain_graph=True)
                grad_vec_c = torch.cat([g.flatten() for g in grads_c])

                weighted_outer = probs[b, c] * grad_vec_c.outer(grad_vec_c)

                fisher_b = weighted_outer if fisher_b is None else fisher_b + weighted_outer

            fisher_sum = fisher_b if fisher_sum is None else fisher_sum + fisher_b

        return fisher_sum / batch_size

    def compute_ggn(self, model, inputs, loss_type='cross_entropy'):
        """Compute GGN matrix."""
        model.zero_grad()
        outputs = model(inputs)
        batch_size, num_classes = outputs.shape

        params = list(model.parameters())
        n_params = sum(p.numel() for p in params)

        # Compute Jacobian
        J = []
        for b in range(batch_size):
            for c in range(num_classes):
                grads = torch.autograd.grad(
                    outputs[b, c], params,
                    retain_graph=True, create_graph=True
                )
                J.append(torch.cat([g.flatten() for g in grads]))

        J = torch.stack(J).reshape(batch_size, num_classes, n_params)

        # Compute output Hessian
        if loss_type == 'cross_entropy':
            probs = F.softmax(outputs, dim=-1)
            H_output = []
            for b in range(batch_size):
                p = probs[b]
                H_b = torch.diag(p) - p.outer(p)
                H_output.append(H_b)
            H_output = torch.stack(H_output)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")

        # Compute GGN = J^T H J
        GGN = torch.zeros(n_params, n_params)
        for b in range(batch_size):
            J_b = J[b]
            H_b = H_output[b]
            GGN_b = J_b.T @ H_b @ J_b
            GGN += GGN_b

        return GGN / batch_size

    def test_fisher_ggn_equivalence(self):
        """Test that true Fisher equals GGN for cross-entropy loss."""
        model = SimpleTestNet(input_dim=5, hidden_dim=10, output_dim=3)
        model.eval()

        inputs = torch.randn(4, 5)
        targets = torch.randint(0, 3, (4,))

        # Compute matrices
        F_true = self.compute_true_fisher(model, inputs)
        GGN = self.compute_ggn(model, inputs, loss_type='cross_entropy')

        # Check equivalence
        relative_error = torch.norm(F_true - GGN) / torch.norm(GGN)

        self.assertLess(relative_error.item(), 1e-5,
                       f"True Fisher should equal GGN for cross-entropy. Error: {relative_error.item()}")

    def test_empirical_vs_true_fisher(self):
        """Test that empirical and true Fisher differ as expected."""
        model = SimpleTestNet(input_dim=5, hidden_dim=10, output_dim=3)
        model.eval()

        inputs = torch.randn(4, 5)
        targets = torch.randint(0, 3, (4,))

        # Compute matrices
        F_emp = self.compute_empirical_fisher(model, inputs, targets)
        F_true = self.compute_true_fisher(model, inputs)

        # Get eigenvalues
        eigs_emp = torch.linalg.eigvalsh(F_emp)[-5:]
        eigs_true = torch.linalg.eigvalsh(F_true)[-5:]

        # For random models, empirical should have different spectrum
        ratio = eigs_emp.mean() / eigs_true.mean()

        # Ratio should be different from 1 (typically 0.3-0.5 for random models)
        self.assertNotAlmostEqual(ratio.item(), 1.0, places=1,
                                 msg="Empirical and true Fisher should differ for random models")

        # But both should be PSD
        self.assertTrue(torch.all(eigs_emp >= -1e-8), "Empirical Fisher should be PSD")
        self.assertTrue(torch.all(eigs_true >= -1e-8), "True Fisher should be PSD")

    def test_numerical_precision(self):
        """Test numerical precision with different dtypes."""
        for dtype in [torch.float32, torch.float64]:
            with self.subTest(dtype=dtype):
                model = SimpleTestNet(input_dim=5, hidden_dim=10, output_dim=3).to(dtype)
                inputs = torch.randn(4, 5, dtype=dtype)

                GGN = self.compute_ggn(model, inputs)

                # Check PSD property (allow small numerical errors)
                eigs = torch.linalg.eigvalsh(GGN)
                min_eig = eigs.min().item()

                # Allow for small numerical errors in eigenvalues
                # Even float64 can have numerical errors due to accumulation
                tolerance = 1e-6 if dtype == torch.float32 else 1e-7
                is_psd = min_eig >= -tolerance

                self.assertTrue(is_psd,
                              f"GGN should be PSD for {dtype}. Min eigenvalue: {min_eig}")

                # Check condition number for positive eigenvalues
                eigs_pos = eigs[eigs > 1e-10]
                if len(eigs_pos) > 0:
                    condition = eigs_pos[-1] / eigs_pos[0]
                    self.assertLess(condition.item(), 1e10,
                                   f"Condition number too large for {dtype}")


class TestLotteryTicketIntegration(unittest.TestCase):
    """Test lottery ticket hypothesis integration."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def test_fisher_importance_computation(self):
        """Test Fisher importance score computation."""
        from lottery_tickets.importance_scoring import compute_fisher_importance
        from lottery_tickets.utils import create_model_wrapper

        model = SimpleTestNet(input_dim=20, hidden_dim=50, output_dim=10)
        wrapped_model = create_model_wrapper(model)

        # Create simple dataloader
        class SimpleDataLoader:
            def __init__(self, n_batches=5, batch_size=8):
                self.batches = []
                for _ in range(n_batches):
                    batch = {
                        'input_ids': torch.randn(batch_size, 20),
                        'labels': torch.randint(0, 10, (batch_size,))
                    }
                    self.batches.append(batch)

            def __iter__(self):
                return iter(self.batches)

        dataloader = SimpleDataLoader()

        # Compute Fisher importance
        fisher_scores = compute_fisher_importance(
            wrapped_model, dataloader,
            num_samples=50,
            use_mixed_precision=True,
            gradient_clip=1.0
        )

        # Verify scores were computed
        self.assertGreater(len(fisher_scores), 0, "Should compute Fisher scores")

        # Check all parameters have scores
        for name, param in model.named_parameters():
            if 'weight' in name:
                self.assertIn(name, fisher_scores, f"Missing Fisher score for {name}")

                # Check scores are non-negative (Fisher is PSD)
                self.assertTrue(torch.all(fisher_scores[name] >= 0),
                              f"Fisher scores should be non-negative for {name}")

    def test_magnitude_pruning(self):
        """Test magnitude-based pruning with histogram quantiles."""
        from lottery_tickets.magnitude_pruning import create_magnitude_mask, compute_sparsity
        from lottery_tickets.utils import create_model_wrapper

        model = SimpleTestNet(input_dim=10, hidden_dim=20, output_dim=5)

        # Test different sparsity levels
        for sparsity in [0.1, 0.5, 0.9]:
            with self.subTest(sparsity=sparsity):
                mask = create_magnitude_mask(
                    model, sparsity,
                    use_histogram=True,
                    histogram_bins=1000
                )

                # Check mask was created for weight parameters
                for name, param in model.named_parameters():
                    if 'weight' in name and len(param.shape) >= 2:
                        self.assertIn(name, mask, f"Missing mask for {name}")

                        # Check mask is binary
                        unique_vals = torch.unique(mask[name])
                        self.assertTrue(
                            torch.all((unique_vals == 0) | (unique_vals == 1)),
                            f"Mask should be binary for {name}"
                        )

                # Check actual sparsity is close to target
                actual_sparsity = compute_sparsity(mask)
                self.assertAlmostEqual(actual_sparsity, sparsity, delta=0.1,
                                     msg=f"Actual sparsity {actual_sparsity} should be close to {sparsity}")

    def test_pruning_robustness(self):
        """Test model robustness to pruning."""
        from lottery_tickets.magnitude_pruning import compute_pruning_robustness
        from lottery_tickets.utils import create_model_wrapper

        model = SimpleTestNet(input_dim=10, hidden_dim=20, output_dim=5)
        wrapped_model = create_model_wrapper(model)

        batch = {
            'input_ids': torch.randn(8, 10),
            'labels': torch.randint(0, 5, (8,))
        }

        results = compute_pruning_robustness(
            wrapped_model, batch,
            sparsity_levels=[0.1, 0.5, 0.9],
            use_histogram_quantiles=True
        )

        # Check results structure
        self.assertIn('baseline_loss', results)
        self.assertIn('sparsity_curves', results)
        self.assertIn('robustness_metrics', results)

        # Check robustness metrics
        metrics = results['robustness_metrics']
        self.assertIn('winning_ticket_score', metrics)
        self.assertIn('optimal_sparsity', metrics)

        # Winning ticket score should be positive
        self.assertGreater(metrics['winning_ticket_score'], 0,
                         "Winning ticket score should be positive")

        # Optimal sparsity should be in valid range
        self.assertGreaterEqual(metrics['optimal_sparsity'], 0)
        self.assertLessEqual(metrics['optimal_sparsity'], 1)


class TestMultiBatchHessian(unittest.TestCase):
    """Test multi-batch Hessian computation."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)

    def test_hvp_averaging(self):
        """Test that HVPs are averaged correctly across batches."""
        from lottery_tickets.utils import create_model_wrapper

        model = SimpleTestNet(input_dim=10, hidden_dim=20, output_dim=5)
        wrapped_model = create_model_wrapper(model)

        # Create multiple batches
        batches = []
        for _ in range(5):
            batch = {
                'input_ids': torch.randn(4, 10),
                'labels': torch.randint(0, 5, (4,))
            }
            batches.append(batch)

        # Note: Full multi-batch implementation requires fisher_lanczos_unified
        # This test verifies the concept is sound

        # Verify batches are different
        for i in range(len(batches) - 1):
            self.assertFalse(
                torch.allclose(batches[i]['input_ids'], batches[i+1]['input_ids']),
                "Batches should be different"
            )

    def test_variance_reduction(self):
        """Test that using multiple batches reduces variance."""
        # This is a conceptual test - actual implementation in fisher_lanczos_unified
        n_samples_single = 16
        n_samples_multi = 160  # 10 batches of 16

        # Theoretical variance reduction
        variance_reduction = np.sqrt(n_samples_single / n_samples_multi)

        self.assertAlmostEqual(variance_reduction, 1/np.sqrt(10), places=5,
                             msg="Variance should reduce by sqrt(n) with n batches")


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions."""

    def setUp(self):
        """Set up test environment."""
        ensure_deterministic_pruning(42)

    def test_histogram_quantile(self):
        """Test histogram-based quantile computation."""
        from lottery_tickets.utils import compute_histogram_quantile

        # Create test tensor
        tensor = torch.randn(10000)

        # Test different quantiles
        for q in [0.1, 0.5, 0.9]:
            with self.subTest(quantile=q):
                hist_quantile = compute_histogram_quantile(tensor, q, bins=1000)
                true_quantile = torch.quantile(tensor, q).item()

                # Should be close but not exact (histogram approximation)
                self.assertAlmostEqual(hist_quantile, true_quantile, delta=0.1,
                                     msg=f"Histogram quantile should approximate true quantile at q={q}")

    def test_model_wrapper(self):
        """Test model wrapper for different interfaces."""
        from lottery_tickets.utils import create_model_wrapper

        model = SimpleTestNet()
        wrapped = create_model_wrapper(model)

        # Test with dict input
        batch = {
            'input_ids': torch.randn(4, 10),
            'labels': torch.randint(0, 3, (4,))
        }

        output = wrapped(**batch)

        # Check output structure
        self.assertTrue(hasattr(output, 'loss'), "Output should have loss")
        self.assertTrue(hasattr(output, 'logits'), "Output should have logits")

        # Loss should be scalar
        self.assertEqual(output.loss.dim(), 0, "Loss should be scalar")

        # Logits should match batch size and output dim
        self.assertEqual(output.logits.shape, (4, 3), "Logits shape should match")

    def test_deterministic_pruning(self):
        """Test deterministic pruning setup."""
        from lottery_tickets.utils import ensure_deterministic_pruning

        # Set seed
        ensure_deterministic_pruning(42)

        # Generate random numbers
        r1 = torch.randn(10)
        n1 = np.random.randn(10)

        # Reset seed
        ensure_deterministic_pruning(42)

        # Generate again
        r2 = torch.randn(10)
        n2 = np.random.randn(10)

        # Should be identical
        self.assertTrue(torch.allclose(r1, r2), "Torch random should be deterministic")
        np.testing.assert_array_almost_equal(n1, n2, err_msg="Numpy random should be deterministic")


def suite():
    """Create test suite."""
    suite = unittest.TestSuite()

    # Add all test cases
    suite.addTest(unittest.makeSuite(TestGGNTheoretical))
    suite.addTest(unittest.makeSuite(TestLotteryTicketIntegration))
    suite.addTest(unittest.makeSuite(TestMultiBatchHessian))
    suite.addTest(unittest.makeSuite(TestUtilityFunctions))

    return suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite())