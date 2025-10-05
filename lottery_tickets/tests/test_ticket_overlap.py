"""
ICML 2026: Unit tests for compute_ticket_overlap
=================================================
Tests theoretical correctness, numerical precision, and edge cases.
"""

import unittest
import torch
from lottery_tickets.evaluation import compute_ticket_overlap


class TestTicketOverlapTheoretical(unittest.TestCase):
    """Test theoretical correctness of overlap metrics."""

    def test_jaccard_index_basic(self):
        """Test basic Jaccard index computation."""
        mask1 = {'layer1': torch.tensor([1, 1, 0, 0], dtype=torch.bool)}
        mask2 = {'layer1': torch.tensor([1, 0, 1, 0], dtype=torch.bool)}

        result = compute_ticket_overlap(mask1, mask2, method='jaccard')

        # Expected: intersection=1, union=3, jaccard=1/3
        expected = 1.0 / 3.0
        self.assertAlmostEqual(result['overall_overlap'], expected, places=10)

    def test_dice_coefficient_basic(self):
        """Test basic Dice coefficient computation."""
        mask1 = {'layer1': torch.tensor([1, 1, 0, 0], dtype=torch.bool)}
        mask2 = {'layer1': torch.tensor([1, 0, 1, 0], dtype=torch.bool)}

        result = compute_ticket_overlap(mask1, mask2, method='dice')

        # Expected: intersection=1, |A|=2, |B|=2, dice=2*1/(2+2)=0.5
        self.assertAlmostEqual(result['overall_overlap'], 0.5, places=10)

    def test_overlap_coefficient_basic(self):
        """Test basic overlap coefficient computation."""
        mask1 = {'layer1': torch.tensor([1, 1, 0, 0], dtype=torch.bool)}
        mask2 = {'layer1': torch.tensor([1, 0, 1, 0], dtype=torch.bool)}

        result = compute_ticket_overlap(mask1, mask2, method='overlap')

        # Expected: intersection=1, min(|A|,|B|)=2, overlap=1/2=0.5
        self.assertAlmostEqual(result['overall_overlap'], 0.5, places=10)


class TestTicketOverlapEdgeCases(unittest.TestCase):
    """Test edge case handling."""

    def test_both_masks_empty(self):
        """CRITICAL: Both masks are empty (all zeros)."""
        mask1 = {'layer1': torch.zeros(100, dtype=torch.bool)}
        mask2 = {'layer1': torch.zeros(100, dtype=torch.bool)}

        for method in ['jaccard', 'dice', 'overlap']:
            with self.subTest(method=method):
                result = compute_ticket_overlap(mask1, mask2, method=method)
                self.assertEqual(result['overall_overlap'], 1.0,
                               f"{method}: Empty masks should return 1.0")
                self.assertTrue(any('empty' in w for w in result['warnings']),
                              f"{method}: Should warn about empty masks")

    def test_one_mask_empty(self):
        """CRITICAL: One mask is empty, the other is not."""
        mask1 = {'layer1': torch.zeros(100, dtype=torch.bool)}
        mask2 = {'layer1': torch.ones(100, dtype=torch.bool)}

        for method in ['jaccard', 'dice', 'overlap']:
            with self.subTest(method=method):
                result = compute_ticket_overlap(mask1, mask2, method=method)
                self.assertEqual(result['overall_overlap'], 0.0,
                               f"{method}: One empty mask should return 0.0")

    def test_identical_masks(self):
        """Test identical masks (perfect overlap)."""
        mask = torch.tensor([1, 1, 0, 1, 0, 0, 1, 1], dtype=torch.bool)
        mask1 = {'layer1': mask.clone()}
        mask2 = {'layer1': mask.clone()}

        for method in ['jaccard', 'dice', 'overlap']:
            with self.subTest(method=method):
                result = compute_ticket_overlap(mask1, mask2, method=method)
                self.assertAlmostEqual(result['overall_overlap'], 1.0, places=10,
                                     msg=f"{method}: Identical masks should return 1.0")

    def test_disjoint_masks(self):
        """Test completely disjoint masks (no overlap)."""
        mask1 = {'layer1': torch.tensor([1, 1, 0, 0, 0, 0], dtype=torch.bool)}
        mask2 = {'layer1': torch.tensor([0, 0, 1, 1, 0, 0], dtype=torch.bool)}

        for method in ['jaccard', 'dice', 'overlap']:
            with self.subTest(method=method):
                result = compute_ticket_overlap(mask1, mask2, method=method)
                self.assertEqual(result['overall_overlap'], 0.0,
                               f"{method}: Disjoint masks should return 0.0")

    def test_shape_mismatch(self):
        """Test handling of shape mismatches."""
        mask1 = {'layer1': torch.ones(10, dtype=torch.bool)}
        mask2 = {'layer1': torch.ones(20, dtype=torch.bool)}

        result = compute_ticket_overlap(mask1, mask2, method='jaccard')

        self.assertIn('layer1', result['skipped_layers'])
        self.assertTrue(any('mismatch' in w.lower() for w in result['warnings']))

    def test_missing_layers(self):
        """Test handling of missing layers."""
        mask1 = {
            'layer1': torch.ones(10, dtype=torch.bool),
            'layer2': torch.ones(10, dtype=torch.bool)
        }
        mask2 = {
            'layer2': torch.ones(10, dtype=torch.bool),
            'layer3': torch.ones(10, dtype=torch.bool)
        }

        result = compute_ticket_overlap(mask1, mask2, method='jaccard')

        self.assertIn('layer1', result['skipped_layers'])
        self.assertIn('layer3', result['skipped_layers'])
        self.assertEqual(len(result['warnings']), 2)


class TestTicketOverlapNumerical(unittest.TestCase):
    """Test numerical precision and reproducibility."""

    def test_numerical_precision_moderate_scale(self):
        """Test numerical precision with moderate-sized masks (10M parameters)."""
        n_params = 10_000_000
        sparsity = 0.9
        n_active = int(n_params * (1 - sparsity))

        torch.manual_seed(42)
        mask1_indices = torch.randperm(n_params)[:n_active]
        mask2_indices = torch.randperm(n_params)[:n_active]

        # Ground truth using sets
        set1 = set(mask1_indices.tolist())
        set2 = set(mask2_indices.tolist())
        intersection_gt = len(set1 & set2)
        union_gt = len(set1 | set2)
        jaccard_gt = intersection_gt / union_gt

        # Create tensor masks
        m1 = torch.zeros(n_params, dtype=torch.bool)
        m2 = torch.zeros(n_params, dtype=torch.bool)
        m1[mask1_indices] = True
        m2[mask2_indices] = True

        mask1_dict = {'layer1': m1}
        mask2_dict = {'layer1': m2}

        result = compute_ticket_overlap(mask1_dict, mask2_dict, method='jaccard')
        jaccard_actual = result['overall_overlap']

        self.assertAlmostEqual(jaccard_actual, jaccard_gt, places=10,
                              msg="Numerical precision error at 10M scale")

    def test_reproducibility(self):
        """Test reproducibility across runs."""
        torch.manual_seed(42)
        mask1 = {'layer1': torch.rand(1000) > 0.5}
        mask2 = {'layer1': torch.rand(1000) > 0.5}

        results = []
        for _ in range(5):
            result = compute_ticket_overlap(mask1, mask2, method='jaccard')
            results.append(result['overall_overlap'])

        # All should be identical
        self.assertEqual(len(set(results)), 1, "Results should be deterministic")

    def test_all_metrics_consistency(self):
        """Test consistency between metrics for identical masks."""
        mask = torch.tensor([1, 0, 1, 1, 0], dtype=torch.bool)
        mask1 = {'layer1': mask.clone()}
        mask2 = {'layer1': mask.clone()}

        jaccard = compute_ticket_overlap(mask1, mask2, method='jaccard')['overall_overlap']
        dice = compute_ticket_overlap(mask1, mask2, method='dice')['overall_overlap']
        overlap = compute_ticket_overlap(mask1, mask2, method='overlap')['overall_overlap']

        self.assertAlmostEqual(jaccard, 1.0, places=10)
        self.assertAlmostEqual(dice, 1.0, places=10)
        self.assertAlmostEqual(overlap, 1.0, places=10)


class TestTicketOverlapSummary(unittest.TestCase):
    """Test summary statistics."""

    def test_summary_contains_required_fields(self):
        """Test that summary contains all required fields."""
        mask1 = {'layer1': torch.ones(10, dtype=torch.bool)}
        mask2 = {'layer1': torch.zeros(10, dtype=torch.bool)}

        result = compute_ticket_overlap(mask1, mask2, method='jaccard')

        required_fields = [
            'total_intersection', 'total_union', 'total_params_mask1',
            'total_params_mask2', 'sparsity_mask1', 'sparsity_mask2',
            'layers_processed', 'layers_skipped', 'layers_mask1', 'layers_mask2'
        ]

        for field in required_fields:
            self.assertIn(field, result['summary'],
                         f"Summary missing required field: {field}")

    def test_sparsity_calculation(self):
        """Test sparsity calculation in summary."""
        # Mask with 90% zeros (90% sparsity)
        mask = torch.cat([torch.ones(10), torch.zeros(90)]).bool()
        mask1 = {'layer1': mask.clone()}
        mask2 = {'layer1': mask.clone()}

        result = compute_ticket_overlap(mask1, mask2, method='jaccard')

        expected_sparsity = 0.9
        self.assertAlmostEqual(result['summary']['sparsity_mask1'], expected_sparsity,
                              places=5, msg="Sparsity calculation incorrect")


if __name__ == '__main__':
    unittest.main(verbosity=2)