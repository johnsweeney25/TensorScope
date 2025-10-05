"""
Compatibility Adapter for LotteryTicketAnalysis
================================================
Maps old LotteryTicketAnalysis to new lottery_tickets module.
"""

import lottery_tickets
from typing import Any


class LotteryTicketAnalysis:
    """Compatibility wrapper - redirects to new lottery_tickets module."""

    def compute_pruning_robustness(self, *args, **kwargs):
        return lottery_tickets.compute_pruning_robustness(*args, **kwargs)

    def compute_gradient_importance(self, *args, **kwargs):
        return lottery_tickets.compute_gradient_importance(*args, **kwargs)

    def compute_iterative_magnitude_pruning(self, *args, **kwargs):
        return lottery_tickets.compute_iterative_magnitude_pruning(*args, **kwargs)

    def compute_early_bird_tickets(self, *args, **kwargs):
        return lottery_tickets.compute_early_bird_tickets(*args, **kwargs)

    def compute_layerwise_magnitude_ticket(self, *args, **kwargs):
        return lottery_tickets.compute_layerwise_magnitude_ticket(*args, **kwargs)

    def compute_ticket_overlap(self, *args, **kwargs):
        # Handle self parameter
        if 'initial_model' in kwargs:
            mask1 = lottery_tickets.create_magnitude_mask(
                kwargs['initial_model'],
                kwargs.get('sparsity', 0.9)
            )
        else:
            mask1 = kwargs.get('mask1', {})

        if 'trained_model' in kwargs:
            mask2 = lottery_tickets.create_magnitude_mask(
                kwargs['trained_model'],
                kwargs.get('sparsity', 0.9)
            )
        else:
            mask2 = kwargs.get('mask2', {})

        return lottery_tickets.compute_ticket_overlap(mask1, mask2)

    def compute_fisher_importance(self, *args, **kwargs):
        return lottery_tickets.compute_fisher_importance(*args, **kwargs)

    def compute_lottery_ticket_quality(self, *args, **kwargs):
        return lottery_tickets.compute_lottery_ticket_quality(*args, **kwargs)

    # Additional utility methods for compatibility
    def ensure_labels(self, batch, task_type='auto'):
        """Compatibility method."""
        if 'labels' not in batch and 'input_ids' in batch:
            batch['labels'] = batch['input_ids'].clone()
        return batch

    def get_batch_size(self, batch):
        """Compatibility method."""
        if isinstance(batch, dict):
            for key in ['input_ids', 'labels', 'x']:
                if key in batch:
                    return batch[key].shape[0]
        return 1


# For direct imports
compute_pruning_robustness = lottery_tickets.compute_pruning_robustness
compute_gradient_importance = lottery_tickets.compute_gradient_importance
compute_iterative_magnitude_pruning = lottery_tickets.compute_iterative_magnitude_pruning
compute_early_bird_tickets = lottery_tickets.compute_early_bird_tickets
compute_layerwise_magnitude_ticket = lottery_tickets.compute_layerwise_magnitude_ticket
