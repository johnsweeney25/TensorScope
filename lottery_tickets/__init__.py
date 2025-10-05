"""
Lottery Tickets Module
======================
Well-organized, memory-efficient lottery ticket hypothesis analysis.

No god classes - each module has a single responsibility.
"""

# Core pruning methods
from .magnitude_pruning import (
    compute_pruning_robustness,
    compute_layerwise_magnitude_ticket,
    create_magnitude_mask
)

# Importance scoring methods
from .importance_scoring import (
    compute_gradient_importance,
    compute_fisher_importance,
    compute_taylor_importance,
    compute_magnitude_importance
)

# Early bird detection
from .early_bird import (
    compute_early_bird_tickets,
    detect_early_bird_convergence
)

# Evaluation and metrics
from .evaluation import (
    compute_lottery_ticket_quality,
    validate_pruning_correctness,
    compute_ticket_overlap
)

# IMP wrapper (for compatibility)
from .imp_wrapper import (
    compute_iterative_magnitude_pruning
)

# Utilities
from .utils import (
    ensure_deterministic_pruning,
    apply_mask,
    remove_mask,
    compute_sparsity
)

__all__ = [
    # Magnitude pruning
    'compute_pruning_robustness',
    'compute_layerwise_magnitude_ticket',
    'create_magnitude_mask',

    # Importance scoring
    'compute_gradient_importance',
    'compute_fisher_importance',
    'compute_taylor_importance',
    'compute_magnitude_importance',

    # Early bird
    'compute_early_bird_tickets',
    'detect_early_bird_convergence',

    # Evaluation
    'compute_lottery_ticket_quality',
    'validate_pruning_correctness',
    'compute_ticket_overlap',

    # IMP (compatibility)
    'compute_iterative_magnitude_pruning',

    # Utils
    'ensure_deterministic_pruning',
    'apply_mask',
    'remove_mask',
    'compute_sparsity'
]

# Version info
__version__ = '2.0.0'
__author__ = 'TensorScope Team'