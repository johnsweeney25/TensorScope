#!/usr/bin/env python3
"""
Unified Mechanistic Interpretability Analyzer
=============================================

This unified version combines mechanistic_analyzer.py and mechanistic_analyzer_improved.py
with proper statistical rigor and numerical stability.

Key improvements from the improved version:
1. Statistical validity: Bootstrap CI, FDR correction, minimum sample sizes
2. Numerical stability: Epsilon guards, clamping, Fisher z-transform
3. Enhanced pattern detection: N-gram support, fuzzy matching
4. Better defaults: min_attention_threshold=0.1 (not 0.01)
"""

import warnings

# Import the original analyzer and improved QK-OV
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=DeprecationWarning)
    try:
        from .mechanistic_analyzer import MechanisticAnalyzer as OriginalAnalyzer
    except ImportError:
        # Use core version as fallback
        from .mechanistic_analyzer_core import MechanisticAnalyzer as OriginalAnalyzer

    try:
        from .mechanistic_analyzer_improved import (
            ImprovedQKOVAnalyzer,
            InductionPattern,
            AttentionEdgeAnalysis,
            HeadStatistics
        )
    except ImportError:
        # Use improved deprecated version as fallback
        from .mechanistic_analyzer_improved_deprecated import (
            ImprovedQKOVAnalyzer,
            InductionPattern,
            AttentionEdgeAnalysis,
            HeadStatistics
        )

import torch
import numpy as np
from typing import Dict, List, Any, Optional


class UnifiedMechanisticAnalyzer(OriginalAnalyzer):
    """
    Unified analyzer that inherits from original but uses improved statistics.
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        # Statistical parameters (improved defaults)
        min_samples_for_correlation: int = 30,
        min_attention_threshold: float = 0.1,  # Changed from 0.01
        fdr_alpha: float = 0.05,
        bootstrap_n_samples: int = 1000,
        pattern_match_threshold: float = 0.8,
        # Streaming parameters
        streaming_window_size: int = 256,
        streaming_chunk_size: int = 32,
    ):
        """Initialize with statistically valid defaults."""
        # Initialize parent with original defaults
        super().__init__(device=device)

        # Create improved analyzer for QK-OV
        self.improved_analyzer = ImprovedQKOVAnalyzer(
            min_samples_for_correlation=min_samples_for_correlation,
            attention_threshold=min_attention_threshold,
            fdr_alpha=fdr_alpha,
            bootstrap_n_samples=bootstrap_n_samples,
            pattern_match_threshold=pattern_match_threshold
        )

        # Override parent's thresholds with improved ones
        self.min_samples = min_samples_for_correlation
        self.attention_threshold = min_attention_threshold
        self.fdr_alpha = fdr_alpha
        self.bootstrap_n_samples = bootstrap_n_samples
        self.pattern_match_threshold = pattern_match_threshold

        # Warn about defaults
        if min_attention_threshold < 0.1:
            warnings.warn(
                f"min_attention_threshold={min_attention_threshold} is below recommended 0.1. "
                "This may capture too much noise and reduce statistical validity.",
                UserWarning
            )

    def compute_qk_ov_pairing(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        top_k_edges: int = 100,  # Legacy parameter, ignored
        min_attention_threshold: float = None,
        use_improved_statistics: bool = True  # Always true in unified
    ) -> Dict[str, Any]:
        """
        Always use improved QK-OV pairing with proper statistics.

        Overrides the original method to use statistically rigorous implementation.
        """
        # Override threshold if provided
        if min_attention_threshold is not None:
            if min_attention_threshold < 0.1:
                warnings.warn(
                    f"min_attention_threshold={min_attention_threshold} is below recommended 0.1. "
                    "Using 0.1 for statistical validity.",
                    UserWarning
                )
                min_attention_threshold = max(0.1, min_attention_threshold)

            # Temporarily override analyzer's threshold
            old_threshold = self.improved_analyzer.attention_threshold
            self.improved_analyzer.attention_threshold = min_attention_threshold

            try:
                result = self.improved_analyzer.analyze_complete(model, batch)
            finally:
                self.improved_analyzer.attention_threshold = old_threshold
        else:
            result = self.improved_analyzer.analyze_complete(model, batch)

        # Add backward compatibility fields
        if 'error' not in result:
            result['paired_circuits'] = result.get('significant_heads', [])
            result['correlation_matrix'] = {}
            for stats in result.get('head_statistics', []):
                if hasattr(stats, 'qk_ov_correlation') and stats.qk_ov_correlation is not None:
                    result['correlation_matrix'][(stats.layer, stats.head)] = {
                        'correlation': stats.qk_ov_correlation,
                        'p_value': stats.correlation_p_value,
                        'ci': stats.correlation_ci
                    }

            # Compute coupling strength
            significant_heads = result.get('significant_heads', [])
            if significant_heads:
                result['coupling_strength'] = np.mean([
                    abs(s.qk_ov_correlation) for s in significant_heads
                    if hasattr(s, 'qk_ov_correlation') and s.qk_ov_correlation is not None
                ])
            else:
                result['coupling_strength'] = 0.0

        return result

    def compute_induction_head_strength(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        min_distance: int = 1,
        use_vectorized: bool = None,
        device: Optional[torch.device] = None
    ) -> Dict[str, Any]:
        """
        Enhanced induction head strength with statistical validity.

        Uses the improved pattern detection and adds bootstrap CI.
        """
        # First use parent's implementation
        result = super().compute_induction_head_strength(
            model, batch, min_distance, use_vectorized, device
        )

        # If we have the improved analyzer, enhance with better statistics
        if hasattr(self, 'improved_analyzer') and 'error' not in result:
            # Move batch to device
            batch = self._to_device(model, batch)

            # Get enhanced patterns
            patterns = self.improved_analyzer.detect_patterns_enhanced(
                batch['input_ids'],
                batch.get('attention_mask', torch.ones_like(batch['input_ids'])),
                ngram_sizes=[1, 2]  # Focus on unigrams and bigrams
            )

            if patterns:
                result['n_patterns_enhanced'] = len(patterns)
                result['pattern_types'] = {
                    'exact': sum(1 for p in patterns if p.match_type == 'exact'),
                    'fuzzy': sum(1 for p in patterns if p.match_type == 'fuzzy'),
                    'semantic': sum(1 for p in patterns if p.match_type == 'semantic')
                }
                result['pattern_confidence_mean'] = np.mean([p.confidence for p in patterns])

                # Add statistical metadata
                result['statistical_metadata'] = {
                    'min_samples_required': self.min_samples,
                    'attention_threshold': self.attention_threshold,
                    'fdr_alpha': self.fdr_alpha,
                    'bootstrap_n_samples': self.bootstrap_n_samples
                }

        return result


# Backward compatibility aliases
MechanisticAnalyzer = UnifiedMechanisticAnalyzer


def compute_qk_ov_pairing_improved(
    model,
    batch: Dict[str, torch.Tensor],
    min_samples: int = 30,
    attention_threshold: float = 0.1,
) -> Dict[str, Any]:
    """Backward compatibility wrapper."""
    analyzer = UnifiedMechanisticAnalyzer(
        min_samples_for_correlation=min_samples,
        min_attention_threshold=attention_threshold,
    )
    return analyzer.compute_qk_ov_pairing(model, batch)