"""
DEPRECATED: This file is deprecated. Use mechanistic_analyzer_unified.py instead.

Improved QK-OV Pairing Analysis with Proper Statistical Validity
This module implements theoretically sound and statistically valid analysis
of attention head mechanisms in transformer models.
"""

import warnings
warnings.warn(
    "mechanistic_analyzer_improved.py is deprecated and will be removed in a future version. "
    "Please use mechanistic_analyzer_unified.py instead which incorporates all improvements.",
    DeprecationWarning,
    stacklevel=2
)

import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from scipy.stats import pearsonr, spearmanr, bootstrap, ttest_ind
from scipy.special import rel_entr
try:
    from statsmodels.stats.multitest import multipletests
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    # Simple Bonferroni correction as fallback
    def multipletests(pvals, alpha=0.05, method='bonferroni'):
        """Simple Bonferroni correction fallback."""
        corrected = np.array(pvals) * len(pvals)
        corrected = np.minimum(corrected, 1.0)
        rejected = corrected < alpha
        return rejected, corrected, None, None
import warnings
import logging

logger = logging.getLogger(__name__)

@dataclass
class InductionPattern:
    """Represents a detected induction pattern with statistical metadata."""
    batch_idx: int
    query_pos: int
    key_pos: int
    pattern_length: int
    pattern_tokens: List[int]
    target_token: int
    actual_token: int
    is_correct: bool
    match_type: str  # 'exact', 'fuzzy', 'semantic'
    confidence: float  # Statistical confidence in pattern detection

@dataclass
class AttentionEdgeAnalysis:
    """Complete analysis of an attention edge with statistical measures."""
    layer: int
    head: int
    edge_type: str  # 'induction', 'copying', 'positional', 'semantic'

    # QK Analysis (Attention Formation)
    qk_score: float
    attention_weight: float
    attention_entropy: float  # Entropy of attention distribution
    attention_rank: int  # Rank of this edge in attention distribution

    # OV Analysis (Value Transformation)
    ov_contribution: float  # Contribution to correct token
    ov_kl_divergence: float  # KL from predicted to actual distribution
    value_norm: float  # Norm of value vector

    # Statistical Measures
    confidence_interval: Tuple[float, float]
    p_value: float
    effect_size: float  # Cohen's d

@dataclass
class HeadStatistics:
    """Statistical summary for a single attention head."""
    layer: int
    head: int

    # Pattern Statistics
    n_patterns: int
    pattern_detection_rate: float
    copying_accuracy: float

    # QK Statistics (with confidence intervals)
    qk_mean: float
    qk_std: float
    qk_ci: Tuple[float, float]

    # OV Statistics (with confidence intervals)
    ov_mean: float
    ov_std: float
    ov_ci: Tuple[float, float]

    # Signal Detection Theory
    d_prime: float  # Discriminability index
    beta: float  # Response bias

    # Correlation Analysis (optional)
    qk_ov_correlation: Optional[float] = None
    correlation_p_value: Optional[float] = None
    correlation_ci: Optional[Tuple[float, float]] = None

    # Bootstrap Results (optional)
    bootstrap_samples: Optional[np.ndarray] = None


class ImprovedQKOVAnalyzer:
    """Statistically rigorous analyzer for QK-OV mechanisms in transformers."""

    def __init__(
        self,
        min_samples_for_correlation: int = 30,
        attention_threshold: float = 0.1,  # 10% minimum attention
        fdr_alpha: float = 0.05,
        bootstrap_n_samples: int = 1000,
        pattern_match_threshold: float = 0.8,  # For fuzzy matching
    ):
        self.min_samples = min_samples_for_correlation
        self.attention_threshold = attention_threshold
        self.fdr_alpha = fdr_alpha
        self.bootstrap_n_samples = bootstrap_n_samples
        self.pattern_match_threshold = pattern_match_threshold

    def detect_patterns_enhanced(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        allow_fuzzy: bool = True,
        max_edit_distance: int = 1,
        ngram_sizes: List[int] = [1, 2, 3],
    ) -> List[InductionPattern]:
        """
        Enhanced pattern detection with fuzzy matching and n-gram support.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            embeddings: Token embeddings for semantic matching [batch_size, seq_len, hidden_dim]
            allow_fuzzy: Enable approximate matching
            max_edit_distance: Maximum edit distance for fuzzy matching
            ngram_sizes: Sizes of n-grams to detect

        Returns:
            List of detected induction patterns with confidence scores
        """
        patterns = []
        batch_size, seq_len = input_ids.shape

        for batch_idx in range(batch_size):
            for n in ngram_sizes:
                patterns.extend(
                    self._detect_ngram_patterns(
                        input_ids[batch_idx],
                        attention_mask[batch_idx],
                        n,
                        batch_idx,
                        embeddings[batch_idx] if embeddings is not None else None,
                        allow_fuzzy,
                        max_edit_distance
                    )
                )

        # Compute confidence scores based on pattern frequency
        pattern_counts = defaultdict(int)
        for p in patterns:
            key = tuple(p.pattern_tokens)
            pattern_counts[key] += 1

        # Normalize confidence by frequency
        total_patterns = len(patterns)
        for p in patterns:
            key = tuple(p.pattern_tokens)
            p.confidence = pattern_counts[key] / total_patterns

        return patterns

    def _detect_ngram_patterns(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        n: int,
        batch_idx: int,
        embeddings: Optional[torch.Tensor],
        allow_fuzzy: bool,
        max_edit_distance: int,
    ) -> List[InductionPattern]:
        """Detect n-gram induction patterns with various matching strategies."""
        patterns = []
        seq_len = len(tokens)

        # Build n-gram index
        ngram_positions = defaultdict(list)
        for i in range(seq_len - n + 1):
            if all(mask[i:i+n]):
                ngram = tuple(tokens[i:i+n].tolist())
                ngram_positions[ngram].append(i)

        # Find repeated n-grams
        for ngram, positions in ngram_positions.items():
            if len(positions) > 1:
                for i, pos1 in enumerate(positions[:-1]):
                    for pos2 in positions[i+1:]:
                        # Check if we can predict the next token
                        if pos1 + n < seq_len and pos2 + n < seq_len:
                            if mask[pos1 + n] and mask[pos2 + n]:
                                target = tokens[pos1 + n].item()
                                actual = tokens[pos2 + n].item()
                                patterns.append(InductionPattern(
                                    batch_idx=batch_idx,
                                    query_pos=pos2 + n,
                                    key_pos=pos1,
                                    pattern_length=n,
                                    pattern_tokens=list(ngram),
                                    target_token=target,
                                    actual_token=actual,
                                    is_correct=(target == actual),
                                    match_type='exact',
                                    confidence=1.0
                                ))

        # Add fuzzy matching if enabled
        if allow_fuzzy and n == 1:  # Fuzzy matching for single tokens
            patterns.extend(
                self._detect_fuzzy_patterns(
                    tokens, mask, batch_idx, embeddings, max_edit_distance
                )
            )

        return patterns

    def _detect_fuzzy_patterns(
        self,
        tokens: torch.Tensor,
        mask: torch.Tensor,
        batch_idx: int,
        embeddings: Optional[torch.Tensor],
        max_edit_distance: int,
    ) -> List[InductionPattern]:
        """Detect patterns with approximate matching."""
        patterns = []

        if embeddings is None:
            return patterns

        seq_len = len(tokens)

        for i in range(2, seq_len):
            if not (mask[i] and mask[i-1]):
                continue

            # Find similar tokens to tokens[i-1] in earlier positions
            query_emb = embeddings[i-1]

            for j in range(i-1):
                if not mask[j] or j + 1 >= seq_len or not mask[j+1]:
                    continue

                # Compute embedding similarity
                key_emb = embeddings[j]
                similarity = torch.cosine_similarity(
                    query_emb.unsqueeze(0),
                    key_emb.unsqueeze(0)
                ).item()

                if similarity >= self.pattern_match_threshold:
                    target = tokens[j + 1].item()
                    actual = tokens[i].item()
                    patterns.append(InductionPattern(
                        batch_idx=batch_idx,
                        query_pos=i,
                        key_pos=j,
                        pattern_length=1,
                        pattern_tokens=[tokens[j].item()],
                        target_token=target,
                        actual_token=actual,
                        is_correct=(target == actual),
                        match_type='fuzzy',
                        confidence=similarity
                    ))

        return patterns

    def analyze_qk_circuits(
        self,
        model,
        hidden_states: List[torch.Tensor],
        attention_weights: List[torch.Tensor],
        patterns: List[InductionPattern],
    ) -> Dict[Tuple[int, int], List[float]]:
        """
        Analyze QK circuits (attention formation) separately.

        Returns:
            Dictionary mapping (layer, head) to list of QK scores for pattern edges
        """
        qk_scores = defaultdict(list)

        for pattern in patterns:
            for layer_idx, attn in enumerate(attention_weights):
                batch_idx = pattern.batch_idx
                query_pos = pattern.query_pos
                key_pos = pattern.key_pos

                n_heads = attn.shape[1]
                for head_idx in range(n_heads):
                    # Get attention weight for this edge
                    attn_weight = attn[batch_idx, head_idx, query_pos, key_pos].item()

                    if attn_weight >= self.attention_threshold:
                        # Compute attention entropy for context
                        attn_dist = attn[batch_idx, head_idx, query_pos, :]
                        entropy = -torch.sum(
                            attn_dist * torch.log(attn_dist + 1e-10)
                        ).item()

                        # Store QK score (attention weight normalized by entropy)
                        qk_score = attn_weight / (entropy + 1e-10)
                        qk_scores[(layer_idx, head_idx)].append(qk_score)

        return qk_scores

    def analyze_ov_circuits(
        self,
        model,
        hidden_states: List[torch.Tensor],
        attention_weights: List[torch.Tensor],
        patterns: List[InductionPattern],
        logits: torch.Tensor,
    ) -> Dict[Tuple[int, int], List[float]]:
        """
        Analyze OV circuits (value transformation) separately.

        Returns:
            Dictionary mapping (layer, head) to list of OV contributions
        """
        ov_contributions = defaultdict(list)

        # Get vocabulary size
        vocab_size = logits.shape[-1]

        for pattern in patterns:
            batch_idx = pattern.batch_idx
            query_pos = pattern.query_pos
            key_pos = pattern.key_pos
            target_token = pattern.target_token

            # Skip if target token is out of bounds
            if target_token >= vocab_size:
                continue

            # Get predicted distribution at query position
            pred_logits = logits[batch_idx, query_pos]
            pred_probs = torch.softmax(pred_logits, dim=-1)

            # Compute KL divergence from predicted to ideal (one-hot at target)
            ideal_probs = torch.zeros_like(pred_probs)
            ideal_probs[target_token] = 1.0

            kl_div = torch.sum(
                ideal_probs * torch.log((ideal_probs + 1e-10) / (pred_probs + 1e-10))
            ).item()

            # Contribution is inverse of KL divergence
            ov_contribution = 1.0 / (1.0 + kl_div)

            # Store for each head that attended to this edge
            for layer_idx, attn in enumerate(attention_weights):
                n_heads = attn.shape[1]
                for head_idx in range(n_heads):
                    attn_weight = attn[batch_idx, head_idx, query_pos, key_pos].item()
                    if attn_weight >= self.attention_threshold:
                        # Weight OV contribution by attention strength
                        weighted_ov = ov_contribution * attn_weight
                        ov_contributions[(layer_idx, head_idx)].append(weighted_ov)

        return ov_contributions

    def compute_head_statistics(
        self,
        qk_scores: Dict[Tuple[int, int], List[float]],
        ov_contributions: Dict[Tuple[int, int], List[float]],
        patterns: List[InductionPattern],
    ) -> List[HeadStatistics]:
        """
        Compute comprehensive statistics for each head with bootstrapping.
        """
        head_stats = []

        # Get all heads that have data
        all_heads = set(qk_scores.keys()) | set(ov_contributions.keys())

        for (layer, head) in all_heads:
            qk_data = np.array(qk_scores.get((layer, head), []))
            ov_data = np.array(ov_contributions.get((layer, head), []))

            # Skip if insufficient data
            if len(qk_data) < 3 or len(ov_data) < 3:
                continue

            # Compute basic statistics
            stats = HeadStatistics(
                layer=layer,
                head=head,
                n_patterns=len(qk_data),
                pattern_detection_rate=len(qk_data) / max(1, len(patterns)),
                copying_accuracy=np.mean([p.is_correct for p in patterns]),
                qk_mean=np.mean(qk_data),
                qk_std=np.std(qk_data),
                qk_ci=(0, 0),  # Will be filled by bootstrap
                ov_mean=np.mean(ov_data),
                ov_std=np.std(ov_data),
                ov_ci=(0, 0),  # Will be filled by bootstrap
                d_prime=0,  # Will be computed
                beta=0,  # Will be computed
            )

            # Bootstrap confidence intervals
            if len(qk_data) >= self.min_samples:
                qk_ci = self._bootstrap_ci(qk_data)
                stats.qk_ci = qk_ci

            if len(ov_data) >= self.min_samples:
                ov_ci = self._bootstrap_ci(ov_data)
                stats.ov_ci = ov_ci

            # Compute correlation if sufficient paired data
            if len(qk_data) == len(ov_data) and len(qk_data) >= self.min_samples:
                corr, p_val = pearsonr(qk_data, ov_data)
                stats.qk_ov_correlation = corr
                stats.correlation_p_value = p_val

                # Bootstrap correlation CI
                corr_ci = self._bootstrap_correlation_ci(qk_data, ov_data)
                stats.correlation_ci = corr_ci

            # Compute signal detection metrics
            stats.d_prime, stats.beta = self._compute_signal_detection(
                qk_data, ov_data, patterns
            )

            head_stats.append(stats)

        return head_stats

    def _bootstrap_ci(
        self,
        data: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Compute bootstrap confidence interval for mean."""

        def statistic(x):
            return np.mean(x)

        res = bootstrap(
            (data,),
            statistic,
            n_resamples=self.bootstrap_n_samples,
            confidence_level=confidence_level,
            method='percentile'
        )

        return (res.confidence_interval.low, res.confidence_interval.high)

    def _bootstrap_correlation_ci(
        self,
        x: np.ndarray,
        y: np.ndarray,
        confidence_level: float = 0.95
    ) -> Tuple[float, float]:
        """Bootstrap confidence interval for correlation with Fisher z-transform."""

        def correlation_statistic(x, y):
            if len(x) < 3:
                return 0.0
            corr, _ = pearsonr(x, y)
            # Fisher z-transformation for stability
            z = np.arctanh(corr)
            return z

        correlations = []
        for _ in range(self.bootstrap_n_samples):
            indices = np.random.choice(len(x), len(x), replace=True)
            x_sample = x[indices]
            y_sample = y[indices]
            z = correlation_statistic(x_sample, y_sample)
            correlations.append(z)

        correlations = np.array(correlations)
        ci_low = np.percentile(correlations, (1 - confidence_level) * 50)
        ci_high = np.percentile(correlations, (1 + confidence_level) * 50)

        # Transform back from Fisher z
        return (np.tanh(ci_low), np.tanh(ci_high))

    def _compute_signal_detection(
        self,
        qk_data: np.ndarray,
        ov_data: np.ndarray,
        patterns: List[InductionPattern],
    ) -> Tuple[float, float]:
        """
        Compute signal detection theory metrics (d' and beta).
        """
        if len(patterns) == 0:
            return 0.0, 0.0

        # Separate hits (correct predictions) from false alarms
        hits = [p for p in patterns if p.is_correct]
        false_alarms = [p for p in patterns if not p.is_correct]

        hit_rate = len(hits) / max(1, len(patterns))
        fa_rate = len(false_alarms) / max(1, len(patterns))

        # Avoid extreme values
        hit_rate = np.clip(hit_rate, 0.01, 0.99)
        fa_rate = np.clip(fa_rate, 0.01, 0.99)

        # Compute d' (discriminability)
        from scipy.stats import norm
        z_hit = norm.ppf(hit_rate)
        z_fa = norm.ppf(fa_rate)
        d_prime = z_hit - z_fa

        # Compute beta (response bias)
        beta = np.exp(-0.5 * (z_hit**2 - z_fa**2))

        return float(d_prime), float(beta)

    def apply_multiple_comparison_correction(
        self,
        head_stats: List[HeadStatistics],
    ) -> List[HeadStatistics]:
        """
        Apply FDR correction for multiple comparisons across heads.
        """
        if not head_stats:
            return head_stats

        # Extract p-values for heads with correlations
        p_values = []
        head_indices = []

        for i, stats in enumerate(head_stats):
            if stats.correlation_p_value is not None:
                p_values.append(stats.correlation_p_value)
                head_indices.append(i)

        if not p_values:
            return head_stats

        # Apply Benjamini-Hochberg FDR correction
        rejected, corrected_p, _, _ = multipletests(
            p_values,
            alpha=self.fdr_alpha,
            method='fdr_bh'
        )

        # Update head statistics with corrected p-values
        for idx, head_idx in enumerate(head_indices):
            head_stats[head_idx].correlation_p_value = corrected_p[idx]

        return head_stats

    def compute_null_distribution(
        self,
        model,
        batch: Dict[str, torch.Tensor],
        n_permutations: int = 100,
    ) -> Dict[str, np.ndarray]:
        """
        Compute null distribution by shuffling inputs.
        """
        null_qk_scores = []
        null_ov_scores = []

        for _ in range(n_permutations):
            # Shuffle input tokens within each sequence
            shuffled_batch = batch.copy()
            input_ids = shuffled_batch['input_ids'].clone()

            for i in range(input_ids.shape[0]):
                perm = torch.randperm(input_ids.shape[1])
                input_ids[i] = input_ids[i][perm]

            shuffled_batch['input_ids'] = input_ids

            # Run analysis on shuffled data
            with torch.no_grad():
                outputs = model(**shuffled_batch, output_attentions=True, output_hidden_states=True)

            # Simplified scoring for null distribution
            if outputs.attentions:
                for attn in outputs.attentions:
                    null_qk_scores.append(attn.mean().item())

            if hasattr(outputs, 'logits'):
                null_ov_scores.append(outputs.logits.mean().item())

        return {
            'null_qk': np.array(null_qk_scores),
            'null_ov': np.array(null_ov_scores)
        }

    def compute_power_analysis(
        self,
        head_stats: List[HeadStatistics],
        effect_size_threshold: float = 0.5,
        power_threshold: float = 0.8,
    ) -> Dict[str, Any]:
        """
        Compute statistical power and required sample sizes.
        """
        try:
            from statsmodels.stats.power import tt_ind_solve_power
        except ImportError:
            # Return empty if statsmodels not available
            return {'error': 'statsmodels not available for power analysis'}

        results = {}

        for stats in head_stats:
            if stats.qk_ov_correlation is not None:
                # Convert correlation to Cohen's d
                r = stats.qk_ov_correlation
                d = 2 * r / np.sqrt(1 - r**2)

                # Compute power for current sample size
                try:
                    power = tt_ind_solve_power(
                        effect_size=d,
                        nobs1=stats.n_patterns,
                        alpha=self.fdr_alpha,
                        alternative='two-sided'
                    )

                    # Compute required sample size for desired power
                    required_n = tt_ind_solve_power(
                        effect_size=effect_size_threshold,
                        power=power_threshold,
                        alpha=self.fdr_alpha,
                        alternative='two-sided'
                    )

                    results[f'L{stats.layer}H{stats.head}'] = {
                        'current_power': power,
                        'required_n': int(required_n),
                        'current_n': stats.n_patterns,
                        'effect_size': d
                    }
                except:
                    pass

        return results

    def analyze_complete(
        self,
        model,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, Any]:
        """
        Complete statistically rigorous analysis of QK-OV pairing.
        """
        # Move to model device
        device = next(model.parameters()).device
        batch = {k: v.to(device) for k, v in batch.items()}

        # Get model outputs
        with torch.no_grad():
            outputs = model(**batch, output_attentions=True, output_hidden_states=True)

        if not outputs.attentions or not outputs.hidden_states:
            return {'error': 'Model does not return attention/hidden states'}

        # Detect patterns
        patterns = self.detect_patterns_enhanced(
            batch['input_ids'],
            batch.get('attention_mask', torch.ones_like(batch['input_ids'])),
            embeddings=outputs.hidden_states[0]  # Use input embeddings
        )

        if len(patterns) < self.min_samples:
            return {
                'error': f'Insufficient patterns detected: {len(patterns)} < {self.min_samples}',
                'n_patterns': len(patterns)
            }

        # Separate QK and OV analysis
        qk_scores = self.analyze_qk_circuits(
            model, outputs.hidden_states, outputs.attentions, patterns
        )

        ov_contributions = self.analyze_ov_circuits(
            model, outputs.hidden_states, outputs.attentions, patterns, outputs.logits
        )

        # Compute head statistics with bootstrapping
        head_stats = self.compute_head_statistics(
            qk_scores, ov_contributions, patterns
        )

        # Apply multiple comparison correction
        head_stats = self.apply_multiple_comparison_correction(head_stats)

        # Compute null distribution
        null_dist = self.compute_null_distribution(model, batch)

        # Power analysis
        power_analysis = self.compute_power_analysis(head_stats)

        # Filter significant heads
        significant_heads = [
            stats for stats in head_stats
            if stats.correlation_p_value is not None
            and stats.correlation_p_value < self.fdr_alpha
        ]

        return {
            'n_patterns_detected': len(patterns),
            'n_heads_analyzed': len(head_stats),
            'n_significant_heads': len(significant_heads),
            'head_statistics': head_stats,
            'significant_heads': significant_heads,
            'null_distribution': null_dist,
            'power_analysis': power_analysis,
            'pattern_summary': {
                'exact_matches': sum(1 for p in patterns if p.match_type == 'exact'),
                'fuzzy_matches': sum(1 for p in patterns if p.match_type == 'fuzzy'),
                'copying_accuracy': np.mean([p.is_correct for p in patterns]),
                'pattern_confidence_mean': np.mean([p.confidence for p in patterns]),
            },
            'statistical_summary': {
                'fdr_alpha': self.fdr_alpha,
                'min_samples_required': self.min_samples,
                'bootstrap_n_samples': self.bootstrap_n_samples,
                'attention_threshold': self.attention_threshold,
            }
        }


# Integration with existing code
def compute_qk_ov_pairing_improved(
    model,
    batch: Dict[str, torch.Tensor],
    min_samples: int = 30,
    attention_threshold: float = 0.1,
) -> Dict[str, Any]:
    """
    Drop-in replacement for the original compute_qk_ov_pairing with proper statistics.
    """
    analyzer = ImprovedQKOVAnalyzer(
        min_samples_for_correlation=min_samples,
        attention_threshold=attention_threshold,
    )

    return analyzer.analyze_complete(model, batch)