#!/usr/bin/env python3
"""
Prompt Robustness Analyzer

Comprehensive prompt analysis combining all Robinson paper insights to
assess and improve prompt stability across LLMs.

This is the practical application of all our singularity detection methods.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import warnings

# Import all our analysis modules
from .singularity_mapper import SingularityMapper, SingularityProfile
from .robinson_fiber_bundle_test import RobinsonFiberBundleTest
from .polysemy_detector import PolysemyDetector


@dataclass
class TokenRisk:
    """Risk assessment for a single token in prompt."""
    token_idx: int
    token_str: str
    position: int  # Position in prompt

    # Risk factors
    singularity_type: str
    polysemy_risk: float
    geometric_risk: float
    volume_growth_risk: float
    local_signal_dimension: float

    # Overall assessment
    risk_score: float  # 0-1, higher = riskier
    risk_category: str  # 'safe', 'monitor', 'caution', 'avoid'

    # Alternatives
    suggested_alternatives: List[Tuple[str, float]]  # (token, similarity)
    replacement_impact: float  # Expected improvement if replaced


@dataclass
class PromptRobustnessReport:
    """Complete robustness analysis for a prompt."""
    prompt: str
    tokens: List[str]

    # Overall metrics
    overall_robustness: float  # 0-1, higher = more robust
    overall_risk_level: str  # 'low', 'medium', 'high', 'critical'
    confidence: float

    # Detailed risks
    token_risks: List[TokenRisk]
    high_risk_tokens: List[TokenRisk]

    # Stability predictions
    expected_output_variance: float  # 0-1
    cross_model_consistency: float  # 0-1
    semantic_stability: float  # 0-1

    # Improvement suggestions
    suggested_rewrites: List[str]
    improvement_potential: float  # How much can be improved
    critical_positions: List[int]  # Token positions needing attention

    # Detailed analysis
    polysemous_count: int
    singularity_count: int
    avg_local_dimension: float

    # Recommendations
    primary_recommendation: str
    detailed_suggestions: List[str]


class PromptRobustnessAnalyzer:
    """
    Analyze and improve prompt robustness using all Robinson paper insights.
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        tokenizer=None,
        use_all_methods: bool = True,
        risk_thresholds: Optional[Dict[str, float]] = None
    ):
        """
        Initialize analyzer with embeddings and configuration.

        Args:
            embeddings: Token embedding matrix
            tokenizer: Tokenizer for encoding/decoding
            use_all_methods: Use all detection methods
            risk_thresholds: Custom risk thresholds
        """
        self.embeddings = embeddings
        self.tokenizer = tokenizer
        self.use_all_methods = use_all_methods

        # Initialize analyzers
        self.singularity_mapper = SingularityMapper()
        self.robinson_test = RobinsonFiberBundleTest()
        self.polysemy_detector = PolysemyDetector()

        # Risk thresholds
        self.risk_thresholds = risk_thresholds or {
            'safe': 0.2,
            'monitor': 0.4,
            'caution': 0.7,
            'avoid': 0.9
        }

        # Cache for already analyzed tokens
        self.token_cache = {}

    def analyze_prompt(
        self,
        prompt: str,
        detailed: bool = True,
        suggest_alternatives: bool = True
    ) -> PromptRobustnessReport:
        """
        Perform comprehensive robustness analysis on a prompt.

        Args:
            prompt: Input prompt text
            detailed: Include detailed per-token analysis
            suggest_alternatives: Generate alternative suggestions

        Returns:
            PromptRobustnessReport with complete analysis
        """
        # Tokenize prompt
        if self.tokenizer:
            token_ids = self.tokenizer.encode(prompt)
            tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        else:
            # Fallback to simple splitting
            tokens = prompt.split()
            token_ids = [hash(t) % len(self.embeddings) for t in tokens]

        # Analyze each token
        token_risks = []
        high_risk_tokens = []

        for position, (token_id, token_str) in enumerate(zip(token_ids, tokens)):
            risk = self._analyze_token_risk(
                token_id, token_str, position, suggest_alternatives
            )
            token_risks.append(risk)

            if risk.risk_score > self.risk_thresholds['caution']:
                high_risk_tokens.append(risk)

        # Compute overall metrics
        overall_robustness = self._compute_overall_robustness(token_risks)
        overall_risk_level = self._determine_overall_risk(overall_robustness)

        # Predict stability
        output_variance = self._predict_output_variance(token_risks)
        cross_model_consistency = self._predict_cross_model_consistency(token_risks)
        semantic_stability = self._predict_semantic_stability(token_risks)

        # Generate improvement suggestions
        suggested_rewrites = []
        if suggest_alternatives:
            suggested_rewrites = self._generate_rewrite_suggestions(
                prompt, tokens, token_risks
            )

        # Compute statistics
        polysemous_count = sum(1 for r in token_risks if r.polysemy_risk > 0.5)
        singularity_count = sum(1 for r in token_risks if r.singularity_type != 'none')
        avg_local_dim = np.mean([r.local_signal_dimension for r in token_risks])

        # Identify critical positions
        critical_positions = [
            r.position for r in token_risks
            if r.risk_score > self.risk_thresholds['caution']
        ]

        # Generate recommendations
        primary_rec = self._generate_primary_recommendation(
            overall_risk_level, high_risk_tokens
        )

        detailed_suggestions = self._generate_detailed_suggestions(
            token_risks, overall_robustness
        )

        # Calculate improvement potential
        improvement_potential = self._calculate_improvement_potential(token_risks)

        # Confidence in analysis
        confidence = self._calculate_confidence(len(tokens), token_risks)

        return PromptRobustnessReport(
            prompt=prompt,
            tokens=tokens,
            overall_robustness=overall_robustness,
            overall_risk_level=overall_risk_level,
            confidence=confidence,
            token_risks=token_risks if detailed else [],
            high_risk_tokens=high_risk_tokens,
            expected_output_variance=output_variance,
            cross_model_consistency=cross_model_consistency,
            semantic_stability=semantic_stability,
            suggested_rewrites=suggested_rewrites,
            improvement_potential=improvement_potential,
            critical_positions=critical_positions,
            polysemous_count=polysemous_count,
            singularity_count=singularity_count,
            avg_local_dimension=avg_local_dim,
            primary_recommendation=primary_rec,
            detailed_suggestions=detailed_suggestions
        )

    def _analyze_token_risk(
        self,
        token_id: int,
        token_str: str,
        position: int,
        suggest_alternatives: bool
    ) -> TokenRisk:
        """
        Analyze risk for a single token.
        """
        # Check cache
        if token_id in self.token_cache:
            cached = self.token_cache[token_id]
            cached.position = position
            return cached

        # Get singularity profile
        profile = self.singularity_mapper.map_singularity(
            self.embeddings, token_id, token_str, self.tokenizer
        )

        # Compute risk components
        polysemy_risk = profile.polysemy_confidence
        geometric_risk = profile.geometric_irregularity
        volume_growth_risk = float(profile.volume_growth_violation)

        # Overall risk score
        risk_score = (
            0.3 * polysemy_risk +
            0.3 * geometric_risk +
            0.2 * volume_growth_risk +
            0.2 * min(1.0, profile.local_signal_dimension / 20)
        )

        # Determine risk category
        if risk_score < self.risk_thresholds['safe']:
            risk_category = 'safe'
        elif risk_score < self.risk_thresholds['monitor']:
            risk_category = 'monitor'
        elif risk_score < self.risk_thresholds['caution']:
            risk_category = 'caution'
        else:
            risk_category = 'avoid'

        # Get alternatives
        alternatives = []
        replacement_impact = 0.0

        if suggest_alternatives and risk_category in ['caution', 'avoid']:
            alternatives = self._find_safer_alternatives(
                token_id, profile
            )
            replacement_impact = self._estimate_replacement_impact(
                risk_score, alternatives
            )

        token_risk = TokenRisk(
            token_idx=token_id,
            token_str=token_str,
            position=position,
            singularity_type=profile.singularity_type,
            polysemy_risk=polysemy_risk,
            geometric_risk=geometric_risk,
            volume_growth_risk=volume_growth_risk,
            local_signal_dimension=profile.local_signal_dimension,
            risk_score=risk_score,
            risk_category=risk_category,
            suggested_alternatives=alternatives,
            replacement_impact=replacement_impact
        )

        # Cache result
        self.token_cache[token_id] = token_risk

        return token_risk

    def _compute_overall_robustness(self, token_risks: List[TokenRisk]) -> float:
        """
        Compute overall prompt robustness.
        """
        if not token_risks:
            return 1.0

        # Weight by position (earlier tokens matter more)
        weighted_risks = []
        for risk in token_risks:
            position_weight = 1.0 / (1.0 + 0.1 * risk.position)
            weighted_risks.append(risk.risk_score * position_weight)

        # Robustness is inverse of average weighted risk
        avg_risk = np.mean(weighted_risks)
        robustness = 1.0 - avg_risk

        # Penalize for high-risk tokens
        high_risk_penalty = sum(1 for r in token_risks if r.risk_category == 'avoid') * 0.1
        robustness = max(0.0, robustness - high_risk_penalty)

        return float(robustness)

    def _determine_overall_risk(self, robustness: float) -> str:
        """
        Determine overall risk level.
        """
        if robustness > 0.8:
            return 'low'
        elif robustness > 0.6:
            return 'medium'
        elif robustness > 0.3:
            return 'high'
        else:
            return 'critical'

    def _predict_output_variance(self, token_risks: List[TokenRisk]) -> float:
        """
        Predict expected output variance.
        """
        if not token_risks:
            return 0.0

        # Based on local signal dimensions
        dimensions = [r.local_signal_dimension for r in token_risks]
        avg_dim = np.mean(dimensions)

        # Higher dimension = more variance
        variance = min(1.0, avg_dim / 20)

        # Amplify for polysemous tokens
        polysemy_factor = sum(r.polysemy_risk for r in token_risks) / len(token_risks)
        variance = min(1.0, variance + 0.3 * polysemy_factor)

        return float(variance)

    def _predict_cross_model_consistency(self, token_risks: List[TokenRisk]) -> float:
        """
        Predict cross-model consistency.
        """
        if not token_risks:
            return 1.0

        # Geometric risks indicate model-specific structure
        geometric_risks = [r.geometric_risk for r in token_risks]
        avg_geometric = np.mean(geometric_risks)

        # Consistency is inverse of geometric risk
        consistency = 1.0 - avg_geometric

        # Penalize for singularities
        singularity_penalty = sum(
            1 for r in token_risks if r.singularity_type != 'none'
        ) / len(token_risks)

        consistency = max(0.0, consistency - 0.3 * singularity_penalty)

        return float(consistency)

    def _predict_semantic_stability(self, token_risks: List[TokenRisk]) -> float:
        """
        Predict semantic stability.
        """
        if not token_risks:
            return 1.0

        # Polysemy directly affects semantic stability
        polysemy_risks = [r.polysemy_risk for r in token_risks]
        avg_polysemy = np.mean(polysemy_risks)

        # Volume growth risks also matter
        volume_risks = [r.volume_growth_risk for r in token_risks]
        avg_volume = np.mean(volume_risks)

        stability = 1.0 - (0.6 * avg_polysemy + 0.4 * avg_volume)

        return float(max(0.0, stability))

    def _find_safer_alternatives(
        self,
        token_id: int,
        profile: SingularityProfile
    ) -> List[Tuple[str, float]]:
        """
        Find safer alternative tokens.
        """
        alternatives = []

        # Use profile's alternatives if available
        for alt_id, alt_str, similarity in profile.alternative_tokens[:3]:
            # Would need to test alternative's risk
            # For now, just return them
            alternatives.append((alt_str, similarity))

        return alternatives

    def _estimate_replacement_impact(
        self,
        current_risk: float,
        alternatives: List[Tuple[str, float]]
    ) -> float:
        """
        Estimate improvement from replacement.
        """
        if not alternatives:
            return 0.0

        # Assume best alternative reduces risk by similarity factor
        best_similarity = max(sim for _, sim in alternatives) if alternatives else 0
        impact = current_risk * best_similarity * 0.7  # Conservative estimate

        return float(impact)

    def _generate_rewrite_suggestions(
        self,
        prompt: str,
        tokens: List[str],
        token_risks: List[TokenRisk]
    ) -> List[str]:
        """
        Generate suggested prompt rewrites.
        """
        suggestions = []

        # Find highest risk token
        if token_risks:
            highest_risk = max(token_risks, key=lambda r: r.risk_score)

            if highest_risk.suggested_alternatives:
                # Suggest replacing highest risk token
                alt_token, _ = highest_risk.suggested_alternatives[0]
                new_tokens = tokens.copy()
                new_tokens[highest_risk.position] = alt_token
                suggestion = ' '.join(new_tokens)
                suggestions.append(suggestion)

        # Could add more sophisticated rewriting strategies

        return suggestions[:3]  # Return top 3 suggestions

    def _generate_primary_recommendation(
        self,
        risk_level: str,
        high_risk_tokens: List[TokenRisk]
    ) -> str:
        """
        Generate primary recommendation.
        """
        if risk_level == 'low':
            return "Prompt is robust and stable. Safe to use across models."
        elif risk_level == 'medium':
            return "Prompt has moderate stability. Test thoroughly before production use."
        elif risk_level == 'high':
            if high_risk_tokens:
                problematic = ', '.join(t.token_str for t in high_risk_tokens[:3])
                return f"High risk detected. Consider replacing: {problematic}"
            return "High risk detected. Significant rewriting recommended."
        else:  # critical
            return "CRITICAL: Prompt is highly unstable. Major revision required."

    def _generate_detailed_suggestions(
        self,
        token_risks: List[TokenRisk],
        overall_robustness: float
    ) -> List[str]:
        """
        Generate detailed improvement suggestions.
        """
        suggestions = []

        # Check for polysemy issues
        polysemous = [r for r in token_risks if r.polysemy_risk > 0.5]
        if polysemous:
            suggestions.append(
                f"Replace polysemous tokens: {', '.join(t.token_str for t in polysemous[:3])}"
            )

        # Check for high local dimensions
        high_dim = [r for r in token_risks if r.local_signal_dimension > 15]
        if high_dim:
            suggestions.append(
                "Tokens with high variance detected. Use more specific terms."
            )

        # Check for singularities
        singularities = [r for r in token_risks if r.singularity_type != 'none']
        if len(singularities) > len(token_risks) * 0.3:
            suggestions.append(
                "Many irregular tokens found. Consider complete rephrasing."
            )

        # Position-based suggestion
        early_risks = [r for r in token_risks[:5] if r.risk_category in ['caution', 'avoid']]
        if early_risks:
            suggestions.append(
                "High-risk tokens at prompt start. Revise opening for stability."
            )

        return suggestions

    def _calculate_improvement_potential(self, token_risks: List[TokenRisk]) -> float:
        """
        Calculate how much the prompt can be improved.
        """
        if not token_risks:
            return 0.0

        # Sum replacement impacts for high-risk tokens
        potential = sum(
            r.replacement_impact for r in token_risks
            if r.risk_category in ['caution', 'avoid']
        )

        # Normalize by prompt length
        potential = min(1.0, potential / len(token_risks))

        return float(potential)

    def _calculate_confidence(
        self,
        n_tokens: int,
        token_risks: List[TokenRisk]
    ) -> float:
        """
        Calculate confidence in analysis.
        """
        # Base confidence on prompt length
        if n_tokens < 5:
            base_confidence = 0.7
        elif n_tokens < 20:
            base_confidence = 0.85
        else:
            base_confidence = 0.95

        # Adjust for analysis completeness
        # (In reality, would check if all analyses succeeded)

        return float(base_confidence)


def analyze_prompt_robustness(
    prompt: str,
    embeddings: np.ndarray,
    tokenizer=None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Convenience function for prompt robustness analysis.

    Args:
        prompt: Prompt text to analyze
        embeddings: Token embeddings
        tokenizer: Optional tokenizer
        verbose: Print results

    Returns:
        Dictionary with analysis results
    """
    analyzer = PromptRobustnessAnalyzer(embeddings, tokenizer)
    report = analyzer.analyze_prompt(prompt)

    if verbose:
        print(f"\nPrompt: {prompt}")
        print("=" * 50)
        print(f"Overall Robustness: {report.overall_robustness:.2%}")
        print(f"Risk Level: {report.overall_risk_level}")
        print(f"Expected Output Variance: {report.expected_output_variance:.2%}")
        print(f"Cross-Model Consistency: {report.cross_model_consistency:.2%}")

        if report.high_risk_tokens:
            print(f"\nHigh-Risk Tokens:")
            for risk in report.high_risk_tokens[:3]:
                print(f"  '{risk.token_str}' - {risk.singularity_type} ({risk.risk_score:.2f})")

        print(f"\nRecommendation: {report.primary_recommendation}")

        if report.suggested_rewrites:
            print(f"\nSuggested Rewrite:")
            print(f"  {report.suggested_rewrites[0]}")

    return {
        'report': report,
        'robustness': report.overall_robustness,
        'risk_level': report.overall_risk_level,
        'recommendations': report.detailed_suggestions
    }


if __name__ == "__main__":
    # Example usage
    print("Prompt Robustness Analyzer - Example")
    print("=" * 50)

    # Generate synthetic embeddings
    np.random.seed(42)
    vocab_size = 10000
    embed_dim = 768
    embeddings = np.random.randn(vocab_size, embed_dim)

    # Test prompts
    test_prompts = [
        "Explain quantum computing in simple terms",
        "What is the affect of climate change on polar bears?",  # 'affect' is polysemous
        "Calculate the sum of 123 and 456",  # Numeric tokens
        "The quick brown fox jumps over the lazy dog",  # Common phrase
        "##fragment tokens can cause instability##"  # Fragments
    ]

    analyzer = PromptRobustnessAnalyzer(embeddings)

    for prompt in test_prompts:
        print(f"\n{'='*60}")
        result = analyze_prompt_robustness(
            prompt,
            embeddings,
            verbose=True
        )

        # Show improvement potential
        report = result['report']
        if report.improvement_potential > 0.1:
            print(f"\nImprovement Potential: {report.improvement_potential:.1%}")