#!/usr/bin/env python3
"""
Token Stability Analyzer for Language Models.

This module applies fiber bundle hypothesis testing to LLM token embeddings
to identify problematic tokens that may cause unstable model behavior.

Based on: "Token embeddings violate the manifold hypothesis" (Robinson et al.)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Set, Optional, Tuple, Any
from dataclasses import dataclass
import warnings

# Import core fiber bundle test
# from .fiber_bundle_core import FiberBundleTest, FiberBundleTestResult, batch_test_fiber_bundle  # Not implemented
from .fiber_bundle_hypothesis_test import FiberBundleHypothesisTest as FiberBundleTest, FiberBundleTestResult

def batch_test_fiber_bundle(*args, **kwargs):
    """Placeholder for batch testing."""
    return []


@dataclass
class TokenAnalysisResult:
    """Results from analyzing a token's stability."""
    token_id: int
    token_str: str
    fiber_bundle_result: FiberBundleTestResult
    stability_risk: str  # 'low', 'medium', 'high'
    semantic_neighbors: List[Tuple[str, float]]  # (token, distance)
    recommendation: str


@dataclass
class PromptStabilityResult:
    """Results from analyzing prompt stability."""
    prompt: str
    token_ids: List[int]
    token_strings: List[str]
    overall_stability: float  # 0-1, higher is more stable
    problematic_tokens: List[Dict[str, Any]]
    risk_level: str  # 'low', 'medium', 'high'
    confidence: float
    suggestions: List[str]


class TokenStabilityAnalyzer:
    """
    Analyze token stability in language models using fiber bundle hypothesis.

    This class bridges the mathematical fiber bundle test with practical
    LLM token analysis.
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: Any,
        cache_results: bool = True,
        alpha: float = 0.05
    ):
        """
        Initialize token stability analyzer.

        Args:
            model: Language model with accessible embeddings
            tokenizer: Tokenizer compatible with the model
            cache_results: Whether to cache test results
            alpha: Significance level for hypothesis testing
        """
        self.model = model
        self.tokenizer = tokenizer
        self.cache_results = cache_results
        self.alpha = alpha

        # Extract embeddings
        self.embeddings = self._extract_embeddings()
        self.vocab_size, self.embed_dim = self.embeddings.shape

        # Initialize fiber bundle tester
        self.fiber_test = FiberBundleTest(
            alpha=alpha,
            n_neighbors_small=min(10, self.vocab_size // 100),
            n_neighbors_large=min(50, self.vocab_size // 20)
        )

        # Cache for results
        self.cached_results: Dict[int, TokenAnalysisResult] = {}
        self.problematic_tokens: Set[int] = set()

    def analyze_token(
        self,
        token_id: int,
        include_neighbors: bool = True
    ) -> TokenAnalysisResult:
        """
        Analyze stability of a specific token.

        Args:
            token_id: Token ID to analyze
            include_neighbors: Whether to include semantic neighbors

        Returns:
            TokenAnalysisResult with stability assessment
        """
        # Check cache
        if self.cache_results and token_id in self.cached_results:
            return self.cached_results[token_id]

        # Get token string
        try:
            token_str = self.tokenizer.decode([token_id])
        except:
            token_str = f"<token_{token_id}>"

        # Run fiber bundle test
        embeddings_np = self.embeddings.detach().cpu().numpy()
        fb_result = self.fiber_test.test_point(embeddings_np, token_id)

        # Assess stability risk
        stability_risk = self._assess_stability_risk(fb_result)

        # Get semantic neighbors if requested
        semantic_neighbors = []
        if include_neighbors:
            semantic_neighbors = self._get_semantic_neighbors(token_id, k=5)

        # Generate recommendation
        recommendation = self._generate_recommendation(fb_result, stability_risk)

        # Create result
        result = TokenAnalysisResult(
            token_id=token_id,
            token_str=token_str,
            fiber_bundle_result=fb_result,
            stability_risk=stability_risk,
            semantic_neighbors=semantic_neighbors,
            recommendation=recommendation
        )

        # Cache if enabled
        if self.cache_results:
            self.cached_results[token_id] = result
            if fb_result.reject_null:
                self.problematic_tokens.add(token_id)

        return result

    def analyze_vocabulary(
        self,
        sample_size: Optional[int] = None,
        prioritize_common: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze entire vocabulary or sample.

        Args:
            sample_size: Number of tokens to analyze (None = all)
            prioritize_common: Whether to prioritize common tokens

        Returns:
            Summary statistics and problematic tokens
        """
        # Select tokens to analyze
        if sample_size and sample_size < self.vocab_size:
            if prioritize_common:
                # Common tokens typically have lower IDs
                test_indices = list(range(min(sample_size, self.vocab_size)))
            else:
                # Random sample
                test_indices = np.random.choice(
                    self.vocab_size, sample_size, replace=False
                ).tolist()
        else:
            test_indices = list(range(self.vocab_size))

        # Run batch analysis
        embeddings_np = self.embeddings.detach().cpu().numpy()
        fb_results = batch_test_fiber_bundle(
            embeddings_np,
            test_indices,
            alpha=self.alpha,
            verbose=False
        )

        # Process results
        high_risk = []
        medium_risk = []
        regime_counts = {"small_radius": 0, "large_radius": 0, "boundary": 0}

        for token_id, fb_result in fb_results.items():
            # Assess risk
            risk = self._assess_stability_risk(fb_result)

            # Get token string
            try:
                token_str = self.tokenizer.decode([token_id])
            except:
                token_str = f"<token_{token_id}>"

            # Track high risk tokens
            if risk == "high":
                high_risk.append({
                    "token_id": token_id,
                    "token": token_str,
                    "p_value": fb_result.p_value,
                    "irregularity": fb_result.irregularity_score
                })
            elif risk == "medium":
                medium_risk.append({
                    "token_id": token_id,
                    "token": token_str,
                    "p_value": fb_result.p_value
                })

            # Count regimes
            regime_counts[fb_result.regime] += 1

            # Update cache
            if fb_result.reject_null:
                self.problematic_tokens.add(token_id)

        # Calculate statistics
        n_tested = len(fb_results)
        n_rejected = sum(r.reject_null for r in fb_results.values())
        rejection_rate = n_rejected / n_tested if n_tested > 0 else 0

        return {
            "n_tested": n_tested,
            "n_rejected": n_rejected,
            "rejection_rate": rejection_rate,
            "high_risk_tokens": high_risk[:20],  # Top 20
            "medium_risk_tokens": medium_risk[:20],
            "n_high_risk": len(high_risk),
            "n_medium_risk": len(medium_risk),
            "regime_distribution": {
                k: v/n_tested for k, v in regime_counts.items()
            },
            "mean_irregularity": np.mean([
                r.irregularity_score for r in fb_results.values()
            ])
        }

    def analyze_prompt(
        self,
        prompt: str,
        return_alternatives: bool = False
    ) -> PromptStabilityResult:
        """
        Analyze stability of a text prompt.

        Args:
            prompt: Text prompt to analyze
            return_alternatives: Whether to generate alternative prompts

        Returns:
            PromptStabilityResult with assessment
        """
        # Tokenize prompt
        token_ids = self.tokenizer.encode(prompt)
        token_strings = [self.tokenizer.decode([tid]) for tid in token_ids]

        # Analyze each token
        problematic = []
        stability_scores = []

        for i, token_id in enumerate(token_ids):
            # Skip special tokens
            if token_id >= self.vocab_size:
                continue

            # Get or compute analysis
            result = self.analyze_token(token_id, include_neighbors=False)

            # Track stability
            if result.fiber_bundle_result.reject_null:
                stability_scores.append(1.0 - result.fiber_bundle_result.p_value)
            else:
                stability_scores.append(1.0)

            # Track problematic tokens
            if result.stability_risk in ["high", "medium"]:
                problematic.append({
                    "position": i,
                    "token_id": token_id,
                    "token": token_strings[i],
                    "risk": result.stability_risk,
                    "p_value": result.fiber_bundle_result.p_value,
                    "regime": result.fiber_bundle_result.regime
                })

        # Calculate overall stability
        if stability_scores:
            overall_stability = np.mean(stability_scores)
        else:
            overall_stability = 1.0

        # Determine risk level
        if any(p["risk"] == "high" for p in problematic):
            risk_level = "high"
        elif len(problematic) > len(token_ids) * 0.3:
            risk_level = "medium"
        elif problematic:
            risk_level = "low"
        else:
            risk_level = "low"

        # Calculate confidence
        confidence = self._calculate_prompt_confidence(
            len(token_ids), len(problematic), overall_stability
        )

        # Generate suggestions
        suggestions = []
        if return_alternatives and problematic:
            suggestions = self._generate_alternative_suggestions(
                prompt, problematic, token_ids, token_strings
            )

        return PromptStabilityResult(
            prompt=prompt,
            token_ids=token_ids,
            token_strings=token_strings,
            overall_stability=overall_stability,
            problematic_tokens=problematic,
            risk_level=risk_level,
            confidence=confidence,
            suggestions=suggestions
        )

    def find_stable_alternatives(
        self,
        token_id: int,
        k: int = 5
    ) -> List[Tuple[int, str, float]]:
        """
        Find stable alternative tokens semantically similar to given token.

        Args:
            token_id: Token to find alternatives for
            k: Number of alternatives to return

        Returns:
            List of (token_id, token_str, similarity) tuples
        """
        # Get embedding of target token
        target_embedding = self.embeddings[token_id]

        # Compute similarities to all tokens
        similarities = torch.cosine_similarity(
            self.embeddings,
            target_embedding.unsqueeze(0),
            dim=1
        )

        # Sort by similarity
        sorted_indices = torch.argsort(similarities, descending=True)

        # Find stable alternatives
        alternatives = []
        for idx in sorted_indices[1:].tolist():  # Skip self
            # Check if stable (not in problematic set)
            if idx not in self.problematic_tokens:
                try:
                    token_str = self.tokenizer.decode([idx])
                except:
                    token_str = f"<token_{idx}>"

                similarity = similarities[idx].item()
                alternatives.append((idx, token_str, similarity))

                if len(alternatives) >= k:
                    break

        return alternatives

    def _extract_embeddings(self) -> torch.Tensor:
        """Extract token embeddings from model."""
        # Try different common embedding locations
        if hasattr(self.model, 'embeddings'):
            if hasattr(self.model.embeddings, 'word_embeddings'):
                return self.model.embeddings.word_embeddings.weight.data
            elif hasattr(self.model.embeddings, 'token_embeddings'):
                return self.model.embeddings.token_embeddings.weight.data

        if hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'wte'):
                return self.model.transformer.wte.weight.data
            elif hasattr(self.model.transformer, 'word_embeddings'):
                return self.model.transformer.word_embeddings.weight.data

        if hasattr(self.model, 'embed_tokens'):
            return self.model.embed_tokens.weight.data

        raise ValueError("Could not locate token embeddings in model")

    def _assess_stability_risk(self, fb_result: FiberBundleTestResult) -> str:
        """Assess stability risk from fiber bundle test result."""
        # Strong rejection = high risk
        if fb_result.p_value < 0.001:
            return "high"
        elif fb_result.p_value < 0.01:
            return "high" if fb_result.regime == "boundary" else "medium"
        elif fb_result.reject_null:
            return "medium"

        # High irregularity even without rejection
        if fb_result.irregularity_score > 0.7:
            return "medium"

        return "low"

    def _get_semantic_neighbors(
        self,
        token_id: int,
        k: int = 5
    ) -> List[Tuple[str, float]]:
        """Get semantically similar tokens."""
        # Compute cosine similarities
        target = self.embeddings[token_id]
        similarities = torch.cosine_similarity(
            self.embeddings,
            target.unsqueeze(0),
            dim=1
        )

        # Get top k (excluding self)
        top_k = torch.topk(similarities, k + 1)

        neighbors = []
        for idx, sim in zip(top_k.indices[1:], top_k.values[1:]):
            try:
                token_str = self.tokenizer.decode([idx.item()])
            except:
                token_str = f"<token_{idx.item()}>"
            neighbors.append((token_str, sim.item()))

        return neighbors

    def _generate_recommendation(
        self,
        fb_result: FiberBundleTestResult,
        risk: str
    ) -> str:
        """Generate recommendation based on analysis."""
        if risk == "high":
            if fb_result.regime == "boundary":
                return (
                    "HIGH RISK: Token in boundary regime with significant irregularity. "
                    "Expect unstable model behavior. Consider using alternative tokens."
                )
            else:
                return (
                    "HIGH RISK: Token violates fiber bundle structure. "
                    "Model responses may vary significantly even with identical context."
                )
        elif risk == "medium":
            return (
                "MEDIUM RISK: Token shows some geometric irregularity. "
                "May experience occasional response variations."
            )
        else:
            return "LOW RISK: Token appears geometrically stable."

    def _calculate_prompt_confidence(
        self,
        n_tokens: int,
        n_problematic: int,
        stability: float
    ) -> float:
        """Calculate confidence in prompt stability assessment."""
        # Base confidence on stability score
        conf = stability

        # Adjust for sample size
        if n_tokens < 5:
            conf *= 0.8  # Low confidence for very short prompts
        elif n_tokens > 50:
            conf *= 1.1  # Higher confidence for longer prompts

        # Adjust for problematic ratio
        problem_ratio = n_problematic / max(n_tokens, 1)
        if problem_ratio > 0.5:
            conf *= 0.7

        return float(np.clip(conf, 0.1, 1.0))

    def _generate_alternative_suggestions(
        self,
        prompt: str,
        problematic: List[Dict],
        token_ids: List[int],
        token_strings: List[str]
    ) -> List[str]:
        """Generate alternative prompt suggestions."""
        suggestions = []

        # Sort problematic tokens by risk
        high_risk = [p for p in problematic if p["risk"] == "high"]

        for problem in high_risk[:3]:  # Up to 3 suggestions
            position = problem["position"]
            token_id = problem["token_id"]

            # Find stable alternatives
            alternatives = self.find_stable_alternatives(token_id, k=3)

            if alternatives:
                # Create suggestion with best alternative
                alt_token_id, alt_token_str, _ = alternatives[0]

                # Reconstruct prompt with substitution
                new_tokens = token_strings.copy()
                new_tokens[position] = alt_token_str

                suggestion = self.tokenizer.decode(
                    token_ids[:position] + [alt_token_id] + token_ids[position+1:]
                )
                suggestions.append(suggestion)

        return suggestions


def create_token_risk_database(
    model: nn.Module,
    tokenizer: Any,
    output_path: str,
    sample_size: Optional[int] = None
) -> Dict[str, Any]:
    """
    Create a database of token stability risks.

    Args:
        model: Language model
        tokenizer: Model tokenizer
        output_path: Path to save results
        sample_size: Number of tokens to analyze

    Returns:
        Summary of analysis
    """
    print("Creating token risk database...")
    print("=" * 50)

    # Initialize analyzer
    analyzer = TokenStabilityAnalyzer(model, tokenizer)

    # Analyze vocabulary
    print(f"Analyzing {sample_size or 'all'} tokens...")
    results = analyzer.analyze_vocabulary(sample_size=sample_size)

    print(f"\nResults:")
    print(f"  Tokens tested: {results['n_tested']}")
    print(f"  Rejection rate: {results['rejection_rate']:.2%}")
    print(f"  High risk tokens: {results['n_high_risk']}")
    print(f"  Medium risk tokens: {results['n_medium_risk']}")

    # Save results
    import json
    with open(output_path, 'w') as f:
        json.dump({
            "summary": results,
            "problematic_token_ids": list(analyzer.problematic_tokens),
            "high_risk_tokens": results["high_risk_tokens"]
        }, f, indent=2)

    print(f"\nRisk database saved to: {output_path}")

    return results


if __name__ == "__main__":
    print("Token Stability Analyzer")
    print("=" * 50)
    print("\nThis module provides:")
    print("1. Token-level stability analysis")
    print("2. Prompt stability assessment")
    print("3. Alternative token suggestions")
    print("4. Risk database creation")
    print("\nReady for integration with language models.")