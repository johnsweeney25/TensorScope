#!/usr/bin/env python3
"""
Fiber Bundle Hypothesis Test for Token Embeddings

Based on "Token embeddings violate the manifold hypothesis" (Robinson et al.)

This module implements statistical tests to detect violations of the manifold
hypothesis in token embedding spaces, identifying irregularities that could
lead to unstable model behavior.

Key innovations:
1. Tests for smooth fiber bundle structure (generalization of manifolds)
2. Identifies problematic tokens that cause instability
3. Distinguishes between small and large radius regimes
4. Provides actionable insights for prompt engineering
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
import warnings
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

# Import our manifold analysis tools
# Import from the fixed version with correct function names
from .tractable_manifold_curvature_fixed import (
    compute_ricci_curvature_debiased as compute_ricci_curvature_tractable,
    compute_intrinsic_dimension_fixed as compute_intrinsic_dimension_twonn,
    compute_manifold_metrics_fixed
)

# Define a simple sectional curvature function as fallback
def compute_sectional_curvature_tractable(embeddings, k=10):
    """Simple sectional curvature estimation."""
    return 0.0  # Placeholder - would need proper implementation


@dataclass
class FiberBundleTestResult:
    """Results of fiber bundle hypothesis test."""
    token_id: int
    token_str: str
    p_value: float
    reject_null: bool
    irregularity_score: float
    neighborhood_radius: float
    local_dimension: float
    curvature_variance: float
    stability_risk: str  # 'low', 'medium', 'high'
    regime: str  # 'small_radius', 'large_radius', 'boundary'


class FiberBundleHypothesisTest:
    """
    Statistical test for fiber bundle structure in token embeddings.

    The null hypothesis assumes a smooth fiber bundle structure, which is
    a generalization of manifolds that allows for more complex topology.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_neighbors_small: int = 10,
        n_neighbors_large: int = 50,
        n_bootstrap: int = 100
    ):
        """
        Args:
            alpha: Significance level for hypothesis testing
            n_neighbors_small: Neighbors for small radius regime
            n_neighbors_large: Neighbors for large radius regime
            n_bootstrap: Number of bootstrap samples for p-value estimation
        """
        self.alpha = alpha
        self.n_neighbors_small = n_neighbors_small
        self.n_neighbors_large = n_neighbors_large
        self.n_bootstrap = n_bootstrap

    def test_token_neighborhood(
        self,
        embeddings: torch.Tensor,
        token_idx: int,
        tokenizer=None
    ) -> FiberBundleTestResult:
        """
        Test fiber bundle hypothesis for a specific token's neighborhood.

        Args:
            embeddings: (vocab_size, embedding_dim) token embeddings
            token_idx: Index of token to test
            tokenizer: Optional tokenizer for token string

        Returns:
            FiberBundleTestResult with test statistics
        """
        vocab_size, embed_dim = embeddings.shape

        # Get token string if tokenizer provided
        token_str = ""
        if tokenizer is not None:
            try:
                token_str = tokenizer.decode([token_idx])
            except:
                token_str = f"token_{token_idx}"

        # Extract neighborhoods at different scales
        small_neighbors, small_dists = self._get_neighbors(
            embeddings, token_idx, self.n_neighbors_small
        )
        large_neighbors, large_dists = self._get_neighbors(
            embeddings, token_idx, self.n_neighbors_large
        )

        # Test 1: Local dimension consistency
        dim_score = self._test_dimension_consistency(
            embeddings, token_idx, small_neighbors, large_neighbors
        )

        # Test 2: Curvature regularity
        curv_score = self._test_curvature_regularity(
            embeddings, token_idx, small_neighbors
        )

        # Test 3: Tangent space alignment (fiber bundle property)
        tangent_score = self._test_tangent_alignment(
            embeddings, token_idx, small_neighbors
        )

        # Test 4: Transition smoothness between regimes
        transition_score = self._test_regime_transition(
            small_dists, large_dists
        )

        # Combine test statistics
        test_statistic = self._combine_test_statistics(
            dim_score, curv_score, tangent_score, transition_score
        )

        # Bootstrap for p-value
        p_value = self._bootstrap_p_value(
            embeddings, token_idx, test_statistic
        )

        # Determine regime
        regime = self._identify_regime(small_dists, large_dists, embed_dim)

        # Calculate stability risk
        stability_risk = self._assess_stability_risk(
            p_value, test_statistic, regime
        )

        return FiberBundleTestResult(
            token_id=token_idx,
            token_str=token_str,
            p_value=p_value,
            reject_null=p_value < self.alpha,
            irregularity_score=test_statistic,
            neighborhood_radius=float(small_dists.mean()),
            local_dimension=float(self._estimate_local_dimension(
                embeddings[small_neighbors]
            )),
            curvature_variance=curv_score,
            stability_risk=stability_risk,
            regime=regime
        )

    def _get_neighbors(
        self,
        embeddings: torch.Tensor,
        token_idx: int,
        n_neighbors: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get k nearest neighbors and distances."""
        nbrs = NearestNeighbors(n_neighbors=n_neighbors+1)
        nbrs.fit(embeddings.cpu().numpy())

        dists, indices = nbrs.kneighbors(
            embeddings[token_idx:token_idx+1].cpu().numpy()
        )

        # Exclude self
        return indices[0, 1:], dists[0, 1:]

    def _test_dimension_consistency(
        self,
        embeddings: torch.Tensor,
        token_idx: int,
        small_neighbors: np.ndarray,
        large_neighbors: np.ndarray
    ) -> float:
        """
        Test if intrinsic dimension is consistent across scales.
        Returns irregularity score.
        """
        # Estimate dimension at small scale
        small_points = embeddings[small_neighbors]
        dim_small = compute_intrinsic_dimension_twonn(small_points)

        # Estimate dimension at large scale
        large_points = embeddings[large_neighbors]
        dim_large = compute_intrinsic_dimension_twonn(large_points)

        # Irregularity: dimension should be similar across scales
        irregularity = abs(dim_small - dim_large) / max(dim_small, dim_large)

        return float(irregularity)

    def _test_curvature_regularity(
        self,
        embeddings: torch.Tensor,
        token_idx: int,
        neighbors: np.ndarray
    ) -> float:
        """
        Test if curvature is regular in neighborhood.
        Returns curvature variance.
        """
        # Include center token
        local_points = torch.cat([
            embeddings[token_idx:token_idx+1],
            embeddings[neighbors]
        ])

        # Compute curvature statistics
        mean_ricci, std_ricci = compute_ricci_curvature_tractable(
            local_points,
            k_neighbors=min(5, len(neighbors)-1),
            n_samples=10
        )

        # High variance indicates irregularity
        return float(std_ricci)

    def _test_tangent_alignment(
        self,
        embeddings: torch.Tensor,
        token_idx: int,
        neighbors: np.ndarray
    ) -> float:
        """
        Test alignment of local tangent spaces (fiber bundle property).
        Returns misalignment score.
        """
        # Get local neighborhood
        local_points = embeddings[neighbors].cpu().numpy()
        center_point = embeddings[token_idx].cpu().numpy()

        # Estimate tangent space at center using PCA
        centered = local_points - center_point
        pca_center = PCA(n_components=min(5, len(centered)-1))
        pca_center.fit(centered)

        # For each neighbor, estimate its tangent space
        misalignments = []
        for i, neighbor_idx in enumerate(neighbors[:5]):  # Test first 5
            # Get neighbors of this neighbor
            sub_nbrs = NearestNeighbors(n_neighbors=min(6, len(embeddings)))
            sub_nbrs.fit(embeddings.cpu().numpy())
            _, sub_indices = sub_nbrs.kneighbors(
                embeddings[neighbor_idx:neighbor_idx+1].cpu().numpy()
            )

            if len(sub_indices[0]) < 3:
                continue

            sub_points = embeddings[sub_indices[0, 1:6]].cpu().numpy()
            sub_centered = sub_points - embeddings[neighbor_idx].cpu().numpy()

            pca_neighbor = PCA(n_components=min(3, len(sub_centered)-1))
            pca_neighbor.fit(sub_centered)

            # Measure subspace angle between tangent spaces
            # Using principal angles via SVD
            U1 = pca_center.components_[:3].T
            U2 = pca_neighbor.components_[:min(3, pca_neighbor.n_components_)].T

            # Grassmann distance
            _, s, _ = np.linalg.svd(U1.T @ U2)
            s = np.clip(s, -1, 1)
            angles = np.arccos(s)
            misalignment = np.mean(angles)
            misalignments.append(misalignment)

        if misalignments:
            return float(np.mean(misalignments))
        return 0.0

    def _test_regime_transition(
        self,
        small_dists: np.ndarray,
        large_dists: np.ndarray
    ) -> float:
        """
        Test smoothness of transition between small and large radius regimes.
        Returns transition irregularity score.
        """
        # Analyze distance distribution scaling
        small_mean = np.mean(small_dists)
        large_mean = np.mean(large_dists)

        # Expected scaling for smooth manifold
        expected_ratio = np.sqrt(len(large_dists) / len(small_dists))
        actual_ratio = large_mean / small_mean

        # Deviation from expected scaling
        irregularity = abs(np.log(actual_ratio / expected_ratio))

        return float(irregularity)

    def _combine_test_statistics(
        self,
        dim_score: float,
        curv_score: float,
        tangent_score: float,
        transition_score: float
    ) -> float:
        """Combine individual test scores into overall statistic."""
        # Weight different aspects
        weights = [0.3, 0.3, 0.25, 0.15]  # Dimension and curvature most important
        scores = [dim_score, curv_score, tangent_score, transition_score]

        # Normalize scores
        scores = [s / (s + 1) for s in scores]  # Map to [0, 1]

        # Weighted combination
        combined = sum(w * s for w, s in zip(weights, scores))

        return float(combined)

    def _bootstrap_p_value(
        self,
        embeddings: torch.Tensor,
        token_idx: int,
        observed_stat: float
    ) -> float:
        """
        Estimate p-value using bootstrap under null hypothesis.
        """
        bootstrap_stats = []

        for _ in range(self.n_bootstrap):
            # Sample random token as null
            random_idx = np.random.randint(0, len(embeddings))

            # Skip if same as test token
            if random_idx == token_idx:
                continue

            # Get neighbors
            neighbors, _ = self._get_neighbors(
                embeddings, random_idx, self.n_neighbors_small
            )

            # Quick irregularity estimate
            local_points = embeddings[neighbors]
            dim = compute_intrinsic_dimension_twonn(local_points)
            _, std_curv = compute_ricci_curvature_tractable(
                local_points, n_samples=5
            )

            # Simple statistic
            stat = (dim / embeddings.shape[1]) + std_curv
            bootstrap_stats.append(stat)

        if bootstrap_stats:
            # P-value: proportion of bootstrap samples >= observed
            p_value = np.mean([s >= observed_stat for s in bootstrap_stats])
            return float(p_value)

        return 0.5  # Uninformative if bootstrap fails

    def _identify_regime(
        self,
        small_dists: np.ndarray,
        large_dists: np.ndarray,
        embed_dim: int
    ) -> str:
        """Identify spatial regime of token."""
        small_radius = np.mean(small_dists)
        large_radius = np.mean(large_dists)

        # Heuristic thresholds based on embedding dimension
        small_threshold = np.sqrt(embed_dim) / 10
        large_threshold = np.sqrt(embed_dim)

        if small_radius < small_threshold:
            return "small_radius"
        elif large_radius > large_threshold:
            return "large_radius"
        else:
            return "boundary"

    def _assess_stability_risk(
        self,
        p_value: float,
        test_statistic: float,
        regime: str
    ) -> str:
        """Assess risk of instability in model responses."""
        # Boundary regime is inherently less stable
        if regime == "boundary":
            base_risk = 0.3
        else:
            base_risk = 0.0

        # Add risk from test results
        if p_value < 0.01:
            stat_risk = 0.6
        elif p_value < 0.05:
            stat_risk = 0.4
        elif p_value < 0.10:
            stat_risk = 0.2
        else:
            stat_risk = 0.0

        total_risk = base_risk + stat_risk

        if total_risk >= 0.6:
            return "high"
        elif total_risk >= 0.3:
            return "medium"
        else:
            return "low"

    def _estimate_local_dimension(self, points: torch.Tensor) -> float:
        """Quick local dimension estimate."""
        if len(points) < 3:
            return float(points.shape[1])

        try:
            return compute_intrinsic_dimension_twonn(points)
        except:
            return float(points.shape[1])


class TokenStabilityAnalyzer:
    """
    Analyze token stability implications for prompt engineering.
    """

    def __init__(self, model: nn.Module, tokenizer):
        """
        Args:
            model: Language model with token embeddings
            tokenizer: Tokenizer for the model
        """
        self.model = model
        self.tokenizer = tokenizer
        self.fiber_test = FiberBundleHypothesisTest()

        # Cache test results
        self.problematic_tokens: Set[int] = set()
        self.test_results: Dict[int, FiberBundleTestResult] = {}

    def analyze_vocabulary(
        self,
        sample_size: Optional[int] = None,
        focus_on_common: bool = True
    ) -> Dict[str, any]:
        """
        Analyze entire vocabulary or sample for manifold violations.

        Args:
            sample_size: Number of tokens to test (None = all)
            focus_on_common: Prioritize common tokens

        Returns:
            Analysis results and statistics
        """
        # Get token embeddings
        embeddings = self._get_embeddings()
        vocab_size = len(embeddings)

        # Select tokens to test
        if sample_size and sample_size < vocab_size:
            if focus_on_common:
                # Test most common tokens (usually lower indices)
                test_indices = list(range(min(sample_size, vocab_size)))
            else:
                # Random sample
                test_indices = np.random.choice(
                    vocab_size, sample_size, replace=False
                )
        else:
            test_indices = list(range(vocab_size))

        # Run tests
        results = []
        for idx in test_indices:
            result = self.fiber_test.test_token_neighborhood(
                embeddings, idx, self.tokenizer
            )
            results.append(result)

            # Cache problematic tokens
            if result.reject_null:
                self.problematic_tokens.add(idx)
                self.test_results[idx] = result

        # Compute statistics
        rejection_rate = np.mean([r.reject_null for r in results])
        high_risk_tokens = [r for r in results if r.stability_risk == "high"]

        # Group by regime
        regime_stats = {
            "small_radius": [],
            "large_radius": [],
            "boundary": []
        }
        for r in results:
            regime_stats[r.regime].append(r)

        return {
            "total_tested": len(results),
            "rejection_rate": float(rejection_rate),
            "high_risk_count": len(high_risk_tokens),
            "high_risk_tokens": high_risk_tokens[:10],  # Top 10
            "regime_distribution": {
                k: len(v) / len(results) for k, v in regime_stats.items()
            },
            "mean_irregularity": float(np.mean([r.irregularity_score for r in results])),
            "problematic_token_ids": list(self.problematic_tokens)[:100]  # First 100
        }

    def check_prompt_stability(self, prompt: str) -> Dict[str, any]:
        """
        Check if a prompt contains problematic tokens.

        Args:
            prompt: Text prompt to analyze

        Returns:
            Stability assessment and recommendations
        """
        # Tokenize prompt
        tokens = self.tokenizer.encode(prompt)

        # Check for problematic tokens
        issues = []
        for token_id in tokens:
            if token_id in self.problematic_tokens:
                result = self.test_results.get(token_id)
                if result:
                    issues.append({
                        "token": self.tokenizer.decode([token_id]),
                        "position": tokens.index(token_id),
                        "risk": result.stability_risk,
                        "irregularity": result.irregularity_score
                    })

        # Overall stability score
        if issues:
            max_risk = max(i["risk"] for i in issues)
            stability_score = {"high": 0.3, "medium": 0.6, "low": 0.8}[max_risk]
        else:
            stability_score = 1.0

        return {
            "prompt": prompt,
            "stability_score": stability_score,
            "problematic_tokens": issues,
            "recommendation": self._get_recommendation(issues)
        }

    def suggest_alternative_prompt(
        self,
        prompt: str,
        max_alternatives: int = 3
    ) -> List[str]:
        """
        Suggest alternative prompts with better stability.

        Args:
            prompt: Original prompt
            max_alternatives: Maximum alternatives to generate

        Returns:
            List of alternative prompts
        """
        tokens = self.tokenizer.encode(prompt)
        alternatives = []

        # Find problematic tokens
        problem_positions = [
            i for i, t in enumerate(tokens)
            if t in self.problematic_tokens
        ]

        if not problem_positions:
            return [prompt]  # No issues

        # Try synonyms or rephrasing
        # This is a simplified approach - in practice would use
        # more sophisticated NLP techniques
        for pos in problem_positions[:max_alternatives]:
            # Create alternative by masking and regenerating
            masked_tokens = tokens.copy()
            masked_tokens[pos] = self.tokenizer.mask_token_id

            # Decode and clean
            masked_text = self.tokenizer.decode(masked_tokens)
            alternatives.append(masked_text.replace("[MASK]", "[REPHRASE]"))

        return alternatives

    def _get_embeddings(self) -> torch.Tensor:
        """Extract token embeddings from model."""
        if hasattr(self.model, 'embeddings'):
            if hasattr(self.model.embeddings, 'word_embeddings'):
                return self.model.embeddings.word_embeddings.weight.data
            elif hasattr(self.model.embeddings, 'token_embeddings'):
                return self.model.embeddings.token_embeddings.weight.data
        elif hasattr(self.model, 'transformer'):
            if hasattr(self.model.transformer, 'wte'):
                return self.model.transformer.wte.weight.data

        raise ValueError("Could not find token embeddings in model")

    def _get_recommendation(self, issues: List[Dict]) -> str:
        """Generate recommendation based on issues found."""
        if not issues:
            return "Prompt appears stable - no problematic tokens detected."

        high_risk = [i for i in issues if i["risk"] == "high"]

        if high_risk:
            return (
                f"CAUTION: {len(high_risk)} high-risk token(s) detected. "
                "Consider rephrasing to avoid potential instability in model responses."
            )
        elif len(issues) > 3:
            return (
                "Multiple problematic tokens detected. "
                "Prompt may exhibit some response variability."
            )
        else:
            return (
                "Minor stability concerns detected. "
                "Prompt should work but may have slight variations in responses."
            )


def run_fiber_bundle_analysis(
    model: nn.Module,
    tokenizer,
    test_size: int = 100
) -> Dict[str, any]:
    """
    Run complete fiber bundle analysis on a model's token embeddings.

    Args:
        model: Language model
        tokenizer: Model's tokenizer
        test_size: Number of tokens to test

    Returns:
        Complete analysis results
    """
    print("=" * 60)
    print("FIBER BUNDLE HYPOTHESIS ANALYSIS")
    print("Testing for manifold violations in token embeddings")
    print("=" * 60)

    analyzer = TokenStabilityAnalyzer(model, tokenizer)

    # Analyze vocabulary
    print(f"\nTesting {test_size} tokens...")
    results = analyzer.analyze_vocabulary(sample_size=test_size)

    print(f"\nResults:")
    print(f"  Rejection rate: {results['rejection_rate']:.2%}")
    print(f"  High-risk tokens: {results['high_risk_count']}")
    print(f"  Mean irregularity: {results['mean_irregularity']:.4f}")

    print(f"\nRegime distribution:")
    for regime, pct in results['regime_distribution'].items():
        print(f"  {regime}: {pct:.2%}")

    # Test example prompts
    test_prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a poem about nature."
    ]

    print("\nPrompt stability analysis:")
    for prompt in test_prompts:
        stability = analyzer.check_prompt_stability(prompt)
        print(f"\nPrompt: '{prompt[:50]}...'")
        print(f"  Stability score: {stability['stability_score']:.2f}")
        print(f"  Issues: {len(stability['problematic_tokens'])}")

    return {
        "vocabulary_analysis": results,
        "analyzer": analyzer
    }


if __name__ == "__main__":
    print("Fiber Bundle Hypothesis Test Module")
    print("Ready for integration with language models")
    print("\nThis module provides:")
    print("1. Statistical test for manifold violations")
    print("2. Token stability analysis")
    print("3. Prompt engineering recommendations")
    print("4. Alternative prompt suggestions")