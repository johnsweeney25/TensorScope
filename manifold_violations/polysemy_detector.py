#!/usr/bin/env python3
"""
Clustering-Based Polysemy Detection (NOT Robinson et al. Method)
================================================================

CRITICAL DISTINCTION - THIS IS NOT THE ROBINSON METHOD:
-------------------------------------------------------
Robinson et al. (2025) Method (see robinson_fiber_bundle_test.py):
  - Statistical hypothesis testing on volume-radius scaling curves
  - Tests H0: "token neighborhood is a smooth manifold"
  - Outputs: p-values, slope changes, statistical violations

This Module (Our Original Contribution):
  - Clustering analysis of k-nearest neighbor embeddings
  - Tests: "do neighbors form distinct semantic clusters?"
  - Outputs: cluster counts, separation scores, polysemy classification

Why This Exists:
Robinson observed that polysemous tokens create geometric "singularities."
We built a practical detector using clustering, which is computationally
simpler and more interpretable than manifold hypothesis testing.

For actual Robinson test: use robinson_fiber_bundle_test.py
For practical polysemy detection: use this module
For research validation: use BOTH and compare results
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from scipy.spatial.distance import pdist
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score, pairwise_distances
import warnings


@dataclass
class PolysemyResult:
    """Result of polysemy detection for a token."""
    token_idx: int
    token_str: str
    is_polysemous: bool
    confidence: float
    num_meanings: int
    meaning_clusters: List[List[int]]  # Indices of tokens in each meaning cluster
    coherence_score: float  # How well-separated the meanings are
    example_contexts: List[str]  # Example usages for each meaning
    polysemy_type: str  # 'homonym', 'contranym', 'multi-sense', 'none'


@dataclass
class PolysemyAnalysis:
    """Overall polysemy analysis results."""
    total_tokens: int
    polysemous_tokens: List[PolysemyResult]
    polysemy_rate: float
    high_risk_tokens: List[int]  # Indices of highly polysemous tokens
    homonyms: List[Tuple[int, str]]  # (idx, token) pairs
    contranyms: List[Tuple[int, str]]  # Tokens with opposite meanings
    summary_stats: Dict[str, float]


class PolysemyDetector:
    """
    Detect polysemous tokens in embedding spaces.

    Based on Robinson et al.'s observation that polysemy creates
    singularities and irregular neighborhoods in token embeddings.
    """

    def __init__(
        self,
        n_neighbors: int = 50,
        clustering_method: str = 'dbscan',
        min_cluster_size: int = 3,
        coherence_threshold: float = 0.7,
        polysemy_confidence_threshold: float = 0.6,
        max_embeddings_for_exact: int = 10000,  # For computational efficiency
        subsample_size: int = 10000,  # Subsample size for approximate k-NN
        metric: str = 'cosine',  # Distance metric for all operations
        random_state: Optional[int] = None  # For reproducibility
    ):
        """
        Initialize polysemy detector.

        Args:
            n_neighbors: Number of neighbors to analyze
            clustering_method: 'dbscan' or 'hierarchical'
            min_cluster_size: Minimum size for a meaning cluster
            coherence_threshold: Threshold for cluster separation
            polysemy_confidence_threshold: Confidence threshold for polysemy
            max_embeddings_for_exact: Maximum embeddings before using bootstrap sampling
            bootstrap_samples: Size of bootstrap sample for approximate k-NN (statistically principled)
            confidence_level: Statistical confidence level for neighbor identification

        Statistical Note:
            When vocabulary size exceeds max_embeddings_for_exact, we use subsampling
            to approximate k-nearest neighbors. This is a standard approximation technique
            that trades exact neighbor identification for computational efficiency.
            For production use, consider using approximate nearest neighbor methods
            (e.g., FAISS, Annoy) for better recall guarantees.
        """
        self.n_neighbors = n_neighbors
        self.clustering_method = clustering_method
        self.min_cluster_size = min_cluster_size
        self.coherence_threshold = coherence_threshold
        self.polysemy_confidence_threshold = polysemy_confidence_threshold
        self.max_embeddings_for_exact = max_embeddings_for_exact
        self.subsample_size = subsample_size
        self.metric = metric
        self.rng = np.random.default_rng(random_state)

    def detect_polysemy(
        self,
        embeddings: np.ndarray,
        token_idx: int,
        token_str: str = "",
        tokenizer=None
    ) -> PolysemyResult:
        """
        Detect if a token is polysemous based on its embedding neighborhood.

        Args:
            embeddings: (n_tokens, embed_dim) embedding matrix
            token_idx: Index of token to analyze
            token_str: String representation of token
            tokenizer: Optional tokenizer for context generation

        Returns:
            PolysemyResult with detection details
        """
        n_tokens, embed_dim = embeddings.shape
        embeddings = embeddings.astype(np.float32)  # Ensure float32 for numerics

        # Normalize embeddings if using cosine metric
        if self.metric == 'cosine':
            X = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
            center = X[token_idx]
        else:
            X = embeddings
            center = embeddings[token_idx]

        # Get nearest neighbors with efficient sampling for large vocabularies
        if n_tokens <= self.max_embeddings_for_exact:
            # For small vocabularies, compute exact distances
            if self.metric == 'cosine':
                distances = 1.0 - X @ center  # Cosine distance
            else:
                distances = np.linalg.norm(X - center, axis=1)  # Euclidean

            neighbor_indices = np.argsort(distances)[1:self.n_neighbors+1]
            neighbor_distances = distances[neighbor_indices]
        else:
            # For large vocabularies, use subsampling for efficiency
            # Note: This is approximate and may miss true neighbors

            # Sample from indices excluding the token itself
            all_other_indices = np.setdiff1d(np.arange(n_tokens), [token_idx])
            sample_size = min(self.subsample_size, len(all_other_indices))

            candidate_indices = self.rng.choice(
                all_other_indices,
                size=sample_size,
                replace=False
            )

            # Compute distances only to sampled candidates
            if self.metric == 'cosine':
                candidate_distances = 1.0 - X[candidate_indices] @ center
            else:
                candidate_distances = np.linalg.norm(X[candidate_indices] - center, axis=1)

            # Get k-nearest from candidates
            nearest_in_sample = np.argsort(candidate_distances)[:self.n_neighbors]
            neighbor_indices = candidate_indices[nearest_in_sample]
            neighbor_distances = candidate_distances[nearest_in_sample]

        neighbor_embeddings = embeddings[neighbor_indices]

        # Detect clusters in neighborhood (potential different meanings)
        clusters, n_clusters, features_used = self._cluster_neighbors(neighbor_embeddings)

        # Compute coherence of clusters using same features as clustering
        coherence = self._compute_cluster_coherence(
            features_used, clusters, n_clusters
        )

        # Calculate adaptive min cluster size used in clustering
        adaptive_min_cluster_size = max(self.min_cluster_size, int(0.06 * self.n_neighbors))

        # Identify meaning clusters with same threshold as clustering
        meaning_clusters = self._extract_meaning_clusters(
            neighbor_indices, clusters, n_clusters, adaptive_min_cluster_size
        )

        # Determine polysemy type
        polysemy_type = self._classify_polysemy_type(
            meaning_clusters, coherence, n_clusters
        )

        # Calculate confidence
        confidence = self._compute_polysemy_confidence(
            n_clusters, coherence, neighbor_distances
        )

        # Determine if polysemous
        is_polysemous = (
            n_clusters > 1 and
            confidence > self.polysemy_confidence_threshold and
            coherence > self.coherence_threshold
        )

        # Generate example contexts if tokenizer available
        example_contexts = self._generate_example_contexts(
            meaning_clusters, tokenizer
        )

        return PolysemyResult(
            token_idx=token_idx,
            token_str=token_str,
            is_polysemous=is_polysemous,
            confidence=confidence,
            num_meanings=n_clusters,
            meaning_clusters=meaning_clusters,
            coherence_score=coherence,
            example_contexts=example_contexts,
            polysemy_type=polysemy_type
        )

    def _cluster_neighbors(
        self,
        neighbor_embeddings: np.ndarray
    ) -> Tuple[np.ndarray, int, np.ndarray]:
        """
        Cluster neighbor embeddings to find potential meanings.

        Returns:
            clusters: Cluster labels
            n_clusters: Number of clusters
            features_used: The actual features used for clustering (normalized if cosine)
        """
        if len(neighbor_embeddings) < self.min_cluster_size:
            return np.zeros(len(neighbor_embeddings)), 1, neighbor_embeddings

        # Check for NaN values and handle them
        if np.any(np.isnan(neighbor_embeddings)):
            # Remove rows with NaN
            valid_mask = ~np.any(np.isnan(neighbor_embeddings), axis=1)
            if np.sum(valid_mask) < self.min_cluster_size:
                return np.zeros(len(neighbor_embeddings)), 1, neighbor_embeddings
            neighbor_embeddings = neighbor_embeddings[valid_mask]

        # Prepare features - normalize once for cosine
        if self.metric == 'cosine':
            features_used = neighbor_embeddings / (np.linalg.norm(neighbor_embeddings, axis=1, keepdims=True) + 1e-12)
        else:
            features_used = neighbor_embeddings

        # Use relative min_cluster_size
        adaptive_min_cluster_size = max(self.min_cluster_size, int(0.06 * self.n_neighbors))

        if self.clustering_method == 'dbscan':
            # Use DBSCAN for density-based clustering
            # Adaptively estimate eps from k-dist curve

            # Compute pairwise distances
            D = pairwise_distances(features_used, metric=self.metric)

            # Fix: Exclude diagonal (self-distances) for proper k-NN
            np.fill_diagonal(D, np.inf)

            k = max(adaptive_min_cluster_size, 5)

            # Estimate eps from k-nearest distances (excluding self)
            if len(D) > k:
                # Use partition for efficiency, get (k-1)th distance
                knn_dists = np.partition(D, kth=k-1, axis=1)[:, k-1]
                eps = float(np.median(knn_dists))  # Use median for robustness
                # Guard against eps=0 when all points identical
                if eps == 0.0:
                    eps = 0.01  # Small positive value
            else:
                eps = 0.5  # Fallback for small neighborhoods

            clustering = DBSCAN(
                eps=eps,
                min_samples=adaptive_min_cluster_size,
                metric=self.metric
            )
            clusters = clustering.fit_predict(features_used)

            # Count actual clusters (excluding noise)
            unique_clusters = np.unique(clusters)
            n_clusters = len(unique_clusters[unique_clusters != -1])

            # Don't relabel noise - it affects coherence computation
            # Keep noise as -1 for proper handling

        else:  # hierarchical
            # Use hierarchical clustering with appropriate metric
            if self.metric == 'cosine':
                # For cosine, use average linkage (Ward requires Euclidean)
                # Use normalized features for consistency
                condensed_dists = pdist(features_used, metric='cosine')
                linkage_matrix = linkage(condensed_dists, method='average')

                # Smart threshold selection: aim for natural cluster separation
                # Use elbow method: find largest gap in sorted distances
                sorted_dists = np.sort(condensed_dists)
                # Look for large jumps in distance
                gaps = np.diff(sorted_dists)
                # Find percentile of largest gap
                if len(gaps) > 0:
                    largest_gap_idx = np.argmax(gaps)
                    # Use distance just before the largest gap
                    threshold = sorted_dists[largest_gap_idx] + gaps[largest_gap_idx] * 0.5
                    # But constrain to reasonable range (30th to 70th percentile)
                    threshold = np.clip(threshold,
                                       np.percentile(condensed_dists, 30),
                                       np.percentile(condensed_dists, 70))
                else:
                    # Fallback to 40th percentile (more conservative than before)
                    threshold = np.percentile(condensed_dists, 40)
            else:
                # For Euclidean, can use Ward
                linkage_matrix = linkage(features_used, method='ward')
                # Use adaptive threshold based on data variance
                data_std = np.std(pairwise_distances(features_used))
                threshold = 1.0 * data_std  # Lower threshold for more sensitivity

            clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')
            n_clusters = len(np.unique(clusters))

        return clusters, n_clusters, features_used

    def _compute_cluster_coherence(
        self,
        embeddings: np.ndarray,
        clusters: np.ndarray,
        n_clusters: int
    ) -> float:
        """
        Compute how well-separated the clusters are.
        """
        if n_clusters <= 1:
            return 0.0

        # Filter out noise and singleton clusters for silhouette
        if self.clustering_method == 'dbscan':
            # Exclude noise points
            valid_mask = clusters != -1
            if np.sum(valid_mask) < 2:
                return 0.0
            valid_embeddings = embeddings[valid_mask]
            valid_clusters = clusters[valid_mask]
        else:
            valid_embeddings = embeddings
            valid_clusters = clusters

        # Check for sufficient non-singleton clusters
        unique_labels, counts = np.unique(valid_clusters, return_counts=True)
        non_singleton_labels = unique_labels[counts >= 2]

        if len(non_singleton_labels) < 2:
            return 0.0  # Can't compute meaningful silhouette

        # Only compute silhouette on non-singleton points
        non_singleton_mask = np.isin(valid_clusters, non_singleton_labels)
        if np.sum(non_singleton_mask) < 2:
            return 0.0

        try:
            # Use silhouette score for cluster quality
            score = silhouette_score(
                valid_embeddings[non_singleton_mask],
                valid_clusters[non_singleton_mask],
                metric=self.metric
            )
            # Normalize to [0, 1]
            return (score + 1) / 2
        except Exception as e:
            warnings.warn(f"Silhouette computation failed: {e}")
            return 0.0

    def _extract_meaning_clusters(
        self,
        neighbor_indices: np.ndarray,
        clusters: np.ndarray,
        n_clusters: int,
        min_cluster_size: Optional[int] = None
    ) -> List[List[int]]:
        """
        Extract token indices for each meaning cluster.
        """
        meaning_clusters = []

        # Use provided min_cluster_size or fall back to instance default
        effective_min_size = min_cluster_size if min_cluster_size is not None else self.min_cluster_size

        # Get unique cluster labels (handling both DBSCAN and hierarchical)
        unique_labels = np.unique(clusters)

        # For DBSCAN, exclude noise cluster (-1)
        if self.clustering_method == 'dbscan':
            unique_labels = unique_labels[unique_labels != -1]

        for cluster_label in unique_labels:
            cluster_mask = clusters == cluster_label
            cluster_tokens = neighbor_indices[cluster_mask].tolist()

            # Only include clusters with minimum size
            if len(cluster_tokens) >= effective_min_size:
                meaning_clusters.append(cluster_tokens)

        return meaning_clusters

    def _classify_polysemy_type(
        self,
        meaning_clusters: List[List[int]],
        coherence: float,
        n_clusters: int
    ) -> str:
        """
        Classify the type of polysemy.
        """
        if n_clusters == 1:
            return 'none'
        elif n_clusters == 2 and coherence > 0.8:
            # Two very distinct clusters - likely homonym
            return 'homonym'
        elif n_clusters == 2 and coherence > 0.6:
            # Two clusters with some separation - could be contranym
            return 'contranym'
        elif n_clusters > 2:
            # Multiple meanings
            return 'multi-sense'
        else:
            return 'ambiguous'

    def _compute_polysemy_confidence(
        self,
        n_clusters: int,
        coherence: float,
        distances: np.ndarray
    ) -> float:
        """
        Compute confidence score for polysemy detection.

        This is a heuristic score based on:
        1. Number of clusters found (more clusters = higher confidence)
        2. Cluster separation quality (higher silhouette = higher confidence)
        3. Distance distribution irregularity (higher variance = higher confidence)

        Note: This is NOT a calibrated probability. For calibrated scores,
        train a logistic regression on labeled data (e.g., WordNet senses).

        Returns:
            Uncalibrated confidence score in [0, 1]
        """
        if n_clusters <= 1:
            return 0.0

        # 1. Cluster count signal (log scale for diminishing returns)
        # 2 clusters = 0.69, 3 = 0.81, 4 = 0.86, etc.
        cluster_signal = np.log(n_clusters) / (np.log(n_clusters) + 1)

        # 2. Quality signal: How well-separated are they?
        # If coherence is 0 or negative (poor separation), use small positive value
        # to still allow detection if we have many clusters
        separation_quality = max(coherence, 0.1)

        # 3. Stability signal: Variance in distances
        if len(distances) > 1:
            # Coefficient of variation (std/mean) - more interpretable
            cv = np.std(distances) / (np.mean(distances) + 1e-12)
            # Higher CV = more irregular neighborhood = likely polysemous
            # Map to [0, 1] with sigmoid-like function
            stability_signal = np.tanh(cv * 2)  # cv of 0.5 gives ~0.76
        else:
            stability_signal = 0.0

        # Weighted combination
        # Give equal weight to having clusters and their quality
        confidence = (0.4 * cluster_signal +
                     0.4 * separation_quality +
                     0.2 * stability_signal)

        return float(np.clip(confidence, 0, 1))

    def _generate_example_contexts(
        self,
        meaning_clusters: List[List[int]],
        tokenizer
    ) -> List[str]:
        """
        Generate example contexts for each meaning (placeholder).
        """
        if tokenizer is None:
            return [f"Meaning cluster {i+1}" for i in range(len(meaning_clusters))]

        # TODO: Use tokenizer to generate actual example contexts
        contexts = []
        for i, cluster in enumerate(meaning_clusters):
            # Get representative tokens from cluster
            if len(cluster) > 0:
                # In real implementation, decode tokens and create context
                contexts.append(f"Context for meaning {i+1} with {len(cluster)} similar tokens")
            else:
                contexts.append(f"Meaning {i+1}")

        return contexts

    def analyze_vocabulary(
        self,
        embeddings: np.ndarray,
        sample_size: Optional[int] = None,
        tokenizer=None,
        verbose: bool = True
    ) -> PolysemyAnalysis:
        """
        Analyze entire vocabulary for polysemy.

        Args:
            embeddings: (vocab_size, embed_dim) embedding matrix
            sample_size: Number of tokens to sample (None = all)
            tokenizer: Optional tokenizer
            verbose: Print progress

        Returns:
            PolysemyAnalysis with comprehensive results
        """
        vocab_size = len(embeddings)

        # Sample tokens if requested
        if sample_size and sample_size < vocab_size:
            token_indices = self.rng.choice(vocab_size, sample_size, replace=False)
        else:
            token_indices = np.arange(vocab_size)

        polysemous_tokens = []
        homonyms = []
        contranyms = []
        high_risk = []

        for i, token_idx in enumerate(token_indices):
            if verbose and i % 100 == 0:
                print(f"Analyzing token {i}/{len(token_indices)}")

            # Get token string if possible
            token_str = ""
            if tokenizer:
                try:
                    token_str = tokenizer.decode([token_idx])
                except:
                    token_str = f"token_{token_idx}"

            # Detect polysemy
            result = self.detect_polysemy(
                embeddings, token_idx, token_str, tokenizer
            )

            if result.is_polysemous:
                polysemous_tokens.append(result)

                if result.polysemy_type == 'homonym':
                    homonyms.append((token_idx, token_str))
                elif result.polysemy_type == 'contranym':
                    contranyms.append((token_idx, token_str))

                if result.confidence > 0.8:
                    high_risk.append(token_idx)

        # Compute summary statistics
        polysemy_rate = len(polysemous_tokens) / len(token_indices)
        avg_meanings = np.mean([p.num_meanings for p in polysemous_tokens]) if polysemous_tokens else 0
        avg_confidence = np.mean([p.confidence for p in polysemous_tokens]) if polysemous_tokens else 0

        return PolysemyAnalysis(
            total_tokens=len(token_indices),
            polysemous_tokens=polysemous_tokens,
            polysemy_rate=polysemy_rate,
            high_risk_tokens=high_risk,
            homonyms=homonyms,
            contranyms=contranyms,
            summary_stats={
                'polysemy_rate': polysemy_rate,
                'avg_meanings_per_polysemous_token': avg_meanings,
                'avg_confidence': avg_confidence,
                'num_homonyms': len(homonyms),
                'num_contranyms': len(contranyms),
                'num_high_risk': len(high_risk)
            }
        )


def identify_problematic_tokens(
    embeddings: np.ndarray,
    tokenizer=None,
    top_k: int = 100
) -> Dict[str, List[Tuple[int, str, float]]]:
    """
    Identify the most problematic polysemous tokens.

    Args:
        embeddings: Token embedding matrix
        tokenizer: Optional tokenizer
        top_k: Number of top problematic tokens to return

    Returns:
        Dictionary with categories of problematic tokens
    """
    detector = PolysemyDetector()

    # Sample vocabulary for efficiency
    sample_size = min(5000, len(embeddings))
    analysis = detector.analyze_vocabulary(
        embeddings,
        sample_size=sample_size,
        tokenizer=tokenizer,
        verbose=False
    )

    # Sort by confidence
    sorted_polysemous = sorted(
        analysis.polysemous_tokens,
        key=lambda x: x.confidence,
        reverse=True
    )[:top_k]

    # Categorize
    results = {
        'high_polysemy': [],
        'homonyms': [],
        'contranyms': [],
        'multi_sense': []
    }

    for result in sorted_polysemous:
        entry = (result.token_idx, result.token_str, result.confidence)

        if result.num_meanings > 3:
            results['high_polysemy'].append(entry)

        if result.polysemy_type == 'homonym':
            results['homonyms'].append(entry)
        elif result.polysemy_type == 'contranym':
            results['contranyms'].append(entry)
        elif result.polysemy_type == 'multi-sense':
            results['multi_sense'].append(entry)

    return results


if __name__ == "__main__":
    # Example usage
    print("Polysemy Detector - Example")
    print("=" * 50)

    # Generate synthetic embeddings with polysemous structure
    np.random.seed(42)
    vocab_size = 1000
    embed_dim = 128

    # Create base embeddings
    embeddings = np.random.randn(vocab_size, embed_dim) * 0.1

    # Add some polysemous tokens (multiple clusters)
    polysemous_indices = [10, 50, 100, 200, 500]

    for poly_idx in polysemous_indices:
        # Create 2-3 meaning clusters around this token
        n_meanings = np.random.randint(2, 4)

        for meaning in range(n_meanings):
            # Add cluster of similar tokens for each meaning
            cluster_size = np.random.randint(5, 15)
            cluster_center = embeddings[poly_idx] + np.random.randn(embed_dim) * 2

            for _ in range(cluster_size):
                neighbor_idx = np.random.randint(vocab_size)
                if neighbor_idx != poly_idx:
                    embeddings[neighbor_idx] = cluster_center + np.random.randn(embed_dim) * 0.3

    # Test detector
    detector = PolysemyDetector()

    print("\nTesting known polysemous tokens:")
    for idx in polysemous_indices[:3]:
        result = detector.detect_polysemy(
            embeddings, idx, f"token_{idx}"
        )
        print(f"\nToken {idx}:")
        print(f"  Polysemous: {result.is_polysemous}")
        print(f"  Confidence: {result.confidence:.3f}")
        print(f"  Num meanings: {result.num_meanings}")
        print(f"  Type: {result.polysemy_type}")

    print("\nTesting regular token:")
    regular_idx = 1
    result = detector.detect_polysemy(
        embeddings, regular_idx, f"token_{regular_idx}"
    )
    print(f"\nToken {regular_idx}:")
    print(f"  Polysemous: {result.is_polysemous}")
    print(f"  Confidence: {result.confidence:.3f}")

    # Analyze sample of vocabulary
    print("\n" + "=" * 50)
    print("Analyzing vocabulary sample...")
    analysis = detector.analyze_vocabulary(
        embeddings,
        sample_size=100,
        verbose=False
    )

    print(f"\nResults:")
    print(f"  Polysemy rate: {analysis.polysemy_rate:.1%}")
    print(f"  Homonyms found: {len(analysis.homonyms)}")
    print(f"  High-risk tokens: {len(analysis.high_risk_tokens)}")
    print(f"  Avg meanings per polysemous token: {analysis.summary_stats['avg_meanings_per_polysemous_token']:.1f}")