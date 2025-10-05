#!/usr/bin/env python3
"""
Token Dimension Classifier

Classifies tokens into categories matching the paper's dimension distribution:
- Numerics, months, days, cardinals (high dimension ~800)
- Single English words (medium dimension 200-400)
- Word fragments and some single words (low dimension 50-200)
"""

import re
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TokenClassification:
    """Classification result for a token."""
    token: str
    category: str
    expected_dimension_range: Tuple[int, int]
    actual_dimension: Optional[float] = None


class TokenDimensionClassifier:
    """
    Classify tokens into dimension categories based on the paper's findings.

    The paper shows distinct dimension clusters for different token types.
    """

    def __init__(self):
        """Initialize with patterns for token classification."""

        # High dimension patterns (~800)
        self.numeric_pattern = re.compile(r'^\d+$|^\d+\.\d+$|^-?\d+$')
        self.months = ['january', 'february', 'march', 'april', 'may', 'june',
                      'july', 'august', 'september', 'october', 'november', 'december',
                      'jan', 'feb', 'mar', 'apr', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        self.days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday',
                    'saturday', 'sunday', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'sun']
        self.cardinals = ['north', 'south', 'east', 'west', 'northeast', 'northwest',
                         'southeast', 'southwest', 'n', 's', 'e', 'w', 'ne', 'nw', 'se', 'sw']

        # Medium dimension patterns (200-400) - common English words
        # Load from a word frequency list or use heuristics
        self.common_words = set([
            'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i',
            'it', 'for', 'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at',
            'this', 'but', 'his', 'by', 'from', 'they', 'we', 'say', 'her', 'she',
            'or', 'an', 'will', 'my', 'one', 'all', 'would', 'there', 'their',
            'what', 'so', 'up', 'out', 'if', 'about', 'who', 'get', 'which', 'go'
        ])

        # Low dimension patterns (50-200) - fragments and subwords
        self.fragment_patterns = [
            re.compile(r'^##'),  # BERT-style subword
            re.compile(r'^▁'),   # SentencePiece subword
            re.compile(r'^Ġ'),   # GPT-style space token
            re.compile(r'^[^a-zA-Z0-9]+$'),  # Pure punctuation
            re.compile(r'^[a-z]{1,2}$'),  # Very short fragments
        ]

        # Dimension ranges from paper
        self.dimension_ranges = {
            'high_dimension': (600, 850),  # Numerics, dates, etc.
            'medium_dimension': (200, 400),  # Common words
            'low_dimension': (50, 200),  # Fragments
        }

    def classify_token(self, token: str) -> TokenClassification:
        """
        Classify a single token into dimension category.

        Args:
            token: Token string to classify

        Returns:
            TokenClassification with category and expected dimension range
        """
        token_lower = token.lower()

        # Check high dimension categories
        if self.numeric_pattern.match(token):
            return TokenClassification(
                token=token,
                category='numeric',
                expected_dimension_range=self.dimension_ranges['high_dimension']
            )

        if token_lower in self.months:
            return TokenClassification(
                token=token,
                category='month',
                expected_dimension_range=self.dimension_ranges['high_dimension']
            )

        if token_lower in self.days:
            return TokenClassification(
                token=token,
                category='day_of_week',
                expected_dimension_range=self.dimension_ranges['high_dimension']
            )

        if token_lower in self.cardinals:
            return TokenClassification(
                token=token,
                category='cardinal_direction',
                expected_dimension_range=self.dimension_ranges['high_dimension']
            )

        # Check for fragments (low dimension)
        for pattern in self.fragment_patterns:
            if pattern.match(token):
                return TokenClassification(
                    token=token,
                    category='fragment',
                    expected_dimension_range=self.dimension_ranges['low_dimension']
                )

        # Check if it's a short/truncated word
        if len(token) <= 3 and not token_lower in self.common_words:
            return TokenClassification(
                token=token,
                category='fragment',
                expected_dimension_range=self.dimension_ranges['low_dimension']
            )

        # Check for common English words (medium dimension)
        if token_lower in self.common_words or (len(token) > 3 and token.isalpha()):
            return TokenClassification(
                token=token,
                category='english_word',
                expected_dimension_range=self.dimension_ranges['medium_dimension']
            )

        # Default to fragment for unknown
        return TokenClassification(
            token=token,
            category='unknown',
            expected_dimension_range=self.dimension_ranges['low_dimension']
        )

    def classify_tokens(self, tokens: List[str]) -> List[TokenClassification]:
        """Classify multiple tokens."""
        return [self.classify_token(token) for token in tokens]

    def analyze_dimension_distribution(
        self,
        tokens: List[str],
        dimensions: Optional[List[float]] = None
    ) -> Dict:
        """
        Analyze dimension distribution across token categories.

        Args:
            tokens: List of tokens
            dimensions: Optional list of actual dimensions for each token

        Returns:
            Dictionary with distribution analysis
        """
        classifications = self.classify_tokens(tokens)

        # If dimensions provided, attach to classifications
        if dimensions and len(dimensions) == len(tokens):
            for cls, dim in zip(classifications, dimensions):
                cls.actual_dimension = dim

        # Group by category
        category_dims = defaultdict(list)
        for cls in classifications:
            if cls.actual_dimension is not None:
                category_dims[cls.category].append(cls.actual_dimension)

        # Compute statistics
        stats = {}
        for category, dims in category_dims.items():
            if dims:
                stats[category] = {
                    'count': len(dims),
                    'mean_dimension': np.mean(dims),
                    'std_dimension': np.std(dims),
                    'min_dimension': np.min(dims),
                    'max_dimension': np.max(dims)
                }

        return {
            'classifications': classifications,
            'category_stats': stats,
            'total_tokens': len(tokens)
        }

    def plot_dimension_histogram(
        self,
        tokens: List[str],
        dimensions: List[float],
        title: str = "Token Dimension Distribution (Following Paper's Figure 2)"
    ):
        """
        Create histogram matching the paper's Figure 2.

        Args:
            tokens: List of tokens
            dimensions: List of dimensions for each token
            title: Plot title
        """
        # Classify tokens
        classifications = self.classify_tokens(tokens)
        for cls, dim in zip(classifications, dimensions):
            cls.actual_dimension = dim

        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        # Top plot: Histogram of all dimensions
        ax1.hist(dimensions, bins=50, alpha=0.7, color='darkblue', edgecolor='black')
        ax1.set_xlabel('Dimension')
        ax1.set_ylabel('Probability Density')
        ax1.set_title('Distribution of Local Dimensions')
        ax1.set_xlim(0, 850)
        ax1.set_ylim(0, 0.03)

        # Add category annotations
        ax1.annotate('Numerics, months, days of week, cardinal directions',
                    xy=(700, 0.025), fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.3))
        ax1.annotate('Single English words',
                    xy=(300, 0.02), fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.3))
        ax1.annotate('Mostly word fragments\nand a few single words',
                    xy=(100, 0.015), fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.3))

        # Bottom plot: Category breakdown
        categories = defaultdict(list)
        for cls in classifications:
            if cls.actual_dimension:
                categories[cls.category].append(cls.actual_dimension)

        # Create box plot
        category_data = []
        category_labels = []
        for cat in ['numeric', 'month', 'day_of_week', 'cardinal_direction',
                   'english_word', 'fragment', 'unknown']:
            if cat in categories:
                category_data.append(categories[cat])
                category_labels.append(cat.replace('_', ' ').title())

        if category_data:
            bp = ax2.boxplot(category_data, labels=category_labels, patch_artist=True)
            for patch in bp['boxes']:
                patch.set_facecolor('lightblue')
            ax2.set_ylabel('Dimension')
            ax2.set_xlabel('Token Category')
            ax2.set_title('Dimension Distribution by Token Type')
            ax2.grid(True, alpha=0.3)

        plt.suptitle(title)
        plt.tight_layout()
        return fig

    def validate_against_paper(self, classifications: List[TokenClassification]) -> Dict:
        """
        Validate classifications against paper's expected dimensions.

        Returns metrics on how well actual dimensions match expected ranges.
        """
        correct = 0
        total_with_dims = 0

        misclassified = []

        for cls in classifications:
            if cls.actual_dimension is not None:
                total_with_dims += 1
                min_dim, max_dim = cls.expected_dimension_range

                if min_dim <= cls.actual_dimension <= max_dim:
                    correct += 1
                else:
                    misclassified.append({
                        'token': cls.token,
                        'category': cls.category,
                        'expected_range': cls.expected_dimension_range,
                        'actual_dimension': cls.actual_dimension
                    })

        accuracy = correct / total_with_dims if total_with_dims > 0 else 0

        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total_with_dims,
            'misclassified': misclassified[:10]  # Top 10 misclassified
        }


def test_classifier():
    """Test the token dimension classifier."""
    classifier = TokenDimensionClassifier()

    # Test tokens
    test_tokens = [
        '123', '456.78', '-42',  # Numerics
        'January', 'feb', 'Wednesday',  # Dates
        'north', 'SW',  # Cardinals
        'the', 'and', 'computer',  # English words
        '##ing', '▁the', 'a',  # Fragments
        '@', '!!!', ','  # Punctuation
    ]

    print("Token Classification Tests:")
    print("-" * 50)

    for token in test_tokens:
        cls = classifier.classify_token(token)
        print(f"{token:15} -> {cls.category:20} (expected dim: {cls.expected_dimension_range})")

    # Test with mock dimensions
    print("\n\nDimension Distribution Analysis:")
    print("-" * 50)

    # Generate mock dimensions based on categories
    mock_dims = []
    for token in test_tokens:
        cls = classifier.classify_token(token)
        min_d, max_d = cls.expected_dimension_range
        mock_dims.append(np.random.uniform(min_d, max_d))

    analysis = classifier.analyze_dimension_distribution(test_tokens, mock_dims)

    for category, stats in analysis['category_stats'].items():
        print(f"\n{category}:")
        print(f"  Count: {stats['count']}")
        print(f"  Mean dimension: {stats['mean_dimension']:.1f}")
        print(f"  Std dimension: {stats['std_dimension']:.1f}")


if __name__ == "__main__":
    test_classifier()