#!/usr/bin/env python3
"""
Optimized dataset loaders with caching for efficient Fisher analysis.
- AIME-24 + AIME-25 + filtered MATH-500 for math data
- C4 samples for non-math data
- Efficient batch processing and caching
"""

import json
import re
import torch
import random
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OptimizedDataLoader:
    """Optimized data loader with caching for math and non-math datasets."""

    def __init__(self, data_dir: str = "./data", verbose: bool = False):
        """
        Initialize optimized data loader.

        Args:
            data_dir: Directory containing data files
            verbose: Whether to show detailed logging
        """
        self.data_dir = Path(data_dir)
        self.verbose = verbose

        # Caches
        self.math_cache = None
        self.c4_cache = None
        self.math_samples_cache = None  # Cache raw text samples
        self.c4_samples_cache = None

        # Set logging level
        if not verbose:
            logger.setLevel(logging.WARNING)

    def load_all_math_data(
        self,
        tokenizer,
        num_samples: Optional[int] = None,  # If specified, return only this many samples
        max_length: int = 256,
        batch_size: int = 256
    ) -> Dict[str, torch.Tensor]:
        """
        Load all math data: AIME-24, AIME-25, and filtered MATH-500.

        Args:
            tokenizer: Tokenizer to use
            num_samples: Number of samples to return (None = all, default)
            max_length: Maximum sequence length
            batch_size: Batch size for tokenization

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Return cached if available (and sufficient samples)
        if self.math_cache is not None:
            if num_samples is None:
                logger.info("Using cached math data (all samples)")
                return self.math_cache
            elif len(self.math_cache['input_ids']) >= num_samples:
                logger.info(f"Using cached math data (first {num_samples} samples)")
                return {
                    'input_ids': self.math_cache['input_ids'][:num_samples],
                    'attention_mask': self.math_cache['attention_mask'][:num_samples]
                }
            # Fall through to load more if cache doesn't have enough

        # Load samples if not cached
        if self.math_samples_cache is None:
            all_samples = []
            counts = {}

            # Load AIME-24 (240 problems)
            aime24_path = self.data_dir / "aime24x8" / "test.jsonl"
            if aime24_path.exists():
                with open(aime24_path, 'r') as f:
                    aime24_samples = []
                    for line in f:
                        item = json.loads(line)
                        # Extract numeric part from answer (handle degrees, etc)
                        answer_str = str(item['answer'])
                        # Remove degree symbols and other non-numeric characters
                        numeric_answer = re.sub(r'[^\d]', '', answer_str.split()[0])
                        if numeric_answer:
                            answer = f"{int(numeric_answer):03d}"
                            text = f"Problem: {item['problem']}\nAnswer: {answer}"
                            aime24_samples.append(text)
                    all_samples.extend(aime24_samples)
                    counts['AIME-24'] = len(aime24_samples)

            # Load AIME-25 (240 problems)
            aime25_path = self.data_dir / "aime25x8" / "test.jsonl"
            if aime25_path.exists():
                with open(aime25_path, 'r') as f:
                    aime25_samples = []
                    for line in f:
                        item = json.loads(line)
                        # Extract numeric part from answer (handle degrees, etc)
                        answer_str = str(item['answer'])
                        # Remove degree symbols and other non-numeric characters
                        numeric_answer = re.sub(r'[^\d]', '', answer_str.split()[0])
                        if numeric_answer:
                            answer = f"{int(numeric_answer):03d}"
                            text = f"Problem: {item['problem']}\nAnswer: {answer}"
                            aime25_samples.append(text)
                    all_samples.extend(aime25_samples)
                    counts['AIME-25'] = len(aime25_samples)

            # Load filtered MATH-500 (simple integer answers to get exactly 288)
            math500_path = self.data_dir / "math500" / "test.jsonl"
            if math500_path.exists():
                with open(math500_path, 'r') as f:
                    math500_samples = []
                    total_math500 = 0

                    for line in f:
                        total_math500 += 1
                        try:
                            item = json.loads(line)
                            # Extract answer and clean it
                            answer_str = str(item['answer']).strip()

                            # First priority: 1-3 digit positive, 1-2 digit negative
                            if re.fullmatch(r"\d{1,3}", answer_str) or re.fullmatch(r"-\d{1,2}", answer_str):
                                text = f"Problem: {item['problem']}\nAnswer: {answer_str}"
                                math500_samples.append(text)
                            # Second priority: 4-digit integers (to fill up to 288)
                            elif len(math500_samples) < 288 and re.fullmatch(r"\d{4}", answer_str):
                                text = f"Problem: {item['problem']}\nAnswer: {answer_str}"
                                math500_samples.append(text)

                            # Stop at exactly 288 samples
                            if len(math500_samples) >= 288:
                                break
                        except json.JSONDecodeError:
                            continue  # Skip corrupted lines if any

                    # If we're still short, take the 288 we have
                    math500_samples = math500_samples[:288]
                    all_samples.extend(math500_samples)
                    counts['MATH-500'] = len(math500_samples)

                    if self.verbose:
                        logger.info(f"Filtered MATH-500: kept {len(math500_samples)}/{total_math500} problems")

            # Shuffle with fixed seed for reproducibility
            random.seed(42)
            random.shuffle(all_samples)

            self.math_samples_cache = all_samples

            # Log summary
            total = sum(counts.values())
            summary = ", ".join([f"{k}: {v}" for k, v in counts.items()])
            logger.info(f"Loaded {total} math samples ({summary})")
        else:
            all_samples = self.math_samples_cache
            logger.info(f"Using cached {len(all_samples)} math samples")

        # Tokenize in batches
        self.math_cache = self._tokenize_in_batches(
            all_samples, tokenizer, max_length, batch_size
        )

        # Return requested subset if num_samples specified
        if num_samples is not None and len(self.math_cache['input_ids']) > num_samples:
            logger.info(f"Returning first {num_samples} of {len(self.math_cache['input_ids'])} math samples")
            return {
                'input_ids': self.math_cache['input_ids'][:num_samples],
                'attention_mask': self.math_cache['attention_mask'][:num_samples]
            }

        return self.math_cache

    def load_c4_samples(
        self,
        tokenizer,
        num_samples: int = 768,  # Match math data size (240+240+288)
        max_length: int = 256,
        batch_size: int = 256
    ) -> Dict[str, torch.Tensor]:
        """
        Load C4 samples efficiently with caching.

        Args:
            tokenizer: Tokenizer to use
            num_samples: Number of samples to load
            max_length: Maximum sequence length
            batch_size: Batch size for tokenization

        Returns:
            Dictionary with input_ids and attention_mask
        """
        # Return cached if we have enough samples
        if self.c4_cache is not None and len(self.c4_cache['input_ids']) >= num_samples:
            logger.info(f"Using cached C4 data (first {num_samples} samples)")
            return {
                'input_ids': self.c4_cache['input_ids'][:num_samples],
                'attention_mask': self.c4_cache['attention_mask'][:num_samples]
            }

        # Load samples if not cached
        if self.c4_samples_cache is None or len(self.c4_samples_cache) < num_samples:
            # Try to load from local file first
            local_c4_path = self.data_dir / "c4_validation_samples.jsonl"

            if local_c4_path.exists():
                logger.info(f"Loading C4 samples from local file: {local_c4_path}")
                samples = []
                with open(local_c4_path, 'r') as f:
                    for line in f:
                        item = json.loads(line)
                        samples.append(item['text'])
                        if len(samples) >= num_samples * 1.5:  # Get some extra
                            break
            else:
                # Fallback to downloading if local file doesn't exist
                logger.warning(f"Local C4 file not found at {local_c4_path}")
                logger.info("Run fetch_c4_streaming.py to create local cache")

                # Use simple non-math samples as fallback
                samples = [
                    "The history of artificial intelligence began in antiquity.",
                    "Climate change includes both global warming and weather shifts.",
                    "The Renaissance was a period in European history.",
                    "Jazz originated in the African-American communities.",
                    "The Internet is a global system of computer networks.",
                ] * (num_samples // 5 + 1)  # Repeat to get enough samples

            # Set seed for reproducibility
            random.seed(42)

            # Shuffle and trim to exact number
            random.shuffle(samples)
            self.c4_samples_cache = samples[:num_samples]

            logger.info(f"Loaded {len(self.c4_samples_cache)} C4 samples (from local cache)")
        else:
            logger.info(f"Using cached C4 text samples")

        # Use only the requested number of samples
        samples_to_use = self.c4_samples_cache[:num_samples]

        # Tokenize in batches
        tokenized = self._tokenize_in_batches(
            samples_to_use, tokenizer, max_length, batch_size
        )

        # Update cache if we tokenized more samples
        if self.c4_cache is None or len(tokenized['input_ids']) > len(self.c4_cache.get('input_ids', [])):
            self.c4_cache = tokenized

        return tokenized

    def load_mixed_batch(
        self,
        tokenizer,
        num_samples: int = 1000,
        max_length: int = 256,
        math_ratio: float = 0.5,
        batch_size: int = 256
    ) -> Dict[str, torch.Tensor]:
        """
        Load a mixed batch of math and non-math data.

        Args:
            tokenizer: Tokenizer to use
            num_samples: Total number of samples
            max_length: Maximum sequence length
            math_ratio: Fraction that should be math (0-1)
            batch_size: Batch size for tokenization

        Returns:
            Dictionary with input_ids and attention_mask
        """
        num_math = int(num_samples * math_ratio)
        num_c4 = num_samples - num_math

        # Load both datasets
        math_data = self.load_all_math_data(tokenizer, max_length, batch_size)
        c4_data = self.load_c4_samples(tokenizer, num_c4, max_length, batch_size)

        # Take required number from each
        math_subset = {
            'input_ids': math_data['input_ids'][:num_math],
            'attention_mask': math_data['attention_mask'][:num_math]
        }

        # Combine
        combined_input_ids = torch.cat([
            math_subset['input_ids'],
            c4_data['input_ids']
        ], dim=0)

        combined_attention_mask = torch.cat([
            math_subset['attention_mask'],
            c4_data['attention_mask']
        ], dim=0)

        # Shuffle
        perm = torch.randperm(num_samples)

        return {
            'input_ids': combined_input_ids[perm],
            'attention_mask': combined_attention_mask[perm]
        }

    def _tokenize_in_batches(
        self,
        samples: List[str],
        tokenizer,
        max_length: int,
        batch_size: int
    ) -> Dict[str, torch.Tensor]:
        """
        Tokenize samples in batches for efficiency.

        Args:
            samples: List of text samples
            tokenizer: Tokenizer to use
            max_length: Maximum sequence length
            batch_size: Batch size for tokenization

        Returns:
            Dictionary with input_ids and attention_mask
        """
        all_input_ids = []
        all_attention_masks = []

        # Process in batches
        for i in range(0, len(samples), batch_size):
            batch_samples = samples[i:i + batch_size]

            if self.verbose and i % (batch_size * 10) == 0:
                logger.info(f"Tokenizing batch {i//batch_size + 1}/{(len(samples) + batch_size - 1)//batch_size}")

            encodings = tokenizer(
                batch_samples,
                truncation=True,
                padding='max_length',
                max_length=max_length,
                return_tensors='pt'
            )

            all_input_ids.append(encodings['input_ids'])
            all_attention_masks.append(encodings['attention_mask'])

        # Create labels for language modeling (shifted input_ids)
        all_labels = []
        for input_ids in all_input_ids:
            labels = input_ids.clone()
            # Set padding tokens to -100 so they're ignored in loss
            labels[labels == tokenizer.pad_token_id] = -100
            all_labels.append(labels)

        return {
            'input_ids': torch.cat(all_input_ids, dim=0),
            'attention_mask': torch.cat(all_attention_masks, dim=0),
            'labels': torch.cat(all_labels, dim=0)  # Added for language modeling
        }

    def clear_cache(self):
        """Clear all caches to free memory."""
        self.math_cache = None
        self.c4_cache = None
        self.math_samples_cache = None
        self.c4_samples_cache = None
        logger.info("Cleared all data caches")


# Convenience functions for backward compatibility
def create_optimized_loader(verbose: bool = False) -> OptimizedDataLoader:
    """Create an optimized data loader instance."""
    return OptimizedDataLoader(verbose=verbose)


if __name__ == "__main__":
    # Test the optimized loader
    from transformers import AutoTokenizer

    print("Testing Optimized Data Loader")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    loader = OptimizedDataLoader(verbose=True)

    # Test math data loading
    print("\n1. Loading Math Data (AIME + filtered MATH-500):")
    math_data = loader.load_all_math_data(tokenizer)
    print(f"   Shape: {math_data['input_ids'].shape}")

    # Test C4 loading
    print("\n2. Loading C4 Data:")
    c4_data = loader.load_c4_samples(tokenizer, num_samples=780)
    print(f"   Shape: {c4_data['input_ids'].shape}")

    # Test mixed loading
    print("\n3. Loading Mixed Batch:")
    mixed_data = loader.load_mixed_batch(tokenizer, num_samples=512, math_ratio=0.5)
    print(f"   Shape: {mixed_data['input_ids'].shape}")

    print("\n✓ Optimized data loader ready!")
    print("Benefits:")
    print("  - One-time loading with caching")
    print("  - Clean MATH-500 filtering")
    print("  - Efficient batch processing")
    print("  - Reproducible with fixed seeds")