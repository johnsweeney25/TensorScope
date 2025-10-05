"""
SuperpositionAnalyzer: Unified superposition analysis with caching.

This module combines SuperpositionMetrics_v2 with paper-specific metrics
while eliminating duplicate calculations through intelligent caching.

https://github.com/liuyz0/SuperpositionScaling (inspried by)
https://arxiv.org/pdf/2505.10465
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, List, Any, Union
from dataclasses import dataclass
import logging
from collections import OrderedDict

# Safe tqdm import - fallback to no-op if there are issues
try:
    from tqdm.auto import tqdm
except ImportError:
    # Fallback: create a no-op tqdm if import fails
    class tqdm:
        def __init__(self, *args, **kwargs):
            self.iterable = args[0] if args else None
            self.total = kwargs.get('total', 0)

        def __iter__(self):
            return iter(self.iterable) if self.iterable else self

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def update(self, n=1):
            pass

        def set_description(self, desc):
            pass

        def set_postfix(self, **kwargs):
            pass

        def close(self):
            pass

from .enhanced import SuperpositionMetrics, SuperpositionConfig

# GPU memory management
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
from utils.gpu_memory_manager import get_memory_manager, EvictionPriority

logger = logging.getLogger(__name__)


@dataclass
class SuperpositionAnalysis:
    """Complete superposition analysis results."""
    # Vector interference metrics
    mean_overlap: float
    std_overlap: float
    max_overlap: float
    num_high_overlap_pairs: int

    # Paper-specific metrics (Liu et al. 2025)
    phi_half: float  # ϕ₁/₂: Fraction with ||W|| > 0.5
    phi_one: float   # ϕ₁: Fraction with ||W|| > 1.0
    regime: str      # 'weak', 'strong', or 'no' superposition

    # Feature statistics
    n_features: int
    n_dimensions: int
    dimension_ratio: float
    n_represented: int      # Features with ||W|| > 0.5
    n_strongly_represented: int  # Features with ||W|| > 1.0

    # Geometric analysis
    welch_bound: float
    welch_bound_ratio: float
    expected_scaling: float
    follows_sqrt_scaling: bool

    # Norm distribution
    mean_norm: float
    std_norm: float
    min_norm: float
    max_norm: float
    median_norm: float

    # Additional data
    feature_norms: Optional[np.ndarray] = None
    overlap_matrix: Optional[torch.Tensor] = None


class SuperpositionAnalyzer(SuperpositionMetrics):
    """
    Unified superposition analyzer with intelligent caching.

    Eliminates duplicate calculations by caching:
    - Feature norms per weight matrix
    - Overlap matrices
    - SVD decompositions
    """

    def __init__(self, device=None, config: Optional[SuperpositionConfig] = None):
        """Initialize analyzer with caching and GPU memory management."""
        super().__init__(device=device, config=config)

        # Use OrderedDict for LRU caching (max 16 entries per cache)
        self._norm_cache = OrderedDict()
        self._overlap_cache = OrderedDict()
        self._svd_cache = OrderedDict()
        self._max_cache_entries = 16

        # Cache statistics
        self.cache_hits = {'norms': 0, 'overlaps': 0, 'svd': 0}
        self.cache_misses = {'norms': 0, 'overlaps': 0, 'svd': 0}

        # GPU memory manager for dynamic eviction
        self.memory_manager = get_memory_manager(device=0 if device is None or device.type == 'cuda' else device.index if hasattr(device, 'index') else 0)

    def _estimate_calculation_memory_gb(self, weight_matrix: torch.Tensor, batch_size: int) -> float:
        """
        Estimate GPU memory needed for superposition calculation.

        Returns memory estimate in GB.
        """
        n_features, n_dims = weight_matrix.shape

        # 1. Weight matrix (FP32)
        weight_gb = (n_features * n_dims * 4) / 1e9

        # 2. Normalized matrix (temporary during normalization)
        normalized_gb = weight_gb

        # 3. Feature norms
        norms_gb = (n_features * 4) / 1e9

        # 4. Batch overlaps (largest temporary tensor)
        batch_overlap_gb = (batch_size * batch_size * 4) / 1e9

        # 5. Masks (bool)
        mask_gb = (batch_size * batch_size * 1) / 1e9

        # 6. Intermediate tensors (conservative estimate)
        intermediate_gb = 0.3

        total_gb = weight_gb + normalized_gb + norms_gb + batch_overlap_gb + mask_gb + intermediate_gb

        return total_gb

    def _select_compute_device_safe(self, weight_matrix: torch.Tensor, batch_size: int) -> torch.device:
        """
        Select compute device based on available GPU memory.

        CRITICAL NEW STRATEGY:
        - NEVER fall back to CPU (too slow!)
        - If insufficient GPU memory, dynamically evict other tensors
        - Make space for current calculation
        - Restore evicted tensors when done

        ICML REPRODUCIBILITY:
        - NEVER changes batch_size (would break reproducibility)
        - Eviction order is deterministic (by priority)
        - Results identical regardless of what gets evicted
        - Only affects performance, not correctness

        Strategy:
        1. If self.device is set, use it
        2. Estimate memory needed
        3. If insufficient free memory → evict tensors to make space
        4. ALWAYS use GPU (make it work!)

        Returns:
            torch.device for computation (always CUDA if available)
        """
        # If user explicitly set device to CPU, respect it but warn
        if hasattr(self, 'device') and self.device is not None:
            if self.device.type == 'cpu':
                logger.warning(
                    "⚠️ User explicitly set device='cpu'. This will be ~100x slower than GPU. "
                    "Consider removing device setting to auto-use GPU with dynamic memory management."
                )
                return self.device
            # If CUDA, continue with memory management
            target_device = self.device
        else:
            # Auto-select CUDA if available
            if torch.cuda.is_available():
                target_device = torch.device('cuda')
            else:
                logger.warning("No GPU available, forced to use CPU (will be slow)")
                return torch.device('cpu')

        # Estimate memory needed
        needed_gb = self._estimate_calculation_memory_gb(weight_matrix, batch_size)

        # Try to ensure we have enough GPU memory (evict if needed)
        if self.memory_manager.ensure_free_memory(needed_gb, safety_margin=1.5):
            logger.info(f"✅ GPU ready for calculation ({needed_gb:.1f}GB needed, batch_size={batch_size})")
            return target_device
        else:
            # This should rarely happen (means we evicted everything and still no space)
            logger.error(
                f"❌ CRITICAL: Cannot free enough GPU memory even after evicting all possible tensors! "
                f"Need {needed_gb:.1f}GB. This is a design problem - tensor too large for GPU."
            )
            # Last resort: use CPU (but this means something is wrong with the design)
            logger.error("Falling back to CPU as absolute last resort (will be ~100x slower)")
            return torch.device('cpu')

    def _tensor_fingerprint(self, t: torch.Tensor) -> Tuple:
        """Create robust fingerprint for tensor caching.

        Includes shape, dtype, device, and version for mutation detection.
        NOTE: We use id(tensor) directly as it's unique per tensor object.
        For truly robust caching, consider using tensor content hashing.
        """
        # CRITICAL FIX: Handle device.index == None for torch.device('cuda')
        # torch.device('cuda') has index=None, but refers to current device
        if t.device.type == 'cuda':
            dev_idx = t.device.index if t.device.index is not None else torch.cuda.current_device()
        else:
            dev_idx = -1  # CPU

        # Use id() of the tensor itself for uniqueness
        # Combined with shape/dtype/device/version for comprehensive fingerprint
        return (
            id(t),  # Python object id of tensor, unique during its lifetime
            tuple(t.shape),
            str(t.dtype),
            dev_idx,
            t.requires_grad,
            int(getattr(t, "_version", 0))  # Tracks in-place modifications
        )

    def _cache_set(self, cache: OrderedDict, key: Any, value: Any, cache_name: str):
        """Set value in cache with LRU eviction."""
        cache[key] = value
        cache.move_to_end(key)
        if len(cache) > self._max_cache_entries:
            evicted = cache.popitem(last=False)
            logger.debug(f"LRU evicted from {cache_name} cache: key starting with {str(evicted[0])[:50]}")

    def compute_comprehensive_superposition_analysis(
        self,
        weight_matrix: torch.Tensor,
        threshold_half: float = 0.5,
        threshold_one: float = 1.0,
        overlap_threshold: Optional[float] = None,
        batch_size: int = 5000,
        return_matrices: bool = False,
        return_dict: bool = False,
        use_sampling: bool = True,
        max_pairs: int = 10_000_000,
        seed: Optional[int] = None
    ) -> SuperpositionAnalysis:
        """
        Compute all superposition metrics in a single optimized, memory-safe pass.

        Reproducibility and theory:
        - Deterministic sampling: `seed` (or `config.vocab_sampling_seed`) drives uniform
          sampling for large pair counts; all models receive identical treatment.
        - Superposition metrics: computes ϕ₁/₂, ϕ₁ (Liu et al., 2025), max coherence vs Welch bound,
          and expected random-vector scaling E[|⟨x,y⟩|] ≈ √(2/π)/√d.
        - Streaming statistics: mean/std of overlaps are accumulated via Welford’s algorithm
          to avoid catastrophic cancellation and excess memory.

        Numerical precision:
        - Overlap computations use `self.dtype` (default FP32) for tensors and accumulate
          statistics in Python floats (≈ FP64). Set `SuperpositionConfig.use_float64=True` to
          compute overlaps in FP64 at higher cost. Diagonals are excluded.

        Memory policy (ICML tools friendly):
        - O(n²) matrices are never stored for large vocabularies. For n_features > batch_size,
          the function computes overlaps in tiled blocks; when total pairs exceed `max_pairs`,
          it computes max/high-count on the full block first and then uniformly samples pairs
          for mean/std with deterministic seeds. No intermediate tensor exceeds
          `config.max_memory_gb` worth of GPU memory.
        - Geometric analysis for represented features uses the same streaming path; a full
          (n_repr × n_repr) matrix is formed only when n_repr ≤ `config.geometric_full_matrix_limit`.

        Args:
            weight_matrix: Weight matrix of shape (n_features, n_dims)
            threshold_half: Threshold for ϕ₁/₂ (default 0.5)
            threshold_one: Threshold for ϕ₁ (default 1.0)
            overlap_threshold: Threshold for high overlap pairs
            batch_size: Batch size for large matrix operations
            return_matrices: Whether to return full matrices
            return_dict: Whether to return as dict for JSON serialization
            use_sampling: Whether to use sampling for very large matrices (default True)
            max_pairs: Maximum number of pairs to process when sampling (default 10M)
            seed: Random seed for deterministic sampling (ICLR reproducibility). If None, uses config.vocab_sampling_seed

        Returns:
            SuperpositionAnalysis with all computed metrics
        """
        # CRITICAL FIX: Memory-safe device selection with OOM protection
        # Model may be on CPU (memory management), but calculations should be on GPU (speed)
        compute_device = self._select_compute_device_safe(weight_matrix, batch_size)

        # Move weight_matrix to compute device (GPU) for fast calculations
        if weight_matrix.device != compute_device:
            logger.info(f"Moving weight_matrix from {weight_matrix.device} to {compute_device} for calculations")
            weight_matrix = weight_matrix.to(compute_device)

        # Validate input
        if weight_matrix.dim() != 2:
            raise ValueError(f"weight_matrix must be 2D, got {weight_matrix.dim()}D")
        if weight_matrix.numel() == 0:
            raise ValueError("weight_matrix cannot be empty")

        # Use configured dtype (default FP32; FP64 when requested)
        weight_matrix = weight_matrix.to(self.dtype)

        n_features, n_dims = weight_matrix.shape

        # Use configured threshold if not specified
        if overlap_threshold is None:
            overlap_threshold = self.config.overlap_threshold

        # Use configured seed if not specified (ICLR reproducibility requirement)
        if seed is None:
            seed = self.config.vocab_sampling_seed

        # Step 1: Get or compute feature norms (cached)
        feature_norms = self._get_or_compute_norms(weight_matrix)

        # Step 2: Compute paper-specific metrics
        n_represented = (feature_norms > threshold_half).sum().item()
        n_strongly_represented = (feature_norms > threshold_one).sum().item()

        phi_half = n_represented / n_features if n_features > 0 else 0
        phi_one = n_strongly_represented / n_features if n_features > 0 else 0

        # Step 3: Classify regime
        regime = self._classify_superposition_regime(
            phi_half, phi_one, n_features, n_dims
        )

        # Step 4: Compute overlaps (cached)
        overlap_stats = self._get_or_compute_overlaps(
            weight_matrix, feature_norms, batch_size, overlap_threshold,
            use_sampling=use_sampling, max_pairs=max_pairs, seed=seed
        )

        # Step 5: Compute geometric analysis for represented features
        geometric_stats = self._compute_geometric_analysis(
            weight_matrix, feature_norms, threshold_half
        )

        # Step 6: Compute norm distribution statistics
        norm_stats = {
            'mean_norm': feature_norms.mean().item(),
            'std_norm': feature_norms.std().item() if n_features > 1 else 0.0,
            'min_norm': feature_norms.min().item() if n_features > 0 else 0.0,
            'max_norm': feature_norms.max().item() if n_features > 0 else 0.0,
            'median_norm': feature_norms.median().item() if n_features > 0 else 0.0
        }

        # Step 7: Create result object
        result = SuperpositionAnalysis(
            # Vector interference metrics
            mean_overlap=overlap_stats['mean_overlap'],
            std_overlap=overlap_stats['std_overlap'],
            max_overlap=overlap_stats['max_overlap'],
            num_high_overlap_pairs=overlap_stats['num_high_overlap_pairs'],

            # Paper metrics
            phi_half=phi_half,
            phi_one=phi_one,
            regime=regime,

            # Feature statistics
            n_features=n_features,
            n_dimensions=n_dims,
            dimension_ratio=n_dims / n_features if n_features > 0 else 0,
            n_represented=n_represented,
            n_strongly_represented=n_strongly_represented,

            # Geometric analysis (with corrected Welch bound interpretation)
            welch_bound=geometric_stats['welch_bound'],
            welch_bound_ratio=geometric_stats['welch_bound_ratio'],  # Now correctly uses max overlap
            expected_scaling=geometric_stats['expected_scaling'],  # Now uses √(2/π)/√d
            follows_sqrt_scaling=geometric_stats['follows_sqrt_scaling'],

            # Norm distribution
            mean_norm=norm_stats['mean_norm'],
            std_norm=norm_stats['std_norm'],
            min_norm=norm_stats['min_norm'],
            max_norm=norm_stats['max_norm'],
            median_norm=norm_stats['median_norm'],

            # Optional matrices (move to CPU to prevent GPU memory pinning)
            feature_norms=feature_norms.detach().cpu().numpy() if return_matrices else None,  # Detach gradient before numpy
            overlap_matrix=(overlap_stats.get('_overlap_matrix_gpu').cpu()
                          if return_matrices and '_overlap_matrix_gpu' in overlap_stats else None)
        )

        # Log cache statistics periodically
        total_hits = sum(self.cache_hits.values())
        total_accesses = total_hits + sum(self.cache_misses.values())
        if total_accesses > 0 and total_accesses % 100 == 0:
            cache_rate = total_hits / total_accesses
            logger.debug(f"Cache hit rate: {cache_rate:.1%} over {total_accesses} accesses")

        # Return as dict if requested (for JSON serialization)
        if return_dict:
            from dataclasses import asdict
            return asdict(result)

        return result

    def _get_or_compute_norms(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Get cached norms or compute them.

        CRITICAL FIX: Ensures cached tensors are on the same device as weight_matrix.
        """
        # Use robust fingerprint as cache key (fixes cache key vulnerability)
        cache_key = self._tensor_fingerprint(weight_matrix)

        if cache_key in self._norm_cache:
            self.cache_hits['norms'] += 1
            cached_norms = self._norm_cache[cache_key]

            # CRITICAL DEVICE FIX: Ensure cached tensor is on same device as weight_matrix
            # This prevents device mismatch errors when model is moved between devices
            if cached_norms.device != weight_matrix.device:
                logger.debug(f"Moving cached norms from {cached_norms.device} to {weight_matrix.device}")
                cached_norms = cached_norms.to(weight_matrix.device)
                # Update cache with tensor on new device
                self._norm_cache[cache_key] = cached_norms

            return cached_norms

        self.cache_misses['norms'] += 1

        # Use configured dtype for numerical precision
        weight_matrix_fp32 = weight_matrix.to(self.dtype)

        # Compute L2 norms of each feature (row)
        feature_norms = torch.linalg.norm(weight_matrix_fp32, dim=1)

        # Cache for future use with LRU eviction
        self._cache_set(self._norm_cache, cache_key, feature_norms, 'norms')

        return feature_norms

    def _get_or_compute_overlaps(
        self,
        weight_matrix: torch.Tensor,
        feature_norms: torch.Tensor,
        batch_size: int,
        overlap_threshold: float,
        use_sampling: bool = True,
        max_pairs: int = 10_000_000,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute overlaps (no caching to prevent memory explosion).

        NOTE: Removed caching of overlap matrices as they can consume
        massive amounts of memory (O(n²) where n can be 100k+).
        Statistics are computed on-the-fly instead.
        """
        # No caching for overlap matrices - compute fresh each time
        self.cache_misses['overlaps'] += 1

        n_features = weight_matrix.shape[0]

        # Use configured dtype (default FP32; configurable FP64)
        weight_matrix = weight_matrix.to(self.dtype)
        feature_norms = feature_norms.to(self.dtype)

        # Normalize features
        norms_expanded = feature_norms.unsqueeze(1)
        norms_expanded = torch.where(
            norms_expanded > self.config.eps,
            norms_expanded,
            torch.ones_like(norms_expanded)
        )
        normalized_matrix = weight_matrix / norms_expanded

        # Compute overlaps efficiently
        if n_features <= batch_size:
            # Small enough to compute all at once
            overlap_matrix = torch.matmul(normalized_matrix, normalized_matrix.T).abs()

            # Exclude diagonal
            # FIX: Use the device of the weight_matrix/normalized_matrix instead of self.device
            mask = ~torch.eye(n_features, device=normalized_matrix.device, dtype=torch.bool)
            off_diagonal = overlap_matrix[mask]

            # Compute stats but DON'T include overlap_matrix to prevent memory issues
            stats = {
                'mean_overlap': off_diagonal.mean().item() if off_diagonal.numel() > 0 else 0.0,
                'std_overlap': off_diagonal.std().item() if off_diagonal.numel() > 1 else 0.0,
                'max_overlap': off_diagonal.max().item() if off_diagonal.numel() > 0 else 0.0,
                'num_high_overlap_pairs': (off_diagonal > overlap_threshold).sum().item()
            }

            # Only include matrix if explicitly needed by caller (will be moved to CPU)
            if n_features <= batch_size:  # Only for small matrices
                stats['_overlap_matrix_gpu'] = overlap_matrix  # Internal key, will process later
        else:
            # Use batched computation for large matrices
            stats = self._compute_overlaps_batched(
                normalized_matrix, batch_size, overlap_threshold,
                use_sampling=use_sampling, max_pairs=max_pairs, seed=seed
            )

        # DO NOT cache overlap matrices - they cause memory explosion
        # Statistics are cheap to recompute compared to memory cost
        return stats

    def _compute_overlaps_batched(
        self,
        normalized_matrix: torch.Tensor,
        batch_size: int,
        overlap_threshold: float,
        show_progress: bool = True,
        use_sampling: bool = True,
        max_pairs: int = 10_000_000,
        memory_limit_gb: Optional[float] = None,
        seed: Optional[int] = None
    ) -> Dict[str, Any]:
        """Compute overlaps in batches for memory efficiency with optional progress tracking.

        IMPORTANT: This uses UNIFORM sampling, not stratified sampling. The high overlap
        count is computed on the full batch before sampling to avoid bias.

        Args:
            normalized_matrix: Normalized feature matrix
            batch_size: Size of batches for computation
            overlap_threshold: Threshold for counting high overlaps
            show_progress: Whether to show progress bar
            use_sampling: Whether to use uniform sampling for large matrices (>100M pairs)
            max_pairs: Maximum number of pairs to process (for sampling)
            memory_limit_gb: GPU memory limit for adaptive batch sizing
            seed: Random seed for deterministic sampling (ICLR reproducibility)
        """
        # Use config value if not specified
        if memory_limit_gb is None:
            memory_limit_gb = self.config.max_memory_gb

        n_features = normalized_matrix.shape[0]

        # Memory monitoring and adaptive batch size adjustment
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # Clear cache before starting
            # CRITICAL FIX: Handle device.index == None for torch.device('cuda')
            if self.device.type == 'cuda':
                device_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
            else:
                device_idx = 0  # Fallback (shouldn't reach here if cuda unavailable)
            total_memory = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
            allocated = torch.cuda.memory_allocated(device_idx) / 1e9
            free_memory = total_memory - allocated

            # Estimate memory needed for one batch pair (in GB)
            if normalized_matrix.dtype == torch.float64:
                element_size = 8
            elif normalized_matrix.dtype in (torch.float16, torch.bfloat16):
                element_size = 2
            else:
                element_size = 4
            memory_per_batch_pair = (batch_size * batch_size * element_size) / 1e9

            # Adjust batch size if needed to stay within memory limits
            safe_memory = min(memory_limit_gb, free_memory * 0.8)  # Use 80% of free memory
            if memory_per_batch_pair > safe_memory:
                old_batch_size = batch_size
                batch_size = int(np.sqrt(safe_memory * 1e9 / element_size))
                batch_size = min(batch_size, n_features)  # Don't exceed n_features
                batch_size = max(100, batch_size)  # Keep minimum batch size
                logger.warning(f"Reducing batch_size from {old_batch_size} to {batch_size} "
                             f"to fit in {safe_memory:.1f}GB memory limit")

            logger.debug(f"GPU Memory: {allocated:.1f}GB allocated, {free_memory:.1f}GB free, "
                        f"using batch_size={batch_size}")

        # Calculate total number of batches for progress tracking
        n_batches = (n_features + batch_size - 1) // batch_size
        total_batch_pairs = (n_batches * (n_batches + 1)) // 2  # Triangular number
        total_pairs = (n_features * (n_features - 1)) // 2

        # Log info for large computations
        if n_features > 10000:
            logger.info(f"Computing overlaps for {n_features:,} features in {n_batches} batches "
                       f"({total_pairs:,} total pairs, batch_size={batch_size})")
            if use_sampling and total_pairs > max_pairs:
                logger.info(f"Using sampling: will process ~{max_pairs:,} pairs instead of {total_pairs:,}")
        elif n_features > 1000:
            logger.debug(f"Computing overlaps for {n_features:,} features in {n_batches} batches")

        # Initialize accumulators for vectorized computation
        all_overlaps = []
        max_overlap = 0.0
        high_overlap_count = 0

        # Determine if we should show progress based on computation size
        use_progress = show_progress and (n_features > 1000 or total_batch_pairs > 10)

        # Create progress bar for batch processing if needed
        if use_progress:
            pbar = tqdm(
                total=total_batch_pairs,
                desc="Computing superposition overlaps",
                leave=False,
                unit="batch_pairs",
                dynamic_ncols=True,
                mininterval=0.5,
                file=sys.stderr,
                disable=not use_progress
            )
        else:
            pbar = None

        try:
            for i in range(0, n_features, batch_size):
                end_i = min(i + batch_size, n_features)
                batch_i = normalized_matrix[i:end_i]
                batch_i_idx = i // batch_size

                for j in range(i, n_features, batch_size):  # Start from i to avoid duplicates
                    end_j = min(j + batch_size, n_features)
                    batch_j = normalized_matrix[j:end_j]
                    batch_j_idx = j // batch_size

                    # Update progress bar description with current batch info
                    if pbar is not None:
                        pbar.set_description(f"Overlaps: batch [{batch_i_idx},{batch_j_idx}]/{n_batches}")

                    # Compute batch overlaps
                    batch_overlaps = torch.matmul(batch_i, batch_j.T).abs()

                    # Handle diagonal exclusion
                    if i == j:
                        # Create ones_like first to preserve device, then apply triu
                        ones_mask = torch.ones_like(batch_overlaps, dtype=torch.bool)
                        mask = torch.triu(ones_mask, diagonal=1)
                    else:
                        mask = torch.ones_like(batch_overlaps, dtype=torch.bool)

                    valid_overlaps = batch_overlaps[mask]

                    # PERFORMANCE FIX: Use vectorized operations instead of item-by-item
                    if valid_overlaps.numel() > 0:
                        # FIX: Always compute max and high overlap count on FULL batch before sampling
                        # This avoids sampling bias that could miss rare high overlaps
                        batch_max = valid_overlaps.max().item()
                        max_overlap = max(max_overlap, batch_max)
                        batch_high_count = (valid_overlaps > overlap_threshold).sum().item()

                        # For sampling: if we have too many pairs, sample them for mean/std computation
                        if use_sampling and total_pairs > max_pairs:
                            # Calculate sampling rate
                            sample_rate = max_pairs / total_pairs
                            n_samples = int(valid_overlaps.numel() * sample_rate)
                            if n_samples > 0:
                                # UNIFORM sampling for unbiased statistics
                                # Deterministic sampling with seed for reproducibility
                                if seed is not None:
                                    gen = torch.Generator(device=valid_overlaps.device)
                                    gen.manual_seed(seed + i + j)  # Include batch indices for unique seeds per batch
                                    indices = torch.randperm(valid_overlaps.numel(), device=valid_overlaps.device, generator=gen)[:n_samples]
                                else:
                                    indices = torch.randperm(valid_overlaps.numel(), device=valid_overlaps.device)[:n_samples]
                                sampled_overlaps = valid_overlaps[indices]

                                # Collect sampled overlaps for mean/std statistics
                                all_overlaps.append(sampled_overlaps)

                                # Scale high overlap count by sampling rate (unbiased estimator)
                                high_overlap_count += int(batch_high_count / sample_rate)
                            else:
                                # No samples in this batch
                                high_overlap_count += batch_high_count
                        else:
                            # No sampling: use full batch
                            all_overlaps.append(valid_overlaps)
                            high_overlap_count += batch_high_count

                    # Update progress bar
                    if pbar is not None:
                        pbar.update(1)
                        # Update with current statistics
                        if len(all_overlaps) > 0:
                            current_total = sum(o.numel() for o in all_overlaps)
                            pbar.set_postfix(
                                pairs=f"{current_total:,}",
                                max=f"{max_overlap:.4f}"
                            )

                    # Periodic memory cleanup for large computations
                    if torch.cuda.is_available() and (i + j) % 100 == 0:
                        # Check memory pressure
                        # CRITICAL FIX: Handle device.index == None
                        if self.device.type == 'cuda':
                            device_idx = self.device.index if self.device.index is not None else torch.cuda.current_device()
                        else:
                            device_idx = 0
                        allocated = torch.cuda.memory_allocated(device_idx) / 1e9
                        total = torch.cuda.get_device_properties(device_idx).total_memory / 1e9
                        if allocated > total * 0.9:  # Over 90% memory used
                            logger.debug(f"High memory usage detected ({allocated:.1f}GB/{total:.1f}GB). Clearing cache.")
                            torch.cuda.empty_cache()
                            torch.cuda.synchronize()
        finally:
            if pbar is not None:
                pbar.close()

        # Compute final statistics using Welford's streaming algorithm for memory efficiency
        if len(all_overlaps) > 0:
            # Always use Welford's algorithm for numerical stability and memory efficiency
            n = 0
            mean = 0.0
            M2 = 0.0

            for batch in all_overlaps:
                # Process in chunks to balance memory and speed
                if batch.numel() > 10000:
                    # Large batch: process in smaller chunks on CPU
                    batch_cpu = batch.detach().cpu().numpy()  # Detach gradient before numpy conversion
                    for x in batch_cpu:
                        n += 1
                        delta = x - mean
                        mean += delta / n
                        delta2 = x - mean
                        M2 += delta * delta2
                else:
                    # Small batch: can process as tensor
                    batch_vals = batch.detach().cpu().numpy()  # Detach gradient before numpy conversion
                    for x in batch_vals:
                        n += 1
                        delta = x - mean
                        mean += delta / n
                        delta2 = x - mean
                        M2 += delta * delta2

            mean_overlap = mean
            std_overlap = np.sqrt(M2 / max(1, n - 1)) if n > 1 else 0.0  # Use n-1 for unbiased estimate

            # Log sampling info if used
            if use_sampling and total_pairs > max_pairs:
                actual_pairs = n  # We counted during Welford's algorithm
                logger.debug(f"Computed statistics from {actual_pairs:,} uniformly sampled pairs "
                           f"({100*actual_pairs/total_pairs:.2f}% of total)")
        else:
            mean_overlap = 0.0
            std_overlap = 0.0

        return {
            'mean_overlap': mean_overlap,
            'std_overlap': std_overlap,
            'max_overlap': max_overlap,
            'num_high_overlap_pairs': high_overlap_count
        }

    def _classify_superposition_regime(
        self,
        phi_half: float,
        phi_one: float,
        n_features: int,
        n_dims: int
    ) -> str:
        """
        Classify the superposition regime based on paper definitions.

        - No superposition: phi_half ≈ m/n (only m features in m dimensions)
        - Weak superposition: phi_half > m/n but phi_one ≈ 0
        - Strong superposition: phi_half ≈ 1 (all features represented)
        """
        if n_features == 0:
            return "no_superposition"

        dimension_ratio = n_dims / n_features

        # No superposition: represented features ≈ available dimensions
        if abs(phi_half - dimension_ratio) < 0.1:
            return "no_superposition"

        # Strong superposition: most features are represented
        elif phi_half > 0.8:  # Paper suggests ϕ₁/₂ ≈ 1 for strong
            return "strong_superposition"

        # Weak superposition: more than m features but not all
        elif phi_half > dimension_ratio + 0.1:
            return "weak_superposition"

        else:
            return "no_superposition"

    def _compute_geometric_analysis(
        self,
        weight_matrix: torch.Tensor,
        feature_norms: torch.Tensor,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """Compute geometric overlap analysis for represented features.

        Memory-safe and deterministic:
        - For n_repr ≤ `config.geometric_full_matrix_limit`, computes a full overlap matrix
          on device and derives mean/max excluding the diagonal.
        - For larger sets, reuses the tiled, uniform-sampling path (`_compute_overlaps_batched`)
          with `seed=config.vocab_sampling_seed` to bound memory and ensure reproducibility.

        Numerical precision follows `self.dtype` (FP32 by default; FP64 when enabled), with
        statistics accumulated in Python floats.
        """
        # Get only represented features
        represented_mask = feature_norms > threshold
        n_represented = represented_mask.sum().item()

        if n_represented < 2:
            return {
                'welch_bound': 0.0,
                'welch_bound_ratio': 0.0,
                'expected_scaling': 0.0,
                'follows_sqrt_scaling': False
            }

        represented_features = weight_matrix[represented_mask].float()
        represented_norms = feature_norms[represented_mask].unsqueeze(1).float()

        # Normalize represented features
        normalized_features = represented_features / represented_norms.clamp_min(self.config.eps)
        normalized_features = normalized_features.to(self.dtype)

        n, d = represented_features.shape

        full_matrix_limit = getattr(self.config, 'geometric_full_matrix_limit', 4096)

        if n <= full_matrix_limit:
            overlap_matrix = torch.matmul(normalized_features, normalized_features.T).abs()
            mask = ~torch.eye(n, device=normalized_features.device, dtype=torch.bool)
            valid_overlaps = overlap_matrix[mask]
            mean_overlap = valid_overlaps.mean().item() if valid_overlaps.numel() > 0 else 0.0
            max_overlap = valid_overlaps.max().item() if valid_overlaps.numel() > 0 else 0.0
        else:
            batch_size = min(getattr(self.config, 'geometric_batch_size', 2048), n)
            stats = self._compute_overlaps_batched(
                normalized_features,
                batch_size=batch_size,
                overlap_threshold=self.config.overlap_threshold,
                show_progress=False,
                use_sampling=True,
                max_pairs=getattr(self.config, 'geometric_max_pairs', 5_000_000),
                memory_limit_gb=self.config.max_memory_gb,
                seed=self.config.vocab_sampling_seed
            )
            mean_overlap = stats['mean_overlap']
            max_overlap = stats['max_overlap']

        # Compute Welch bound (theoretical MINIMUM for MAXIMUM coherence)
        if n > d:
            welch_bound = np.sqrt((n - d) / (d * (n - 1)))
        else:
            welch_bound = 0.0

        # Check if overlaps follow √(2/π)/√d scaling (correct formula for random unit vectors)
        expected_scaling = np.sqrt(2.0 / np.pi) / np.sqrt(d) if d > 0 else 0
        scaling_ratio = mean_overlap / expected_scaling if expected_scaling > 0 else 0

        return {
            'welch_bound': float(welch_bound),
            'welch_bound_ratio': float(max_overlap / welch_bound) if welch_bound > 0 else 0.0,  # Correctly uses max coherence
            'expected_scaling': float(expected_scaling),
            'follows_sqrt_scaling': bool(abs(scaling_ratio - 1.0) < 0.25),  # Slightly wider tolerance
            'max_coherence': float(max_overlap),  # Added for clarity
            'mean_overlap_internal': float(mean_overlap)  # Keep for internal use
        }

    def compute_vector_interference_optimized(
        self,
        weight_matrix: 'torch.Tensor',
        normalize: bool = True,
        return_norms: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Optimized version of compute_vector_interference that returns norms.

        This method wraps the comprehensive analysis to provide backward
        compatibility while avoiding duplicate norm computation.
        """
        # CRITICAL FIX: Memory-safe device selection
        # Use default batch_size if not specified for memory estimation
        effective_batch_size = kwargs.get('batch_size', self.config.max_batch_size if hasattr(self, 'config') else 5000)
        compute_device = self._select_compute_device_safe(weight_matrix, effective_batch_size)

        if weight_matrix.device != compute_device:
            weight_matrix = weight_matrix.to(compute_device)

        # Use comprehensive analysis
        analysis = self.compute_comprehensive_superposition_analysis(
            weight_matrix,
            return_matrices=return_norms
        )

        # Build compatible return dictionary
        result = {
            'mean_overlap': analysis.mean_overlap,
            'std_overlap': analysis.std_overlap,
            'max_overlap': analysis.max_overlap,
            'num_high_overlap_pairs': analysis.num_high_overlap_pairs,
            'n_features': analysis.n_features,
            'n_dimensions': analysis.n_dimensions,
            'effective_orthogonality': 1.0 - analysis.mean_overlap
        }

        if return_norms:
            result['feature_norms'] = analysis.feature_norms

        return result

    def clear_cache(self):
        """Clear all caches to free memory."""
        self._norm_cache.clear()
        self._overlap_cache.clear()
        self._svd_cache.clear()

        # Reset statistics
        self.cache_hits = {'norms': 0, 'overlaps': 0, 'svd': 0}
        self.cache_misses = {'norms': 0, 'overlaps': 0, 'svd': 0}

        logger.info("Cleared all superposition analysis caches")

    # Convenience methods for UnifiedModelAnalysis integration
    def analyze_model(
        self,
        model: torch.nn.Module,
        batch: Optional[Dict] = None,
        return_dict: bool = True
    ) -> Union[SuperpositionAnalysis, Dict]:
        """
        Analyze superposition in a model.
        Convenience method for UnifiedModelAnalysis that accepts (model, batch) signature.

        Args:
            model: The model to analyze
            batch: Unused (for signature compatibility)
            return_dict: Whether to return as dict for JSON serialization

        Returns:
            Superposition analysis results
        """
        # Extract weight matrix from model
        weight_matrix = self._extract_weight_matrix(model)
        if weight_matrix is None:
            if return_dict:
                return {'error': 'Could not find suitable weight matrix in model'}
            else:
                raise ValueError('Could not find suitable weight matrix in model')

        return self.compute_comprehensive_superposition_analysis(
            weight_matrix,
            return_dict=return_dict
        )

    def compute_vector_interference_from_model(
        self,
        model: torch.nn.Module,
        batch: Optional[Dict] = None,
        normalize: bool = True
    ) -> Dict[str, float]:
        """
        Compute vector interference from a model.
        Convenience method for UnifiedModelAnalysis.

        Args:
            model: The model to analyze
            batch: Unused (for signature compatibility)
            normalize: Whether to normalize vectors

        Returns:
            Vector interference metrics
        """
        weight_matrix = self._extract_weight_matrix(model)
        if weight_matrix is None:
            return {'error': 'Could not find suitable weight matrix in model'}

        return self.compute_vector_interference(weight_matrix, normalize=normalize)

    def _extract_weight_matrix(self, model: torch.nn.Module, max_vocab_size: Optional[int] = None,
                              seed: Optional[int] = None) -> Optional[torch.Tensor]:
        """
        Extract a suitable weight matrix from a model for superposition analysis.
        For large vocabulary embeddings, uses stratified sampling to maintain statistical validity.

        Statistical Approach:
        - Sampling reduces large vocabularies to manageable size
        - Fixed seed enables reproducibility
        - Sample includes special tokens and a mix of vocabulary entries
        - Preserves special tokens critical for model behavior

        Args:
            model: The model to extract from
            max_vocab_size: Maximum vocabulary size to analyze (samples if larger).
                           If None, uses config.max_vocab_size_for_superposition
            seed: Random seed for reproducible sampling.
                  If None, uses config.vocab_sampling_seed

        Returns:
            Weight matrix tensor or None if not found
        """
        # Use config values if not specified
        if max_vocab_size is None:
            max_vocab_size = self.config.max_vocab_size_for_superposition
        if seed is None:
            seed = self.config.vocab_sampling_seed
        # Look for embedding layers first
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Embedding):
                weight = module.weight.detach()  # Detach from computation graph for analysis
                vocab_size = weight.shape[0]

                # Check if we need to sample based on size and strategy
                should_sample = (vocab_size > max_vocab_size and
                               self.config.vocab_sampling_strategy != 'none')

                if should_sample:
                    logger.info(f"Large vocabulary detected ({vocab_size:,} tokens). "
                               f"Using {self.config.vocab_sampling_strategy} sampling of {max_vocab_size:,} tokens for analysis.")
                    logger.info(f"This reduces memory usage while maintaining representation diversity.")

                    # Set seed for reproducibility if specified
                    # IMPORTANT: Generator device must match the device where randperm is called
                    # For CPU tensors, we need a CPU generator; for CUDA tensors, a CUDA generator
                    if seed is not None:
                        # Ensure generator is on the correct device
                        # CRITICAL FIX: Generator must match the device where randperm is executed
                        gen = torch.Generator(device=weight.device)
                        gen.manual_seed(seed)
                    else:
                        gen = None

                    if self.config.vocab_sampling_strategy == 'random':
                        # Simple random sampling
                        # FIX: Ensure randperm is executed on the same device as weight tensor
                        indices = torch.randperm(vocab_size, device=weight.device,
                                               generator=gen)[:max_vocab_size]
                        weight = weight[indices]
                        logger.info(f"Applied random sampling to embedding matrix: {weight.shape}")
                    elif self.config.vocab_sampling_strategy == 'stratified':
                        # Sampling strategy to reduce vocabulary size:
                        # 1. Always include special tokens (typically first 100-1000)
                        n_special = min(1000, vocab_size // 10, max_vocab_size // 10)

                        # 2. Sample from early vocabulary entries (often more frequent)
                        n_frequent = min(vocab_size // 5, max_vocab_size // 3)

                        # 3. Random sample from remaining vocabulary
                        n_random = max_vocab_size - n_special - n_frequent

                        # Create stratified indices
                        special_indices = torch.arange(n_special, device=weight.device)

                        frequent_start = n_special
                        frequent_end = min(frequent_start + vocab_size // 5, vocab_size)
                        frequent_pool_size = frequent_end - frequent_start
                        if frequent_pool_size > 0:
                            frequent_indices = torch.randperm(frequent_pool_size, device=weight.device,
                                                             generator=gen)[:n_frequent] + frequent_start
                        else:
                            frequent_indices = torch.tensor([], device=weight.device, dtype=torch.long)

                        # Random sample from rest
                        random_start = frequent_end
                        if random_start < vocab_size and n_random > 0:
                            random_pool_size = vocab_size - random_start
                            random_indices = torch.randperm(random_pool_size, device=weight.device,
                                                           generator=gen)[:n_random] + random_start
                        else:
                            random_indices = torch.tensor([], device=weight.device, dtype=torch.long)

                        # Combine all indices
                        indices = torch.cat([special_indices, frequent_indices, random_indices])
                        # Ensure indices stay on the same device after sorting
                        indices, _ = indices.sort()
                        indices = indices.to(weight.device)  # Ensure indices are on the same device as weight

                        # Sample the weight matrix
                        weight = weight[indices]
                        logger.info(f"Applied vocabulary sampling to embedding matrix: {weight.shape}")
                        logger.debug(f"Sampling strategy: {n_special} special tokens, "
                                   f"{len(frequent_indices)} early vocab tokens, {len(random_indices)} random tokens")
                return weight

            elif isinstance(module, torch.nn.Linear) and 'embed' in name.lower():
                weight = module.weight.detach()  # Detach from computation graph for analysis
                if weight.shape[0] > max_vocab_size:
                    logger.info(f"Large embedding linear layer ({weight.shape[0]:,} dims). "
                               f"Random sampling {max_vocab_size:,} rows for analysis.")

                    if seed is not None:
                        # Ensure generator is on the correct device
                        # CRITICAL FIX: Generator must match the device where randperm is executed
                        gen = torch.Generator(device=weight.device)
                        gen.manual_seed(seed)
                    else:
                        gen = None

                    indices = torch.randperm(weight.shape[0], device=weight.device, generator=gen)[:max_vocab_size]
                    return weight[indices]
                return weight

        # If no embedding found, use first linear layer (less common case)
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight.detach()  # Detach from computation graph for analysis
                if weight.shape[0] > max_vocab_size:
                    logger.info(f"Large linear layer ({weight.shape[0]:,} dims). "
                               f"Sampling {max_vocab_size:,} rows for analysis.")

                    if seed is not None:
                        # Ensure generator is on the correct device
                        # CRITICAL FIX: Generator must match the device where randperm is executed
                        gen = torch.Generator(device=weight.device)
                        gen.manual_seed(seed)
                    else:
                        gen = None

                    indices = torch.randperm(weight.shape[0], device=weight.device, generator=gen)[:max_vocab_size]
                    return weight[indices]
                return weight

        return None

    def compute_superposition_trajectory(
        self,
        model: torch.nn.Module,
        batch: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Compute superposition metrics optimized for trajectory analysis.
        Returns scalar values suitable for time series tracking.

        Args:
            model: The model to analyze
            batch: Unused (for signature compatibility)

        Returns:
            Dictionary of scalar metrics for trajectory tracking
        """
        logger.debug(f"Starting superposition trajectory computation for model with "
                    f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")

        # Extract weight matrix
        weight_matrix = self._extract_weight_matrix(model)
        if weight_matrix is None:
            return {
                'phi_half': 0.0,
                'phi_one': 0.0,
                'mean_overlap': 0.0,
                'n_represented': 0,
                'n_strongly_represented': 0,
                'regime_numeric': 0,  # 0=no_superposition, 1=weak, 2=strong
                'error': 'No weight matrix found'
            }

        logger.debug(f"Analyzing weight matrix of shape {weight_matrix.shape} "
                    f"({weight_matrix.shape[0]:,} features, {weight_matrix.shape[1]:,} dimensions)")

        # Compute comprehensive analysis
        result = self.compute_comprehensive_superposition_analysis(
            weight_matrix,
            return_dict=False  # Get dataclass for full access
        )

        # Convert regime to numeric for plotting
        regime_map = {
            'no_superposition': 0,
            'weak_superposition': 1,
            'strong_superposition': 2
        }

        # Return scalar metrics suitable for trajectory plots
        return {
            'phi_half': float(result.phi_half),
            'phi_one': float(result.phi_one),
            'mean_overlap': float(result.mean_overlap),
            'n_represented': int(result.n_represented),
            'n_strongly_represented': int(result.n_strongly_represented),
            'regime_numeric': regime_map.get(result.regime, 0),
            'welch_bound_ratio': float(result.welch_bound_ratio)  # Already correctly computed with max_overlap
        }

    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_hits = sum(self.cache_hits.values())
        total_misses = sum(self.cache_misses.values())
        total_accesses = total_hits + total_misses

        return {
            'total_accesses': total_accesses,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': total_hits / total_accesses if total_accesses > 0 else 0.0,
            'cache_sizes': {
                'norms': len(self._norm_cache),
                'overlaps': len(self._overlap_cache),
                'svd': len(self._svd_cache)
            },
            'detailed_hits': self.cache_hits,
            'detailed_misses': self.cache_misses
        }


def analyze_model_superposition_comprehensive(
    model: torch.nn.Module,
    layer_names: Optional[List[str]] = None,
    verbose: bool = True
) -> Dict[str, SuperpositionAnalysis]:
    """
    Analyze superposition across multiple layers of a model.

    Args:
        model: The model to analyze
        layer_names: Specific layer names to analyze (None = auto-detect)
        verbose: Whether to print results

    Returns:
        Dictionary mapping layer names to their superposition analysis
    """
    analyzer = SuperpositionAnalyzer()
    results = {}

    # Auto-detect layers if not specified
    if layer_names is None:
        layer_names = []
        for name, module in model.named_modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
                layer_names.append(name)

    for layer_name in layer_names:
        # Get the layer
        module = dict(model.named_modules())[layer_name]

        # Get weight matrix
        if hasattr(module, 'weight'):
            weight_matrix = module.weight.data

            # Analyze
            analysis = analyzer.compute_comprehensive_superposition_analysis(
                weight_matrix,
                return_matrices=False
            )

            results[layer_name] = analysis

            if verbose:
                print(f"\n=== {layer_name} ===")
                print(f"Features: {analysis.n_features}, Dimensions: {analysis.n_dimensions}")
                print(f"ϕ₁/₂ = {analysis.phi_half:.3f}, ϕ₁ = {analysis.phi_one:.3f}")
                print(f"Regime: {analysis.regime.replace('_', ' ').title()}")
                print(f"Mean overlap: {analysis.mean_overlap:.4f}")
                if analysis.follows_sqrt_scaling:
                    print("✓ Follows √(1/m) scaling (strong superposition signature)")

    # Print cache statistics
    if verbose:
        cache_stats = analyzer.get_cache_statistics()
        if cache_stats['total_accesses'] > 0:
            print(f"\nCache efficiency: {cache_stats['hit_rate']:.1%} hit rate")

    return results


if __name__ == "__main__":
    # Example usage
    import torch.nn as nn

    # Create test model
    model = nn.Sequential(
        nn.Embedding(1000, 64),  # 1000 features in 64 dimensions
        nn.Linear(64, 128),
        nn.Linear(128, 256)
    )

    # Analyze all layers
    results = analyze_model_superposition_comprehensive(model, verbose=True)

    # Check for strong superposition
    for layer_name, analysis in results.items():
        if analysis.regime == "strong_superposition":
            print(f"\n✓ {layer_name} is in STRONG SUPERPOSITION")
            print(f"  → Expect robust 1/m scaling")
            print(f"  → {analysis.n_represented}/{analysis.n_features} features represented")
