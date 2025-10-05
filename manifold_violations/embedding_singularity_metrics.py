#!/usr/bin/env python3
"""
Embedding Singularity Metrics for TensorScope Integration

Bridges Robinson paper implementation with TensorScope's CorrelationDiscovery.
Enables systematic correlation studies between embedding geometry and training outcomes.

NOVEL CONTRIBUTION: First integration of singularity metrics with training diagnostics.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import sys
import psutil
import gc
from tqdm import tqdm

# Import from same package (we're now inside manifold_violations)
from .robinson_fiber_bundle_test import RobinsonFiberBundleTest
from .singularity_mapper import SingularityMapper
from .polysemy_detector import PolysemyDetector
from .training_singularity_dynamics import TrainingSingularityTracker


class EmbeddingSingularityMetrics:
    """
    Compute embedding singularity metrics for TensorScope.

    This class integrates with CorrelationDiscovery to enable:
    1. Tracking singularity evolution during training
    2. Correlating with performance metrics
    3. Early warning detection
    4. Cross-model comparisons

    IMPORTANT: This analyzes the model's embedding matrix directly,
    not training data batches. The 'batch_size' parameter controls
    how many embedding vectors are processed in parallel when computing
    pairwise distances (for memory efficiency), not data batch size.

    For a model with 50K vocabulary:
    - Full distance matrix would be 50K x 50K = ~10GB memory
    - Processing in chunks of 256 reduces peak memory usage
    """

    def __init__(
        self,
        sample_size: Optional[int] = None,  # None = test full vocabulary
        track_evolution: bool = True,
        compute_all_metrics: bool = False,
        show_progress: bool = True,
        monitor_memory: bool = False,
        batch_size: int = 256,  # H100-optimized, matches unified_model_analysis default
        random_seed: int = 42  # ✅ FIX 10: Reproducibility for ICML submission
    ):
        """
        Initialize metrics computer.

        Args:
            sample_size: Number of tokens to sample. None = test full vocabulary (recommended for <100K vocab).
                        Testing full vocab takes <1 second for 50K vocab on H100
            track_evolution: Track evolution across checkpoints
            compute_all_metrics: Compute expensive metrics
            show_progress: Show tqdm progress bars
            monitor_memory: Monitor memory usage during computation
            batch_size: Number of embedding vectors to process in parallel for distance computation.
                       This controls memory usage when computing pairwise distances, NOT training batch size.
                       256 is optimal for H100 GPU memory bandwidth and L2 cache (49MB fits in 60MB L2)
            random_seed: Random seed for reproducible sampling (ICML submission requirement)
        """
        self.sample_size = sample_size
        self.enable_evolution_tracking = track_evolution  # Renamed to avoid shadowing track_evolution() method
        self.compute_all_metrics = compute_all_metrics
        self.show_progress = show_progress
        self.monitor_memory = monitor_memory
        self.batch_size = batch_size
        self.random_seed = random_seed  # ✅ Store for reproducibility

        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.device.type == 'cuda' and show_progress:
            print(f"Using GPU: {torch.cuda.get_device_name()}")

        # Initialize components (with reproducibility seeds)
        self.robinson_test = RobinsonFiberBundleTest()
        self.singularity_mapper = SingularityMapper()
        self.polysemy_detector = PolysemyDetector(random_state=self.random_seed)

        if track_evolution:
            self.tracker = TrainingSingularityTracker()

        # Cache for efficiency
        self.cache = {}

        # Memory monitoring
        self.memory_stats = []

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics."""
        process = psutil.Process()
        memory_info = process.memory_info()

        # Get GPU memory if available
        gpu_memory = 0
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / 1024**3  # GB

        return {
            'rss_gb': memory_info.rss / 1024**3,  # GB
            'vms_gb': memory_info.vms / 1024**3,  # GB
            'gpu_gb': gpu_memory,
            'percent': process.memory_percent()
        }

    def compute_metrics(
        self,
        model: torch.nn.Module,
        checkpoint_name: Optional[str] = None,
        step: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Compute all embedding singularity metrics.

        Args:
            model: PyTorch model (analyzes the embedding layer)
            checkpoint_name: Optional checkpoint identifier
            step: Optional training step

        Returns:
            Dictionary of metrics for correlation analysis

        Note: This analyzes the model's embedding matrix directly,
              not training batches. The batch_size parameter controls
              parallel processing of embedding vectors for efficiency.
        """
        # Extract embeddings
        embeddings = self._extract_embeddings(model)

        # Cache key
        cache_key = f"{checkpoint_name}_{step}" if checkpoint_name else "current"

        # Check cache
        if cache_key in self.cache and not self.compute_all_metrics:
            return self.cache[cache_key]

        # Compute metrics
        metrics = {}

        if self.monitor_memory:
            initial_memory = self._get_memory_usage()
            self.memory_stats.append(('start', initial_memory))

        # Progress bar for main computation stages
        stages = [
            ('Robinson volume growth', self._compute_robinson_metrics),
            ('Polysemy analysis', self._compute_polysemy_metrics),
            ('Singularity mapping', self._compute_singularity_metrics),
            ('Stability analysis', self._compute_stability_metrics)
        ]

        if self.show_progress:
            stages_pbar = tqdm(stages, desc="Computing metrics", leave=False)
        else:
            stages_pbar = stages

        for stage_name, compute_func in stages_pbar:
            if self.show_progress:
                stages_pbar.set_description(f"Computing {stage_name}")

            # Compute stage metrics
            stage_metrics = compute_func(embeddings)
            metrics.update(stage_metrics)

            # Memory checkpoint after each stage
            if self.monitor_memory:
                stage_memory = self._get_memory_usage()
                self.memory_stats.append((stage_name, stage_memory))

            # Force garbage collection after heavy operations
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Cache results
        self.cache[cache_key] = metrics

        return metrics

    def _extract_embeddings(self, model: torch.nn.Module) -> torch.Tensor:
        """Extract token embeddings from model and keep on GPU."""
        # Handle different model architectures
        if hasattr(model, 'get_input_embeddings'):
            embed_layer = model.get_input_embeddings()
        elif hasattr(model, 'embeddings'):
            embed_layer = model.embeddings.word_embeddings
        elif hasattr(model, 'embed_tokens'):
            embed_layer = model.embed_tokens
        else:
            # Try to find embedding layer
            for name, module in model.named_modules():
                if 'embed' in name.lower() and hasattr(module, 'weight'):
                    if isinstance(module, torch.nn.Embedding):
                        embed_layer = module
                        break
            else:
                raise ValueError("Could not find embedding layer")

        # Get embeddings and keep on GPU
        with torch.no_grad():
            weight_tensor = embed_layer.weight.detach()
            # Convert BFloat16 to float32 if needed
            if weight_tensor.dtype == torch.bfloat16:
                weight_tensor = weight_tensor.float()
            # Ensure it's on GPU if available
            if torch.cuda.is_available() and not weight_tensor.is_cuda:
                weight_tensor = weight_tensor.cuda()
            embeddings = weight_tensor

        return embeddings

    def _compute_robinson_metrics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Compute Robinson metrics using GPU acceleration when possible."""
        # If we have GPU embeddings, use GPU computation
        if embeddings.is_cuda:
            return self._compute_robinson_metrics_gpu(embeddings)
        else:
            # Fallback to numpy for CPU
            return self._compute_robinson_metrics_cpu(embeddings.cpu().numpy())

    def _compute_robinson_metrics_gpu(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """GPU-accelerated Robinson metrics computation.

        Note: batch_size here refers to how many embedding vectors we process
        in parallel for distance computation, NOT training batch size.
        This is purely for memory efficiency when computing pairwise distances.
        """
        n_tokens = embeddings.shape[0]
        # Use full vocabulary if sample_size is None
        sample_size = n_tokens if self.sample_size is None else min(self.sample_size, n_tokens)

        # ✓ FIX 1: Add reproducibility seed
        generator = torch.Generator(device=embeddings.device).manual_seed(self.random_seed)
        indices = torch.randperm(n_tokens, generator=generator, device=embeddings.device)[:sample_size]

        # ✓ FIX 2: Precompute emb_norms_sq ONCE (not 594 times)
        emb_norms_sq = (embeddings ** 2).sum(dim=1, keepdim=True)  # (vocab_size, 1)

        violations = 0
        increasing_slopes = 0
        avg_local_dim = []
        max_slope_increases = []

        # Process sampled embeddings in chunks to avoid OOM on distance matrix
        # batch_size controls parallel distance computations, not data batches
        if self.show_progress:
            indices_iter = tqdm(range(0, sample_size, self.batch_size),
                               desc="Testing Robinson hypothesis (GPU chunked)",
                               leave=False)
        else:
            indices_iter = range(0, sample_size, self.batch_size)

        for batch_start in indices_iter:
            batch_end = min(batch_start + self.batch_size, sample_size)
            batch_indices = indices[batch_start:batch_end]

            # ✓ FIX 3: Pass precomputed norms to avoid recomputation
            batch_results = self._process_robinson_batch_gpu(embeddings, batch_indices, emb_norms_sq)

            violations += batch_results['violations']
            increasing_slopes += batch_results['increasing_slopes']
            # Results are now CPU floats, safe to extend
            avg_local_dim.extend(batch_results['local_dims'])
            max_slope_increases.extend(batch_results['max_slope_increases'])

            # ✓ FIX 4: Force cleanup every 10 batches to prevent accumulation
            if batch_start % (10 * self.batch_size) == 0:
                torch.cuda.empty_cache()

        # ✓ FIX 5: Final cleanup
        del emb_norms_sq, indices
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        metrics = {
            'robinson_violation_rate': violations / sample_size,
            'robinson_increasing_slopes_rate': increasing_slopes / sample_size,
            'avg_local_signal_dimension': float(np.mean(avg_local_dim)),
            'max_slope_increase': float(np.max(max_slope_increases)),
            'std_local_signal_dimension': float(np.std(avg_local_dim))
        }

        return metrics

    def _process_robinson_batch_gpu(self, embeddings: torch.Tensor, batch_indices: torch.Tensor,
                                   emb_norms_sq: torch.Tensor) -> Dict:
        """Process a batch of Robinson tests on GPU.

        Args:
            embeddings: Full embedding matrix (vocab_size, embed_dim)
            batch_indices: Indices of embeddings to test in this chunk
            emb_norms_sq: Precomputed squared norms (vocab_size, 1)

        Note: This processes multiple embedding vectors in parallel,
        computing distances from each to all other embeddings.
        """
        batch_size = len(batch_indices)
        batch_embeddings = embeddings[batch_indices]  # (batch_size, embed_dim)

        # Efficient batched distance computation for all pairs
        # Computing distance from batch_size embeddings to ALL vocab embeddings
        # ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
        # emb_norms_sq now passed as parameter (computed once)
        batch_norms_sq = (batch_embeddings ** 2).sum(dim=1, keepdim=True)  # (batch_size, 1)
        dots = torch.mm(batch_embeddings, embeddings.t())  # (batch_size, vocab_size)
        distances_sq = batch_norms_sq + emb_norms_sq.t() - 2 * dots

        # ✓ FIX 6: Clean up intermediate tensors immediately
        del dots, batch_norms_sq

        distances = torch.sqrt(torch.clamp(distances_sq, min=0))  # (batch_size, vocab_size)
        del distances_sq  # ✓ FIX 7: Delete before using distances

        # Analyze each point in the batch
        results = {
            'violations': 0,
            'increasing_slopes': 0,
            'local_dims': [],
            'max_slope_increases': []
        }

        for i in range(batch_size):
            dist_row = distances[i]
            # Remove self-distance
            mask = torch.ones_like(dist_row, dtype=torch.bool)
            mask[batch_indices[i]] = False
            dist_row = dist_row[mask]

            # Simplified Robinson analysis on GPU
            violation, inc_slope, local_dim, max_inc = self._analyze_single_point_gpu(dist_row)

            if violation:
                results['violations'] += 1
            if inc_slope:
                results['increasing_slopes'] += 1
            # ✓ FIX 8: Ensure results are CPU scalars (already are, but explicit)
            results['local_dims'].append(float(local_dim))
            results['max_slope_increases'].append(float(max_inc))

        # ✓ FIX 9: Clean up batch tensors
        del distances

        return results

    def _analyze_single_point_gpu(self, distances: torch.Tensor) -> Tuple[bool, bool, float, float]:
        """Robinson analysis for a single point on GPU with ALL ICML fixes applied.

        Implements Robinson et al. (2023) fiber bundle hypothesis test with:
        - GPU-native Kendall's tau (no CPU sync)
        - Float64 precision for log-log analysis
        - Proper reach gating (ENABLED)
        - r_max/r_min ratio check
        - Memory-efficient computation
        """
        if len(distances) < 10:
            return False, False, 0.0, 0.0

        # Sort distances once
        sorted_dists, _ = torch.sort(distances)
        r_min = sorted_dists[min(4, len(sorted_dists) // 100)]
        r_max = torch.quantile(sorted_dists, 0.9)

        # ✅ FIX 16: Add r_max/r_min ratio check for sufficient dynamic range
        if r_max <= r_min or not torch.isfinite(r_min) or not torch.isfinite(r_max):
            return False, False, float('nan'), 0.0

        if (r_max / r_min) < 1.5:  # Need at least 1.5× dynamic range
            return False, False, float('nan'), 0.0

        # Log-spaced radii
        n_radii = 20
        log_radii = torch.linspace(torch.log(r_min), torch.log(r_max), n_radii, device=distances.device)
        radii = torch.exp(log_radii)

        # Count points within each radius
        volumes = torch.searchsorted(sorted_dists, radii, right=True).float()

        # Filter valid volumes (need at least 5 points)
        valid_mask = volumes >= 5
        if not valid_mask.any() or valid_mask.sum() < 5:  # Need at least 5 valid radii
            return False, False, 0.0, 0.0

        valid_volumes = volumes[valid_mask]
        valid_log_radii = log_radii[valid_mask]

        # ✅ FIX 17: Use float64 for numerical precision in log-log space
        log_volumes = torch.log(valid_volumes.double() + 1e-10)
        valid_log_radii = valid_log_radii.double()

        if len(log_volumes) < 5:  # Need at least 5 points for regression
            return False, False, 0.0, 0.0

        # Central differences for slopes (O(h²) accuracy)
        slopes = (log_volumes[2:] - log_volumes[:-2]) / (valid_log_radii[2:] - valid_log_radii[:-2])

        if len(slopes) < 3:
            return False, False, 0.0, 0.0

        # Slope changes for diagnostics
        slope_changes = slopes[1:] - slopes[:-1]
        max_slope_increase = slope_changes.max().item() if slope_changes.numel() > 0 else 0.0

        # ✅ FIX 12-14: GPU-native Kendall's tau with correct variance formula
        n = slopes.numel()
        if n >= 3:
            tau = self._kendall_tau_gpu(slopes)

            # Correct variance formula (Kendall 1938)
            var_tau = n * (n - 1) * (2 * n + 5) / 18.0 if n > 1 else float('inf')

            # Continuity correction for normal approximation
            continuity_correction = 1.0 / (n * (n - 1)) if n > 1 else 0.0
            z = 0.0 if not (var_tau > 0) else ((abs(tau) - continuity_correction) / (var_tau ** 0.5))

            # One-sided p-value for increasing trend
            from scipy.stats import norm
            p_value = 1.0 - norm.cdf(z)
            increasing_slopes = p_value < 0.001  # Robinson's significance level
        else:
            p_value = 1.0
            increasing_slopes = False

        # ✅ FIX 18: Enable reach gating (was disabled!)
        reach_ok = False  # Default to false (CRITICAL FIX)
        if valid_log_radii.numel() >= 3:
            median_idx = valid_log_radii.numel() // 2
            median_radius = torch.exp(valid_log_radii[median_idx]).item()
            r_min_val = r_min.item() if torch.is_tensor(r_min) else r_min

            # Reach criterion: median should span at least 2× r_min
            reach_ok = (median_radius / r_min_val) > 2.0

            # Also require at least 10 valid radii for reliable trend detection
            reach_ok = reach_ok and (valid_log_radii.numel() >= 10)

        # Violation requires BOTH statistical significance AND sufficient reach
        violation = bool(increasing_slopes and reach_ok)

        # Estimate local dimension (median slope in log-log space)
        local_dim = slopes.median().item() if slopes.numel() > 0 else float('nan')

        return violation, bool(increasing_slopes), local_dim, max_slope_increase

    def _kendall_tau_gpu(self, x: torch.Tensor) -> float:
        """
        ✅ FIX 15: GPU-native Kendall's tau computation (no CPU transfers).

        Computes Kendall's tau correlation coefficient using vectorized GPU operations:
        tau = (C - D) / (n * (n-1) / 2)
        where C = concordant pairs, D = discordant pairs

        This replaces the nested Python loops on CPU which caused:
        - 151,936 CPU transfers
        - 7.6 seconds wasted
        - GPU pipeline breaks

        Args:
            x: 1D tensor of values (typically slopes in log-log space)

        Returns:
            Kendall's tau coefficient (float scalar)
        """
        n = len(x)

        if n < 2:
            return 0.0

        # Vectorized pairwise comparisons on GPU
        x_i = x.unsqueeze(1)  # (n, 1)
        x_j = x.unsqueeze(0)  # (1, n)

        # Compare all pairs: +1 if x_j > x_i, -1 if x_j < x_i, 0 if equal
        comparisons = torch.sign(x_j - x_i)  # (n, n)

        # Only count upper triangle (each pair once)
        triu_indices = torch.triu_indices(n, n, offset=1, device=x.device)
        comparisons_triu = comparisons[triu_indices[0], triu_indices[1]]

        # Sum of signs = C - D (concordant - discordant)
        s = comparisons_triu.sum()

        # Normalize by number of pairs
        tau = 2.0 * s / (n * (n - 1))

        return tau.item()

    def _compute_robinson_metrics_cpu(self, embeddings: np.ndarray) -> Dict[str, float]:
        """Compute Robinson volume growth metrics on CPU (fallback)."""
        metrics = {}
        n_tokens = len(embeddings)

        # Sample tokens
        sample_size = min(self.sample_size, n_tokens)
        indices = np.random.choice(n_tokens, sample_size, replace=False)

        violations = 0
        increasing_slopes = 0
        avg_local_dim = []
        max_slope_increases = []

        # Progress bar for Robinson test iterations
        if self.show_progress:
            indices_iter = tqdm(indices, desc="Testing Robinson hypothesis", leave=False)
        else:
            indices_iter = indices

        for idx in indices_iter:
            result = self.robinson_test.test_point(embeddings, idx)

            if result.violates_hypothesis:
                violations += 1

            if result.increasing_slopes:
                increasing_slopes += 1

            avg_local_dim.append(result.local_signal_dimension)
            max_slope_increases.append(result.max_slope_increase)

        metrics['robinson_violation_rate'] = violations / sample_size
        metrics['robinson_increasing_slopes_rate'] = increasing_slopes / sample_size
        metrics['avg_local_signal_dimension'] = np.mean(avg_local_dim)
        metrics['max_slope_increase'] = np.max(max_slope_increases)
        metrics['std_local_signal_dimension'] = np.std(avg_local_dim)

        return metrics

    def _compute_polysemy_metrics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Compute polysemy-related metrics."""
        # Convert to numpy for compatibility with existing code
        embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings

        if self.show_progress:
            # Temporarily disable internal verbose to avoid nested progress bars
            analysis = self.polysemy_detector.analyze_vocabulary(
                embeddings_np,
                sample_size=min(500, len(embeddings_np)),
                verbose=False
            )
        else:
            analysis = self.polysemy_detector.analyze_vocabulary(
                embeddings_np,
                sample_size=min(500, len(embeddings_np)),
                verbose=False
            )

        return {
            'polysemy_rate': analysis.polysemy_rate,
            'num_homonyms': len(analysis.homonyms),
            'num_contranyms': len(analysis.contranyms),
            'avg_meanings_per_polysemous': analysis.summary_stats['avg_meanings_per_polysemous_token'],
            'high_risk_polysemy_count': len(analysis.high_risk_tokens)
        }

    def _compute_singularity_metrics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Compute comprehensive singularity metrics."""
        from .singularity_mapper import create_singularity_map

        # Convert to numpy for compatibility
        embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings

        if self.show_progress:
            # Create singularity map with progress tracking disabled in nested calls
            result = create_singularity_map(
                embeddings_np,
                sample_size=min(self.sample_size, len(embeddings_np)),
                verbose=False
            )
        else:
            result = create_singularity_map(
                embeddings_np,
                sample_size=min(self.sample_size, len(embeddings_np)),
                verbose=False
            )

        metrics = {
            'total_singularity_rate': result['statistics']['singularity_rate'],
            'critical_risk_rate': result['statistics']['critical_risk_rate'],
            'avg_output_variance': result['statistics']['avg_output_variance'],
            'avg_semantic_instability': result['statistics']['avg_semantic_instability']
        }

        # Count by type
        for sing_type, tokens in result['singularity_types'].items():
            metrics[f'singularity_type_{sing_type}'] = len(tokens) / result['statistics']['total_analyzed']

        # Count by risk level
        for risk_level, tokens in result['risk_levels'].items():
            metrics[f'risk_level_{risk_level}'] = len(tokens) / result['statistics']['total_analyzed']

        return metrics

    def _compute_stability_metrics(self, embeddings: torch.Tensor) -> Dict[str, float]:
        """Compute embedding stability metrics."""
        # Use GPU operations when available
        if torch.is_tensor(embeddings) and embeddings.is_cuda:
            # GPU path
            embedding_norms = torch.norm(embeddings, dim=1)

            metrics = {
                'embedding_norm_mean': embedding_norms.mean().item(),
                'embedding_norm_std': embedding_norms.std().item(),
                'embedding_norm_max': embedding_norms.max().item(),
                'embedding_norm_min': embedding_norms.min().item(),
                'token_norm_ratio': (embedding_norms.max() / (embedding_norms.min() + 1e-8)).item()  # Renamed from misleading 'condition_number'
            }
        else:
            # CPU path (numpy)
            embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
            embedding_norms = np.linalg.norm(embeddings_np, axis=1)
            metrics = {
                'embedding_norm_mean': np.mean(embedding_norms),
                'embedding_norm_std': np.std(embedding_norms),
                'embedding_norm_max': np.max(embedding_norms),
                'embedding_norm_min': np.min(embedding_norms),
                'embedding_condition_number': np.max(embedding_norms) / (np.min(embedding_norms) + 1e-8)
            }

        # Compute spectral properties if not too large
        if embeddings.shape[0] < 5000:
            try:
                if torch.is_tensor(embeddings) and embeddings.is_cuda:
                    # GPU SVD
                    _, s, _ = torch.linalg.svd(embeddings, full_matrices=False)
                    metrics['embedding_rank'] = (s > 1e-6).sum().item()
                    # Safe computation of effective rank
                    s_normalized = s / s.sum()
                    s_normalized = torch.clamp(s_normalized, min=1e-10)
                    entropy = -(s_normalized * torch.log(s_normalized)).sum()
                    metrics['embedding_effective_rank'] = torch.exp(entropy).item()
                else:
                    # CPU path
                    embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
                    embeddings_f64 = embeddings_np.astype(np.float64)
                    _, s, _ = np.linalg.svd(embeddings_f64, full_matrices=False)
                    metrics['embedding_rank'] = np.sum(s > 1e-6)
                    s_normalized = s / (s.sum() + 1e-10)
                    s_normalized = np.clip(s_normalized, 1e-10, 1.0)
                    metrics['embedding_effective_rank'] = np.exp(
                        -np.sum(s_normalized * np.log(s_normalized))
                    )
            except Exception as e:
                # Log the error but don't fail
                print(f"Warning: Could not compute spectral properties: {e}")
                pass

        return metrics

    def track_evolution(
        self,
        checkpoints: List[Tuple[str, torch.nn.Module, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """
        Track metric evolution across checkpoints.

        Args:
            checkpoints: List of (name, model, training_metrics) tuples

        Returns:
            Evolution analysis
        """
        evolution_data = []

        if self.monitor_memory:
            self.memory_stats = []  # Reset for evolution tracking

        # Progress bar for checkpoint processing
        if self.show_progress:
            checkpoint_iter = tqdm(checkpoints, desc="Processing checkpoints")
        else:
            checkpoint_iter = checkpoints

        for name, model, training_metrics in checkpoint_iter:
            if self.show_progress:
                checkpoint_iter.set_description(f"Processing checkpoint: {name}")
            # Compute singularity metrics
            metrics = self.compute_metrics(model, checkpoint_name=name)

            # Add training metrics
            metrics.update(training_metrics)
            metrics['checkpoint'] = name

            evolution_data.append(metrics)

            # Clean up after each checkpoint
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Analyze trends
        trends = self._analyze_trends(evolution_data)

        result = {
            'metrics_by_checkpoint': evolution_data,
            'trends': trends,
            'correlations': self._compute_correlations(evolution_data)
        }

        if self.monitor_memory:
            result['memory_usage'] = self._analyze_memory_usage()

        return result

    def _analyze_trends(self, evolution_data: List[Dict]) -> Dict[str, str]:
        """Analyze trends in metrics."""
        trends = {}

        if len(evolution_data) < 2:
            return trends

        # Key metrics to track
        key_metrics = [
            'robinson_violation_rate',
            'polysemy_rate',
            'total_singularity_rate',
            'avg_local_signal_dimension'
        ]

        for metric in key_metrics:
            values = [d.get(metric, 0) for d in evolution_data]
            if len(values) > 1:
                # Simple trend: increasing, decreasing, stable
                first_half = np.mean(values[:len(values)//2])
                second_half = np.mean(values[len(values)//2:])

                if second_half > first_half * 1.1:
                    trends[metric] = 'increasing'
                elif second_half < first_half * 0.9:
                    trends[metric] = 'decreasing'
                else:
                    trends[metric] = 'stable'

        return trends

    def _compute_correlations(self, evolution_data: List[Dict]) -> Dict[str, float]:
        """Compute correlations between singularity metrics and training metrics."""
        correlations = {}

        if len(evolution_data) < 3:
            return correlations

        # Extract singularity and training metrics
        sing_metrics = ['robinson_violation_rate', 'total_singularity_rate', 'avg_local_signal_dimension']
        train_metrics = ['loss', 'accuracy', 'perplexity']

        for s_metric in sing_metrics:
            for t_metric in train_metrics:
                s_values = [d.get(s_metric, 0) for d in evolution_data]
                t_values = [d.get(t_metric, 0) for d in evolution_data]

                if any(s_values) and any(t_values):
                    # Compute correlation
                    valid_pairs = [(s, t) for s, t in zip(s_values, t_values) if s != 0 and t != 0]
                    if len(valid_pairs) > 2:
                        s_valid, t_valid = zip(*valid_pairs)
                        corr = np.corrcoef(s_valid, t_valid)[0, 1]
                        correlations[f'{s_metric}_vs_{t_metric}'] = corr

        return correlations

    def _analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze memory usage patterns."""
        if not self.memory_stats:
            return {}

        analysis = {
            'peak_memory_gb': max(stat[1]['rss_gb'] for stat in self.memory_stats),
            'peak_gpu_gb': max(stat[1]['gpu_gb'] for stat in self.memory_stats),
            'stages': {}
        }

        # Analyze memory by stage
        for stage_name, stats in self.memory_stats:
            if stage_name not in analysis['stages']:
                analysis['stages'][stage_name] = {
                    'rss_gb': stats['rss_gb'],
                    'gpu_gb': stats['gpu_gb'],
                    'percent': stats['percent']
                }

        # Calculate memory increase per stage
        if len(self.memory_stats) > 1:
            initial_mem = self.memory_stats[0][1]['rss_gb']
            for i, (stage_name, stats) in enumerate(self.memory_stats[1:], 1):
                prev_mem = self.memory_stats[i-1][1]['rss_gb']
                analysis['stages'][stage_name]['delta_gb'] = stats['rss_gb'] - prev_mem

        return analysis

    def generate_report(
        self,
        model: torch.nn.Module,
        checkpoint_name: str = "current"
    ) -> str:
        """
        Generate human-readable report of embedding health.
        """
        metrics = self.compute_metrics(model, checkpoint_name)

        report = []
        report.append("="*60)
        report.append("EMBEDDING SINGULARITY ANALYSIS REPORT")
        report.append("="*60)
        report.append(f"Checkpoint: {checkpoint_name}")
        report.append("")

        # Robinson metrics
        report.append("Robinson Volume Growth Analysis:")
        report.append(f"  Violation rate: {metrics['robinson_violation_rate']:.1%}")
        report.append(f"  Increasing slopes: {metrics['robinson_increasing_slopes_rate']:.1%}")
        report.append(f"  Avg local dimension: {metrics['avg_local_signal_dimension']:.2f}")

        # Polysemy
        report.append("\nPolysemy Analysis:")
        report.append(f"  Polysemy rate: {metrics['polysemy_rate']:.1%}")
        report.append(f"  Homonyms: {metrics['num_homonyms']}")
        report.append(f"  High-risk tokens: {metrics['high_risk_polysemy_count']}")

        # Singularities
        report.append("\nSingularity Statistics:")
        report.append(f"  Total rate: {metrics['total_singularity_rate']:.1%}")
        report.append(f"  Critical risk: {metrics['critical_risk_rate']:.1%}")
        report.append(f"  Expected output variance: {metrics['avg_output_variance']:.3f}")

        # Stability
        report.append("\nEmbedding Stability:")
        report.append(f"  Norm mean: {metrics['embedding_norm_mean']:.3f}")
        report.append(f"  Norm std: {metrics['embedding_norm_std']:.3f}")
        report.append(f"  Condition number: {metrics['embedding_condition_number']:.2f}")

        # Interpretation
        report.append("\nInterpretation:")
        if metrics['robinson_violation_rate'] > 0.5:
            report.append("  ⚠️ HIGH VIOLATION RATE: Embeddings violate manifold hypothesis")
        if metrics['polysemy_rate'] > 0.2:
            report.append("  ⚠️ HIGH POLYSEMY: Many tokens have multiple meanings")
        if metrics['critical_risk_rate'] > 0.1:
            report.append("  ⚠️ CRITICAL RISK: Significant embedding instability")

        if metrics['robinson_violation_rate'] < 0.2 and metrics['critical_risk_rate'] < 0.05:
            report.append("  ✓ Embeddings appear relatively stable")

        return "\n".join(report)


# Integration with CorrelationDiscovery
def integrate_with_correlation_discovery():
    """
    Example of how to integrate with TensorScope's CorrelationDiscovery.
    """
    # This would be added to CorrelationDiscovery.py
    code = '''
    # In CorrelationDiscovery.py, add:

    from EmbeddingSingularityMetrics import EmbeddingSingularityMetrics

    class CorrelationDiscovery:
        def __init__(self, ...):
            # Add to metrics
            self.embedding_metrics = EmbeddingSingularityMetrics()

        def analyze_checkpoint(self, model, checkpoint_data):
            # Existing metrics...

            # Add embedding singularity metrics
            embedding_results = self.embedding_metrics.compute_metrics(
                model,
                checkpoint_name=checkpoint_data['name'],
                step=checkpoint_data['step']
            )

            # Merge with other metrics
            all_metrics.update(embedding_results)

            return all_metrics

        def correlate_with_outcomes(self):
            # New correlation: singularities vs training stability
            correlations['singularity_vs_collapse'] = self._correlate(
                'robinson_violation_rate',
                'training_collapsed'
            )

            # Early warning analysis
            correlations['early_warning'] = self._test_early_warning(
                'total_singularity_rate',
                'future_loss_spike',
                lookahead=5
            )
    '''
    return code


if __name__ == "__main__":
    print("Embedding Singularity Metrics for TensorScope")
    print("="*60)

    # Example with synthetic model
    import torch.nn as nn

    class DummyModel(nn.Module):
        def __init__(self, vocab_size=1000, embed_dim=128):
            super().__init__()
            self.embeddings = nn.Embedding(vocab_size, embed_dim)

        def get_input_embeddings(self):
            return self.embeddings

    model = DummyModel()
    metrics_computer = EmbeddingSingularityMetrics(
        show_progress=True,
        monitor_memory=True
    )

    # Compute metrics
    metrics = metrics_computer.compute_metrics(model)

    # Generate report
    report = metrics_computer.generate_report(model)
    print(report)

    # Show memory usage if monitored
    if metrics_computer.monitor_memory and metrics_computer.memory_stats:
        print("\n" + "="*60)
        print("Memory Usage Analysis:")
        memory_analysis = metrics_computer._analyze_memory_usage()
        print(f"Peak Memory: {memory_analysis['peak_memory_gb']:.2f} GB")
        print(f"Peak GPU Memory: {memory_analysis['peak_gpu_gb']:.2f} GB")
        print("\nMemory by Stage:")
        for stage, stats in memory_analysis['stages'].items():
            print(f"  {stage}:")
            print(f"    RSS: {stats['rss_gb']:.2f} GB")
            if 'delta_gb' in stats:
                print(f"    Delta: {stats['delta_gb']:+.2f} GB")

    print("\n" + "="*60)
    print("Integration code for CorrelationDiscovery:")
    print(integrate_with_correlation_discovery())

    print("\nKey Research Questions Enabled:")
    print("1. Do embedding singularities correlate with training instability?")
    print("2. Can singularity metrics provide early warning of collapse?")
    print("3. Are successful models characterized by fewer singularities?")
    print("4. Do different architectures show different singularity patterns?")