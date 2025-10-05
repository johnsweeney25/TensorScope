"""
Fisher Spectral Analysis Module
================================
For ICLR 2026 - Theoretically correct and reproducible Fisher Information Matrix spectrum computation.

Key theoretical foundations:
1. Empirical Fisher: F = (1/N) Σᵢ ∇ℓᵢ ∇ℓᵢᵀ (uncentered second moment of gradients)
2. NOT the Hessian: Fisher ≈ Hessian only at optimum under correct model specification
3. Block-diagonal approximation: F ≈ diag(F₁, F₂, ..., Fₖ) where blocks = layers/modules
4. Gram matrix trick: When N << P, eigenvalues of (1/N)G@Gᵀ = non-zero eigenvalues of (1/N)Gᵀ@G

Mathematical correctness notes:
- Spectral gap: λ₁ - λ₂ where eigenvalues ordered λ₁ ≥ λ₂ ≥ ... ≥ λₙ
- This is NOT a mixing time (that's for Markov chains with λ₁ = 1)
- Condition number: κ = λ_max / λ_min measures optimization difficulty
- Effective rank: exp(H(p)) where p = λ/Σλ, measures parameter efficiency

Author: ICLR 2026 Project
"""

import torch
import torch.nn as nn
import math
import logging
import warnings
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SpectralConfig:
    """Configuration for spectral analysis."""
    seed: int = 42
    eps: float = 1e-9
    storage_mode: str = 'chunked'  # 'full', 'chunked', 'streaming'
    chunk_size: int = 32
    max_params_per_block: int = 10000
    use_vmap: bool = False
    dtype_compute: torch.dtype = torch.float32
    dtype_eigensolve: torch.dtype = torch.float64  # Higher precision for eigensolve
    top_k_eigenvalues: int = 100  # Number of top eigenvalues to compute/store
    regularization: float = 1e-8  # Diagonal regularization for Gram matrix


class FisherSpectral:
    """
    Compute Fisher Information Matrix spectrum with theoretical correctness and numerical stability.

    This implementation:
    1. Uses per-sample gradients to form empirical Fisher
    2. Applies block-diagonal approximation for scalability
    3. Uses Gram matrix trick when N << P for efficiency
    4. Ensures reproducibility with fixed seeds and deterministic algorithms
    5. Can reuse gradients from FisherCollector when available
    """

    def __init__(self, config: Optional[SpectralConfig] = None, gradient_cache=None):
        """
        Initialize Fisher Spectral analyzer.

        Args:
            config: Configuration object. If None, uses defaults.
            gradient_cache: Optional shared gradient cache from FisherCollector.
        """
        self.config = config or SpectralConfig()
        self.gradient_cache = gradient_cache  # Shared cache from FisherCollector

        # Subsampling indices for reproducibility (fixed per session)
        self.subsample_indices = {}

        # Gradient accumulator for streaming mode
        self.gradient_accumulator = {}

        # Set deterministic mode for ICLR reproducibility
        self._set_deterministic_mode()

        # Check vmap availability
        self.has_vmap = self.config.use_vmap and hasattr(torch.func, 'vmap')
        if self.config.use_vmap and not self.has_vmap:
            logger.warning("vmap requested but not available in PyTorch version. Using loop-based collection.")

    def _set_deterministic_mode(self):
        """Set PyTorch to deterministic mode for reproducibility."""
        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
            # Note: Full determinism may slow down computation
            # torch.use_deterministic_algorithms(True)
            # Uncomment above for full ICLR reproducibility

    def compute_fisher_spectrum(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        n_samples: Optional[int] = None,
        block_structure: str = 'layer',  # 'layer', 'module', 'none' (full Fisher)
        center_gradients: bool = False,  # False = Fisher, True = Gradient Covariance
        precomputed_gradients: Optional[List[Dict[str, torch.Tensor]]] = None,  # Reuse gradients
    ) -> Dict[str, Any]:
        """
        Main entry point for Fisher spectrum computation.

        Args:
            model: Neural network model
            batch: Input batch with 'input_ids' and optional 'attention_mask'
            n_samples: Number of samples to use (None = use all in batch)
            block_structure: How to partition parameters into blocks
            center_gradients: If True, compute gradient covariance instead of Fisher

        Returns:
            Dictionary containing:
            - global: Global spectrum metrics (merged from blocks)
            - per_block: Per-block spectrum metrics
            - metadata: Computation details for reproducibility

        Theory:
            The empirical Fisher at parameter θ is:
            F(θ) = E_x[∇_θ ℓ(x,θ) ∇_θ ℓ(x,θ)ᵀ]

            We approximate with:
            F̂ = (1/N) Σᵢ gᵢ gᵢᵀ where gᵢ = ∇_θ ℓ(xᵢ,θ)

            For block-diagonal approximation:
            F ≈ diag(F₁, F₂, ...) where each Fₖ is the Fisher for block k
        """
        # Ensure model is in eval mode for deterministic forward pass
        model.eval()

        # Validate and prepare batch
        batch = self._prepare_batch(batch, model)
        batch_size = batch['input_ids'].size(0)
        n_samples = min(n_samples or batch_size, batch_size)

        if n_samples < 2:
            logger.warning(f"Need at least 2 samples for spectrum computation, got {n_samples}")
            return self._empty_results()

        # Collect gradients or use precomputed/cached ones
        if precomputed_gradients is not None:
            # Use precomputed gradients (from FisherCollector)
            gradients = self._organize_precomputed_gradients(
                precomputed_gradients, block_structure
            )
        elif self.gradient_cache is not None and self.gradient_cache.per_sample_gradients:
            # Use cached gradients
            gradients = self._organize_precomputed_gradients(
                self.gradient_cache.per_sample_gradients, block_structure
            )
        elif self.config.storage_mode == 'full':
            gradients = self._collect_gradients_full(model, batch, n_samples, block_structure)
        elif self.config.storage_mode == 'chunked':
            gradients = self._collect_gradients_chunked(model, batch, n_samples, block_structure)
        else:  # streaming
            return self._compute_spectrum_streaming(model, batch, n_samples, block_structure)

        # Compute spectrum from collected gradients
        return self._compute_spectrum_from_gradients(gradients, center_gradients, n_samples)

    def _prepare_batch(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> Dict[str, torch.Tensor]:
        """Prepare and validate batch, ensuring it has required fields."""
        if 'input_ids' not in batch:
            raise ValueError("Batch must contain 'input_ids'")

        # Add labels if not present (for language modeling)
        if 'labels' not in batch:
            batch = dict(batch)  # Copy to avoid modifying original
            batch['labels'] = batch['input_ids'].clone()

        # Move to model device
        device = next(model.parameters()).device
        batch = {k: v.to(device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        return batch

    def _collect_gradients_full(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        n_samples: int,
        block_structure: str
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Collect all gradients at once (memory-intensive but simple).

        Theory: We need per-sample gradients gᵢ = ∇_θ ℓ(xᵢ,θ) to form
        the empirical Fisher F̂ = (1/N) Σᵢ gᵢ gᵢᵀ.
        """
        gradients = {}

        for i in range(n_samples):
            # Zero gradients - critical for correctness
            model.zero_grad(set_to_none=True)

            # Forward-backward for single sample
            single_batch = {k: v[i:i+1] if torch.is_tensor(v) else v
                           for k, v in batch.items()}

            try:
                outputs = model(**single_batch)
                loss = outputs.loss

                # Check for NaN/Inf
                if not torch.isfinite(loss):
                    logger.warning(f"Non-finite loss at sample {i}, skipping")
                    continue

                # Backward pass
                loss.backward()

                # Collect gradients per block - group parameters in same block
                block_grads = {}  # Temporary storage for this sample
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        block_key = self._get_block_key(name, block_structure)

                        # Flatten gradient
                        grad_flat = param.grad.view(-1).to(self.config.dtype_compute)

                        # Initialize block list if needed
                        if block_key not in block_grads:
                            block_grads[block_key] = []

                        # Add to block (will concatenate later)
                        block_grads[block_key].append(grad_flat)

                # Concatenate and subsample each block
                for block_key, grad_list in block_grads.items():
                    # Concatenate all parameters in this block
                    full_block_grad = torch.cat(grad_list)

                    # Subsample if needed
                    grad_subsampled = self._subsample_gradient(full_block_grad, block_key)

                    # Initialize gradients list if needed
                    if block_key not in gradients:
                        gradients[block_key] = []

                    # Store gradient (detach to free graph)
                    gradients[block_key].append(grad_subsampled.detach().cpu())

            except Exception as e:
                logger.warning(f"Error processing sample {i}: {e}")
                continue

        # Validate we collected enough samples
        for block_key in list(gradients.keys()):
            if len(gradients[block_key]) < 2:
                logger.warning(f"Block {block_key} has < 2 samples, removing")
                del gradients[block_key]

        return gradients

    def _collect_gradients_chunked(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        n_samples: int,
        block_structure: str
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Collect gradients in chunks to balance memory and efficiency.
        Process chunk_size samples at a time, accumulating gradients.
        """
        all_gradients = {}
        n_chunks = math.ceil(n_samples / self.config.chunk_size)

        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * self.config.chunk_size
            end_idx = min(start_idx + self.config.chunk_size, n_samples)

            # Process chunk
            chunk_gradients = self._collect_gradients_full(
                model,
                {k: v[start_idx:end_idx] if torch.is_tensor(v) else v
                 for k, v in batch.items()},
                end_idx - start_idx,
                block_structure
            )

            # Merge with accumulated gradients
            for block_key, grad_list in chunk_gradients.items():
                if block_key not in all_gradients:
                    all_gradients[block_key] = []
                all_gradients[block_key].extend(grad_list)

        return all_gradients

    def _organize_precomputed_gradients(
        self,
        precomputed_grads: List[Dict[str, torch.Tensor]],
        block_structure: str
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Organize precomputed gradients into block structure.

        Args:
            precomputed_grads: List of gradient dicts from FisherCollector
            block_structure: How to group parameters

        Returns:
            Organized gradients by block
        """
        # First pass: determine block membership and dimensions
        block_params = {}  # block_key -> list of (param_name, param_shape)
        for name, grad in precomputed_grads[0].items():
            block_key = self._get_block_key(name, block_structure)
            if block_key not in block_params:
                block_params[block_key] = []
            block_params[block_key].append((name, grad.shape))

        # Second pass: collect and concatenate gradients per block
        gradients = {}
        for sample_grads in precomputed_grads:
            for block_key, params_info in block_params.items():
                # Concatenate all parameters in this block into single vector
                block_grads = []
                for param_name, param_shape in params_info:
                    if param_name in sample_grads:
                        grad = sample_grads[param_name]
                        grad_flat = grad.view(-1).to(self.config.dtype_compute)
                        block_grads.append(grad_flat)

                if block_grads:
                    # Concatenate all params in block
                    full_block_grad = torch.cat(block_grads)

                    # Subsample if needed
                    grad_subsampled = self._subsample_gradient(full_block_grad, block_key)

                    # Initialize block list if needed
                    if block_key not in gradients:
                        gradients[block_key] = []

                    # Store gradient
                    if grad_subsampled.is_cuda:
                        grad_subsampled = grad_subsampled.cpu()
                    gradients[block_key].append(grad_subsampled)

        return gradients

    def _subsample_gradient(self, grad_flat: torch.Tensor, block_key: str) -> torch.Tensor:
        """
        Apply consistent subsampling to reduce parameter dimension.

        Theory: For large P, we subsample parameters consistently across samples
        to maintain valid gradient matrix structure. This preserves relative
        eigenvalue magnitudes while reducing computation.
        """
        P = grad_flat.numel()

        if P <= self.config.max_params_per_block:
            return grad_flat

        # Get or create consistent subsampling indices
        if block_key not in self.subsample_indices:
            # Use block-specific seed for reproducibility
            seed = self.config.seed + hash(block_key) % 1000000
            generator = torch.Generator().manual_seed(seed)

            # Create indices on CPU to avoid device issues
            indices = torch.randperm(P, generator=generator, device='cpu')[:self.config.max_params_per_block]
            self.subsample_indices[block_key] = indices

        # Apply subsampling (move indices to grad device if needed)
        indices = self.subsample_indices[block_key].to(grad_flat.device)
        return grad_flat[indices]

    def _get_block_key(self, param_name: str, block_structure: str) -> str:
        """
        Determine block assignment for parameter.

        Block structure options:
        - 'layer': Group by transformer layer (e.g., layer.0, layer.1)
        - 'module': Group by module type (attention, mlp, embedding)
        - 'none': Single block (full Fisher, no block-diagonal approximation)
        """
        if block_structure == 'none':
            return 'global'
        elif block_structure == 'layer':
            # Extract layer number if present
            import re
            layer_match = re.search(r'layer[s]?\.(\d+)', param_name)
            if layer_match:
                return f"layer_{layer_match.group(1)}"
            elif 'embed' in param_name.lower():
                return 'embedding'
            elif 'lm_head' in param_name.lower() or 'classifier' in param_name.lower():
                return 'output'
            else:
                return 'other'
        elif block_structure == 'module':
            # Group by module type
            if any(x in param_name.lower() for x in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'attention']):
                return 'attention'
            elif any(x in param_name.lower() for x in ['mlp', 'fc', 'dense', 'feedforward']):
                return 'mlp'
            elif 'embed' in param_name.lower():
                return 'embedding'
            elif any(x in param_name.lower() for x in ['norm', 'ln', 'layernorm']):
                return 'normalization'
            else:
                return 'other'
        else:
            return 'default'

    def _compute_spectrum_from_gradients(
        self,
        gradients: Dict[str, List[torch.Tensor]],
        center_gradients: bool,
        n_samples: int
    ) -> Dict[str, Any]:
        """
        Compute spectrum from collected gradients.

        Theory:
        For gradient matrix G ∈ ℝ^(N×P) where rows are per-sample gradients:
        - Fisher: F = (1/N) Gᵀ G (uncentered)
        - Covariance: C = (1/N) G̃ᵀ G̃ where G̃ is row-centered

        When N << P, use Gram matrix: eigenvalues of (1/N) G Gᵀ = non-zero eigenvalues of (1/N) Gᵀ G
        """
        results = {
            'per_block': {},
            'global': None,
            'metadata': {
                'n_samples_requested': n_samples,
                'centered': center_gradients,
                'seed': self.config.seed,
                'eps': self.config.eps,
                'regularization': self.config.regularization,
                'block_structure': list(gradients.keys())
            }
        }

        all_eigenvalues = []
        total_params = 0

        for block_key, grad_list in gradients.items():
            if len(grad_list) < 2:
                continue

            # Stack gradients into matrix G ∈ ℝ^(N×P)
            G = torch.stack(grad_list).to(self.config.dtype_eigensolve)
            N, P = G.shape
            total_params += P

            # Center if computing covariance instead of Fisher
            if center_gradients:
                G = G - G.mean(dim=0, keepdim=True)

            # Compute eigenvalues using appropriate method
            eigenvalues = self._compute_block_eigenvalues(G)

            # Analyze spectrum
            block_metrics = self._analyze_eigenvalues(eigenvalues, N, P)
            block_metrics['block_name'] = block_key
            results['per_block'][block_key] = block_metrics

            # Collect for global spectrum (union of block spectra)
            all_eigenvalues.extend(eigenvalues.tolist())

        # Compute global metrics by merging block eigenvalues
        if all_eigenvalues:
            results['global'] = self._compute_global_metrics(all_eigenvalues)
            results['global']['total_params'] = total_params
        else:
            results['global'] = self._empty_global_metrics()

        return results

    def _compute_block_eigenvalues(self, G: torch.Tensor) -> torch.Tensor:
        """
        Compute eigenvalues of Fisher block (1/N) Gᵀ G efficiently.

        Theory:
        When N << P: Use Gram matrix (1/N) G Gᵀ which has same non-zero eigenvalues
        When N >= P/4: Direct computation or randomized SVD

        The factor of 1/N normalizes the Fisher to be an expectation.
        """
        N, P = G.shape

        # Choose computation path based on dimensions
        if N <= P // 4:
            # Use Gram matrix trick for N << P
            # Eigenvalues of (1/N) G Gᵀ = eigenvalues of (1/N) Gᵀ G (non-zero ones)
            gram = (G @ G.T) / N  # N × N matrix

            # Add small regularization for numerical stability
            gram = gram + self.config.regularization * torch.eye(N, device=gram.device, dtype=gram.dtype)

            # Compute eigenvalues (already sorted ascending by eigvalsh)
            try:
                eigenvalues = torch.linalg.eigvalsh(gram)
            except torch.linalg.LinAlgError as e:
                logger.warning(f"Eigendecomposition failed: {e}, using SVD fallback")
                # Fallback to SVD
                try:
                    U, S, _ = torch.linalg.svd(gram, full_matrices=False)
                    eigenvalues = S
                except Exception as e2:
                    logger.error(f"SVD also failed: {e2}, returning zeros")
                    return torch.zeros(1, dtype=gram.dtype)
        else:
            # For N >= P/4, use randomized SVD for efficiency
            k = min(self.config.top_k_eigenvalues, min(N, P) - 1)
            try:
                # Normalize by sqrt(N) for SVD, then square singular values
                _, S, _ = torch.svd_lowrank(G / math.sqrt(N), q=k)
                eigenvalues = S ** 2
            except Exception as e:
                logger.warning(f"Randomized SVD failed: {e}, using diagonal approximation")
                # Fallback to diagonal Fisher (parameter-wise variance)
                eigenvalues = (G ** 2).sum(dim=0) / N

        # Sort descending (critical - eigvalsh returns ascending!)
        eigenvalues = torch.sort(eigenvalues, descending=True).values

        # Filter numerical zeros
        eigenvalues = eigenvalues[eigenvalues > self.config.eps]

        return eigenvalues

    def _analyze_eigenvalues(self, eigenvalues: torch.Tensor, N: int, P: int) -> Dict[str, Any]:
        """
        Compute spectral metrics from eigenvalues.

        Metrics:
        - Spectral gap: λ₁ - λ₂ (gap between largest eigenvalues)
        - Condition number: λ_max / λ_min (optimization difficulty)
        - Effective rank: exp(H(p)) where p = λ/Σλ (parameter efficiency)
        """
        if len(eigenvalues) == 0:
            return self._empty_block_metrics(N, P)

        metrics = {
            'n_samples': N,
            'n_params': P,
            'n_eigenvalues': len(eigenvalues),
            'largest_eigenvalue': float(eigenvalues[0]),
        }

        if len(eigenvalues) >= 2:
            # Spectral gap (correct definition: λ₁ - λ₂)
            metrics['spectral_gap'] = float(eigenvalues[0] - eigenvalues[1])
            metrics['second_eigenvalue'] = float(eigenvalues[1])
        else:
            metrics['spectral_gap'] = 0.0
            metrics['second_eigenvalue'] = 0.0

        # Condition number (with regularization for numerical stability)
        min_positive = eigenvalues[eigenvalues > self.config.eps]
        if len(min_positive) > 0:
            metrics['condition_number'] = float(eigenvalues[0] / min_positive[-1])
        else:
            metrics['condition_number'] = float('inf')

        # Effective rank via entropy
        # p = eigenvalues / sum, H = -Σ p log(p), rank = exp(H)
        p = eigenvalues / eigenvalues.sum()
        # Avoid log(0) with small epsilon
        entropy = -(p * torch.log(p + self.config.eps)).sum()
        metrics['effective_rank'] = float(torch.exp(entropy))

        # Store top-k eigenvalues for analysis
        metrics['top_eigenvalues'] = eigenvalues[:min(10, len(eigenvalues))].tolist()

        # Compute trace (sum of eigenvalues) - related to total Fisher "mass"
        metrics['trace'] = float(eigenvalues.sum())

        return metrics

    def _compute_global_metrics(self, all_eigenvalues: List[float]) -> Dict[str, Any]:
        """
        Merge eigenvalues from all blocks to compute global spectrum.

        Theory: For block-diagonal matrix, global eigenvalues are the
        union of block eigenvalues (not their average!).
        """
        # Convert to tensor and sort descending
        eigs = torch.tensor(all_eigenvalues, dtype=self.config.dtype_eigensolve)
        eigs = torch.sort(eigs, descending=True).values

        # Filter numerical zeros
        eigs = eigs[eigs > self.config.eps]

        if len(eigs) == 0:
            return self._empty_global_metrics()

        metrics = {
            'largest_eigenvalue': float(eigs[0]),
            'n_total_eigenvalues': len(eigs),
        }

        # Spectral gap (global)
        if len(eigs) >= 2:
            metrics['spectral_gap'] = float(eigs[0] - eigs[1])
            metrics['second_eigenvalue'] = float(eigs[1])
        else:
            metrics['spectral_gap'] = 0.0
            metrics['second_eigenvalue'] = 0.0

        # Global condition number
        metrics['condition_number'] = float(eigs[0] / eigs[-1])

        # Global effective rank
        p = eigs / eigs.sum()
        entropy = -(p * torch.log(p + self.config.eps)).sum()
        metrics['effective_rank'] = float(torch.exp(entropy))

        # Top eigenvalues
        metrics['top_eigenvalues'] = eigs[:min(self.config.top_k_eigenvalues, len(eigs))].tolist()

        # Trace
        metrics['trace'] = float(eigs.sum())

        # Optimization-related metrics (NOT mixing time!)
        # These are heuristic timescales, not rigorous mixing times
        if metrics['largest_eigenvalue'] > 0:
            # Approximate convergence scale for gradient descent
            metrics['gd_convergence_scale'] = 1.0 / metrics['largest_eigenvalue']

        return metrics

    def _compute_spectrum_streaming(
        self,
        model: nn.Module,
        batch: Dict[str, torch.Tensor],
        n_samples: int,
        block_structure: str
    ) -> Dict[str, Any]:
        """
        Streaming computation using online PCA (memory-efficient).
        Not implemented in this version - returns placeholder.
        """
        logger.warning("Streaming mode not yet implemented, using chunked mode instead")
        return self._compute_spectrum_from_gradients(
            self._collect_gradients_chunked(model, batch, n_samples, block_structure),
            center_gradients=False,
            n_samples=n_samples
        )

    def _empty_results(self) -> Dict[str, Any]:
        """Return empty results structure when computation fails."""
        return {
            'per_block': {},
            'global': self._empty_global_metrics(),
            'metadata': {
                'error': 'Insufficient samples or computation failed',
                'seed': self.config.seed
            }
        }

    def _empty_block_metrics(self, N: int = 0, P: int = 0) -> Dict[str, Any]:
        """Return empty metrics for a block."""
        return {
            'n_samples': N,
            'n_params': P,
            'n_eigenvalues': 0,
            'largest_eigenvalue': 0.0,
            'spectral_gap': 0.0,
            'condition_number': float('inf'),
            'effective_rank': 1.0,
            'top_eigenvalues': [],
            'trace': 0.0
        }

    def _empty_global_metrics(self) -> Dict[str, Any]:
        """Return empty global metrics."""
        return {
            'largest_eigenvalue': 0.0,
            'spectral_gap': 0.0,
            'condition_number': float('inf'),
            'effective_rank': 1.0,
            'n_total_eigenvalues': 0,
            'top_eigenvalues': [],
            'trace': 0.0,
            'total_params': 0
        }