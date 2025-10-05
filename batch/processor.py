"""
Batch processor module for memory-efficient batch processing.
This is the core batch processor that batch_processor_integration.py extends.
"""

import torch
import logging
from typing import Dict, Any, Callable, Optional, Union
from enum import Enum
from contextlib import contextmanager

logger = logging.getLogger(__name__)


class ProcessingMode(Enum):
    """Processing modes for batch handling."""
    ADAPTIVE = 'adaptive'
    FIXED = 'fixed'
    DYNAMIC = 'dynamic'


class BatchConfig:
    """Configuration for batch processing with proper attribute access."""

    def __init__(
        self,
        mode: Union[ProcessingMode, str] = ProcessingMode.ADAPTIVE,
        chunk_size: int = 32,
        max_size: int = 128,
        seed: Optional[int] = None,
        weighted: bool = True,
        clear_cache: bool = True,
        deterministic: bool = True,
        verbose: bool = False
    ):
        """Initialize batch configuration.

        Args:
            mode: Processing mode (can be string or ProcessingMode enum)
            chunk_size: Size of chunks for processing
            max_size: Maximum batch size
            seed: Random seed for reproducibility
            weighted: Use weighted averaging
            clear_cache: Clear CUDA cache between chunks
            deterministic: Use deterministic algorithms
            verbose: Enable verbose output
        """
        # Handle string mode conversion
        if isinstance(mode, str):
            mode_map = {
                'adaptive': ProcessingMode.ADAPTIVE,
                'fixed': ProcessingMode.FIXED,
                'dynamic': ProcessingMode.DYNAMIC
            }
            self.mode = mode_map.get(mode, ProcessingMode.ADAPTIVE)
        else:
            self.mode = mode

        self.chunk_size = chunk_size
        self.max_size = max_size
        self.seed = seed
        self.weighted = weighted
        self.clear_cache = clear_cache
        self.deterministic = deterministic
        self.verbose = verbose

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BatchConfig':
        """Create BatchConfig from dictionary."""
        return cls(**{k: v for k, v in config_dict.items()
                     if k in cls.__init__.__code__.co_varnames})


class BatchProcessor:
    """Core batch processor for memory-efficient computation."""

    def __init__(self):
        """Initialize batch processor."""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.debug(f"BatchProcessor initialized on {self.device}")

    @contextmanager
    def process_context(self, model: Optional[torch.nn.Module] = None):
        """
        Context manager for batch processing with state preservation.

        Automatically:
        - Saves model state (training mode, requires_grad) if model provided
        - Clears GPU cache on exit
        - Restores model state on exit

        Args:
            model: Optional model to manage state for

        Example:
            with processor.process_context(model):
                # Model state automatically managed
                result = compute_something(model, batch_chunk)
            # Model state restored, cache cleared
        """
        # Save model state if provided
        original_training = None
        original_requires_grad = {}

        if model is not None:
            original_training = model.training
            # Save requires_grad state (though we don't modify it here)
            for name, param in model.named_parameters():
                original_requires_grad[name] = param.requires_grad

        try:
            yield
        finally:
            # Restore model state if we saved it
            if model is not None and original_training is not None:
                model.train(original_training)
                # Restore requires_grad if it was modified
                for name, param in model.named_parameters():
                    if name in original_requires_grad:
                        param.requires_grad = original_requires_grad[name]

            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def compute_gradients(
        self,
        model: torch.nn.Module,
        batch: Dict[str, torch.Tensor],
        config_override: Optional[Union[Dict, BatchConfig]] = None
    ) -> Dict[str, torch.Tensor]:
        """Compute gradients for a model given a batch.

        Specialized method for gradient computation with proper:
        - Micro-batching for memory efficiency
        - Gradient accumulation across chunks
        - Automatic label generation for causal LM

        Args:
            model: The model to compute gradients for
            batch: Input batch with 'input_ids' and optional 'labels'
            config_override: Override configuration

        Returns:
            Dictionary mapping parameter names to gradient tensors
        """
        # Ensure batch has labels for loss computation
        if 'labels' not in batch:
            batch = batch.copy()
            batch['labels'] = batch['input_ids'].clone() if torch.is_tensor(batch['input_ids']) else batch['input_ids']

        # Store original model state
        original_training = model.training
        model.eval()  # Use eval mode for deterministic analysis (gradients still work!)

        # Enable gradients for all parameters (critical for pretrained models)
        original_requires_grad = {}
        for name, param in model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = True

        # Define gradient computation function for each chunk
        def compute_chunk_gradients(chunk):
            # Ensure chunk is on model device
            model_device = next(model.parameters()).device
            chunk_on_device = {k: v.to(model_device) if torch.is_tensor(v) else v
                             for k, v in chunk.items()}

            # Forward and backward pass
            with torch.enable_grad():
                outputs = model(**chunk_on_device)
                loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]

                # Validate loss reduction mode
                # HuggingFace models use reduction='mean' by default
                # If model uses reduction='sum', chunked gradients will be wrong
                chunk_size = chunk_on_device['input_ids'].shape[0]
                if loss > 100:  # Heuristic: summed losses are typically large
                    logger.warning(
                        f"⚠️  Large loss detected ({loss:.2f}). "
                        f"Ensure model uses reduction='mean', not 'sum'. "
                        f"Chunked gradient computation assumes averaged loss. "
                        f"If using reduction='sum', gradients will be {chunk_size}× too large!"
                    )
                    # Optionally normalize (uncomment if you want auto-fix):
                    # logger.warning(f"Auto-normalizing by chunk_size={chunk_size}")
                    # loss = loss / chunk_size

                if loss is None:
                    logger.error("Model returned None loss")
                    return {}

                # Clear existing gradients
                model.zero_grad()

                # Backward pass
                loss.backward()

                # Extract gradients
                gradients = {}
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        gradients[name] = param.grad.detach().clone()

                # Clear gradients to save memory
                model.zero_grad(set_to_none=True)

                return gradients

        # Use process_batch with gradient-specific reduction
        gradients = self.process_batch(
            batch=batch,
            compute_fn=compute_chunk_gradients,
            reduction='mean',  # Average gradients across micro-batches
            config_override=config_override
        )

        # Restore original model state
        if not original_training:
            model.eval()

        # Restore original requires_grad state
        for name, param in model.named_parameters():
            if name in original_requires_grad:
                param.requires_grad = original_requires_grad[name]

        return gradients if gradients else {}

    def process_batch(
        self,
        batch: Dict[str, torch.Tensor],
        compute_fn: Callable,
        reduction: str = 'mean',
        config_override: Optional[Union[Dict, BatchConfig]] = None
    ) -> Any:
        """Process a batch with optional chunking for memory efficiency.

        Args:
            batch: Input batch dictionary
            compute_fn: Function to compute on each chunk
            reduction: How to reduce results ('mean', 'sum', 'none')
            config_override: Override configuration (dict or BatchConfig)

        Returns:
            Computed result
        """
        # Convert dict config to BatchConfig if needed
        config = self._get_config(config_override)

        # Apply deterministic settings if requested
        if config.deterministic:
            self._set_deterministic(config.seed)

        # Process based on configuration
        if config.chunk_size and len(batch.get('input_ids', [])) > config.chunk_size:
            # Process in chunks for large batches
            result = self._process_chunked(batch, compute_fn, config, reduction)
        else:
            # Process full batch
            result = compute_fn(batch)

        # Clear cache if requested
        if config.clear_cache and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def _get_config(self, config_override: Optional[Union[Dict, BatchConfig]]) -> BatchConfig:
        """Get configuration, converting dict to BatchConfig if needed."""
        if config_override is None:
            return BatchConfig()
        elif isinstance(config_override, dict):
            return BatchConfig.from_dict(config_override)
        else:
            return config_override

    def _set_deterministic(self, seed: Optional[int] = None):
        """Set deterministic mode for reproducibility."""
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _process_chunked(
        self,
        batch: Dict[str, torch.Tensor],
        compute_fn: Callable,
        config: BatchConfig,
        reduction: str
    ) -> Any:
        """Process batch in chunks to save memory."""
        batch_size = batch['input_ids'].shape[0]
        chunk_size = min(config.chunk_size, batch_size)

        results = []
        chunk_sizes = []  # Track chunk sizes for weighted averaging
        for i in range(0, batch_size, chunk_size):
            end_idx = min(i + chunk_size, batch_size)
            chunk = {k: v[i:end_idx] if torch.is_tensor(v) else v
                    for k, v in batch.items()}

            chunk_result = compute_fn(chunk)
            results.append(chunk_result)
            # Track chunk size for weighted averaging
            chunk_sizes.append(end_idx - i)

            # Clear cache periodically for memory management
            if config.clear_cache and i % (chunk_size * 4) == 0:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # Reduce results
        if reduction == 'none':
            return results
        elif reduction == 'mean':
            # Use weighted mean to handle unequal chunk sizes correctly
            # For unbiased estimation: E[f(x)] = Σ (nᵢ/N) × E[f(xᵢ)]
            total_samples = sum(chunk_sizes)
            weights = [n / total_samples for n in chunk_sizes]

            if torch.is_tensor(results[0]):
                # Weighted average of tensors
                weighted_sum = sum(r * w for r, w in zip(results, weights))
                return weighted_sum
            elif isinstance(results[0], dict):
                # For dictionaries (like gradients), weighted average each key
                averaged = {}
                for key in results[0].keys():
                    values = [r[key] for r in results if key in r]
                    if values:
                        if torch.is_tensor(values[0]):
                            # Weighted average of tensors
                            weighted_sum = sum(v * w for v, w in zip(values, weights))
                            averaged[key] = weighted_sum
                        else:
                            # Weighted average of scalars
                            averaged[key] = sum(v * w for v, w in zip(values, weights))
                return averaged
            else:
                # Weighted average of scalars
                return sum(r * w for r, w in zip(results, weights))
        elif reduction == 'sum':
            if isinstance(results[0], dict):
                # Sum each key in dictionaries
                summed = {}
                for key in results[0].keys():
                    values = [r[key] for r in results if key in r]
                    if values:
                        if torch.is_tensor(values[0]):
                            summed[key] = torch.stack(values).sum(dim=0)
                        else:
                            summed[key] = sum(values)
                return summed
            else:
                return sum(results)
        else:
            return results


def create_batch(
    data_batch: Dict[str, torch.Tensor],
    batch_config: Union[Dict, BatchConfig],
    device: torch.device
) -> Dict[str, torch.Tensor]:
    """Create and configure a batch for processing.

    Args:
        data_batch: Original data batch
        batch_config: Batch configuration (dict or BatchConfig)
        device: Target device

    Returns:
        Configured batch on the specified device
    """
    # Convert config if needed
    if isinstance(batch_config, dict):
        config = BatchConfig.from_dict(batch_config)
    else:
        config = batch_config

    # Move batch to device
    batch = {k: v.to(device) if torch.is_tensor(v) else v
             for k, v in data_batch.items()}

    # Limit batch size if specified
    if config.max_size is not None:
        for key in batch:
            if torch.is_tensor(batch[key]) and len(batch[key]) > config.max_size:
                batch[key] = batch[key][:config.max_size]

    return batch