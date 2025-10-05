"""
Fisher Accumulator: Data-agnostic Fisher computation with automatic batching.

This module provides a standalone Fisher accumulator that can handle any batch size
internally through automatic chunking and EMA accumulation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class FisherAccumulator:
    """
    Data-agnostic Fisher accumulator that handles any batch size internally.
    Processes data in chunks and maintains EMA accumulation.
    """

    def __init__(self, model: nn.Module, device_batch_size: int = 32, ema_decay: float = 0.99):
        """
        Initialize Fisher accumulator.

        Args:
            model: The model to compute Fisher for
            device_batch_size: Max samples to process at once on device
            ema_decay: Decay factor for exponential moving average
        """
        self.model = model
        self.device_batch_size = device_batch_size
        self.ema_decay = ema_decay
        self.fisher_ema = {}
        self.total_samples_seen = 0
        self.total_tokens_seen = 0  # Track token weight for proper Welford weighting
        self.device = next(model.parameters()).device

    def process_data(
        self,
        data: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
        max_samples: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Process any type of data and compute Fisher.

        Args:
            data: Can be:
                - Tensor: Single batch (will be auto-chunked)
                - List[Tensor]: List of batches
                - List[Dict]: List of batch dicts
                - Dict: Single batch dict with 'input_ids', 'attention_mask', etc.
                - DataLoader: PyTorch DataLoader
            max_samples: Optional limit on total samples to process

        Returns:
            Fisher EMA values for all parameters
        """
        if isinstance(data, torch.Tensor):
            # Single tensor - treat as input_ids
            self._process_batch({'input_ids': data}, max_samples)

        elif isinstance(data, list):
            # List of batches
            for batch in data:
                if max_samples and self.total_samples_seen >= max_samples:
                    break
                if isinstance(batch, torch.Tensor):
                    self._process_batch({'input_ids': batch}, max_samples)
                else:
                    self._process_batch(batch, max_samples)

        elif isinstance(data, dict):
            # Single batch dict
            self._process_batch(data, max_samples)

        else:
            # Try to iterate (DataLoader case)
            try:
                for batch in data:
                    if max_samples and self.total_samples_seen >= max_samples:
                        break
                    if isinstance(batch, torch.Tensor):
                        self._process_batch({'input_ids': batch}, max_samples)
                    else:
                        self._process_batch(batch, max_samples)
            except TypeError:
                raise ValueError(f"Unsupported data type: {type(data)}")

        return self.fisher_ema

    def _process_batch(self, batch: Dict[str, torch.Tensor], max_samples: Optional[int] = None):
        """Process a single batch, auto-chunking if needed."""
        # Move batch to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # Determine batch size
        if 'input_ids' in batch:
            batch_size = batch['input_ids'].size(0)
        else:
            # Find first tensor in batch
            for v in batch.values():
                if torch.is_tensor(v):
                    batch_size = v.size(0)
                    break
            else:
                logger.warning("No tensors found in batch")
                return

        # Check if we've hit max samples
        if max_samples and self.total_samples_seen >= max_samples:
            remaining = max_samples - self.total_samples_seen
            if remaining <= 0:
                return
            # Process only remaining samples
            batch_size = min(batch_size, remaining)
            batch = {k: v[:batch_size] if torch.is_tensor(v) else v
                    for k, v in batch.items()}

        # Process in chunks if needed
        if batch_size > self.device_batch_size:
            for start_idx in range(0, batch_size, self.device_batch_size):
                if max_samples and self.total_samples_seen >= max_samples:
                    break

                end_idx = min(start_idx + self.device_batch_size, batch_size)
                chunk = {k: v[start_idx:end_idx] if torch.is_tensor(v) else v
                        for k, v in batch.items()}
                self._compute_fisher_chunk(chunk)
                self.total_samples_seen += (end_idx - start_idx)
                # Track tokens for proper Welford weighting
                if 'attention_mask' in chunk:
                    self.total_tokens_seen += chunk['attention_mask'].sum().item()
                else:
                    self.total_tokens_seen += chunk['input_ids'].numel()
        else:
            self._compute_fisher_chunk(batch)
            self.total_samples_seen += batch_size
            # Track tokens for proper Welford weighting
            if 'attention_mask' in batch:
                self.total_tokens_seen += batch['attention_mask'].sum().item()
            else:
                self.total_tokens_seen += batch['input_ids'].numel()

    def _compute_fisher_chunk(self, chunk: Dict[str, torch.Tensor]):
        """Compute Fisher for a device-sized chunk."""
        was_training = self.model.training

        # CRITICAL: Enable gradients for ALL parameters (pretrained models load with requires_grad=False)
        original_requires_grad = {}
        for name, param in self.model.named_parameters():
            original_requires_grad[name] = param.requires_grad
            param.requires_grad = True

        self.model.eval()  # Use eval mode for deterministic Fisher (gradients still work!)

        # Add labels if missing (for sampling from output distribution)
        if 'labels' not in chunk:
            chunk = self._add_labels(chunk)

        try:
            with torch.enable_grad():
                self.model.zero_grad(set_to_none=True)

                # Forward pass
                outputs = self.model(**chunk)

                if not hasattr(outputs, 'loss') or outputs.loss is None:
                    logger.warning("Loss is None, skipping Fisher computation for this chunk")
                    return

                loss = outputs.loss.float()

                # Check for NaN
                if torch.isnan(loss):
                    logger.warning("NaN loss detected, skipping Fisher computation for this chunk")
                    return

                # Compute gradients
                grads = torch.autograd.grad(
                    loss,
                    [p for p in self.model.parameters() if p.requires_grad],
                    create_graph=False,
                    retain_graph=False
                )

                # Update EMA
                self._update_ema(grads)

        except Exception as e:
            logger.warning(f"Error computing Fisher for chunk: {e}")

        finally:
            # Restore original requires_grad states
            for name, param in self.model.named_parameters():
                param.requires_grad = original_requires_grad[name]

            # Restore training mode
            if was_training:
                self.model.train()
            else:
                self.model.eval()

    def _add_labels(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Add labels by sampling from output distribution (Fisher sampling)."""
        try:
            with torch.no_grad():
                outputs = self.model(**batch)

                if not hasattr(outputs, 'logits'):
                    logger.warning("Model outputs don't have logits, cannot add labels")
                    return batch

                logits = outputs.logits

                # Handle different logit shapes
                if logits.dim() == 2:
                    # [batch_size, vocab_size] - single token prediction
                    probs = F.softmax(logits, dim=-1)
                    labels = torch.multinomial(probs, num_samples=1).squeeze(-1)
                elif logits.dim() == 3:
                    # [batch_size, seq_len, vocab_size] - sequence prediction
                    batch_size, seq_len, vocab_size = logits.shape

                    # Sample from the output distribution
                    probs = F.softmax(logits, dim=-1)

                    # Reshape for sampling
                    probs_flat = probs.view(-1, vocab_size)
                    labels = torch.multinomial(probs_flat, num_samples=1).squeeze(-1)
                    labels = labels.view(batch_size, seq_len)

                    # Mask padding positions
                    if 'attention_mask' in batch:
                        labels = labels * batch['attention_mask'] + (-100) * (1 - batch['attention_mask'])
                else:
                    logger.warning(f"Unexpected logits shape: {logits.shape}")
                    return batch

                batch['labels'] = labels.long()

        except Exception as e:
            logger.warning(f"Error adding labels: {e}")

        return batch

    def _update_ema(self, grads):
        """Update exponential moving average of Fisher diagonal."""
        param_names = [name for name, p in self.model.named_parameters() if p.requires_grad]

        for name, grad in zip(param_names, grads):
            if grad is None:
                continue

            # Compute Fisher diagonal (gradient squared)
            fisher_diag = grad.pow(2).detach()

            # Update EMA
            if name not in self.fisher_ema:
                self.fisher_ema[name] = fisher_diag.float()
            else:
                self.fisher_ema[name] = (
                    self.ema_decay * self.fisher_ema[name] +
                    (1 - self.ema_decay) * fisher_diag.float()
                )

    def get_fisher_values(self) -> Dict[str, torch.Tensor]:
        """Get the computed Fisher values."""
        return self.fisher_ema

    def reset(self):
        """Reset the accumulator."""
        self.fisher_ema = {}
        self.total_samples_seen = 0