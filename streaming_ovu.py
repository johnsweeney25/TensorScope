#!/usr/bin/env python3
"""
Streaming OV→U Computation for Memory-Efficient Analysis

This module implements streaming computation of OV→U contributions,
allowing analysis of very long sequences without memory explosion.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Generator, Optional, Any, Tuple
import numpy as np
from dataclasses import dataclass
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class StreamingOpportunity:
    """Represents an induction opportunity in the stream."""
    batch_idx: int
    query_pos: int
    key_pos: int
    target_token: int
    timestamp: int  # When this was added to stream


class StreamingOVUComputer:
    """
    Computes OV→U contributions in a streaming fashion.

    Instead of collecting all opportunities first, this processes them
    in chunks, maintaining a sliding window of recent opportunities.
    """

    def __init__(
        self,
        model,
        window_size: int = 256,
        chunk_size: int = 32,
        device=None
    ):
        """
        Initialize streaming computer.

        Args:
            model: The transformer model
            window_size: Size of sliding window for opportunities
            chunk_size: Number of opportunities to process at once
            device: Computation device
        """
        self.model = model
        self.window_size = window_size
        self.chunk_size = chunk_size
        self.device = device or next(model.parameters()).device

        # Sliding window of opportunities
        self.opportunity_window = deque(maxlen=window_size)

        # Running statistics
        self.total_processed = 0
        self.running_contributions = {}  # (layer, head) -> running stats

    def _detect_model_type(self) -> str:
        """Detect model architecture."""
        model_class = self.model.__class__.__name__.lower()
        if 'gpt2' in model_class:
            return 'gpt2'
        elif 'llama' in model_class:
            return 'llama'
        elif 'neox' in model_class:
            return 'neox'
        elif 'phi' in model_class:
            return 'phi'
        else:
            return 'unknown'

    def stream_opportunities(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        min_distance: int = 1
    ) -> Generator[StreamingOpportunity, None, None]:
        """
        Stream induction opportunities from input sequences.

        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask (optional)
            min_distance: Minimum distance for induction

        Yields:
            StreamingOpportunity objects
        """
        batch_size, seq_len = input_ids.shape

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        for b in range(batch_size):
            for i in range(min_distance + 1, seq_len):
                if attention_mask[b, i] == 0 or attention_mask[b, i-1] == 0:
                    continue

                prev_token = input_ids[b, i-1].item()

                # Look for earlier occurrences
                for j in range(i - min_distance):
                    if attention_mask[b, j] == 0:
                        continue

                    if input_ids[b, j].item() == prev_token and j + 1 < seq_len:
                        if attention_mask[b, j+1] == 0:
                            continue

                        target_token = input_ids[b, j+1].item()

                        yield StreamingOpportunity(
                            batch_idx=b,
                            query_pos=i,
                            key_pos=j,
                            target_token=target_token,
                            timestamp=self.total_processed
                        )

                        self.total_processed += 1

    def process_chunk(
        self,
        opportunities: List[StreamingOpportunity],
        hidden_states: List[torch.Tensor],
        attentions: List[torch.Tensor],
        W_U: torch.Tensor
    ) -> Dict[Tuple[int, int], float]:
        """
        Process a chunk of opportunities.

        Args:
            opportunities: List of opportunities to process
            hidden_states: Hidden states from model
            attentions: Attention weights from model
            W_U: Unembedding matrix

        Returns:
            Dictionary mapping (layer, head) to contribution
        """
        model_type = self._detect_model_type()
        contributions = {}

        # Get model layers
        if hasattr(self.model, 'model'):
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            layers = self.model.transformer.h
        else:
            return contributions

        for layer_idx, layer in enumerate(layers):
            if layer_idx >= len(attentions):
                break

            # Get attention module and weights
            if model_type == 'llama':
                attn_module = layer.self_attn
                n_heads = attn_module.num_heads
                num_kv_heads = getattr(attn_module, 'num_key_value_heads', n_heads)
                head_dim = attn_module.head_dim
                W_O = attn_module.o_proj.weight
                v_proj = attn_module.v_proj
                norm = layer.input_layernorm
            elif model_type == 'gpt2':
                attn_module = layer.attn
                n_heads = attn_module.n_head if hasattr(attn_module, 'n_head') else attn_module.num_heads
                hidden_dim = hidden_states[0].shape[-1]
                head_dim = hidden_dim // n_heads
                W_O = attn_module.c_proj.weight if hasattr(attn_module, 'c_proj') else None
                norm = layer.ln_1
            else:
                continue

            if W_O is None:
                continue

            # Apply normalization
            hidden = hidden_states[layer_idx]
            x_in = norm(hidden) if norm is not None else hidden

            # Get value vectors
            if model_type == 'llama':
                values = v_proj(x_in)
                batch_size, seq_len = x_in.shape[:2]
                values = values.view(batch_size, seq_len, num_kv_heads, head_dim)
                values = values.transpose(1, 2)
            elif hasattr(attn_module, 'c_attn'):
                qkv = attn_module.c_attn(x_in)
                hidden_dim = qkv.shape[-1] // 3
                q, k, v = qkv.split(hidden_dim, dim=-1)
                batch_size, seq_len = v.shape[:2]
                values = v.view(batch_size, seq_len, n_heads, head_dim)
                values = values.transpose(1, 2)
            else:
                continue

            attn_weights = attentions[layer_idx]

            # Process each head
            for head_idx in range(n_heads):
                total_contrib = 0.0
                count = 0

                for opp in opportunities:
                    b = opp.batch_idx
                    i = opp.query_pos
                    j = opp.key_pos
                    target = opp.target_token

                    # Get attention weight
                    attn = attn_weights[b, head_idx, i, j].float()

                    # Get value vector
                    if model_type == 'llama' and num_kv_heads != n_heads:
                        kv_head_idx = head_idx % num_kv_heads
                        v_j = values[b, kv_head_idx, j, :].float()
                    else:
                        v_j = values[b, head_idx, j, :].float()

                    # Compute OV→U contribution
                    weighted_v = attn * v_j
                    head_start = head_idx * head_dim
                    head_end = (head_idx + 1) * head_dim

                    W_O_f = W_O.float()
                    W_U_f = W_U.float()

                    head_out = W_O_f[:, head_start:head_end] @ weighted_v
                    logit_contrib = torch.dot(head_out, W_U_f[:, target])

                    total_contrib += logit_contrib.item()
                    count += 1

                if count > 0:
                    contributions[(layer_idx, head_idx)] = total_contrib / count

        return contributions

    def update_running_stats(self, contributions: Dict[Tuple[int, int], float]):
        """
        Update running statistics with new contributions.

        Uses Welford's online algorithm for numerical stability.
        """
        for (layer, head), contrib in contributions.items():
            if (layer, head) not in self.running_contributions:
                self.running_contributions[(layer, head)] = {
                    'count': 0,
                    'mean': 0.0,
                    'M2': 0.0,  # Sum of squared differences
                    'max': float('-inf'),
                    'min': float('inf')
                }

            stats = self.running_contributions[(layer, head)]
            stats['count'] += 1
            delta = contrib - stats['mean']
            stats['mean'] += delta / stats['count']
            delta2 = contrib - stats['mean']
            stats['M2'] += delta * delta2
            stats['max'] = max(stats['max'], contrib)
            stats['min'] = min(stats['min'], contrib)

    def compute_streaming_ovu(
        self,
        batch: Dict[str, torch.Tensor],
        max_seq_len: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Compute OV→U contributions in streaming fashion.

        Args:
            batch: Input batch
            max_seq_len: Maximum sequence length to process

        Returns:
            Streaming analysis results
        """
        input_ids = batch['input_ids']
        attention_mask = batch.get('attention_mask')

        # Optionally truncate for memory
        if max_seq_len and input_ids.shape[1] > max_seq_len:
            input_ids = input_ids[:, :max_seq_len]
            if attention_mask is not None:
                attention_mask = attention_mask[:, :max_seq_len]
            batch = {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }

        # Get model outputs
        with torch.inference_mode():
            outputs = self.model(**batch, output_attentions=True, output_hidden_states=True)

            if not hasattr(outputs, 'attentions') or outputs.attentions is None:
                return {
                    'error': 'No attention weights available',
                    'head_contributions': []
                }

            attentions = outputs.attentions
            hidden_states = outputs.hidden_states

        # Get unembedding matrix
        if hasattr(self.model, 'lm_head'):
            W_U = self.model.lm_head.weight
        elif hasattr(self.model, 'wte'):
            W_U = self.model.wte.weight.T
        else:
            return {
                'error': 'No unembedding matrix found',
                'head_contributions': []
            }

        # Stream through opportunities
        opportunity_buffer = []

        for opp in self.stream_opportunities(input_ids, attention_mask):
            opportunity_buffer.append(opp)

            # Process chunk when buffer is full
            if len(opportunity_buffer) >= self.chunk_size:
                chunk_contributions = self.process_chunk(
                    opportunity_buffer,
                    hidden_states,
                    attentions,
                    W_U
                )
                self.update_running_stats(chunk_contributions)
                opportunity_buffer = []

        # Process remaining opportunities
        if opportunity_buffer:
            chunk_contributions = self.process_chunk(
                opportunity_buffer,
                hidden_states,
                attentions,
                W_U
            )
            self.update_running_stats(chunk_contributions)

        # Compile final results
        head_contributions = []
        for (layer, head), stats in self.running_contributions.items():
            if stats['count'] > 0:
                variance = stats['M2'] / stats['count'] if stats['count'] > 1 else 0
                std = np.sqrt(variance)

                head_contributions.append({
                    'layer': layer,
                    'head': head,
                    'layer_head': f'L{layer}.H{head}',
                    'ov_contribution': stats['mean'],
                    'std': std,
                    'min': stats['min'],
                    'max': stats['max'],
                    'samples': stats['count']
                })

        # Sort by contribution
        head_contributions.sort(key=lambda x: abs(x['ov_contribution']), reverse=True)

        return {
            'head_contributions': head_contributions,
            'total_opportunities_processed': self.total_processed,
            'streaming_window_size': self.window_size,
            'chunk_size': self.chunk_size,
            'note': f'Processed {self.total_processed} opportunities in streaming mode'
        }


def compare_streaming_vs_batch(
    model,
    batch: Dict[str, torch.Tensor],
    batch_fn: callable
) -> Dict[str, Any]:
    """
    Compare streaming vs batch computation for validation.

    Args:
        model: Model to analyze
        batch: Input batch
        batch_fn: Function for batch computation

    Returns:
        Comparison results
    """
    # Streaming computation
    streamer = StreamingOVUComputer(model)
    streaming_results = streamer.compute_streaming_ovu(batch)

    # Batch computation
    batch_results = batch_fn(model, batch)

    # Compare top heads
    streaming_top = streaming_results['head_contributions'][:10]
    batch_top = batch_results.get('head_contributions', [])[:10]

    # Compute agreement
    streaming_heads = {(h['layer'], h['head']) for h in streaming_top}
    batch_heads = {(h['layer'], h['head']) for h in batch_top}
    overlap = len(streaming_heads & batch_heads)
    agreement = overlap / max(len(streaming_heads), 1)

    return {
        'streaming_results': streaming_results,
        'batch_results': batch_results,
        'top_head_agreement': agreement,
        'streaming_opportunities': streaming_results['total_opportunities_processed'],
        'batch_opportunities': batch_results.get('opportunity_count', 0),
        'note': f'{agreement*100:.1f}% agreement between streaming and batch modes'
    }


if __name__ == "__main__":
    # Example usage
    import transformers

    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")

    # Create test input
    text = "The quick brown fox jumps over the lazy dog. " * 10
    inputs = tokenizer(text, return_tensors="pt")

    # Run streaming computation
    streamer = StreamingOVUComputer(model)
    results = streamer.compute_streaming_ovu(inputs)

    print(f"Processed {results['total_opportunities_processed']} opportunities")
    print(f"Top 5 heads by OV→U contribution:")
    for head in results['head_contributions'][:5]:
        print(f"  {head['layer_head']}: {head['ov_contribution']:.4f} (±{head['std']:.4f})")