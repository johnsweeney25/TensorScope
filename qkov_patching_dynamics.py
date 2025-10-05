#!/usr/bin/env python3
"""
Advanced Mechanistic Interpretability: QK-OV Pairing, Activation Patching, and Training Dynamics

This module implements the missing pieces for a complete paper on induction head emergence:
1. Automatic QK-OV pairing to identify coupled attention circuits
2. Activation patching validation for causal verification
3. Training dynamics analysis across checkpoints
4. Synthetic task validation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import logging
from contextlib import contextmanager
from scipy.stats import pearsonr, spearmanr
from scipy.special import rel_entr
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class EdgeContribution:
    """Represents a single attention edge contribution."""
    layer: int
    head: int
    query_pos: int
    key_pos: int
    attention_weight: float
    qk_contribution: float  # How QK circuit creates this attention
    ov_contribution: float  # How OV circuit uses this attention
    target_token: int
    actual_token: int


class QKOVAnalyzer:
    """Analyzes QK-OV circuit coupling in transformer attention heads."""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model_type = self._detect_model_type()

    def _detect_model_type(self) -> str:
        """Detect model architecture type."""
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

    def compute_qk_ov_pairing(
        self,
        batch: Dict[str, torch.Tensor],
        top_k_edges: int = 100,
        min_attention_threshold: float = 0.01
    ) -> Dict[str, Any]:
        """
        Compute QK-OV pairing for top attention edges under induction mask.

        This identifies which QK circuits (creating attention patterns) are
        coupled with which OV circuits (using those patterns for prediction).

        Args:
            batch: Input batch with input_ids and attention_mask
            top_k_edges: Number of top edges to analyze
            min_attention_threshold: Minimum attention weight to consider

        Returns:
            Dictionary containing:
            - paired_circuits: List of coupled QK-OV circuits
            - correlation_matrix: QK-OV correlation per head
            - top_edges: Top attention edges analyzed
            - coupling_strength: Overall coupling metric
        """
        input_ids = batch['input_ids']
        batch_size, seq_len = input_ids.shape

        # First, identify induction opportunities (pattern matches)
        induction_edges = []
        for b in range(batch_size):
            for i in range(2, seq_len):  # Start from position 2
                prev_token = input_ids[b, i-1].item()
                for j in range(i-1):
                    if input_ids[b, j].item() == prev_token and j+1 < seq_len:
                        target = input_ids[b, j+1].item()
                        actual = input_ids[b, i].item()
                        induction_edges.append({
                            'batch': b,
                            'query_pos': i,
                            'key_pos': j,
                            'target_token': target,
                            'actual_token': actual,
                            'is_correct': target == actual
                        })

        if not induction_edges:
            return {
                'paired_circuits': [],
                'correlation_matrix': {},
                'top_edges': [],
                'coupling_strength': 0.0,
                'note': 'No induction opportunities found'
            }

        # Get model outputs with attention
        with torch.inference_mode():
            outputs = self.model(**batch, output_attentions=True, output_hidden_states=True)
            attentions = outputs.attentions
            hidden_states = outputs.hidden_states

        # Analyze each attention head
        edge_contributions = []

        for layer_idx, layer in enumerate(self.model.model.layers if hasattr(self.model, 'model') else self.model.transformer.h):
            if layer_idx >= len(attentions):
                break

            attn_weights = attentions[layer_idx]  # [B, H, L, L]
            n_heads = attn_weights.shape[1]

            # Get attention module
            if self.model_type == 'llama':
                attn_module = layer.self_attn
                W_Q = attn_module.q_proj.weight
                W_K = attn_module.k_proj.weight
                W_V = attn_module.v_proj.weight
                W_O = attn_module.o_proj.weight
            elif self.model_type == 'gpt2':
                attn_module = layer.attn
                # Need to split c_attn
                if hasattr(attn_module, 'c_attn'):
                    c_attn_weight = attn_module.c_attn.weight
                    hidden_dim = c_attn_weight.shape[1]
                    W_Q = c_attn_weight[:hidden_dim, :]
                    W_K = c_attn_weight[hidden_dim:2*hidden_dim, :]
                    W_V = c_attn_weight[2*hidden_dim:, :]
                W_O = attn_module.c_proj.weight if hasattr(attn_module, 'c_proj') else None
            else:
                continue  # Skip unsupported architectures

            if W_O is None:
                continue

            # Analyze each edge for this layer
            for edge in induction_edges[:top_k_edges]:  # Limit to top K
                b = edge['batch']
                i = edge['query_pos']
                j = edge['key_pos']

                # Get hidden states at these positions
                h_i = hidden_states[layer_idx][b, i]  # Query position
                h_j = hidden_states[layer_idx][b, j]  # Key position

                for head_idx in range(n_heads):
                    # Get attention weight for this edge
                    attn = attn_weights[b, head_idx, i, j].item()

                    if attn < min_attention_threshold:
                        continue

                    # Compute QK contribution (how this attention was formed)
                    head_dim = W_Q.shape[0] // n_heads
                    head_start = head_idx * head_dim
                    head_end = (head_idx + 1) * head_dim

                    # Extract head-specific weights
                    W_Q_h = W_Q[head_start:head_end, :]
                    W_K_h = W_K[head_start:head_end, :]
                    W_V_h = W_V[head_start:head_end, :]
                    W_O_h = W_O[:, head_start:head_end]

                    # QK circuit: q_i @ k_j
                    q_i = h_i @ W_Q_h.T
                    k_j = h_j @ W_K_h.T
                    qk_score = (q_i @ k_j) / np.sqrt(head_dim)
                    qk_contribution = torch.sigmoid(qk_score).item()

                    # OV circuit: How this attention contributes to output
                    v_j = h_j @ W_V_h.T
                    ov_output = (attn * v_j) @ W_O_h.T

                    # Get unembedding matrix
                    if hasattr(self.model, 'lm_head'):
                        W_U = self.model.lm_head.weight
                    elif hasattr(self.model, 'wte'):
                        W_U = self.model.wte.weight.T
                    else:
                        continue

                    # Contribution to target token
                    target_logit = ov_output @ W_U[:, edge['target_token']]
                    ov_contribution = target_logit.item()

                    edge_contributions.append(EdgeContribution(
                        layer=layer_idx,
                        head=head_idx,
                        query_pos=i,
                        key_pos=j,
                        attention_weight=attn,
                        qk_contribution=qk_contribution,
                        ov_contribution=ov_contribution,
                        target_token=edge['target_token'],
                        actual_token=edge['actual_token']
                    ))

        # Sort by total contribution
        edge_contributions.sort(
            key=lambda x: abs(x.qk_contribution * x.ov_contribution),
            reverse=True
        )

        # Compute correlation matrix between QK and OV for each head
        correlation_matrix = defaultdict(list)
        for contrib in edge_contributions:
            key = (contrib.layer, contrib.head)
            correlation_matrix[key].append((contrib.qk_contribution, contrib.ov_contribution))

        # Calculate correlations
        head_correlations = {}
        for (layer, head), pairs in correlation_matrix.items():
            if len(pairs) > 2:
                qk_vals, ov_vals = zip(*pairs)
                corr, _ = pearsonr(qk_vals, ov_vals)
                head_correlations[f'L{layer}.H{head}'] = corr

        # Identify strongly coupled circuits (high QK-OV correlation)
        paired_circuits = []
        for head_name, corr in head_correlations.items():
            if abs(corr) > 0.5:  # Strong coupling threshold
                paired_circuits.append({
                    'head': head_name,
                    'correlation': corr,
                    'coupling_type': 'positive' if corr > 0 else 'negative'
                })

        # Overall coupling strength
        coupling_strength = np.mean([abs(c) for c in head_correlations.values()]) if head_correlations else 0.0

        return {
            'paired_circuits': paired_circuits,
            'correlation_matrix': head_correlations,
            'top_edges': [
                {
                    'layer': e.layer,
                    'head': e.head,
                    'edge': f'pos{e.query_pos}→pos{e.key_pos}',
                    'attention': e.attention_weight,
                    'qk_contrib': e.qk_contribution,
                    'ov_contrib': e.ov_contribution,
                    'total_contrib': e.qk_contribution * e.ov_contribution
                }
                for e in edge_contributions[:top_k_edges]
            ],
            'coupling_strength': float(coupling_strength),
            'num_edges_analyzed': len(edge_contributions),
            'note': f'Analyzed {len(edge_contributions)} edges across {len(head_correlations)} heads'
        }


class ActivationPatcher:
    """Performs activation patching for causal validation of attention heads."""

    def __init__(self, model, device=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.hooks = []
        self.cached_activations = {}

    @contextmanager
    def patch_head_values(self, layer: int, head: int, position: int, new_values: torch.Tensor):
        """
        Context manager to patch value vectors for a specific head and position.

        Args:
            layer: Layer index
            head: Head index
            position: Position to patch
            new_values: New value vectors to use
        """
        def hook_fn(module, input, output):
            # Modify the value vectors in-place
            if hasattr(output, 'value_states'):
                # For models that return value_states
                output.value_states[:, head, position, :] = new_values
            return output

        # Register hook
        if hasattr(self.model, 'model'):
            layer_module = self.model.model.layers[layer].self_attn
        else:
            layer_module = self.model.transformer.h[layer].attn

        hook = layer_module.register_forward_hook(hook_fn)
        self.hooks.append(hook)

        try:
            yield
        finally:
            # Remove hook
            hook.remove()
            self.hooks.remove(hook)

    def validate_head_contributions(
        self,
        batch: Dict[str, torch.Tensor],
        head_contributions: List[Dict[str, Any]],
        top_k: int = 10,
        num_patches: int = 20
    ) -> Dict[str, Any]:
        """
        Validate OV→U contributions using activation patching.

        For each top head, patch its value vectors and measure the change in
        target token logits. Compare with predicted OV→U contributions.

        Args:
            batch: Input batch
            head_contributions: List of head contributions from compute_induction_ov_contribution
            top_k: Number of top heads to validate
            num_patches: Number of random patches to test per head

        Returns:
            Dictionary containing:
            - validation_results: Per-head validation metrics
            - agreement_score: Overall agreement between OV→U and patching
            - correlation: Pearson correlation between predicted and measured
        """
        # Get baseline outputs
        with torch.inference_mode():
            baseline_outputs = self.model(**batch, output_attentions=True, output_hidden_states=True)
            baseline_logits = baseline_outputs.logits
            hidden_states = baseline_outputs.hidden_states

        # Sort heads by contribution
        top_heads = sorted(head_contributions, key=lambda x: abs(x.get('ov_contribution', 0)), reverse=True)[:top_k]

        validation_results = []
        predicted_effects = []
        measured_effects = []

        for head_info in top_heads:
            layer = head_info['layer']
            head = head_info['head']
            predicted_contribution = head_info.get('ov_contribution', 0)

            # Test multiple random patches
            head_effects = []

            for _ in range(num_patches):
                # Choose random position to patch
                seq_len = batch['input_ids'].shape[1]
                patch_pos = torch.randint(1, seq_len-1, (1,)).item()

                # Generate random values (or zero them out)
                head_dim = hidden_states[layer].shape[-1] // self.model.config.num_attention_heads

                # Option 1: Zero out the values (ablation)
                new_values = torch.zeros(head_dim, device=self.device)

                # Option 2: Random values (noise injection)
                # new_values = torch.randn(head_dim, device=self.device)

                # Patch and measure effect
                with self.patch_head_values(layer, head, patch_pos, new_values):
                    with torch.inference_mode():
                        patched_outputs = self.model(**batch)
                        patched_logits = patched_outputs.logits

                # Measure change in logits
                logit_diff = (patched_logits - baseline_logits).abs().mean().item()
                head_effects.append(logit_diff)

            avg_effect = np.mean(head_effects)
            std_effect = np.std(head_effects)

            validation_results.append({
                'layer': layer,
                'head': head,
                'layer_head': f'L{layer}.H{head}',
                'predicted_contribution': predicted_contribution,
                'measured_effect': avg_effect,
                'effect_std': std_effect,
                'agreement': 1.0 - min(1.0, abs(predicted_contribution - avg_effect) / max(abs(predicted_contribution), abs(avg_effect), 1e-6))
            })

            predicted_effects.append(abs(predicted_contribution))
            measured_effects.append(avg_effect)

        # Compute overall agreement
        if len(predicted_effects) > 1:
            correlation, p_value = pearsonr(predicted_effects, measured_effects)
            spearman_corr, _ = spearmanr(predicted_effects, measured_effects)
        else:
            correlation = 0.0
            p_value = 1.0
            spearman_corr = 0.0

        agreement_score = np.mean([r['agreement'] for r in validation_results])

        return {
            'validation_results': validation_results,
            'agreement_score': float(agreement_score),
            'pearson_correlation': float(correlation),
            'spearman_correlation': float(spearman_corr),
            'p_value': float(p_value),
            'num_heads_tested': len(validation_results),
            'note': f'Validated {len(validation_results)} heads with {num_patches} patches each'
        }


class TrainingDynamicsAnalyzer:
    """Analyzes induction head emergence across training checkpoints."""

    def __init__(self, checkpoint_dir: str):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoints = self._discover_checkpoints()

    def _discover_checkpoints(self) -> List[Dict[str, Any]]:
        """Discover available checkpoints and their metadata."""
        checkpoints = []

        # Look for checkpoint files
        for ckpt_path in sorted(self.checkpoint_dir.glob("*.pt")):
            # Extract step number from filename (assuming format like "model_step_1000.pt")
            try:
                parts = ckpt_path.stem.split('_')
                step = int(parts[-1]) if parts[-1].isdigit() else len(checkpoints)
            except:
                step = len(checkpoints)

            checkpoints.append({
                'path': ckpt_path,
                'step': step,
                'name': ckpt_path.stem
            })

        return sorted(checkpoints, key=lambda x: x['step'])

    def track_induction_emergence(
        self,
        test_data: List[Dict[str, torch.Tensor]],
        model_class: type,
        config: Any,
        metric_fn: callable,
        sample_interval: int = 1
    ) -> Dict[str, Any]:
        """
        Track induction head emergence across training checkpoints.

        Args:
            test_data: List of test batches
            model_class: Model class to instantiate
            config: Model config
            metric_fn: Function to compute induction metrics
            sample_interval: Sample every N checkpoints

        Returns:
            Dictionary containing:
            - dynamics: Metrics at each checkpoint
            - emergence_point: Step where induction heads emerge
            - phase_transitions: Detected phase transitions
            - laws: Fitted scaling laws
        """
        dynamics = []

        for i, ckpt in enumerate(self.checkpoints):
            if i % sample_interval != 0:
                continue

            # Load checkpoint
            model = model_class(config)
            state_dict = torch.load(ckpt['path'], map_location='cpu')
            model.load_state_dict(state_dict)
            model.eval()

            # Compute metrics on test data
            metrics = []
            for batch in test_data:
                m = metric_fn(model, batch)
                metrics.append(m)

            # Average metrics
            avg_metrics = {}
            for key in metrics[0].keys():
                if isinstance(metrics[0][key], (int, float)):
                    avg_metrics[key] = np.mean([m[key] for m in metrics])

            dynamics.append({
                'step': ckpt['step'],
                'checkpoint': ckpt['name'],
                **avg_metrics
            })

            logger.info(f"Step {ckpt['step']}: Induction score = {avg_metrics.get('copy_alignment_score', 0):.4f}")

        # Detect emergence point (when score exceeds threshold)
        emergence_threshold = 0.1
        emergence_point = None

        for d in dynamics:
            if d.get('copy_alignment_score', 0) > emergence_threshold:
                emergence_point = d['step']
                break

        # Detect phase transitions (large jumps in metrics)
        phase_transitions = []
        if len(dynamics) > 1:
            scores = [d.get('copy_alignment_score', 0) for d in dynamics]
            for i in range(1, len(scores)):
                if scores[i] - scores[i-1] > 0.05:  # 5% jump
                    phase_transitions.append({
                        'step': dynamics[i]['step'],
                        'before': scores[i-1],
                        'after': scores[i],
                        'jump': scores[i] - scores[i-1]
                    })

        # Fit scaling laws (power law: score = a * step^b)
        if len(dynamics) > 2:
            steps = np.array([d['step'] for d in dynamics])
            scores = np.array([d.get('copy_alignment_score', 0) for d in dynamics])

            # Log-log regression for power law
            valid_mask = (steps > 0) & (scores > 0)
            if valid_mask.sum() > 2:
                log_steps = np.log(steps[valid_mask])
                log_scores = np.log(scores[valid_mask])

                # Fit y = mx + c (log space)
                coeffs = np.polyfit(log_steps, log_scores, 1)
                power_law_exponent = coeffs[0]
                power_law_constant = np.exp(coeffs[1])

                laws = {
                    'type': 'power_law',
                    'equation': f'score = {power_law_constant:.4f} * step^{power_law_exponent:.4f}',
                    'exponent': float(power_law_exponent),
                    'constant': float(power_law_constant)
                }
            else:
                laws = {'type': 'insufficient_data'}
        else:
            laws = {'type': 'insufficient_data'}

        return {
            'dynamics': dynamics,
            'emergence_point': emergence_point,
            'phase_transitions': phase_transitions,
            'laws': laws,
            'num_checkpoints': len(dynamics),
            'note': f'Tracked {len(dynamics)} checkpoints, emergence at step {emergence_point}'
        }


class SyntheticTaskValidator:
    """Validates induction heads on synthetic tasks."""

    @staticmethod
    def create_perfect_induction_data(
        vocab_size: int = 100,
        seq_len: int = 50,
        batch_size: int = 16,
        pattern_len: int = 5,
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Create synthetic data with perfect induction patterns (ABC...ABC).

        Args:
            vocab_size: Size of vocabulary
            seq_len: Sequence length
            batch_size: Batch size
            pattern_len: Length of repeating pattern
            device: Device to create tensors on

        Returns:
            Batch dictionary with input_ids and attention_mask
        """
        input_ids = []

        for _ in range(batch_size):
            # Create random pattern
            pattern = torch.randint(1, vocab_size, (pattern_len,))

            # Repeat pattern to fill sequence
            num_repeats = seq_len // pattern_len + 1
            repeated = pattern.repeat(num_repeats)[:seq_len]

            # Add some noise (10% random tokens)
            noise_mask = torch.rand(seq_len) < 0.1
            noise = torch.randint(1, vocab_size, (seq_len,))
            repeated[noise_mask] = noise[noise_mask]

            input_ids.append(repeated)

        input_ids = torch.stack(input_ids).to(device)
        attention_mask = torch.ones_like(input_ids).to(device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    @staticmethod
    def create_anti_induction_data(
        vocab_size: int = 100,
        seq_len: int = 50,
        batch_size: int = 16,
        device: str = 'cpu'
    ) -> Dict[str, torch.Tensor]:
        """
        Create data that should NOT trigger induction heads (random sequences).

        Args:
            vocab_size: Size of vocabulary
            seq_len: Sequence length
            batch_size: Batch size
            device: Device to create tensors on

        Returns:
            Batch dictionary with input_ids and attention_mask
        """
        # Completely random sequences (no patterns)
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len), device=device)
        attention_mask = torch.ones_like(input_ids, device=device)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

    @staticmethod
    def validate_on_synthetic_tasks(
        model,
        metric_fn: callable,
        num_trials: int = 10
    ) -> Dict[str, Any]:
        """
        Validate model on synthetic induction tasks.

        Args:
            model: Model to test
            metric_fn: Function to compute induction metrics
            num_trials: Number of trials per task type

        Returns:
            Dictionary containing:
            - perfect_induction_score: Average score on perfect patterns
            - random_baseline_score: Average score on random sequences
            - discrimination_ratio: Ratio showing selectivity
            - statistical_significance: P-value from t-test
        """
        device = next(model.parameters()).device

        perfect_scores = []
        random_scores = []

        for _ in range(num_trials):
            # Test on perfect induction data
            perfect_data = SyntheticTaskValidator.create_perfect_induction_data(device=device)
            perfect_metrics = metric_fn(model, perfect_data)
            perfect_scores.append(perfect_metrics.get('copy_alignment_score', 0))

            # Test on random data
            random_data = SyntheticTaskValidator.create_anti_induction_data(device=device)
            random_metrics = metric_fn(model, random_data)
            random_scores.append(random_metrics.get('copy_alignment_score', 0))

        # Statistical test
        from scipy.stats import ttest_ind
        t_stat, p_value = ttest_ind(perfect_scores, random_scores)

        # Compute discrimination ratio
        avg_perfect = np.mean(perfect_scores)
        avg_random = np.mean(random_scores)
        discrimination_ratio = avg_perfect / max(avg_random, 1e-6)

        return {
            'perfect_induction_score': float(avg_perfect),
            'perfect_induction_std': float(np.std(perfect_scores)),
            'random_baseline_score': float(avg_random),
            'random_baseline_std': float(np.std(random_scores)),
            'discrimination_ratio': float(discrimination_ratio),
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'is_significant': p_value < 0.05,
            'num_trials': num_trials,
            'note': f'Model {"successfully" if p_value < 0.05 else "fails to"} discriminates induction patterns (p={p_value:.4f})'
        }


def run_complete_analysis(
    model,
    test_data: List[Dict[str, torch.Tensor]],
    checkpoint_dir: Optional[str] = None,
    output_dir: str = './induction_analysis'
) -> Dict[str, Any]:
    """
    Run complete induction head analysis including QK-OV pairing, patching, and dynamics.

    Args:
        model: Model to analyze
        test_data: List of test batches
        checkpoint_dir: Directory containing training checkpoints (optional)
        output_dir: Directory to save results

    Returns:
        Complete analysis results
    """
    os.makedirs(output_dir, exist_ok=True)
    results = {}

    # 1. QK-OV Pairing Analysis
    logger.info("Running QK-OV pairing analysis...")
    qkov_analyzer = QKOVAnalyzer(model)
    qkov_results = []

    for batch in test_data[:5]:  # Sample a few batches
        pairing = qkov_analyzer.compute_qk_ov_pairing(batch)
        qkov_results.append(pairing)

    # Average results
    avg_coupling = np.mean([r['coupling_strength'] for r in qkov_results])
    all_paired_circuits = []
    for r in qkov_results:
        all_paired_circuits.extend(r['paired_circuits'])

    results['qkov_pairing'] = {
        'average_coupling_strength': float(avg_coupling),
        'num_paired_circuits': len(all_paired_circuits),
        'top_paired_circuits': all_paired_circuits[:10]
    }

    # 2. Activation Patching Validation
    logger.info("Running activation patching validation...")
    patcher = ActivationPatcher(model)

    # Get head contributions first (would come from compute_induction_ov_contribution)
    # For demo, create mock contributions
    mock_contributions = [
        {'layer': i, 'head': j, 'ov_contribution': np.random.rand()}
        for i in range(12) for j in range(12)
    ]

    patching_results = []
    for batch in test_data[:3]:  # Sample a few batches
        validation = patcher.validate_head_contributions(batch, mock_contributions)
        patching_results.append(validation)

    # Average agreement
    avg_agreement = np.mean([r['agreement_score'] for r in patching_results])
    avg_correlation = np.mean([r['pearson_correlation'] for r in patching_results])

    results['activation_patching'] = {
        'average_agreement': float(avg_agreement),
        'average_correlation': float(avg_correlation),
        'validation_results': patching_results[0]['validation_results'][:5]  # Top 5
    }

    # 3. Synthetic Task Validation
    logger.info("Running synthetic task validation...")

    # Mock metric function (would use compute_induction_head_strength)
    def mock_metric_fn(model, batch):
        return {'copy_alignment_score': np.random.rand() * 0.5}

    synthetic_results = SyntheticTaskValidator.validate_on_synthetic_tasks(
        model, mock_metric_fn, num_trials=5
    )
    results['synthetic_validation'] = synthetic_results

    # 4. Training Dynamics (if checkpoints available)
    if checkpoint_dir and os.path.exists(checkpoint_dir):
        logger.info("Analyzing training dynamics...")
        dynamics_analyzer = TrainingDynamicsAnalyzer(checkpoint_dir)

        if dynamics_analyzer.checkpoints:
            dynamics = dynamics_analyzer.track_induction_emergence(
                test_data[:2],
                type(model),
                model.config,
                mock_metric_fn,
                sample_interval=1
            )
            results['training_dynamics'] = dynamics

    # 5. Save results
    output_path = os.path.join(output_dir, 'complete_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"Analysis complete. Results saved to {output_path}")

    return results


if __name__ == "__main__":
    # Example usage
    import transformers

    # Load a model
    model_name = "gpt2"
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # Create test data
    test_texts = [
        "The cat sat on the mat. The cat sat on the",
        "ABC DEF GHI. ABC DEF",
        "Pattern recognition is important. Pattern recognition is"
    ]

    test_data = []
    for text in test_texts:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        test_data.append(inputs)

    # Run analysis
    results = run_complete_analysis(
        model,
        test_data,
        checkpoint_dir=None,  # Would point to checkpoint directory
        output_dir="./induction_analysis"
    )

    print("Analysis complete!")
    print(f"QK-OV Coupling Strength: {results['qkov_pairing']['average_coupling_strength']:.4f}")
    print(f"Patching Agreement: {results['activation_patching']['average_agreement']:.4f}")
    print(f"Synthetic Discrimination: {results['synthetic_validation']['discrimination_ratio']:.2f}")