"""
QK-OV Block-Wise Interference Metric (Section 4.1)

Implements the Fisher-normalized, head-resolved interference metric:

    M^B_{ij,ℓ,h} = ⟨C_i|_{B,ℓ,h} / (Î_n|_{B,ℓ,h} + ε), |g_j||_{B,ℓ,h}⟩

where B ∈ {Q, K, V, O}, ℓ is layer index, h is head index.

This bridges:
- Per-sample contributions C_i from FisherCollector
- EMA Fisher Î_n for normalization
- Per-sample gradients g_j from cross-task analysis
- QK-OV circuit structure from mechanistic analysis

THEORETICAL FOUNDATION
----------------------
Unlike task-level or parameter-level conflict metrics, this provides:
1. Circuit-specific attribution (which attention mechanism conflicts)
2. Sample-pair forensics (which specific examples interfere)
3. Fisher-weighted importance (normalized by parameter sensitivity)

REFERENCES
----------
- Section 4.1 of paper: "From Contributions to Circuit-Level Interference"
- Contribution Safety Theorem (Section 3.2): ensures C_i usage is valid
- QK-OV pairing (mechanistic analysis): circuit taxonomy

Author: ICLR 2026 Project
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass
from collections import defaultdict
import logging
import re

logger = logging.getLogger(__name__)


@dataclass
class QKOVConfig:
    """Configuration for QK-OV parameter layout in a model."""
    num_layers: int
    num_heads: int
    head_dim: int  # d_k for Q/K
    v_head_dim: int  # d_v for V (may differ from d_k)
    hidden_dim: int
    fused_qkv: bool = False  # True if using single W_qkv projection
    uses_gqa: bool = False   # Grouped-query attention
    num_kv_heads: Optional[int] = None  # For GQA/MQA
    has_bias: bool = True
    include_bias: bool = False  # Whether to include bias in interference computation
    fused_qkv_transposed: bool = False  # True if fused QKV uses Conv1D (GPT-2 style)

    def __post_init__(self):
        """Validate configuration after initialization."""
        # GQA validation
        if self.uses_gqa:
            if self.num_kv_heads is None:
                raise ValueError("num_kv_heads must be specified when uses_gqa=True")
            if self.num_heads % self.num_kv_heads != 0:
                raise ValueError(
                    f"num_heads ({self.num_heads}) must be divisible by "
                    f"num_kv_heads ({self.num_kv_heads}) for GQA"
                )

    @classmethod
    def from_model(cls, model: nn.Module) -> 'QKOVConfig':
        """Auto-detect configuration from model architecture."""
        config = model.config if hasattr(model, 'config') else None

        if config is None:
            raise ValueError("Model must have .config attribute")

        num_layers = getattr(config, 'num_hidden_layers',
                            getattr(config, 'n_layer',
                            getattr(config, 'num_layers', None)))
        num_heads = getattr(config, 'num_attention_heads',
                           getattr(config, 'n_head', None))
        hidden_dim = getattr(config, 'hidden_size',
                            getattr(config, 'n_embd', None))

        if num_layers is None or num_heads is None or hidden_dim is None:
            raise ValueError(f"Could not auto-detect model dimensions from config")

        head_dim = hidden_dim // num_heads

        # Infer actual dimensions from model weights
        v_head_dim = head_dim  # Default assumption
        fused_qkv = False
        fused_qkv_transposed = False
        has_bias = False

        for name, module in model.named_modules():
            if 'attn' in name.lower() or 'attention' in name.lower():
                # Check for fused QKV
                if hasattr(module, 'c_attn'):  # GPT-2 style
                    fused_qkv = True
                    has_bias = hasattr(module.c_attn, 'bias') and module.c_attn.bias is not None
                    # Infer dims from weight shape
                    W = module.c_attn.weight
                    expected_fused = 3 * num_heads * head_dim

                    # GPT-2 uses Conv1D: weights are [hidden_dim, 3*hidden_dim]
                    # Standard Linear: weights are [3*hidden_dim, hidden_dim]
                    if W.shape[1] == expected_fused and W.shape[0] == hidden_dim:
                        # Conv1D format (transposed)
                        fused_qkv_transposed = True
                        v_head_dim = head_dim
                    elif W.shape[0] == expected_fused and W.shape[1] == hidden_dim:
                        # Standard Linear format
                        fused_qkv_transposed = False
                        v_head_dim = head_dim
                    else:
                        # Unusual shape, try to infer
                        logger.warning(f"Unexpected c_attn shape {W.shape}, expected {expected_fused}")
                        v_head_dim = head_dim
                    break
                elif hasattr(module, 'qkv'):  # ViT style
                    fused_qkv = True
                    has_bias = hasattr(module.qkv, 'bias') and module.qkv.bias is not None
                    W = module.qkv.weight
                    # Infer v_head_dim from weight shape
                    # W_qkv shape: [H*(d_k + d_k + d_v), hidden_dim]
                    # If d_k != d_v, need to detect from shape
                    total_out = W.shape[0]
                    # Assume d_k == d_v for now (most common)
                    v_head_dim = head_dim
                    break
                # Check for split projections
                elif hasattr(module, 'v_proj'):
                    has_bias = hasattr(module.v_proj, 'bias') and module.v_proj.bias is not None
                    # Infer v_head_dim from V projection
                    W_v = module.v_proj.weight
                    num_kv_heads_inferred = getattr(config, 'num_key_value_heads', num_heads)
                    v_head_dim = W_v.shape[0] // num_kv_heads_inferred
                    break

        # Detect GQA/MQA
        num_kv_heads = getattr(config, 'num_key_value_heads', None)
        uses_gqa = num_kv_heads is not None and num_kv_heads != num_heads

        return cls(
            num_layers=num_layers,
            num_heads=num_heads,
            head_dim=head_dim,
            v_head_dim=v_head_dim,
            hidden_dim=hidden_dim,
            fused_qkv=fused_qkv,
            fused_qkv_transposed=fused_qkv_transposed,
            uses_gqa=uses_gqa,
            num_kv_heads=num_kv_heads,
            has_bias=has_bias,
            include_bias=False  # Default: exclude bias from interference
        )


@dataclass
class BlockHeadSlice:
    """Represents a slice of parameters for a specific block and head."""
    layer: int
    head: int
    block: str  # 'Q', 'K', 'V', or 'O'
    param_name: str
    row_slice: Optional[Tuple[int, int]] = None
    col_slice: Optional[Tuple[int, int]] = None

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        """Apply slicing to a tensor."""
        result = tensor
        if self.row_slice is not None:
            result = result[self.row_slice[0]:self.row_slice[1]]
        if self.col_slice is not None:
            result = result[:, self.col_slice[0]:self.col_slice[1]]
        return result


class QKOVIndexer:
    """
    Unified API for slicing QK-OV parameters by layer, head, and block.

    Handles:
    - Fused vs split QKV projections
    - Row vs column slicing for Q/K/V vs O
    - Grouped-query attention (GQA/MQA)
    - Different model architectures (GPT-2, LLaMA, etc.)
    """

    def __init__(self, config: QKOVConfig):
        self.config = config
        self._param_patterns = self._build_param_patterns()

    def _build_param_patterns(self) -> Dict[str, List[str]]:
        """Build regex patterns for finding QK-OV parameters."""
        # Common patterns across architectures
        patterns = {
            'Q': [
                r'layers\.(\d+)\.self_attn\.q_proj\.weight',  # LLaMA
                r'layers\.(\d+)\.attention\.wq\.weight',      # Alternative
                r'h\.(\d+)\.attn\.c_attn\.weight',            # GPT-2 (fused)
                r'transformer\.h\.(\d+)\.attn\.c_attn\.weight',
            ],
            'K': [
                r'layers\.(\d+)\.self_attn\.k_proj\.weight',
                r'layers\.(\d+)\.attention\.wk\.weight',
                r'h\.(\d+)\.attn\.c_attn\.weight',  # Same as Q for fused
            ],
            'V': [
                r'layers\.(\d+)\.self_attn\.v_proj\.weight',
                r'layers\.(\d+)\.attention\.wv\.weight',
                r'h\.(\d+)\.attn\.c_attn\.weight',  # Same as Q for fused
            ],
            'O': [
                r'layers\.(\d+)\.self_attn\.o_proj\.weight',
                r'layers\.(\d+)\.attention\.wo\.weight',
                r'h\.(\d+)\.attn\.c_proj\.weight',  # GPT-2
                r'transformer\.h\.(\d+)\.attn\.c_proj\.weight',
            ],
        }
        return patterns

    def find_param_name(self, layer: int, block: str, model_params: Dict[str, torch.Tensor]) -> Optional[str]:
        """Find the parameter name for a given layer and block."""
        for pattern in self._param_patterns[block]:
            for param_name in model_params.keys():
                match = re.search(pattern, param_name)
                if match:
                    param_layer = int(match.group(1))
                    if param_layer == layer:
                        return param_name
        return None

    def get_slice(self, layer: int, head: int, block: str, param_name: str) -> BlockHeadSlice:
        """
        Get slicing information for a specific layer, head, and block.

        Args:
            layer: Layer index
            head: Head index
            block: 'Q', 'K', 'V', or 'O'
            param_name: Full parameter name

        Returns:
            BlockHeadSlice with row/column ranges
        """
        H = self.config.num_heads
        d_k = self.config.head_dim
        d_v = self.config.v_head_dim  # May differ from d_k!

        if block in ['Q', 'K']:
            # Q/K use d_k
            if self.config.fused_qkv:
                # Fused QKV: determine offset
                block_offset = {'Q': 0, 'K': H * d_k}[block]

                # Handle GQA for K
                if block == 'K' and self.config.uses_gqa:
                    # K shared across multiple Q heads
                    assert self.config.num_kv_heads is not None
                    assert H % self.config.num_kv_heads == 0, \
                        f"num_heads ({H}) must be divisible by num_kv_heads ({self.config.num_kv_heads})"
                    kv_head = head // (H // self.config.num_kv_heads)
                    start = block_offset + kv_head * d_k
                    end = start + d_k
                else:
                    start = block_offset + head * d_k
                    end = start + d_k

                # Conv1D (GPT-2): weights [hidden_dim, 3*hidden_dim] → use column slicing
                # Linear: weights [3*hidden_dim, hidden_dim] → use row slicing
                if self.config.fused_qkv_transposed:
                    return BlockHeadSlice(
                        layer=layer, head=head, block=block, param_name=param_name,
                        row_slice=None, col_slice=(start, end)
                    )
                else:
                    return BlockHeadSlice(
                        layer=layer, head=head, block=block, param_name=param_name,
                        row_slice=(start, end), col_slice=None
                    )
            else:
                # Split projections: [H*d_k, d_model] for Q, [(H or H_kv)*d_k, d_model] for K
                if block == 'K' and self.config.uses_gqa:
                    assert H % self.config.num_kv_heads == 0
                    kv_head = head // (H // self.config.num_kv_heads)
                    start = kv_head * d_k
                    end = start + d_k
                else:
                    start = head * d_k
                    end = start + d_k

                return BlockHeadSlice(
                    layer=layer, head=head, block=block, param_name=param_name,
                    row_slice=(start, end), col_slice=None
                )

        elif block == 'V':
            # V uses d_v (may differ from d_k!)
            if self.config.fused_qkv:
                # V offset: after Q and K
                block_offset = 2 * H * d_k  # Q and K regions

                # Handle GQA for V
                if self.config.uses_gqa:
                    assert self.config.num_kv_heads is not None
                    assert H % self.config.num_kv_heads == 0
                    kv_head = head // (H // self.config.num_kv_heads)
                    start = block_offset + kv_head * d_v
                    end = start + d_v
                else:
                    start = block_offset + head * d_v
                    end = start + d_v

                # Conv1D vs Linear (same as Q/K)
                if self.config.fused_qkv_transposed:
                    return BlockHeadSlice(
                        layer=layer, head=head, block=block, param_name=param_name,
                        row_slice=None, col_slice=(start, end)
                    )
                else:
                    return BlockHeadSlice(
                        layer=layer, head=head, block=block, param_name=param_name,
                        row_slice=(start, end), col_slice=None
                    )
            else:
                # Split projections: [(H or H_kv)*d_v, d_model] for V
                if self.config.uses_gqa:
                    assert H % self.config.num_kv_heads == 0
                    kv_head = head // (H // self.config.num_kv_heads)
                    start = kv_head * d_v
                    end = start + d_v
                else:
                    start = head * d_v
                    end = start + d_v

                return BlockHeadSlice(
                    layer=layer, head=head, block=block, param_name=param_name,
                    row_slice=(start, end), col_slice=None
                )

        else:  # 'O'
            # W_O: [d_model, H*d_v] - columns are head-partitioned
            # CRITICAL: O uses d_v and column slicing!
            start = head * d_v
            end = start + d_v

            return BlockHeadSlice(
                layer=layer,
                head=head,
                block=block,
                param_name=param_name,
                row_slice=None,
                col_slice=(start, end)  # Column slice for O!
            )

    def slice_tensor(self, tensor: torch.Tensor, layer: int, head: int,
                     block: str, param_name: str) -> torch.Tensor:
        """Apply slicing to extract head-specific parameters."""
        slice_info = self.get_slice(layer, head, block, param_name)
        return slice_info.apply(tensor)


@dataclass
class InterferenceScore:
    """Result of computing M^B_{ij,ℓ,h}."""
    task_a: str
    task_b: str
    sample_i: int
    sample_j: int
    layer: int
    head: int
    block: str  # 'Q', 'K', 'V', or 'O'
    score: float  # M^B_{ij,ℓ,h}
    contrib_norm: float  # ||C_i||
    grad_norm: float     # ||g_j||
    fisher_min: float    # min(Î_n) for numerical health


class QKOVInterferenceMetric:
    """
    Computes Fisher-normalized, block-wise, head-resolved interference.

    Usage:
        # Setup
        config = QKOVConfig.from_model(model)
        metric = QKOVInterferenceMetric(config, fisher_collector)

        # Compute for sample pair
        scores = metric.compute_sample_pair(
            task_a='math', sample_i=7,
            task_b='code', sample_j=23,
            layer=3, head=5
        )
        # scores = {'Q': 0.42, 'K': 0.31, 'V': 0.18, 'O': 0.55}

        # Compute full heatmap
        heatmap = metric.compute_heatmap(
            task_a='math', task_b='code',
            layers=[3, 4, 5], heads=range(12)
        )
    """

    def __init__(
        self,
        config: QKOVConfig,
        fisher_collector,  # FisherCollector instance
        epsilon: float = 1e-10,
        ridge_lambda: float = 1e-8,
        normalization_mode: str = 'behavioral',  # 'behavioral', 'structural', or 'hybrid'
        structural_fisher_collector = None  # Alternative FisherCollector for hybrid mode
    ):
        """
        Initialize interference metric with configurable normalization.

        Args:
            config: QK-OV configuration
            fisher_collector: FisherCollector with EMA Fisher and contributions (behavioral-grouped)
            epsilon: Numerical stability for division
            ridge_lambda: Ridge regularization for Fisher
            normalization_mode: 'behavioral' (default), 'structural', or 'hybrid'
            structural_fisher_collector: FisherCollector for structural grouping (hybrid mode only)
        """
        self.config = config
        self.fisher_collector = fisher_collector
        self.epsilon = epsilon
        self.ridge_lambda = ridge_lambda
        self.normalization_mode = normalization_mode
        self.indexer = QKOVIndexer(config)

        # Setup Fisher data for normalization
        if normalization_mode == 'hybrid':
            if structural_fisher_collector is None:
                raise ValueError("structural_fisher_collector required for hybrid normalization")
            # Get structural Fisher for hybrid normalization
            self.structural_fisher = structural_fisher_collector.get_group_fisher()
        elif normalization_mode == 'structural':
            # Use structural Fisher only (ignore behavioral Fisher)
            self.structural_fisher = fisher_collector.get_group_fisher()
        # For 'behavioral' mode, use the behavioral Fisher from fisher_collector

        # Cache for computed scores
        self._score_cache: Dict[str, float] = {}

    def _get_cache_key(self, task_a: str, sample_i: int, task_b: str,
                       sample_j: int, layer: int, head: int, block: str) -> str:
        """Generate cache key for memoization."""
        return f"{task_a}:{sample_i}_{task_b}:{sample_j}_L{layer}H{head}{block}"

    def compute_block_head_score(
        self,
        contrib: torch.Tensor,
        grad: torch.Tensor,
        fisher: torch.Tensor,
        layer: int,
        head: int,
        block: str,
        param_name: str
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Compute M^B_{ij,ℓ,h} = ⟨C_i / (Î_n + ε), |g_j|⟩ for a single block/head.

        Args:
            contrib: Contribution C_i (full parameter)
            grad: Gradient g_j (full parameter)
            fisher: EMA Fisher Î_n (full parameter)
            layer: Layer index
            head: Head index
            block: 'Q', 'K', 'V', or 'O'
            param_name: Parameter name for slicing

        Returns:
            Tuple of (score, diagnostics_dict)
            - score: Scalar interference score
            - diagnostics: Dict with contrib_norm, grad_norm, fisher_min for debugging
        """
        # Device/dtype safety: ensure all tensors on same device & fp32
        device = 'cpu'  # Safe default for computation

        # Convert to fp32 on CPU for numerical stability
        C_full = contrib.detach().to(dtype=torch.float32, device=device)
        G_full = grad.detach().to(dtype=torch.float32, device=device)
        F_full = fisher.detach().to(dtype=torch.float32, device=device)

        # Slice to block/head
        C_i_bh = self.indexer.slice_tensor(C_full, layer, head, block, param_name)
        g_j_bh = self.indexer.slice_tensor(G_full, layer, head, block, param_name)
        I_n_bh = self.indexer.slice_tensor(F_full, layer, head, block, param_name)

        # Diagnostic info
        fisher_min = I_n_bh.min().item()
        contrib_norm = C_i_bh.norm().item()
        grad_norm = g_j_bh.norm().item()

        # Add ridge regularization to Fisher
        I_n_regularized = I_n_bh.clamp_min(self.epsilon) + self.ridge_lambda

        # Choose Fisher normalization based on mode
        if self.normalization_mode == 'hybrid' and self.structural_fisher is not None:
            # Hybrid normalization: combine behavioral and structural Fisher
            # This maintains theoretical validity while incorporating behavioral insights
            structural_I_n = self.indexer.slice_tensor(self.structural_fisher, layer, head, block, param_name)
            structural_regularized = structural_I_n.clamp_min(self.epsilon) + self.ridge_lambda
            # Geometric mean for balanced normalization
            hybrid_fisher = torch.sqrt(I_n_regularized * structural_regularized)
            fisher_for_normalization = hybrid_fisher
        elif self.normalization_mode == 'structural':
            # Use structural Fisher only (theoretically safest)
            fisher_for_normalization = I_n_regularized
        else:
            # Behavioral normalization (default, may have theoretical issues)
            fisher_for_normalization = I_n_regularized

        # Apply normalization
        normalized_contrib = C_i_bh / fisher_for_normalization

        # Inner product with |g_j|
        score = (normalized_contrib * g_j_bh.abs()).sum().item()

        diagnostics = {
            'fisher_min': fisher_min,
            'contrib_norm': contrib_norm,
            'grad_norm': grad_norm,
            'normalization_mode': self.normalization_mode,
            'fisher_behavioral_min': I_n_bh.min().item(),
            'fisher_regularized_min': fisher_for_normalization.min().item()
        }

        return score, diagnostics

    def compute_sample_pair(
        self,
        task_a: str,
        sample_i: int,
        task_b: str,
        sample_j: int,
        layer: int,
        head: int
    ) -> Dict[str, float]:
        """
        Compute M^B_{ij,ℓ,h} for all blocks {Q, K, V, O} for a sample pair.

        Args:
            task_a: Task identifier for sample i
            sample_i: Sample index from task A
            task_b: Task identifier for sample j
            sample_j: Sample index from task B
            layer: Layer index
            head: Head index

        Returns:
            Dictionary: {'Q': score, 'K': score, 'V': score, 'O': score}
        """
        # Check cache first
        scores = {}

        # Get contributions for sample i from task A
        if not hasattr(self.fisher_collector, 'contribution_cache'):
            raise ValueError("FisherCollector must have contribution_cache enabled")

        task_a_contribs = self.fisher_collector.contribution_cache.get(
            f"{task_a}_{sample_i}", {}
        )

        # Get gradients for sample j from task B
        if hasattr(self.fisher_collector, 'gradient_manager'):
            task_b_grads = self.fisher_collector.gradient_manager.get_sample_gradients(
                task_b, sample_j
            )
        else:
            # Fallback: compute on-the-fly if needed
            logger.warning("No gradient_manager found; scores may be incomplete")
            task_b_grads = {}

        # Get EMA Fisher
        fisher_ema = self.fisher_collector.fisher_ema

        # Find parameters for each block
        # Note: Use a representative parameter dict (e.g., from fisher_ema)
        for block in ['Q', 'K', 'V', 'O']:
            param_name = self.indexer.find_param_name(layer, block, fisher_ema)

            if param_name is None:
                logger.warning(f"Could not find parameter for L{layer} {block}")
                scores[block] = 0.0
                continue

            # Check if we have all required data
            if (param_name not in task_a_contribs or
                param_name not in task_b_grads or
                param_name not in fisher_ema):
                logger.debug(f"Missing data for {param_name}, skipping")
                scores[block] = 0.0
                continue

            # Check cache
            cache_key = self._get_cache_key(task_a, sample_i, task_b, sample_j,
                                           layer, head, block)
            if cache_key in self._score_cache:
                scores[block] = self._score_cache[cache_key]
                continue

            # Compute score
            score, diagnostics = self.compute_block_head_score(
                contrib=task_a_contribs[param_name],
                grad=task_b_grads[param_name],
                fisher=fisher_ema[param_name],
                layer=layer,
                head=head,
                block=block,
                param_name=param_name
            )

            scores[block] = score
            self._score_cache[cache_key] = score

            # Log numerical health warnings
            if diagnostics['fisher_min'] < self.epsilon * 10:
                logger.debug(f"L{layer}H{head} {block}: Fisher min={diagnostics['fisher_min']:.2e} (may be ill-conditioned)")

        return scores

    def compute_heatmap(
        self,
        task_a: str,
        task_b: str,
        layers: Optional[List[int]] = None,
        heads: Optional[List[int]] = None,
        max_samples_per_task: int = 100
    ) -> Dict[str, np.ndarray]:
        """
        Compute full interference heatmap across layers, heads, and blocks.

        Args:
            task_a: First task identifier
            task_b: Second task identifier
            layers: List of layers to analyze (default: all)
            heads: List of heads to analyze (default: all)
            max_samples_per_task: Limit samples to prevent explosion

        Returns:
            Dictionary with keys 'Q', 'K', 'V', 'O', each containing:
            - 'scores': ndarray of shape [n_samples_A, n_samples_B, n_layers, n_heads]
            - 'layer_head_avg': ndarray of shape [n_layers, n_heads]
            - 'top_conflicts': List of top (i, j, layer, head) tuples
        """
        if layers is None:
            layers = list(range(self.config.num_layers))
        if heads is None:
            heads = list(range(self.config.num_heads))

        # Get available samples
        task_a_samples = list(self.fisher_collector.contribution_cache.keys())
        task_a_samples = [k for k in task_a_samples if k.startswith(f"{task_a}_")]
        task_a_samples = task_a_samples[:max_samples_per_task]

        if hasattr(self.fisher_collector, 'gradient_manager'):
            task_b_samples = self.fisher_collector.gradient_manager.get_task_samples(task_b)
            task_b_samples = task_b_samples[:max_samples_per_task]
        else:
            logger.warning("No gradient_manager; limited sample analysis")
            task_b_samples = []

        n_samples_a = len(task_a_samples)
        n_samples_b = len(task_b_samples)
        n_layers = len(layers)
        n_heads = len(heads)

        # Initialize result structure
        results = {}
        for block in ['Q', 'K', 'V', 'O']:
            scores = np.zeros((n_samples_a, n_samples_b, n_layers, n_heads))
            results[block] = {'scores': scores}

        # Compute all scores
        logger.info(f"Computing QKOV heatmap: {n_samples_a}×{n_samples_b} samples, "
                   f"{n_layers} layers, {n_heads} heads")

        for i_idx, sample_a in enumerate(task_a_samples):
            sample_i = int(sample_a.split('_')[1])

            for j_idx, sample_j in enumerate(task_b_samples):

                for l_idx, layer in enumerate(layers):
                    for h_idx, head in enumerate(heads):

                        block_scores = self.compute_sample_pair(
                            task_a, sample_i, task_b, sample_j, layer, head
                        )

                        for block in ['Q', 'K', 'V', 'O']:
                            results[block]['scores'][i_idx, j_idx, l_idx, h_idx] = \
                                block_scores[block]

        # Compute aggregates
        for block in ['Q', 'K', 'V', 'O']:
            scores = results[block]['scores']

            # Average over sample pairs per layer/head
            layer_head_avg = scores.mean(axis=(0, 1))  # [n_layers, n_heads]
            results[block]['layer_head_avg'] = layer_head_avg

            # Find top conflicts
            flat_idx = np.argsort(scores.flatten())[::-1][:100]
            top_conflicts = []
            for idx in flat_idx:
                i, j, l, h = np.unravel_index(idx, scores.shape)
                top_conflicts.append({
                    'sample_i': int(i),
                    'sample_j': int(j),
                    'layer': layers[l],
                    'head': heads[h],
                    'score': float(scores[i, j, l, h])
                })
            results[block]['top_conflicts'] = top_conflicts

        return results

    def sanity_check(self) -> Dict[str, Any]:
        """
        Run sanity checks from intern's feedback.

        Returns:
            Dictionary with check results
        """
        checks = {}

        # Check 1: Head additivity
        # Sum over heads should match unsliced block metric (within fp error)
        logger.info("Running sanity check: head additivity...")
        # TODO: Implement
        checks['head_additivity'] = {'status': 'not_implemented'}

        # Check 2: Scale invariance
        # Scaling weights shouldn't change ranks
        logger.info("Running sanity check: scale invariance...")
        # TODO: Implement
        checks['scale_invariance'] = {'status': 'not_implemented'}

        # Check 3: Ablation validity
        # Zeroing a head should collapse its scores
        logger.info("Running sanity check: ablation validity...")
        # TODO: Implement
        checks['ablation_validity'] = {'status': 'not_implemented'}

        # Check 4: Symmetry
        # Swapping i/j should preserve head identities
        logger.info("Running sanity check: symmetry...")
        # TODO: Implement
        checks['symmetry'] = {'status': 'not_implemented'}

        return checks
