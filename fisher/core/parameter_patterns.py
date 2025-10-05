"""
Unified Parameter Pattern Recognition System
============================================

Handles parameter naming conventions across different model architectures
(Qwen, LLaMA, GPT, BERT, etc.) to ensure consistent Fisher computation.

This is the single source of truth for parameter pattern matching.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum


class ModelArchitecture(Enum):
    """Known model architectures and their naming patterns."""
    QWEN = "qwen"
    LLAMA = "llama"
    GPT = "gpt"
    GPTNEO = "gpt_neo"
    GPT2 = "gpt2"
    BERT = "bert"
    ROBERTA = "roberta"
    T5 = "t5"
    MISTRAL = "mistral"
    UNKNOWN = "unknown"


@dataclass
class ParameterPattern:
    """Defines a parameter pattern for matching."""
    pattern: str  # Regex pattern
    category: str  # Category name (e.g., 'attention_q', 'mlp_gate')
    component: str  # Component type (e.g., 'attention', 'mlp', 'norm')
    architectures: Set[ModelArchitecture]  # Which architectures use this


class UnifiedParameterMatcher:
    """
    Unified system for matching parameter patterns across model architectures.

    This class provides consistent parameter categorization regardless of
    the underlying model architecture (Qwen, GPT, LLaMA, etc.).
    """

    def __init__(self):
        """Initialize with comprehensive parameter patterns."""
        self.patterns = self._initialize_patterns()
        self._compiled_patterns = self._compile_patterns()

    def _initialize_patterns(self) -> List[ParameterPattern]:
        """Define all known parameter patterns across architectures."""
        patterns = []

        # ============================================================
        # ATTENTION PATTERNS
        # ============================================================

        # Qwen/LLaMA style: self_attn.{q,k,v,o}_proj
        patterns.extend([
            ParameterPattern(
                r'self_attn\.q_proj',
                'attention_q',
                'attention',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL}
            ),
            ParameterPattern(
                r'self_attn\.k_proj',
                'attention_k',
                'attention',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL}
            ),
            ParameterPattern(
                r'self_attn\.v_proj',
                'attention_v',
                'attention',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL}
            ),
            ParameterPattern(
                r'self_attn\.o_proj',
                'attention_o',
                'attention',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL}
            ),
        ])

        # Generic patterns without self_attn prefix (older models)
        patterns.extend([
            ParameterPattern(
                r'(?<!self_attn\.)q_proj',
                'attention_q',
                'attention',
                {ModelArchitecture.UNKNOWN}
            ),
            ParameterPattern(
                r'(?<!self_attn\.)k_proj',
                'attention_k',
                'attention',
                {ModelArchitecture.UNKNOWN}
            ),
            ParameterPattern(
                r'(?<!self_attn\.)v_proj',
                'attention_v',
                'attention',
                {ModelArchitecture.UNKNOWN}
            ),
            ParameterPattern(
                r'(?<!self_attn\.)o_proj',
                'attention_o',
                'attention',
                {ModelArchitecture.UNKNOWN}
            ),
        ])

        # GPT style: attn.c_attn (combined QKV), attn.c_proj (output)
        patterns.extend([
            ParameterPattern(
                r'attn\.c_attn',
                'attention_qkv',
                'attention',
                {ModelArchitecture.GPT, ModelArchitecture.GPT2, ModelArchitecture.GPTNEO}
            ),
            ParameterPattern(
                r'attn\.c_proj',
                'attention_o',
                'attention',
                {ModelArchitecture.GPT, ModelArchitecture.GPT2, ModelArchitecture.GPTNEO}
            ),
        ])

        # BERT style: attention.self.{query,key,value}
        patterns.extend([
            ParameterPattern(
                r'attention\.self\.query',
                'attention_q',
                'attention',
                {ModelArchitecture.BERT, ModelArchitecture.ROBERTA}
            ),
            ParameterPattern(
                r'attention\.self\.key',
                'attention_k',
                'attention',
                {ModelArchitecture.BERT, ModelArchitecture.ROBERTA}
            ),
            ParameterPattern(
                r'attention\.self\.value',
                'attention_v',
                'attention',
                {ModelArchitecture.BERT, ModelArchitecture.ROBERTA}
            ),
            ParameterPattern(
                r'attention\.output\.dense',
                'attention_o',
                'attention',
                {ModelArchitecture.BERT, ModelArchitecture.ROBERTA}
            ),
        ])

        # T5 style
        patterns.extend([
            ParameterPattern(
                r'SelfAttention\.q',
                'attention_q',
                'attention',
                {ModelArchitecture.T5}
            ),
            ParameterPattern(
                r'SelfAttention\.k',
                'attention_k',
                'attention',
                {ModelArchitecture.T5}
            ),
            ParameterPattern(
                r'SelfAttention\.v',
                'attention_v',
                'attention',
                {ModelArchitecture.T5}
            ),
            ParameterPattern(
                r'SelfAttention\.o',
                'attention_o',
                'attention',
                {ModelArchitecture.T5}
            ),
        ])

        # ============================================================
        # MLP PATTERNS
        # ============================================================

        # Qwen/LLaMA style: mlp.{gate,up,down}_proj
        patterns.extend([
            ParameterPattern(
                r'mlp\.gate_proj',
                'mlp_gate',
                'mlp',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL}
            ),
            ParameterPattern(
                r'mlp\.up_proj',
                'mlp_up',
                'mlp',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL}
            ),
            ParameterPattern(
                r'mlp\.down_proj',
                'mlp_down',
                'mlp',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL}
            ),
        ])

        # GPT style: mlp.c_fc, mlp.c_proj
        patterns.extend([
            ParameterPattern(
                r'mlp\.c_fc',
                'mlp_up',
                'mlp',
                {ModelArchitecture.GPT, ModelArchitecture.GPT2, ModelArchitecture.GPTNEO}
            ),
            ParameterPattern(
                r'mlp\.c_proj',
                'mlp_down',
                'mlp',
                {ModelArchitecture.GPT, ModelArchitecture.GPT2, ModelArchitecture.GPTNEO}
            ),
        ])

        # BERT style: intermediate.dense, output.dense
        patterns.extend([
            ParameterPattern(
                r'intermediate\.dense',
                'mlp_up',
                'mlp',
                {ModelArchitecture.BERT, ModelArchitecture.ROBERTA}
            ),
            ParameterPattern(
                r'output\.dense',
                'mlp_down',
                'mlp',
                {ModelArchitecture.BERT, ModelArchitecture.ROBERTA}
            ),
        ])

        # Generic MLP patterns
        patterns.extend([
            ParameterPattern(
                r'fc1|ffn\.fc1',
                'mlp_up',
                'mlp',
                {ModelArchitecture.UNKNOWN}
            ),
            ParameterPattern(
                r'fc2|ffn\.fc2',
                'mlp_down',
                'mlp',
                {ModelArchitecture.UNKNOWN}
            ),
        ])

        # ============================================================
        # NORMALIZATION PATTERNS
        # ============================================================

        # Qwen/LLaMA style
        patterns.extend([
            ParameterPattern(
                r'input_layernorm|ln_1',
                'norm_pre',
                'norm',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL}
            ),
            ParameterPattern(
                r'post_attention_layernorm|ln_2',
                'norm_post',
                'norm',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA, ModelArchitecture.MISTRAL}
            ),
        ])

        # GPT style
        patterns.extend([
            ParameterPattern(
                r'ln_1',
                'norm_pre',
                'norm',
                {ModelArchitecture.GPT, ModelArchitecture.GPT2}
            ),
            ParameterPattern(
                r'ln_2',
                'norm_post',
                'norm',
                {ModelArchitecture.GPT, ModelArchitecture.GPT2}
            ),
            ParameterPattern(
                r'ln_f',
                'norm_final',
                'norm',
                {ModelArchitecture.GPT, ModelArchitecture.GPT2}
            ),
        ])

        # BERT style
        patterns.extend([
            ParameterPattern(
                r'LayerNorm',
                'norm',
                'norm',
                {ModelArchitecture.BERT, ModelArchitecture.ROBERTA}
            ),
        ])

        # Generic norm patterns
        patterns.extend([
            ParameterPattern(
                r'norm|layernorm|layer_norm|rmsnorm',
                'norm',
                'norm',
                {ModelArchitecture.UNKNOWN}
            ),
        ])

        # ============================================================
        # EMBEDDING PATTERNS
        # ============================================================

        patterns.extend([
            ParameterPattern(
                r'embed_tokens|tok_embeddings|word_embeddings|wte',
                'embeddings',
                'embedding',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA,
                 ModelArchitecture.GPT, ModelArchitecture.BERT}
            ),
            ParameterPattern(
                r'embed_positions|pos_embeddings|position_embeddings|wpe',
                'position_embeddings',
                'embedding',
                {ModelArchitecture.GPT, ModelArchitecture.BERT}
            ),
        ])

        # ============================================================
        # OUTPUT PATTERNS
        # ============================================================

        patterns.extend([
            ParameterPattern(
                r'lm_head|output_projection|cls\.predictions',
                'lm_head',
                'output',
                {ModelArchitecture.QWEN, ModelArchitecture.LLAMA,
                 ModelArchitecture.GPT, ModelArchitecture.BERT}
            ),
        ])

        # ============================================================
        # BIAS PATTERNS
        # ============================================================

        patterns.append(
            ParameterPattern(
                r'\.bias$',
                'bias',
                'bias',
                set(ModelArchitecture)
            )
        )

        return patterns

    def _compile_patterns(self) -> Dict[str, List[Tuple[re.Pattern, ParameterPattern]]]:
        """Compile regex patterns for efficiency."""
        compiled = {'all': []}
        for pattern in self.patterns:
            compiled_regex = re.compile(pattern.pattern, re.IGNORECASE)
            compiled['all'].append((compiled_regex, pattern))

            # Also organize by component for faster lookup
            if pattern.component not in compiled:
                compiled[pattern.component] = []
            compiled[pattern.component].append((compiled_regex, pattern))

        return compiled

    def detect_architecture(self, param_names: List[str]) -> ModelArchitecture:
        """
        Detect model architecture from parameter names.

        Args:
            param_names: List of parameter names from the model

        Returns:
            Detected ModelArchitecture
        """
        architecture_scores = {arch: 0 for arch in ModelArchitecture}

        for name in param_names[:100]:  # Check first 100 params for speed
            for regex, pattern in self._compiled_patterns['all']:
                if regex.search(name):
                    for arch in pattern.architectures:
                        architecture_scores[arch] += 1

        # Find architecture with highest score
        if architecture_scores:
            best_arch = max(architecture_scores, key=architecture_scores.get)
            if architecture_scores[best_arch] > 0:
                return best_arch

        return ModelArchitecture.UNKNOWN

    def categorize_parameter(
        self,
        param_name: str,
        architecture: Optional[ModelArchitecture] = None
    ) -> Tuple[str, str]:
        """
        Categorize a parameter based on its name.

        Args:
            param_name: Parameter name to categorize
            architecture: Optional architecture hint for better matching

        Returns:
            Tuple of (category, component)
        """
        # First pass: honor architecture-specific patterns when provided
        if architecture is not None:
            for regex, pattern in self._compiled_patterns['all']:
                if regex.search(param_name) and architecture in pattern.architectures:
                    return pattern.category, pattern.component

            # Fallback pass: if no arch-specific match, allow generic patterns
            for regex, pattern in self._compiled_patterns['all']:
                if regex.search(param_name):
                    return pattern.category, pattern.component

        else:
            # No architecture hint provided: use any matching pattern
            for regex, pattern in self._compiled_patterns['all']:
                if regex.search(param_name):
                    return pattern.category, pattern.component

        # Default fallback
        return 'other', 'unknown'

    def is_attention_parameter(self, param_name: str) -> bool:
        """Check if parameter is attention-related."""
        category, component = self.categorize_parameter(param_name)
        return component == 'attention'

    def is_mlp_parameter(self, param_name: str) -> bool:
        """Check if parameter is MLP-related."""
        category, component = self.categorize_parameter(param_name)
        return component == 'mlp'

    def get_attention_parameters(self, param_names: List[str]) -> List[str]:
        """Filter list to only attention parameters."""
        return [name for name in param_names if self.is_attention_parameter(name)]

    def get_mlp_parameters(self, param_names: List[str]) -> List[str]:
        """Filter list to only MLP parameters."""
        return [name for name in param_names if self.is_mlp_parameter(name)]

    def normalize_parameter_name(self, param_name: str) -> str:
        """
        Normalize parameter name to a standard format.

        This helps with cross-architecture comparisons.

        Args:
            param_name: Original parameter name

        Returns:
            Normalized name
        """
        category, component = self.categorize_parameter(param_name)

        # Extract layer number if present
        layer_match = re.search(r'\.(layers?|blocks?|h)\.(\d+)\.', param_name)
        layer_num = f"layer_{layer_match.group(2)}" if layer_match else "no_layer"

        # Create normalized name
        return f"{layer_num}.{component}.{category}"


# Global instance for easy access
PARAMETER_MATCHER = UnifiedParameterMatcher()


def get_parameter_matcher() -> UnifiedParameterMatcher:
    """Get the global parameter matcher instance."""
    return PARAMETER_MATCHER
