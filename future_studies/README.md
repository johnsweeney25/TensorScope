# Future Studies: Advanced Experimental Interventions

## Overview
This module contains cutting-edge experimental techniques for mechanistic interpretability and causal analysis of neural networks. These methods allow surgical interventions on model computations to understand causal relationships between components.

⚠️ **WARNING**: These techniques modify model forward passes. Always work with model copies and validate results carefully.

## Understanding Attention Mechanisms

### Standard Multi-Head Attention (MHA)
In traditional transformers, attention works through three projections:
- **Query (Q)**: What information am I looking for?
- **Key (K)**: What information do I contain?
- **Value (V)**: What information should I output?

Each attention head computes: `Attention(Q,K,V) = softmax(QK^T/√d)V`

### Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)
Modern efficient architectures like Llama 2 70B and Mistral use GQA/MQA:
- **GQA**: Multiple query heads share the same key-value heads (e.g., 32 Q heads, 8 KV heads)
- **MQA**: All query heads share a single key-value head (extreme case of GQA)

This reduces memory usage and speeds up inference while maintaining performance.

### QK and OV Circuits
Attention can be decomposed into two independent circuits:
1. **QK Circuit**: Controls *where* to look (attention patterns)
   - Computes attention scores via Q·K^T
   - Determines information routing
2. **OV Circuit**: Controls *what* to output (value transformation)
   - Applies attention weights to values
   - Transforms and mixes information

These circuits compose linearly: changing one doesn't affect the other mathematically.

## Modules

### 1. Attention Circuit Freezing (`attention_circuit_freezing.py`)
**Updated with GQA/MQA support!** Selective freezing of QK (attention pattern) and OV (value mixing) circuits within attention heads.

#### Key Features:
- **Separate QK and OV interventions**: Understand which circuit drives behavior
- **Architecture agnostic**: Works with both fused (GPT-2) and separate (LLaMA) QKV
- **GQA/MQA compatible**: Properly handles grouped and multi-query attention architectures
- **Theoretically grounded**: Interventions are mathematically equivalent across architectures
- **Gradient control**: Two modes - `stopgrad` (blocks gradients) or `ste` (straight-through estimator preserves gradient flow)

#### Quick Start:
```python
from future_studies import freeze_qk_circuit, freeze_ov_circuit
import copy

# Always work with a copy
model_copy = copy.deepcopy(model)

# Freeze QK circuit in heads 3,7 of layers 0,1
hooks = freeze_qk_circuit(model_copy, layer_indices=[0, 1], head_indices=[3, 7])

# Run your analysis
output = model_copy(input_ids)

# Always remove hooks when done
for hook in hooks: hook.remove()
```

### 2. Experimental Interventions (`experimental_interventions.py`)
Original intervention methods including full head freezing and intervention vector discovery.

#### Features:
- Head freezing (entire heads only)
- Intervention vector discovery
- Model compatibility checking

## Supported Architectures

### ✅ Fully Supported (QK/OV Separation)
| Model Family | Architecture | QKV Type | Attention Type | Notes |
|--------------|-------------|----------|----------------|-------|
| Llama-2 7B | Separate QKV | 3 matrices | MHA (32 heads) | Full support |
| Llama-2 70B | Separate QKV | 3 matrices | **GQA** (64Q, 8KV) | Full GQA support |
| Mistral 7B | Separate QKV | 3 matrices | **GQA** (32Q, 8KV) | Full GQA support |
| Falcon | Separate QKV | 3 matrices | **MQA** (71Q, 1KV) | Full MQA support |
| BLOOM | Separate QKV | 3 matrices | MHA | Tested |
| OPT | Separate QKV | 3 matrices | MHA | Tested |
| GPT-2 | Fused QKV | 1 matrix | MHA | Tensor splitting |
| GPT-J | Fused QKV | 1 matrix | MHA | Works with splitting |
| Qwen | Fused QKV | 1 matrix | MHA | Requires careful handling |

### ❌ Not Supported
| Model Type | Reason | Workaround |
|------------|---------|------------|
| Flash Attention 2 | Fused CUDA kernel | Use non-flash version or freeze entire heads |
| xFormers | Optimized kernel | Disable optimization |
| Quantized models | No proper hooks | Use full precision |

## Installation

```bash
# The module is already part of the main framework
# Just import and use:
from future_studies import AttentionCircuitFreezer, CircuitType, FreezeType
```

## Usage Examples

### Example 1: Compare QK vs OV Contribution
```python
from future_studies import AttentionCircuitFreezer, CircuitType, FreezeType, InterventionConfig
import torch
import copy

# Setup
freezer = AttentionCircuitFreezer()
base_model = load_model("gpt2")
input_ids = tokenizer("The capital of France is", return_tensors="pt").input_ids

# Baseline
with torch.no_grad():
    baseline_output = base_model(input_ids).logits

# Test QK circuit
qk_model = copy.deepcopy(base_model)
config = InterventionConfig(
    layer_indices=[5, 6],  # Middle layers
    head_indices=[0, 1, 2, 3],  # First 4 heads
    circuit=CircuitType.QK,
    freeze_type=FreezeType.ZERO
)
qk_hooks = freezer.freeze_circuits(qk_model, config)

with torch.no_grad():
    qk_output = qk_model(input_ids).logits

# Test OV circuit
ov_model = copy.deepcopy(base_model)
config.circuit = CircuitType.OV
ov_hooks = freezer.freeze_circuits(ov_model, config)

with torch.no_grad():
    ov_output = ov_model(input_ids).logits

# Compare effects
qk_effect = (baseline_output - qk_output).abs().mean()
ov_effect = (baseline_output - ov_output).abs().mean()

print(f"QK circuit importance: {qk_effect:.3f}")
print(f"OV circuit importance: {ov_effect:.3f}")

# Cleanup
freezer.remove_hooks(qk_hooks)
freezer.remove_hooks(ov_hooks)
```

### Example 2: Working with GQA/MQA Models
```python
from future_studies import AttentionCircuitFreezer, InterventionConfig

# Load a GQA model (e.g., Llama 2 70B)
model = load_model("meta-llama/Llama-2-70b-hf")
freezer = AttentionCircuitFreezer()

# The module automatically detects GQA/MQA
config = freezer.get_model_config(model)
print(f"Query heads: {config['num_heads']}")      # 64
print(f"KV heads: {config['num_kv_heads']}")      # 8

# When freezing head 0-7, it affects:
# - Query heads 0-7 individually
# - KV head 0 (shared by query heads 0-7)
config = InterventionConfig(
    layer_indices=[20],
    head_indices=list(range(8)),  # First 8 query heads
    circuit=CircuitType.QK
)

# The freezing properly handles the shared KV heads
hooks = freezer.freeze_circuits(model, config)
```

### Example 3: Automated Architecture Detection
```python
from future_studies import AttentionCircuitFreezer

freezer = AttentionCircuitFreezer()

# Automatically detect architecture
architecture = freezer.detect_architecture(model)
print(f"Detected architecture: {architecture.value}")

if architecture == ModelArchitecture.FLASH_ATTENTION:
    print("Warning: Flash Attention detected. QK/OV separation not possible.")
    print("Falling back to full head freezing...")
```

### Example 3: Debug and Verify Gradient Behavior
```python
from future_studies import AttentionCircuitFreezer, InterventionConfig, CircuitType
import copy

# Create freezer with debug mode enabled
freezer = AttentionCircuitFreezer(debug_gradients=True)

# Test stopgrad mode (blocks gradients)
model_sg = copy.deepcopy(model)
config_sg = InterventionConfig(
    layer_indices=[0, 1],
    head_indices=[0, 1, 2],
    circuit=CircuitType.QK,
    backward_mode="stopgrad"  # Blocks gradients
)
hooks_sg = freezer.freeze_circuits(model_sg, config_sg)

# Check gradient behavior
test_input = torch.randn(1, 10, 768, requires_grad=True)
grad_stats = freezer.check_gradient_behavior(model_sg, test_input, config_sg)

print(f"Validation: {'PASSED' if grad_stats['validation_passed'] else 'FAILED'}")
print(f"Loss: {grad_stats['loss_value']:.6f}")

# Example gradient statistics output:
# self_attn.q_proj.weight      | norm: 0.000001 | mean: 0.000000 | max: 0.000002 | nonzero:  0.1%
# self_attn.k_proj.weight      | norm: 0.000001 | mean: 0.000000 | max: 0.000001 | nonzero:  0.1%
# → Near-zero gradients confirm stopgrad is working

# Test STE mode (preserves gradient flow)
model_ste = copy.deepcopy(model)
config_ste = InterventionConfig(
    layer_indices=[0, 1],
    head_indices=[0, 1, 2],
    circuit=CircuitType.QK,
    backward_mode="ste"  # Preserves gradients via straight-through
)
hooks_ste = freezer.freeze_circuits(model_ste, config_ste)

grad_stats_ste = freezer.check_gradient_behavior(model_ste, test_input, config_ste)
# → Non-zero gradients confirm STE is preserving flow

# Verify intervention actually changes outputs
effect_stats = freezer.verify_intervention_effect(
    model_baseline=model,
    model_intervened=model_sg,
    test_input=test_input,
    config=config_sg
)
print(f"Max output change: {effect_stats['max_absolute_diff']:.6f}")
```

### Example 4: Publication-Ready Experiment
```python
def run_circuit_ablation_study(model, dataset, layers, heads):
    """
    Run systematic ablation study for paper.
    Returns results ready for LaTeX tables.
    """
    results = []
    freezer = AttentionCircuitFreezer()

    for circuit_type in [CircuitType.QK, CircuitType.OV, CircuitType.BOTH]:
        model_copy = copy.deepcopy(model)

        config = InterventionConfig(
            layer_indices=layers,
            head_indices=heads,
            circuit=circuit_type,
            freeze_type=FreezeType.ZERO,
            preserve_gradients=True  # Important for gradient analysis
        )

        hooks = freezer.freeze_circuits(model_copy, config)

        # Evaluate on dataset
        metrics = evaluate_model(model_copy, dataset)

        results.append({
            'circuit': circuit_type.value,
            'perplexity': metrics['perplexity'],
            'accuracy': metrics['accuracy'],
            'effect_size': compute_effect_size(baseline_metrics, metrics)
        })

        freezer.remove_hooks(hooks)

    return pd.DataFrame(results)
```

## Gradient Behavior and Backward Modes

### Understanding Gradient Control

The module offers two backward modes for controlling gradient flow during interventions:

1. **`stopgrad` (default)**: Complete gradient blocking
   - Forward pass: Uses replacement value (zero, mean, or noise)
   - Backward pass: Zero gradient with respect to original activation
   - Use case: Causal analysis where you want to completely isolate circuit effects

2. **`ste` (Straight-Through Estimator)**: Gradient preservation
   - Forward pass: Uses replacement value (zero, mean, or noise)
   - Backward pass: Gradient flows as if original activation was unchanged
   - Use case: Training analysis where you want to maintain gradient flow for learning

```python
# Example: Causal analysis with gradient blocking
config = InterventionConfig(
    layer_indices=[5],
    head_indices=[0, 1],
    circuit=CircuitType.QK,
    backward_mode="stopgrad"  # Blocks gradients completely
)

# Example: Training analysis with gradient preservation
config = InterventionConfig(
    layer_indices=[5],
    head_indices=[0, 1],
    circuit=CircuitType.QK,
    backward_mode="ste"  # Preserves gradient flow via straight-through
)
```

### Important Notes on KV Caching

⚠️ **Warning**: During autoregressive generation, models use KV caches (`past_key_values`). The current implementation only affects the current token's projections, not the cached history. For generation experiments:

- Consider clearing caches when applying interventions
- Be aware that frozen circuits only affect new tokens, not cached attention from previous tokens
- Future versions will include explicit cache handling

## Theoretical Foundation

### Mathematical Equivalence
For fused QKV (GPT-2) and separate QKV (LLaMA) architectures:

```
Fused:    [Q,K,V] = X @ W_qkv, then split
Separate: Q = X @ W_q, K = X @ W_k, V = X @ W_v

Our intervention operates on Q, K, V after projection, making it
mathematically equivalent regardless of the projection method.
```

### GQA/MQA Head Mapping
In Grouped-Query Attention, multiple query heads share KV heads:

```python
# Example: 32 query heads, 8 KV heads (4:1 ratio)
query_head_0-3   → kv_head_0
query_head_4-7   → kv_head_1
query_head_8-11  → kv_head_2
...

# Intervention mapping
When freezing query_head_5:
- Freezes Q projection for head 5
- Freezes K,V projections for kv_head_1 (affects heads 4-7)
```

This sharing pattern is automatically detected and handled by our implementation.

### Circuit Definitions
- **QK Circuit**: Controls attention pattern formation via Query-Key dot products
- **OV Circuit**: Controls information flow via Output-Value multiplication
- **Independence**: QK and OV circuits operate independently and compose linearly
- **GQA Consideration**: In GQA, freezing affects groups of heads sharing KV projections

## Validation & Testing

### Running Tests
```bash
cd future_studies
python -m pytest tests/test_circuit_freezing.py -v
```

### Key Validation Checks
1. **Architecture detection** works correctly
2. **Gradient flow** preserved when configured
3. **Circuit independence**: QK doesn't affect OV and vice versa
4. **Equivalence**: Same results on fused vs separate QKV models

## Common Issues & Solutions

### Issue 1: "Could not detect model architecture"
**Solution**: Manually specify architecture:
```python
from future_studies import ModelArchitecture
hooks = freezer.freeze_circuits(model, config, architecture=ModelArchitecture.SEPARATE_QKV)
```

### Issue 2: "Flash Attention not supported"
**Solution**: Disable flash attention or use full head freezing:
```python
# Disable flash attention (model-specific)
model.config.use_flash_attention = False

# Or use original head freezing
from future_studies import ExperimentalInterventions
exp = ExperimentalInterventions()
hooks = exp.freeze_attention_heads(model, {0: [1, 2]})
```

### Issue 3: Gradient errors during training
**Solution**: Enable gradient preservation:
```python
config = InterventionConfig(
    # ...
    preserve_gradients=True  # Use multiplication by 0 instead of assignment
)
```

### Issue 4: Unexpected behavior with GQA/MQA models
**Note**: When freezing heads in GQA/MQA models:
- Freezing a query head may affect neighboring heads due to shared KV projections
- Example: In Llama-2 70B, freezing head 0 affects KV head 0, which is shared by query heads 0-7
- This is expected behavior that reflects the model's architecture

**Solution**: To freeze individual query heads without affecting others:
```python
# For precise control, freeze only the Q circuit
config = InterventionConfig(
    head_indices=[5],  # Only affects Q of head 5
    circuit=CircuitType.QK,  # Still affects shared K
    # Consider using custom head mapping for fine-grained control
)
```

## Citation

If you use these methods in your research, please cite:

```bibtex
@article{circuitfreezing2024,
    title={Dissecting Attention: Causal Analysis of QK and OV Circuits in Transformers},
    author={Your Name},
    journal={arXiv preprint arXiv:2024.xxxxx},
    year={2024}
}
```

## Safety & Ethics

⚠️ **Important Considerations**:
1. These interventions can reveal model vulnerabilities
2. Always validate that interventions work as intended
3. Be transparent about limitations (e.g., Flash Attention)
4. Document any model-specific adaptations

## Future Work

- [ ] Support for cross-attention circuits
- [ ] Intervention strength gradients (partial freezing)
- [ ] Automatic importance scoring
- [ ] Integration with activation patching
- [ ] Support for MoE models

## Contributing

This is experimental research code. Contributions welcome:
1. Test on new model architectures
2. Add validation metrics
3. Improve efficiency
4. Add visualization tools

## License

Part of the Unified Model Analysis Framework
MIT License - See main repository

---
*Last updated: September 2024*
*Version: 0.1.0*