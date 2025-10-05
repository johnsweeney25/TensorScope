# QK/OV Circuit Freezing Design Document

## Executive Summary
Design for implementing selective freezing of QK (attention pattern) and OV (value mixing) circuits in transformer models. This allows for fine-grained causal analysis of attention head sub-components.

## Background: Model Architecture Variations

### 1. Fused QKV Models (Single Matrix)
**Models**: GPT-2, GPT-J, GPT-NeoX, CodeGen, Qwen
```python
# GPT-2 style: Single matrix projects to Q, K, V concatenated
self.c_attn = nn.Linear(hidden_size, 3 * hidden_size)  # Fused QKV
qkv = self.c_attn(x)
q, k, v = qkv.split(hidden_size, dim=-1)
```
**Challenge**: Cannot freeze Q/K independently at projection level since they come from same matrix

### 2. Separate QKV Models (Three Matrices)
**Models**: LLaMA, Mistral, Falcon, BLOOM, OPT
```python
# LLaMA style: Separate projections
self.q_proj = nn.Linear(hidden_size, hidden_size)
self.k_proj = nn.Linear(hidden_size, hidden_size)
self.v_proj = nn.Linear(hidden_size, hidden_size)
q = self.q_proj(x)
k = self.k_proj(x)
v = self.v_proj(x)
```
**Advantage**: Can easily freeze Q/K/V independently at projection level

### 3. Flash Attention Models
**Models**: Any model using Flash Attention 2
```python
# Flash attention: Fused kernel, no intermediate access
output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
# No access to attention weights!
```
**Challenge**: IMPOSSIBLE to freeze QK/OV separately - computation is fused in CUDA kernel

## Proposed Design

### Core Architecture
```python
class AttentionCircuitFreezer:
    """Selective freezing of QK and OV circuits in attention heads."""

    def __init__(self):
        self.model_type = None  # 'fused_qkv', 'separate_qkv', 'flash'
        self.active_hooks = []

    def detect_attention_type(self, model) -> str:
        """Auto-detect model's attention implementation."""
        # Check for flash attention
        # Check for fused vs separate QKV
        # Return type string

    def freeze_circuits(
        self,
        model,
        layer_indices: List[int],
        head_indices: List[int],
        circuit: str = 'both',  # 'qk', 'ov', 'both'
        freeze_type: str = 'zero',  # 'zero', 'noise', 'mean'
        model_type: str = 'auto'
    ):
        """Main entry point for circuit freezing."""
```

### Implementation Strategies by Model Type

#### Strategy 1: Separate QKV Models (EASIEST)
```python
def freeze_separate_qkv(self, model, layer_idx, heads, circuit, freeze_type):
    """For models with separate q_proj, k_proj, v_proj."""

    if circuit == 'qk':
        # Option A: Zero Q and K projections for specific heads
        def hook_q(module, input, output):
            # output shape: [batch, seq, hidden]
            # Zero out dimensions corresponding to target heads
            head_dim = output.shape[-1] // num_heads
            for head_idx in heads:
                start = head_idx * head_dim
                end = (head_idx + 1) * head_dim
                output[..., start:end] = 0
            return output

        # Register hooks on q_proj and k_proj
        hooks.append(layer.self_attn.q_proj.register_forward_hook(hook_q))
        hooks.append(layer.self_attn.k_proj.register_forward_hook(hook_k))

    elif circuit == 'ov':
        # Zero V projections for specific heads
        hooks.append(layer.self_attn.v_proj.register_forward_hook(hook_v))
```

#### Strategy 2: Fused QKV Models (HARDER)
```python
def freeze_fused_qkv(self, model, layer_idx, heads, circuit, freeze_type):
    """For models with fused c_attn producing concatenated QKV."""

    def hook_qkv(module, input, output):
        # output shape: [batch, seq, 3 * hidden]
        hidden_size = output.shape[-1] // 3
        q, k, v = output.split(hidden_size, dim=-1)

        # Reshape to per-head
        batch, seq = q.shape[:2]
        num_heads = self.get_num_heads(layer_idx)
        head_dim = hidden_size // num_heads

        q = q.view(batch, seq, num_heads, head_dim)
        k = k.view(batch, seq, num_heads, head_dim)
        v = v.view(batch, seq, num_heads, head_dim)

        if circuit == 'qk':
            # Zero Q and K for target heads
            for head_idx in heads:
                q[:, :, head_idx, :] = 0
                k[:, :, head_idx, :] = 0
        elif circuit == 'ov':
            # Zero V for target heads
            for head_idx in heads:
                v[:, :, head_idx, :] = 0

        # Reshape back and concatenate
        q = q.view(batch, seq, hidden_size)
        k = k.view(batch, seq, hidden_size)
        v = v.view(batch, seq, hidden_size)

        return torch.cat([q, k, v], dim=-1)

    # Single hook on c_attn
    hooks.append(layer.attn.c_attn.register_forward_hook(hook_qkv))
```

#### Strategy 3: Attention Scores Level (MOST PRECISE)
```python
def freeze_at_attention_scores(self, model, layer_idx, heads, circuit):
    """Hook at attention computation level - works for both model types."""

    def hook_attention_forward(module, input, kwargs, output):
        # Need to intercept the attention forward pass
        # This is model-specific and complex

        if circuit == 'qk':
            # Replace attention scores with uniform attention
            # attn_weights = torch.ones_like(attn_weights) / seq_len
            pass
        elif circuit == 'ov':
            # Keep attention pattern but zero values
            # v[:, :, head_idx, :] = 0
            pass

    # Hook the entire attention module's forward
    layer.self_attn.register_forward_hook(hook_attention_forward)
```

### Flash Attention Handling
```python
def handle_flash_attention(self, model):
    """Flash attention models cannot be selectively frozen."""

    # Option 1: Raise error
    raise NotImplementedError(
        "Flash Attention does not expose intermediate values. "
        "Cannot freeze QK/OV circuits separately. "
        "Consider using a non-flash model or freezing entire heads."
    )

    # Option 2: Fall back to entire head freezing
    warnings.warn(
        "Flash Attention detected. Falling back to full head freezing. "
        "QK/OV separation not possible with fused kernels."
    )
    return self.freeze_entire_heads(...)
```

## Critical Pitfalls and Challenges

### 1. Model Detection Issues
```python
# PITFALL: Model architecture variations
# GPT2Model vs GPT2LMHeadModel have different structures
# Some models wrap layers differently (model.transformer vs model.model)

# SOLUTION: Robust detection
def find_attention_layers(model):
    """Recursively find attention layers regardless of wrapper."""
    attention_layers = []
    for module in model.modules():
        if hasattr(module, 'q_proj') or hasattr(module, 'c_attn'):
            attention_layers.append(module)
    return attention_layers
```

### 2. Head Dimension Calculations
```python
# PITFALL: Different head counts and dimensions
# - GPT-2: 12 heads for 768 dim (64 dim/head)
# - LLaMA-70B: 64 heads for 8192 dim (128 dim/head)
# - Some models use multi-query attention (shared K,V)

# SOLUTION: Get from config
def get_head_config(model, layer_idx):
    config = model.config
    return {
        'num_heads': config.num_attention_heads,
        'head_dim': config.hidden_size // config.num_attention_heads,
        'num_kv_heads': getattr(config, 'num_key_value_heads', config.num_attention_heads)
    }
```

### 3. Gradient Flow Issues
```python
# PITFALL: Zeroing tensors can break gradients
# PyTorch might optimize away zero branches

# SOLUTION: Use multiplication or addition
def safe_freeze(tensor, freeze_type='zero'):
    if freeze_type == 'zero':
        return tensor * 0  # Maintains gradient flow
    elif freeze_type == 'noise':
        noise = torch.randn_like(tensor) * 0.01
        return noise  # Replace with small noise
    elif freeze_type == 'mean':
        return torch.ones_like(tensor) * tensor.mean()
```

### 4. Memory Layout Assumptions
```python
# PITFALL: Contiguous memory assumptions
# Reshaping might fail if tensor not contiguous

# SOLUTION: Ensure contiguous
def safe_reshape(tensor, *shape):
    return tensor.contiguous().view(*shape)
```

## Feasibility Assessment

### ✅ FEASIBLE for These Models:
- **LLaMA/Mistral/Falcon**: Separate QKV projections
- **GPT-2/GPT-J** (with caveats): Fused QKV but can split
- **BERT/RoBERTa**: Separate QKV projections
- **OPT/BLOOM**: Separate projections

### ⚠️ DIFFICULT but POSSIBLE:
- **Qwen**: Fused QKV, need careful tensor manipulation
- **GPT-NeoX**: Fused QKV with parallel attention
- **CodeGen**: Fused with special rotary embeddings

### ❌ IMPOSSIBLE:
- **Flash Attention 2 models**: No intermediate access
- **Optimized kernels** (xFormers, etc.): Fused computation
- **Quantized models**: May not expose proper hooks

## Implementation Priority

### Phase 1: Separate QKV Models (Week 1)
- Implement for LLaMA-style models
- Test on small models (125M-1B params)
- Validate with known attention head functions

### Phase 2: Fused QKV Models (Week 2)
- Add support for GPT-2 style models
- Handle tensor splitting and reshaping
- Test gradient flow preservation

### Phase 3: Robustness (Week 3)
- Add model auto-detection
- Handle edge cases
- Create comprehensive test suite

### Phase 4: Documentation (Week 4)
- Usage examples
- Known limitations
- Troubleshooting guide

## Testing Strategy

### Unit Tests
```python
def test_qk_freezing():
    """Test that QK freezing affects attention pattern."""
    model = load_small_model()

    # Get baseline attention pattern
    baseline_attn = get_attention_weights(model, input)

    # Freeze QK circuit
    freeze_circuits(model, circuit='qk', heads=[0])
    frozen_attn = get_attention_weights(model, input)

    # Should be uniform or very different
    assert not torch.allclose(baseline_attn, frozen_attn)
```

### Integration Tests
```python
def test_ov_freezing():
    """Test that OV freezing affects output but not attention pattern."""
    # Attention pattern should be same
    # Output should be different
```

### Validation Tests
```python
def test_circuit_independence():
    """Verify QK and OV circuits are independent."""
    # Freeze QK: attention pattern changes, values processed normally
    # Freeze OV: attention pattern same, values zeroed
    # Freeze both: equivalent to full head freezing
```

## Risk Mitigation

1. **Start with known-working models** (LLaMA with separate QKV)
2. **Extensive logging** of tensor shapes and operations
3. **Gradient checking** to ensure backprop works
4. **Incremental implementation** - one model type at a time
5. **Fallback options** when circuit separation impossible

## Conclusion

Implementing QK/OV circuit freezing is:
- **FEASIBLE** for most standard transformer models
- **IMPOSSIBLE** for Flash Attention models
- **COMPLEX** but manageable for fused QKV models

The key is to:
1. Start with easy cases (separate QKV)
2. Build robust model detection
3. Accept that some models (Flash Attention) cannot support this
4. Provide clear documentation of limitations

---
*Document created: September 29, 2024*
*Estimated implementation time: 2-4 weeks for production-ready code*