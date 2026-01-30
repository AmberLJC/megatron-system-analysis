# Fused RoPE (Rotary Position Embedding)

## Context

Rotary embeddings require rotation matrix computation and application, typically 2-3 kernels.

**Solution:** Single fused kernel computes and applies rotary embeddings.

## Implementation

```python
# Conceptual implementation
def fused_rope(query, key, position_ids, rotary_dim):
    """
    Fused RoPE computation and application.
    Single kernel replaces 3 separate operations:
    1. Compute sin/cos
    2. Rotate query
    3. Rotate key
    """
    return fused_rope_cuda(query, key, position_ids, rotary_dim)
```

## Code Location

- **Implementation:** `megatron/core/fusions/fused_mla_yarn_rope_apply.py`

## Performance Impact

- **1.5-2x faster** vs unfused RoPE
- Small but cumulative (applied every attention layer)
- For 96 layers: ~5ms saved per step

## When to Use

**Automatically used** for models using RoPE:
- GPT-NeoX
- LLaMA
- Mistral
- etc.

## Related Optimizations

- [Fused Softmax](22_compute_fused_softmax.md) - Fusion in attention
- [Kernel Fusion](21_compute_bias_activation_fusion.md) - General fusion principle

## References

- Paper: [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864)
- [LLaMA RoPE Implementation](https://github.com/facebookresearch/llama)

