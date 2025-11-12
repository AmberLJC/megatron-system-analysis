# Fused Softmax with Masking

## Context

Standard softmax with masking requires separate operations: mask application, softmax computation, and scaling. Each has overhead.

## Implementation

Three fused variants:
- **ScaledUpperTriangMaskedSoftmax:** Causal attention (most common)
- **ScaledMaskedSoftmax:** General masking
- **ScaledSoftmax:** No mask

```python
class ScaledUpperTriangMaskedSoftmax(torch.autograd.Function):
    """Fused causal mask + scaled softmax"""
    
    @staticmethod
    def forward(ctx, inputs, scale):
        # Single kernel does: mask + scale + softmax
        # Much faster than 3 separate operations
        outputs = fused_upper_triang_masked_softmax_cuda(inputs, scale)
        ctx.save_for_backward(outputs)
        ctx.scale = scale
        return outputs
```

## Code Location

- **Causal attention:** `megatron/core/fusions/fused_softmax.py` lines 11-57
- **General masking:** Lines 60-105
- **No mask:** Lines 108-151

## Performance Impact

- **2-3x faster** vs unfused softmax
- **Critical for attention** (often bottleneck in transformer)

## When to Use

**Automatically used** in attention layers when available.

## Related Optimizations

- [Fused Layer Norm](23_compute_fused_layernorm.md) - Similar fusion for normalization
- [Bias + Activation Fusion](21_compute_bias_activation_fusion.md) - Fusion principle

## References

- [Flash Attention](https://arxiv.org/abs/2205.14135) - Advanced attention optimization
- [Softmax Optimization](https://arxiv.org/abs/1805.02867)

