# Kernel Fusion - Bias + Activation

## Context

Separate bias addition and activation require:
- 3 memory accesses (read input, write temp, read temp)
- 2 kernel launches (~100μs overhead)

**Solution:** Single fused kernel, everything in registers.

## Implementation

Available fusions:
- Bias + GELU
- Bias + SwiGLU
- Bias + GEGLU

```python
# From fused_bias_gelu.py (conceptual)
@torch.jit.script
def fused_bias_gelu(input, bias):
    """Fused bias add + GELU activation"""
    # Single kernel does both operations
    # No intermediate materialization!
    x = input + bias
    return 0.5 * x * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))
```

## Code Location

- **Bias + GELU:** `megatron/core/fusions/fused_bias_gelu.py`
- **Bias + SwiGLU:** `megatron/core/fusions/fused_bias_swiglu.py`
- **Bias + GEGLU:** `megatron/core/fusions/fused_bias_geglu.py`

## Performance Impact

| Metric | Unfused | Fused | Improvement |
|--------|---------|-------|-------------|
| Kernels | 2 | 1 | 2x |
| Memory access | 3× data | 1× data | 3x bandwidth |
| Time per layer | 100μs | 50μs | 2x |
| 96 layers | 9.6ms | 4.8ms | **4.8ms saved** |

## When to Use

**Automatically used** when Apex/TransformerEngine available. No configuration needed!

## Related Optimizations

- [CUDA Graphs](20_compute_cuda_graphs.md) - Fewer kernels → better graphs
- [Fused Softmax](22_compute_fused_softmax.md) - Similar fusion principle

## References

- [NVIDIA Apex](https://github.com/NVIDIA/apex)
- [Kernel Fusion](https://en.wikipedia.org/wiki/Loop_fusion)

