# Fused Layer Normalization

## Context

PyTorch's layer norm uses multiple kernels (mean, variance, normalize). Each has overhead and separate memory traffic.

**Solution:** Single APEX fused kernel computes all in one pass.

## Implementation

```python
# From fused_layer_norm.py
from apex.normalization import FusedLayerNorm

# Replaces torch.nn.LayerNorm
norm = FusedLayerNorm(hidden_size, eps=1e-5)
```

## Code Location

- **Wrapper:** `megatron/core/fusions/fused_layer_norm.py` line 30
- **Implementation:** From APEX library

## Performance Impact

- **2-4x faster** vs PyTorch LayerNorm
- For 96 layers with 2 norms each: **9.6ms saved per step**

## When to Use

**Automatically used** when Apex available.

## Related Optimizations

- [Fused Softmax](22_compute_fused_softmax.md) - Fusion for attention
- [Bias + Activation Fusion](21_compute_bias_activation_fusion.md) - Similar principle

## References

- [NVIDIA Apex Normalization](https://github.com/NVIDIA/apex/tree/master/apex/normalization)
- [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)

