# Gradient Accumulation Fusion

## Context

Standard gradient accumulation: compute grad into temp tensor, then add to main_grad (extra kernel, extra memory).

**Solution:** Weight gradients accumulate directly into main_grad (FP32) using custom CUDA kernel.

## Implementation

```python
# Instead of:
# 1. grad_weight = compute_grad(...)      # Temporary FP16
# 2. main_grad += grad_weight.to(fp32)    # Separate add kernel

# Do this:
# 1. fused_wgrad_gemm_accum_fp32(         # Single fused kernel!
#        output=main_grad,                 # Accumulate directly to FP32
#        ...)
```

## Code Location

- **Config:** `megatron/core/tensor_parallel/layers.py` lines 44-48
- **Uses:** `fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32()` from TE/Apex

## Performance Impact

### Kernels Saved

For 80-layer model with 8 microbatches:
- Layers: 80
- Operations per layer: 2 (QKV + MLP)
- Microbatches: 8
- **Total kernels saved:** 80 × 2 × 8 = 1280 kernels
- **Time saved:** 1280 × 5μs = **6.4ms per step**

### Cumulative Benefit

- **2-5% speedup** end-to-end
- Better with more microbatches
- Combines well with other fusions

## When to Use

**Always use** when available:
- Requires Transformer Engine or Apex
- `gradient_accumulation_fusion=True`

**Configuration:**

```python
config = TransformerConfig(
    gradient_accumulation_fusion=True,  # Enable!
)
```

## Related Optimizations

- [Tensor Parallelism Overlap](04_communication_tp_overlap.md) - Complements fusion
- [Bias + Activation Fusion](21_compute_bias_activation_fusion.md) - Similar fusion principle

## References

- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [Gradient Accumulation](https://arxiv.org/abs/1711.00489)

