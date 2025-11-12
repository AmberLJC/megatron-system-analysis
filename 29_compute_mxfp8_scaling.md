# MXFP8 Blockwise Scaling

## Context

Standard FP8 uses per-tensor scaling—one scale factor for entire tensor. This loses precision when tensor has varying magnitude.

**Solution:** MXFP8 uses **per-block scaling**—different scale for each block (e.g., 32 elements).

## Implementation

```python
# Standard FP8 (per-tensor):
scale = tensor.abs().max() / fp8_max
tensor_fp8 = (tensor / scale).to(fp8)
# Problem: Small values lose precision with global scale

# MXFP8 (per-block):
for block in tensor.split(block_size=32):
    block_scale = block.abs().max() / fp8_max
    block_fp8 = (block / block_scale).to(fp8)
    scales.append(block_scale)
# Better: Each block optimally scaled!
```

## Code Location

- **Infrastructure:** `megatron/core/fp8_utils.py` (MXFP8 recipe sections)
- **Requires:** TransformerEngine 2.3+

## Performance Impact

### Accuracy Comparison

| Method | Accuracy Loss | Performance |
|--------|---------------|-------------|
| BF16 | 0% (baseline) | 1.0x |
| FP8 (per-tensor) | 0.5-2% | 1.8x |
| **MXFP8** (per-block) | **0.1-0.5%** | **1.8x** |

**Best of both worlds:** Speed of FP8, accuracy closer to BF16!

## When to Use

**Use when:**
- FP8 training with accuracy critical
- H100+ hardware
- TransformerEngine 2.3+

**Configuration:**

```python
config = TransformerConfig(
    fp8='hybrid',
    fp8_format='mxfp8',  # Enable MXFP8!
)
```

## Related Optimizations

- [FP8 Training](28_compute_fp8_training.md) - Base FP8 infrastructure
- [MXFP8 Buffer Sharing](11_memory_mxfp8_buffer_sharing.md) - Memory optimization

## References

- Paper: [MICROSCALING DATA FORMATS FOR DEEP LEARNING](https://arxiv.org/abs/2302.13007)
- [Transformer Engine MXFP8](https://github.com/NVIDIA/TransformerEngine)

