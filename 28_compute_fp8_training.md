# FP8 Training Infrastructure

## Context

FP8 uses 8-bit floating point for computation, reducing memory bandwidth and enabling tensor core acceleration on H100+.

**Trade-off:** 1.5-2x speedup with minimal accuracy loss (when done correctly).

## Implementation

Complex system with:
- **Recipe management:** Delayed, tensorwise, blockwise, MXFP8
- **Context management:** First/last layer BF16 override
- **Dynamic scaling:** Factors updated every iteration
- **Parameter quantization:** On-the-fly conversion

```python
# Simplified conceptual example
class FP8Linear:
    """Linear layer with FP8 computation"""
    
    def forward(self, input, weight):
        # Quantize to FP8
        input_fp8, input_scale = quantize_to_fp8(input)
        weight_fp8, weight_scale = quantize_to_fp8(weight)
        
        # FP8 GEMM (1.5-2x faster on H100!)
        output_fp8 = fp8_gemm(input_fp8, weight_fp8)
        
        # Dequantize
        output = dequantize_from_fp8(output_fp8, input_scale * weight_scale)
        
        return output
```

## Code Location

- **Full infrastructure:** `megatron/core/fp8_utils.py` (701 lines)
- **Recipe management:** Lines 459-514
- **Context management:** Lines 516-574
- **Quantization:** Lines 142-433

## Performance Impact

### Speedup

| GPU | Precision | Training Speed | Notes |
|-----|-----------|----------------|-------|
| H100 | BF16 | 1.0x | Baseline |
| H100 | FP8 | **1.5-2x** | With TE 2.3+ |
| A100 | FP8 | No benefit | No HW support |

### Memory Benefits

- Reduced bandwidth → improved effective compute
- Can fit larger models
- Faster gradient communication

## When to Use

**Use when:**
- H100+ GPUs with FP8 tensor core support
- Model accuracy maintained with FP8
- Want maximum performance

**Configuration:**

```python
config = TransformerConfig(
    fp8='hybrid',  # or 'e4m3', hybrid uses different precisions
    fp8_margin=0,
    fp8_interval=1,
    fp8_amax_history_len=1024,
    fp8_amax_compute_algo='max',
    
    # MXFP8 for better accuracy
    fp8_format='mxfp8',  # Requires TE 2.3+
)
```

**Skip if:**
- Older GPUs without FP8 (A100, V100)
- Accuracy degradation unacceptable
- Stability issues

## FP8 Formats

### E4M3 (Standard FP8)
- **4 exponent bits, 3 mantissa bits**
- Range: 2^-6 to 448
- Good for forward pass

### E5M2 (Alternative FP8)
- **5 exponent bits, 2 mantissa bits**
- Wider range, less precision
- Good for gradients

### MXFP8 (Microscaling)
- **Block-based scaling** (not per-tensor)
- **Better accuracy** than standard FP8
- Requires TE 2.3+

## Troubleshooting

**Symptom:** Training divergence with FP8

**Fix:**
```python
# 1. Try MXFP8 (better precision)
config.fp8_format = 'mxfp8'

# 2. Keep first/last layers in BF16
# (automatically done by framework)

# 3. Adjust scaling parameters
config.fp8_amax_history_len = 2048  # Longer history
```

**Symptom:** No speedup with FP8

**Cause:** Not using H100+ or TransformerEngine not installed

**Fix:**
```bash
# Install TransformerEngine
pip install transformer-engine[pytorch]>=2.3
```

## Related Optimizations

- [MXFP8 Blockwise Scaling](29_compute_mxfp8_scaling.md) - Better FP8 accuracy
- [FP8 Inference Padding](18_memory_fp8_padding.md) - FP8 for inference
- [MXFP8 Buffer Sharing](11_memory_mxfp8_buffer_sharing.md) - Memory optimization with FP8

## References

- Paper: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- [H100 Tensor Core](https://www.nvidia.com/en-us/data-center/h100/)

