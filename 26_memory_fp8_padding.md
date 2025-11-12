# 26. FP8 Inference Padding

## Context

FP8 GEMM requires input aligned to 16/32 elements. Variable sequence lengths violate this constraint.

## Implementation

Auto-pads sequences before FP8 GEMM, unpads after. Trades wasted computation for FP8 speedup.

## Core Code

- `megatron/core/fp8_utils.py:593-691` - Padding wrapper layers

## Code Snippet

```python
# FP8 padding for variable sequence lengths
def fp8_linear_with_padding(input, weight):
    """Pad input for FP8 alignment"""
    
    seq_len = input.size(0)
    alignment = 16  # FP8 requires 16-element alignment
    
    # Pad to alignment
    padded_len = ((seq_len + alignment - 1) // alignment) * alignment
    
    if padded_len > seq_len:
        # Pad with zeros
        padding = torch.zeros(
            padded_len - seq_len, *input.shape[1:],
            dtype=input.dtype, device=input.device
        )
        padded_input = torch.cat([input, padding], dim=0)
    else:
        padded_input = input
    
    # FP8 GEMM on padded input (fast!)
    padded_output = fp8_gemm(padded_input, weight)
    
    # Unpad result
    output = padded_output[:seq_len]
    
    return output
```

## When to Use

**FP8 inference only** with variable sequence lengths

**Skip if:**
- Fixed sequence lengths (no padding needed)
- Padding overhead > FP8 benefit

## Performance Impact

**Cost:** Wasted computation on padding (typically <10%)
**Benefit:** Enables FP8 (1.5-2x speedup)
**Net:** Usually positive with sequence variance <50%

## References

- FP8 training: See optimization #36
- Implementation: `megatron/core/fp8_utils.py`

