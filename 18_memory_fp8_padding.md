# FP8 Inference Padding

## Context

FP8 GEMM requires input aligned to 16/32 elements. Variable sequence lengths violate this requirement, preventing FP8 acceleration.

**Trade-off:** Wasted computation on padding vs 1.5-2x FP8 speedup.

## Implementation

Auto-pad sequences before FP8 GEMM, unpad after. Net benefit when sequence length variance < 50%.

```python
class FP8LinearWithPadding:
    """FP8 linear layer with automatic padding"""
    
    def forward(self, input, weight):
        original_seq_len = input.size(0)
        
        # Pad to multiple of 16 for FP8
        padded_seq_len = ((original_seq_len + 15) // 16) * 16
        
        if padded_seq_len > original_seq_len:
            # Pad input
            padding = padded_seq_len - original_seq_len
            input_padded = F.pad(input, (0, 0, 0, padding))
        else:
            input_padded = input
        
        # FP8 GEMM (fast!)
        output_padded = fp8_gemm(input_padded, weight)
        
        # Unpad output
        output = output_padded[:original_seq_len]
        
        return output
```

## Code Location

- **Implementation:** `megatron/core/fp8_utils.py` lines 593-691

## Performance Impact

### Cost vs Benefit

| Seq Length | Padding % | Wasted Compute | FP8 Speedup | Net Benefit |
|------------|-----------|----------------|-------------|-------------|
| 1024 → 1024 | 0% | 0% | 1.8x | +80% |
| 1000 → 1008 | 0.8% | 0.8% | 1.8x | +78% |
| 900 → 912 | 1.3% | 1.3% | 1.8x | +77% |
| 600 → 608 | 1.3% | 1.3% | 1.8x | +77% |

**Rule:** Net benefit when padding < 50% (almost always!)

## When to Use

**Use when:**
- FP8 inference
- Variable sequence lengths
- FP8 speedup > padding cost (usual case)

**Skip if:**
- Fixed sequence lengths (no padding needed)
- Padding overhead > FP8 benefit (rare)

## Related Optimizations

- [FP8 Training](28_compute_fp8_training.md) - FP8 training infrastructure
- [Sequence Parallelism](03_communication_sequence_parallel.md) - Affects sequence length

## References

- [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)

