# 19. MXFP8 Parameter/Gradient Buffer Sharing

## Context

Storing both BF16 parameters and FP32 gradients doubles memory usage. With FP8, can share a single buffer!

Standard approach: Separate buffers for parameters and gradients
- BF16 parameters: 175GB (for 175B model)
- FP32 gradients: 350GB
- **Total:** 525GB

## Implementation

For MXFP8 training with distributed optimizer:
- First half of buffer: BF16 parameters
- After parameter all-gather: Zero out to reuse for FP32/BF16 gradients

**Constraint:** Only with `reuse_grad_buf_for_mxfp8_param_ag=True`

## Core Code

- `megatron/core/distributed/param_and_grad_buffer.py:716-733` - Buffer sharing logic
- Line 728: BF16 params in first half
- Line 328: Zero out after all-gather

## Code Snippet

```python
# From param_and_grad_buffer.py:716-733
def _allocate_shared_buffer_fp8(self):
    """Share buffer between parameters and gradients for FP8"""
    
    if self.reuse_grad_buf_for_mxfp8_param_ag:
        # Calculate sizes
        param_size = self.param_numel  # BF16 parameters
        grad_size = self.grad_numel     # FP32/BF16 gradients
        
        # Allocate single buffer (max of param or grad size)
        buffer_size = max(param_size, grad_size * 2)  # *2 for FP32
        
        self.shared_buffer = torch.zeros(
            buffer_size, dtype=torch.uint8, device='cuda'
        )
        
        # Phase 1 (forward): Use for parameters
        self.param_data = self.shared_buffer[:param_size].view(
            torch.bfloat16
        )
        
        # Phase 2 (backward): Reuse for gradients
        # After all-gather, parameters no longer needed
        # Zero out and reuse for gradients!
        
    else:
        # Standard: separate buffers
        self.param_data = torch.zeros(self.param_numel, dtype=torch.bfloat16)
        self.grad_data = torch.zeros(self.grad_numel, dtype=torch.float32)


def _reuse_buffer_after_all_gather(self):
    """After parameter all-gather, reuse buffer for gradients"""
    
    # All-gather completed, parameters materialized
    # Now zero out buffer to reuse for gradients
    self.shared_buffer.zero_()
    
    # Create gradient view into same buffer
    self.grad_data = self.shared_buffer[:self.grad_numel].view(
        self.grad_dtype
    )
```

## When to Use

**FP8 training only** with distributed optimizer:

```python
ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,
    reuse_grad_buf_for_mxfp8_param_ag=True,  # Enable buffer sharing
)

config = TransformerConfig(
    fp8='hybrid',  # Enable FP8
)
```

**Skip if:**
- Not using FP8
- Don't need extreme memory savings

## Performance Impact

**Memory saved:** Up to 50% of parameter buffer size
- 175B model: ~175GB saved across DP group
- Per rank with DP=8: ~22GB saved

**Example:** GPT-3 175B, FP8, DP=8
- Standard: 44GB params + 44GB grads = 88GB
- Shared: max(44GB, 44GB) = 44GB
- **Saved:** 44GB per rank (50%)

## References

- FP8 training: See optimization #36
- Implementation: `megatron/core/distributed/param_and_grad_buffer.py`

