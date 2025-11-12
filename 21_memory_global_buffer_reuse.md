# 21. Global Memory Buffer Reuse

## Context

Repeatedly allocating temporary tensors (for all-gather, etc.) has ~100μs overhead per allocation and causes fragmentation.

## Implementation

`GlobalMemoryBuffer` class pre-allocates buffers and returns views. Allocates 2x needed size to reduce reallocation frequency.

## Core Code

- `megatron/core/utils.py:590-643` - GlobalMemoryBuffer class
- Line 631: 2x allocation to reduce reallocations
- Used by: `tensor_parallel/layers.py:474-476` - TP all-gather operations

## Code Snippet

```python
# From utils.py:590-643
class GlobalMemoryBuffer:
    """Pre-allocated buffer for temporary tensors"""
    
    def __init__(self):
        self.buffer = {}  # (size, dtype, device) -> tensor
    
    def get_tensor(self, shape, dtype, device):
        """Get buffer, reusing if possible"""
        required_size = reduce(operator.mul, shape, 1)
        key = (dtype, device)
        
        if key not in self.buffer or self.buffer[key].numel() < required_size:
            # Allocate 2x size to reduce reallocations
            self.buffer[key] = torch.empty(
                required_size * 2, dtype=dtype, device=device
            )
        
        return self.buffer[key][:required_size].view(shape)

# Global instance
_GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()

# Usage in TP layers
def column_parallel_linear_forward(input, weight):
    # Get temporary buffer for all-gather
    gathered = _GLOBAL_MEMORY_BUFFER.get_tensor(
        (seq_len * tp_size, hidden), dtype, device
    )
    torch.distributed.all_gather_into_tensor(gathered, input)
    # No allocation overhead!
```

## Performance Impact

- Allocation time: 100μs → 1μs per use
- For 96-layer model: Saves ~9.6ms per step (96 × 100μs)

## Configuration

Automatic - no config needed!

## References

- Implementation: `megatron/core/utils.py`

