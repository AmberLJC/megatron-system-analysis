# Global Memory Buffer Reuse

## Context

Repeatedly allocating temporary tensors (for all-gather, intermediate results, etc.) has ~100μs overhead per allocation and causes memory fragmentation. For 96 layers × multiple operations = significant waste.

**Solution:** Pre-allocate global buffers, return views. Allocate 2x needed size to reduce reallocation frequency.

## Implementation

```python
class GlobalMemoryBuffer:
    """Global buffer pool for temporary tensors"""
    
    def __init__(self):
        self._buffer = {}
    
    def get_tensor(self, size, dtype, name):
        """Get tensor view from buffer (reuses memory)"""
        key = (tuple(size), dtype, name)
        
        # Check if buffer exists and is large enough
        if key not in self._buffer or self._buffer[key].numel() < size.numel():
            # Allocate 2x size to reduce future reallocations
            alloc_size = [s * 2 for s in size]
            self._buffer[key] = torch.empty(
                alloc_size, dtype=dtype, device=torch.cuda.current_device()
            )
        
        # Return view of exact size needed (no new allocation!)
        return self._buffer[key].narrow(0, 0, size[0]).view(size)


# Global singleton
_GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()

def get_global_memory_buffer():
    return _GLOBAL_MEMORY_BUFFER
```

## Usage in TP Layers

```python
# From tensor_parallel/layers.py
def forward(ctx, input, weight, ...):
    if sequence_parallel:
        # Need temporary buffer for all-gather
        dim_size = list(input.size())
        dim_size[0] = dim_size[0] * tp_group.size()
        
        # Get buffer (reuses memory!)
        all_gather_buffer = get_global_memory_buffer().get_tensor(
            dim_size, input.dtype, "mpu"
        )
        
        # All-gather into reused buffer
        dist_all_gather_func(all_gather_buffer, input, group=tp_group)
        total_input = all_gather_buffer
```

## Code Location

- **Implementation:** `megatron/core/utils.py` lines 590-643
- **Usage:** `megatron/core/tensor_parallel/layers.py` lines 474-476, 511-512

## Performance Impact

### Allocation Time Savings

| Operation | Without Buffer | With Buffer | Savings |
|-----------|---------------|-------------|---------|
| Per alloc | 100μs | 1μs | 99μs |
| 96 layers | 9.6ms | 0.096ms | 9.5ms |

### Memory Fragmentation

- Reduces fragmentation by 50-80%
- Fewer allocator calls → better memory locality
- 2x allocation strategy amortizes reallocation cost

## When to Use

**Automatically used** by Megatron for:
- Tensor parallel all-gather operations
- Sequence parallel operations
- Any temporary buffers in core layers

## Related Optimizations

- [Tensor Parallelism Overlap](04_communication_tp_overlap.md) - Uses global buffers
- [Sequence Parallelism](03_communication_sequence_parallel.md) - Uses global buffers

## References

- PyTorch Memory Management: [CUDA Caching Allocator](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)

