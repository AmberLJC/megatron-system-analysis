# Cached Bucket Shards

## Context

Creating tensor views via slicing has 10-50μs CPU overhead per operation. With 20 buckets × 2 operations (param AG + grad RS) × many iterations = significant cumulative overhead.

**Solution:** Pre-compute and cache tensor shard views, reuse across iterations.

## Implementation

```python
class _ParamAndGradBucketGroup:
    """Bucket group with cached shard views"""
    
    def __init__(self, ...):
        # Pre-compute shard views for all buckets
        self.cached_param_buffer_shard_list = []
        self.cached_grad_buffer_shard_list = []
        
        for bucket in self.buckets:
            # Cache parameter shards for all ranks
            param_shards = []
            for rank in range(self.data_parallel_size):
                start = rank * bucket.shard_size
                end = (rank + 1) * bucket.shard_size
                shard = bucket.param_data[start:end]  # View, not copy!
                param_shards.append(shard)
            
            self.cached_param_buffer_shard_list.append(param_shards)
            
            # Cache gradient shards
            grad_shards = []
            for rank in range(self.data_parallel_size):
                start = rank * bucket.shard_size
                end = (rank + 1) * bucket.shard_size
                shard = bucket.grad_data[start:end]
                grad_shards.append(shard)
            
            self.cached_grad_buffer_shard_list.append(grad_shards)


def start_grad_sync(self):
    """Use cached shards (no slicing overhead!)"""
    for idx, bucket in enumerate(self.buckets):
        rank = torch.distributed.get_rank(self.data_parallel_group)
        
        # Use pre-computed shard view (fast!)
        local_shard = self.cached_grad_buffer_shard_list[idx][rank]
        
        torch.distributed.reduce_scatter_tensor(
            local_shard, bucket.grad_data, ...
        )
```

## Code Location

- **Cache initialization:** `megatron/core/distributed/param_and_grad_buffer.py` lines 168-174
- **Parameter shard caching:** Lines 249-255
- **Gradient shard caching:** Lines 397-403

## Performance Impact

### CPU Overhead Savings

| Scenario | Operations | Without Cache | With Cache | Saved |
|----------|------------|---------------|------------|-------|
| 20 buckets/step | 40 | 40 × 30μs = 1.2ms | ~0 | 1.2ms |
| 1000 steps | 40K | 40s | ~0 | 40s |

### Per-Step Savings

- 20 buckets: ~1ms CPU time saved per step
- Reduces CPU bottleneck in communication
- Enables better overlap of compute and communication

## When to Use

**Automatically used** with:
- Gradient bucketing
- Distributed optimizer
- Any reduce-scatter/all-gather operations

## Related Optimizations

- [Gradient Buffer](10_memory_gradient_buffer.md) - Creates the buckets that are cached
- [Distributed Optimizer](19_memory_distributed_optimizer.md) - Uses cached shards

## References

- PyTorch Tensor Views: [Tensor Views](https://pytorch.org/docs/stable/tensor_view.html)

