# 22. Cached Bucket Shards

## Context

Creating tensor views via slicing has 10-50μs CPU overhead per operation. For 20 buckets per step, that's 200-1000μs wasted!

## Implementation

Pre-computes and caches tensor shard views for each bucket, reusing across iterations.

## Core Code

- `megatron/core/distributed/param_and_grad_buffer.py:168-174` - Cache initialization
- Lines 249-255: Parameter buffer shard caching
- Lines 397-403: Gradient buffer shard caching

## Code Snippet

```python
# From param_and_grad_buffer.py:168-174, 397-403
class _ParamAndGradBucketGroup:
    def __init__(self, buckets, ...):
        # Pre-compute and cache shard views
        self.cached_grad_buffer_shard_list = []
        
        for bucket in self.buckets:
            # Create list of shard views (one per DP rank)
            shard_views = shard_buffer(
                bucket.grad_data,
                self.data_parallel_world_size
            )
            self.cached_grad_buffer_shard_list.append(shard_views)
            # ^ Computed ONCE, reused every iteration!
    
    def start_grad_sync(self):
        """Use cached shards (no slicing overhead)"""
        for idx, bucket in enumerate(self.buckets):
            # Get pre-computed shard view (no overhead!)
            local_shard = self.cached_grad_buffer_shard_list[idx][rank]
            
            # Reduce-scatter with cached view
            torch.distributed.reduce_scatter_tensor(
                local_shard, bucket.grad_data, group=self.dp_group
            )
```

## Performance Impact

- CPU overhead saved: 10-50μs per bucket per iteration
- For 20 buckets: ~200-1000μs saved per step

## Configuration

Automatic - no config needed!

## References

- Implementation: `megatron/core/distributed/param_and_grad_buffer.py`

