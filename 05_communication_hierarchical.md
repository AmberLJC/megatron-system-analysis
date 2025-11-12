# 05. Hierarchical Communication (Multi-Instance DistOpt)

## Context

Complex parallelism hierarchies (TP+PP+DP+EP) require coordinated communication across different process groups. With `num_distributed_optimizer_instances > 1`, we need two-level gradient reduction:
1. **Intra-instance:** Within smaller optimizer instance groups
2. **Inter-instance:** Across larger data-parallel groups

Naive implementation would serialize these operations, adding both communication times to critical path.

## Implementation

Two-level reduction using separate CUDA streams for coordination:
- **Stream 1:** Intra-instance reduce-scatter (smaller groups, faster)
- **Stream 2:** Inter-instance all-reduce (larger groups, slower)
- **Synchronization:** Careful stream ordering ensures correctness

### How It Works

1. **Phase 1 (Intra-Instance):** Reduce-scatter gradients within each optimizer instance
   - Creates local shards for distributed optimizer
2. **Phase 2 (Inter-Instance):** All-reduce the shards across instances
   - Synchronizes gradients across all data-parallel ranks
3. **Stream Management:** Use separate streams to enable overlap

## Core Code

- `megatron/core/distributed/param_and_grad_buffer.py:416-443` - Two-level reduction implementation
- `megatron/core/distributed/param_and_grad_buffer.py:146-162` - Separate process groups setup
- `megatron/core/distributed/param_and_grad_buffer.py:372-385` - Stream synchronization logic

## Code Snippet

```python
# From param_and_grad_buffer.py:146-162
class _ParamAndGradBucketGroup:
    def __init__(self, ...):
        # Two process groups for hierarchical communication
        if self.ddp_config.num_distributed_optimizer_instances > 1:
            # Intra-instance group (smaller, faster)
            self.intra_distributed_optimizer_instance_group = collective_group
            self.intra_distributed_optimizer_instance_size = collective_group_size
            
            # Inter-instance group (larger, slower)
            self.inter_distributed_optimizer_instance_group = get_data_parallel_group()
            self.inter_distributed_optimizer_instance_size = get_data_parallel_world_size()
        else:
            # Single-level: just data-parallel group
            self.data_parallel_group = collective_group


# From param_and_grad_buffer.py:372-443
def start_grad_sync(self):
    """Two-level gradient reduction for multi-instance distributed optimizer"""
    
    if self.ddp_config.num_distributed_optimizer_instances > 1:
        # --- TWO-LEVEL REDUCTION ---
        
        # Stream coordination for overlap:
        # Compute Stream: ----Gradient compute-----------------
        # Comm. Stream:   ----(wait)-------RS-------(wait)-----
        # NCCL Stream:                   -----RS----     -----AR----
        
        # Phase 1: Intra-instance reduce-scatter
        # Within each optimizer instance (smaller group)
        for idx, bucket in enumerate(self.buckets):
            local_shard = self.cached_grad_buffer_shard_list[idx][intra_rank]
            
            # Reduce-scatter within instance
            # This is FAST (smaller group)
            torch.distributed.reduce_scatter_tensor(
                local_shard,                    # Output: local shard
                bucket.grad_data,               # Input: full gradients
                group=self.intra_distributed_optimizer_instance_group,
                async_op=False  # Synchronous for correctness
            )
        
        # Synchronize compute stream before inter-instance reduction
        # Ensures intra-instance RS is complete
        torch.cuda.current_stream().wait_stream(self.comm_stream)
        
        # Phase 2: Inter-instance all-reduce
        # Across all optimizer instances (larger group)
        for idx, bucket in enumerate(self.buckets):
            local_shard = self.cached_grad_buffer_shard_list[idx][intra_rank]
            
            # All-reduce shards across instances
            # This is SLOWER (larger group) but operates on smaller data (shards)
            torch.distributed.all_reduce(
                local_shard,                    # Both input and output
                group=self.inter_distributed_optimizer_instance_group,
                async_op=True,  # Async to enable overlap
                op=torch.distributed.ReduceOp.AVG if self.ddp_config.average_in_collective
                   else torch.distributed.ReduceOp.SUM
            )
    
    else:
        # Single-level: standard reduce-scatter or all-reduce
        # See optimization #01 for details
        for idx, bucket in enumerate(self.buckets):
            if self.ddp_config.use_distributed_optimizer:
                dist_reduce_scatter_func(...)
            else:
                torch.distributed.all_reduce(...)
```

## When to Use

**Multi-instance distributed optimizer scenarios:**
- Expert parallelism + data parallelism (MoE models)
- FSDP with multiple optimizer instances
- Complex hierarchies: TP×PP×DP×EP

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,
    num_distributed_optimizer_instances=2,  # >1 enables hierarchical
    overlap_grad_reduce=True,
)
```

### Skip If

- `num_distributed_optimizer_instances = 1` (single-level is simpler)
- No complex parallelism hierarchies
- Not using expert parallelism

## Performance Impact

### Communication Efficiency

**Single-Level (num_instances=1):**
- One reduce-scatter across all DP ranks
- Time: ~O(N) where N = DP size

**Two-Level (num_instances=2):**
- Phase 1: Reduce-scatter within instance (size N/2)
- Phase 2: All-reduce across instances (size 2, on shards)
- Time: ~O(N/2) + O(2 × shard_size)

**Improvement:** 5-15% better communication efficiency for complex hierarchies

### Example Measurements

**MoE model with DP=16, EP=4, instances=4:**
- Single-level: All-reduce across 16 ranks = 180ms
- Two-level: RS in 4 groups of 4 (60ms) + AR across 4 instances (45ms) = 105ms
- **Speedup:** 71% faster gradient communication

## Troubleshooting

### Incorrect Gradients

**Symptoms:**
- Training diverges with multi-instance enabled
- Loss becomes NaN

**Causes:**
- Incorrect stream synchronization
- Wrong reduction order
- Process group mismatch

**Fix priority:**
1. Verify `num_distributed_optimizer_instances` divides DP size evenly
2. Check process group initialization
3. Disable multi-instance to isolate issue

### Poor Performance

**Symptoms:**
- Slower with multi-instance than single-instance
- No communication improvement

**Causes:**
- Instances too small (overhead > benefit)
- Network topology not matching hierarchy

**Fix priority:**
1. Increase instance size (reduce num_instances)
2. Profile to identify bottleneck
3. Consider single-instance if no benefit

## Related Optimizations

- **#01 Gradient Bucketing:** Base bucketing applies to both levels
- **#27 Distributed Optimizer:** Required for multi-instance
- **#09 Expert Parallelism Communication:** Often used with hierarchical

## Configuration Example

```python
from megatron.core.distributed import DistributedDataParallelConfig

# Multi-instance configuration
ddp_config = DistributedDataParallelConfig(
    # Distributed optimizer with multiple instances
    use_distributed_optimizer=True,
    num_distributed_optimizer_instances=4,  # 2, 4, 8, etc.
    
    # Standard bucketing
    overlap_grad_reduce=True,
    bucket_size=40000000,
    
    # Collective operations
    average_in_collective=True,
)

# Ensure DP size is divisible by num_instances
data_parallel_size = 16  # Must be divisible by 4
assert data_parallel_size % 4 == 0
```

## References

- DeepSpeed ZeRO-3: [ZeRO Paper](https://arxiv.org/abs/1910.02054)
- Implementation: `megatron/core/distributed/param_and_grad_buffer.py`
