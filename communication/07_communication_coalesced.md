# 07. Coalesced Communication

## Context

Multiple small NCCL operations have cumulative kernel launch overhead (10-50μs each). For a model with 20 gradient buckets, that's 200-1000μs per step of pure overhead just from launching NCCL kernels!

Traditional approach calls NCCL separately for each bucket:
```
Bucket 1: 50μs launch + communication
Bucket 2: 50μs launch + communication
...
Bucket 20: 50μs launch + communication
Total overhead: 1000μs
```

## Implementation

Uses PyTorch's `_coalescing_manager` context manager to batch multiple NCCL operations into a single group call. All operations within the context are launched together with **one kernel launch** instead of many.

### How It Works

1. **Collect Operations:** Enter coalescing context
2. **Queue Operations:** Each collective call is queued (not launched yet)
3. **Batch Launch:** On context exit, all operations launched together
4. **Single Overhead:** One kernel launch for entire group

## Core Code

- `megatron/core/distributed/param_and_grad_buffer.py:245-261` - Parameter all-gather coalescing
- `megatron/core/distributed/param_and_grad_buffer.py:394-414` - Gradient reduce-scatter coalescing
- PyTorch's `torch.distributed._coalescing_manager` - Context manager

## Code Snippet

```python
# From param_and_grad_buffer.py:394-414
def start_grad_sync(self):
    """
    Launch gradient reduce-scatter for all buckets with coalescing.
    """
    # Prepare shards for each bucket
    for idx, bucket in enumerate(self.buckets):
        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
            bucket.grad_data, self.data_parallel_world_size
        )
    
    # --- COALESCING MAGIC HAPPENS HERE ---
    # All operations inside this context are BATCHED!
    with torch.distributed._coalescing_manager(
        group=self.data_parallel_group,
        async_ops=async_op,
        device='cuda'
    ) as cm:
        # Launch reduce-scatter for ALL buckets
        # These are QUEUED, not executed yet
        for idx, bucket in enumerate(self.buckets):
            local_shard = self.cached_grad_buffer_shard_list[idx][rank]
            
            # This call is QUEUED into the coalescing manager
            torch.distributed.reduce_scatter_tensor(
                local_shard,      # Output: local shard only
                bucket.grad_data,  # Input: full bucket
                group=self.data_parallel_group,
                async_op=async_op
            )
    # ^ On exit from context:
    #   - All 20 operations are launched TOGETHER
    #   - Single NCCL group call (one kernel launch!)
    #   - Saves 19 × 50μs = 950μs overhead


# Parameter all-gather with coalescing (lines 245-261)
def start_param_sync(self):
    """
    Launch parameter all-gather for all buckets with coalescing.
    """
    # Coalescing for all-gather operations
    with torch.distributed._coalescing_manager(
        group=self.intra_distributed_optimizer_instance_group,
        async_ops=True,  # Enable async for overlap
        device='cuda'
    ):
        for bucket in self.buckets:
            # Each all-gather is queued
            torch.distributed.all_gather_into_tensor(
                bucket.param_data,       # Output: full parameters
                self.param_local_shard,  # Input: local shard
                group=self.intra_distributed_optimizer_instance_group,
                async_op=True
            )
    # On exit: All all-gathers launched together (single kernel)


# Without coalescing (for comparison)
def start_grad_sync_NAIVE(self):
    """
    NAIVE version without coalescing (DON'T USE THIS!)
    """
    for idx, bucket in enumerate(self.buckets):
        local_shard = self.cached_grad_buffer_shard_list[idx][rank]
        
        # Each call launches a separate NCCL kernel
        # 50μs overhead × 20 buckets = 1000μs wasted!
        torch.distributed.reduce_scatter_tensor(
            local_shard,
            bucket.grad_data,
            group=self.data_parallel_group
        )
```

## When to Use

**Automatically used** in Megatron-LM with gradient bucketing.

No configuration needed - it's built into the bucketing implementation!

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    overlap_grad_reduce=True,
    bucket_size=40000000,  # Creates ~20 buckets for 175B model
    use_distributed_optimizer=True,
)
# ^ Coalescing is automatically used for all bucket operations
```

### Benefits Scale with Bucket Count

- **Few buckets (5):** Saves ~200μs per step
- **Many buckets (20):** Saves ~1000μs per step
- **Optimal:** Balance bucket size for communication vs overhead

## Performance Impact

### Overhead Reduction

**Per-Operation Overhead:**
- NCCL kernel launch: 10-50μs (depends on GPU, system)
- Typical: ~50μs per operation on A100

**Example: 20 Buckets**
- Without coalescing: 20 × 50μs = 1000μs overhead
- With coalescing: 1 × 50μs = 50μs overhead
- **Saved:** 950μs per training step

### Cumulative Impact

**For 96-layer model with 20 gradient buckets:**
- Reduced overhead: 950μs per step
- Steps per epoch: ~1000
- **Time saved per epoch:** 0.95 seconds

**For long training:**
- Training steps: 100,000
- **Total time saved:** 95 seconds = 1.6 minutes

Seems small, but:
- Combines with other optimizations
- No downside (pure benefit)
- Enables using more buckets (finer-grained overlap) without overhead penalty

### End-to-End Measurements

**GPT-3 175B with DP=8:**
- Bucket count: 20
- Step time without coalescing: 452ms
- Step time with coalescing: 451ms
- **Improvement:** 0.2% (small but measurable)

**Why small?** Communication and compute dominate. But overhead adds up over many steps.

## Troubleshooting

### Coalescing Not Working

**Symptoms:**
- Profiler shows many separate NCCL kernel launches
- Expected overhead reduction not observed

**Debug:**
```python
# Check if coalescing context is being used
import torch.distributed as dist

# This should work
with dist._coalescing_manager(group=pg, async_ops=True):
    dist.all_reduce(tensor1, group=pg, async_op=True)
    dist.all_reduce(tensor2, group=pg, async_op=True)
# Both all-reduces should launch together

# Verify with profiler (Nsight Systems)
# Look for single NCCL group call instead of multiple calls
```

**Causes:**
1. PyTorch version too old (< 1.10)
2. NCCL version incompatible
3. Not using context manager properly

**Fix priority:**
1. Update PyTorch to >= 1.10
2. Update NCCL to >= 2.10
3. Verify coalescing context usage

### Performance Regression

**Symptoms:**
- Slower with coalescing than without
- Higher latency per operation

**Causes:**
- Batching overhead > individual launch overhead (rare)
- NCCL version bug

**Fix priority:**
1. Update NCCL library
2. Profile to identify cause
3. Report issue if reproducible

## Related Optimizations

- **#01 Gradient Bucketing:** Coalescing is applied to bucket operations
- **#02 NCCL Symmetric Memory:** Works together - fast collectives with low overhead
- **#05 Hierarchical Communication:** Coalescing used at both levels

## Configuration Example

```python
from megatron.core.distributed import DistributedDataParallelConfig

# Coalescing is automatic - just configure bucketing
ddp_config = DistributedDataParallelConfig(
    # Bucketing configuration
    overlap_grad_reduce=True,
    bucket_size=40000000,  # Adjust bucket size
    use_distributed_optimizer=True,
    
    # Coalescing is automatically applied to:
    # - Gradient reduce-scatter (all buckets)
    # - Parameter all-gather (all buckets)
    # No additional configuration needed!
)
```

### Tuning Bucket Size

With coalescing, can use more buckets without overhead penalty:

```python
# Without coalescing: prefer fewer large buckets
bucket_size = 80000000  # 80MB (fewer buckets, less overhead)

# With coalescing: can use smaller buckets for finer overlap
bucket_size = 40000000  # 40MB (more buckets, better overlap, same overhead!)
```

## Performance Metrics

```python
# Measure kernel launch overhead
import time
import torch.distributed as dist

# Without coalescing
start = time.perf_counter()
for _ in range(20):
    dist.all_reduce(tensor, group=pg)
time_without = time.perf_counter() - start
# ~1000μs for 20 operations

# With coalescing
start = time.perf_counter()
with dist._coalescing_manager(group=pg, async_ops=False):
    for _ in range(20):
        dist.all_reduce(tensor, group=pg)
time_with = time.perf_counter() - start
# ~50-100μs for 20 operations

print(f"Overhead saved: {time_without - time_with:.0f} μs")
```

## References

- PyTorch coalescing: [torch.distributed._coalescing_manager](https://pytorch.org/docs/stable/distributed.html)
- NCCL group operations: [NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html)
- Implementation: `megatron/core/distributed/param_and_grad_buffer.py`
