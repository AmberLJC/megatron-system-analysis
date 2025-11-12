# 01. Gradient Bucketing and Overlap

## Context

In naive distributed training, GPUs idle during gradient synchronization. For large models, this idle time can be 20-40% of total training time. The traditional approach waits until all gradients are computed before starting communication, wasting precious time.

## Implementation

Groups gradients into ~40MB buckets in **reverse topological order**. As each bucket completes during backward pass, asynchronous all-reduce/reduce-scatter is triggered automatically.

**Key optimization:** Reverse-order bucketing ensures communication starts as soon as first gradients are ready (which are computed last in the backward pass, since they correspond to the first layers in forward pass).

### How It Works

1. **Bucket Creation:** Parameters are grouped into buckets of ~40MB each
2. **Gradient Hooks:** Register hooks on each parameter to detect when `.grad` is ready
3. **Auto-Trigger:** When all parameters in a bucket have gradients, automatically launch async communication
4. **Overlap:** Communication happens while later layers compute their gradients

## Core Code

- `megatron/core/distributed/param_and_grad_buffer.py:330-507` - Async communication logic
- `megatron/core/distributed/param_and_grad_buffer.py:532-748` - Bucket creation
- `megatron/core/distributed/param_and_grad_buffer.py:490-507` - Auto-trigger mechanism
- `megatron/core/distributed/param_and_grad_buffer.py:62-105` - Bucket class definition

## Code Snippet

```python
# From param_and_grad_buffer.py:330-358
def start_grad_sync(self):
    """
    Initiates grad sync (all-reduce or reduce-scatter) communication operations
    for all buckets in the bucket group.
    """
    assert (
        self.grad_reduce_handle is None
    ), "Should not have multiple communication calls outstanding at once"

    # Scale gradients BEFORE communication
    # gradient_scaling_factor handles averaging and MoE scaling
    for bucket in self.buckets:
        if bucket.gradient_scaling_factor != 1.0:
            bucket.grad_data *= bucket.gradient_scaling_factor

    # Decide reduce operation (SUM or AVG)
    reduce_op = torch.distributed.ReduceOp.SUM
    if self.ddp_config.average_in_collective:
        reduce_op = torch.distributed.ReduceOp.AVG

    # Enable async operation for overlap - this is THE KEY!
    async_op = (
        self.ddp_config.overlap_grad_reduce
        and self.ddp_config.num_distributed_optimizer_instances == 1
    )

    # Launch coalesced communication for all buckets
    # Using coalescing manager batches multiple ops into single NCCL call
    with _coalescing_manager(group, async_ops=async_op):
        for idx, bucket in enumerate(self.buckets):
            if self.ddp_config.use_distributed_optimizer:
                # Reduce-scatter (ZeRO-style): Each rank gets its shard
                dist_reduce_scatter_func(
                    local_shard, bucket.grad_data,
                    group=group, async_op=async_op
                )
            else:
                # Standard all-reduce: All ranks get full gradient
                torch.distributed.all_reduce(
                    bucket.grad_data, group=group, 
                    op=reduce_op, async_op=async_op
                )

# Auto-trigger mechanism (lines 490-507)
def register_grad_ready(self, param: torch.nn.Parameter):
    """
    Called by autograd hooks when param.grad is ready.
    Automatically launches communication when bucket is complete.
    """
    assert param in self.params, "Param not recognized"
    
    bucket = self.param_to_bucket[param]
    self.params_with_grad.add(param)

    # Auto-trigger when ALL params in bucket group have gradients ready
    if len(self.params_with_grad) == len(self.params):
        self.start_grad_sync()  # Launch communication NOW!
        # ^ At this point, communication starts while later layers 
        #   are still computing gradients (perfect overlap!)
```

### Bucket Structure

```python
# From param_and_grad_buffer.py:62-105
class _ParamAndGradBucket:
    """
    Bucket to keep track of a subset of the model's parameters and gradients.
    """
    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],  # View in param buffer
        grad_data: torch.Tensor,              # View in grad buffer
        offset: int,                          # Offset in buffer
        numel_unpadded: int,                  # Actual elements (no padding)
        gradient_scaling_factor: float,       # For averaging/MoE scaling
        bucket_id: int,
    ):
        self.params_list = params
        self.grad_data = grad_data  # Contiguous buffer for efficient NCCL
        self.gradient_scaling_factor = gradient_scaling_factor
        # ... rest of initialization
```

## When to Use

**Always enable** - This is essential for any distributed training.

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    overlap_grad_reduce=True,              # ENABLE async gradient reduction
    bucket_size=40000000,                  # 40MB buckets (tune: 20-80MB)
    use_distributed_optimizer=True,        # Use reduce-scatter instead of all-reduce
    average_in_collective=True,            # Use AVG not SUM
)
```

### Tuning Bucket Size

- **Smaller buckets (20MB):** More fine-grained overlap, but more kernel launches
- **Larger buckets (80MB):** Fewer kernel launches, but coarser overlap
- **Sweet spot:** 40MB for most models (default)

## Performance Impact

### Throughput Improvement
- **20-40% throughput improvement** vs naive all-reduce after backward
- With proper overlap: **80-95% of communication time hidden** behind computation
- Critical path reduction: Hides ~seconds of communication per training step

### Communication Timeline

**Without Bucketing:**
```
Forward → Backward (all layers) → [IDLE] All-Reduce → Optimizer Step
                                   ^^^^^ 2-5 seconds wasted
```

**With Bucketing + Overlap:**
```
Forward → Backward Layer 96 → [Bucket 1 All-Reduce starts]
                 Layer 95 → [Bucket 2 All-Reduce starts]  
                 ...           [Communication overlaps!]
                 Layer 1  → [All buckets complete] → Optimizer Step
```

### Example Measurements

For GPT-3 175B model with DP=64:
- Gradient size: ~350GB across all ranks
- Without overlap: 3.2s communication time on critical path
- With overlap: 0.3s exposed communication (90% hidden)
- **Result:** 2.9s saved per step = 30% throughput improvement

## Troubleshooting

### Communication Not Overlapping

**Symptoms:**
- High exposed communication time in profiler
- Low GPU utilization during backward pass

**Causes:**
1. `overlap_grad_reduce=False` (check config!)
2. `CUDA_DEVICE_MAX_CONNECTIONS ≠ 1` (for TP overlap)
3. Bucket size too large (reduces overlap opportunities)

**Fix priority:**
1. Verify `overlap_grad_reduce=True` in config
2. Reduce bucket size to 20-40MB
3. Check with profiler (Nsight Systems) to see overlap

### Low Communication Bandwidth

**Symptoms:**
- Communication takes longer than expected
- Bandwidth << peak (e.g., 50 GB/s instead of 300 GB/s on DGX)

**Causes:**
1. Bucket size not optimal for network
2. Not using proper NCCL algorithms
3. Unpadded buffers (see optimization #18)

**Fix priority:**
1. Enable NCCL symmetric memory (#02)
2. Tune bucket size (try 40MB, 80MB)
3. Verify padding is enabled

## Related Optimizations

- **#02 NCCL Symmetric Memory:** Improves bandwidth of the actual communication
- **#07 Coalesced Communication:** Reduces kernel launch overhead for buckets
- **#27 Distributed Optimizer:** Changes all-reduce to reduce-scatter (same overlap principle)
- **#16 Gradient Sync in Bubbles:** Hides gradient sync in pipeline bubbles

## Configuration Example

```python
# Complete configuration for gradient bucketing
ddp_config = DistributedDataParallelConfig(
    # Core bucketing settings
    overlap_grad_reduce=True,              # Enable async overlap
    bucket_size=40000000,                  # 40MB buckets
    
    # Distributed optimizer (use reduce-scatter)
    use_distributed_optimizer=True,        # Shard optimizer state
    
    # Collective operation settings
    average_in_collective=True,            # Use AVG not SUM
    reduce_scatter_with_fp32_accumulation=True,  # Better precision
    
    # Advanced
    check_for_nan_in_grad=False,           # Disable for performance
    bucket_cap_mb=40,                      # Same as bucket_size
)
```

## References

- Original paper: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- PyTorch DDP uses similar bucketing: [torch.nn.parallel.DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- Megatron-LM implementation: `megatron/core/distributed/param_and_grad_buffer.py`
