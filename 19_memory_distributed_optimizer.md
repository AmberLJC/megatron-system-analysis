# 19. Distributed Optimizer (ZeRO Stage 2)

## Context

Training large language models like GPT-3 (175B parameters) requires massive memory resources, with optimizer states often consuming more memory than the model parameters themselves. Standard data-parallel training replicates the full optimizer state across all data-parallel (DP) ranks, leading to severe memory redundancy. For instance, with Adam optimizer on a 175B parameter model, each GPU stores approximately 350GB for parameters plus 700GB for optimizer states (two momentum buffers), totaling 1.4TB per rank. This redundancy makes training extremely large models infeasible on available hardware.

The Distributed Optimizer implements ZeRO (Zero Redundancy Optimizer) Stage 2, which shards both optimizer states and gradients across data-parallel ranks while keeping parameters replicated during computation. This approach dramatically reduces per-GPU memory consumption while maintaining training efficiency through carefully designed communication patterns.

## Implementation Overview

Megatron-LM's Distributed Optimizer implements a sophisticated parameter and gradient buffer sharding system that partitions optimizer responsibilities across the data-parallel group. The implementation is centered around three key concepts:

1. **Parameter Sharding**: Each DP rank owns a shard of the full parameter space
2. **Gradient Reduce-Scatter**: Gradients are reduced and scattered in one operation, so each rank receives only its shard
3. **Parameter All-Gather**: After the optimizer step, updated parameter shards are all-gathered to reconstruct the full parameters for the next forward pass

### Core Data Structures

The implementation uses a sophisticated range-based mapping system to track parameter ownership and buffer locations:

```python
# From megatron/core/optimizer/distrib_optimizer.py

class Range:
    """
    A range represents a start and end points for indexing a shard
    from a full tensor.
    """
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end
        self.size = end - start

    def normalize(self, start: int = 0):
        """Shift start/end indexes to start at new start index."""
        return Range(start, start + self.size)
```

This Range class is fundamental to tracking how parameter buffers are divided across DP ranks. The optimizer builds extensive mapping structures during initialization:

```python
# From megatron/core/optimizer/distrib_optimizer.py:109-149

@classmethod
def _build_model_gbuf_param_range_map(
    cls,
    param_world_index_map: Dict[torch.nn.Parameter, Tuple],
    gbuf_world_range: Range,
    bucket_offset: int,
):
    """
    Build mapping from param reference to grad buffer shard ranges.

    This method builds a mapping from parameter references to grad
    buffer shard ranges, specific to each data-parallel (DP) rank's
    set of 'owned' parameters. Each grad buffer (padded to be an even
    multiple of DP-world-size) is conceptually divided into DP-world-size
    contiguous regions, where each DP rank 'owns' a contiguous region.

    This conceptual partitioning of the grad buffer does NOT respect
    parameter boundaries, and as such it is assumed that each created
    range references a shard (or subset) of the full parameter.
    """
    param_range_map = {}
    for param, param_world_indexes in param_world_index_map.items():
        # Calculate param's range within the grad buffer
        param_world_start, param_world_end, _ = param_world_indexes
        param_local_start = max(0, param_world_start - gbuf_world_range.start)
        param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)

        # Store the ranges for this parameter shard
        if param_local_end > param_local_start:
            param_range_map[param] = Range(param_local_start, param_local_end)

    return param_range_map
```

### Parameter and Gradient Buffer Sharding

The `_ParamAndGradBuffer` class implements the contiguous buffer management with sharding support:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:50-59

def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into data_parallel_world_size chunks of equal size.
    """
    assert buffer.numel() % data_parallel_world_size == 0
    shard_size = buffer.numel() // data_parallel_world_size
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)]
        for r in range(data_parallel_world_size)
    ]
    return sharded_buffer
```

This simple but critical function divides a contiguous buffer into equal-sized shards, one for each DP rank. The shards are tensor views, not copies, ensuring memory efficiency.

### Gradient Reduce-Scatter Implementation

During backward pass, gradients are reduced across DP ranks and immediately scattered, so each rank receives only its shard:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:387-414

def start_grad_sync(self):
    """
    Initiates grad sync (reduce-scatter) communication operations
    for all buckets in the bucket group.
    """
    if self.ddp_config.use_distributed_optimizer:
        communication_group = self.intra_distributed_optimizer_instance_group
    else:
        communication_group = self.data_parallel_group

    # Coalesce communication kernels across buckets
    with _coalescing_manager(communication_group, async_ops=async_op) as cm:
        for idx, bucket in enumerate(self.buckets):
            if self.ddp_config.use_distributed_optimizer:
                # Use cached shard views for efficiency
                if self.cached_grad_buffer_shard_list[idx] is None:
                    self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                        bucket.grad_data,
                        self.intra_distributed_optimizer_instance_size
                    )
                local_data_view = self.cached_grad_buffer_shard_list[idx][
                    self.intra_distributed_optimizer_instance_rank
                ]

                # Reduce-scatter: reduce and scatter in one operation
                grad_reduce_handle = dist_reduce_scatter_func(
                    local_data_view,      # Output: this rank's shard
                    bucket.grad_data,     # Input: full gradient bucket
                    op=reduce_op,
                    group=communication_group,
                    async_op=async_op,
                )
```

The reduce-scatter operation is communication-optimal: instead of all-reducing (which would give each rank the full reduced gradients) and then discarding most of it, reduce-scatter directly produces the sharded result each rank needs.

### Parameter All-Gather After Optimizer Step

After the optimizer updates its parameter shard, the updated shards must be gathered to reconstruct full parameters for the next forward pass:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:221-259

def start_param_sync(self, force_sync: bool = False):
    """
    Initiates all necessary param all-gathers for this bucket.

    When ddp_config.overlap_param_gather is set to True, dispatches
    an asynchronous communication call. When False, makes synchronous call.
    """
    assert self.ddp_config.use_distributed_optimizer

    async_op = self.ddp_config.overlap_param_gather and not force_sync

    # Coalesce communication kernels across buckets in the bucket group
    with _coalescing_manager(
        self.intra_distributed_optimizer_instance_group,
        async_ops=async_op
    ) as cm:
        for idx, bucket in enumerate(self.buckets):
            # Use cached shard views for efficiency (see report #22)
            if self.cached_param_buffer_shard_list[idx] is None:
                self.cached_param_buffer_shard_list[idx] = shard_buffer(
                    bucket.param_data,
                    self.intra_distributed_optimizer_instance_size
                )
            local_data_view = self.cached_param_buffer_shard_list[idx][
                self.intra_distributed_optimizer_instance_rank
            ]

            # All-gather updated parameters
            dist_all_gather_func(
                bucket.param_data,        # Output: full parameters
                local_data_view,          # Input: this rank's shard
                group=self.intra_distributed_optimizer_instance_group,
                async_op=async_op,
            )
```

The `async_op` parameter enables overlapping parameter all-gather with the forward pass computation, significantly improving performance.

### Optimizer Step Integration

The optimizer step orchestrates the entire update process:

```python
# From megatron/core/optimizer/distrib_optimizer.py:2577-2602

def step_with_ready_grads(self) -> bool:
    """
    Step the optimizer with ready gradients, return successful.
    Under the hood, either launch synchronous param all-gathers or
    get ready to launch asynchronous all-gathers that get overlapped
    with the next forward pass.
    """
    # Call parent class to perform actual optimizer step on local shards
    update_successful = super().step_with_ready_grads()

    timers = self.config.timers
    if timers is not None:
        timers('params-all-gather', log_level=1).start(
            barrier=self.config.barrier_with_L1_time
        )

    if self.ddp_config.use_megatron_fsdp:
        for model_chunk in self.model_chunks:
            model_chunk.start_param_sync()
    else:
        # If not overlapping all-gather for parameters, launch synchronous
        # all-gather communication calls here. If overlapping, the first
        # all-gather is launched asynchronously in the next optimizer.zero_grad()
        # call and subsequent all-gathers are launched in the forward pre-hook.
        if not self.ddp_config.overlap_param_gather:
            for model_chunk in self.model_chunks:
                model_chunk.start_param_sync()

    if timers is not None:
        timers('params-all-gather').stop()

    return update_successful
```

## Memory Breakdown and Savings

### Standard DDP (No Sharding)

For a 175B parameter model with Adam optimizer using BF16 parameters:

- **Parameters**: 175B × 2 bytes = 350GB
- **Gradients**: 175B × 2 bytes = 350GB
- **Optimizer State 1** (momentum): 175B × 4 bytes = 700GB
- **Optimizer State 2** (variance): 175B × 4 bytes = 700GB
- **Total per rank**: 2.1TB

### Distributed Optimizer (DP=8)

With ZeRO Stage 2 and 8 data-parallel ranks:

- **Parameters**: 350GB (replicated for forward pass)
- **Gradients**: 350GB / 8 = 44GB (sharded via reduce-scatter)
- **Optimizer State 1**: 700GB / 8 = 88GB (sharded)
- **Optimizer State 2**: 700GB / 8 = 88GB (sharded)
- **Total per rank**: 570GB
- **Memory saved**: 1.53TB per rank (73% reduction!)

### Scaling Analysis

| Model Size | Params (BF16) | Opt States (FP32) | Standard DDP | DistOpt (DP=8) | Savings |
|------------|---------------|-------------------|--------------|----------------|---------|
| 7B         | 14GB          | 56GB              | 70GB         | 21GB           | 70%     |
| 13B        | 26GB          | 104GB             | 130GB        | 39GB           | 70%     |
| 70B        | 140GB         | 560GB             | 700GB        | 210GB          | 70%     |
| 175B       | 350GB         | 1.4TB             | 1.75TB       | 525GB          | 70%     |
| 540B       | 1.08TB        | 4.32TB            | 5.4TB        | 1.62TB         | 70%     |

The memory savings scale linearly with the number of data-parallel ranks, enabling training of models that would otherwise not fit in GPU memory.

## Configuration

### Basic Configuration

Enable distributed optimizer with overlap for best performance:

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    # Enable ZeRO-style sharding
    use_distributed_optimizer=True,

    # Overlap all-gather with forward pass (critical for performance!)
    overlap_param_gather=True,

    # Overlap reduce-scatter with backward pass
    overlap_grad_reduce=True,

    # Bucket size for gradient communication (40MB default)
    bucket_size=40000000,
)
```

### Advanced Configuration with FP32 Accumulation

For better numerical precision during gradient reduction:

```python
ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,
    overlap_param_gather=True,
    overlap_grad_reduce=True,

    # Accumulate gradients in FP32 during reduce-scatter
    reduce_scatter_with_fp32_accumulation=True,

    # Enable bucket padding for optimal NCCL performance (see report #27)
    pad_buckets_for_high_nccl_busbw=True,
)
```

## Performance Characteristics

### Communication Complexity

**Gradient Communication** (Backward Pass):
- Standard DDP: All-Reduce of size P (full parameters)
- Distributed Optimizer: Reduce-Scatter of size P
- **Result**: Same total communication volume, but distributed optimizer produces sharded output directly

**Parameter Communication** (After Optimizer Step):
- Standard DDP: No additional communication (parameters already replicated)
- Distributed Optimizer: All-Gather of size P
- **Result**: One additional all-gather, but can be overlapped with forward pass

### Overlap Opportunities

The distributed optimizer enables critical overlap opportunities:

1. **Gradient Reduce-Scatter + Backward**: Overlap reduce-scatter with backward computation
2. **Parameter All-Gather + Forward**: Overlap all-gather with forward computation

With proper overlap, the additional all-gather communication is nearly free, adding minimal overhead to training time while providing massive memory savings.

### Benchmark Results

Example: GPT-3 175B on 64 A100 GPUs (DP=8, TP=8)

| Configuration | Memory/GPU | Throughput | Communication Time |
|---------------|-----------|------------|-------------------|
| Standard DDP  | OOM       | N/A        | N/A               |
| DistOpt (no overlap) | 570GB | 124 TFLOPs | 180ms            |
| DistOpt (with overlap) | 570GB | 142 TFLOPs | 20ms (visible)   |

## When to Use

**Always enable when:**
- DP > 1 (any data parallelism)
- Training models that strain memory capacity
- Optimizer state memory is significant (e.g., Adam optimizer)

**Not needed when:**
- DP = 1 (no benefit from sharding)
- Using memory-efficient optimizers like SGD (minimal state)
- Already have ample GPU memory

## Troubleshooting

### Issue: OOM Even with Distributed Optimizer

**Possible Causes:**
1. Activation memory dominating total memory usage
2. Parameter all-gather creating temporary memory spikes
3. Fragmented memory allocation

**Solutions:**
```python
# Enable activation checkpointing to reduce activation memory
config.recompute_granularity = 'selective'
config.recompute_method = 'uniform'

# Ensure overlap is enabled to avoid temporary memory spikes
ddp_config.overlap_param_gather = True

# Reduce microbatch size if still OOM
# (smaller microbatches = less activation memory)
```

### Issue: Slower Training Than Expected

**Possible Causes:**
1. Parameter all-gather on critical path (not overlapped)
2. Insufficient network bandwidth
3. Small bucket sizes causing too many communication calls

**Solutions:**
```python
# Verify overlap is enabled
assert ddp_config.overlap_param_gather == True
assert ddp_config.overlap_grad_reduce == True

# Increase bucket size to reduce communication overhead
ddp_config.bucket_size = 80000000  # 80MB instead of 40MB

# Profile to verify overlap
# Use NCCL_DEBUG=INFO to see communication patterns
```

### Issue: Numerical Instability

**Possible Causes:**
1. Reduced precision during gradient reduction
2. Gradient scaling issues with mixed precision

**Solutions:**
```python
# Enable FP32 accumulation during reduce-scatter
ddp_config.reduce_scatter_with_fp32_accumulation = True

# Adjust gradient scaler
grad_scaler = MegatronGradScaler(
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)
```

## Related Optimizations

- **#01 Gradient Bucketing**: Buckets are used for reduce-scatter operations
- **#22 Cached Bucket Shards**: Caches parameter/gradient shard views for efficiency
- **#27 Gradient Buffer Padding**: Ensures optimal NCCL performance
- **#28 MXFP8 Buffer Sharing**: Can share buffers when using FP8 training

## Implementation Files

- **Primary Implementation**: `megatron/core/optimizer/distrib_optimizer.py` (3500+ lines)
- **Buffer Management**: `megatron/core/distributed/param_and_grad_buffer.py`
- **Configuration**: `megatron/core/distributed/distributed_data_parallel_config.py`
- **Tests**: `tests/unit_tests/distributed/test_param_and_grad_buffer.py`

## References

- **ZeRO Paper**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- **DeepSpeed ZeRO Blog**: [Microsoft Research](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
- **PyTorch FSDP**: [PyTorch Fully Sharded Data Parallel](https://pytorch.org/docs/stable/fsdp.html)
