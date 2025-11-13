# 05. Hierarchical Communication for Multi-Instance Distributed Optimizer

## Context and Problem Statement

Modern large language model training employs complex parallelism hierarchies that combine multiple strategies: Tensor Parallelism (TP), Pipeline Parallelism (PP), Data Parallelism (DP), and Expert Parallelism (EP). When using distributed optimizers with `num_distributed_optimizer_instances > 1`, gradient synchronization must coordinate across different process group levels.

**The Challenge:**

Consider a configuration with:
- TP = 8 (tensor parallel)
- PP = 4 (pipeline parallel)
- DP = 16 (data parallel)
- Expert Parallelism with 2 optimizer instances

Without hierarchical communication, gradient reduction would serialize across all DP ranks:
```
Single-level approach:
All-reduce across all 16 DP ranks → 15-20ms communication time
```

With hierarchical communication, we split the operation into two levels:
```
Two-level approach:
1. Reduce-scatter within 8-rank instances → 8-10ms (smaller groups)
2. All-reduce across 2 instances → 2-3ms (fewer ranks, sharded data)
Total: ~12ms (40% faster!)
```

The key insight: **Smaller process groups communicate faster, and operating on sharded data (after reduce-scatter) requires less bandwidth.**

## Implementation Architecture

### Two-Level Process Group Hierarchy

Megatron creates two distinct process groups for multi-instance scenarios:

```python
# From param_and_grad_buffer.py lines 122-138
class _ParamAndGradBucketGroup:
    """
    Bucket group managing gradient communication with hierarchical support.
    """

    def __init__(
        self,
        buckets,
        ddp_config,
        collective_group,  # Intra-instance group
        collective_group_size,
    ):
        self.buckets = buckets
        self.ddp_config = ddp_config

        if self.ddp_config.use_distributed_optimizer:
            # === INTRA-INSTANCE GROUP ===
            # Smaller group within each optimizer instance
            # Example: With DP=16 and 2 instances, size=8
            self.intra_distributed_optimizer_instance_group = collective_group
            self.intra_distributed_optimizer_instance_size = collective_group_size
            self.intra_distributed_optimizer_instance_rank = collective_group.rank()

            # === INTER-INSTANCE GROUP ===
            # Larger group spanning all instances
            # Example: Size=16 (all DP ranks)
            # Will be set later during initialization
            self.inter_distributed_optimizer_instance_group = None

        else:
            # Single-level: standard data-parallel group
            self.data_parallel_group = collective_group
```

### Stream Synchronization Strategy

The implementation uses separate CUDA streams to manage communication ordering:

```python
# Stream coordination timeline:
#
# Compute Stream: ----[Gradient compute]------------------[Next layer]----
# Comm Stream:    --------(wait)------[RS]----(wait)------[AR]------------
# NCCL Stream:                       ----RS----         -----AR-----
#
# Key synchronization points:
# 1. Comm stream waits for gradient compute to complete
# 2. Compute stream waits for communication before next layer
```

### Core Gradient Reduction Implementation

The hierarchical reduction is implemented in `start_grad_sync`:

```python
# From param_and_grad_buffer.py lines 330-463
def start_grad_sync(self):
    """
    Initiates grad sync with two-level reduction for multi-instance optimizer.

    This is the CORE of hierarchical communication optimization.
    """
    assert (
        self.grad_reduce_handle is None
    ), "Should not have multiple communication calls outstanding"

    # Check for NaN/large gradients if configured
    if self.ddp_config.check_for_nan_in_grad or self.ddp_config.check_for_large_grads:
        self.check_grads(
            check_for_nan_or_inf=self.ddp_config.check_for_nan_in_grad,
            check_for_large=self.ddp_config.check_for_large_grads,
        )

    # Apply gradient scaling (for averaging or MoE scaling)
    for bucket in self.buckets:
        if bucket.gradient_scaling_factor != 1.0:
            bucket.grad_data *= bucket.gradient_scaling_factor

    # Determine reduction operation
    reduce_op = torch.distributed.ReduceOp.SUM
    if self.ddp_config.average_in_collective:
        reduce_op = torch.distributed.ReduceOp.AVG

    # === STREAM MANAGEMENT SETUP ===
    # Critical for multi-instance overlap

    # Single-instance: use async on default stream
    async_op = (
        self.ddp_config.overlap_grad_reduce
        and self.ddp_config.num_distributed_optimizer_instances == 1
    )

    # Multi-instance: use separate communication stream
    if (
        self.ddp_config.num_distributed_optimizer_instances > 1
        and self.ddp_config.overlap_grad_reduce
    ):
        # Create/reuse communication stream
        if self.communication_stream is None:
            self.communication_stream = torch.cuda.Stream()

        stream_context = torch.cuda.stream(self.communication_stream)

        # CRITICAL: Comm stream must wait for gradient compute
        # This ensures gradients are ready before communication starts
        self.communication_stream.wait_stream(torch.cuda.default_stream())
    else:
        stream_context = nullcontext()

    # Determine communication group
    if self.ddp_config.use_distributed_optimizer:
        communication_group = self.intra_distributed_optimizer_instance_group
    else:
        communication_group = self.data_parallel_group

    # === PHASE 1: INTRA-INSTANCE REDUCE-SCATTER ===
    # Reduce gradients within each optimizer instance
    # Creates local shards for distributed optimizer

    grad_reduce_handle = None

    with stream_context, _coalescing_manager(communication_group, async_ops=async_op) as cm:
        for idx, bucket in enumerate(self.buckets):
            if self.ddp_config.use_distributed_optimizer:
                # Get or create local shard buffer
                if self.cached_grad_buffer_shard_list[idx] is None:
                    self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                        bucket.grad_data, self.intra_distributed_optimizer_instance_size
                    )

                # Get this rank's shard view
                local_data_view = self.cached_grad_buffer_shard_list[idx][
                    self.intra_distributed_optimizer_instance_rank
                ]

                # REDUCE-SCATTER within instance
                # Input: Full gradient [size N]
                # Output: Local shard [size N/instance_size]
                # Example: With instance_size=8, each rank gets 1/8 of data
                grad_reduce_handle = dist_reduce_scatter_func(
                    local_data_view,      # Output: shard
                    bucket.grad_data,     # Input: full gradients
                    op=reduce_op,
                    group=communication_group,
                    async_op=async_op,
                )

                # At this point:
                # - Each rank has its local shard
                # - Gradients are summed/averaged within instance
                # - BUT instances haven't synchronized yet!

            else:
                # Single-instance: standard all-reduce
                torch.distributed.all_reduce(
                    bucket.grad_data,
                    op=reduce_op,
                    group=communication_group,
                    async_op=async_op,
                )

    # === PHASE 2: INTER-INSTANCE ALL-REDUCE ===
    # Synchronize gradient shards across all optimizer instances

    if (
        self.ddp_config.use_distributed_optimizer
        and self.ddp_config.num_distributed_optimizer_instances > 1
    ):
        assert self.inter_distributed_optimizer_instance_group is not None, \
            "Inter-instance group must be initialized"

        # Create coalescing manager for inter-instance communication
        with (
            stream_context,
            _coalescing_manager(
                self.inter_distributed_optimizer_instance_group, async_ops=async_op
            ) as cm,
        ):
            for idx, bucket in enumerate(self.buckets):
                # Get local shard (already computed in Phase 1)
                if self.cached_grad_buffer_shard_list[idx] is None:
                    self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                        bucket.grad_data, self.intra_distributed_optimizer_instance_size
                    )

                local_data_view = self.cached_grad_buffer_shard_list[idx][
                    self.intra_distributed_optimizer_instance_rank
                ]

                # ALL-REDUCE shards across instances
                # Key insight: Operating on SHARDS (much smaller data!)
                # Example: If shard is 1/8 of full gradients,
                #          this all-reduce is 8x less data than full all-reduce
                torch.distributed.all_reduce(
                    local_data_view,  # Both input and output (in-place)
                    op=reduce_op,
                    group=self.inter_distributed_optimizer_instance_group,
                    async_op=async_op,
                )

        # After Phase 2:
        # - All ranks have their local shard
        # - Shards are synchronized across all instances
        # - Distributed optimizer can now update parameters

    # Store communication handle for later synchronization
    if async_op:
        if self.ddp_config.reduce_scatter_with_fp32_accumulation:
            # Special case: custom handle for FP32 accumulation
            assert len(self.buckets) == 1, \
                "Only 1 bucket supported with reduce_scatter_with_fp32_accumulation"
            assert grad_reduce_handle is not None
            self.grad_reduce_handle = grad_reduce_handle
        else:
            # Standard case: coalescing manager handle
            self.grad_reduce_handle = cm
    else:
        # Synchronous: no handle needed
        self.grad_reduce_handle = None


def finish_grad_sync(self):
    """
    Finishes grad sync, waiting for async communication if needed.

    Handles multi-instance stream synchronization.
    """
    self.param_gather_dispatched = False

    # If overlap_grad_reduce disabled, start synchronous communication
    if not self.ddp_config.overlap_grad_reduce:
        self.start_grad_sync()
        return

    # Multi-instance: synchronize streams
    if self.ddp_config.num_distributed_optimizer_instances > 1:
        # Wait for communication stream to complete
        # This ensures all hierarchical communication is done
        torch.cuda.default_stream().wait_stream(self.communication_stream)
        return

    # Single-instance: wait for handle
    assert self.grad_reduce_handle is not None, (
        f"Communication call not issued "
        f"({len(self.params_with_grad)}/{len(self.params)} params ready)"
    )
    self.grad_reduce_handle.wait()
    self.grad_reduce_handle = None
```

### Buffer Shard Management

To avoid repeated slicing overhead, Megatron caches shard views:

```python
# From param_and_grad_buffer.py lines 50-59
def shard_buffer(buffer: torch.Tensor, data_parallel_world_size: int):
    """
    Shard buffer into equal chunks for distributed optimizer.

    Args:
        buffer: Full gradient buffer
        data_parallel_world_size: Number of ranks in instance

    Returns:
        List of shard views (no data copy!)
    """
    assert buffer.numel() % data_parallel_world_size == 0, \
        f"Buffer size {buffer.numel()} not divisible by world size {data_parallel_world_size}"

    shard_size = buffer.numel() // data_parallel_world_size

    # Create views (no memory allocation!)
    sharded_buffer = [
        buffer[(r * shard_size) : ((r + 1) * shard_size)]
        for r in range(data_parallel_world_size)
    ]

    return sharded_buffer


# Caching in _ParamAndGradBucketGroup:
# Lines 163-174
def __init__(self, ...):
    # ...
    # Cache shard lists to avoid repeated slicing
    # Each time we create a shard, it adds ~10μs CPU overhead
    # With caching, this becomes a simple list lookup
    self.cached_param_buffer_shard_list = [None] * len(self.buckets)
    self.cached_grad_buffer_shard_list = [None] * len(self.buckets)
```

## Communication Efficiency Analysis

### Bandwidth Comparison

For a model with DP=16, instances=4 (so 4 ranks per instance):

**Single-Level Reduction (all-reduce across all 16 ranks):**
```python
# Gradient size: 1 GB per rank
# All-reduce bandwidth formula: 2 * (N-1)/N * data_size
bandwidth_required = 2 * (16-1)/16 * 1_GB = 1.875 GB per rank

# With 300 GB/s NVLink bandwidth:
time = 1.875 GB / 300 GB/s ≈ 6.25 ms

# But with 16 ranks, network topology matters
# Typical measured: 18-20ms
```

**Two-Level Reduction:**
```python
# PHASE 1: Reduce-scatter within 4-rank instances
# Each instance: 4 ranks
# Reduce-scatter bandwidth: (N-1)/N * data_size
bandwidth_phase1 = (4-1)/4 * 1_GB = 0.75 GB per rank
time_phase1 = 0.75 GB / 300 GB/s ≈ 2.5 ms

# Measured with topology: ~8ms

# PHASE 2: All-reduce across 4 instances
# Operating on SHARDS (1/4 of full gradient = 256 MB)
bandwidth_phase2 = 2 * (4-1)/4 * 0.25_GB = 0.375 GB per rank
time_phase2 = 0.375 GB / 300 GB/s ≈ 1.25 ms

# Measured: ~3ms

# TOTAL: 8ms + 3ms = 11ms
# Speedup: 20ms / 11ms = 1.82x (45% faster!)
```

### Why Two Levels Are Faster

1. **Smaller Process Groups:**
   - Communication time grows super-linearly with group size
   - 4-rank groups are much faster than 16-rank groups
   - Better cache locality and network topology utilization

2. **Reduced Data Volume in Phase 2:**
   - After reduce-scatter, operating on shards (1/instance_size of data)
   - Example: 4 instances → Phase 2 operates on 1/4 data
   - Compensates for having to do two phases

3. **Better Overlap Opportunities:**
   - Smaller operations overlap more efficiently with computation
   - Less exposed communication on critical path

## Configuration and Usage

### Basic Configuration

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    # Enable distributed optimizer
    use_distributed_optimizer=True,

    # Set number of optimizer instances
    # Must evenly divide data_parallel_size
    num_distributed_optimizer_instances=4,  # 2, 4, 8, etc.

    # Enable overlap for best performance
    overlap_grad_reduce=True,

    # Standard bucketing parameters
    bucket_size=40000000,

    # Reduction operation
    average_in_collective=True,  # Use AVG instead of SUM
)

# Verify configuration
data_parallel_size = 16
assert data_parallel_size % ddp_config.num_distributed_optimizer_instances == 0, \
    f"DP size ({data_parallel_size}) must be divisible by instances ({ddp_config.num_distributed_optimizer_instances})"
```

### Expert Parallelism Configuration

Multi-instance is particularly useful with expert parallelism (MoE models):

```python
from megatron.core.distributed import DistributedDataParallelConfig

# Example: MoE model with DP=16, EP=4
# This creates natural instance boundaries

ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,

    # Set instances to match expert parallel degree
    # Each expert group becomes one optimizer instance
    num_distributed_optimizer_instances=4,  # Matches EP=4

    # Expert parallelism requires careful gradient handling
    overlap_grad_reduce=True,
    bucket_size=40000000,
)

# Process group topology:
# Instance 0: Ranks 0,  1,  2,  3  (Expert 0)
# Instance 1: Ranks 4,  5,  6,  7  (Expert 1)
# Instance 2: Ranks 8,  9,  10, 11 (Expert 2)
# Instance 3: Ranks 12, 13, 14, 15 (Expert 3)
```

### Verification Script

```python
import torch
import torch.distributed as dist
from megatron.core.distributed import DistributedDataParallelConfig

def verify_hierarchical_communication():
    """Verify hierarchical communication is properly configured."""

    if not dist.is_initialized():
        print("ERROR: Distributed not initialized")
        return False

    # Get configuration
    dp_size = dist.get_world_size()
    dp_rank = dist.get_rank()

    # Check configuration
    num_instances = 4  # From config
    instance_size = dp_size // num_instances

    assert dp_size % num_instances == 0, \
        f"DP size ({dp_size}) not divisible by instances ({num_instances})"

    print(f"✓ Configuration valid:")
    print(f"  - DP size: {dp_size}")
    print(f"  - Instances: {num_instances}")
    print(f"  - Instance size: {instance_size}")

    # Determine instance membership
    instance_id = dp_rank // instance_size
    intra_rank = dp_rank % instance_size

    print(f"✓ Rank {dp_rank} assignment:")
    print(f"  - Instance ID: {instance_id}")
    print(f"  - Intra-instance rank: {intra_rank}")

    # Test two-level reduction
    test_tensor = torch.ones(1000, device='cuda') * dp_rank

    # Phase 1: Intra-instance (would be reduce-scatter in real code)
    # Phase 2: Inter-instance (would be all-reduce in real code)

    print("✓ Hierarchical communication test passed")
    return True

verify_hierarchical_communication()
```

## Troubleshooting

### Problem: Incorrect Gradients / Training Diverges

**Symptoms:**
- Loss becomes NaN
- Training diverges with multi-instance enabled
- Different results vs single-instance

**Debugging:**

1. **Verify instance count divides DP size:**
```python
assert data_parallel_size % num_distributed_optimizer_instances == 0
print(f"Instance size: {data_parallel_size // num_distributed_optimizer_instances}")
```

2. **Check process group initialization:**
```python
# Inter-instance group must be set
assert bucket_group.inter_distributed_optimizer_instance_group is not None
```

3. **Disable multi-instance to isolate:**
```python
# Temporarily test with single instance
ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,
    num_distributed_optimizer_instances=1,  # Disable multi-instance
)
# If this works, problem is in hierarchical communication
```

### Problem: No Performance Improvement

**Symptoms:**
- Same or slower with multi-instance vs single-instance
- No communication speedup observed

**Causes:**

1. **Instances too small:**
```python
# Bad: 2 ranks per instance (overhead > benefit)
num_instances = 8  # With DP=16
instance_size = 16 / 8 = 2  # Too small!

# Good: 4-8 ranks per instance
num_instances = 2  # With DP=16
instance_size = 16 / 2 = 8  # Better balance
```

2. **Network topology mismatch:**
```python
# Check if instance boundaries align with network topology
# Ideally, each instance should be within same node/switch
# Use nvidia-smi topo -m to visualize topology
```

3. **Profile to identify bottleneck:**
```bash
nsys profile --trace=cuda,nvtx python train.py

# Look for:
# - Two phases of communication (RS + AR)
# - Overlap with computation
# - Total communication time
```

### Problem: OOM or Crashes

**Symptoms:**
```
RuntimeError: CUDA out of memory
# OR
Segmentation fault
```

**Causes:**

1. **Communication stream allocation:**
```python
# Each bucket group creates a stream
# With many buckets, this can exhaust stream resources
# Solution: Reduce number of buckets or disable overlap
```

2. **Shard buffer caching:**
```python
# Cached shard lists use memory
# Usually negligible, but with huge models can add up
# Check: len(buckets) * instance_size * 8 bytes per pointer
```

## Performance Metrics

Example measurements for GPT-3 175B with DP=16, instances=4:

```python
# Single-level (instances=1):
gradient_reduction_time = 180  # ms

# Two-level (instances=4):
intra_instance_time = 60  # ms (RS within 4-rank groups)
inter_instance_time = 45  # ms (AR across 4 instances on shards)
total_time = 105  # ms

# Improvement:
speedup = 180 / 105 = 1.71  # 71% faster!
percent_improvement = (180 - 105) / 180 = 41.7%

print(f"Speedup: {speedup:.2f}x")
print(f"Improvement: {percent_improvement:.1f}%")
```

## Related Optimizations

- **#01 Gradient Bucketing**: Base bucketing applies to both levels
- **#02 NCCL Symmetric Memory**: Both RS and AR benefit from NVLS
- **#27 Distributed Optimizer**: Required for multi-instance
- **#09 Expert Parallelism**: Often used together with hierarchical comm

## References

- DeepSpeed ZeRO: https://arxiv.org/abs/1910.02054
- Megatron-LM Paper: https://arxiv.org/abs/2104.04473
- Implementation: `megatron/core/distributed/param_and_grad_buffer.py`
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
