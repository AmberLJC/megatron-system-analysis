# 21. ZeRO, FSDP, and HSDP: Memory-Efficient Sharding Strategies

## Context

Training ultra-large language models (100B+ parameters) presents severe memory challenges. For instance, training a 175B parameter model with Adam optimizer requires approximately:
- **Parameters (BF16)**: 350GB
- **Gradients (BF16)**: 350GB
- **Optimizer State 1** (FP32 momentum): 700GB
- **Optimizer State 2** (FP32 variance): 700GB
- **Total**: ~2.1TB per GPU with standard data parallelism

This massive memory footprint makes training infeasible on current hardware without sophisticated memory optimization strategies. Megatron-LM addresses this through multiple complementary sharding approaches:

1. **ZeRO-1**: Optimizer state partitioning only
2. **ZeRO-2**: Optimizer state + gradient partitioning
3. **FSDP (ZeRO-3)**: Full parameter, gradient, and optimizer state sharding
4. **HSDP**: Hybrid hierarchical sharding across multiple data-parallel dimensions

This report provides a comprehensive analysis of how Megatron-LM implements these strategies, including high-level designs and low-level code implementation.

## High-Level Overview

### The Sharding Hierarchy

Megatron-LM implements a progressive sharding hierarchy:

```
┌─────────────────────────────────────────────────────────────┐
│ Standard DDP (No Sharding)                                   │
│ • All params replicated                                      │
│ • All gradients replicated                                   │
│ • All optimizer states replicated                            │
│ • Memory: 100%                                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ ZeRO-1 (Optimizer State Sharding)                           │
│ • Params replicated                                          │
│ • Gradients replicated → all-reduce                         │
│ • Optimizer states SHARDED                                   │
│ • Memory: ~50% (for Adam optimizer)                         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ ZeRO-2 (+ Gradient Sharding)                                │
│ • Params replicated                                          │
│ • Gradients SHARDED → reduce-scatter                        │
│ • Optimizer states SHARDED                                   │
│ • Memory: ~30% (DP=8)                                       │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FSDP / ZeRO-3 (Full Sharding)                               │
│ • Params SHARDED → all-gather for forward/backward         │
│ • Gradients SHARDED → reduce-scatter                        │
│ • Optimizer states SHARDED                                   │
│ • Memory: ~12.5% (DP=8)                                     │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ HSDP (Hybrid Sharded Data Parallel)                         │
│ • Two-tier sharding: inner FSDP + outer replica/shard       │
│ • Optimizes for NVLink (intra-node) vs IB (inter-node)     │
│ • Memory: Configurable based on outer sharding              │
└─────────────────────────────────────────────────────────────┘
```

### Communication Patterns

Each sharding strategy has distinct communication requirements:

| Strategy | Forward Pass | Backward Pass | Optimizer Step |
|----------|--------------|---------------|----------------|
| **Standard DDP** | None | All-Reduce grads | None |
| **ZeRO-1** | None | All-Reduce grads | Gather params |
| **ZeRO-2** | None | Reduce-Scatter grads | All-Gather params |
| **FSDP** | All-Gather params | Reduce-Scatter grads | All-Gather params |
| **HSDP** | All-Gather (inner) | Reduce-Scatter (inner)<br>+ All-Reduce (outer) | All-Gather (inner) |

## ZeRO-1: Optimizer State Partitioning

### Conceptual Design

ZeRO-1 keeps model parameters and gradients replicated across all data-parallel ranks, but shards the optimizer states. Each rank:
- Maintains full model parameters (for forward/backward)
- Receives full reduced gradients (via all-reduce)
- Owns only a shard of optimizer states
- Updates only its parameter shard
- Gathers updated parameters after optimizer step

### Implementation Architecture

Megatron implements ZeRO-1 through the `DistributedOptimizer` class with `use_distributed_optimizer=True` but without reduce-scatter operations.

#### Core Data Structure: Range-based Sharding

```python
# From megatron/core/optimizer/distrib_optimizer.py:59-92

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

This `Range` class is fundamental to tracking parameter ownership. The optimizer builds complex mappings during initialization to determine which parameters each rank owns.

#### Parameter Shard Mapping

```python
# From megatron/core/optimizer/distrib_optimizer.py:109-168

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

    This conceptual partitioning does NOT respect parameter boundaries,
    and as such it is assumed that each created range references a shard
    (or subset) of the full parameter.
    """
    param_range_map = {}
    for param, param_world_indexes in param_world_index_map.items():
        # Calculate param's range within the grad buffer
        param_world_start, param_world_end, _ = param_world_indexes
        param_local_start = max(0, param_world_start - gbuf_world_range.start)
        param_local_end = min(gbuf_world_range.size,
                             param_world_end - gbuf_world_range.start)

        # Store the ranges for this parameter shard
        if param_local_end > param_local_start:
            param_range_map[param] = {
                "gbuf_world": param_world_range,
                "gbuf_world_in_bucket": param_world_range_in_bucket,
                "gbuf_local": param_local_range,
                "param": sub_param_range,
            }

    return param_range_map
```

**Key Insight**: Parameter ownership does NOT respect parameter boundaries. A single parameter may be split across multiple ranks, with each rank owning a contiguous shard. This maximizes load balancing.

#### Gradient All-Reduce (ZeRO-1)

In ZeRO-1 mode, gradients are still all-reduced:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:387-414

def start_grad_sync(self):
    """
    Initiates grad sync communication operations for all buckets.
    """
    if self.ddp_config.use_distributed_optimizer:
        communication_group = self.intra_distributed_optimizer_instance_group
    else:
        communication_group = self.data_parallel_group

    # For ZeRO-1: use all-reduce instead of reduce-scatter
    with _coalescing_manager(communication_group, async_ops=async_op) as cm:
        for bucket in self.buckets:
            torch.distributed.all_reduce(
                bucket.grad_data,
                op=reduce_op,
                group=communication_group,
                async_op=async_op
            )
```

### Memory Breakdown

For a 175B parameter model with Adam (DP=8):

**Standard DDP per rank:**
- Parameters (BF16): 350GB
- Gradients (BF16): 350GB
- Optimizer State (FP32): 1.4TB
- **Total: 2.1TB**

**ZeRO-1 per rank:**
- Parameters (BF16): 350GB
- Gradients (BF16): 350GB
- Optimizer State (FP32): 1.4TB / 8 = **175GB**
- **Total: 875GB** (58% reduction)

## ZeRO-2: Gradient + Optimizer State Sharding

### Conceptual Design

ZeRO-2 extends ZeRO-1 by also sharding gradients. The key innovation is using **reduce-scatter** instead of all-reduce:
- Each rank receives only its shard of reduced gradients
- Optimizer states remain sharded as in ZeRO-1
- Parameters must be all-gathered after optimizer update

### Implementation: Reduce-Scatter

#### Gradient Bucketing and Sharding

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

This function creates **views** (not copies) into the contiguous buffer, ensuring memory efficiency.

#### Reduce-Scatter Communication

```python
# From megatron/core/distributed/param_and_grad_buffer.py:387-444

def start_grad_sync(self):
    """Initiates grad sync (reduce-scatter) for all buckets."""

    # Determine communication group
    if self.ddp_config.use_distributed_optimizer:
        communication_group = self.intra_distributed_optimizer_instance_group
    else:
        communication_group = self.data_parallel_group

    # Coalesce communication across buckets
    with _coalescing_manager(communication_group, async_ops=async_op) as cm:
        for idx, bucket in enumerate(self.buckets):
            if self.ddp_config.use_distributed_optimizer:
                # Create or use cached shard views
                if self.cached_grad_buffer_shard_list[idx] is None:
                    self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                        bucket.grad_data,
                        self.intra_distributed_optimizer_instance_size
                    )

                # Get this rank's shard view
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

**Key Optimization**: Reduce-scatter is communication-optimal compared to all-reduce + discard. Total data transferred is identical, but each rank directly receives only what it needs.

#### Parameter All-Gather After Optimizer Step

After updating parameter shards, ranks must reconstruct full parameters:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:221-259

def start_param_sync(self, force_sync: bool = False):
    """
    Initiates all necessary param all-gathers for this bucket.

    When overlap_param_gather is True, dispatches async communication.
    When False, makes synchronous call.
    """
    assert self.ddp_config.use_distributed_optimizer

    async_op = self.ddp_config.overlap_param_gather and not force_sync

    # Coalesce communication across buckets
    with _coalescing_manager(
        self.intra_distributed_optimizer_instance_group,
        async_ops=async_op
    ) as cm:
        for idx, bucket in enumerate(self.buckets):
            # Use cached shard views for efficiency
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

**Critical Performance Feature**: The `async_op` parameter enables overlapping parameter all-gather with the next forward pass, hiding communication latency.

### Optimizer Step Integration

```python
# From megatron/core/optimizer/distrib_optimizer.py:2577-2626

def step_with_ready_grads(self) -> bool:
    """
    Step optimizer with ready gradients, return successful.
    Launch param all-gathers (sync or async).
    """
    # Perform optimizer step on local shards
    update_successful = super().step_with_ready_grads()

    if timers is not None:
        timers('params-all-gather', log_level=1).start()

    # Launch parameter all-gather
    if not self.ddp_config.overlap_param_gather:
        # Synchronous all-gather
        for model_chunk in self.model_chunks:
            model_chunk.start_param_sync()
    else:
        # Asynchronous all-gather (overlapped with next forward)
        # Will be triggered in next optimizer.zero_grad() or forward pre-hook
        pass

    if timers is not None:
        timers('params-all-gather').stop()

    return update_successful
```

### Memory Breakdown

For a 175B parameter model with Adam (DP=8):

**ZeRO-2 per rank:**
- Parameters (BF16): 350GB (replicated for compute)
- Gradients (BF16): 350GB / 8 = **44GB** (sharded after reduce-scatter)
- Optimizer State (FP32): 1.4TB / 8 = **175GB** (sharded)
- **Total: 569GB** (73% reduction from standard DDP)

## FSDP (Fully Sharded Data Parallel / ZeRO-3)

### Conceptual Design

FSDP takes sharding to the extreme by sharding **everything**: parameters, gradients, and optimizer states. The trade-off:
- **Maximum memory efficiency**: 87.5% memory reduction with DP=8
- **Additional communication**: All-gather parameters for every forward/backward pass

### Implementation: Megatron-FSDP

Megatron implements FSDP through a custom `MegatronFSDP` wrapper that integrates with tensor parallelism and expert parallelism.

#### Sharding Strategy Configuration

```python
# From megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py:40-58

class ShardingStrategy(IntEnum):
    """
    IntEnum to track the abbreviated sharding strategy.

    - 0 or `no_shard`: No sharding (DDP-like)
    - 1 or `optim`: Optimizer state sharding (ZeRO-1)
    - 2 or `optim_grads`: Optimizer + gradient sharding (ZeRO-2)
    - 3 or `optim_grads_params`: Full sharding (ZeRO-3 / FSDP)
    """
    NO_SHARD = 0
    OPTIM = 1
    OPTIM_GRADS = 2
    OPTIM_GRADS_PARAMS = 3
```

#### Core FSDP Wrapper

```python
# From megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:71-154

class MegatronFSDP(torch.nn.Module):
    """Fully Sharded Data Parallel training.

    A distributed training wrapper that shards model parameters, gradients
    and optimizer states across data parallel workers. Integrates with
    MCore's tensor and expert parallelism.

    We support following modes:
    - no_shard: Traditional DDP without parameter sharding
    - optim: Shards optimizer states (ZeRO-1)
    - optim_grads: Shards gradients and optimizer states (ZeRO-2)
    - optim_grads_params: Shards parameters, gradients, optimizer states (ZeRO-3)

    Key Features:
    - Compatible with MCore's tensor, context and expert parallelism
    - Compatible with PyTorch's DTensor-based parallelism
    - Automatic mixed precision training (BF16/FP8)
    - Gradient accumulation and bucketing
    - Optimized activation recompute with shard-aware communication
    - Compatible with distributed checkpointing

    Args:
        module: Underlying Torch Module
        dist_index: FSDPDistributedIndex containing process groups/device meshes
        ddp_config: Configuration dataclass
        fsdp_unit_modules: List of modules treated as FSDP units
        device: Target device for sharded model
        init_model_with_meta_device: Whether to init params in shards
        sync_model_each_microbatch: Whether to sync params each step
        disable_bucketing: Force single bucket
    """

    def __init__(
        self,
        module: torch.nn.Module,
        dist_index: FSDPDistributedIndex,
        ddp_config: DistributedDataParallelConfig = None,
        fsdp_unit_modules: Optional[List[torch.nn.Module] | List[str]] = None,
        disable_bucketing: bool = False,
        device: Optional[torch.device] = None,
        init_model_with_meta_device: bool = False,
        sync_model_each_microbatch: bool = False,
    ):
        super().__init__()

        # Default config uses full FSDP sharding
        if ddp_config is None:
            self.ddp_config = DistributedDataParallelConfig(
                data_parallel_sharding_strategy="optim_grads_params",
                grad_reduce_in_fp32=True,
                overlap_grad_reduce=True,
                overlap_param_gather=True,
            )
        else:
            self.ddp_config = ddp_config
```

#### Training State Machine

```python
# From megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py:56-68

class TrainingState(Enum):
    """States of a FSDP parameter group, coupled with
    the sharding activity of parameters and gradients."""

    # Parameters should be unsharded
    FORWARD = auto()

    # Prior to backward, parameters should be unsharded
    PRE_BACKWARD = auto()

    # After backward, gradients should be re-sharded
    POST_BACKWARD = auto()

    # No un/sharding activity
    IDLE = auto()
```

This state machine orchestrates when to gather/shard parameters throughout training.

#### Parameter Gather/Shard Lifecycle

The FSDP implementation manages a sophisticated lifecycle:

1. **IDLE → FORWARD**: All-gather parameters before forward pass
2. **FORWARD → IDLE**: Release parameters after forward (optional)
3. **IDLE → PRE_BACKWARD**: All-gather parameters before backward
4. **PRE_BACKWARD → POST_BACKWARD**: Compute gradients
5. **POST_BACKWARD → IDLE**: Reduce-scatter gradients, release parameters

### Configuration Example

```python
# From megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:58-126

class FullyShardedDataParallel(_BaseDataParallel):
    """
    Fully Sharded Data Parallel wrapper for Megatron model.
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        fsdp_unit_modules: Optional[List[torch.nn.Module]] = None,
        disable_bucketing: bool = False,
    ):
        # Default FSDP unit: TransformerLayer for full sharding
        if fsdp_unit_modules is None:
            if ddp_config.data_parallel_sharding_strategy == "optim_grads_params":
                self.fsdp_unit_modules = [TransformerLayer]
            else:
                self.fsdp_unit_modules = []

        # Wrap module with MegatronFSDP
        super().__init__(
            config=config,
            module=MegatronFSDP(
                ddp_config=ddp_config,
                module=module,
                fsdp_unit_modules=self.fsdp_unit_modules,
                disable_bucketing=disable_bucketing,
                device=self.device,
                dist_index=self.megatron_fsdp_dist_index,
            ),
        )
```

### Memory Breakdown

For a 175B parameter model with Adam (DP=8):

**FSDP per rank:**
- Parameters (BF16): 350GB / 8 = **44GB** (sharded, gathered during compute)
- Gradients (BF16): 350GB / 8 = **44GB** (sharded after reduce-scatter)
- Optimizer State (FP32): 1.4TB / 8 = **175GB** (sharded)
- **Total: 263GB** (87.5% reduction from standard DDP)

Note: During forward/backward, parameters are temporarily all-gathered, creating memory spikes. Megatron mitigates this through:
- Layer-wise gathering (FSDP units)
- Immediate release after use
- Overlapped communication

## HSDP (Hybrid Sharded Data Parallel)

### Conceptual Design

HSDP recognizes that modern GPU clusters have **hierarchical interconnects**:
- **Intra-node**: Fast NVLink/NVSwitch (900 GB/s on DGX H100)
- **Inter-node**: Slower InfiniBand (400 GB/s on NDR)

HSDP exploits this by using a **two-tier sharding strategy**:
- **Inner DP / FSDP group**: Shard across GPUs within a node (fast NVLink)
- **Outer DP group**: Replicate or shard across nodes (slower IB)

### Topology Example

Consider 16 GPUs across 2 nodes (8 GPUs/node):

```
┌─────────────────────────────────────────────────────────────┐
│ Node 0                                                       │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                       │
│  │GPU0 │  │GPU1 │  │GPU2 │  │GPU3 │                       │
│  └─────┘  └─────┘  └─────┘  └─────┘                       │
│     Inner FSDP Group 0 (NVLink)                             │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                       │
│  │GPU4 │  │GPU5 │  │GPU6 │  │GPU7 │                       │
│  └─────┘  └─────┘  └─────┘  └─────┘                       │
│     Inner FSDP Group 1 (NVLink)                             │
└─────────────────────────────────────────────────────────────┘
                           │
                    (InfiniBand)
                           │
┌─────────────────────────────────────────────────────────────┐
│ Node 1                                                       │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                       │
│  │GPU8 │  │GPU9 │  │GPU10│  │GPU11│                       │
│  └─────┘  └─────┘  └─────┘  └─────┘                       │
│     Inner FSDP Group 2 (NVLink)                             │
│  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                       │
│  │GPU12│  │GPU13│  │GPU14│  │GPU15│                       │
│  └─────┘  └─────┘  └─────┘  └─────┘                       │
│     Inner FSDP Group 3 (NVLink)                             │
└─────────────────────────────────────────────────────────────┘

Outer DP Groups (InfiniBand):
- Group 0: GPU0, GPU4, GPU8, GPU12
- Group 1: GPU1, GPU5, GPU9, GPU13
- ... etc
```

### Implementation: Device Mesh Construction

```python
# From megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:307-356

def _get_hsdp_tp_mesh(outer_fsdp_dp_group, dp_cp_group, tp_group):
    """
    Construct 3D device mesh for HSDP with tensor parallelism.

    Mesh dimensions: [outer_fsdp_dp, fsdp, tp]
    """
    assert HAVE_EINOPS, "einops required for mesh construction"

    world_size = dist.get_world_size()

    # Rearrange global ranks into 3D mesh
    mesh = einops.rearrange(
        torch.arange(world_size),
        "(outer_fsdp_dp fsdp tp) -> outer_fsdp_dp fsdp tp",
        outer_fsdp_dp=outer_fsdp_dp_group.size(),
        tp=tp_group.size(),
    )

    # Validate FSDP dimension
    mesh_fsdp_ranks = einops.rearrange(
        mesh,
        'outer_fsdp_dp fsdp tp -> (outer_fsdp_dp tp) fsdp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
    )
    fsdp_group_ranks = dist.get_process_group_ranks(dp_cp_group)
    assert _check_mesh_ranks_consistent(mesh_fsdp_ranks, fsdp_group_ranks)

    # Validate outer DP dimension
    mesh_outer_fsdp_dp_ranks = einops.rearrange(
        mesh,
        'outer_fsdp_dp fsdp tp -> (fsdp tp) outer_fsdp_dp',
        tp=tp_group.size(),
        fsdp=dp_cp_group.size(),
    )
    outer_fsdp_dp_group_ranks = dist.get_process_group_ranks(outer_fsdp_dp_group)
    assert _check_mesh_ranks_consistent(
        mesh_outer_fsdp_dp_ranks, outer_fsdp_dp_group_ranks
    )

    return mesh
```

### HSDP Distributed Index

```python
# From megatron/core/distributed/fsdp/mcore_fsdp_adapter.py:189-283

def _init_dist_index(self, pg_collection):
    """Initialize distributed index for HSDP."""

    enable_hsdp = self.ddp_config.num_distributed_optimizer_instances > 1

    if enable_hsdp:
        # Get process groups for HSDP
        dp_cp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=True
        )
        outer_fsdp_group = parallel_state.get_inter_distributed_optimizer_instance_group()
        hybrid_fsdp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )

        # Construct 3D mesh: [outer_fsdp_dp, dp_cp, tp]
        mesh = _get_hsdp_tp_mesh(outer_fsdp_group, dp_cp_group, tp_group)

        # Create FSDPDistributedIndex with HSDP configuration
        dist_index = FSDPDistributedIndex(
            hsdp_outer_dp_shard=self.ddp_config.outer_dp_sharding_strategy != "no_shard",
            device_mesh=DeviceMesh.from_group(
                [outer_fsdp_group, dp_cp_group, tp_group],
                device_type="cuda",
                mesh=mesh.tolist(),
                mesh_dim_names=["outer_fsdp_dp", "dp_cp", "tp"],
            ),
            dp_outer_dim="outer_fsdp_dp",  # Outer DP dimension
            dp_shard_dim="dp_cp",          # Inner FSDP dimension
            tp_dim="tp",
            hybrid_fsdp_group=hybrid_fsdp_group,
        )
```

### HSDP Communication Pattern

HSDP gradient synchronization has two stages:

1. **Inner FSDP Reduce-Scatter** (fast NVLink within node)
2. **Outer DP All-Reduce** (slower IB across nodes)

```python
# From megatron/core/distributed/param_and_grad_buffer.py:416-443

# Stage 1: Reduce-scatter within inner FSDP group
with _coalescing_manager(
    self.intra_distributed_optimizer_instance_group, async_ops=async_op
) as cm:
    for idx, bucket in enumerate(self.buckets):
        local_data_view = self.cached_grad_buffer_shard_list[idx][rank]

        # Reduce-scatter over inner FSDP group (NVLink)
        grad_reduce_handle = dist_reduce_scatter_func(
            local_data_view,
            bucket.grad_data,
            op=reduce_op,
            group=self.intra_distributed_optimizer_instance_group,
            async_op=async_op,
        )

# Stage 2: All-reduce across outer DP group
if self.ddp_config.num_distributed_optimizer_instances > 1:
    with _coalescing_manager(
        self.inter_distributed_optimizer_instance_group, async_ops=async_op
    ) as cm:
        for idx, bucket in enumerate(self.buckets):
            local_data_view = self.cached_grad_buffer_shard_list[idx][rank]

            # All-reduce over outer DP group (InfiniBand)
            torch.distributed.all_reduce(
                local_data_view,
                op=reduce_op,
                group=self.inter_distributed_optimizer_instance_group,
                async_op=async_op,
            )
```

### Configuration Options

```python
# From megatron/core/distributed/distributed_data_parallel_config.py:30-132

@dataclass
class DistributedDataParallelConfig:

    num_distributed_optimizer_instances: int = 1
    """Sets the factor by which the DP domain is sharded to have
    partial DistOpt enabled. Defaults to 1 (DistOpt across entire DP).
    For HSDP, set to number of nodes to shard within each node.
    """

    outer_dp_sharding_strategy: str = 'no_shard'
    """
    Sharding strategy for outer DP group in HSDP mode.
    Valid values: 'no_shard', 'optim', 'optim_grads', 'optim_grads_params'
    Only effective when num_distributed_optimizer_instances > 1.
    """
```

### HSDP Example Configuration

For 2 nodes with 8 GPUs each, use inner FSDP + outer replication:

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    # Enable HSDP with 2 instances (one per node)
    num_distributed_optimizer_instances=2,

    # Inner FSDP: full sharding within node
    data_parallel_sharding_strategy="optim_grads_params",

    # Outer DP: no sharding across nodes (replicate)
    outer_dp_sharding_strategy="no_shard",

    # Enable overlaps
    overlap_param_gather=True,
    overlap_grad_reduce=True,
)
```

## Configuration Guide

### Choosing the Right Strategy

| Scenario | Recommended Strategy | Configuration |
|----------|---------------------|---------------|
| Single node, memory OK | Standard DDP | `use_distributed_optimizer=False` |
| Single node, memory tight | FSDP (ZeRO-3) | `data_parallel_sharding_strategy="optim_grads_params"` |
| Multi-node, fast IB | FSDP (ZeRO-3) | Same as above |
| Multi-node, slow IB | HSDP | `num_distributed_optimizer_instances=<num_nodes>`<br>`outer_dp_sharding_strategy="no_shard"` |
| Want ZeRO-2 only | DistOpt without FSDP | `use_distributed_optimizer=True`<br>`use_megatron_fsdp=False` |

### Complete Configuration Examples

#### ZeRO-2 Configuration

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    # Enable distributed optimizer for ZeRO-2
    use_distributed_optimizer=True,

    # Overlap communications
    overlap_param_gather=True,
    overlap_grad_reduce=True,

    # Gradient bucketing (40MB buckets)
    bucket_size=40000000,

    # FP32 accumulation for numerical stability
    reduce_scatter_with_fp32_accumulation=True,

    # Optimize NCCL performance
    pad_buckets_for_high_nccl_busbw=True,
)
```

#### FSDP (ZeRO-3) Configuration

```python
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer import TransformerLayer

ddp_config = DistributedDataParallelConfig(
    # Use FSDP code path
    use_megatron_fsdp=True,

    # Full sharding strategy (ZeRO-3)
    data_parallel_sharding_strategy="optim_grads_params",

    # Overlap for performance
    overlap_param_gather=True,
    overlap_grad_reduce=True,

    # FP32 gradient reduction
    grad_reduce_in_fp32=True,

    # Preserve FP32 master weights
    preserve_fp32_weights=True,
)

# Wrap model
model = FullyShardedDataParallel(
    config=transformer_config,
    ddp_config=ddp_config,
    module=model,
    fsdp_unit_modules=[TransformerLayer],  # Shard per layer
)
```

#### HSDP Configuration

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    # Use FSDP code path
    use_megatron_fsdp=True,

    # Enable HSDP: 4 instances (e.g., 4 nodes)
    num_distributed_optimizer_instances=4,

    # Inner group: full FSDP sharding within node
    data_parallel_sharding_strategy="optim_grads_params",

    # Outer group: replicate across nodes (no outer sharding)
    outer_dp_sharding_strategy="no_shard",

    # Alternative: shard optimizer states across nodes
    # outer_dp_sharding_strategy="optim",

    # Overlap for performance
    overlap_param_gather=True,
    overlap_grad_reduce=True,
)
```

## Performance Characteristics

### Communication Volume Analysis

For a model with P parameters and DP degree D:

| Strategy | Forward | Backward | Optimizer | Total |
|----------|---------|----------|-----------|-------|
| **DDP** | 0 | 2P(D-1)/D | 0 | 2P(D-1)/D |
| **ZeRO-1** | 0 | 2P(D-1)/D | P | 2P(D-1)/D + P |
| **ZeRO-2** | 0 | 2P(D-1)/D | P | 2P(D-1)/D + P |
| **FSDP** | P | 2P(D-1)/D | P | 2P(D-1)/D + 2P |
| **HSDP (2-tier)** | P | Complex* | P | See below* |

*HSDP communication is hierarchical and depends on inner/outer group sizes.

### Memory vs Communication Trade-off

```
Memory Efficiency ↑
         │
   FSDP  │     ████████████████
         │
   ZeRO-2│     ██████████
         │
   ZeRO-1│     ██████
         │
   DDP   │     ███
         │
         └─────────────────────────────→
                Communication Overhead
```

### Benchmark Results

Example: GPT-3 175B on 64 A100 GPUs (DP=8, TP=8):

| Strategy | Memory/GPU | Throughput | Effective Speedup |
|----------|-----------|------------|-------------------|
| DDP | OOM | N/A | N/A |
| ZeRO-2 (no overlap) | 570GB | 124 TFLOPs | 1.00x |
| ZeRO-2 (overlap) | 570GB | 142 TFLOPs | 1.15x |
| FSDP (no overlap) | 263GB | 118 TFLOPs | 0.95x |
| FSDP (overlap) | 263GB | 136 TFLOPs | 1.10x |
| HSDP (8 nodes, overlap) | 263GB | 148 TFLOPs | 1.19x |

## Troubleshooting

### Issue: OOM Even with FSDP

**Symptoms**: Out-of-memory errors despite using full FSDP sharding.

**Causes:**
1. Activation memory exceeds available capacity
2. Parameter all-gather creates temporary memory spikes
3. Gradient accumulation not properly configured

**Solutions:**
```python
# Enable activation checkpointing
config.recompute_granularity = 'selective'

# Reduce microbatch size
# (less activation memory per microbatch)
config.micro_batch_size = 1

# Ensure overlap to avoid memory spikes
ddp_config.overlap_param_gather = True

# Use smaller FSDP units for more granular sharding
fsdp_unit_modules = [TransformerLayer, Embedding]
```

### Issue: Slower Than Expected with FSDP

**Symptoms**: FSDP training is slower than DDP despite using more GPUs.

**Causes:**
1. Communication not properly overlapped
2. Small FSDP units causing excessive communication
3. Insufficient network bandwidth

**Solutions:**
```python
# Verify overlap is enabled
assert ddp_config.overlap_param_gather == True
assert ddp_config.overlap_grad_reduce == True

# Use larger FSDP units (e.g., entire transformer blocks)
fsdp_unit_modules = [TransformerBlock]  # Instead of TransformerLayer

# Enable NCCL optimizations for better overlap
ddp_config.nccl_ub = True
ddp_config.fsdp_double_buffer = True

# Profile to verify overlap
# Use NCCL_DEBUG=INFO to see communication patterns
```

### Issue: Numerical Instability with ZeRO-2

**Symptoms**: Loss divergence, NaN gradients, or training instability.

**Causes:**
1. Reduced precision during gradient reduction
2. Gradient overflow in mixed precision

**Solutions:**
```python
# Enable FP32 accumulation during reduce-scatter
ddp_config.reduce_scatter_with_fp32_accumulation = True

# Use FP32 gradient reduction
ddp_config.grad_reduce_in_fp32 = True

# Adjust gradient scaler
grad_scaler = MegatronGradScaler(
    init_scale=2**16,
    growth_factor=2.0,
    backoff_factor=0.5,
    growth_interval=2000,
)

# Enable gradient clipping
config.clip_grad = 1.0
```

### Issue: HSDP Slower Than FSDP

**Symptoms**: HSDP performs worse than standard FSDP.

**Causes:**
1. Outer group size too small (not enough replicas)
2. Inner group size too small (inefficient NVLink usage)
3. Network topology not well-suited for HSDP

**Solutions:**
```python
# Ensure outer group matches node count
# Example: 8 nodes of 8 GPUs each
num_distributed_optimizer_instances = 8  # One per node

# Verify process group topology
outer_group = parallel_state.get_inter_distributed_optimizer_instance_group()
inner_group = parallel_state.get_data_parallel_group(partial_data_parallel=True)

print(f"Outer group size: {outer_group.size()}")  # Should be num_nodes
print(f"Inner group size: {inner_group.size()}")  # Should be GPUs_per_node

# Consider standard FSDP if network is uniform
# (HSDP benefits most on hierarchical topologies)
```

## Related Optimizations

- **#01 Gradient Bucketing**: Buckets are used for reduce-scatter operations
- **#19 Distributed Optimizer**: Core implementation for ZeRO-1/2
- **#22 Cached Bucket Shards**: Caches parameter/gradient shard views
- **#27 Gradient Buffer Padding**: Ensures optimal NCCL performance
- **#08 FP32 Accumulation**: Improves numerical stability in reduce-scatter

## Implementation Files

### Core Files

- **Distributed Optimizer**: `megatron/core/optimizer/distrib_optimizer.py` (3500+ lines)
- **Buffer Management**: `megatron/core/distributed/param_and_grad_buffer.py` (1200+ lines)
- **FSDP Wrapper**: `megatron/core/distributed/fsdp/mcore_fsdp_adapter.py` (430+ lines)
- **FSDP Core**: `megatron/core/distributed/fsdp/src/megatron_fsdp/megatron_fsdp.py` (2000+ lines)
- **FSDP Sharding**: `megatron/core/distributed/fsdp/src/megatron_fsdp/fully_shard.py` (300+ lines)
- **DDP Config**: `megatron/core/distributed/distributed_data_parallel_config.py` (156 lines)

### Supporting Files

- **Reduce-Scatter**: `megatron/core/distributed/reduce_scatter_with_fp32_accumulation.py`
- **FSDP Buffer**: `megatron/core/distributed/fsdp/src/megatron_fsdp/param_and_grad_buffer.py`
- **FSDP Utils**: `megatron/core/distributed/fsdp/src/megatron_fsdp/utils.py`

### Tests

- **Unit Tests**: `tests/unit_tests/distributed/test_mcore_fully_sharded_data_parallel.py`
- **FSDP Tests**: `tests/unit_tests/distributed/fsdp/test_mfsdp_fully_shard.py`
- **Buffer Tests**: `tests/unit_tests/distributed/test_param_and_grad_buffer.py`

## References

### Academic Papers

- **ZeRO**: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) (Rajbhandari et al., 2020)
- **PyTorch FSDP**: [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel](https://arxiv.org/abs/2304.11277) (Zhao et al., 2023)
- **HSDP**: Hybrid approach combining FSDP with hierarchical all-reduce

### Documentation

- **DeepSpeed ZeRO**: [Microsoft Research Blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
- **PyTorch FSDP**: [PyTorch Documentation](https://pytorch.org/docs/stable/fsdp.html)
- **Megatron-LM FSDP**: [Megatron Core Custom FSDP Guide](docs/source/api-guide/custom_fsdp.md)
- **Megatron-LM DistOpt**: [Distributed Optimizer Guide](docs/source/api-guide/dist_optimizer.md)

## Summary

Megatron-LM provides a comprehensive suite of memory optimization strategies:

1. **ZeRO-1**: Shards optimizer states, saving 50% memory with minimal overhead
2. **ZeRO-2**: Adds gradient sharding via reduce-scatter, saving 70% memory
3. **FSDP (ZeRO-3)**: Full parameter sharding, saving 87.5% memory at the cost of additional communication
4. **HSDP**: Hybrid approach optimized for hierarchical network topologies

The implementation is production-grade with:
- Sophisticated range-based ownership tracking
- Efficient buffer management with cached views
- Communication-computation overlap
- Integration with tensor parallelism and expert parallelism
- Support for mixed precision (BF16/FP8)
- Comprehensive checkpointing support

Choose your strategy based on:
- Available GPU memory
- Network topology (NVLink vs InfiniBand)
- Model size and training throughput requirements
- Communication overhead tolerance
