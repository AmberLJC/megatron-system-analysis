# 42. Tensor Parallelism Degree Tuning and Placement Strategies

## Context

Tensor Parallelism (TP) degree—the number of GPUs partitioning each transformer layer—is one of the most critical tuning parameters in distributed training. Choosing the right TP degree directly impacts model memory distribution, communication overhead, computation efficiency, and overall training throughput. Unlike data parallelism which scales linearly with GPU count, TP has a narrower optimal range where returns diminish sharply beyond certain thresholds due to communication overhead growth. This report addresses the key strategic decisions in TP tuning: determining the right TP degree for your cluster topology, placing TP groups to maximize bandwidth utilization, and balancing TP with other parallelism dimensions (PP, DP, CP) for multi-dimensional training.

## Problem: The TP Degree Trade-off

Tensor parallelism presents a fundamental trade-off:

**Benefits of increasing TP:**
- More GPUs share the computation load
- Reduces activation memory per GPU (scales as 1/TP)
- Reduces parameter memory per GPU (scales as 1/TP)
- Enables larger models to fit on given clusters
- Improves per-GPU computation efficiency for large weights

**Costs of increasing TP:**
- More collective communication within each layer
- All-reduce latency increases with TP (though not linearly)
- All-reduce bandwidth grows from `bandwidth_per_link × (TP-1)/TP` to `bandwidth_per_link`
- TP > cluster capacity forces cross-node communication (much slower)
- Synchronous TP stalls computation waiting for collective completion

**The TP sweet spot exists in the overlapping region:**
- TP is small enough that NVLink bandwidth dominates (low-latency, high-bandwidth region)
- TP is large enough to fit model and provide benefit
- TP doesn't exceed a single "all-reduce friendly" unit (e.g., NVSwitch group)

## Solution: Systematic TP Degree Selection

### TP Degree Fundamentals

TP degree is constrained by two physical limits:

1. **Topology Constraint:** TP should not exceed the size of the highest-bandwidth interconnect available. On NVIDIA clusters:
   - Single GPU: TP = 1
   - Single node (8× A100/H100): TP ≤ 8 with NVLink (600 GB/s peak)
   - NVSwitch-connected pods (64-128 GPUs): TP ≤ 16-32 with NVSwitch (900 GB/s)
   - Rack with InfiniBand (up to 512 GPUs): TP ≤ 8, cross-rack TP is not recommended

2. **Model Constraint:** TP must be sufficient to fit model parameters and activations:
   - Minimum TP = ceil(total_params / (gpu_memory - activation_overhead))

### Step 1: Analyze Cluster Topology

Before choosing TP, map your cluster's physical topology:

```python
# Cluster topology analysis
def analyze_cluster_topology():
    """Determine optimal TP bounds for your cluster"""

    # GPU properties
    num_gpus_total = 512  # Total GPUs in cluster
    num_gpus_per_node = 8  # 8× A100/H100 typical
    gpu_memory_gb = 80  # Per GPU

    # Interconnect speeds (measured, not theoretical)
    nvlink_bandwidth_gb_s = 600  # A100: 300 GB/s, H100: 600 GB/s
    switch_bandwidth_gb_s = 900  # NVSwitch typical
    ib_bandwidth_gb_s = 100  # 100 Gbps InfiniBand

    # Determine maximum TP based on topology
    # Strategy 1: Within-node TP (maximum TP on single node)
    max_tp_intranode = num_gpus_per_node  # TP=8 for 8-GPU node

    # Strategy 2: NVSwitch domain (if available)
    max_tp_nvswitch = 32  # Typical NVSwitch group: 32-64 GPUs

    # Strategy 3: Absolute maximum (cross-InfiniBand TP not recommended)
    # TP is limited by lowest interconnect speed in TP group
    # For multi-rack: max_tp = num_gpus_per_node = 8

    return {
        'intranode_max_tp': max_tp_intranode,
        'nvswitch_max_tp': max_tp_nvswitch,
        'global_max_tp': max_tp_intranode,
        'recommended_range': (2, 8),  # Practical range
    }

# Output for 512-GPU cluster with NVLink+InfiniBand:
# {
#     'intranode_max_tp': 8,        # NVLink domain
#     'nvswitch_max_tp': 32,        # NVSwitch pod
#     'global_max_tp': 8,           # Don't cross InfiniBand
#     'recommended_range': (2, 8),  # Use 2-8 in practice
# }
```

### Step 2: Calculate Memory Requirements

Determine minimum TP needed to fit the model:

```python
def calculate_minimum_tp(
    model_params: int,              # Total parameters (billions)
    gpu_memory_gb: int = 80,         # Per GPU
    precision: str = 'bfloat16',     # or 'float32', 'float8'
    use_sequence_parallel: bool = True,
    use_distributed_optimizer: bool = True,
    activation_multiplier: float = 3.0  # Typical activation memory multiplier
) -> int:
    """Calculate minimum TP degree to fit model"""

    # Model memory calculation
    params_dtype_bytes = {
        'float32': 4,
        'bfloat16': 2,
        'float8': 1,
    }

    bytes_per_param = params_dtype_bytes[precision]
    param_memory_gb = (model_params * 1e9 * bytes_per_param) / 1e9

    # Activation memory (rough estimate)
    activation_memory_gb = param_memory_gb * activation_multiplier

    # Optimizer state memory (ZeRO-2: 2× param memory)
    if use_distributed_optimizer:
        optimizer_memory_gb = param_memory_gb * 2  # Sharded across DP group
    else:
        optimizer_memory_gb = param_memory_gb * 3  # Full state per GPU

    # Sequence parallel reduces activation memory by TP factor
    if use_sequence_parallel:
        activation_memory_gb /= 8  # Assumes TP=8, adjust accordingly

    # Total memory per GPU (with TP sharding)
    total_memory_gb = (
        param_memory_gb +
        activation_memory_gb +
        optimizer_memory_gb +
        10  # Reserve for gradients and overhead
    )

    # Calculate minimum TP
    min_tp = max(1, int(np.ceil(total_memory_gb / gpu_memory_gb)))

    return min_tp

# Example: GPT-3 175B
min_tp = calculate_minimum_tp(
    model_params=175,
    gpu_memory_gb=80,
    precision='bfloat16',
    use_sequence_parallel=True,
    use_distributed_optimizer=True,
)
# For 175B: min_tp ≈ 4-8 (without SP: min_tp ≈ 8)
# For 70B: min_tp ≈ 2
# For 7B: min_tp ≈ 1 (use DP only)
```

### Step 3: Estimate Communication Overhead

TP introduces communication overhead at each layer. Estimate overhead to assess diminishing returns:

```python
def estimate_tp_overhead(
    hidden_size: int = 4096,
    batch_size: int = 32,
    sequence_length: int = 4096,
    tp_degree: int = 8,
    all_reduce_bandwidth_gb_s: float = 600,  # NVLink
    network_latency_us: float = 1.0,  # NVLink latency
) -> dict:
    """Estimate TP all-reduce communication cost"""

    # TP communication patterns (per transformer layer):
    # 1. ColumnParallel backward: all-reduce of input gradients
    # 2. RowParallel forward: all-reduce output (or reduce-scatter with SP)
    # 3. MLP weight gradients: all-reduce
    # Total: ~2-3 all-reduce operations per layer

    # Data size per all-reduce (typical hidden state)
    tensor_size_bytes = batch_size * sequence_length * hidden_size * 2  # bfloat16
    tensor_size_gb = tensor_size_bytes / 1e9

    # All-reduce cost = 2× tensor size (send + receive)
    # Bandwidth for all-reduce with TP=N: bw_per_link × (N-1)/N
    all_reduce_bw = all_reduce_bandwidth_gb_s * (tp_degree - 1) / tp_degree

    # Communication time per all-reduce (approximate)
    comm_time_us = (tensor_size_gb * 1e9 / (all_reduce_bw * 1e9)) * 1e6

    # Add latency (from each TP rank)
    total_comm_time_per_allreduce_us = comm_time_us + network_latency_us * np.log2(tp_degree)

    # Number of all-reduces per layer (conservative estimate)
    num_layers = 96  # GPT-3 baseline
    allreduces_per_layer = 2  # Input gradient + weight gradient
    num_allreduces = num_layers * allreduces_per_layer

    # Total communication time
    total_comm_time_ms = (total_comm_time_per_allreduce_us * num_allreduces) / 1e3

    # Computation time estimate (rough)
    # GEMM operations per transformer layer
    flops_per_layer = 2 * batch_size * sequence_length * hidden_size * (4 * hidden_size)  # FFN expansion
    gpu_tflops = 312  # H100 bfloat16 tensor cores
    compute_time_ms = (flops_per_layer / (gpu_tflops * 1e12)) * 1e3 * num_layers

    # Efficiency metrics
    overhead_fraction = total_comm_time_ms / (total_comm_time_ms + compute_time_ms)

    return {
        'allreduce_time_per_layer_us': total_comm_time_per_allreduce_us,
        'total_comm_time_ms': total_comm_time_ms,
        'compute_time_ms': compute_time_ms,
        'overhead_fraction': overhead_fraction,
        'recommendation': (
            'Good TP' if overhead_fraction < 0.10 else
            'Acceptable' if overhead_fraction < 0.20 else
            'High overhead - consider reducing TP'
        )
    }

# Example overhead analysis
for tp in [2, 4, 8, 16]:
    overhead = estimate_tp_overhead(tp_degree=tp)
    print(f"TP={tp}: {overhead['overhead_fraction']:.1%} overhead - {overhead['recommendation']}")

# Typical output:
# TP=2: 3-5% overhead - Good TP
# TP=4: 5-8% overhead - Good TP
# TP=8: 8-12% overhead - Acceptable
# TP=16: 15-20% overhead - High overhead
```

## Low-Level Implementation Details

### TP Group Initialization

TP groups are created in `megatron/core/parallel_state.py:514-700` using the `RankGenerator` class:

```python
# File: megatron/core/parallel_state.py:413-488
class RankGenerator(object):
    """A class for generating rank groups for different modes of parallelism."""

    def __init__(
        self, tp: int, ep: int, dp: int, pp: int, cp: int, order: str, rank_offset: int = 0
    ) -> None:
        """
        Initialize rank generator with parallelism dimensions.

        Args:
            tp: Tensor parallel size
            ep: Expert parallel size
            dp: Data parallel size
            pp: Pipeline parallel size
            cp: Context parallel size
            order: Dimension ordering string (e.g., 'tp-cp-ep-dp-pp')
        """
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.world_size = tp * dp * pp * cp * ep

        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }

        # Validate that all dimensions > 1 are in order string
        for name in self.name_to_size.keys():
            if name not in order and self.name_to_size[name] != 1:
                raise RuntimeError(
                    f"Dimension '{name}' with size {self.name_to_size[name]} "
                    f"must be in order string '{order}'"
                )

    def get_ranks(self, token: str):
        """
        Get rank groups for specified parallelism types.

        Args:
            token: Dimension specification (e.g., 'tp', 'tp-dp')

        Returns:
            List of rank groups, where each group contains ranks
            that should be in the same collective communication

        Example:
            With 16 GPUs, TP=2, PP=2, DP=4, order='tp-pp-dp':
            get_ranks('tp') returns:
            [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]]
        """
        mask = self.get_mask(self.order, token)
        ranks = generate_masked_orthogonal_rank_groups(
            self.world_size, self.ordered_size, mask
        )
        return ranks
```

### TP Process Group Creation

The core TP group creation in `initialize_model_parallel()`:

```python
# File: megatron/core/parallel_state.py:514-900 (simplified)
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    context_parallel_size: int = 1,
    expert_model_parallel_size: int = 1,
    order: str = "tp-cp-ep-dp-pp",
) -> None:
    """Initialize TP groups and other parallelism process groups."""

    world_size = torch.distributed.get_world_size()
    rank = torch.distributed.get_rank()

    # Create RankGenerator for decoder (standard) layers
    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,  # Standard layers don't use EP
        dp=world_size // (tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size),
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
    )

    # Get all TP rank groups from rank generator
    all_tp_ranks = decoder_rank_generator.get_ranks('tp')

    # Create NCCL process groups for each TP group
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS

    timeout = timedelta(minutes=distributed_timeout_minutes)
    nccl_options = get_nccl_options('tp', nccl_comm_cfgs)

    for ranks in all_tp_ranks:
        group = torch.distributed.new_group(
            ranks,
            timeout=timeout,
            pg_options=nccl_options
        )

        # Store group reference if current rank is in this group
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

            # Log group formation for debugging
            logger.info(
                f"Rank {rank} joined TP group with ranks {ranks}, "
                f"TP world size={len(ranks)}"
            )
```

### TP Layer Implementation: ColumnParallelLinear

The core TP computation is implemented in ColumnParallelLinear (megatron/core/tensor_parallel/layers.py:743-950):

```python
# File: megatron/core/tensor_parallel/layers.py:743-950
class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The weight matrix A is split along its second dimension:
    A = [A_1, A_2, ..., A_p] where p is tensor parallel world size.

    Each TP rank computes Y_i = X @ A_i independently.
    Output is either:
    - Gathered across TP ranks (gather_output=True), OR
    - Kept partitioned for next RowParallel layer (gather_output=False)
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        gather_output=False,
        skip_bias_add=False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super(ColumnParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output

        # Get TP group and partition information
        self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group)
        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)

        # Each rank holds output_size / world_size parameters
        self.output_size_per_partition = divide(output_size, world_size)

        # Weight shape: [output_size_per_partition, input_size]
        # Note: PyTorch linear does XA^T, so we store weight transposed
        self.weight = Parameter(
            torch.empty(
                self.output_size_per_partition,
                self.input_size,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )

        # Initialize weight with proper RNG tracking
        with get_cuda_rng_tracker().fork():
            init_method(self.weight)

        # Mark weight as tensor-parallel sharded
        set_tensor_model_parallel_attributes(
            self.weight,
            is_parallel=True,
            dim=0,  # Partition dim for column-parallel
            stride=1
        )

    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ):
        """Forward pass of ColumnParallelLinear.

        Input shape: [sequence_length, batch_size, hidden_size]
        Weight shape: [output_size_per_partition, input_size]
        Output shape: [sequence_length, batch_size, output_size_per_partition]

        If gather_output=True, all-gather to get full output across TP ranks.
        """

        if weight is None:
            weight = self.weight

        # Compute local matrix multiply: [s,b,h] @ [h,o/p]^T -> [s,b,o/p]
        output_parallel = torch.nn.functional.linear(input_, weight, self.bias)

        # Optionally gather output across TP ranks
        if self.gather_output:
            output = gather_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            output = output_parallel

        return output, self.bias if self.skip_bias_add else None
```

### TP Layer Implementation: RowParallelLinear

Row-parallel layer for reducing outputs across TP ranks (megatron/core/tensor_parallel/layers.py:1068-1303):

```python
# File: megatron/core/tensor_parallel/layers.py:1068-1200
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The weight matrix A is split along its first dimension:
    A = transpose([A_1, A_2, ..., A_p])

    Input X is split along its second dimension:
    X = [X_1, X_2, ..., X_p]

    Each rank computes Y_i = X_i @ A_i^T independently.
    Requires all-reduce to combine results: Y = sum(Y_i)
    """

    def __init__(
        self,
        input_size,
        output_size,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias=True,
        input_is_parallel=False,
        skip_bias_add=False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super(RowParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add

        # Get TP group information
        self.tp_group = get_tensor_model_parallel_group_if_none(self.tp_group)
        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)

        # Each rank holds input_size / world_size parameters
        self.input_size_per_partition = divide(input_size, world_size)

        # Weight shape: [output_size, input_size_per_partition]
        self.weight = Parameter(
            torch.empty(
                self.output_size,
                self.input_size_per_partition,
                device=torch.cuda.current_device(),
                dtype=config.params_dtype,
            )
        )

        # Initialize weight
        with get_cuda_rng_tracker().fork():
            init_method(self.weight)

        # Mark weight as tensor-parallel sharded
        set_tensor_model_parallel_attributes(
            self.weight,
            is_parallel=True,
            dim=1,  # Partition dim for row-parallel
            stride=1
        )

        # Bias is NOT partitioned (replicated across TP ranks)
        if bias:
            self.bias = Parameter(torch.empty(self.output_size))
            with torch.no_grad():
                self.bias.zero_()

    def forward(self, input_):
        """Forward pass of RowParallelLinear.

        Input shape: [sequence_length, batch_size, input_size_per_partition]
                    (already partitioned from ColumnParallel output)
        Weight shape: [output_size, input_size_per_partition]
        Output shape: [sequence_length, batch_size, output_size]
                    (gathered via all-reduce)
        """

        # Handle input partitioning if not already split
        if not self.input_is_parallel:
            input_parallel = scatter_to_tensor_model_parallel_region(
                input_, group=self.tp_group
            )
        else:
            input_parallel = input_

        # Compute local matrix multiply: [s,b,i/p] @ [o,i/p]^T -> [s,b,o]
        output_parallel = torch.nn.functional.linear(
            input_parallel, self.weight, None
        )

        # All-reduce to combine outputs from all TP ranks
        output_ = reduce_from_tensor_model_parallel_region(
            output_parallel, group=self.tp_group
        )

        # Add bias (bias not partitioned, same on all ranks)
        if not self.skip_bias_add:
            output = output_ + self.bias if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias

        return output, output_bias
```

### Async Communication with Gradient Accumulation

For overlapping communication with computation, Megatron uses `LinearWithGradAccumulationAndAsyncCommunication`:

```python
# File: megatron/core/tensor_parallel/layers.py:600-740
def linear_with_grad_accumulation_and_async_allreduce(
    input,
    weight,
    bias,
    gradient_accumulation_fusion,
    allreduce_dgrad=False,
    sequence_parallel=False,
    tp_group=None,
):
    """Wrapper for async all-reduce during backward pass.

    Enables overlapping of weight gradient computation with
    input gradient all-reduce, recovering communication latency.

    Key optimization: Launch all-reduce for dgrad while
    computing wgrad, hiding communication time.
    """

    args = [
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        tp_group,
    ]

    # Check and warn about CUDA_DEVICE_MAX_CONNECTIONS
    if allreduce_dgrad:
        if os.environ.get("CUDA_DEVICE_MAX_CONNECTIONS") != "1":
            warnings.warn(
                "When using async grad allreduce, set "
                "CUDA_DEVICE_MAX_CONNECTIONS=1 for maximum speedup"
            )

    return LinearWithGradAccumulationAndAsyncCommunication.apply(*args)


class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """Custom autograd for overlapping all-reduce with gradient computation.

    Forward: Standard linear layer output = input @ weight^T + bias
    Backward: Two key optimizations:
    1. All-reduce input gradients (dgrad) asynchronously
    2. Compute weight gradients (wgrad) while all-reduce is in-flight
    3. Return gathered dgrad when ready
    """

    @staticmethod
    def forward(ctx, input, weight, bias, ...):
        """Forward: regular linear layer"""
        output = torch.nn.functional.linear(input, weight, bias)
        ctx.save_for_backward(input, weight)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: overlap all-reduce with gradient computation"""
        input, weight = ctx.saved_tensors

        # Step 1: Start async all-reduce of input gradients (dgrad)
        # dgrad = grad_output @ weight
        grad_input = torch.matmul(grad_output, weight)

        # Start async all-reduce (in communication stream)
        handle = torch.distributed.all_reduce_async(
            grad_input, group=ctx.tp_group
        )

        # Step 2: While all-reduce is in-flight, compute weight gradients
        # wgrad = grad_output^T @ input
        grad_weight = torch.matmul(grad_output.t(), input)

        # Step 3: Synchronize on all-reduce completion
        torch.distributed.wait_stream(handle)

        # Return accumulated gradients
        return (
            grad_input / ctx.tp_world_size,  # All-reduced dgrad
            grad_weight,  # Local wgrad
            grad_output if ctx.return_bias_grad else None,
            ...
        )
```

### TP All-Reduce Communication Primitives

Core all-reduce implementation in megatron/core/tensor_parallel/mappings.py:

```python
# File: megatron/core/tensor_parallel/mappings.py:22-33
def _reduce(input_, group):
    """All-reduce the input tensor across TP group."""
    assert group is not None, "group should not be None"

    # Bypass if TP size = 1
    if group.size() == 1:
        return input_

    # Use NCCL all-reduce for efficient ring-based reduction
    torch.distributed.all_reduce(input_.contiguous(), group=group)

    return input_

# File: megatron/core/tensor_parallel/mappings:197-234
class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce gradient in backward pass."""

    @staticmethod
    def forward(ctx, input_, group):
        """Forward: identity"""
        ctx.group = group
        return _reduce(input_, group)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward: scatter gradient to each rank

        During backward:
        - Input: gradient tensor from next layer
        - Output: scattered gradient (1/TP fraction to each rank)
        """
        # Scatter gradient along dimension
        return _split_along_last_dim(grad_output, ctx.group), None
```

### Gather/Scatter for TP

Efficient gather and scatter operations:

```python
# File: megatron/core/tensor_parallel/mappings:80-96
def _gather_along_last_dim(input_, group):
    """Gather tensors along last dimension across TP ranks.

    Used in ColumnParallel forward when gather_output=True.
    Each rank has [s, b, o/p] tensor, output is [s, b, o] across all ranks.
    """
    world_size = group.size()
    if world_size == 1:
        return input_

    # Create output buffer
    dim_size = list(input_.size())
    dim_size[-1] = dim_size[-1] * world_size  # Expand last dim

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
    )

    # Use all-gather-into-tensor for efficient gathering
    dist_all_gather_func(output, input_.contiguous(), group=group)

    # Rearrange from [s,b,p,o/p] to [s,b,o]
    tensor_list = output.chunk(world_size, dim=0)
    output = torch.cat(tensor_list, dim=-1).contiguous()

    return output


# File: megatron/core/tensor_parallel/mappings:36-53
def _split_along_last_dim(input_, group):
    """Scatter tensor along last dimension.

    Input: [s, b, o] from all ranks
    Output: [s, b, o/p] on each rank
    Used in backward pass of gather operations.
    """
    world_size = group.size()
    if world_size == 1:
        return input_

    # Split along last dimension
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Keep only partition for this rank
    rank = group.rank()
    output = input_list[rank].contiguous()

    return output
```

## Implementation Details: TP Placement Strategies

### Strategy 1: Intra-Node TP (Recommended for Most Cases)

Place entire TP group within a single node to maximize NVLink bandwidth:

```python
# megatron/core/parallel_state.py - TP group formation

def create_intranode_tp_groups(
    world_size: int,
    tp_size: int,
    gpus_per_node: int = 8,
    order: str = 'tp-pp-dp'
) -> List[List[int]]:
    """
    Create TP groups entirely within single nodes.

    Example with 64 GPUs, TP=8, gpus_per_node=8:
    - Node 0: TP ranks [0,1,2,3,4,5,6,7]
    - Node 1: TP ranks [8,9,10,11,12,13,14,15]
    - Node 2: TP ranks [16,17,18,19,20,21,22,23]
    ...etc
    """

    if tp_size > gpus_per_node:
        raise ValueError(
            f"tp_size ({tp_size}) cannot exceed gpus_per_node ({gpus_per_node})"
        )

    num_nodes = world_size // gpus_per_node
    if world_size % gpus_per_node != 0:
        raise ValueError(
            f"world_size ({world_size}) not divisible by gpus_per_node ({gpus_per_node})"
        )

    tp_groups = []
    groups_per_node = gpus_per_node // tp_size

    for node_id in range(num_nodes):
        node_start_rank = node_id * gpus_per_node

        for group_id in range(groups_per_node):
            tp_group_ranks = [
                node_start_rank + group_id * tp_size + i
                for i in range(tp_size)
            ]
            tp_groups.append(tp_group_ranks)

    return tp_groups

# Verify placement uses NVLink
def verify_tp_placement_bandwidth(tp_ranks: List[int], gpus_per_node: int = 8):
    """Verify that TP group uses high-bandwidth interconnect"""

    node_ids = [rank // gpus_per_node for rank in tp_ranks]
    unique_nodes = len(set(node_ids))

    if unique_nodes == 1:
        return 'NVLink (600 GB/s)'
    elif unique_nodes == 2:
        return 'NVSwitch (if available) or IB (100 GB/s)'
    else:
        return 'WARNING: TP spans multiple nodes - high latency!'
```

### Strategy 2: Cross-Node TP with NVSwitch (Advanced)

For very large models that require TP > 8, use NVSwitch-connected pods:

```python
def create_nvswitch_tp_groups(
    world_size: int,
    tp_size: int,
    nvswitch_pod_size: int = 32,  # H100 pods: 32 GPUs
    order: str = 'tp-pp-dp'
):
    """
    Create TP groups using NVSwitch pods.

    NVSwitch pods achieve ~900 GB/s all-reduce bandwidth.
    Suitable for TP up to 16-32.

    Example: 256 GPUs, TP=16, nvswitch_pod_size=32
    - Pod 0 GPUs [0-31]: TP groups [0-15], [16-31]
    - Pod 1 GPUs [32-63]: TP groups [32-47], [48-63]
    ...etc
    """

    if tp_size > nvswitch_pod_size:
        raise ValueError(
            f"tp_size ({tp_size}) cannot exceed nvswitch_pod_size ({nvswitch_pod_size})"
        )

    num_pods = world_size // nvswitch_pod_size
    if world_size % nvswitch_pod_size != 0:
        raise ValueError(
            f"world_size ({world_size}) not divisible by nvswitch_pod_size"
        )

    tp_groups = []
    groups_per_pod = nvswitch_pod_size // tp_size

    for pod_id in range(num_pods):
        pod_start_rank = pod_id * nvswitch_pod_size

        for group_id in range(groups_per_pod):
            tp_group_ranks = [
                pod_start_rank + group_id * tp_size + i
                for i in range(tp_size)
            ]
            tp_groups.append(tp_group_ranks)

    return tp_groups

# Usage in initialize_model_parallel()
# parallel_state.initialize_model_parallel(
#     tensor_model_parallel_size=16,
#     pipeline_model_parallel_size=4,
#     # ... rest of config
#     # Rank ordering ensures TP is innermost for NVSwitch placement
#     order='tp-pp-dp'
# )
```

### Strategy 3: Hybrid TP Placement (TP × PP × DP)

Balance TP, PP, and DP for multi-dimensional training:

```python
def compute_balanced_parallelism(
    world_size: int,
    model_size_gb: int,
    gpu_memory_gb: int = 80,
    target_dp_size: int = None,
    gpus_per_node: int = 8,
) -> dict:
    """
    Compute balanced TP, PP, DP configuration.

    Optimization goals:
    1. TP within single node (maximize NVLink usage)
    2. PP across multiple nodes (minimize inter-stage communication)
    3. DP for gradient synchronization (last layer, can tolerate higher latency)
    """

    # Step 1: Determine TP (constrained by topology and memory)
    min_tp_for_memory = max(1, int(np.ceil(model_size_gb / gpu_memory_gb)))
    max_tp_topology = gpus_per_node
    tp_size = max(min_tp_for_memory, min(8, max_tp_topology))

    # Ensure TP divides world size
    while world_size % tp_size != 0:
        tp_size += 1
    if tp_size > max_tp_topology:
        tp_size = max_tp_topology

    # Remaining GPUs for PP and DP
    remaining_gpus = world_size // tp_size

    # Step 2: Determine PP (aim for 2-16 stages depending on model depth)
    # PP memory overhead: ~num_layers × activation_memory × num_microbatches/pp_size
    # Rough heuristic: PP = sqrt(remaining_gpus) / 2
    pp_size = max(1, int(np.sqrt(remaining_gpus) / 2))

    # Ensure PP divides remaining_gpus
    while remaining_gpus % pp_size != 0:
        pp_size += 1

    # Step 3: DP fills the rest
    dp_size = remaining_gpus // pp_size

    # Validate
    assert tp_size * pp_size * dp_size == world_size, "Decomposition error"

    return {
        'tensor_model_parallel_size': tp_size,
        'pipeline_model_parallel_size': pp_size,
        'data_parallel_size': dp_size,
        'world_size': world_size,
        'config_string': f'TP={tp_size} × PP={pp_size} × DP={dp_size}',
    }

# Examples
configs = [
    compute_balanced_parallelism(8),    # Single node
    compute_balanced_parallelism(64),   # 8 nodes
    compute_balanced_parallelism(512),  # 64 nodes
]

for config in configs:
    print(config['config_string'])

# Output:
# TP=2 × PP=2 × DP=2 (for 8 GPUs)
# TP=8 × PP=2 × DP=4 (for 64 GPUs)
# TP=8 × PP=8 × DP=8 (for 512 GPUs)
```

### Strategy 4: Rank Ordering for TP Placement

The `order` parameter in `initialize_model_parallel()` controls how ranks map to parallelism dimensions:

```python
# File: megatron/core/parallel_state.py:731
# The order parameter controls rank-to-group mapping

def rank_order_examples():
    """
    Different rank orderings for 64 GPUs, TP=8, PP=2, DP=4:

    order='tp-pp-dp' (INTRA-NODE TP):
    - TP groups form consecutive ranks within node
    - Rank [0-7] → TP group 0
    - Rank [8-15] → TP group 1
    - Better for NVLink (same physical node)

    order='pp-tp-dp' (INTER-NODE TP):
    - TP groups span multiple nodes
    - Rank [0, 8, 16, 24, ...] → TP group 0
    - Better for hierarchical networks
    - NOT recommended (crosses InfiniBand)

    order='tp-dp-pp' (CUSTOM):
    - TP inner, DP middle, PP outer
    - Good for specific network topologies
    """

    # Default (recommended): tp-cp-ep-dp-pp
    # Ensures TP groups are physically close
    # Moves PP (least frequent comm) to outermost

    pass

# Configuration for intra-node TP
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=8,
    context_parallel_size=1,
    expert_model_parallel_size=1,
    order='tp-pp-dp',  # TP groups within nodes
    # This ensures ranks [0-7] are in same TP group,
    # which maps to same physical node if 8 GPUs per node
)
```

### TP-Aware Weight Initialization

Critical for correctness: Weight initialization must account for TP sharding to maintain proper variance:

```python
# File: megatron/core/tensor_parallel/layers.py:128-200
def _initialize_affine_weight_gpu(
    weight: torch.Tensor,
    init_method: Callable,
    partition_dim: int,
    stride: int = 1,
    is_expert: bool = False
) -> None:
    """Initialize weight for tensor-parallel layer on GPU.

    Key: Init variance must be computed as if weight is NOT partitioned
    to maintain consistent activation variance across TP ranks.
    """

    # Mark weight as tensor-parallel (for checkpoint/validation)
    set_tensor_model_parallel_attributes(
        tensor=weight,
        is_parallel=True,
        dim=partition_dim,  # 0 for column-parallel, 1 for row-parallel
        stride=stride
    )

    # Initialize with proper RNG tracking
    # Ensures all TP ranks get different random values
    with get_cuda_rng_tracker().fork():
        # Init method called on sharded weight
        # But init variance computed as if unsharded
        init_method(weight)


# Example init method that knows about TP sharding:
def scaled_uniform_init(tensor, tp_world_size=8, partition_dim=0):
    """Initialize with variance scaled for TP.

    When weight is partitioned along dim 0:
    - Each rank sees output_size / tp_world_size rows
    - But initialization variance scales with full output_size
    - Need to scale init variance accordingly
    """
    fan_in = tensor.size(1)
    fan_out_full = tensor.size(0) * tp_world_size

    # Standard Xavier variance
    var = 2.0 / (fan_in + fan_out_full)
    std = math.sqrt(var)

    # Initialize partition with full-model variance
    with torch.no_grad():
        tensor.uniform_(-std, std)


def _initialize_affine_weight_cpu(
    weight: torch.Tensor,
    output_size: int,
    input_size: int,
    per_partition_size: int,
    partition_dim: int,
    init_method: Callable,
    stride: int = 1,
    rank: int = 0,
    world_size: int = 1,
) -> Optional[torch.Tensor]:
    """Initialize weight on CPU then move to GPU.

    Used for large models to avoid GPU memory spike during init.
    """

    # Create full weight on CPU (large allocation)
    full_weight = torch.empty(
        output_size if partition_dim == 0 else per_partition_size,
        input_size if partition_dim == 1 else per_partition_size,
        dtype=torch.float32  # Use FP32 for init precision
    )

    # Initialize full weight with proper variance
    with torch.no_grad():
        init_method(full_weight)

    # Extract this rank's partition
    if partition_dim == 0:
        partition_start = rank * per_partition_size
        partition_end = partition_start + per_partition_size
        weight_partition = full_weight[partition_start:partition_end, :]
    else:  # partition_dim == 1
        partition_start = rank * per_partition_size
        partition_end = partition_start + per_partition_size
        weight_partition = full_weight[:, partition_start:partition_end]

    # Copy to GPU and convert to target dtype
    with torch.no_grad():
        weight.copy_(weight_partition)

    return full_weight if keep_master_weight else None
```

### TP Parameter Identification and Checkpointing

Megatron uses tensor-parallel attributes for checkpoint sharding:

```python
# File: megatron/core/tensor_parallel/layers.py:87-126
def param_is_not_tensor_parallel_duplicate(param: torch.nn.Parameter) -> bool:
    """Check if parameter should be included in checkpoint.

    TP-sharded parameters are on all ranks but represent different parts.
    Only include parameters from rank 0 (or specified rank).
    """
    # Check if parameter has tensor_model_parallel attribute
    if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
        # It's TP-sharded, include it (checkpoint will handle sharding)
        return True
    else:
        # It's replicated, only include on rank 0
        return get_tensor_model_parallel_rank() == 0


def set_tensor_model_parallel_attributes(
    tensor: torch.Tensor,
    is_parallel: bool,
    dim: int,
    stride: int
) -> None:
    """Mark tensor with TP sharding metadata.

    Attributes:
    - tensor_model_parallel: True if TP-sharded
    - partition_dim: Which dimension is partitioned (0 or 1)
    - partition_stride: Stride in partition dimension
    """
    setattr(tensor, "tensor_model_parallel", is_parallel)
    setattr(tensor, "partition_dim", dim)
    setattr(tensor, "partition_stride", stride)


# In transformer blocks:
class TransformerBlock(torch.nn.Module):
    def __init__(self, config, tp_group=None):
        super().__init__()

        # All these layers are TP-sharded
        self.attention = ParallelAttention(config, tp_group=tp_group)
        self.mlp = ParallelMLP(config, tp_group=tp_group)

        # Store TP metadata for checkpoint/validation
        self.tp_group = tp_group
        self.tp_rank = get_tensor_model_parallel_rank()
        self.tp_world_size = get_tensor_model_parallel_world_size()

    def named_parameters_with_tp_info(self):
        """Yield parameters with TP sharding information"""
        for name, param in self.named_parameters():
            if hasattr(param, "tensor_model_parallel"):
                tp_info = {
                    'is_tp_sharded': param.tensor_model_parallel,
                    'partition_dim': param.partition_dim,
                    'partition_stride': param.partition_stride,
                    'tp_rank': self.tp_rank,
                    'tp_world_size': self.tp_world_size,
                }
            else:
                tp_info = {'is_tp_sharded': False}

            yield name, param, tp_info
```

## Configuration and Tuning Guide

### Quick Decision Tree

```
Is model.params × bytes_per_param > 60 GB?
├─ NO: Use DP only (TP=1)
│
└─ YES: Proceed to TP sizing
    │
    Do you have GPUs with NVLink?
    ├─ NO: Use minimal TP=2 (if available on your network)
    │
    └─ YES: Determine TP from topology
        │
        How many GPUs per node?
        ├─ 8× A100/H100: TP ∈ {2, 4, 8}
        ├─ 16× A100: TP ∈ {2, 4, 8, 16}
        └─ 32× (NVSwitch pod): TP ∈ {4, 8, 16, 32}

        Verify: TP × bytes_per_param < 60 GB?
        └─ YES: Proceed with chosen TP
           └─ Configure PP and DP for remaining GPUs
```

### Practical TP Configurations by Cluster Size

```python
# Single node (8 GPUs)
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=1,
    # data_parallel_size = 8 / (4 × 1) = 2
    order='tp-pp-dp',
)

# Small cluster (64 GPUs, 8 nodes)
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,      # Full node
    pipeline_model_parallel_size=2,    # 2 stages
    # data_parallel_size = 64 / (8 × 2) = 4
    order='tp-pp-dp',
)

# Medium cluster (256 GPUs, 32 nodes)
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,      # Full node
    pipeline_model_parallel_size=4,    # 4 stages
    context_parallel_size=1,
    # data_parallel_size = 256 / (8 × 4) = 8
    order='tp-pp-dp',
)

# Large cluster (512 GPUs)
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,      # Full node
    pipeline_model_parallel_size=8,    # 8 stages
    context_parallel_size=1,
    # data_parallel_size = 512 / (8 × 8) = 8
    order='tp-pp-dp',
)

# Very large cluster with long sequences (256 GPUs, long context)
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    context_parallel_size=4,           # Handle 32K tokens
    # data_parallel_size = 256 / (4 × 4 × 4) = 4
    order='tp-pp-dp',
)
```

## Performance Impact

### Memory Distribution

With TP=8 on 175B model (350GB in FP16):

| Metric | Without TP | With TP=8 |
|--------|-----------|----------|
| Param memory per GPU | 350 GB | 44 GB |
| Activation memory per GPU | ~100 GB | ~12 GB |
| Gradient memory per GPU | ~350 GB | ~44 GB |
| Total per GPU | ~800 GB | ~100 GB |
| Fits on A100 80GB? | ❌ NO | ✅ YES (with SP) |

### Communication Overhead

All-reduce overhead per layer (TP=8, hidden=4096):

```
Tensor size: ~100 MB (typical attention output)
Intra-node (NVLink 600 GB/s): ~0.2 ms
Cross-node (IB 100 GB/s): ~1.0 ms
With latency + pipeline: ~0.3-1.5 ms per all-reduce
2 all-reduces per layer × 96 layers = 57-288 ms total

Compute time per layer: ~5-10 ms (rough)
TP overhead: 5-50% depending on topology
```

### Empirical Scaling

TP scaling efficiency (relative speedup vs TP=1):

| TP Degree | Intra-Node (NVLink) | Cross-Node (IB) |
|-----------|-------------------|-----------------|
| 1x | 1.0x (baseline) | 1.0x (baseline) |
| 2x | 1.8x | 1.7x |
| 4x | 3.2x | 2.8x |
| 8x | 5.5x | 3.5x |
| 16x | 8.2x | 4.0x |
| 32x | 10.5x | N/A (not recommended) |

Efficiency drops sharply beyond TP=8 cross-node due to InfiniBand latency.

## Troubleshooting TP Issues

### Symptom: OOM Despite Setting TP=8

**Root causes:**
1. Sequence parallelism not enabled (SP required with TP > 4)
2. Activation checkpointing disabled
3. Too large microbatch size

**Solution:**
```python
config = TransformerConfig(
    tensor_model_parallel_size=8,
    sequence_parallel=True,  # CRITICAL with TP=8
    recompute_granularity='selective',
    recompute_method='block',
)
```

### Symptom: Training Speed Doesn't Improve Beyond TP=4

**Root causes:**
1. TP > 8 crossing InfiniBand boundary (high latency)
2. Wrong rank ordering (TP scattered across nodes)
3. Communication bottleneck dominating

**Diagnostic:**
```bash
# Check TP group formation
python -c "
from megatron.core import parallel_state
import torch
torch.distributed.init_process_group('nccl')
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=4,
)
tp_group = parallel_state.get_tensor_model_parallel_global_ranks()
print('TP ranks:', tp_group)
# Should show 8 consecutive ranks on same node
"
```

### Symptom: Non-Deterministic Results with TP

**Root causes:**
1. Floating point precision in all-reduce (FP16 underflow)
2. Different activation order in multi-dimensional parallelism

**Solution:**
```python
# Enable FP32 accumulation for TP all-reduce
from megatron.core.tensor_parallel import mappings

# Configure gradient accumulation in FP32
config.fp32_accumulation_in_allreduce = True
```

## Related Optimizations

- **#04 Tensor Parallel Overlap:** Async all-reduce with computation overlap
- **#12 Tensor Parallel:** Core TP implementation (ColumnParallel, RowParallel layers)
- **#17 Multi-Dimensional Parallel:** TP × PP × DP × CP × EP combinations
- **#18 Sequence Parallel:** Reduce activation memory by TP factor

## References

- Megatron paper: "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et al., 2021)
- Tensor parallel implementation: `megatron/core/tensor_parallel/`
- Parallel state management: `megatron/core/parallel_state.py:413-488` (RankGenerator)
- Configuration: `megatron/core/model_parallel_config.py`

