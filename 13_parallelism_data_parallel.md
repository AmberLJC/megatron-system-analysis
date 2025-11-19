# 13. Data Parallelism

## Overview

Data parallelism (DP) is the foundational scaling dimension in Megatron-LM, allowing training to scale beyond the limits of tensor and pipeline parallelism by replicating the entire model across multiple GPUs and sharding the training data. Each replica processes different batches of data in parallel, with gradients synchronized across replicas to maintain training consistency. Megatron-LM implements sophisticated data parallelism with two primary modes: standard Distributed Data Parallel (DDP) using all-reduce operations, and a distributed optimizer approach inspired by ZeRO that uses reduce-scatter operations for memory-efficient gradient synchronization.

The data parallel dimension is automatically calculated as the remaining GPUs after accounting for tensor and pipeline parallelism: `data_parallel_size = total_gpus / (tensor_parallel_size × pipeline_parallel_size)`. This makes data parallelism the final and most flexible scaling dimension, capable of utilizing hundreds or thousands of GPUs with near-linear scaling efficiency when properly configured.

## Core Architecture

### Process Group Initialization

Megatron-LM establishes data parallel process groups during initialization in `megatron/core/parallel_state.py`. The framework creates multiple types of data parallel groups to handle different communication patterns:

```python
# From parallel_state.py (lines ~880-900) - Data parallel group initialization
# Build the data-parallel groups.
global _DATA_PARALLEL_GROUP
global _DATA_PARALLEL_GLOBAL_RANKS
global _DATA_PARALLEL_GROUP_GLOO
assert _DATA_PARALLEL_GROUP is None, 'data parallel group is already initialized'

for ranks in decoder_rank_generator.get_ranks('dp', independent_ep=True):
    # Create NCCL group for GPU communication
    group = create_group(
        ranks,
        timeout=timeout,
        pg_options=get_nccl_options("dp", nccl_comm_cfgs),
        use_local_synchronization=use_local_synchronization,
        group_desc="DATA_PARALLEL_GROUP",
    )
    if rank in ranks:
        _DATA_PARALLEL_GROUP = group
        _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # Create Gloo group for CPU communication
    if ranks_with_config is not None:
        gloo_group = create_group(
            ranks,
            timeout=timeout,
            backend="gloo",
            use_local_synchronization=use_local_synchronization,
            group_desc="DATA_PARALLEL_GROUP_GLOO",
        )
        if rank in ranks:
            _DATA_PARALLEL_GROUP_GLOO = gloo_group
```

The initialization creates both NCCL groups (for high-performance GPU-to-GPU communication) and Gloo groups (for CPU-based operations like checkpoint synchronization). This dual-group approach enables optimal performance for different types of collective operations.

### DistributedDataParallel Wrapper

The `DistributedDataParallel` class in `megatron/core/distributed/distributed_data_parallel.py` wraps model modules to enable data parallel training. This implementation extends beyond PyTorch's standard DDP with sophisticated gradient bucketing, overlapped communication, and support for both standard all-reduce and distributed optimizer patterns:

```python
# From distributed_data_parallel.py:23-147 - Core DDP wrapper class
class DistributedDataParallel(_BaseDataParallel):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).
    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config, module=module)

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000, 1000000 * parallel_state.get_data_parallel_world_size()
            )
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        self.ddp_config = ddp_config

        # Setup process groups
        self.dp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=False, partial_data_parallel=False
        )
        self.dp_cp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=False
        )
        self.intra_dp_cp_group = parallel_state.get_data_parallel_group(
            with_context_parallel=True, partial_data_parallel=True
        )
        self.expt_dp_group = parallel_state.get_expert_data_parallel_group()
        self.intra_expt_dp_group = parallel_state.get_expert_data_parallel_group(
            partial_expert_data_parallel=True
        )

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size
        if isinstance(self.pp_group, list):
            pp_rank = self.pp_group[0].rank()
        else:
            pp_rank = self.pp_group.rank()
        if disable_bucketing or pp_rank > 0:
            self.bucket_size = None

        self.param_to_bucket_group = {}

        # Group parameters by their gradient type.
        param_to_name = {}
        dense_params = []
        expert_parallel_params = []
        self.params_with_grad = []
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            # Track params with grad to enable direct setting
            # of param.grad_added_to_main_grad
            self.params_with_grad.append(param)

            param.grad_added_to_main_grad = False
            param_to_name[param] = name

            if getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                expert_parallel_params.append(param)
```

The constructor performs intelligent bucket size calculation, increasing bucket sizes proportionally with data parallel world size to ensure NCCL communication remains bandwidth-bound rather than latency-bound at scale. Parameters are automatically categorized into dense and expert-parallel groups for optimized communication patterns. The critical insight is that bucketing is disabled for non-first pipeline stages since their gradient synchronization is not on the critical path.

### Gradient Buffer Management

Parameters and gradients are organized into contiguous buffers with sophisticated bucketing strategies in `megatron/core/distributed/param_and_grad_buffer.py`:

```python
# From param_and_grad_buffer.py:510-690 - Buffer initialization
class _ParamAndGradBuffer:
    """
    Groups parameters and gradients into a contiguous buffer, and then breaks the buffer into
    buckets with roughly `bucket_size` parameters each.

    Args:
        ddp_config: DistributedDataParallel config object.
        param_dtype: Type of param tensor.
        grad_dtype: Type of grad tensor.
        params: List of parameters whose parameters and gradients are collated in the underlying
            tensor.
        data_parallel_group: Data-parallel process group.
        bucket_size: The rough size of each bucket in terms of number of parameters.
        param_to_name: Mapping from `torch.nn.Parameter` to name (for logging purposes).
        gradient_scaling_factor: This factor is utilized to scale gradients prior to their
            communication. Its application is twofold: it facilitates the averaging of gradients
            and the scaling of gradients in the context of the Mixture of Experts (MoE) model.
    """

    def __init__(
        self,
        ddp_config: DistributedDataParallelConfig,
        param_dtype: torch.dtype,
        grad_dtype: torch.dtype,
        params: List[torch.nn.Parameter],
        data_parallel_group: torch.distributed.ProcessGroup,
        bucket_size: int,
        param_to_name: Dict[torch.nn.Parameter, str],
        gradient_scaling_factor: float,
        param_indices: List[int],
        nccl_ub: bool,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        self.ddp_config = ddp_config
        self.params = params
        self.param_dtype = param_dtype
        self.grad_dtype = grad_dtype
        self.data_parallel_group = data_parallel_group
        self.data_parallel_world_size = self.data_parallel_group.size()
        self.gradient_scaling_factor = gradient_scaling_factor

        # Data structures to store underlying buckets and relevant indexing data.
        self.buckets = []
        self.param_to_bucket = {}  # Param -> bucket mapping.
        self.param_index_map = {}  # Param -> location in buffer mapping (used in dist. optimizer).

        def _pad_end_of_bucket_if_needed(bucket_end_index: int) -> int:
            """
            Pads end index of bucket if using distributed optimizer (to ensure uniform sharding).
            """
            if self.ddp_config.use_distributed_optimizer:
                # Workaround for TE bug causing cuBLAS to pick an incompatible algorithm.
                # This also helps cuBLAS pick more efficient algorithms for GEMMs.
                # We now ensure that all buckets start at a memory address that is 256-byte
                # aligned (128 values since params and grads use >= 16-bit precision).
                if self.ddp_config.pad_buckets_for_high_nccl_busbw:
                    # Make sure the bucket size is divisible by a large power of 2 (2^16) to
                    # ensure NCCL collectives have high bus bandwidth at large DP counts,
                    # since NCCL message size (which for ring algorithms is bucket_size /
                    # dp_size) apparently needs to be divisible by a power of 2 for high busbw.
                    bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128, 2**16)
                else:
                    bucket_size_divisor = math.lcm(self.data_parallel_world_size, 128)
                return _pad(bucket_end_index, bucket_size_divisor)
            return bucket_end_index

        for param in params[::-1]:
            # Iterate through parameters in reverse order to roughly follow backprop order.
            this_numel = param.data.nelement()
            param_start_index = _pad_start_of_param_if_needed(param_start_index)

            param_end_index = param_start_index + this_numel
            self.param_index_map[param] = (param_start_index, param_end_index, bucket_id)
            bucket_params.add(param)

            # If we have enough elements already or the current param is part of the shared
            # embedding layer and needs a separate bucket, form a new bucket.
            if (
                bucket_size is not None and (param_end_index - bucket_start_index) >= bucket_size
            ) or _does_param_require_new_bucket(param):
                bucket_end_index = _update_bucket_metadata(param_end_index)
                param_start_index = bucket_end_index
            else:
                param_start_index = param_end_index

        # Next, create underlying storage for buffer (with numel elements that includes
        # padding as necessary).
        self.numel = bucket_end_index
        self.numel_unpadded = sum(per_bucket_numel_unpadded)
        assert self.numel_unpadded <= self.numel
        if self.ddp_config.use_distributed_optimizer:
            assert self.numel % self.data_parallel_world_size == 0
```

The buffer management system creates contiguous memory regions for all parameters and gradients, reducing memory fragmentation and enabling efficient communication. Parameters are processed in reverse order to align with backward pass execution, allowing gradient synchronization to begin as soon as each bucket's gradients are ready. The padding logic ensures proper alignment for both cuBLAS (256-byte boundaries) and NCCL (power-of-2 divisibility for high bandwidth).

## Gradient Synchronization Strategies

### Standard DDP: All-Reduce Pattern

In standard DDP mode, gradient synchronization uses all-reduce operations where every rank receives the complete averaged gradients:

```python
# From param_and_grad_buffer.py:330-462 - Gradient sync implementation
class _ParamAndGradBucketGroup:
    """
    Put multiple buckets into a group so that their communications can be aggregated together.
    Provides functionality to register when params in the bucket group have grads ready to be
    synced; an asynchronous communication call is automatically launched when _all_ params in
    the bucket group have grads ready.
    """

    def start_grad_sync(self):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all buckets in the bucket group.

        When ddp_config.overlap_grad_reduce is set to True, dispatches an asynchronous
        communication call. When ddp_config.overlap_grad_reduce is set to False, makes
        synchronous call.
        """
        assert (
            self.grad_reduce_handle is None
        ), "Should not have multiple communication calls outstanding at once"

        # gradient_scaling_factor already takes into account whether we are computing
        # an average or sum in the data-parallel collective.
        for bucket in self.buckets:
            if bucket.gradient_scaling_factor != 1.0:
                bucket.grad_data *= bucket.gradient_scaling_factor

        # Decide reduce_op.
        reduce_op = torch.distributed.ReduceOp.SUM
        if self.ddp_config.average_in_collective:
            reduce_op = torch.distributed.ReduceOp.AVG

        # Use async communications only when overlap_grad_reduce is True.
        async_op = (
            self.ddp_config.overlap_grad_reduce
            and self.ddp_config.num_distributed_optimizer_instances == 1
        )

        if (
            self.ddp_config.num_distributed_optimizer_instances > 1
            and self.ddp_config.overlap_grad_reduce
        ):
            # Assign a communication stream if we have multiple DistOpt instances and we
            # need to overlap communication.
            stream_context = torch.cuda.stream(self.communication_stream)

            # The RS/AR communication stream needs to wait for the default stream
            # to complete its gradient computation before launching the next
            # gradient reduction collective.
            self.communication_stream.wait_stream(torch.cuda.default_stream())
        else:
            stream_context = nullcontext()

        if self.ddp_config.use_distributed_optimizer:
            communication_group = self.intra_distributed_optimizer_instance_group
        else:
            communication_group = self.data_parallel_group

        # Coalesce communication kernels across buckets in the bucket group.
        grad_reduce_handle = None
        with stream_context, _coalescing_manager(communication_group, async_ops=async_op) as cm:
            for idx, bucket in enumerate(self.buckets):
                if self.ddp_config.use_distributed_optimizer:
                    if self.cached_grad_buffer_shard_list[idx] is None:
                        self.cached_grad_buffer_shard_list[idx] = shard_buffer(
                            bucket.grad_data, self.intra_distributed_optimizer_instance_size
                        )
                    local_data_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]
                    grad_reduce_handle = dist_reduce_scatter_func(
                        local_data_view,
                        bucket.grad_data,
                        op=reduce_op,
                        group=communication_group,
                        async_op=async_op,
                    )
                else:
                    torch.distributed.all_reduce(
                        bucket.grad_data, op=reduce_op, group=communication_group, async_op=async_op
                    )

        # With multiple DistOpt instances, we need to all-reduce across instances.
        if (
            self.ddp_config.use_distributed_optimizer
            and self.ddp_config.num_distributed_optimizer_instances > 1
        ):
            assert self.inter_distributed_optimizer_instance_group is not None
            with (
                stream_context,
                _coalescing_manager(
                    self.inter_distributed_optimizer_instance_group, async_ops=async_op
                ) as cm,
            ):
                for idx, bucket in enumerate(self.buckets):
                    local_data_view = self.cached_grad_buffer_shard_list[idx][
                        self.intra_distributed_optimizer_instance_rank
                    ]

                    torch.distributed.all_reduce(
                        local_data_view,
                        op=reduce_op,
                        group=self.inter_distributed_optimizer_instance_group,
                        async_op=async_op,
                    )

        if async_op:
            self.grad_reduce_handle = cm
        else:
            self.grad_reduce_handle = None
```

The synchronization system intelligently selects between reduce operations based on configuration. The `average_in_collective` option determines whether averaging occurs during the reduction (ReduceOp.AVG) or via pre-scaling (ReduceOp.SUM with gradient_scaling_factor). Communication is coalesced across buckets when possible to reduce kernel launch overhead. The dual-stream design for multi-instance distributed optimizer enables perfect overlap by using dedicated communication streams.

### Distributed Optimizer: Reduce-Scatter Pattern

When using the distributed optimizer, Megatron-LM employs a reduce-scatter strategy inspired by ZeRO, where each rank maintains only a shard of the optimizer states:

```python
# From distrib_optimizer.py:107-234 - Distributed optimizer gradient partitioning
class DistributedOptimizer(MixedPrecisionOptimizer):
    """
    Distributed optimizer, for all data types (fp16, bf16, and fp32).
    Each DP rank 'owns' a contiguous region of the gradient buffer and is responsible
    for reducing the relevant subset of grads and updating the relevant subset of params.
    """

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
        Ownership in this sense means DP rank is responsible for reducing
        the relevant subset of grads, and updating the relevant subset of
        params.

        This conceptual partitioning of the grad buffer does NOT respect
        parameter boundaries, and as such it is assumed that each created
        range references a shard (or subset) of the full parameter. It is
        easiest to think of each DP rank as operating (i.e., reducing,
        gathering) purely on views into the grad buffer, for all model-to-
        main & main-to-model operations.

        This method creates four ranges:
        - The param's range within the entire grad buffer (i.e., world index).
        - The param's range within the relevant grad bucket's buffer.
        - The param's range within the DP rank's local view of the grad buffer.
        - The param's range within itself (i.e., its shard).
        """

        # Param range map.
        param_range_map = {}
        for param, param_world_indexes in param_world_index_map.items():

            # Param range.
            param_world_start, param_world_end, _ = param_world_indexes
            param_local_start = max(0, param_world_start - gbuf_world_range.start)
            param_local_end = min(gbuf_world_range.size, param_world_end - gbuf_world_range.start)

            # Add param, if within local gbuf range.
            if param_local_end > param_local_start:
                param_local_range = Range(param_local_start, param_local_end)
                param_world_range = param_local_range.normalize(
                    param_local_start + gbuf_world_range.start
                )
                param_world_range_in_bucket = Range(
                    param_world_range.start - bucket_offset, param_world_range.end - bucket_offset
                )
                sub_param_start = max(0, gbuf_world_range.start - param_world_start)
                sub_param_range = param_local_range.normalize(sub_param_start)
                param_range_map[param] = {
                    "gbuf_world": param_world_range,
                    "gbuf_world_in_bucket": param_world_range_in_bucket,
                    "gbuf_local": param_local_range,
                    "param": sub_param_range,
                }

        return param_range_map

    @classmethod
    def _build_model_gbuf_range(cls, param_and_grad_buffer: _ParamAndGradBuffer, bucket_index: int):
        """
        Build mapping between params and their grad buffers.

        This method does the initial setup for the method above. This setup
        includes determining the shard ranges into the param_and_grad_buffer
        for each data-parallel (DP) rank. Each DP rank keeps range info for
        all other DP ranks, for the purpose of creating args for
        reduce-scatter and all-gather.
        """

        data_parallel_rank = param_and_grad_buffer.data_parallel_group.rank()
        data_parallel_world_size = param_and_grad_buffer.data_parallel_group.size()

        bucket = param_and_grad_buffer.buckets[bucket_index]
        gbuf_size = bucket.grad_data.numel()
        assert (
            gbuf_size % data_parallel_world_size == 0
        ), f"Each bucket's buffer size should be divisible by {data_parallel_world_size}"
        max_gbuf_range_size = gbuf_size // data_parallel_world_size

        # All world ranges (i.e., across all data parallel ranks).
        gbuf_world_all_ranges = []
        for r in range(data_parallel_world_size):
            # Compute start of chunk in this bucket.
            gbuf_world_start = r * max_gbuf_range_size
            gbuf_world_end = min(gbuf_size, gbuf_world_start + max_gbuf_range_size)
            # Add bucket's offset in grad buffer.
            gbuf_world_range = Range(
                gbuf_world_start + bucket.offset, gbuf_world_end + bucket.offset
            )
            gbuf_world_all_ranges.append(gbuf_world_range)

        # Local DP's ranges.
        gbuf_world_range = gbuf_world_all_ranges[data_parallel_rank]

        # Get each param's ranges.
        param_range_map = cls._build_model_gbuf_param_range_map(
            param_and_grad_buffer.param_index_map, gbuf_world_range, bucket.offset
        )

        # Group into dict.
        data = {"param_map": param_range_map}

        return data
```

The distributed optimizer creates a sophisticated mapping system that partitions parameters and gradients across data parallel ranks. Unlike standard DDP where each rank holds complete copies of all gradients and optimizer states, this approach shards both gradients and optimizer states, significantly reducing memory consumption. The key innovation is that parameter boundaries are not respected—each rank owns a contiguous slice of the gradient buffer, and parameters may be split across multiple ranks. This maximizes load balancing but requires careful index tracking through the four-level range hierarchy.

## Overlapped Communication

### Asynchronous Gradient Reduction

Megatron-LM achieves high efficiency by overlapping gradient communication with backward pass computation. This is implemented through backward hooks that trigger communication as soon as each bucket's gradients are ready:

```python
# From distributed_data_parallel.py:441-467 - Backward hook for async communication
def _make_backward_post_hook(self, param: torch.nn.Parameter):
    """
    Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
    ready (i.e., when all grads in a bucket have been computed in all microbatches
    in a batch).
    """

    def hook(*unused):
        if is_graph_capturing():
            return

        if param in self.param_to_bucket_group:
            assert param.requires_grad
            if self.ddp_config.overlap_grad_reduce:
                assert (
                    param.grad is not None
                ), 'param.grad being None is not safe when overlap_grad_reduce is True'
            if param.grad is not None and (
                not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
            ):
                param.main_grad.add_(param.grad.data)
            param.grad = None

            if self.ddp_config.overlap_grad_reduce:
                self.param_to_bucket_group[param].register_grad_ready(param)

    return hook

# Register hooks for all parameters
for param in self.module.parameters():
    if param.requires_grad:
        # Expand so we get access to grad_fn.
        param_tmp = param.expand_as(param)
        # Get the gradient accumulator function.
        grad_acc = param_tmp.grad_fn.next_functions[0][0]
        grad_acc.register_hook(self._make_backward_post_hook(param))
        self.grad_accs.append(grad_acc)
```

Each parameter's backward hook adds gradients to the main gradient buffer and registers the parameter as ready. The hook accumulates parameter gradients into `param.main_grad`, then clears `param.grad` to save memory. When all parameters in a bucket group are ready, communication is automatically dispatched without waiting for the entire backward pass to complete.

### Bucket-based Communication Triggering

The bucket group system tracks gradient readiness and automatically triggers communication:

```python
# From param_and_grad_buffer.py:490-507 - Automatic communication dispatch
def register_grad_ready(self, param: torch.nn.Parameter):
    """
    Registers grads for the passed-in param to be "ready" for grad sync.

    When the number of microbatches is greater than 1, we only want to register
    grads as ready when processing the last microbatch and ddp_config.overlap_grad_reduce
    is True.
    """
    assert (
        self.ddp_config.overlap_grad_reduce
    ), "register_grad_ready() should only be called when overlap_grad_reduce is True"
    if self.is_last_microbatch:
        assert param in self.param_to_bucket, "Param is not in the bucket group"
        assert param not in self.params_with_grad, "Cannot set grad twice"
        self.params_with_grad.add(param)
        # If all params in bucket group have grads available, issue communication call.
        if len(self.params_with_grad) == len(self.params):
            self.start_grad_sync()
```

This design enables near-perfect overlap of communication with computation. As the backward pass proceeds from the end of the model to the beginning (following PyTorch's reverse-mode autodiff), buckets at the end are ready for communication while earlier layers are still computing gradients. The `is_last_microbatch` flag ensures that with gradient accumulation, communication is only triggered after all microbatches have contributed their gradients.

## Advanced Features

### Gradient Scaling and Averaging

Megatron-LM supports flexible gradient scaling strategies to handle both standard averaging and expert parallelism in MoE models:

```python
# From distributed_data_parallel.py:275-323 - Gradient scaling configuration
if config.calculate_per_token_loss:
    gradient_scaling_factor = 1.0
    expert_gradient_scaling_factor = 1.0
else:
    # The goal is to scale reduced gradients by 1/dp_size.
    # This can be achieved in two ways:
    #
    # Case 1: average_in_collective=True
    # - Non-expert parameters:
    #   1. No pre-scaling (gradient_scaling_factor=1.0)
    #   2. Do average reduction over dp group (equals to sum then divide by dp_size)
    #   3. Final result is scaled by 1/dp_size as desired
    #
    # - Expert parameters:
    #   1. Scale by edp_size/dp_size before reduction
    #   2. Do average reduction over edp group (equals to sum then divide by edp_size)
    #   3. Resulted scaling: (edp_size/dp_size) * (1/edp_size) = 1/dp_size as desired
    #   (edp_size = expert data parallel world size)
    #
    # Case 2: average_in_collective=False
    # - Both expert and non-expert parameters:
    #   1. Scale gradients by 1/dp_size before reduction
    #   2. Do sum reduction across data parallel ranks
    #   3. Final result is scaled by 1/dp_size as desired
    if self.ddp_config.average_in_collective:
        gradient_scaling_factor = 1.0
        expert_gradient_scaling_factor = self.expt_dp_group.size() / self.dp_cp_group.size()
    else:
        data_parallel_world_size = self.dp_cp_group.size()

        gradient_scaling_factor = 1.0 / data_parallel_world_size
        expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

# Allocate the param+grad buffers for dense params' grads.
self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
    dense_params, self.intra_dp_cp_group, gradient_scaling_factor=gradient_scaling_factor
)

# Allocate separate param+grad buffers for expert parallel params' grads.
self.expert_parallel_buffers, self.expert_parallel_bucket_groups = (
    _allocate_buffers_for_parameters(
        expert_parallel_params,
        self.intra_expt_dp_group,
        gradient_scaling_factor=expert_gradient_scaling_factor,
    )
)
```

This dual-scaling approach ensures correct gradient averaging across different parallel dimensions. For MoE models, expert parameters use a different data parallel group (expert DP group) which may have a different size than the standard DP group. The scaling factor compensates for this difference, ensuring that all parameters end up with gradients scaled by 1/dp_size regardless of which communication group they use.

### Parameter Broadcasting

At initialization, parameters must be synchronized across all data parallel replicas to ensure consistent starting points:

```python
# From distributed_data_parallel.py:569-585 - Parameter synchronization
def broadcast_params(self):
    """
    Syncs parameters across all DP ranks.
    """
    for param in self.module.parameters():
        is_expert_parallel = not getattr(param, 'allreduce', True)

        if is_expert_parallel:
            data_parallel_group = self.expt_dp_group
        else:
            data_parallel_group = self.dp_cp_group
        torch.distributed.broadcast(
            param.data,
            src=torch.distributed.get_global_rank(data_parallel_group, 0),
            group=data_parallel_group,
        )
```

This ensures that random initialization on rank 0 is replicated to all other ranks, or that loaded checkpoint parameters are consistently distributed across all replicas before training begins.

### No-Sync Context Manager

```python
# From distributed_data_parallel.py:469-480 - No-sync context
@contextmanager
def no_sync(self):
    """
    Context manager that turns off gradient synchronization.
    """
    for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
        bucket_group.is_last_microbatch = False
    try:
        yield
    finally:
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.is_last_microbatch = True
```

This context manager is critical for gradient accumulation and pipeline parallelism. It temporarily disables gradient synchronization by marking all microbatches as non-final, preventing communication until explicitly triggered. This allows pipeline parallelism to defer all gradient synchronization to the pipeline cooldown phase, maximizing overlap opportunities.

## Performance Characteristics

### Scaling Efficiency

Data parallelism achieves near-linear scaling efficiency through several optimizations:

1. **Gradient bucketing**: Amortizes communication overhead by coalescing many small messages into fewer large ones
2. **Overlapped communication**: Hides communication latency by executing it during backward computation
3. **Optimized bucket sizing**: Dynamically adjusts based on data parallel world size to maintain bandwidth-bound communication
4. **Bandwidth optimization**: Padding to 2^16 boundaries ensures NCCL ring algorithms operate at peak efficiency

Typical scaling efficiency: 85-95% at moderate scales (DP ≤ 64), maintaining 80%+ efficiency to DP = 512+. The slight degradation at larger scales comes from increased synchronization overhead and the difficulty of perfectly overlapping communication at the boundaries.

### Memory Considerations

Memory usage varies significantly between standard DDP and distributed optimizer modes:

**Standard DDP (all-reduce)**:
- Each rank: Full model + full optimizer states + full gradients
- Memory per rank: Model_size + Optimizer_states + Gradients

**Distributed Optimizer (reduce-scatter)**:
- Each rank: Full model + sharded optimizer states + sharded gradients
- Memory per rank: Model_size + (Optimizer_states + Gradients) / DP_size
- Memory savings: ~(Optimizer_states + Gradients) × (DP_size - 1) / DP_size

For a typical AdamW optimizer with fp32 master weights and states (2 moment estimates), distributed optimizer reduces memory by approximately 12 bytes per parameter × (DP_size - 1) / DP_size. At DP=8, this translates to an 87.5% reduction in optimizer memory.

### Communication Volume

**All-reduce per iteration**: 2 × Gradients_size × (DP_size - 1) / DP_size
**Reduce-scatter per iteration**: Gradients_size × (DP_size - 1) / DP_size

Additionally, distributed optimizer requires all-gather for parameters when using `overlap_param_gather`, adding parameter communication volume of Params_size × (DP_size - 1) / DP_size per iteration. However, this enables the memory savings and can be overlapped with the forward pass.

## Configuration and Usage

### Basic Configuration

```python
# Example configuration for 512 GPUs
total_gpus = 512
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 8
# Data parallel size automatically calculated
data_parallel_size = total_gpus // (8 * 8)  # = 8

# DDP configuration
ddp_config = DistributedDataParallelConfig(
    overlap_grad_reduce=True,              # Enable async communication
    use_distributed_optimizer=True,        # Use ZeRO-style reduce-scatter
    average_in_collective=True,            # Average during reduction
    bucket_size=None,                      # Auto-sized based on DP world size
    pad_buckets_for_high_nccl_busbw=True, # Ensure 2^16 alignment
)
```

### Performance Tuning Parameters

- **overlap_grad_reduce**: Enable to overlap communication with backward pass (recommended: True for almost all cases)
- **bucket_size**: Controls granularity of communication; default auto-sizing works well for most workloads
- **use_distributed_optimizer**: Enable for memory efficiency at large scale (trades communication volume for memory)
- **average_in_collective**: Use ReduceOp.AVG instead of pre-scaling (can improve numerical stability)
- **pad_buckets_for_high_nccl_busbw**: Pad to 2^16 boundaries for optimal NCCL performance at very large DP sizes (>= 64)
- **num_distributed_optimizer_instances**: Split DP domain hierarchically to match network topology

## Integration with Other Parallelism Dimensions

Data parallelism composes orthogonally with tensor and pipeline parallelism to enable 3D parallelism:

```python
# 3D parallelism example: GPT-175B on 512 GPUs
tensor_parallel_size = 8      # Split attention/MLP across 8 GPUs
pipeline_parallel_size = 8    # Split layers across 8 pipeline stages
data_parallel_size = 8        # 8 data-parallel replicas

# Total: 8 × 8 × 8 = 512 GPUs
# Each model replica uses: 8 (TP) × 8 (PP) = 64 GPUs
# Number of replicas: 8 (DP)

# Effective batch size
micro_batch_size = 4
global_batch_size = micro_batch_size * data_parallel_size  # 4 × 8 = 32
```

The composition allows flexible resource allocation: increase TP/PP to fit larger models, increase DP to process more data in parallel and improve training throughput.

## Conclusion

Data parallelism in Megatron-LM provides essential scalability for training at massive scale. Through sophisticated gradient bucketing, overlapped asynchronous communication, and optional distributed optimizer integration (ZeRO-2 style), it achieves excellent scaling efficiency of 85-95% while offering flexibility in memory-performance trade-offs. The implementation seamlessly handles complex scenarios including mixed precision training, expert parallelism in MoE models, and tight integration with tensor and pipeline parallelism. This makes data parallelism a robust and efficient foundation for multi-dimensional parallel training strategies capable of scaling to thousands of GPUs.
