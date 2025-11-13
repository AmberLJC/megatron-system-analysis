# 12. Tensor Parallelism

## Context

Tensor parallelism (TP) is a critical intra-layer model parallelism technique that enables training of large language models that exceed the memory capacity of a single GPU. Modern large models like GPT-3 (175B parameters) require approximately 350GB of memory in FP16 precision just for parameters alone, far exceeding the 80GB capacity of NVIDIA A100 GPUs. Tensor parallelism addresses this fundamental limitation by partitioning individual weight matrices across multiple GPUs, allowing each GPU to store and compute only a fraction of each layer while maintaining mathematical equivalence through collective communication.

Unlike pipeline parallelism which distributes entire layers across GPUs, tensor parallelism splits the computations within each transformer layer. This approach provides fine-grained parallelism with predictable memory distribution and allows flexible scaling based on model size and available GPU interconnect bandwidth. Megatron-LM implements sophisticated tensor parallel primitives that handle weight initialization, forward/backward passes, and gradient synchronization across distributed GPUs.

## Implementation Architecture

Megatron-LM's tensor parallelism implementation is built on two fundamental layer types that partition linear transformations along different dimensions: **ColumnParallelLinear** and **RowParallelLinear**. These layers are strategically placed in transformer blocks to minimize communication overhead while maintaining correct gradients during backpropagation.

### Column-Parallel Linear Layer

The `ColumnParallelLinear` class (megatron/core/tensor_parallel/layers.py:743-1067) implements linear layers where the output dimension is partitioned across GPUs. Given a linear transformation `Y = XA + b` where `A` has shape `[input_size, output_size]`, column parallelism splits `A` along the second dimension as `A = [A_1, A_2, ..., A_p]` where `p` is the tensor parallel world size.

Here's the actual implementation from the codebase:

```python
class ColumnParallelLinear(torch.nn.Module):
    """Linear layer with column parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along
    its second dimension as A = [A_1, ..., A_p].

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias
        gather_output: If true, call all-gather on output and make Y available
                       to all GPUs, otherwise, every GPU will have its output
                       which is Y_i = XA_i
        skip_bias_add: If True, do not add the bias term, instead return it
                       to be added by the caller. This enables performance
                       optimizations where bias can be fused with other
                       elementwise operations.
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
        stride=1,
        keep_master_weight_for_test=False,
        skip_bias_add=False,
        is_expert: bool = False,
        disable_grad_reduce: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super(ColumnParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        self.skip_bias_add = skip_bias_add
        self.is_expert = is_expert

        # Get tensor parallel group and calculate partition size
        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, is_expert=self.is_expert
        )
        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)

        # Divide output dimension across GPUs
        self.output_size_per_partition = divide(output_size, world_size)

        # Initialize weight - note transpose for torch.nn.functional.linear
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.output_size_per_partition,
                    0,  # partition_dim=0 for column parallel
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size_per_partition,
                    self.input_size,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=0,
                    stride=stride,
                    is_expert=self.is_expert,
                )
```

The forward pass handles input distribution and optional output gathering:

```python
    def forward(
        self,
        input_: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
        runtime_gather_output: Optional[bool] = None,
    ):
        """Forward of ColumnParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]
            weight (optional): weight tensor to use
            runtime_gather_output (bool): Gather output at runtime

        Returns:
            - output
            - bias
        """
        if weight is None:
            weight = self.weight

        bias = self.bias if not self.skip_bias_add else None

        # Handle input parallelization based on configuration
        if (
            self.allreduce_dgrad
            or self.sequence_parallel
            or self.explicit_expert_comm
            or self.disable_grad_reduce
        ):
            input_parallel = input_
        else:
            # Copy input to all tensor parallel ranks
            input_parallel = copy_to_tensor_model_parallel_region(
                input_, group=self.tp_group
            )

        # Perform local matrix multiplication
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=weight,
            bias=bias,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=self.allreduce_dgrad,
            sequence_parallel=self.sequence_parallel,
            tp_group=self.tp_group,
        )

        # Optionally gather output across tensor parallel ranks
        gather_out = self.gather_output
        if runtime_gather_output is not None:
            gather_out = runtime_gather_output

        if gather_out:
            output = gather_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            output = output_parallel

        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias
```

### Row-Parallel Linear Layer

The `RowParallelLinear` class (megatron/core/tensor_parallel/layers.py:1068-1303) implements linear layers where the input dimension is partitioned. For transformation `Y = XA + b`, the weight matrix `A` is transposed and parallelized along the first dimension, while input `X = [X_1, X_2, ..., X_p]` is split along the second dimension.

```python
class RowParallelLinear(torch.nn.Module):
    """Linear layer with row parallelism.

    The linear layer is defined as Y = XA + b. A is parallelized along its
    first dimension and X along its second dimension.
    A = transpose([A_1 .. A_p])
    X = [X_1, ..., X_p]

    Args:
        input_size: first dimension of matrix A.
        output_size: second dimension of matrix A.
        bias: If true, add bias. Note that bias is not parallelized.
        input_is_parallel: If true, we assume that the input is already
                           split across the GPUs and we do not split again.
        skip_bias_add: If True, do not add the bias term, instead return it
                       to be added by the caller.
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        *,
        config: ModelParallelConfig,
        init_method: Callable,
        bias: bool,
        input_is_parallel: bool,
        skip_bias_add: bool,
        stride: int = 1,
        is_expert: bool = False,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super(RowParallelLinear, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        self.skip_bias_add = skip_bias_add
        self.config = config
        self.is_expert = is_expert
        self.gradient_accumulation_fusion = config.gradient_accumulation_fusion
        self.sequence_parallel = config.sequence_parallel

        if self.sequence_parallel and not self.input_is_parallel:
            raise RuntimeError(
                "To enable `sequence_parallel`, `input_is_parallel` must be `True`"
            )

        # Get tensor parallel group
        self.tp_group = get_tensor_model_parallel_group_if_none(
            self.tp_group, is_expert=self.is_expert
        )
        world_size = get_pg_size(self.tp_group)
        rank = get_pg_rank(self.tp_group)

        # Divide input dimension across GPUs
        self.input_size_per_partition = divide(input_size, world_size)

        # Initialize weight
        if config.use_cpu_initialization:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    dtype=config.params_dtype
                )
            )
            if config.perform_initialization:
                self.master_weight = _initialize_affine_weight_cpu(
                    self.weight,
                    self.output_size,
                    self.input_size,
                    self.input_size_per_partition,
                    1,  # partition_dim=1 for row parallel
                    init_method,
                    stride=stride,
                    return_master_weight=keep_master_weight_for_test,
                    params_dtype=config.params_dtype,
                    rank=rank,
                    world_size=world_size,
                )
        else:
            self.weight = Parameter(
                torch.empty(
                    self.output_size,
                    self.input_size_per_partition,
                    device=torch.cuda.current_device(),
                    dtype=config.params_dtype,
                )
            )
            if config.perform_initialization:
                _initialize_affine_weight_gpu(
                    self.weight,
                    init_method,
                    partition_dim=1,
                    stride=stride,
                    is_expert=self.is_expert,
                )
```

The forward pass performs local computation followed by reduction across GPUs:

```python
    def forward(self, input_):
        """Forward of RowParallelLinear

        Args:
            input_: 3D tensor whose order of dimension is [sequence, batch, hidden]

        Returns:
            - output
            - bias
        """

        # Handle input splitting if not already parallel
        if self.input_is_parallel:
            input_parallel = input_
        else:
            assert not self.sequence_parallel
            input_parallel = scatter_to_tensor_model_parallel_region(
                input_, group=self.tp_group
            )

        # Perform local matrix multiplication with weight shard
        output_parallel = self._forward_impl(
            input=input_parallel,
            weight=self.weight,
            bias=None,
            gradient_accumulation_fusion=self.gradient_accumulation_fusion,
            allreduce_dgrad=allreduce_dgrad,
            sequence_parallel=False,
            tp_group=None,
            grad_output_buffer=None,
        )

        # Reduce across all tensor parallel partitions
        if self.explicit_expert_comm:
            assert self.skip_bias_add
            output_ = output_parallel
        elif self.sequence_parallel:
            # Use reduce-scatter for memory efficiency
            output_ = reduce_scatter_to_sequence_parallel_region(
                output_parallel, group=self.tp_group
            )
        else:
            # Standard all-reduce
            output_ = reduce_from_tensor_model_parallel_region(
                output_parallel, group=self.tp_group
            )

        if not self.skip_bias_add:
            output = (output_ + self.bias) if self.bias is not None else output_
            output_bias = None
        else:
            output = output_
            output_bias = self.bias

        return output, output_bias
```

## Communication Primitives

The tensor parallel communication operations are implemented in `megatron/core/tensor_parallel/mappings.py` using custom autograd functions to ensure correct gradient flow during backpropagation.

### Core Collective Operations

```python
def _reduce(input_, group):
    """All-reduce the input tensor across model parallel group."""
    assert group is not None, "group should not be None"

    # Bypass the function if we are using only 1 GPU.
    if group.size() == 1:
        return input_

    # All-reduce.
    torch.distributed.all_reduce(input_.contiguous(), group=group)

    return input_


def _split_along_last_dim(input_, group):
    """Split the tensor along its last dimension and keep the
    corresponding slice."""
    assert group is not None, "group should not be None"

    world_size = group.size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    # Split along last dimension.
    input_list = split_tensor_along_last_dim(input_, world_size)

    # Note: torch.split does not create contiguous tensors by default.
    rank = group.rank()
    output = input_list[rank].contiguous()

    return output


def _gather_along_first_dim(input_, group, output_split_sizes=None, use_global_buffer=False):
    """Gather tensors and concatenate along the first dimension.

    Args:
        input_tensor (torch.Tensor): A tensor to be gathered.
        output_split_sizes (List[int], optional): A list specifying the sizes
            of the output splits along the first dimension. If None, equal
            splitting is assumed. Default: None.

    Returns:
        torch.Tensor: Gathered tensor.
    """

    assert group is not None, "group should not be None"
    world_size = group.size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    dim_size = list(input_.size())
    if output_split_sizes is None:
        dim_size[0] = dim_size[0] * world_size

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        dist_all_gather_func(output, input_.contiguous(), group=group)
    else:
        dim_size[0] = sum(output_split_sizes)
        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output


def _reduce_scatter_along_first_dim(input_, group, input_split_sizes=None, use_global_buffer=False):
    """Reduce-scatter the input tensor across model parallel group.

    Args:
        input_ (torch.Tensor): The input tensor to be reduce-scattered.
        input_split_sizes (List[int], optional): A list specifying the sizes of
            the input splits along the first dimension for each rank. If None,
            equal splitting is assumed. Default: None.
    """
    assert group is not None, "group should not be None"
    world_size = group.size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), "First dimension of the tensor should be divisible by tensor parallel size"

        dim_size[0] = dim_size[0] // world_size

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(dim_size, dtype=input_.dtype, device=torch.cuda.current_device())
        dist_reduce_scatter_func(output, input_.contiguous(), group=group)
    else:
        rank = group.rank()
        input_tensor_list = list(torch.split(input_, input_split_sizes, dim=0))

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(
                input_tensor_list[rank].shape, input_.dtype, "mpu"
            )
        else:
            output = torch.empty_like(input_tensor_list[rank])
        torch.distributed.reduce_scatter(output, input_tensor_list, group=group)
    return output
```

## Transformer Block Integration

Tensor parallelism is applied strategically within transformer blocks to minimize communication. The MLP block demonstrates this pattern clearly (megatron/core/transformer/mlp.py:58-241):

```python
class MLP(MegatronModule):
    """
    MLP will take the input with h hidden state, project it to 4*h
    hidden dimension, perform nonlinear transformation, and project the
    state back into h hidden dimension.

    We use the following notation:
     h: hidden size
     p: number of tensor model parallel partitions
     b: batch size
     s: sequence length
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: MLPSubmodules,
        is_expert: bool = False,
        input_size: Optional[int] = None,
        ffn_hidden_size: int = None,
        tp_group: Optional[torch.distributed.ProcessGroup] = None,
    ):
        super().__init__(config=config)

        self.config: TransformerConfig = config
        self.input_size = input_size if input_size != None else self.config.hidden_size

        tp_group = get_tensor_model_parallel_group_if_none(tp_group, is_expert=is_expert)
        if ffn_hidden_size is None:
            ffn_hidden_size = self.config.ffn_hidden_size

        # If this is a gated linear unit we double the output width
        if self.config.gated_linear_unit:
            ffn_hidden_size *= 2

        # First layer: ColumnParallelLinear (h -> 4h/p on each GPU)
        self.linear_fc1 = build_module(
            submodules.linear_fc1,
            self.input_size,
            ffn_hidden_size,
            config=self.config,
            init_method=self.config.init_method,
            gather_output=False,  # Keep output partitioned
            bias=self.config.add_bias_linear,
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc1",
            tp_group=tp_group,
        )

        self.activation_func = self.config.activation_func

        # Second layer: RowParallelLinear (4h/p -> h with reduction)
        self.linear_fc2 = build_module(
            submodules.linear_fc2,
            self.config.ffn_hidden_size,
            self.config.hidden_size,
            config=self.config,
            init_method=self.config.output_layer_init_method,
            bias=self.config.add_bias_linear,
            input_is_parallel=True,  # Input already partitioned from fc1
            skip_bias_add=True,
            is_expert=is_expert,
            tp_comm_buffer_name="fc2",
            tp_group=tp_group,
        )

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the MLP block."""
        # [s, b, 4 * h/p] - Column parallel output
        intermediate_parallel, bias_parallel = self.linear_fc1(hidden_states)

        # Activation function applied to partitioned outputs
        if self.config.bias_activation_fusion:
            intermediate_parallel = self.activation_func(
                intermediate_parallel, bias_parallel
            )
        else:
            if bias_parallel is not None:
                intermediate_parallel = intermediate_parallel + bias_parallel
            intermediate_parallel = self.activation_func(intermediate_parallel)

        # [s, b, h] - Row parallel with all-reduce
        output, output_bias = self.linear_fc2(intermediate_parallel)

        return output, output_bias
```

## Configuration and Usage

Tensor parallelism is configured through the `ModelParallelConfig` class (megatron/core/model_parallel_config.py):

```python
@dataclass
class ModelParallelConfig:
    # Tensor parallelism size
    tensor_model_parallel_size: int = 1

    # Enable sequence parallelism for memory efficiency
    sequence_parallel: bool = False

    # Expert tensor parallelism for MoE layers
    expert_tensor_parallel_size: Optional[int] = None

    # Weight initialization mode
    use_cpu_initialization: bool = False
    perform_initialization: bool = True

    # Gradient accumulation fusion (requires APEX CUDA extensions)
    gradient_accumulation_fusion: bool = False

    # Parameter data type
    params_dtype: torch.dtype = torch.float32
```

Example configuration for training a 175B model:

```python
# For GPT-3 175B on 8x A100 80GB GPUs
config = ModelParallelConfig(
    tensor_model_parallel_size=8,      # Split across 8 GPUs
    sequence_parallel=True,             # Reduce memory with SP
    use_cpu_initialization=False,       # GPU init for speed
    gradient_accumulation_fusion=True,  # Enable if APEX available
    params_dtype=torch.bfloat16,       # Use BF16 for stability
)
```

## Performance Characteristics

**Memory Distribution:** With TP=8, a 175B parameter model (350GB in FP16) requires only ~44GB per GPU for parameters, fitting comfortably within A100 80GB memory along with activations and optimizer states.

**Communication Volume:** Each transformer layer requires two collective communications:
- ColumnParallelLinear backward: All-reduce of input gradients (unless using sequence parallel)
- RowParallelLinear forward: All-reduce of output activations (or reduce-scatter with sequence parallel)

**Communication Time:** On NVLink-connected GPUs (300-600 GB/s), all-reduce of tensors typically takes 1-3ms per operation. With communication-computation overlap (enabled via CUDA_DEVICE_MAX_CONNECTIONS=1 and async operations), up to 80-95% of this latency can be hidden.

**Scaling Efficiency:** Tensor parallelism scales well up to TP=8 within a single node with NVLink. Beyond this, communication overhead begins to dominate. Cross-node TP is possible but requires high-bandwidth interconnects like InfiniBand or NVSwitch.

## Advanced Features

**Sequence Parallelism Integration:** When `sequence_parallel=True`, Megatron-LM replaces all-reduce operations with reduce-scatter and replaces identity copies with all-gather, reducing activation memory by the tensor parallel size (see report #03 for details).

**Gradient Accumulation Fusion:** With `gradient_accumulation_fusion=True` and APEX CUDA extensions installed, gradient accumulation is fused with communication for improved performance.

**Expert Tensor Parallelism:** Mixture-of-Experts (MoE) layers can use separate tensor parallelism with `expert_tensor_parallel_size`, allowing different parallelization strategies for expert layers vs. dense layers.

**Async Communication:** The implementation uses `LinearWithGradAccumulationAndAsyncCommunication` to overlap all-reduce of gradients with backward computation, significantly reducing effective communication time.

## When to Use Tensor Parallelism

**Enable TP when:**
- Model parameters exceed single GPU memory (>60GB for A100 80GB)
- Training layers with billions of parameters (attention, MLP)
- GPUs connected via high-bandwidth interconnect (NVLink, NVSwitch)
- Need fine-grained memory distribution within layers

**Consider alternatives when:**
- Model fits on single GPU (no TP needed)
- Only modest GPU interconnect available (use pipeline parallelism instead)
- TP size would exceed 8 (diminishing returns, combine with pipeline parallelism)

Tensor parallelism is the foundation of Megatron-LM's multi-dimensional parallelism strategy, typically combined with pipeline parallelism (between nodes) and data parallelism (across data parallel groups) to achieve efficient training of models with hundreds of billions of parameters.
