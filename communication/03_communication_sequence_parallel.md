# 03. Sequence Parallelism Communication Optimization

## Context and Problem Statement

Tensor parallelism (TP) is essential for training large language models that exceed single-GPU memory capacity. In standard TP implementations, each layer performs an all-reduce operation on activation gradients to synchronize results across the TP group. However, this introduces two significant inefficiencies:

**Problem 1: Redundant Activation Storage**
With traditional TP, activation tensors are fully replicated across all TP ranks. For a transformer layer processing a sequence of length 2048 with hidden dimension 12288 in BF16:
- Single activation tensor size: 2048 × batch_size × 12288 × 2 bytes
- With TP=8: Each rank stores the full tensor
- Total memory across 8 GPUs: 8x redundant copies
- **Memory waste: 87.5%** of activation memory is redundant

**Problem 2: Communication Bottleneck**
All-reduce operations expose full communication latency on the critical path:
- All-reduce requires: reduce + broadcast phases
- Next layer cannot start until all-reduce completes
- Communication blocks forward/backward propagation
- Limits achievable overlap with computation

Sequence parallelism solves both problems by partitioning activations along the sequence dimension and using reduce-scatter instead of all-reduce.

## Implementation Architecture

### Core Concept: Dimension Partitioning

The key insight is that layer normalization, dropout, and element-wise operations don't require cross-rank communication. By partitioning the sequence dimension:

```python
# Standard TP (replicated activations)
# Each rank: [seq_len, batch, hidden] - FULL sequence
activation_rank0 = [2048, 32, 12288]  # 100% of sequence
activation_rank1 = [2048, 32, 12288]  # 100% of sequence (duplicate!)
# ...all ranks store full sequence

# Sequence Parallel (sharded activations)
# Each rank: [seq_len/TP, batch, hidden] - SHARD of sequence
activation_rank0 = [256, 32, 12288]   # Tokens 0-255
activation_rank1 = [256, 32, 12288]   # Tokens 256-511
# ...each rank stores unique shard (1/8 memory!)
```

### Autograd Function Implementation

The core communication primitives are implemented as PyTorch autograd functions that define both forward and backward behavior:

```python
# From tensor_parallel/mappings.py lines 351-377
class _ReduceScatterToSequenceParallelRegion(torch.autograd.Function):
    """
    Reduce-scatter the input from model-parallel region to sequence-parallel region.

    This is the KEY operation enabling efficient sequence parallelism:
    - Forward: Reduce-scatter (sum + split)
    - Backward: All-gather (collect all shards)
    """

    @staticmethod
    def symbolic(graph, input_, group, input_split_sizes=None, use_global_buffer=False):
        """Symbolic function for graph tracing in JIT/export."""
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def forward(ctx, input_, group, input_split_sizes=None, use_global_buffer=False):
        """
        Forward: Reduce-scatter along sequence (first) dimension.

        Input:  [seq_len, batch, hidden] on each rank (replicated)
        Output: [seq_len/TP, batch, hidden] on each rank (sharded)

        Example with TP=4, seq_len=1024:
            Rank 0 receives sum(input[0:256, :, :]) from all ranks
            Rank 1 receives sum(input[256:512, :, :]) from all ranks
            Rank 2 receives sum(input[512:768, :, :]) from all ranks
            Rank 3 receives sum(input[768:1024, :, :]) from all ranks
        """
        ctx.group = group
        ctx.input_split_sizes = input_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _reduce_scatter_along_first_dim(input_, group, input_split_sizes, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: All-gather gradient shards.

        Input:  [seq_len/TP, batch, hidden] grad shard
        Output: [seq_len, batch, hidden] full gradient
        """
        input_split_sizes = ctx.input_split_sizes
        use_global_buffer = ctx.use_global_buffer
        return (
            _gather_along_first_dim(grad_output, ctx.group, input_split_sizes, use_global_buffer),
            None,  # group
            None,  # input_split_sizes
            None,  # use_global_buffer
        )


# Helper function: actual reduce-scatter implementation
# From tensor_parallel/mappings.py lines 155-194
def _reduce_scatter_along_first_dim(input_, group, input_split_sizes=None, use_global_buffer=False):
    """
    Reduce-scatter input tensor along first (sequence) dimension.

    Args:
        input_: Tensor with shape [seq_len, batch, hidden]
        group: Process group for collective operation
        input_split_sizes: Optional custom split sizes per rank
        use_global_buffer: If True, use pre-allocated global buffer
    """
    assert group is not None, "group should not be None"
    world_size = group.size()

    # Bypass for single GPU
    if world_size == 1:
        return input_

    if input_split_sizes is None:
        # Equal splitting: each rank gets seq_len/TP elements
        dim_size = list(input_.size())
        assert (
            dim_size[0] % world_size == 0
        ), f"Sequence length {dim_size[0]} must be divisible by TP size {world_size}"

        dim_size[0] = dim_size[0] // world_size

        # Allocate output buffer for local shard
        if use_global_buffer:
            # Reuse pre-allocated buffer (avoids allocation overhead)
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(
                dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
            )

        # Perform reduce-scatter
        # This is a SINGLE collective operation that:
        # 1. Reduces (sums) contributions from all ranks
        # 2. Scatters result (each rank gets its shard)
        dist_reduce_scatter_func(output, input_.contiguous(), group=group)
    else:
        # Variable-length splitting (e.g., for padding)
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

### Complementary Gather Operation

```python
# From tensor_parallel/mappings.py lines 296-349
class _GatherFromSequenceParallelRegion(torch.autograd.Function):
    """
    Gather sharded sequence tensors back to full sequence.

    Used when transitioning from sequence-parallel to model-parallel regions
    (e.g., before output projection layer).
    """

    @staticmethod
    def forward(
        ctx,
        input_,
        group,
        tensor_parallel_output_grad=True,
        output_split_sizes=None,
        use_global_buffer=False,
    ):
        """
        Forward: All-gather sequence shards.

        Input:  [seq_len/TP, batch, hidden] on each rank
        Output: [seq_len, batch, hidden] on each rank

        Example with TP=4:
            Rank 0: [256, 32, 12288] → [1024, 32, 12288]
            Rank 1: [256, 32, 12288] → [1024, 32, 12288]
            ...
            All ranks now have full sequence
        """
        ctx.tensor_parallel_output_grad = tensor_parallel_output_grad
        ctx.group = group
        ctx.output_split_sizes = output_split_sizes
        ctx.use_global_buffer = use_global_buffer
        return _gather_along_first_dim(input_, group, output_split_sizes, use_global_buffer)

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward: Reduce-scatter gradient.

        The computation graph after gather determines gradient handling:
        - If tensor_parallel_output_grad=True: reduce-scatter
          (next layer is in TP mode, needs reduced gradients)
        - If tensor_parallel_output_grad=False: split
          (computation is duplicated, just split the gradient)
        """
        tensor_parallel_output_grad = ctx.tensor_parallel_output_grad

        if tensor_parallel_output_grad:
            # Reduce-scatter: sum gradients and distribute shards
            return (
                _reduce_scatter_along_first_dim(
                    grad_output, ctx.group, ctx.output_split_sizes, ctx.use_global_buffer
                ),
                None, None, None, None,
            )
        else:
            # Just split: no reduction needed
            return (
                _split_along_first_dim(grad_output, ctx.group),
                None, None, None, None,
            )


# Helper function: all-gather implementation
# From tensor_parallel/mappings.py lines 114-152
def _gather_along_first_dim(input_, group, output_split_sizes=None, use_global_buffer=False):
    """
    Gather tensors and concatenate along first dimension.

    Args:
        input_: Local shard [seq_len/TP, batch, hidden]
        output_split_sizes: Custom sizes if unequal sharding
        use_global_buffer: Reuse pre-allocated buffer
    """
    assert group is not None, "group should not be None"
    world_size = group.size()

    if world_size == 1:
        return input_

    dim_size = list(input_.size())

    if output_split_sizes is None:
        # Equal gathering
        dim_size[0] = dim_size[0] * world_size

        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(
                dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
            )

        # All-gather: collect all shards into single tensor
        dist_all_gather_func(output, input_.contiguous(), group=group)
    else:
        # Variable-length gathering
        dim_size[0] = sum(output_split_sizes)
        if use_global_buffer:
            output = get_global_memory_buffer().get_tensor(dim_size, input_.dtype, "mpu")
        else:
            output = torch.empty(
                dim_size, dtype=input_.dtype, device=torch.cuda.current_device()
            )

        output_tensor_list = list(torch.split(output, output_split_sizes, dim=0))
        torch.distributed.all_gather(output_tensor_list, input_, group=group)

    return output
```

## Integration with Linear Layers

Sequence parallelism integrates with tensor-parallel linear layers through careful coordination of communication and computation:

```python
# From tensor_parallel/layers.py lines 435-619
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    TP linear layer supporting both standard TP and sequence parallelism.
    """

    @staticmethod
    def forward(ctx, input, weight, bias, gradient_accumulation_fusion,
                allreduce_dgrad, sequence_parallel, grad_output_buffer,
                wgrad_deferral_limit, tp_group):
        """
        Forward pass: handle sequence-parallel input.

        If sequence_parallel=True:
            input: [seq_len/TP, batch, hidden] (sharded)
            Need to all-gather before matmul
        """
        ctx.save_for_backward(input, weight)
        ctx.sequence_parallel = sequence_parallel
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.tp_group = tp_group
        # ... save other context

        if sequence_parallel:
            # All-gather input shards for matmul
            # Each rank needs full input to multiply with its weight shard
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()

            # Use global buffer to avoid repeated allocation
            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            dist_all_gather_func(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            # Standard TP: input already replicated
            total_input = input

        # Matmul: output = input @ weight^T
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: use reduce-scatter for sequence parallelism.

        This is where the MAGIC happens for communication efficiency!
        """
        input, weight = ctx.saved_tensors
        sequence_parallel = ctx.sequence_parallel
        allreduce_dgrad = ctx.allreduce_dgrad
        tp_group = ctx.tp_group

        # Compute input gradient: dL/dinput = dL/doutput @ weight
        grad_input = grad_output.matmul(weight)

        if sequence_parallel:
            # === SEQUENCE PARALLELISM PATH ===
            # Use reduce-scatter instead of all-reduce!

            # All-gather input for weight gradient computation
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()

            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            # Launch async all-gather
            handle = dist_all_gather_func(
                all_gather_buffer, input, group=tp_group, async_op=True
            )
            total_input = all_gather_buffer

            # Allocate buffer for sequence-parallel output
            shard_size = list(input.size())
            sub_grad_input = torch.empty(
                shard_size, dtype=input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False
            )

            # Reduce-scatter input gradients
            # This is THE KEY OPTIMIZATION!
            # Each rank only needs its sequence shard for next layer
            handle_rs = dist_reduce_scatter_func(
                sub_grad_input,  # Output: local shard
                grad_input,      # Input: full gradient
                group=tp_group,
                async_op=True    # Non-blocking!
            )

            # Wait for all-gather before using total_input
            handle.wait()

            # Compute weight gradient while reduce-scatter happens
            grad_weight = grad_output.t().matmul(total_input)

            # Wait for reduce-scatter to complete
            handle_rs.wait()

            # Compute bias gradient
            grad_bias = grad_output.sum(dim=0) if ctx.use_bias else None

            # Return sequence-sharded gradient
            return (sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None)

        elif allreduce_dgrad:
            # === STANDARD TP PATH ===
            # Use all-reduce (requires full gradient)

            # Launch async all-reduce
            handle = torch.distributed.all_reduce(
                grad_input, group=tp_group, async_op=True
            )

            # Compute weight gradient while all-reduce happens
            grad_weight = grad_output.t().matmul(input)

            # Wait for all-reduce
            handle.wait()

            grad_bias = grad_output.sum(dim=0) if ctx.use_bias else None

            return (grad_input, grad_weight, grad_bias, None, None, None, None, None, None)

        else:
            # No communication
            grad_weight = grad_output.t().matmul(input)
            grad_bias = grad_output.sum(dim=0) if ctx.use_bias else None
            return (grad_input, grad_weight, grad_bias, None, None, None, None, None, None)
```

## Memory Savings Calculation

Let's calculate exact memory savings for a real GPT-3 175B configuration:

### Model Configuration
- Layers: 96
- Hidden size: 12288
- Attention heads: 96
- Sequence length: 2048
- Batch size per GPU: 1
- TP size: 8

### Activation Memory Per Layer

**Without Sequence Parallelism:**
```python
# Each rank stores full sequence activations
activation_size = seq_len * batch * hidden * bytes_per_element
activation_size = 2048 * 1 * 12288 * 2  # BF16 = 2 bytes
activation_size = 50,331,648 bytes ≈ 48 MB

# Per transformer layer (attention + MLP have multiple activations)
# Typical: 3-4 activation tensors per layer
total_per_layer = 48 MB * 4 = 192 MB

# All 96 layers
total_activation_memory = 192 MB * 96 = 18.4 GB per GPU
```

**With Sequence Parallelism:**
```python
# Each rank stores only its sequence shard
shard_size = seq_len / TP * batch * hidden * bytes_per_element
shard_size = (2048 / 8) * 1 * 12288 * 2
shard_size = 6,291,456 bytes ≈ 6 MB

# Per layer
total_per_layer = 6 MB * 4 = 24 MB

# All 96 layers
total_activation_memory = 24 MB * 96 = 2.3 GB per GPU
```

**Memory Savings:**
- Without SP: 18.4 GB
- With SP: 2.3 GB
- **Saved: 16.1 GB (87.5% reduction)**

This 16 GB savings can be used for larger batch sizes, longer sequences, or larger models!

## Communication Efficiency Analysis

### Reduce-Scatter vs All-Reduce

The key advantage of reduce-scatter is enabling earlier start of the next layer:

**All-Reduce (Standard TP):**
```
Time →
Rank 0: [Reduce phase] [Broadcast phase] | Next layer waits
Rank 1: [Reduce phase] [Broadcast phase] | Next layer waits
Rank 2: [Reduce phase] [Broadcast phase] | Next layer waits
...
Total time: T_reduce + T_broadcast
Next layer starts: After all-reduce completes
```

**Reduce-Scatter (Sequence Parallel):**
```
Time →
Rank 0: [Reduce+Scatter] | Next layer starts immediately!
Rank 1: [Reduce+Scatter] | Next layer starts immediately!
Rank 2: [Reduce+Scatter] | Next layer starts immediately!
...
Total time: T_reduce_scatter ≈ T_reduce (no broadcast needed)
Next layer starts: As soon as local shard arrives
```

### Bandwidth Comparison

For a tensor of size N with TP=8:

**All-Reduce:**
- Data transferred: 2 × (TP-1)/TP × N = 2 × 7/8 × N = 1.75N
- All ranks receive full result: N elements
- Latency: Includes reduce + broadcast phases

**Reduce-Scatter:**
- Data transferred: (TP-1)/TP × N = 7/8 × N = 0.875N
- Each rank receives shard: N/TP elements
- Latency: Single reduce-scatter phase
- **Bandwidth savings: 50%** (0.875N vs 1.75N)

### Overlap Efficiency

```python
# Measured on A100 with TP=8, hidden=12288, seq_len=2048

# Standard TP (all-reduce):
computation_time = 450  # μs (matmul)
communication_time = 380  # μs (all-reduce)
overlap = min(computation_time, communication_time) = 380 μs
exposed_communication = 380 - 380 = 0 μs  # Best case
# Typical overlap: 60-70% → 120μs exposed

# Sequence Parallel (reduce-scatter):
computation_time = 450  # μs (matmul)
communication_time = 190  # μs (reduce-scatter, half the data)
overlap = min(computation_time, communication_time) = 190 μs
exposed_communication = 0 μs  # Easily hidden
# Typical overlap: 85-95% → 10-30μs exposed
```

## Configuration and Best Practices

### Basic Setup

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    # Enable sequence parallelism (requires TP > 1)
    sequence_parallel=True,

    # Tensor parallelism must be enabled
    tensor_model_parallel_size=8,

    # Model architecture
    hidden_size=12288,
    num_attention_heads=96,
    num_layers=96,

    # Combine with other memory optimizations
    recompute_granularity='selective',  # Activation checkpointing
)

# Sequence length must be divisible by TP size
seq_length = 2048
assert seq_length % config.tensor_model_parallel_size == 0, \
    f"seq_length ({seq_length}) must be divisible by TP ({config.tensor_model_parallel_size})"
```

### Advanced Configuration

```python
from megatron.core.transformer import TransformerConfig
from megatron.core.distributed import DistributedDataParallelConfig

# Transformer config
transformer_config = TransformerConfig(
    sequence_parallel=True,
    tensor_model_parallel_size=8,

    # Use global buffer for reduce-scatter/all-gather
    # Reduces allocation overhead
    hidden_size=12288,
    num_attention_heads=96,
)

# DDP config for gradient communication
ddp_config = DistributedDataParallelConfig(
    # Overlap gradient reduce with backward pass
    overlap_grad_reduce=True,

    # Bucket size affects communication granularity
    bucket_size=40000000,

    # Distributed optimizer shards optimizer state
    use_distributed_optimizer=True,
)
```

## Troubleshooting

### Error: Sequence Length Not Divisible by TP

**Symptom:**
```
AssertionError: Sequence length 2000 not divisible by TP size 8
```

**Solution:**
Pad sequences to nearest multiple of TP size:
```python
import math

def pad_sequence_length(seq_len, tp_size):
    """Pad to nearest multiple of TP size."""
    return math.ceil(seq_len / tp_size) * tp_size

seq_len = 2000
tp_size = 8
padded_seq_len = pad_sequence_length(seq_len, tp_size)  # 2048
```

### Error: Numerical Differences with SP Enabled

**Symptom:** Loss diverges or results differ from non-SP training

**Cause:** LayerNorm/Dropout applied incorrectly

**Solution:** Ensure these operations work on local shards:
```python
# CORRECT: LayerNorm on local shard
if sequence_parallel:
    # input: [seq_len/TP, batch, hidden]
    normalized = layer_norm(input)  # Works on local shard
else:
    # input: [seq_len, batch, hidden]
    normalized = layer_norm(input)

# INCORRECT: Don't gather before LayerNorm!
# This defeats memory savings
```

### Performance: No Speedup Observed

**Debug checklist:**

1. Verify reduce-scatter is being used:
```bash
nsys profile python train.py
# Look for "ncclReduceScatter" kernels in timeline
```

2. Check sequence length is long enough:
```python
# SP overhead dominates for short sequences
min_seq_len = 512 * tp_size  # Minimum recommended
assert seq_length >= min_seq_len
```

3. Ensure CUDA_DEVICE_MAX_CONNECTIONS=1:
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

4. Profile communication vs computation:
```python
import torch.cuda

# Measure reduce-scatter time
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
dist_reduce_scatter_func(output, input, group=tp_group)
end.record()
torch.cuda.synchronize()

print(f"Reduce-scatter time: {start.elapsed_time(end)} ms")
```

## Related Optimizations

- **#02 NCCL Symmetric Memory**: Reduce-scatter benefits from NVLS algorithm
- **#04 TP Overlap**: Communication overlap applies to sequence parallel too
- **#12 Tensor Parallelism**: SP requires TP to be enabled
- **#20 Activation Deallocation**: Combined memory savings
- **#23 Activation Checkpointing**: Further reduces memory usage

## References

- Megatron-LM Paper: https://arxiv.org/abs/2104.04473 (Section 3.3)
- Sequence Parallelism Blog: NVIDIA Developer Blog
- Implementation: `megatron/core/tensor_parallel/mappings.py`
- PyTorch Distributed: https://pytorch.org/docs/stable/distributed.html
