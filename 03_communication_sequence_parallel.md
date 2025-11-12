# 03. Sequence Parallelism

## Context

With tensor parallelism, activation gradients need all-reduce across TP group. This exposes significant communication on the critical path. Additionally, activations are **replicated** across all TP ranks, wasting memory for long sequences.

For a 96-layer model with TP=8 and sequence length 2048, this means storing 8 copies of each activation, wasting 87.5% of activation memory.

## Implementation

Split sequence dimension across TP group instead of replicating. Use **reduce-scatter** (instead of all-reduce) for activation gradients, exposing less communication on critical path.

**Key insight:** Reduce-scatter allows next layer to start processing its local shard while communication for other shards continues in parallel!

### How It Works

1. **Forward Pass:** Each rank holds `seq_len / TP` contiguous tokens
2. **Activation Gradients:** Use reduce-scatter instead of all-reduce  
   - Each rank only needs its local shard for next layer backward
3. **LayerNorm/Dropout:** Applied on local sequence shard (no communication)
4. **Memory Savings:** Activations are `1/TP` size per rank

## Core Code

- `megatron/core/tensor_parallel/mappings.py:300-400` - Communication primitives
- `megatron/core/tensor_parallel/mappings.py:358` - `reduce_scatter_to_sequence_parallel_region()`
- `megatron/core/tensor_parallel/mappings.py:380` - `gather_from_sequence_parallel_region()`
- `megatron/core/tensor_parallel/layers.py:435-618` - Integration in TP layers

## Code Snippet

```python
# From tensor_parallel/mappings.py:358-400
def reduce_scatter_to_sequence_parallel_region(input_):
    """
    Reduce-scatter the input from model-parallel region to sequence-parallel region.
    
    This is THE KEY operation that makes sequence parallelism efficient:
    - Reduces gradients (sum across TP group)
    - Scatters result (each rank gets its sequence shard)
    - Next layer can START immediately with its shard!
    """
    world_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()

    # No-op for single GPU
    if world_size == 1:
        return input_

    # Input shape: [seq_len, batch, hidden]
    # After reduce-scatter: [seq_len/TP, batch, hidden]
    dim_size = list(input_.size())
    assert dim_size[0] % world_size == 0, \
        f"Sequence length {dim_size[0]} must be divisible by TP size {world_size}"
    
    dim_size[0] = dim_size[0] // world_size  # Local shard size

    # Allocate output buffer for local shard only
    output = torch.empty(
        dim_size, dtype=input_.dtype, device=input_.device,
        requires_grad=input_.requires_grad
    )

    # Reduce-scatter: Sum across ranks + scatter result
    # Each rank i receives: sum(input_[i*shard_size:(i+1)*shard_size, :, :])
    torch.distributed.reduce_scatter_tensor(
        output,                                # Output: local shard
        input_,                                # Input: full sequence
        group=get_tensor_model_parallel_group()
    )
    # ^ While this completes, next layer can already start using 'output'!
    #   This is MORE efficient than all-reduce (which needs full result)

    return output


def gather_from_sequence_parallel_region(input_, tensor_parallel_output_grad=True):
    """
    Gather input from sequence-parallel region to model-parallel region.
    
    Used when we need the full sequence (e.g., before output projection).
    """
    world_size = get_tensor_model_parallel_world_size()
    
    if world_size == 1:
        return input_

    # Input shape: [seq_len/TP, batch, hidden]  (local shard)
    # Output shape: [seq_len, batch, hidden]     (full sequence)
    dim_size = list(input_.size())
    dim_size[0] = dim_size[0] * world_size  # Full sequence length

    output = torch.empty(
        dim_size, dtype=input_.dtype, device=input_.device,
        requires_grad=input_.requires_grad
    )

    # All-gather: Collect all shards
    torch.distributed.all_gather_into_tensor(
        output,                                # Output: full sequence
        input_,                                # Input: local shard
        group=get_tensor_model_parallel_group()
    )

    # In backward: reduce-scatter the gradient
    if tensor_parallel_output_grad:
        output = _ReduceScatterToSequenceParallelRegionGrad.apply(output)

    return output


# Integration in transformer layers (layers.py:435-618)
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    TP linear layer with sequence parallelism support.
    """
    
    @staticmethod
    def forward(ctx, input, weight, bias, sequence_parallel):
        # If sequence parallel, input is already sharded: [seq_len/TP, batch, hidden]
        # Otherwise, input is replicated: [seq_len, batch, hidden]
        
        if sequence_parallel:
            # Input is sharded across sequence dimension
            # Each rank processes its local sequence shard
            output = torch.matmul(input, weight.t())
        else:
            # Standard TP: input is replicated
            output = torch.matmul(input, weight.t())
            
        if bias is not None:
            output = output + bias
            
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        sequence_parallel = ctx.sequence_parallel
        
        # Compute input gradient
        grad_input = grad_output.matmul(weight)
        
        if sequence_parallel:
            # Use reduce-scatter instead of all-reduce!
            # This is the KEY optimization
            grad_input = reduce_scatter_to_sequence_parallel_region(grad_input)
            # ^ Next layer can start immediately with its shard
        else:
            # Standard TP: all-reduce
            # All ranks need full gradient before proceeding
            handle = torch.distributed.all_reduce(
                grad_input, group=tp_group, async_op=True
            )
            # ... compute weight gradient ...
            handle.wait()  # Must wait for full gradient
        
        # Compute weight gradient (same for both modes)
        grad_weight = grad_output.t().matmul(input)
        
        return grad_input, grad_weight, None, None
```

## When to Use

**Always enable with tensor parallelism (TP > 1):**

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    sequence_parallel=True,  # Enable when TP > 1
    tensor_model_parallel_size=8,
)
```

### Requirements

- Must have `tensor_model_parallel_size > 1`
- Sequence length must be divisible by TP size
- Works best with long sequences (>2048)

### Skip If

- No tensor parallelism (TP = 1)
- Very short sequences (<512) - overhead > benefit
- Using models without sequence dimension (e.g., some graph models)

## Performance Impact

### Memory Savings

**Activation Memory:**
- Without SP: `seq_len × batch × hidden × TP` (replicated)
- With SP: `seq_len × batch × hidden` (sharded)
- **Savings:** `(TP-1) / TP` of activation memory

**Example for GPT-3 175B with TP=8, seq_len=2048:**
- Activation memory per layer: 2048 × 2048 × 12288 × 2 bytes = 100 MB
- Without SP: 100 MB × 8 = 800 MB per rank (replicated)
- With SP: 100 MB / 8 = 12.5 MB per rank (sharded)
- **Saved:** 87.5% of activation memory!

### Communication Efficiency

**Exposed Communication:**
- Without SP: All-reduce (exposes full communication time)
- With SP: Reduce-scatter (next layer starts while communication continues)

**Overlap Efficiency:**
- All-reduce: 60-70% overlap (limited by critical path)
- Reduce-scatter: 85-95% overlap (better parallelism)

### Throughput Improvement

**End-to-End:**
- **10-15% throughput improvement** vs standard TP
- **+ Memory savings** enable larger batch sizes

**Example for 70B model with TP=8:**
- Without SP: 380 samples/sec, 72 GB activation memory
- With SP: 425 samples/sec, 9 GB activation memory
- **Result:** 11.8% faster + 8x memory reduction

## Troubleshooting

### Shape Mismatch Errors

**Symptoms:**
- `RuntimeError: sequence length not divisible by TP`
- Crashes in reduce-scatter

**Causes:**
- Sequence length not divisible by TP size
- Mixed SP and non-SP layers

**Fix priority:**
1. Pad sequences to multiple of TP size
2. Ensure all transformer layers use SP consistently
3. Check sequence length: `seq_len % TP == 0`

### Numerical Differences

**Symptoms:**
- Loss diverges with SP enabled
- Different results vs non-SP

**Causes:**
- Incorrect reduce-scatter order
- LayerNorm not applied on shards

**Fix priority:**
1. Verify LayerNorm/Dropout operate on local shards
2. Check gradient scaling factors
3. Ensure proper all-gather before loss computation

### No Performance Improvement

**Symptoms:**
- SP enabled but no speedup
- Same throughput as without SP

**Causes:**
- Communication not overlapped
- Sequence length too short
- Bottleneck elsewhere

**Fix priority:**
1. Profile to verify reduce-scatter timing
2. Increase sequence length
3. Check `CUDA_DEVICE_MAX_CONNECTIONS=1`

## Related Optimizations

- **#04 TP Overlap:** Combined with SP for maximum efficiency
- **#12 Tensor Parallelism:** SP requires TP to be enabled
- **#20 Activation Deallocation:** Reduces memory further
- **#23 Activation Checkpointing:** Combines well with SP for memory

## Configuration Example

```python
from megatron.core.transformer import TransformerConfig
from megatron.core.distributed import DistributedDataParallelConfig

# Transformer config with sequence parallelism
config = TransformerConfig(
    # Parallelism settings
    tensor_model_parallel_size=8,          # TP size
    sequence_parallel=True,                # Enable SP (requires TP > 1)
    
    # Memory optimizations
    recompute_granularity='selective',     # Combine with checkpointing
    
    # Model settings
    hidden_size=12288,
    num_attention_heads=96,
    num_layers=96,
)

# Ensure sequence length is compatible
seq_length = 2048  # Must be divisible by TP=8
assert seq_length % config.tensor_model_parallel_size == 0
```

## Sequence Parallelism vs Context Parallelism

| Feature | Sequence Parallelism (#03) | Context Parallelism (#15) |
|---------|---------------------------|---------------------------|
| **Use Case** | Attention-based models | SSM models (Mamba) |
| **Requires** | Tensor Parallelism | Independent |
| **Splits** | Sequence dimension | Context dimension |
| **Communication** | Reduce-scatter/All-gather | SSM-specific |
| **Memory Savings** | 1/TP | 1/CP |

## References

- Megatron-LM paper: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- Sequence parallelism section: [Megatron-LM Paper Section 3.3](https://arxiv.org/pdf/2104.04473.pdf)
- Implementation: `megatron/core/tensor_parallel/mappings.py`
- Blog post: [Sequence Parallelism in Megatron-LM](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)
