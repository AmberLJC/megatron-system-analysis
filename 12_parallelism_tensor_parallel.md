# 12. Tensor Parallelism

## Context

Large models don't fit on single GPU. A 175B parameter model needs ~350GB for parameters alone (FP16), but A100 has only 80GB.

**Solution:** Split individual layers across multiple GPUs. Each GPU holds a fraction of each weight matrix.

## Implementation

**Column-parallel:** Split weight columns across GPUs
```
Full: Y = XW where W is [hidden, hidden]
Split: Each GPU computes Y_i = X @ W_i where W_i is [hidden, hidden/TP]
```

**Row-parallel:** Split weight rows across GPUs  
```
Full: Y = XW where W is [hidden, hidden]
Split: Each GPU computes Y_i = X_i @ W where X_i is [seq, hidden/TP]
       Then reduce-scatter or all-reduce Y_i across GPUs
```

## Core Code

- `megatron/core/tensor_parallel/layers.py:100-200` - ColumnParallelLinear
- `megatron/core/tensor_parallel/layers.py:202-300` - RowParallelLinear
- `megatron/core/tensor_parallel/mappings.py` - Communication primitives

## Code Snippet

```python
# Column-parallel linear layer
class ColumnParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, ...):
        # Each GPU holds output_size/TP columns
        self.output_size_per_partition = output_size // tp_world_size
        self.weight = torch.nn.Parameter(torch.empty(
            self.output_size_per_partition, input_size
        ))
        
    def forward(self, input):
        # Input: [seq_len, batch, hidden] (replicated or sequence-parallel)
        # Output: [seq_len, batch, hidden/TP] (partitioned)
        
        # All-gather input if needed (or use sequence-parallel)
        if self.sequence_parallel:
            input_parallel = input  # Already sharded by sequence
        else:
            input_parallel = copy_to_tensor_model_parallel_region(input)
        
        # Local matmul with weight shard
        output_parallel = F.linear(input_parallel, self.weight, self.bias)
        
        return output_parallel  # Each GPU has different columns

# Row-parallel linear layer  
class RowParallelLinear(torch.nn.Module):
    def __init__(self, input_size, output_size, ...):
        # Each GPU holds input_size/TP rows
        self.input_size_per_partition = input_size // tp_world_size
        self.weight = torch.nn.Parameter(torch.empty(
            output_size, self.input_size_per_partition
        ))
        
    def forward(self, input):
        # Input: [seq_len, batch, hidden/TP] (partitioned)
        # Output: [seq_len, batch, hidden] (replicated or reduce-scattered)
        
        # Local matmul with weight shard
        output_parallel = F.linear(input, self.weight)
        
        # All-reduce across TP group
        if self.sequence_parallel:
            # Reduce-scatter instead of all-reduce (saves memory)
            output = reduce_scatter_to_sequence_parallel_region(output_parallel)
        else:
            output = reduce_from_tensor_model_parallel_region(output_parallel)
        
        return output  # Sum of partial results from all GPUs
```

## When to Use

**Enable when:**
- Model doesn't fit on single GPU
- Layers are large (billions of parameters)
- NVLink-connected GPUs

```python
tensor_model_parallel_size = 8  # TP=2, 4, or 8 (limited by NVLink topology)
```

**Skip if:**
- Model fits on single GPU
- Slow interconnect (Ethernet)

## Performance Impact

**Communication per layer:**
- Column-parallel: All-gather inputs (or none with SP)
- Row-parallel: All-reduce outputs (or reduce-scatter with SP)
- With overlap: 80-95% hidden (#04)

**Enables training:**
- 175B model: Requires TP=8 on A100 80GB
- 70B model: Requires TP=4 on A100 80GB

**Example:** GPT-3 175B with TP=8
- Per-GPU memory: 350GB / 8 = 43.75GB parameters
- Communication: ~2ms per layer (mostly overlapped)
- End-to-end: Enables training that wouldn't fit otherwise!

## Configuration Example

```python
# Tensor parallelism configuration
tensor_model_parallel_size = 8  # Split across 8 GPUs
sequence_parallel = True        # Enable to reduce memory (#03)

# Environment: MANDATORY for overlap!
# export CUDA_DEVICE_MAX_CONNECTIONS=1
```

## References

- Megatron paper: [Megatron-LM](https://arxiv.org/abs/1909.08053)
- Implementation: `megatron/core/tensor_parallel/`

