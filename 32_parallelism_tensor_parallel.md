# Tensor Parallelism

## Context

Large layers don't fit on single GPU. **Solution:** Split individual weight matrices across GPUs.

## Implementation

Splits along matrix dimensions:
- **Column parallel:** Each GPU holds subset of columns
- **Row parallel:** Each GPU holds subset of rows
- All-gather inputs, compute local matmul, reduce-scatter outputs

```python
class ColumnParallelLinear:
    """Split weight columns across TP group"""
    
    def forward(self, input):
        # Each rank has partial columns of weight
        # Input same on all ranks (or sequence-parallel split)
        output_parallel = torch.matmul(input, self.weight_partition)
        
        # Reduce-scatter or keep partitioned
        if sequence_parallel:
            return output_parallel  # Each rank has partial sequence
        else:
            return all_gather(output_parallel)  # Gather full output
```

## Code Location

- **Column parallel:** `megatron/core/tensor_parallel/layers.py` lines 100-200
- **Row parallel:** Lines 202-300
- **Communication:** `megatron/core/tensor_parallel/mappings.py`

## Performance Impact

### Communication per Layer

- 2 all-gathers
- 2 reduce-scatters
- **With overlap:** 80-95% hidden

### Enables Training

**Without TP:** Model doesn't fit
**With TP:** Can train on multiple GPUs

## When to Use

**Use when:**
- Model doesn't fit on single GPU
- Layers are large (billions of parameters)
- NVLink-connected GPUs

**Typical TP sizes:** 2, 4, 8 (limited by NVLink topology)

**Configuration:**

```python
tensor_model_parallel_size = 4  # Split across 4 GPUs
```

## Related Optimizations

- [Tensor Parallelism Overlap](04_communication_tp_overlap.md) - Hide TP communication
- [Sequence Parallelism](03_communication_sequence_parallel.md) - Reduce TP memory
- [Sequence Parallelism Strategy](33_parallelism_sequence_parallel.md) - Combined strategy

## References

- Paper: [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053)
- [Tensor Parallelism Explained](https://huggingface.co/docs/transformers/v4.15.0/parallelism)

