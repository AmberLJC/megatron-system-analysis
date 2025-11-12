# Sequence Parallelism (Strategy)

## Context

With tensor parallelism, activations are replicated. For long sequences, this wastes memory.

**Solution:** Split sequence dimension across TP group instead of replicating.

## Implementation

See detailed implementation in [Communication: Sequence Parallelism](03_communication_sequence_parallel.md).

**Strategy:** Each rank processes different tokens for its TP slice.

## Performance Impact

- **Memory:** Reduces activation memory by TP factor
- **Communication:** Reduce-scatter instead of all-reduce (less exposed)
- **Throughput:** 10-15% improvement

## When to Use

**Always use with tensor parallelism:**
- TP > 1
- Any sequence length
- Memory + communication benefits

**Configuration:**

```python
tensor_model_parallel_size = 4
sequence_parallel = True  # Always enable with TP!
```

## Related Optimizations

- [Sequence Parallelism Communication](03_communication_sequence_parallel.md) - Implementation details
- [Tensor Parallelism](32_parallelism_tensor_parallel.md) - Requires TP

## References

- Paper: [Reducing Activation Recomputation](https://arxiv.org/abs/2205.05198)

