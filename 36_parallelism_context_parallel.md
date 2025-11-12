# Context Parallelism (SSM)

## Context

SSM models (Mamba) with long contexts need to split sequence dimension differently than attention models.

## Implementation

Splits sequence dimension for SSM-specific operations. Different from sequence parallelism (which is for attention).

## Code Location

- **Implementation:** `megatron/core/ssm/mamba_context_parallel.py`

## Performance Impact

- Enables longer contexts for SSM models
- Similar benefits to sequence parallelism but SSM-specific

## When to Use

**Mamba/SSM models only:**
- Long contexts (> 4096)
- `context_parallel_size > 1`

**Configuration:**

```python
# For Mamba models
context_parallel_size = 2  # or 4

# Don't use with standard attention models!
```

**Skip if:**
- Not using SSM models
- Standard attention (use sequence parallelism instead)

## Related Optimizations

- [Sequence Parallelism](33_parallelism_sequence_parallel.md) - For attention models

## References

- Paper: [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [Megatron Mamba Implementation](https://github.com/NVIDIA/Megatron-LM)

