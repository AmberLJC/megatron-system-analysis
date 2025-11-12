# Grouped GEMM for MoE

## Context

Sequential expert GEMMs in MoE have poor GPU utilization. Each expert GEMM is small, leaving GPU underutilized.

**Solution:** Batch all expert GEMMs into single grouped GEMM operation.

## Implementation

```python
from grouped_gemm import ops as grouped_ops

# Instead of:
for expert_id in range(num_experts):
    output[expert_id] = torch.matmul(input[expert_id], expert_weights[expert_id])
    # ^ Sequential, poor GPU utilization

# Do this:
output = grouped_ops.gmm(
    inputs,          # List of input tensors
    expert_weights,  # List of weight tensors
    batch_sizes      # Tokens per expert
)
# ^ Single batched operation, full GPU utilization!
```

## Code Location

- **Wrapper:** `megatron/core/transformer/moe/grouped_gemm_util.py` lines 1-23

## Performance Impact

- **2-3x faster** vs sequential expert GEMMs
- **Critical for MoE performance**
- Enables training with 64+ experts efficiently

## When to Use

**Use when:**
- MoE models only
- Requires `grouped_gemm` library
- **Highly recommended** for MoE

**Installation:**

```bash
pip install grouped-gemm
```

## Related Optimizations

- [Expert Parallelism Communication](09_communication_expert_parallel.md) - Token routing for MoE
- [Expert Parallelism Strategy](35_parallelism_expert_parallel.md) - Overall MoE parallelism

## References

- [Grouped GEMM Library](https://github.com/fanshiqing/grouped_gemm)
- Paper: [Switch Transformers](https://arxiv.org/abs/2101.03961)
- [MoE Training at Scale](https://arxiv.org/abs/2202.08906)

