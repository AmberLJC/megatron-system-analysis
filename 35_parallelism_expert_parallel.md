# Expert Parallelism (MoE)

## Context

MoE models have many experts. Not all experts needed for all tokens. Parallelize across expert dimension.

## Implementation

Each EP rank holds subset of experts. Tokens routed to appropriate rank via all-to-all.

See detailed implementation in [Expert Parallelism Communication](09_communication_expert_parallel.md).

## Performance Impact

- Enables training MoE with hundreds of experts
- **2-5x speedup** vs dense model of similar quality
- Communication: All-to-all for token routing

## When to Use

**MoE models only:**
- Many experts (≥ 8)
- Expert parallelism size = 2, 4, 8, etc.

**Configuration:**

```python
num_moe_experts = 64
expert_model_parallel_size = 8  # 8 experts per rank

# Total GPUs = TP × PP × EP × DP
# e.g., 4 × 4 × 8 × 4 = 512 GPUs
```

## Related Optimizations

- [Expert Parallelism Communication](09_communication_expert_parallel.md) - Implementation
- [Grouped GEMM](27_compute_grouped_gemm.md) - Efficient expert computation

## References

- Paper: [GShard: Scaling Giant Models](https://arxiv.org/abs/2006.16668)
- Paper: [Switch Transformers](https://arxiv.org/abs/2101.03961)

