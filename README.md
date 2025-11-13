# Megatron-LM System Optimizations

This directory contains detailed documentation of system-level optimizations in Megatron-LM. Each optimization is documented in its own file with:
- **Context:** Why the optimization exists
- **Implementation:** How it works at a high level
- **Core Code:** File paths and line numbers showing where it's implemented
- **Code Snippet:** Low-level implementation details to give you intuition
- **When to Use:** Scenarios where this optimization helps
- **Performance Impact:** Measured improvements

## Table of Contents

### Communication Optimizations (01-09, 20)
Communication is often the bottleneck in distributed training. These optimizations reduce overhead through overlapping, hierarchical strategies, and hardware-aware algorithms.

1. [Gradient Bucketing and Overlap](01_communication_gradient_bucketing.md) - 20-40% throughput improvement
2. [NCCL Symmetric Memory](02_communication_nccl_symmetric.md) - 20% speedup on NVLink systems
3. [Sequence Parallelism](03_communication_sequence_parallel.md) - 10-15% improvement + memory savings
4. [Tensor Parallelism Overlap](04_communication_tp_overlap.md) - Hides 80-95% of TP communication
5. [Hierarchical Communication](05_communication_hierarchical.md) - 5-15% improvement for complex parallelism
6. [P2P Communication Modes](06_communication_p2p_modes.md) - 5-15% pipeline throughput improvement
7. [Coalesced Communication](07_communication_coalesced.md) - Saves ~200-1000μs per step
8. [FP32 Gradient Accumulation](08_communication_fp32_accumulation.md) - Numerical stability at scale
9. [Expert Parallelism Communication](09_communication_expert_parallel.md) - 10-30% MoE training speedup
20. [MoE Batch-Level Overlapping](20_communication_moe_batch_overlap.md) - 15-25% MoE speedup via dense-expert overlap

### Parallelism Strategies (10-18)
Multi-dimensional parallelism enables scaling to thousands of GPUs by partitioning work across different dimensions.

10. [1F1B Pipeline Scheduling](10_parallelism_1f1b.md) - 2-4x better GPU utilization than GPipe
11. [Interleaved 1F1B](11_parallelism_interleaved_1f1b.md) - Half the bubble time of standard 1F1B
12. [Tensor Parallelism](12_parallelism_tensor_parallel.md) - Enables training models that don't fit on single GPU
13. [Data Parallelism](13_parallelism_data_parallel.md) - Linear scaling with communication overlap
14. [Expert Parallelism (MoE)](14_parallelism_expert_parallel.md) - 2-5x speedup vs dense models
15. [Context Parallelism (SSM)](15_parallelism_context_parallel.md) - Long contexts for Mamba/SSM models
16. [Gradient Sync in Pipeline Bubbles](16_parallelism_gradient_sync_bubbles.md) - "Free" gradient synchronization
17. [Multi-Dimensional Parallelism](17_parallelism_multidimensional.md) - Combining TP×PP×DP×EP strategies
18. [Sequence Parallelism](18_parallelism_sequence_parallel.md) - Memory-efficient sequence dimension partitioning

### Memory Optimizations (19, 22, 27-28)
Memory optimizations enable 2-3x larger models or batch sizes through careful memory management.

19. [Distributed Optimizer (ZeRO)](19_memory_distributed_optimizer.md) - O(DP) reduction in optimizer state
22. [Cached Bucket Shards](22_memory_cached_shards.md) - Saves ~200-1000μs per step
27. [Gradient Buffer with Padding](27_memory_gradient_buffer_padding.md) - 20-30% bandwidth improvement
28. [MXFP8 Buffer Sharing](28_memory_mxfp8_buffer_sharing.md) - Up to 50% of parameter buffer size

### Compute Optimizations (29-38)
Compute optimizations improve GPU efficiency through kernel fusion and hardware-specific acceleration.

29. [CUDA Graphs](29_compute_cuda_graphs.md) - 3-8% speedup, 30x kernel launch reduction
30. [Bias + Activation Fusion](30_compute_bias_activation_fusion.md) - 1.5-2x vs separate kernels
31. [Fused Softmax with Masking](31_compute_fused_softmax.md) - 2-3x vs unfused
32. [Fused Layer Normalization](32_compute_fused_layernorm.md) - 2-4x vs PyTorch
33. [Fused Cross Entropy](33_compute_fused_cross_entropy.md) - Saves 100GB+ for large vocab
34. [Fused RoPE](34_compute_fused_rope.md) - 1.5-2x for rotary embeddings
35. [Gradient Accumulation Fusion](35_compute_grad_accumulation_fusion.md) - 2-5% cumulative speedup
36. [Grouped GEMM for MoE](36_compute_grouped_gemm.md) - 2-3x vs sequential expert GEMMs
37. [FP8 Training Infrastructure](37_compute_fp8_training.md) - 1.5-2x training on H100+
38. [MXFP8 Blockwise Scaling](38_compute_mxfp8_scaling.md) - Better accuracy than standard FP8

## Quick Start

### For Training at Scale
Start with these essential optimizations:
1. **Gradient Bucketing** ([#01](01_communication_gradient_bucketing.md)) - Enable `overlap_grad_reduce=True`
2. **1F1B Pipeline** ([#10](10_parallelism_1f1b.md)) - Set `num_microbatches = 4-8 × pipeline_stages`
3. **Distributed Optimizer** ([#19](19_memory_distributed_optimizer.md)) - Enable `use_distributed_optimizer=True`
4. **Sequence Parallelism** ([#03](03_communication_sequence_parallel.md)) - Enable with tensor parallelism

### For Maximum Performance
Add these optimizations after basics:
1. **NCCL Symmetric Memory** ([#02](02_communication_nccl_symmetric.md)) - 20% speedup on NVLink
2. **TP Overlap** ([#04](04_communication_tp_overlap.md)) - Set `export CUDA_DEVICE_MAX_CONNECTIONS=1`
3. **CUDA Graphs** ([#29](29_compute_cuda_graphs.md)) - 3-8% speedup for static shapes
4. **Kernel Fusion** ([#30-38](30_compute_bias_activation_fusion.md)) - Install Apex/TransformerEngine

### For Memory-Constrained Training
Use these to fit larger models:
1. **Sequence Parallelism** ([#18](18_parallelism_sequence_parallel.md)) - Reduce activation memory
2. **Distributed Optimizer** ([#19](19_memory_distributed_optimizer.md)) - O(DP) reduction
3. **Pipeline Parallelism** ([#10](10_parallelism_1f1b.md)) - Split across stages
4. **Gradient Buffer Padding** ([#27](27_memory_gradient_buffer_padding.md)) - Improved bandwidth

## Performance Impact Summary

| Category | Total Potential Improvement |
|----------|---------------------------|
| Communication | 30-50% throughput gain through overlap and fast collectives |
| Parallelism | Enables scaling to 1000+ GPUs with 65-95% efficiency |
| Memory | 2-3x larger models or batch sizes |
| Compute | 20-40% higher MFU through fusion and FP8 |

**Combined:** Well-tuned Megatron-LM can achieve 50-65% MFU on large models (vs 20-30% naive).

## Configuration Template

```python
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.transformer import TransformerConfig

# Communication optimizations
ddp_config = DistributedDataParallelConfig(
    overlap_grad_reduce=True,              # Gradient bucketing (#01)
    use_distributed_optimizer=True,        # ZeRO-style sharding (#27)
    bucket_size=40000000,                  # 40MB buckets
    average_in_collective=True,
    nccl_ub=True,                          # NCCL symmetric memory (#02)
    disable_symmetric_registration=False,
)

# Parallelism configuration
tensor_model_parallel_size = 8             # Tensor parallelism (#12)
pipeline_model_parallel_size = 8           # Pipeline parallelism (#10)
data_parallel_size = 8                     # Data parallelism (#13)
virtual_pipeline_model_parallel_size = 2   # Interleaved 1F1B (#11)
num_microbatches = 64                      # 4-8 × pipeline_stages

# Memory optimizations
config = TransformerConfig(
    sequence_parallel=True,                # Sequence parallelism (#03)
    deallocate_pipeline_outputs=True,      # Activation deallocation (#20)
    recompute_granularity='selective',     # Activation checkpointing (#23)
    recompute_method='uniform',
    
    # Compute optimizations
    gradient_accumulation_fusion=True,     # Gradient fusion (#34)
    cuda_graph=True,                       # CUDA graphs (#28)
    fp8='hybrid',                          # FP8 training (#36)
)
```

## Environment Variables

```bash
# MANDATORY for overlap
export CUDA_DEVICE_MAX_CONNECTIONS=1       # TP overlap (#04)

# Optional for debugging
export NCCL_DEBUG=INFO                     # See NCCL details
export NCCL_NVLS_ENABLE=1                  # Force NVLS (testing)
```

## Troubleshooting

### Low Throughput
1. Check [Gradient Bucketing](#01) is enabled
2. Verify [TP Overlap](#04) environment variable
3. Review [1F1B Pipeline](#10) microbatch settings
4. Profile with Nsight Systems

### Out of Memory (OOM)
1. Enable [Distributed Optimizer](#19)
2. Enable [Sequence Parallelism](#03)
3. Use [Gradient Buffer Padding](#27)
4. Reduce microbatch size or increase pipeline stages

### Poor Scaling Efficiency
1. Verify [Communication Overlap](#01)
2. Check [NCCL Symmetric Memory](#02) on NVLink
3. Review [Multi-Dimensional Parallelism](#17) configuration
4. Profile communication with NCCL_DEBUG

## Documentation Format

Each optimization document follows this structure:

**Context** - Why this optimization exists and what problem it solves

**Implementation** - High-level explanation of how it works

**Core Code** - File paths and line numbers in the codebase

**Code Snippet** - Key implementation details with comments

**When to Use** - Scenarios where this helps (and when to skip)

**Performance Impact** - Measured improvements and trade-offs

## Contributing

When documenting new optimizations:
1. Follow the established format
2. Include code snippets from actual implementation
3. Provide specific file paths and line numbers
4. Include measured performance impacts
5. Add configuration examples
6. Document when NOT to use the optimization

## Related Resources

- [Megatron-LM Main README](../../README.md)
- [API Documentation](../../docs/source/index.rst)
- [Training Examples](../)
- [NVIDIA Blog: Megatron-LM](https://developer.nvidia.com/megatron-lm)

---

**Total:** 36 system optimizations documented with code snippets and performance measurements organized by:
- **Communication** (10 optimizations): 01-09, 20
- **Parallelism** (9 optimizations): 10-18
- **Memory** (4 optimizations): 19, 22, 27-28
- **Compute** (10 optimizations): 29-38
