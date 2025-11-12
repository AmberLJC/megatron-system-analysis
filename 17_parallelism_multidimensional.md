# 17. Multi-Dimensional Parallelism

## Context

Large-scale training requires combining multiple parallelism strategies: Tensor (TP), Pipeline (PP), Data (DP), and Expert (EP) parallelism.

**3D Parallelism:** TP × PP × DP
**4D Parallelism:** TP × PP × DP × EP (for MoE)

Each dimension operates at different timescale and communication pattern.

## Implementation

Combine parallelism strategies with careful process group management:
- **TP:** High frequency, small messages, needs NVLink
- **PP:** Medium frequency, medium messages, works on IB
- **DP:** Low frequency, large messages, can use Ethernet
- **EP:** MoE only, medium frequency, variable messages

## Configuration Examples

### 3D Parallelism (512 GPUs, 175B model)

```python
# Split model across dimensions
tensor_model_parallel_size = 8       # NVLink domain (single node)
pipeline_model_parallel_size = 8     # 96 layers / 8 = 12 layers per stage
data_parallel_size = 8               # 8 × 8 × 8 = 512 GPUs

# Total GPUs = TP × PP × DP = 8 × 8 × 8 = 512
```

### 4D Parallelism (1024 GPUs, MoE model)

```python
# MoE with expert parallelism
tensor_model_parallel_size = 4
pipeline_model_parallel_size = 4
expert_model_parallel_size = 8       # 64 experts / 8 = 8 experts per rank
data_parallel_size = 8               # 4 × 4 × 8 × 8 = 1024 GPUs

# Total GPUs = TP × PP × EP × DP = 1024
```

## Communication Hierarchy

```
Level 1: Tensor Parallelism (TP)
├─ Frequency: Every layer (~μs between ops)
├─ Message size: ~MB per layer
├─ Bandwidth requirement: ~GB/s per layer
└─ Network: Requires NVLink (high bandwidth, low latency)

Level 2: Pipeline Parallelism (PP)
├─ Frequency: Every microbatch (~ms between ops)
├─ Message size: ~MB per stage  
├─ Bandwidth requirement: ~GB/s per stage
└─ Network: Works on InfiniBand (medium latency OK)

Level 3: Data Parallelism (DP)
├─ Frequency: Once per batch (~seconds between ops)
├─ Message size: ~GB total (but bucketed)
├─ Bandwidth requirement: ~GB/s to TB/s aggregate
└─ Network: Can use Ethernet (low frequency amortizes latency)

Level 4: Expert Parallelism (EP, MoE only)
├─ Frequency: Per MoE layer (~ms between ops)
├─ Message size: Variable (depends on routing)
├─ Bandwidth requirement: ~GB/s
└─ Network: Similar to Pipeline (IB recommended)
```

**Key insight:** Each level operates at different timescale. Overlaps at each level compound multiplicatively!

## Decision Tree

### Step 1: Does model fit on single GPU?
- **No** → Use Tensor Parallelism (TP=2, 4, or 8)
- **Yes** → Skip TP

### Step 2: Need to reduce memory further?
- **Yes** → Use Pipeline Parallelism (PP=2, 4, 8, etc.)
- **No** → Skip PP

### Step 3: Have more GPUs to utilize?
- **Yes** → Use Data Parallelism (DP = remaining GPUs / (TP × PP))
- **No** → Done

### Step 4: Using MoE model?
- **Yes** → Replace some DP with Expert Parallelism
- **No** → Done

### Step 5: Long sequences?
- **Yes** → Enable Sequence Parallelism (#03) or Context Parallelism (#15)
- **No** → Done

## Constraints

- **TP:** Limited by NVLink topology (usually ≤8)
- **PP:** Limited by model depth (can't have more stages than layers)
- **DP:** Limited by total GPUs and batch size requirements
- **EP:** Limited by number of experts in MoE

## Performance Tuning

### Pipeline Bubble Reduction
**Goal:** Minimize `(P-1)/(2MV)`

**Strategies:**
1. Increase microbatches: `num_microbatches = 4-8 × PP`
2. Use virtual pipeline: `virtual_pipeline_model_parallel_size = 2`
3. Balance pipeline stages

### Tensor Parallelism Overlap
**Goal:** Hide 80-95% of TP communication

**Strategies:**
1. Set `export CUDA_DEVICE_MAX_CONNECTIONS=1`
2. Enable `sequence_parallel=True`
3. Use `gradient_accumulation_fusion=True`

### Data Parallelism Efficiency
**Goal:** Hide gradient communication

**Strategies:**
1. Enable `overlap_grad_reduce=True`
2. Use `use_distributed_optimizer=True`
3. Tune `bucket_size`
4. Sync gradients in pipeline cooldown (#16)

## Scaling Laws

### Expected Efficiency by Scale

| GPUs | TP×PP×DP | Expected Efficiency | MFU |
|------|----------|---------------------|-----|
| 8 | 2×1×4 | 85-95% | 45-50% |
| 64 | 4×4×4 | 75-85% | 48-55% |
| 512 | 8×8×8 | 70-80% | 50-58% |
| 2048+ | 8×16×16+ | 65-75% | 52-60% |

Efficiency drops at larger scale due to:
- Communication overhead increases
- Load imbalance more visible
- Harder to overlap perfectly

But absolute throughput still increases!

## Best Practices

1. **Start with DP only** if model fits on single GPU
2. **Add TP** only if model doesn't fit (TP=2, 4, 8)
3. **Add PP** if still memory-constrained (PP=2, 4, 8)
4. **Always enable sequence parallelism** with TP
5. **Use virtual pipeline** with PP ≥ 4
6. **Set microbatches** to 4-8 × PP
7. **Profile and tune** bucket size, microbatch size
8. **Verify overlap** with Nsight Systems

## Communication Cost Comparison

| Parallelism | Frequency | Message Size | Bandwidth Need |
|-------------|-----------|--------------|----------------|
| TP | Every layer | ~MB | High (NVLink) |
| PP | Every microbatch | ~MB | Medium (IB OK) |
| DP | Once per batch | ~GB | Low (Ethernet OK) |
| EP | MoE layers | Variable | Medium |

This hierarchy explains why:
- TP requires NVLink (high frequency)
- PP works on InfiniBand (medium frequency)
- DP can work on Ethernet (low frequency)

Choose parallelism strategy based on network topology!

## Complete Configuration Example

```python
# 3D Parallelism: TP × PP × DP = 512 GPUs
config = {
    # Parallelism dimensions
    'tensor_model_parallel_size': 8,
    'pipeline_model_parallel_size': 8,
    'virtual_pipeline_model_parallel_size': 2,
    # data_parallel_size computed automatically: 512/(8×8) = 8
    
    # Pipeline scheduling
    'num_microbatches': 128,  # 8 PP × 8 DP × 2 virtual = 128
    
    # Communication optimizations
    'sequence_parallel': True,
    'overlap_p2p_comm': True,
    
    # Memory optimizations
    'deallocate_pipeline_outputs': True,
    'use_distributed_optimizer': True,
    'recompute_granularity': 'selective',
}

# DDP configuration
ddp_config = DistributedDataParallelConfig(
    overlap_grad_reduce=True,
    bucket_size=40000000,
    use_distributed_optimizer=True,
    nccl_ub=True,
)

# Environment
# export CUDA_DEVICE_MAX_CONNECTIONS=1
```

## References

- Megatron-LM paper: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- 3D parallelism blog: [NVIDIA Blog](https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/)
- ZeRO paper: [ZeRO](https://arxiv.org/abs/1910.02054)

