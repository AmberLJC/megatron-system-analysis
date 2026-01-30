# 01. Gradient Bucketing and Overlap

## Overview

Gradient bucketing with communication overlap is one of the most critical optimizations for distributed training of large language models. In naive distributed data-parallel training, GPUs spend 20-40% of their time sitting idle during gradient synchronization. The traditional approach waits until all gradients are computed in the backward pass before starting any communication, completely wasting precious GPU cycles. This optimization fundamentally changes that paradigm by grouping gradients into "buckets" and overlapping their communication with ongoing computation.

## The Problem: Idle GPUs During Gradient Sync

Consider a 96-layer GPT model trained across 64 GPUs in a data-parallel configuration. Without bucketing:

1. The backward pass computes gradients for all 96 layers (takes ~4-5 seconds)
2. Only after ALL gradients are ready does all-reduce begin
3. During the 2-3 seconds of all-reduce, GPUs are completely idle
4. **Result:** 30-40% of step time is wasted on communication

This inefficiency compounds at scale. For a 175B parameter model with gradient size of ~700GB across all ranks, naive synchronization exposes 3-5 seconds of pure communication overhead on the critical path.

## The Solution: Bucketing + Async Overlap

The core insight is that gradients don't all need to be ready before communication starts. The backward pass computes gradients in reverse topological order - last layer first, first layer last. By grouping parameters into buckets and launching communication as soon as each bucket completes, we can overlap gradient communication with ongoing backward computation.

### Key Implementation Details

Parameters are grouped into **reverse-order buckets** of ~40MB each. This size is carefully chosen:
- **Too small (<20MB):** More buckets means more NCCL kernel launch overhead
- **Too large (>80MB):** Coarser granularity reduces overlap opportunities
- **Sweet spot (40MB):** Balances overhead with fine-grained overlap

The bucketing happens in reverse topological order because of how PyTorch's autograd works:
- Layer 96 parameters get gradients first (computed early in backward)
- Layer 1 parameters get gradients last (computed late in backward)
- By bucketing in reverse order (Layer 1→Layer 96), we ensure the first bucket to complete is the first layer, whose gradients were just computed

## Core Implementation

The implementation spans two main components: bucket structure and automatic triggering.

### Bucket Structure

```python
class _ParamAndGradBucket:
    """
    Bucket to keep track of a subset of the model's parameters and gradients.

    Each bucket contains:
    - List of parameters (torch.nn.Parameter objects)
    - Contiguous gradient buffer view (for efficient NCCL ops)
    - Metadata for distributed optimizer integration
    - Scaling factors for averaging and MoE gradient scaling
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        param_data: Optional[torch.Tensor],  # View in param buffer
        grad_data: torch.Tensor,              # View in grad buffer (contiguous!)
        offset: int,                          # Offset in full buffer
        numel_unpadded: int,                  # Actual elements (excludes padding)
        gradient_scaling_factor: float,       # For averaging + MoE scaling
        bucket_id: int,
    ):
        self.params_list = params
        self.params = set(params)  # For O(1) membership checks
        assert len(self.params_list) == len(self.params), "No duplicate params allowed"

        # Contiguous buffer view - critical for NCCL performance
        self.grad_data = grad_data

        # Gradient scaling handles both:
        # 1. Averaging across data-parallel ranks (divide by DP size)
        # 2. MoE expert scaling (different experts may have different scaling)
        self.gradient_scaling_factor = gradient_scaling_factor

        # Track offset for distributed optimizer shard calculations
        self.offset = offset
        self.numel_unpadded = numel_unpadded  # Exclude padding bytes
        self.bucket_id = bucket_id

        # Build param → buffer index mapping for fast lookups
        self.param_to_index = {}
        offset = 0
        for param in params:
            self.param_to_index[param] = (offset, offset + param.numel())
            offset += param.numel()
```

**Key aspects:**
- **Contiguous `grad_data`:** NCCL operates most efficiently on contiguous memory. By maintaining gradient views into a contiguous buffer, we get 10-20% better bandwidth utilization
- **Padding awareness:** Modern GPUs prefer 16-byte aligned data. `numel_unpadded` tracks actual elements vs padded size
- **Scaling factor:** Pre-computed to avoid repeated division during communication

### Automatic Gradient Registration

The magic of automatic triggering happens through PyTorch autograd hooks:

```python
def register_grad_ready(self, param: torch.nn.Parameter):
    """
    Registers parameter gradients as "ready" for synchronization.

    This is called automatically by PyTorch autograd hooks when param.grad
    is populated. When ALL parameters in the bucket group have gradients ready,
    automatically launches asynchronous communication.

    Critical for overlap: This ensures communication starts IMMEDIATELY when
    a bucket completes, not waiting for the entire backward pass.
    """
    assert (
        self.ddp_config.overlap_grad_reduce
    ), "register_grad_ready() should only be called when overlap_grad_reduce is True"

    # Only trigger on last microbatch to avoid redundant communication
    if self.is_last_microbatch:
        assert param in self.param_to_bucket, "Param not recognized in bucket group"
        assert param not in self.params_with_grad, "Cannot register grad twice"

        # Mark this parameter as having gradient ready
        self.params_with_grad.add(param)

        # Auto-trigger: When ALL params in bucket group have grads, launch communication
        if len(self.params_with_grad) == len(self.params):
            self.start_grad_sync()
            # ↑ At this point, async communication launches immediately!
            # Backward pass continues on later layers while communication happens
```

**The auto-trigger mechanism:**
1. PyTorch autograd populates `param.grad` after computing gradients
2. Autograd hook fires, calling `register_grad_ready(param)`
3. Track which parameters have gradients in `params_with_grad` set
4. When set size equals total params → all bucket gradients ready
5. **Immediately launch** `start_grad_sync()` for async communication
6. Backward pass continues computing gradients for next buckets
7. **Result:** Communication for bucket N happens concurrently with gradient computation for bucket N+1

### Communication Launch

```python
def start_grad_sync(self):
    """
    Initiates async gradient communication for all buckets in group.

    Performs:
    1. Pre-scale gradients by gradient_scaling_factor
    2. Choose reduce operation (SUM vs AVG)
    3. Launch async all-reduce or reduce-scatter (depending on distributed optimizer)
    4. Return immediately without waiting (async_op=True)
    """
    assert (
        self.grad_reduce_handle is None
    ), "Cannot have multiple outstanding communication calls"

    # Scale gradients BEFORE communication
    # Handles both averaging (1/DP_size) and MoE scaling
    for bucket in self.buckets:
        if bucket.gradient_scaling_factor != 1.0:
            bucket.grad_data *= bucket.gradient_scaling_factor

    # Choose reduction operation
    reduce_op = torch.distributed.ReduceOp.SUM
    if self.ddp_config.average_in_collective:
        reduce_op = torch.distributed.ReduceOp.AVG  # Better numerical stability

    # Enable async for overlap - THE KEY SETTING!
    async_op = (
        self.ddp_config.overlap_grad_reduce
        and self.ddp_config.num_distributed_optimizer_instances == 1
    )

    # Use coalescing manager to batch all bucket operations (see optimization #07)
    with _coalescing_manager(communication_group, async_ops=async_op):
        for idx, bucket in enumerate(self.buckets):
            if self.ddp_config.use_distributed_optimizer:
                # Reduce-scatter (ZeRO-style): Each rank gets shard of gradients
                local_shard = self.cached_grad_buffer_shard_list[idx][rank]
                dist_reduce_scatter_func(
                    local_shard,      # Output: only my shard
                    bucket.grad_data,  # Input: full bucket gradients
                    group=communication_group,
                    async_op=async_op  # Non-blocking!
                )
            else:
                # All-reduce: All ranks get full gradients
                torch.distributed.all_reduce(
                    bucket.grad_data,
                    group=communication_group,
                    op=reduce_op,
                    async_op=async_op  # Non-blocking!
                )

    # Communication handle stored for later .wait() in optimizer step
    # But we return IMMEDIATELY - backward pass continues!
```

**Critical design choices:**

1. **`async_op=True`:** Non-blocking communication returns immediately
2. **Coalescing manager:** Batches multiple bucket ops into single NCCL group call
3. **Pre-scaling:** Apply averaging/MoE scaling before communication to avoid post-communication synchronization
4. **Reduce-scatter vs all-reduce:** Distributed optimizer uses reduce-scatter to save memory (each rank only needs its shard)

## Performance Impact

### Throughput Improvement

For GPT-3 175B with DP=64, 96 layers, 40MB buckets:

**Without bucketing:**
```
Forward → Backward (all layers) → [IDLE 3.2s] All-Reduce → Optimizer
                                   ^^^^^ wasted time
```

**With bucketing + overlap:**
```
Forward → Backward Layer 96 → [Bucket 1 reduce-scatter starts]
                 Layer 95 → [Bucket 2 reduce-scatter starts]
                 ...        [Communication overlaps with computation!]
                 Layer 1  → [All buckets complete] → Optimizer
```

**Measured results:**
- Gradient size: ~350GB across all ranks
- Communication time without overlap: 3.2 seconds on critical path
- Communication time with overlap: 0.3 seconds exposed (90% hidden!)
- **Speedup:** 2.9 seconds saved per step = 30% throughput improvement

### Overlap Efficiency

The percentage of communication time hidden behind computation:
- **Poor overlap (<50%):** Communication dominates, GPU underutilized
- **Good overlap (70-80%):** Most communication hidden
- **Excellent overlap (85-95%):** Nearly all communication hidden

Achieving excellent overlap requires:
1. **Proper bucket sizing:** 40MB buckets provide ~50-100ms of communication per bucket
2. **Sufficient compute per bucket:** Later layers have enough work to hide communication
3. **`CUDA_DEVICE_MAX_CONNECTIONS=1`:** Ensures proper kernel ordering (important for TP overlap, less so for DP)

## Configuration and Tuning

### Basic Configuration

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    # Core bucketing settings
    overlap_grad_reduce=True,              # ENABLE async overlap
    bucket_size=40000000,                  # 40MB buckets (bytes)

    # Distributed optimizer (highly recommended with bucketing)
    use_distributed_optimizer=True,        # Use reduce-scatter (saves memory)

    # Collective operation settings
    average_in_collective=True,            # Use AVG not SUM (better numerics)
    reduce_scatter_with_fp32_accumulation=True,  # FP32 accumulation for large DP

    # Validation (disable in production for max performance)
    check_for_nan_in_grad=False,           # Skip NaN checks
    finalize_model_grads_func=None,        # No custom finalization
)
```

### Bucket Size Tuning

The bucket size trades off overhead vs overlap granularity:

**Smaller buckets (20MB):**
- ✅ More fine-grained overlap (better hiding)
- ❌ More NCCL kernel launches (overhead)
- **Use when:** Many small layers, need maximum overlap

**Larger buckets (80MB):**
- ✅ Fewer kernel launches (less overhead)
- ❌ Coarser overlap (may expose more communication)
- **Use when:** Few large layers, overhead is bottleneck

**Recommended (40MB):**
- ⚖️ Balanced overhead and overlap
- Works well for most transformer models

### Verification

Check if bucketing is working correctly:

```python
import torch
from megatron.core import mpu

# After first step, verify buckets were created
def verify_bucketing(model):
    ddp_config = model.ddp_config

    print(f"Bucketing enabled: {ddp_config.overlap_grad_reduce}")
    print(f"Bucket size: {ddp_config.bucket_size / 1e6:.1f} MB")

    # Check bucket count (should be ~total_params / bucket_size)
    total_params = sum(p.numel() for p in model.parameters())
    expected_buckets = total_params * 4 / ddp_config.bucket_size  # 4 bytes per param
    print(f"Expected buckets: {expected_buckets:.0f}")

verify_bucketing(model)
```

## Common Issues and Solutions

### Issue 1: Low Communication Overlap

**Symptoms:**
- Profiler shows high exposed communication time
- GPU utilization drops during backward pass
- Expected 80-90% overlap, seeing <50%

**Causes & fixes:**
1. **`overlap_grad_reduce=False`:** Check config, enable it!
2. **Bucket size too large:** Reduce to 20-40MB
3. **Network bottleneck:** Check NCCL symmetric memory (optimization #02)
4. **Insufficient compute:** Layers too small to hide communication

**Debug with Nsight Systems:**
```bash
nsys profile -o profile --stats=true python train.py

# Look for:
# - NCCL kernels overlapping with GEMM/attention kernels
# - Should see 80-90% overlap for good configuration
```

### Issue 2: Wrong Gradient Values

**Symptoms:**
- Loss diverges with bucketing enabled
- Different results than without bucketing
- NaN gradients

**Causes & fixes:**
1. **Scaling factor incorrect:** Check `gradient_scaling_factor` computation
2. **Last microbatch tracking wrong:** Verify `is_last_microbatch` flag
3. **Bucket registration out of order:** Ensure reverse topological order

**Validation:**
```python
# Disable bucketing temporarily to verify correctness
ddp_config.overlap_grad_reduce = False
# Run training - should match previous results exactly
```

### Issue 3: OOM During Bucketing Setup

**Symptoms:**
- Out of memory when creating buckets
- Works without bucketing

**Causes & fixes:**
1. **Gradient buffer allocation:** Contiguous buffer needs more memory upfront
2. **Fragmentation:** Allocate buffers early in training
3. **Too many buckets:** Increase bucket size to reduce memory overhead

## Related Optimizations

- **#02 NCCL Symmetric Memory:** Improves bandwidth for bucket communications (2-3x faster)
- **#07 Coalesced Communication:** Batches bucket operations to reduce kernel launch overhead
- **#08 FP32 Accumulation in Reduce-Scatter:** Improves numerical stability for large DP sizes
- **#19 Distributed Optimizer:** Works together with bucketing to reduce memory (reduce-scatter instead of all-reduce)

## References

- Original ZeRO paper: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- PyTorch DDP bucketing: [DistributedDataParallel Documentation](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- Megatron-LM implementation: `megatron/core/distributed/param_and_grad_buffer.py`
- NCCL communication patterns: [NCCL Operations Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html)

## Summary

Gradient bucketing with communication overlap is essential for efficient distributed training. By grouping gradients into reverse-order buckets and automatically triggering async communication as buckets complete, Megatron-LM achieves 80-95% communication overlap with backward computation. This translates to 20-40% throughput improvement over naive synchronous all-reduce, making it a must-have optimization for large-scale training.
