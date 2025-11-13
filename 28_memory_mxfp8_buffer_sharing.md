# 28. MXFP8 Parameter/Gradient Buffer Sharing

## Context

FP8 (8-bit floating point) training has emerged as a powerful technique for accelerating large language model training while maintaining model quality. The MXFP8 (Microscaling FP8) format from Transformer Engine provides particularly good numerical stability for training. However, FP8 training introduces a unique memory challenge: the framework must maintain both FP8 parameters (for computation) and higher-precision gradients (BF16 or FP32 for numerical stability during accumulation).

In standard distributed optimizer implementations with FP8:
- **FP8 parameters**: 175B × 1 byte = 175GB (for compute)
- **BF16 gradients**: 175B × 2 bytes = 350GB (for accumulation)
- **Total**: 525GB just for parameters and gradients

The key insight enabling buffer sharing is **temporal locality**: parameter and gradient buffers are never needed simultaneously:
1. **Forward pass**: Uses FP8 parameters (gradients unused)
2. **Parameter all-gather**: Gathers FP8 parameter shards (gradients unused)
3. **Backward pass**: Computes BF16/FP32 gradients (parameters no longer needed)
4. **Gradient reduce-scatter**: Communicates gradients (parameters no longer needed)
5. **Optimizer step**: Updates parameter shards based on gradients
6. **Repeat**: Next iteration starts with parameter all-gather again

Since parameters are only needed until the backward pass completes, and gradients are only needed after backward completes, they can share the same underlying memory buffer. This optimization can reduce memory usage by up to 50% for the parameter/gradient buffers.

## Implementation Overview

Megatron-LM implements buffer sharing within the `_ParamAndGradBuffer` class when MXFP8 parameters are detected. The implementation creates a single shared buffer that serves dual purposes at different phases of the training iteration.

### Shared Buffer Allocation

The shared buffer is allocated during buffer initialization when MXFP8 parameters are present:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:715-733

with mem_alloc_context():
    # For MXFP8 param: Create a shared buffer for param AG and grad RS
    # for memory efficiency. The buffer is mapped to weight gradients
    # whose dtype is either bf16 or FP32. It can be temporarily reused
    # by param AG.
    if (
        self.ddp_config.use_distributed_optimizer
        and any(is_mxfp8tensor(p) for p in params)
    ):
        # Allocate shared buffer sized for gradients (typically larger)
        self.shared_buffer = torch.zeros(
            self.numel,
            dtype=self.grad_dtype,  # BF16 or FP32
            device=torch.cuda.current_device(),
            requires_grad=False,
        )

        # For FP32 weight grads, only half of the buffer is used to store
        # params in bf16 (since FP32 uses 4 bytes, BF16 uses 2 bytes).
        if self.grad_dtype == torch.float32:
            # View first half as BF16 for parameters
            self.param_data = self.shared_buffer[: math.ceil(self.numel / 2)].view(
                torch.bfloat16
            )
        else:
            # grad_dtype is BF16, same size as param dtype
            self.param_data = self.shared_buffer

        # Gradient data shares the same buffer
        self.grad_data = self.shared_buffer
    else:
        # Standard approach: separate buffers
        if self.ddp_config.use_distributed_optimizer:
            self.param_data = torch.zeros(
                self.numel,
                dtype=self.param_dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )
        self.grad_data = torch.zeros(
            self.numel,
            dtype=self.grad_dtype,
            device=torch.cuda.current_device(),
            requires_grad=False,
        )
```

The key optimization is that `self.param_data` and `self.grad_data` reference the **same underlying storage** through different tensor views.

### Parameter Copy After All-Gather

After parameter all-gather completes, parameters must be copied from the shared buffer to the actual parameter tensors, then the shared buffer is zeroed for gradient accumulation:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:311-328

# For the mxfp8_param with "reuse_grad_buf_for_mxfp8_param_ag=True",
# we need to copy the param_data from the shared_param/grad_buffer
# to param.data after the param all-gather.
if (
    self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag
    and self.ddp_config.overlap_param_gather
):
    for bucket in self.buckets:
        for param in bucket.params:
            # Extract parameter's slice from shared buffer
            param_start, param_end = bucket.param_to_index[param]
            param_slice = bucket.param_data.view(-1)[param_start:param_end]

            # Copy to actual parameter tensor
            param.data.copy_(param_slice.view(param.data.shape))

        # All-gathered params are not needed after being copied to param.data.
        # Zero out the param buffer (shared with grad buffer) for gradient
        # accumulation. We cannot zero out the entire grad buffer because one
        # grad buffer may correspond to multiple param buffers. If we zero out
        # the entire grad buffer, it would clear the data of those param buffers
        # that have not yet completed AG.
        bucket.param_data.zero_()
```

This careful dance ensures:
1. All-gathered parameters are materialized in the shared buffer
2. Parameters are copied to their final locations
3. Shared buffer is cleared for gradient accumulation
4. No interference between multiple buckets completing at different times

### Configuration Requirements

Buffer sharing is only enabled when specific conditions are met:

```python
# From megatron/core/distributed/distributed_data_parallel_config.py:65-67, 147-148

reuse_grad_buf_for_mxfp8_param_ag: bool = False
"""If true, reuse the grad buffer for param AG when using mxfp8 recipe.
   Should be set to True only when fp8_recipe is mxfp8 and fp8_param_gather
   is True."""

def __post_init__(self):
    """Check the validity of the config."""
    if self.reuse_grad_buf_for_mxfp8_param_ag:
        assert self.fp8_param_gather, (
            "Reuse grad buffer only when keeping params in MXFP8."
        )
```

The configuration validation ensures that buffer sharing is only enabled when:
- FP8 parameter gather is enabled (parameters remain in FP8 during communication)
- The training recipe is MXFP8 (not standard FP8)

### Parameter Buffer Reuse Logic

The parameter mapping to shared buffer is conditional:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:756-771

for param in params[::-1]:
    param_start_index, param_end_index, bucket_id = self.param_index_map[param]

    # For MXFP8 param: we only need to map weight gradients to the buffer.
    if not self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
        # Standard mode: Assign param.data to appropriate segment of
        # self.param_data
        if self.param_data is not None:
            new_param_data = self._get(
                param.data.shape,
                param_start_index,
                buffer_type=BufferType.PARAM
            )
            if is_float8tensor(param):
                modify_underlying_storage(param, new_param_data)
            else:
                old_param_data = param.data
                param.data = new_param_data
                assert old_param_data._base is None
                # Copy tensor values (from initialization or checkpoint)
                param.data.detach().copy_(old_param_data)
                del old_param_data

    # Always map gradient to buffer (whether shared or not)
    param.main_grad = self._get(
        param.data.shape,
        param_start_index,
        buffer_type=BufferType.GRAD
    )
```

When buffer sharing is enabled (`reuse_grad_buf_for_mxfp8_param_ag=True`), parameters are NOT directly mapped to the shared buffer. Instead:
- Parameters remain in their original storage during compute
- Shared buffer is used temporarily during all-gather
- After all-gather, parameters are copied to their original storage
- Shared buffer is zeroed and reused for gradients

## Memory Savings Analysis

### Memory Layout: Separate Buffers (Standard)

**Standard Distributed Optimizer with FP8**:
```
GPU Memory Layout:
┌─────────────────────────────────────────────────┐
│ FP8 Parameters (175B × 1 byte = 175GB)          │
│   ┌─────────────────────────────────────────┐   │
│   │ Shard 1/8: 22GB per rank                │   │
│   │ (all-gathered to 175GB for forward)     │   │
│   └─────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│ BF16 Gradients (175B × 2 bytes = 350GB)         │
│   ┌─────────────────────────────────────────┐   │
│   │ Shard 1/8: 44GB per rank                │   │
│   └─────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│ FP32 Optimizer States (2 × 175B × 4 = 1.4TB)    │
│   ┌─────────────────────────────────────────┐   │
│   │ Shard 1/8: 175GB per rank               │   │
│   └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘

Total per rank: 22GB (params) + 44GB (grads) + 175GB (opt) = 241GB
```

### Memory Layout: Shared Buffer (Optimized)

**With MXFP8 Buffer Sharing**:
```
GPU Memory Layout:
┌─────────────────────────────────────────────────┐
│ Shared Buffer (max of param/grad size = 350GB)  │
│   ┌─────────────────────────────────────────┐   │
│   │ Phase 1 (forward + AG): BF16 params     │   │
│   │   Shard 1/8: 22GB per rank             │   │
│   │   (after AG: 175GB temporarily)         │   │
│   ├─────────────────────────────────────────┤   │
│   │ Phase 2 (backward): BF16 gradients      │   │
│   │   Shard 1/8: 44GB per rank             │   │
│   └─────────────────────────────────────────┘   │
├─────────────────────────────────────────────────┤
│ FP32 Optimizer States (2 × 175B × 4 = 1.4TB)    │
│   ┌─────────────────────────────────────────┐   │
│   │ Shard 1/8: 175GB per rank               │   │
│   └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘

Total per rank: 44GB (shared) + 175GB (opt) = 219GB

Memory saved: 22GB per rank (10% reduction)
```

### Savings for Different Model Sizes

| Model Size | Param Size (FP8) | Grad Size (BF16) | Standard | With Sharing | Savings per Rank (DP=8) |
|------------|------------------|------------------|----------|--------------|-------------------------|
| 7B         | 7GB              | 14GB             | 21GB     | 14GB         | 7GB (33%)               |
| 13B        | 13GB             | 26GB             | 39GB     | 26GB         | 13GB (33%)              |
| 70B        | 70GB             | 140GB            | 210GB    | 140GB        | 70GB (33%)              |
| 175B       | 175GB            | 350GB            | 525GB    | 350GB        | 175GB (33%)             |
| 540B       | 540GB            | 1.08TB           | 1.62TB   | 1.08TB       | 540GB (33%)             |

With distributed optimizer (DP=8), savings per rank are divided by 8:

| Model Size | Savings per Rank (DP=8) |
|------------|-------------------------|
| 7B         | 0.875GB (11%)           |
| 70B        | 8.75GB (11%)            |
| 175B       | 21.875GB (11%)          |
| 540B       | 67.5GB (11%)            |

### Peak Memory Analysis

A critical consideration is **peak memory usage** during transitions:

**Peak Memory Occurs During Parameter All-Gather**:
```
Timeline:
1. Start: Shared buffer contains parameter shard (22GB for 175B)
2. All-gather: Materialize full parameters (175GB temporarily)
3. Copy: Copy parameters to original storage (175GB briefly doubled!)
4. Zero: Clear shared buffer
5. Backward: Accumulate gradients in shared buffer (44GB)

Peak = max(175GB during AG, 175GB during copy) = 175GB
```

The peak memory during all-gather is **temporarily higher** than the shard size, but still lower than separate buffers would require throughout the entire iteration.

## Configuration

### Enable MXFP8 Buffer Sharing

```python
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core import ModelParallelConfig

# Enable distributed optimizer
ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,
    overlap_param_gather=True,
    overlap_grad_reduce=True,

    # Enable FP8 parameter all-gather
    fp8_param_gather=True,

    # Enable buffer sharing (MXFP8 only!)
    reuse_grad_buf_for_mxfp8_param_ag=True,
)

# Configure FP8 training with MXFP8 recipe
config = ModelParallelConfig(
    fp8='hybrid',  # Enable FP8 training
    # Additional FP8 configuration...
)
```

### Requirements and Constraints

Buffer sharing requires ALL of the following:
1. **Distributed optimizer enabled**: `use_distributed_optimizer=True`
2. **FP8 parameter gather enabled**: `fp8_param_gather=True`
3. **MXFP8 recipe**: FP8 training must use the MXFP8 format
4. **Overlap parameter gather**: `overlap_param_gather=True` (for copy-back logic)

If any requirement is not met, the implementation automatically falls back to separate buffers.

## Performance Considerations

### Additional Copy Overhead

Buffer sharing introduces an additional copy operation after parameter all-gather:

```python
# Copy cost analysis
param_size = 175GB  # Full parameters after AG
copy_bandwidth = 1500 GB/s  # H100 internal bandwidth

copy_time = 175GB / 1500 GB/s ≈ 117ms
```

This copy time must be compared against the memory savings benefit:
- **Pro**: 22GB memory saved per rank (enables larger batch sizes or models)
- **Con**: 117ms additional latency per iteration

### When Buffer Sharing Helps

**Enable buffer sharing when**:
- Memory is constrained (close to OOM)
- Can increase batch size with freed memory → higher throughput offsets copy cost
- Training very large models (540B+) where memory is critical

**Skip buffer sharing when**:
- Memory is ample (not close to OOM)
- Copy overhead would reduce throughput
- Training smaller models where memory is not a bottleneck

### Overlap Opportunities

The copy operation after all-gather **cannot be overlapped** with compute because:
1. Parameters must be copied before forward pass
2. Forward pass requires the copied parameters
3. Copy is on the critical path

However, the zeroing operation can be overlapped with the next iteration's all-gather.

## Implementation Details

### Buffer Sharing with FP32 Gradients

When gradients use FP32 (4 bytes) but parameters use BF16 (2 bytes), special handling is needed:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:726-732

if self.grad_dtype == torch.float32:
    # For FP32 weight grads, only half of the buffer is used to store
    # params in bf16 (since FP32 uses 4 bytes, BF16 uses 2 bytes).
    self.param_data = self.shared_buffer[: math.ceil(self.numel / 2)].view(
        torch.bfloat16
    )
else:
    # BF16 gradients, same size as BF16 params
    self.param_data = self.shared_buffer
```

This ensures that:
- FP32 gradient buffer (350GB for 175B) can store BF16 parameters (175GB) in first half
- No additional memory needed beyond the gradient buffer size
- Second half of buffer remains available for gradient accumulation

### Bucket-Level Zeroing

The shared buffer is zeroed at the **bucket level**, not buffer level:

```python
# From megatron/core/distributed/param_and_grad_buffer.py:328

bucket.param_data.zero_()
```

This fine-grained zeroing is critical because:
1. Different buckets complete all-gather at different times (when overlapping)
2. Zeroing entire buffer would corrupt buckets still being gathered
3. Bucket-level zeroing ensures correctness with asynchronous communication

### Safety Assertions

The implementation includes validation to prevent misuse:

```python
# From megatron/core/distributed/distributed_data_parallel_config.py:147-148

if self.reuse_grad_buf_for_mxfp8_param_ag:
    assert self.fp8_param_gather, (
        "Reuse grad buffer only when keeping params in MXFP8."
    )
```

This prevents enabling buffer sharing without FP8 parameter gather, which would cause correctness issues.

## Troubleshooting

### Issue: OOM After Enabling Buffer Sharing

**Symptoms**: Out of memory despite buffer sharing being enabled

**Possible Causes**:
1. Peak memory during all-gather exceeds available memory
2. Copy operation requires temporary memory
3. Buffer sharing not actually enabled (configuration error)

**Diagnosis**:
```python
# Verify buffer sharing is active
for buffer in model.per_model_buffers.values():
    for buf in buffer:
        if hasattr(buf, 'shared_buffer'):
            print(f"Buffer sharing ACTIVE: shared_buffer size = {buf.shared_buffer.numel()}")
        else:
            print("Buffer sharing INACTIVE: separate param/grad buffers")

# Check peak memory usage
import torch
torch.cuda.reset_peak_memory_stats()
# ... run one iteration ...
peak_mb = torch.cuda.max_memory_allocated() / 1e6
print(f"Peak memory: {peak_mb:.2f} MB")
```

**Solutions**:
1. Reduce microbatch size to lower peak memory
2. Verify all configuration requirements are met
3. Check for memory leaks elsewhere in the model

### Issue: Slower Training with Buffer Sharing

**Symptoms**: Iteration time increases after enabling buffer sharing

**Cause**: Copy overhead dominates, not enough benefit from memory savings

**Diagnosis**:
```python
# Profile copy time
import time

start = time.time()
# ... parameter copy operation ...
copy_time_ms = (time.time() - start) * 1000

# Compare against iteration time
print(f"Copy time: {copy_time_ms:.2f}ms")
print(f"Iteration time: {iter_time_ms:.2f}ms")
print(f"Copy overhead: {copy_time_ms / iter_time_ms * 100:.2f}%")
```

**Solution**: Disable buffer sharing if overhead > 5% and memory is not constrained:
```python
ddp_config.reuse_grad_buf_for_mxfp8_param_ag = False
```

### Issue: Incorrect Gradients with Buffer Sharing

**Symptoms**: Training diverges or produces NaN values

**Possible Causes**:
1. Buffer not properly zeroed between phases
2. Buckets interfering with each other
3. Overlap timing issues

**Diagnosis**:
```python
# Check gradient buffer contents after parameter copy
for bucket in buffer.buckets:
    grad_norm = bucket.grad_data.norm(p=2)
    print(f"Bucket {bucket.bucket_id} grad norm after AG: {grad_norm}")
    # Should be 0 or very small immediately after AG
```

**Solution**: Ensure overlap is properly configured:
```python
ddp_config.overlap_param_gather = True  # Required for correct copy-back timing
```

## Related Optimizations

- **#19 Distributed Optimizer**: Shards parameters and gradients across ranks
- **#22 Cached Bucket Shards**: Caches shard views used during AG/RS
- **#37 FP8 Training**: Enables FP8 compute that makes buffer sharing possible
- **#38 MXFP8 Scaling**: Implements the MXFP8 format that provides numerical stability

## Implementation Files

- **Primary Implementation**: `megatron/core/distributed/param_and_grad_buffer.py:715-733, 311-328, 756-771`
- **Configuration**: `megatron/core/distributed/distributed_data_parallel_config.py:65-67, 147-148`
- **FP8 Utilities**: `megatron/core/fp8_utils.py`
- **Tests**: `tests/unit_tests/test_fp8_param.py`

## Design Trade-offs

### Pros
- **Memory Savings**: 11-33% reduction in parameter/gradient buffer size
- **Enables Larger Models**: Freed memory can be used for larger batch sizes or model sizes
- **Automatic Fallback**: Safely falls back to separate buffers if requirements not met

### Cons
- **Copy Overhead**: Additional parameter copy after all-gather (100-200ms for large models)
- **Complexity**: More complex buffer management logic
- **Peak Memory**: Temporarily higher peak during all-gather + copy phase
- **FP8-Only**: Only benefits FP8 training, not BF16/FP32 training

### Recommendation

Enable buffer sharing when:
```python
if (
    model_size_gb > available_memory_gb * 0.85  # Close to memory limit
    and using_fp8_training
    and can_tolerate_copy_overhead
):
    ddp_config.reuse_grad_buf_for_mxfp8_param_ag = True
```

Otherwise, prefer separate buffers for simplicity and lower latency.

## References

- **MXFP8 Format**: [Microscaling Data Formats](https://arxiv.org/abs/2310.10537)
- **Transformer Engine**: [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine)
- **FP8 Training**: [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433)
- **Memory Management in PyTorch**: [PyTorch Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
