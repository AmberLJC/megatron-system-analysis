# 04. Tensor Parallelism Communication Overlap

## Context and Problem Statement

Tensor parallelism (TP) introduces unavoidable communication operations in both forward and backward passes. For each linear layer with TP enabled:

**Forward Pass:** All-gather input tensors so each rank can compute its portion of the output with its column-sharded weight matrix.

**Backward Pass:** All-reduce input gradients to combine contributions from all weight shards across the TP group.

Without overlap optimization, these operations block the critical path. For a 96-layer GPT-3 model with TP=8:
- Communication per layer (backward): ~400 microseconds
- Total communication overhead: 96 × 400 μs = 38.4 ms per training step
- With 450ms step time: 8.5% of training time wasted on blocked communication

The fundamental challenge is that the naive implementation serializes operations:
```
Sequential execution (NO overlap):
1. Compute input gradient (200 μs)
2. Wait...
3. All-reduce input gradient (400 μs) ← BLOCKS everything
4. Compute weight gradient (500 μs)
Total: 200 + 400 + 500 = 1100 μs
```

This optimization overlaps communication with computation by launching async operations:
```
Overlapped execution (WITH overlap):
1. Compute input gradient (200 μs)
2. Launch async all-reduce (starts immediately, non-blocking)
3. Compute weight gradient (500 μs) ← Happens while all-reduce runs!
4. Wait for all-reduce (usually already done)
Total: 200 + max(400, 500) = 700 μs
Saved: 400 μs per layer → 38ms per 96-layer model
```

## Implementation Architecture

### Critical Environment Variable

The entire optimization depends on a single environment variable:

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

**Why is this MANDATORY?**

CUDA normally uses multiple execution streams (default: 32) to maximize GPU utilization by reordering kernel launches. However, this reordering breaks communication-computation overlap:

**Without `CUDA_DEVICE_MAX_CONNECTIONS=1`:**
```
Stream 1: [input_grad matmul] [weight_grad matmul] ← Scheduled optimally for GPU
Stream 2:                     [all-reduce]         ← Scheduled after both matmuls!
Result: All-reduce happens AFTER weight gradient, NO OVERLAP
```

**With `CUDA_DEVICE_MAX_CONNECTIONS=1`:**
```
Single Stream: [input_grad] [all-reduce] [weight_grad] ← Strict order
Result: All-reduce starts immediately after input_grad, OVERLAPS with weight_grad
```

The single connection forces strict FIFO kernel ordering, ensuring the all-reduce kernel launches before the weight gradient kernel, enabling overlap.

### Core Autograd Implementation

The optimization is implemented in the backward pass of tensor-parallel linear layers:

```python
# From tensor_parallel/layers.py lines 435-619
class LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    Linear layer with tensor parallelism and async communication overlap.

    This autograd function implements the core overlap optimization.
    Each rank holds a column shard of the weight matrix.
    """

    @staticmethod
    @custom_fwd
    def forward(
        ctx,
        input,
        weight,
        bias,
        gradient_accumulation_fusion,
        allreduce_dgrad,
        sequence_parallel,
        grad_output_buffer,
        wgrad_deferral_limit,
        tp_group,
    ):
        """Forward pass: standard matmul, no special overlap logic."""
        # Save tensors for backward
        if gradient_accumulation_fusion and hasattr(weight, "main_grad"):
            main_grad = weight.main_grad
        else:
            main_grad = None

        ctx.save_for_backward(input, weight)
        ctx.main_grad = main_grad
        ctx.use_bias = bias is not None
        ctx.gradient_accumulation_fusion = gradient_accumulation_fusion
        ctx.allreduce_dgrad = allreduce_dgrad
        ctx.sequence_parallel = sequence_parallel
        ctx.tp_group = tp_group

        # Handle sequence-parallel input gathering
        if sequence_parallel:
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()

            # Use pre-allocated global buffer (avoids allocation overhead)
            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            dist_all_gather_func(all_gather_buffer, input, group=tp_group)
            total_input = all_gather_buffer
        else:
            total_input = input

        # Forward matmul: output = input @ weight^T
        output = torch.matmul(total_input, weight.t())
        if bias is not None:
            output = output + bias

        return output

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        """
        Backward pass with communication-computation overlap.

        THIS IS WHERE THE MAGIC HAPPENS!

        Key insight: Weight gradient computation (~500μs) is large enough
        to hide all-reduce communication (~400μs) when properly overlapped.
        """
        input, weight = ctx.saved_tensors
        main_grad = ctx.main_grad
        use_bias = ctx.use_bias
        sequence_parallel = ctx.sequence_parallel
        allreduce_dgrad = ctx.allreduce_dgrad
        gradient_accumulation_fusion = ctx.gradient_accumulation_fusion
        tp_group = ctx.tp_group
        handle = None

        # Restore main_grad pointer for gradient accumulation fusion
        if gradient_accumulation_fusion:
            weight.main_grad = main_grad

        # --- STEP 1: Compute input gradient ---
        # This is relatively fast: ~200 microseconds
        # grad_input = grad_output @ weight
        grad_input = grad_output.matmul(weight)
        # After this point, grad_input contains the local gradient
        # (computed using this rank's weight shard)

        # --- STEP 2: Launch ASYNC all-reduce ---
        if allreduce_dgrad and not sequence_parallel:
            # ALL-REDUCE PATH (standard TP)
            # Each rank computed grad_input using its weight shard
            # Need to sum all contributions: grad_input = Σ(grad_input_i)

            # Launch non-blocking all-reduce
            handle = torch.distributed.all_reduce(
                grad_input,
                group=tp_group,
                async_op=True  # ← CRITICAL: Returns immediately!
            )
            # At this point:
            # - All-reduce kernel is queued in CUDA stream
            # - Python execution continues immediately
            # - With CUDA_DEVICE_MAX_CONNECTIONS=1, this kernel
            #   is guaranteed to start before next kernel launch

        elif sequence_parallel:
            # SEQUENCE PARALLEL PATH (uses reduce-scatter)
            # See optimization #03 for details

            # All-gather input for weight gradient computation
            dim_size = list(input.size())
            dim_size[0] = dim_size[0] * tp_group.size()

            all_gather_buffer = get_global_memory_buffer().get_tensor(
                dim_size, input.dtype, "mpu"
            )
            handle = dist_all_gather_func(
                all_gather_buffer, input, group=tp_group, async_op=True
            )
            total_input = all_gather_buffer

            # Allocate output for reduce-scatter
            shard_size = list(input.size())
            sub_grad_input = torch.empty(
                shard_size,
                dtype=input.dtype,
                device=torch.cuda.current_device(),
                requires_grad=False,
            )

            # Launch async reduce-scatter
            handle_rs = dist_reduce_scatter_func(
                sub_grad_input,
                grad_input,
                group=tp_group,
                async_op=True,
            )

            # Wait for all-gather to complete before using total_input
            handle.wait()

        # --- STEP 3: Compute weight gradient WHILE all-reduce runs ---
        # This is a LARGE matmul: ~500 microseconds
        # grad_weight = grad_output^T @ input

        # By the time this computation finishes, all-reduce is usually
        # 80-95% complete or fully done!

        if gradient_accumulation_fusion:
            # Fused gradient accumulation: accumulate directly into FP32 buffer
            # See optimization #34 for details

            # Handle FSDP case
            if hasattr(weight, "__fsdp_param__"):
                weight.main_grad = weight.get_main_grad()
                torch.matmul(grad_output.t(), input, out=weight.main_grad)
            else:
                # Use fused CUDA kernel for accumulation
                if weight.main_grad.dtype == torch.float32:
                    # FP32 accumulation (most common)
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                        input, grad_output, weight.main_grad
                    )
                elif weight.main_grad.dtype in (torch.float16, torch.bfloat16):
                    # FP16/BF16 accumulation
                    fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp16(
                        input, grad_output, weight.main_grad
                    )
                else:
                    raise RuntimeError(
                        "Unsupported gradient type for gradient accumulation fusion"
                    )

            # Create dummy gradient for PyTorch's autograd engine
            # This prevents backward hooks from running in background threads
            if hasattr(weight, "grad_added_to_main_grad"):
                if getattr(weight, "zero_out_wgrad", False):
                    if HAVE_TE:
                        grad_weight = get_dummy_wgrad(
                            list(weight.main_grad.shape), input.dtype, zero=True
                        )
                    else:
                        grad_weight = torch.zeros(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                else:
                    if HAVE_TE:
                        grad_weight = get_dummy_wgrad(
                            list(weight.main_grad.shape), input.dtype
                        )
                    else:
                        grad_weight = torch.empty(
                            weight.main_grad.shape,
                            dtype=input.dtype,
                            device=torch.cuda.current_device(),
                            requires_grad=False,
                        )
                weight.grad_added_to_main_grad = True
            else:
                grad_weight = None
        else:
            # Standard weight gradient computation
            grad_weight = grad_output.t().matmul(input)

        # Compute bias gradient (fast, after communication)
        grad_bias = grad_output.sum(dim=0) if use_bias else None

        # --- STEP 4: Wait for all-reduce to complete ---
        # By this point, all-reduce is typically 80-95% done or fully complete!
        # If it's done, this is a no-op. If not, we wait the remaining ~20-50μs.

        if sequence_parallel:
            handle_rs.wait()
            return (sub_grad_input, grad_weight, grad_bias, None, None, None, None, None, None)

        if allreduce_dgrad:
            handle.wait()  # Usually doesn't block! Communication already done.

        return (grad_input, grad_weight, grad_bias, None, None, None, None, None, None)
```

### Global Memory Buffer for All-Gather

To reduce allocation overhead in sequence-parallel mode, Megatron uses a global memory buffer:

```python
# From parallel_state.py (conceptual implementation)
class GlobalMemoryBuffer:
    """
    Pre-allocated buffer pool for all-gather operations.

    Avoids repeated allocation overhead (~100μs per allocation).
    Critical for maintaining overlap efficiency.
    """

    def __init__(self):
        self.buffer = {}  # (size, dtype, device) -> tensor

    def get_tensor(self, tensor_shape, dtype, name):
        """
        Get buffer tensor, allocating 2x size to reduce reallocations.

        Args:
            tensor_shape: Shape of required tensor
            dtype: Data type
            name: Buffer name for tracking (e.g., "mpu")

        Returns:
            View into pre-allocated buffer
        """
        from functools import reduce
        import operator

        key = (tuple(tensor_shape), dtype, torch.cuda.current_device(), name)
        required_size = reduce(operator.mul, tensor_shape, 1)

        if key not in self.buffer or self.buffer[key].numel() < required_size:
            # Allocate 2x size to reduce reallocation frequency
            # Tradeoff: 2x memory for better performance
            self.buffer[key] = torch.empty(
                required_size * 2, dtype=dtype, device=torch.cuda.current_device()
            )

        # Return view of exact required size
        return self.buffer[key][:required_size].view(*tensor_shape)

# Global instance
_GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()

def get_global_memory_buffer():
    """Get the global memory buffer instance."""
    return _GLOBAL_MEMORY_BUFFER
```

## Overlap Efficiency Analysis

### Timing Breakdown

Measured on A100 GPU with TP=8, hidden_size=12288:

**Input Gradient Computation:**
```python
# grad_input = grad_output @ weight
# Typical size: [2048, 12288] @ [12288, 1536] (weight shard)
# Time: ~200 microseconds
```

**All-Reduce Communication:**
```python
# Size: [2048, 12288] tensor in BF16 = 50 MB
# Bandwidth with NVLS: ~300 GB/s
# Time: 50 MB / 300 GB/s ≈ 167 μs (theoretical)
# Actual measured: ~400 μs (includes latency, protocol overhead)
```

**Weight Gradient Computation:**
```python
# grad_weight = grad_output^T @ input
# Size: [1536, 2048] @ [2048, 12288] → [1536, 12288]
# Time: ~500 microseconds (large matmul)
```

### Overlap Efficiency Calculation

```python
# WITHOUT OVERLAP (sequential execution):
total_time = input_grad_time + allreduce_time + weight_grad_time
total_time = 200 + 400 + 500 = 1100 microseconds

# WITH OVERLAP (async execution):
total_time = input_grad_time + max(allreduce_time, weight_grad_time)
total_time = 200 + max(400, 500) = 700 microseconds

# Savings per layer
savings = 1100 - 700 = 400 microseconds

# For 96-layer model:
total_savings = 96 * 400 μs = 38.4 ms

# With typical 450ms step time:
speedup = 38.4 / 450 = 8.5% improvement
```

**Overlap efficiency:**
```python
# Ideal case: weight_grad_time >= allreduce_time
# → 100% overlap (communication completely hidden)

# Actual measured:
overlap_percent = min(weight_grad_time / allreduce_time, 1.0)
overlap_percent = min(500 / 400, 1.0) = 1.0 (100%)

# In practice, CUDA scheduling isn't perfect:
# Typical overlap efficiency: 80-95%
exposed_communication = allreduce_time * (1 - 0.85) = 60 μs
```

### Impact of CUDA_DEVICE_MAX_CONNECTIONS

Measured impact with different connection settings:

**CUDA_DEVICE_MAX_CONNECTIONS not set (default: 32):**
```
Timeline:
[input_grad: 200μs] [weight_grad: 500μs] [all-reduce: 400μs]
                                           ↑ Scheduled last!
Total: 1100μs (NO overlap)
Overlap efficiency: 0%
```

**CUDA_DEVICE_MAX_CONNECTIONS=1:**
```
Timeline:
[input_grad: 200μs] [all-reduce: 400μs]
                    [weight_grad: 500μs]
                    ↑ Overlaps with all-reduce!
Total: 700μs
Overlap efficiency: 85-95%
```

**Performance difference:** 400μs per layer = 57% speedup for this operation!

## Configuration and Usage

### Mandatory Setup

```bash
#!/bin/bash
# MUST be set BEFORE launching Python!
export CUDA_DEVICE_MAX_CONNECTIONS=1

# Verify it's set
python -c "import os; assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == '1'"

# Launch training
python train.py
```

### Model Configuration

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    # Enable tensor parallelism
    tensor_model_parallel_size=8,

    # Enable gradient accumulation fusion (maximizes weight grad time)
    gradient_accumulation_fusion=True,

    # Enable sequence parallelism (also benefits from overlap)
    sequence_parallel=True,

    # Model settings
    hidden_size=12288,
    num_attention_heads=96,
    num_layers=96,
)
```

### Verification Script

```python
import os
import torch
import torch.distributed as dist

def verify_overlap_configuration():
    """Verify overlap optimization is properly configured."""

    # 1. Check CUDA_DEVICE_MAX_CONNECTIONS
    max_conn = os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS')
    if max_conn != '1':
        print(f"WARNING: CUDA_DEVICE_MAX_CONNECTIONS={max_conn}, should be '1'")
        print("Overlap will be inefficient!")
        return False

    print("✓ CUDA_DEVICE_MAX_CONNECTIONS=1 (correct)")

    # 2. Test async all-reduce
    if not dist.is_initialized():
        dist.init_process_group(backend='nccl')

    tensor = torch.randn(1000, 1000, device='cuda')

    # Launch async all-reduce
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    handle = dist.all_reduce(tensor, async_op=True)

    # Do computation while all-reduce runs
    result = torch.matmul(tensor, tensor)

    # Wait for all-reduce
    handle.wait()
    end.record()

    torch.cuda.synchronize()
    elapsed = start.elapsed_time(end)

    print(f"✓ Async all-reduce test passed ({elapsed:.2f}ms)")
    return True

verify_overlap_configuration()
```

## Troubleshooting

### Problem: Low Overlap Efficiency

**Symptoms:**
- Profiler shows all-reduce on critical path
- Communication not overlapped with weight gradient

**Debug steps:**

1. **Check environment variable (MOST COMMON ISSUE):**
```bash
# Check if set
echo $CUDA_DEVICE_MAX_CONNECTIONS
# Should print: 1

# If not set, add to job script:
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

2. **Profile with Nsight Systems:**
```bash
nsys profile -o profile.qdrep python train.py

# Open in Nsight Systems GUI and check:
# - NCCL kernels should START before weight gradient GEMM
# - NCCL and GEMM kernels should OVERLAP on timeline
# - If NCCL comes after GEMM, CUDA_DEVICE_MAX_CONNECTIONS not working
```

3. **Add manual timing:**
```python
import torch.cuda

# In backward pass, add events:
start_ar = torch.cuda.Event(enable_timing=True)
end_ar = torch.cuda.Event(enable_timing=True)
start_wgrad = torch.cuda.Event(enable_timing=True)
end_wgrad = torch.cuda.Event(enable_timing=True)

# Time all-reduce
start_ar.record()
handle = dist.all_reduce(grad_input, async_op=True)
end_ar.record()

# Time weight gradient
start_wgrad.record()
grad_weight = compute_weight_gradient(...)
handle.wait()
end_wgrad.record()

torch.cuda.synchronize()

ar_time = start_ar.elapsed_time(end_ar)
wgrad_time = start_wgrad.elapsed_time(end_wgrad)

print(f"All-reduce: {ar_time:.2f}ms, Weight grad: {wgrad_time:.2f}ms")
print(f"Overlap efficiency: {min(wgrad_time/ar_time, 1.0)*100:.1f}%")
```

### Problem: Weight Gradient Too Fast

**Symptoms:**
- Weight gradient completes before all-reduce
- Overlap efficiency < 50%

**Causes:**
- Small hidden dimension
- Gradient accumulation fusion disabled

**Solutions:**

1. Enable gradient accumulation fusion:
```python
config = TransformerConfig(
    gradient_accumulation_fusion=True,  # Lengthens weight grad time
)
```

2. Increase hidden dimension (if model allows)

3. Accept lower overlap for small models (overhead < benefit)

### Problem: OOM with Gradient Fusion

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Cause:** FP32 main_grad buffer allocation

**Solutions:**

1. Enable distributed optimizer (shards main_grad):
```python
ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,  # Shards FP32 buffer
)
```

2. Reduce microbatch size

3. Disable fusion if necessary:
```python
config = TransformerConfig(
    gradient_accumulation_fusion=False,  # Small performance cost
)
```

## Performance Metrics

Track these metrics to verify overlap:

```python
# From profiler or manual timing
tp_comm_time = 38  # ms (total TP all-reduce time)
tp_comm_exposed = 5  # ms (time on critical path after overlap)

overlap_efficiency = 1 - (tp_comm_exposed / tp_comm_time)
# Should be > 80%

print(f"TP Communication Overlap: {overlap_efficiency:.1%}")
# Expected: 80-95%

# Per-layer metrics
layers = 96
exposed_per_layer = tp_comm_exposed / layers  # ~50 μs
total_per_layer = tp_comm_time / layers       # ~400 μs
layer_overlap = 1 - (exposed_per_layer / total_per_layer)
print(f"Per-layer overlap: {layer_overlap:.1%}")
```

## Related Optimizations

- **#03 Sequence Parallelism**: Sequence parallel also uses async reduce-scatter
- **#02 NCCL Symmetric Memory**: Faster communication enables better overlap
- **#34 Gradient Accumulation Fusion**: Maximizes weight gradient time
- **#01 Gradient Bucketing**: Similar overlap principle for data parallelism

## References

- Megatron-LM Paper: https://arxiv.org/abs/2104.04473
- CUDA Streams: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams
- Implementation: `megatron/core/tensor_parallel/layers.py`
- Nsight Systems: https://developer.nvidia.com/nsight-systems
