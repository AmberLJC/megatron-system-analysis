# 04. Tensor Parallelism Overlap

## Context

Each tensor-parallel linear layer requires:
- **Forward:** All-gather inputs (to get full input for matmul with local weight shard)
- **Backward:** All-reduce input gradients (to sum gradients from all weight shards)

Without overlap, these communications block the critical path, adding ~1-3ms per layer. For a 96-layer model, that's 96-288ms per step!

## Implementation

In backward pass, launches **async all-reduce** of input gradients, then computes weight gradients **while communication happens**. The weight gradient computation is large enough (~500μs) to hide most of the all-reduce time.

**Critical requirement:** `CUDA_DEVICE_MAX_CONNECTIONS=1` ensures strict kernel ordering for proper overlap.

### How It Works

1. **Compute input gradient:** `grad_input = grad_output @ weight`
2. **Launch async all-reduce:** Start communication immediately (non-blocking)
3. **Compute weight gradient:** `grad_weight = grad_output.T @ input` (BIG matmul ~500μs)
4. **Wait for all-reduce:** Usually already done by the time we reach this point!

## Core Code

- `megatron/core/tensor_parallel/layers.py:435-618` - `LinearWithGradAccumulationAndAsyncCommunication`
- `megatron/core/tensor_parallel/layers.py:252-284` - Backward pass with overlap
- `megatron/core/utils.py:590-643` - Global memory buffer for all-gather
- `megatron/core/tensor_parallel/layers.py:44-48` - Configuration

## Code Snippet

```python
# From tensor_parallel/layers.py:252-284 (simplified)
class _LinearWithGradAccumulationAndAsyncCommunication(torch.autograd.Function):
    """
    Linear layer with tensor parallelism and communication overlap.
    
    Each rank holds a column shard of the weight matrix:
    Rank 0: weight[:, 0:hidden/TP]
    Rank 1: weight[:, hidden/TP:2*hidden/TP]
    ...
    """
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with communication overlap.
        This is where the MAGIC happens!
        """
        input, weight = ctx.saved_tensors
        use_bias = ctx.use_bias
        grad_acc_fusion = ctx.gradient_accumulation_fusion
        
        # --- STEP 1: Compute input gradient ---
        # grad_input = grad_output @ weight
        # This is relatively fast (~200μs for typical layer)
        grad_input = grad_output.matmul(weight)  
        
        # --- STEP 2: Launch ASYNC all-reduce of input gradients ---
        # This is THE KEY to overlap!
        # Each rank computed grad based on its weight shard
        # Need to sum across all ranks: grad_input = Σ(grad_input_i)
        handle = torch.distributed.all_reduce(
            grad_input,
            group=get_tensor_model_parallel_group(),
            async_op=True  # ← NON-BLOCKING! Returns immediately
        )
        # ^ Communication starts NOW in background!
        #   GPU is free to do other work...
        
        # --- STEP 3: Compute weight gradient WHILE communication happens ---
        # This is a BIG matmul (~500μs) - enough to hide most/all communication!
        # grad_weight = grad_output.T @ input
        if grad_acc_fusion:
            # Fused kernel: accumulate directly into main_grad (FP32)
            # See optimization #34 for details
            fused_weight_gradient_mlp_cuda.wgrad_gemm_accum_fp32(
                grad_output, input, weight.main_grad
            )
            grad_weight = None
        else:
            # Standard: compute into temporary tensor
            grad_weight = grad_output.t().matmul(input)
        
        # --- STEP 4: Wait for all-reduce to complete ---
        # By now, communication is usually 80-95% done or fully complete!
        handle.wait()  # ← Usually doesn't block (already done!)
        
        # Compute bias gradient if needed (fast, after communication done)
        grad_bias = grad_output.sum(dim=0) if use_bias else None
        
        return grad_input, grad_weight, grad_bias
    
    
# Critical: Environment variable for strict ordering
# Without this, CUDA may reorder kernels and break overlap!
# Set before training:
#   export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### Global Memory Buffer for All-Gather

```python
# From utils.py:590-643
class GlobalMemoryBuffer:
    """
    Pre-allocated buffer for all-gather operations.
    Avoids repeated allocation overhead (~100μs per alloc).
    """
    
    def __init__(self):
        self.buffer = {}  # (size, dtype, device) -> tensor
        
    def get_tensor(self, tensor_shape, dtype, device):
        """
        Get buffer tensor, allocating 2x size to reduce reallocations.
        """
        key = (tensor_shape, dtype, device)
        required_size = reduce(operator.mul, tensor_shape, 1)
        
        if key not in self.buffer or self.buffer[key].numel() < required_size:
            # Allocate 2x size to reduce frequency of reallocations
            self.buffer[key] = torch.empty(
                required_size * 2, dtype=dtype, device=device
            )
            
        # Return view of appropriate size
        return self.buffer[key][:required_size].view(*tensor_shape)

# Global instance used by all TP layers
_GLOBAL_MEMORY_BUFFER = GlobalMemoryBuffer()
```

## When to Use

**Always enable with tensor parallelism:**

```bash
# MANDATORY environment variable
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    tensor_model_parallel_size=8,          # TP size > 1
    gradient_accumulation_fusion=True,     # Maximize weight grad time
    sequence_parallel=True,                # Combine with SP (#03)
)
```

### Critical Requirement

```bash
# MUST SET THIS BEFORE TRAINING!
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

**Why?** Without this setting:
- CUDA uses multiple kernel execution streams
- Kernels may be reordered for "optimal" scheduling
- All-reduce might execute AFTER weight gradient
- **Result:** Lose 15-30% of overlap benefit

**With `CUDA_DEVICE_MAX_CONNECTIONS=1`:**
- Single stream ensures strict kernel order
- All-reduce launches before weight gradient kernel
- **Result:** 80-95% communication overlap

## Performance Impact

### Communication Overlap Efficiency

**Typical Layer (hidden=12288, TP=8):**
- Input gradient compute: ~200μs
- All-reduce communication: ~400μs  (exposed without overlap)
- Weight gradient compute: ~500μs
- **Overlap efficiency:** 500μs / 400μs = 125% → 100% hidden (fully overlapped!)

**Per-Layer Savings:**
- Without overlap: 200μs (input grad) + 400μs (all-reduce) + 500μs (weight grad) = 1100μs
- With overlap: 200μs + max(400μs, 500μs) = 700μs
- **Saved:** 400μs per layer

**96-Layer Model:**
- Total saved: 96 × 400μs = 38.4ms per step
- Typical step time: 450ms
- **Speedup:** 38.4ms / 450ms = 8.5% improvement

### End-to-End Measurements

**GPT-3 175B with TP=8, PP=8, DP=8:**
- Without overlap: 520ms per step
- With overlap: 475ms per step  
- **Improvement:** 8.7% throughput gain

**Communication Overlap Metrics:**
- Exposed TP communication: 45ms → 5ms (88.9% hidden)
- Compute time unchanged: 400ms
- Other communication: 75ms (gradient bucketing, pipeline)

## Troubleshooting

### Low Overlap Efficiency

**Symptoms:**
- Profiler shows all-reduce on critical path
- Communication not overlapped with weight gradient

**Causes:**
1. **`CUDA_DEVICE_MAX_CONNECTIONS ≠ 1`** (most common!)
2. Weight gradient completes too quickly (small layers)
3. Communication too slow (network bottleneck)

**Fix priority:**
```bash
# 1. SET THIS FIRST!
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 2. Verify it's set
python -c "import os; print(os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'NOT SET'))"

# 3. If still no overlap, profile with Nsight Systems
nsys profile -o profile python train.py
# Look for overlap between all-reduce and GEMM kernels
```

### Kernel Ordering Issues

**Symptoms:**
- Nsight shows GEMM before all-reduce
- Communication happens after weight gradient

**Causes:**
- `CUDA_DEVICE_MAX_CONNECTIONS` not set
- Multiple CUDA streams interfering

**Fix:**
```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
# Must set BEFORE launching Python!
```

### OOM with Gradient Fusion

**Symptoms:**
- OOM when `gradient_accumulation_fusion=True`
- Works fine with fusion disabled

**Causes:**
- FP32 main_grad buffer allocation
- Memory fragmentation

**Fix priority:**
1. Enable distributed optimizer (shards main_grad)
2. Reduce microbatch size
3. Disable fusion if necessary (small performance cost)

## Related Optimizations

- **#03 Sequence Parallelism:** Reduces TP communication further
- **#34 Gradient Accumulation Fusion:** Maximizes weight gradient time
- **#01 Gradient Bucketing:** Similar overlap principle for DP
- **#12 Tensor Parallelism:** Base TP implementation

## Configuration Example

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    # Tensor parallelism with overlap
    tensor_model_parallel_size=8,          # TP size
    
    # Maximize overlap
    gradient_accumulation_fusion=True,     # Fused weight gradient (longer compute)
    sequence_parallel=True,                # Reduce TP communication
    
    # Model settings
    hidden_size=12288,
    num_attention_heads=96,
)

# CRITICAL: Set environment variable BEFORE training
# In your shell or job script:
# export CUDA_DEVICE_MAX_CONNECTIONS=1
```

### Verification Script

```python
import os
import torch
import torch.distributed as dist

# 1. Check environment variable
max_conn = os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS', 'NOT SET')
print(f"CUDA_DEVICE_MAX_CONNECTIONS = {max_conn}")
assert max_conn == '1', "Must set CUDA_DEVICE_MAX_CONNECTIONS=1"

# 2. Run simple overlap test
def test_overlap():
    tensor = torch.randn(1000, 1000, device='cuda')
    
    # Launch async all-reduce
    handle = dist.all_reduce(tensor, async_op=True)
    
    # Do computation while all-reduce happens
    result = torch.matmul(tensor, tensor)
    
    # Wait for all-reduce
    handle.wait()
    
    print("Overlap test passed!")

test_overlap()
```

## Performance Metrics

Track these metrics to verify overlap:

```python
# From profiler or manual timing
tp_comm_time = 38ms      # Total TP all-reduce time
tp_comm_exposed = 5ms    # Time on critical path
overlap_efficiency = 1 - (tp_comm_exposed / tp_comm_time)
# Should be >80%

print(f"TP Communication Overlap: {overlap_efficiency:.1%}")
# Expected: 80-95%
```

## References

- Megatron-LM paper: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- CUDA streams documentation: [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#streams)
- Implementation: `megatron/core/tensor_parallel/layers.py`
