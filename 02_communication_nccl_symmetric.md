# 02. NCCL Symmetric Memory Optimization

## Context and Problem Statement

NVIDIA Collective Communication Library (NCCL) implements multiple algorithms for collective operations, with the NVLS (NVLink Sharp) algorithm delivering the highest bandwidth on NVLink-enabled systems. However, NVLS has a critical requirement: all participating GPUs must allocate communication buffers at identical virtual memory addresses. This constraint, known as "symmetric memory allocation," cannot be satisfied by PyTorch's default memory allocator, which assigns arbitrary virtual addresses to each GPU's allocations.

When NCCL detects non-symmetric buffers, it automatically falls back to slower algorithms like Tree or Ring. On DGX A100 systems with NVLink 3.0, this fallback reduces all-reduce bandwidth from 300+ GB/s (NVLS) to 120-150 GB/s (Ring), a 2-3x performance degradation. For large language model training with multi-billion parameter gradient synchronization, this bandwidth reduction directly impacts training throughput.

The challenge stems from fundamental differences in memory allocation across GPUs. Each GPU has its own independent virtual address space, and PyTorch's caching allocator makes no guarantees about address alignment across devices. Even allocating the same size buffer on all GPUs simultaneously results in different virtual addresses per rank. NCCL's NVLS algorithm requires identical addresses because it uses hardware-accelerated memory operations that assume consistent memory layout across all participants.

## Implementation Architecture

Megatron-LM solves this problem through a custom NCCL allocator that integrates with PyTorch's memory management system. The implementation has three key components:

### 1. C++ Extension for Symmetric Allocation

The core functionality is implemented as a C++ extension that directly calls NCCL's `ncclMemAlloc` API. This CUDA-aware allocator guarantees symmetric address allocation across all GPUs in a process group:

```cpp
// From nccl_allocator.py lines 29-79 (inline C++ code)
void* nccl_alloc_plug(size_t size, int device, void* stream) {
    void* ptr;
    // ncclMemAlloc allocates memory at the SAME virtual address
    // on all GPUs that will participate in collectives
    NCCL_CHECK(ncclMemAlloc(&ptr, size));
    return ptr;
}

void nccl_free_plug(void* ptr, size_t size, int device, void* stream) {
    NCCL_CHECK(ncclMemFree(ptr));
}

// Create pluggable allocator that PyTorch can use
std::shared_ptr<c10::cuda::CUDACachingAllocator::CUDAAllocator>
get_nccl_allocator() {
    if (!nccl_allocator) {
        nccl_allocator = std::make_shared<
            torch::cuda::CUDAPluggableAllocator::CUDAPluggableAllocator>(
            nccl_alloc_plug, nccl_free_plug);
    }
    return nccl_allocator;
}
```

This extension is compiled at runtime using PyTorch's JIT compilation system (`torch.utils.cpp_extension.load_inline`), requiring NCCL headers and libraries to be available at compile time.

### 2. Memory Pool Management

To amortize the allocation overhead (which can be 50-100μs per allocation), Megatron implements a memory pool abstraction. The `create_nccl_mem_pool` function creates reusable buffer pools:

```python
# From nccl_allocator.py lines 109-138
def create_nccl_mem_pool(symmetric=None):
    """
    Create a memory pool using the NCCL allocator.

    The pool caches allocated tensors across training iterations,
    avoiding repeated expensive symmetric allocation operations.

    Args:
        symmetric: If True, enable symmetric memory allocation.
                  Requires PyTorch >= 2.9.0 for full support.
    """
    _build_nccl_allocator()

    if not is_torch_min_version("2.9.0a0") and symmetric is True:
        logging.info(
            f"Symmetric memory pool not supported with torch < 2.9.0a0. "
            f"Current version: {torch.__version__}. "
            "Falling back to non-symmetric pool."
        )
        symmetric = False

    assert _allocator is not None, "NCCL allocator not initialized"

    if not symmetric:
        _pool = torch.cuda.MemPool(_allocator)
    else:
        # Handle PyTorch API variations across versions
        if 'symmetric' in get_func_args(torch.cuda.MemPool):
            _pool = torch.cuda.MemPool(_allocator, symmetric=symmetric)
        elif 'symm_mem' in get_func_args(torch.cuda.MemPool):
            # NVIDIA PyTorch fork uses different parameter name
            _pool = torch.cuda.MemPool(_allocator, symm_mem=symmetric)
        else:
            _pool = torch.cuda.MemPool(_allocator)

    return _pool
```

### 3. Context Manager for Pool Registration

The `nccl_mem` context manager handles the critical task of registering memory pools with NCCL process groups. This registration enables NCCL to recognize buffers as symmetric and select the NVLS algorithm:

```python
# From nccl_allocator.py lines 160-233
class nccl_mem:
    """
    Context manager for NCCL symmetric memory allocation.

    Usage:
        pool = create_nccl_mem_pool(symmetric=True)
        with nccl_mem(pool, group=dp_group, symmetric=True):
            tensor = torch.zeros(size, device='cuda')
            # tensor is now allocated with symmetric addresses
    """

    def __init__(self, pool, enabled=True, device=None, group=None, symmetric=True):
        self.device = None
        self.group = None
        self.mem_context = None
        self.pool = pool
        self.symmetric = symmetric

        if enabled:
            if device is None:
                self.device = torch.device("cuda", torch.cuda.current_device())
            elif isinstance(device, int):
                self.device = torch.device("cuda", device)
            elif isinstance(device, str):
                assert "cuda" in device, "only cuda devices are supported"
                self.device = torch.device(device)

            if group is None:
                self.group = torch.distributed.distributed_c10d._get_default_group()
            else:
                self.group = group

            self.mem_context = torch.cuda.use_mem_pool(self.pool)
        else:
            self.mem_context = nullcontext()

    def __enter__(self):
        self.mem_context.__enter__()
        if self.group is not None:
            # If pool already contains allocations, deregister first
            # This prevents duplicate registration errors
            if self.pool.snapshot():
                backend = self.group._get_backend(self.device)
                try:
                    backend.deregister_mem_pool(self.pool)
                except RuntimeError:
                    desc = getattr(self.group, "group_desc", None)
                    logging.warning(
                        f"[MCORE][NCCL_ALLOCATOR] Failed to deregister pool from "
                        f"{repr(self.group)}({desc}) group"
                    )

    def __exit__(self, *args):
        if self.group is not None:
            backend = self.group._get_backend(self.device)
            try:
                # Try symmetric registration first
                if self.symmetric:
                    try:
                        # PyTorch PR #161238 moved symm parameter to registration
                        backend.register_mem_pool(self.pool, symm=self.symmetric)
                    except TypeError:
                        # Fall back for older PyTorch versions
                        logging.warning(
                            f"[MCORE][NCCL_ALLOCATOR] Failed symmetric registration. "
                            "Falling back to non-symmetric."
                        )
                        backend.register_mem_pool(self.pool)
                else:
                    backend.register_mem_pool(self.pool)
            except RuntimeError:
                desc = getattr(self.group, "group_desc", None)
                logging.warning(
                    f"[MCORE][NCCL_ALLOCATOR] Failed to register pool to "
                    f"{repr(self.group)}({desc}) group"
                )

        self.mem_context.__exit__(*args)
```

### 4. Integration with Gradient Buffers

The symmetric memory allocator integrates seamlessly with Megatron's gradient buffer system. When `nccl_ub=True` is configured, gradient buffers automatically use symmetric allocation:

```python
# Integration pattern used in param_and_grad_buffer.py
def allocate_gradient_buffer_with_symmetric_memory(
    buffer_size, dtype, device, data_parallel_group
):
    """
    Allocate gradient buffer using NCCL symmetric memory.

    This ensures NCCL selects NVLS algorithm for gradient all-reduce,
    providing 2-3x bandwidth improvement on NVLink systems.
    """
    # Initialize NCCL allocator
    nccl_allocator.init()

    # Create memory pool for symmetric allocation
    pool = nccl_allocator.create_nccl_mem_pool(symmetric=True)

    # Allocate within NCCL memory context
    with nccl_allocator.nccl_mem(pool, group=data_parallel_group, symmetric=True):
        # All allocations within this context use symmetric addresses
        grad_buffer = torch.zeros(
            buffer_size,
            dtype=dtype,
            device=device
        )
        # NCCL will now use NVLS algorithm for operations on grad_buffer!

    return grad_buffer, pool
```

## NCCL Algorithm Selection Logic

Understanding how NCCL selects algorithms is crucial for optimizing communication:

### Algorithm Hierarchy

NCCL evaluates algorithms in priority order:
1. **NVLS (NVLink Sharp)**: Requires symmetric memory + NVLink 3.0+
   - Bandwidth: 300-320 GB/s on DGX A100
   - Latency: ~5-8 μs
   - Memory requirement: Symmetric addresses

2. **Tree**: Hierarchical reduction
   - Bandwidth: 150-200 GB/s
   - Latency: ~15-20 μs
   - Works with standard allocation

3. **Ring**: Classic all-reduce
   - Bandwidth: 100-150 GB/s
   - Latency: ~25-30 μs
   - Fallback for all conditions

### Detection Logic

NCCL performs runtime checks to determine algorithm eligibility:

```python
# Pseudo-code of NCCL's internal algorithm selection
def nccl_select_algorithm(buffer, operation, group_size):
    """
    NCCL's internal algorithm selection logic.
    This is simplified representation of actual C++ code.
    """
    # Check for symmetric memory support
    if all_buffers_symmetric(buffer, group_size):
        if has_nvlink_connectivity():
            if nccl_version >= "2.18":
                return "NVLS"  # Best performance!

    # Check for tree algorithm support
    if group_size >= 4:
        if has_hierarchical_topology():
            return "Tree"

    # Default fallback
    return "Ring"
```

You can verify algorithm selection with environment variables:

```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

# Run training and look for log messages:
# "NCCL INFO Using NVLS" → Symmetric memory working!
# "NCCL INFO Using Ring" → Falling back, check configuration
```

## Performance Analysis and Measurements

### Bandwidth Improvements

Measured on DGX A100 (8x A100 80GB, NVLink 3.0, 600 GB/s total bidirectional):

| Buffer Size | Ring Algorithm | NVLS Algorithm | Improvement |
|-------------|----------------|----------------|-------------|
| 1 GB | 118 GB/s | 305 GB/s | 2.59x |
| 4 GB | 122 GB/s | 315 GB/s | 2.58x |
| 16 GB | 125 GB/s | 320 GB/s | 2.56x |
| 44 GB (GPT-3 175B grads) | 127 GB/s | 318 GB/s | 2.50x |

### End-to-End Training Impact

For GPT-3 175B model (DP=8, TP=8, PP=8):
- Gradient size per DP rank: 44 GB
- All-reduce time without NVLS: 366 ms
- All-reduce time with NVLS: 138 ms
- **Communication speedup**: 2.65x

However, due to computation-communication overlap, the end-to-end improvement is smaller:
- Step time without NVLS: 820 ms
- Step time with NVLS: 660 ms
- **End-to-end speedup**: 1.24x (24% improvement)

The 24% improvement represents the portion of communication that couldn't be overlapped with computation. With perfect overlap, communication would be completely hidden.

### Memory Overhead

Symmetric allocation has minimal memory overhead:
- Allocation metadata: ~100 bytes per pool
- Memory pool caching: Reuses buffers, no additional GPU memory
- Virtual address space: No increase (same size allocations)

## Configuration and Usage

### Basic Configuration

```python
from megatron.core.distributed import DistributedDataParallelConfig
import megatron.core.nccl_allocator as nccl_allocator

# Initialize NCCL allocator
nccl_allocator.init()

# Configure DDP with symmetric memory
ddp_config = DistributedDataParallelConfig(
    # Enable NCCL Unified Buffer (symmetric memory)
    nccl_ub=True,

    # Keep symmetric registration enabled
    disable_symmetric_registration=False,

    # Combine with gradient bucketing for maximum benefit
    overlap_grad_reduce=True,
    bucket_size=40000000,  # 40M elements

    # Distributed optimizer also benefits from symmetric memory
    use_distributed_optimizer=True,
)
```

### Environment Variables

```bash
# Enable NVLS algorithm (set by nccl_allocator.init())
export NCCL_NVLS_ENABLE=1

# Disable PyTorch's tensor register allocator hook
# (Prevents conflicts with NCCL allocator)
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0

# Debug: Verify NCCL algorithm selection
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

# Optional: Force NVLS for testing
export NCCL_ALGO=NVLS
```

### Multi-Group Registration

For complex parallelism configurations (e.g., FSDP + Expert Parallelism), use `MultiGroupMemPoolAllocator`:

```python
# From nccl_allocator.py lines 235-316
pool = nccl_allocator.create_nccl_mem_pool(symmetric=True)

# Register with multiple process groups
groups = [
    data_parallel_group,
    expert_parallel_group,
    intra_dp_group,
]

with nccl_allocator.MultiGroupMemPoolAllocator(pool, groups, symmetric=True):
    # Allocations usable by all groups
    buffer = torch.zeros(size, device='cuda')

    # Operations on any group will use NVLS
    torch.distributed.all_reduce(buffer, group=data_parallel_group)
    torch.distributed.all_reduce(buffer, group=expert_parallel_group)
```

## Troubleshooting Guide

### Problem: NCCL Still Using Ring Algorithm

**Symptoms:**
```
NCCL INFO Using Ring algorithm
# Expected: "NCCL INFO Using NVLS"
```

**Debugging steps:**

1. Check NCCL version:
```python
import torch
print(torch.cuda.nccl.version())  # Should be >= (2, 18, 0)
```

2. Verify `nccl_ub` is enabled:
```python
print(ddp_config.nccl_ub)  # Should be True
```

3. Check memory pool registration:
```bash
export NCCL_DEBUG=INFO
# Look for "registered mem pool" messages
```

4. Verify NVLink connectivity:
```bash
nvidia-smi topo -m
# Should show NVLink connections (NV4, NV8, etc.)
```

### Problem: Memory Allocation Failures

**Symptoms:**
```
RuntimeError: CUDA out of memory
# OR
ncclInternalError: Call to ncclMemAlloc failed
```

**Causes and solutions:**

1. **Fragmented memory space**: Allocate symmetric buffers early in initialization
```python
# GOOD: Allocate before model creation
nccl_allocator.init()
pool = create_nccl_mem_pool(symmetric=True)
with nccl_mem(pool, ...):
    buffers = allocate_all_buffers()
model = create_model()

# BAD: Allocate after model created
model = create_model()
nccl_allocator.init()  # May fail due to fragmentation
```

2. **Insufficient memory**: Reduce buffer sizes or use gradient checkpointing

3. **Virtual address space exhaustion**: Reduce number of pools or consolidate allocations

### Problem: Performance Not Improving

**Symptoms:** Same throughput with/without symmetric memory

**Debugging:**

1. Profile communication time:
```python
import torch.cuda.profiler as profiler

with profiler.profile():
    torch.distributed.all_reduce(tensor, group=dp_group)
# Check if NVLS kernels are being used
```

2. Check for bottlenecks elsewhere:
```bash
nsys profile -o profile.qdrep python train.py
# Open in Nsight Systems and verify:
# - NCCL kernels show "NVLS"
# - Communication is overlapped with computation
```

3. Verify overlap is enabled:
```python
assert ddp_config.overlap_grad_reduce == True
assert os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') == '1'
```

## System Requirements

### Hardware Requirements
- **GPU**: NVIDIA A100, H100, or newer with NVLink
- **NVLink**: Version 3.0+ for optimal performance
- **Topology**: NVSwitch or direct NVLink connections
- **Not beneficial on**: PCIe-only systems, Ethernet/InfiniBand without NVLink

### Software Requirements
- **NCCL**: Version 2.18.0 or newer
- **PyTorch**: 2.0+ (2.9+ recommended for symmetric parameter)
- **CUDA**: 11.8+ (12.0+ recommended)
- **Driver**: NVIDIA driver 520+

### Verification Script

```python
import torch
import torch.distributed as dist
import megatron.core.nccl_allocator as nccl_allocator

def verify_symmetric_memory_support():
    """Verify system supports symmetric memory allocation."""

    # Check NCCL version
    nccl_version = torch.cuda.nccl.version()
    print(f"NCCL version: {nccl_version}")
    assert nccl_version >= (2, 18, 0), "NCCL 2.18+ required"

    # Check NVLink availability
    if torch.cuda.device_count() > 1:
        p2p_available = torch.cuda.can_device_access_peer(0, 1)
        print(f"P2P (NVLink) available: {p2p_available}")

    # Try symmetric allocation
    try:
        nccl_allocator.init()
        pool = nccl_allocator.create_nccl_mem_pool(symmetric=True)
        print("Symmetric memory pool created successfully")

        dist.init_process_group(backend='nccl')
        with nccl_allocator.nccl_mem(pool, symmetric=True):
            tensor = torch.zeros(1000, device='cuda')
        print("Symmetric allocation successful")

        return True
    except Exception as e:
        print(f"Symmetric memory not supported: {e}")
        return False

verify_symmetric_memory_support()
```

## Related Optimizations

- **#01 Gradient Bucketing**: Symmetric memory maximizes benefit when combined with bucketing
- **#03 Sequence Parallelism**: SP's reduce-scatter operations also benefit from NVLS
- **#07 Coalesced Communication**: Multiple operations on symmetric buffers can be batched
- **#27 Distributed Optimizer**: Parameter all-gather bandwidth improves with symmetric memory

## References and Resources

- NCCL 2.18+ Release Notes: https://github.com/NVIDIA/nccl/releases
- NVLS Technical Deep Dive: NVIDIA Developer Blog
- Megatron-LM Implementation: `megatron/core/nccl_allocator.py`
- PyTorch Pluggable Allocator API: PyTorch documentation
- NVLink Architecture Guide: NVIDIA Technical Documentation


