# 02. NCCL Symmetric Memory

## Context

NCCL's NVLS (NVLink Sharp) algorithm can achieve 2-3x higher bandwidth than traditional ring algorithms, but it requires buffers at the **same virtual address** across all GPUs. PyTorch's default allocator can't guarantee this, forcing NCCL to fallback to slower algorithms.

On DGX A100 systems with NVLink, this means getting 120 GB/s instead of 320 GB/s for collective operations.

## Implementation

Custom C++ extension using `ncclMemAlloc_v2` for symmetric memory allocation. Memory pools cache buffers across training steps to amortize allocation overhead.

### How It Works

1. **Symmetric Allocation:** Use NCCL's API to allocate memory at same virtual address on all GPUs
2. **Memory Pooling:** Cache allocated tensors to avoid repeated allocation overhead
3. **Registration:** Register memory pools with NCCL process groups
4. **Usage:** Gradient buffers and parameter buffers use symmetric memory
5. **NVLS Detection:** NCCL automatically detects symmetric buffers and uses fast NVLS algorithm

## Core Code

- `megatron/core/nccl_allocator.py:49-58` - C++ extension wrapper
- `megatron/core/nccl_allocator.py:109-189` - Memory pool management (`NCCLMemPool`)
- `megatron/core/nccl_allocator.py:190-232` - Registration with process groups
- `megatron/core/distributed/param_and_grad_buffer.py:699-714` - Usage in gradient buffers

## Code Snippet

```python
# From nccl_allocator.py:109-189
class NCCLMemPool:
    """
    Memory pool for NCCL-allocated tensors.
    Caches tensors across iterations to amortize allocation cost.
    """

    def __init__(self, symmetric=True):
        self.symmetric = symmetric
        self.pool = {}  # Cache: (size, dtype, device) -> tensor
        self.stats = {'hits': 0, 'misses': 0}

    def allocate(self, size, dtype, device):
        """
        Allocate or retrieve cached tensor.
        
        Returns tensor at SAME VIRTUAL ADDRESS across all GPUs
        when symmetric=True.
        """
        key = (size, dtype, device)

        # Check cache first - avoid repeated allocation
        if key in self.pool:
            self.stats['hits'] += 1
            return self.pool[key]

        self.stats['misses'] += 1

        # Allocate using NCCL's symmetric allocator
        if self.symmetric:
            # This is the CRITICAL call!
            # ncclMemAlloc_v2 ensures same virtual address on all GPUs
            tensor = _NCCL_ALLOCATOR.nccl_mem_alloc(
                size, device, symmetric=True
            )
            # ^ Virtual address of 'tensor' is IDENTICAL across all GPUs!
            #   This enables NCCL NVLS algorithm (2-3x faster)
        else:
            # Standard PyTorch allocation (different addresses)
            tensor = torch.empty(size, dtype=dtype, device=device)

        # Cache for future use
        self.pool[key] = tensor
        return tensor

    def free_all(self):
        """Free all cached tensors (usually at end of training)"""
        self.pool.clear()


# Usage in gradient buffers (param_and_grad_buffer.py:699-714)
def _allocate_gradient_buffer(self, device):
    """Allocate gradient buffer with NCCL symmetric memory"""
    
    if self.nccl_ub:  # NCCL Unified Buffer (symmetric memory)
        # Create memory pool for this process group
        pool = nccl_allocator.create_nccl_mem_pool(symmetric=True)
        
        # Allocate within NCCL memory context
        with nccl_allocator.nccl_mem(pool, group=dp_group, symmetric=True):
            # This allocation will use symmetric memory!
            self.grad_data = torch.zeros(
                self.numel,                 # Total gradient size
                dtype=self.grad_dtype,      # Usually FP32 or BF16
                device=device
            )
            # ^ NCCL detects symmetric buffer and automatically
            #   uses NVLS algorithm → 2-3x bandwidth improvement!
    else:
        # Standard allocation (non-symmetric)
        self.grad_data = torch.zeros(
            self.numel, dtype=self.grad_dtype, device=device
        )


# Registration with process groups (nccl_allocator.py:190-232)
def register_nccl_mem_pool(pool, group):
    """
    Register symmetric memory pool with NCCL process group.
    Required for NCCL to recognize buffers as symmetric.
    """
    if not hasattr(group, '_nccl_mem_pools'):
        group._nccl_mem_pools = []
    
    group._nccl_mem_pools.append(pool)
    
    # Notify NCCL about symmetric buffers
    # NCCL will use NVLS algorithm for collectives on these buffers
    _NCCL_ALLOCATOR.register_pool(pool, group._get_backend_comm())
```

### C++ Extension Wrapper

```python
# From nccl_allocator.py:49-58
class _NCCLAllocator:
    """Wrapper around C++ extension for NCCL memory allocation"""
    
    def __init__(self):
        try:
            # Import custom CUDA extension
            from megatron.core.extensions import nccl_alloc
            self.nccl_alloc = nccl_alloc
            self.available = True
        except ImportError:
            self.available = False
            logger.warning("NCCL allocator not available")

    def nccl_mem_alloc(self, numel, device, symmetric=True):
        """
        Allocate memory using ncclMemAlloc_v2 API.
        
        When symmetric=True, guarantees same virtual address
        across all GPUs in the process group.
        """
        if not self.available:
            return torch.empty(numel, device=device)
        
        # Call C++ extension: uses ncclMemAlloc_v2 from NCCL
        return self.nccl_alloc.alloc_symmetric(numel, device)
```

## When to Use

**Always enable on NVLink systems:**
- DGX A100/H100 systems
- HGX platforms
- Any system with NVLink connectivity

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    nccl_ub=True,                          # Enable NCCL Unified Buffer
    disable_symmetric_registration=False,   # Keep symmetric memory
)
```

### Requirements

1. **NCCL version ≥ 2.18:** Earlier versions don't support symmetric memory API
2. **NVLink connectivity:** On Ethernet/InfiniBand, benefit is minimal
3. **NVIDIA GPUs:** AMD GPUs use RCCL, which has different APIs

### Skip If

- Single GPU training (no communication)
- Non-NVIDIA GPUs
- NCCL version < 2.18
- Ethernet-only systems (no NVLink)

## Performance Impact

### Bandwidth Improvement

**On DGX A100 (NVLink):**
- All-reduce bandwidth: 120 GB/s → 320 GB/s (2.67x improvement)
- Reduce-scatter bandwidth: 100 GB/s → 280 GB/s (2.80x improvement)
- Algorithm used: Ring → NVLS (NVLink Sharp)

**End-to-End Training:**
- Training throughput: ~20% speedup on NVLink systems
- Why only 20%? Communication is ~15% of critical path
  - 2.5x speedup on 15% → saves 9% of total time
  - But overlap hides most communication → net ~20% improvement

### Example Measurements

For GPT-3 175B model with DP=8 on DGX A100:
- Gradient size per rank: ~44 GB
- All-reduce time: 366ms → 138ms (2.65x faster)
- Step time: 820ms → 660ms (19.5% faster)

## Troubleshooting

### NCCL Not Using NVLS

**Symptoms:**
- Bandwidth doesn't improve
- NCCL still using ring algorithm

**Debug with:**
```bash
export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=INIT,GRAPH

# Look for:
# "NCCL INFO Using NVLS" → Good!
# "NCCL INFO Using RING" → Bad, symmetric memory not working
```

**Causes:**
1. NCCL version < 2.18
2. Symmetric memory not properly allocated
3. Buffers not registered with NCCL

**Fix priority:**
1. Check NCCL version: `python -c "import torch; print(torch.cuda.nccl.version())"`
2. Verify `nccl_ub=True` in config
3. Check NCCL debug output for "NVLS" messages

### Memory Allocation Failures

**Symptoms:**
- OOM errors during buffer creation
- Crashes during allocation

**Causes:**
1. Fragmented memory space
2. Not enough free memory
3. Virtual address space exhaustion

**Fix priority:**
1. Allocate buffers earlier (less fragmentation)
2. Reduce buffer sizes
3. Use memory pools (automatic in Megatron)

### Compatibility Issues

**Symptoms:**
- Import errors for C++ extension
- Crashes during initialization

**Causes:**
1. C++ extension not compiled
2. CUDA version mismatch
3. NCCL library not found

**Fix priority:**
1. Recompile Megatron with NCCL support
2. Verify CUDA_HOME and LD_LIBRARY_PATH
3. Check NCCL installation

## Related Optimizations

- **#01 Gradient Bucketing:** Works together - bucketing + symmetric memory = maximum benefit
- **#07 Coalesced Communication:** Combines multiple ops on symmetric buffers
- **#27 Distributed Optimizer:** Parameter all-gather also benefits from symmetric memory

## Configuration Example

```python
from megatron.core.distributed import DistributedDataParallelConfig

ddp_config = DistributedDataParallelConfig(
    # NCCL symmetric memory (20% improvement on NVLink)
    nccl_ub=True,                          # Enable NCCL Unified Buffer
    disable_symmetric_registration=False,   # Keep registration
    
    # Combine with gradient bucketing
    overlap_grad_reduce=True,
    bucket_size=40000000,
    use_distributed_optimizer=True,
)

# Environment variables
import os
os.environ['NCCL_DEBUG'] = 'INFO'          # Debug NCCL algorithm selection
os.environ['NCCL_NVLS_ENABLE'] = '1'       # Force enable NVLS (for testing)
```

## NCCL Algorithms

| Algorithm | Bandwidth | Requirements |
|-----------|-----------|--------------|
| **NVLS (NVLink Sharp)** | 300+ GB/s | Symmetric memory + NVLink |
| **Tree** | 150-200 GB/s | Standard allocation |
| **Ring** | 100-150 GB/s | Fallback |

NCCL automatically selects the best algorithm based on:
1. Buffer alignment and symmetry
2. Operation size
3. Process group topology
4. NVLink connectivity

## References

- NCCL 2.18+ release notes: [NCCL Releases](https://github.com/NVIDIA/nccl/releases)
- NVLS documentation: [NVIDIA Collective Communication Library](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/operations.html)
- Megatron implementation: `megatron/core/nccl_allocator.py`

