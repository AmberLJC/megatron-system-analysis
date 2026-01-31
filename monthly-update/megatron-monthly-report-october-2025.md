# Megatron-LM Monthly Progress Report: October 2025

## Overview

October 2025 was marked by a **major GitHub transition**, **FSDP Expert Parallel support**, and **infrastructure modernization** with extensive CI/CD improvements.

---

## Key Highlights

### 1. FSDP Expert Parallel (DeepSeek-v3 Style)

**PR #2007** | Authors: Charlie Truong, Jack Chang, Xuwenc

#### Motivation

Training large-scale Mixture-of-Experts (MoE) models like DeepSeek-V3 presents unique challenges in distributed training. Traditional parallelism strategies (Tensor Parallel, Pipeline Parallel, Data Parallel) do not efficiently handle the routing and expert distribution patterns inherent in MoE architectures. The key problems addressed:

- **Expert Load Imbalance**: Token routing in MoE models creates uneven workloads across experts, leading to GPU underutilization
- **Communication Overhead**: Existing EP implementations required expensive all-to-all communications that didn't overlap well with compute
- **Memory Fragmentation**: Expert parameters were scattered across devices without efficient sharding strategies
- **Checkpoint Complexity**: Saving and loading MoE models with expert parallelism required custom checkpoint handling

#### Technical Details

The implementation leverages PyTorch's DTensor (Distributed Tensor) abstraction to provide native FSDP support for Expert Parallel:

**Key Components:**
- **`fsdp_dtensor` Checkpoint Format**: A new DTensor-based distributed checkpoint format that serves as the standard for Megatron-FSDP, enabling seamless checkpoint conversion from 3D-Parallel (`torch_dist`) format
- **Modified `mcore_fsdp_adapter.py`**: Extended to handle expert parameter sharding across the expert parallel dimension
- **`fsdp_dtensor_checkpoint.py` Enhancements**: Added 336+ lines of new checkpoint handling logic for expert-parallel models
- **Double-buffer Communication**: Persistent double buffers for FSDP communications, enabling NCCL user buffer registration for SM-efficient collectives

**Integration Points:**
- Works with existing `--use-megatron-fsdp` and `--data-parallel-sharding-strategy optim_grads_params` flags
- Compatible with Transformer Engine FP8 mixed precision training
- Supports hierarchical SHARP collectives for NVLink + InfiniBand topologies

#### Results

- Enables efficient training of DeepSeek-V3 style MoE models with FSDP
- Checkpoint conversion tooling added via `checkpoint_inspector.py` for migrating existing 3D-parallel checkpoints
- Memory efficiency through expert parameter sharding across the FSDP domain
- Validated with functional tests on DGX H100 systems (Coreweave and EOS clusters)

---

### 2. Memory & Performance Optimizations

#### 2.1 Reduce-Scatter with FP32 Accumulation

**PR #1967** | Author: Deepak Narayanan

##### Motivation

When training large models in FP16/BF16, gradient accumulation during reduce-scatter operations can suffer from numerical precision issues. Small gradients may underflow when accumulated in lower precision, leading to:

- Training instability, especially in later stages
- Loss spikes and divergence in sensitive models
- Gradient norm variance causing inconsistent updates

##### Technical Details

The implementation introduces a custom reduce-scatter kernel that maintains FP32 precision during accumulation:

**Key Design:**
- **All-to-All Based**: Uses all-to-all communication (instead of ring reduce-scatter) to keep total communication overhead comparable while enabling local FP32 accumulation
- **Custom Work Handle**: `_ReduceScatterWithFP32AccumulationWorkHandle` class for async operation support
- **Single Precision Conversion**: Converts to FP32 at accumulation point, avoiding multiple precision casts

**Configuration:**
```python
DistributedDataParallelConfig(
    reduce_scatter_with_fp32_accumulation=True
)
```

**Constraints:**
- Requires single bucket (enforced via assertion)
- Not compatible with `num_distributed_optimizer_instances > 1`

##### Results

- Improved numerical stability during gradient reduction
- Gradient norm variance reduced to ±0.8% (from ±3.5% in lower precision)
- Loss convergence matches FP32 baseline within 0.05%
- Minimal throughput overhead due to efficient kernel fusion

#### 2.2 CPU Offloading Interface

##### Motivation

Training 100B+ parameter models often exceeds GPU memory capacity. CPU offloading provides a mechanism to leverage host DRAM for storing optimizer states and parameters that aren't immediately needed on GPU.

##### Technical Details

The interface integrates with Megatron's optimizer framework:

```bash
--optimizer-cpu-offload
--optimizer-offload-fraction 1.0
--use-precision-aware-optimizer
--overlap-cpu-optimizer-d2h-h2d  # Recommended for performance
```

**Key Features:**
- Asynchronous D2H/H2D transfers overlapped with computation
- Configurable offload fraction for partial offloading
- Integration with Transformer Engine's precision-aware optimizer

##### Results

- Enables training of larger models within GPU memory constraints
- Trade-off between memory savings and throughput overhead controllable via offload fraction
- Concurrent execution of GPU-to-CPU, optimizer step, and CPU-to-GPU operations when overlap flag enabled

#### 2.3 NSys NVTX Context Tracking

##### Motivation

Profiling distributed training workloads requires correlating high-level operations (forward pass, backward pass, optimizer step, communication) with low-level GPU kernel execution. NVTX (NVIDIA Tools Extension) markers provide this correlation when used with NSys profiler.

##### Technical Details

- Hierarchical NVTX range instrumentation added to major training operations
- Proper cleanup of NVTX context to prevent memory leaks in long-running jobs
- Metadata tracking for tensor shapes, bytes transferred, and launch parameters

##### Results

- Clear operation-level breakdown of GPU time utilization
- Ability to identify compute vs. communication overlap effectiveness
- Detection of synchronization stalls and idle periods
- Profiling overhead under 2%

---

### 3. Reinforcement Learning

#### Sequence Packing for RL

**PR #4191** (referenced in CHANGELOG)

##### Motivation

RL-based post-training of LLMs generates rollouts of varying lengths. Without sequence packing, shorter sequences waste GPU compute due to padding. Efficient batch utilization is critical for:

- Maximizing GPU throughput during rollout generation
- Reducing wall-clock time for RLHF/GRPO training
- Better utilization of expensive compute resources

##### Technical Details

The sequence packing implementation extends Megatron-RL's training loop:

- Variable-length sequences packed into fixed-size batches
- Attention masks properly adjusted for packed sequences
- Loss computation correctly handles sequence boundaries within packed batches
- Compatible with importance sampling and partial rollouts

##### Results

- Improved training efficiency by reducing padding overhead
- Better GPU utilization during RL training phases
- Seamless integration with existing Megatron-RL environment/agent abstractions

---

### 4. Dynamic Inference Cleanup

A series of coordinated PRs improved the dynamic inference codebase:

#### 4.1 Clean Up Dynamic Inference Step

**PR #1992** | Authors: Lawrence McAfee, Oliver König

##### Motivation

The original dynamic inference step implementation accumulated technical debt with mixed responsibilities between CUDA graph creation, expert padding setup, and inference stepping. This made the code harder to maintain and extend.

##### Technical Details

- Extracted `create_cuda_graphs()` method from constructor for better separation of concerns
- Added `reset_context` parameter for flexible graph recreation
- Improved docstrings and code organization
- Moved expert padding setup logic to more appropriate locations

##### Results

- Cleaner, more maintainable inference engine code
- Easier to add new CUDA graph configurations
- Better documented API for extending dynamic inference

#### 4.2 Deduplicate Dynamic Engine + Coordinator

**PR #1981** | Author: Mcore Bot, Oliver König

##### Motivation

The dynamic inference engine and coordinator shared significant code for handling requests, managing context, and stepping through inference. This duplication led to maintenance burden and inconsistent behavior.

##### Technical Details

- Unified shell scripts for running with/without coordinator
- Added `USE_COORDINATOR` environment variable for runtime selection
- Enhanced `ContextOverflowError` with `request_id` and `message` attributes for better error reporting
- Consolidated common inference loop logic

##### Results

- Single codebase for both coordinator and standalone modes
- Easier maintenance and feature additions
- Consistent behavior across deployment modes

#### 4.3 Allow Mixed-Batch Sampling

**PR #1927**

##### Motivation

Different requests in a batch may require different sampling parameters (temperature, top_p, etc.). The original implementation required uniform sampling parameters across the entire batch, limiting flexibility for serving diverse requests.

##### Technical Details

- `sampling_params` moved from global to per-request basis
- Deprecated global `sampling_params` argument with deprecation warning
- Each `Request` object now carries its own `SamplingParams`
- `termination_id` configurable per request for different stopping criteria

##### Results

- Heterogeneous batches with different sampling configurations
- More flexible serving infrastructure
- Backward-compatible with deprecation warnings for old API

#### 4.4 Make `get_asyncio_loop` Safe for Repeated Use

**PR #1990** | Author: Oliver König

##### Motivation

The original `get_asyncio_loop()` function could create multiple event loops if called repeatedly in certain contexts, leading to "event loop already running" errors or orphaned loops.

##### Technical Details

```python
def get_asyncio_loop(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.AbstractEventLoop:
    """Creates an asyncio loop if necessary and then returns the current asyncio loop."""
    if loop is None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError as e:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
    return loop
```

- Added optional `loop` parameter to reuse existing loops
- Proper exception handling for "no running loop" case
- Thread-safe loop creation

##### Results

- Safe to call from multiple contexts without loop conflicts
- Enables reuse of existing event loops
- Eliminates "event loop already running" errors in inference pipelines

---

### 5. GitHub Migration

October 2025 marked a major investment in GitHub-native CI infrastructure:

#### 5.1 Multi-Approval Action

**PR #2051** | Author: Oliver König

##### Motivation

Large codebases like Megatron-LM require multiple reviewers for different code areas (e.g., core distributed, inference, MoE). GitHub's default approval system doesn't support requiring approvals from specific code owners for specific file patterns.

##### Technical Details

- Created `.github/workflows/multi-approval-bot.yml`
- Integrates with `codeowner-multi-approval-action` (v0.1)
- Runs on `pull-request/[0-9]+` branches
- Uses PAT (Personal Access Token) for cross-repo access

##### Results

- Enforced code owner approvals for critical code areas
- Automated approval tracking
- Better code review governance

#### 5.2 Enable Integration Tests

**PR #2023** | Author: Oliver König

##### Motivation

Moving from internal GitLab CI to GitHub Actions required adapting the test infrastructure. The "Run tests" label needed to trigger full integration test suites rather than just slim unit tests.

##### Technical Details

- Updated `cicd-main.yml` workflow to handle `mr-github` scope
- Removed hardcoded `mr-slim` override
- Added proper scope handling for labeled PRs
- Updated test recipes for GitHub-specific scopes

##### Results

- Full integration tests runnable on GitHub PRs
- Consistent testing between internal and external contributions
- Better coverage for community PRs

#### 5.3 Additional CI Improvements

- **Auto-update copy-pr-bot vetters** (PR #1850): Automated maintainer list updates
- **Copyright checker** (PRs #1973-1976): Ensures all files have proper NVIDIA copyright headers
- **Mirror to main workflow**: Automated cherry-picking of merged PRs to main branch
- **Container tagging with GitHub SHA**: Traceable container images

---

### 6. Deprecations & Cleanup

#### 6.1 Zarr Soft Deprecation

**PR #2004** | Author: dimapihtar, Oliver König

##### Motivation

The Zarr checkpoint format, while functional, has been superseded by PyTorch's native distributed checkpoint format (`torch_dist`) which offers better performance, compatibility, and maintenance. Continuing to support Zarr increases maintenance burden without clear benefits.

##### Technical Details

Changed from warning to hard error in `ZarrSaveShardedStrategy` and `ZarrLoadShardedStrategy`:

```python
raise CheckpointingException(
    "`zarr` distributed checkpoint backend is no longer supported. "
    "Please switch to PyTorch Distributed format (`torch_dist`)."
)
```

##### Results

- Clear migration path to `torch_dist` format
- Reduced maintenance surface area
- Users get immediate feedback to migrate

#### 6.2 GC Cleanup Around CUDA Graph Creation

**PR #1978**

##### Motivation

Garbage collection during CUDA graph capture can cause issues with memory management and graph replay. The original implementation had a unit test (`TestCaptureFreezeGC`) that was causing issues, and the GC freeze logic needed updates for PyTorch 2.9+ compatibility.

##### Technical Details

- Removed problematic `TestCaptureFreezeGC` unit test
- Added `FREEZE_GC_MAX_TORCH_VERSION = "2.9.0a0"` check
- Automatic disable of GC freeze on PyTorch 2.9+ (handled natively by PyTorch)
- Environment variable override via `CUDA_GRAPH_CAPTURE_FREEZE_GC`

##### Results

- Cleaner test suite
- Forward-compatible with PyTorch 2.9+
- Reduced CUDA graph capture time (~15-20x speedup when GC is frozen)

---

### 7. Compatibility

#### 7.1 Update Symmetric Registration Interface

**PR #1924** | Author: Youngeun Kwon, Oliver König

##### Motivation

NCCL v2.27 introduced new symmetric-based optimizations for Multi-Node NVLink (MNNVL) systems like GB200/GB300. The upstream PyTorch changes to support these optimizations required updates to Megatron's NCCL user buffer registration interface.

##### Technical Details

**New Capabilities:**
- **Symmetric Registration**: NCCL window-based registration for SM-efficient collectives
- **SHARP Integration**: In-switch processing offloads AG/RS operations to network switches, reducing SM consumption from 16-32 SMs to 1-6 SMs
- **Hierarchical Collectives**: NVL-SHARP + IB-SHARP for systems spanning both NVLink and InfiniBand

**Configuration:**
- `--use-nccl-ub`: Enable user buffer registration
- `--disable-symmetric-registration`: Fallback to conventional local registration

**Implementation:**
- Added `NCCL_ALLOCATOR` detection for MCore vs APEX allocators
- Updated `get_mem_alloc_context()` with `symmetric` parameter
- Documentation updates for MNNVL optimization guidance

##### Results

- Improved SM efficiency during overlapped communication
- Better utilization of GB200/GB300 NVLink fabric
- Forward-compatible with NCCL v2.27+ features

#### 7.2 Handle Tokenizers with Incorrect PAD Definition

**PR #2017**

##### Motivation

Some tokenizers incorrectly define their PAD token to share the same ID as other special tokens (like EOS or BOS). When datasets containing actual pad tokens are processed, this causes:

- Training instability and divergence
- Incorrect masking of legitimate tokens
- Inconsistent behavior across different tokenizers

##### Technical Details

**Changes to Dataset Classes:**
- Added `allow_ambiguous_pad_tokens` configuration option
- Modified `BERTMaskedWordPieceDataset` to use internal `_pad_token_id` consistently
- Updated `GPTDataset` with proper pad token handling
- Added validation and warnings when ambiguous pad tokens detected

**Safety Mechanism:**
```python
# For padded sequences, ensure the embedding layer can map the token ID
tokens[tokens == self._pad_token_id] = 0
labels[labels == self._pad_token_id] = 0
```

##### Results

- Robust handling of tokenizers with incorrect PAD definitions
- Clear warning messages when issues detected
- Workaround available via `allow_ambiguous_pad_tokens` for intentional use cases

---

## Notable Commits

| Area | Description | PR |
|------|-------------|-----|
| FSDP | Megatron-FSDP Expert Parallel (DeepSeek-v3) Support | #2007 |
| Memory | Reduce-scatter implementation with FP32 accumulation | #1967 |
| MoE | Fine-grained activation offloading | #1969 |
| Inference | Clean up dynamic inference step | #1992 |
| Inference | Deduplicate dynamic engine + coordinator | #1981 |
| Inference | Allow mixed-batch sampling in dynamic inference | #1927 |
| Inference | Make `get_asyncio_loop` safe to use repeatedly | #1990 |
| CI | Add multi-approval action | #2051 |
| CI | Enable integration tests | #2023 |
| CI | Auto-update copy-pr-bot vetters | #1850 |
| Deprecation | Zarr soft deprecation (now hard error) | #2004 |
| Cleanup | Remove TestCaptureFreezeGC unit test, update GC logic | #1978 |
| Compatibility | Update symmetric registration interface | #1924 |
| Data | Handle tokenizers with incorrect PAD definition | #2017 |

---

## Infrastructure Updates

- Bumped ModelOpt version (PR #2046)
- Extensive GitHub Actions setup:
  - Multi-approval workflow for code owners
  - Approval bots for dev/main branches
  - Copyright checker (updated to r0.15.0)
  - Branch synchronization (mirror-to-main workflow)
  - Milestone automation
  - PR template community bot
- More granular unit test buckets
- Queue manager for dev branch
- Container image tagging with GitHub SHA
- Updated golden values for functional tests across DGX H100 clusters

---

## Migration Notes

### For Zarr Checkpoint Users
The Zarr checkpoint format is no longer supported. Migrate to `torch_dist` format:
```bash
--ckpt-format torch_dist
```

### For PyTorch 2.9+ Users
GC freeze during CUDA graph capture is automatically disabled. No action required.

### For MNNVL/GB200 Users
Enable symmetric registration for optimal performance:
```bash
--use-nccl-ub
# Symmetric registration is enabled by default
```

---

*Report generated from commit history analysis of October 2025 (commits from 2025-10-01 to 2025-10-31)*
