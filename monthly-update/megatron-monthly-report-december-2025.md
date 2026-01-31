# Megatron-LM Monthly Progress Report: December 2025

## Executive Summary

December 2025 was a landmark month for Megatron-LM, delivering major advancements in **high-performance fused kernels**, **reinforcement learning production readiness**, **hybrid parallelism for inference**, **comprehensive documentation overhaul**, and **FSDP improvements**. The release targets the **Core 0.16 milestone** with 639 commits spanning performance optimizations, infrastructure upgrades, and expanded model support.

---

## 1. High-Performance NVLS Fused Kernels

### 1.1 Motivation

In tensor-parallel inference, four sequential operations occur frequently in transformer layers: reduce-scatter, residual addition, RMS normalization, and all-gather. These operations were traditionally executed as separate kernels, resulting in:

- **Multiple kernel launch overheads** causing latency accumulation
- **Repeated memory bandwidth consumption** as data is loaded and stored multiple times between operations
- **Synchronization delays** between GPU ranks at each operation boundary

The NVLS (NVLink Symmetric Memory) Fused Kernel addresses these bottlenecks by combining all four operations into a single kernel, leveraging NVIDIA Hopper+ GPU architecture's symmetric memory capabilities for efficient cross-GPU communication.

### 1.2 Technical Details

**Two-Phase Implementation:**

| Phase | PR | Date | Description |
|-------|-----|------|-------------|
| Foundation | #1997 | Dec 6, 2025 | NVLS communication kernels for sequence parallelism |
| Fusion | #2599 | Dec 18, 2025 | Fused reduce-scatter + residual + RMS-norm + all-gather |

**Core Implementation** (`megatron/core/inference/communication/torch_symm_triton/fused_collectives.py`):

The kernel uses Triton-based implementation with inline PTX assembly for performance-critical operations:

```
Phase 1: Reduce-Scatter + Residual Addition
├── Load data from symmetric memory buffer via multicast load (multimem.ld_reduce)
├── Perform hardware-accelerated reduction across ranks
├── Load and add residual values using bf16x2 vector operations
├── Store residual output
└── Compute squared sum for RMS normalization

Phase 2: RMS-Norm + All-Gather
├── Calculate mean of squares across all elements
├── Compute reciprocal square root via PTX assembly (rsqrt.approx.f32)
├── Apply normalization: norm = (value * rrms * weight).cast(bf16)
└── Store to multicast pointers for all-gather broadcast
```

**Hardware Acceleration via PTX Assembly:**
- **Multicast Load**: `multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2` - reads 128-bit values from same address across all peer GPUs with hardware reduction
- **Multicast Store**: `multimem.st.relaxed.sys.global.v4.f32` - writes to all peer GPU memory addresses simultaneously
- **Arithmetic**: `add.bf16x2` for vector addition, `rsqrt.approx.f32` for fast reciprocal square root

**Hardware Requirements:**
- NVIDIA Hopper+ GPU architecture (compute capability ≥ 9)
- PyTorch 2.2+ with symmetric memory support
- Tensor-parallel process group within single NVLink domain
- bfloat16 data type with 128-bit memory alignment

**Automatic Fallback:** If requirements aren't met, the system gracefully reverts to standard NCCL operations.

### 1.3 Results

| Metric | Improvement |
|--------|-------------|
| Kernel Launches | 4 kernels → 1 kernel |
| Memory Round-trips | Eliminated intermediate writes (data stays in registers) |
| Synchronization Points | Multiple barriers → Single sync point |
| Memory Efficiency | 128-bit (8×bf16) aligned access per thread |

**Codebase Impact:**
- 12 files modified, 681 insertions, 142 deletions
- New configuration flag: `--inference-fuse-tp-communication`
- Added `inference_fuse_tp_communication: bool` to `TransformerConfig`
- CUDA graph compatible for inference workloads

---

## 2. Reinforcement Learning Enhancements

### 2.1 Pipeline Parallelism Fix with Sequence Packing Rewriting

#### Motivation

The original RL pipeline had sequence packing logic tightly coupled with utility functions, creating maintenance challenges. The fix addresses pipeline parallelism issues where loss function handling wasn't properly integrated with pipeline stages, and the inference wrapper wasn't correctly accessing wrapped model methods.

#### Technical Details

**PR #2632** (December 16, 2025) - Major refactoring:

| Component | Before | After |
|-----------|--------|-------|
| Code Organization | Monolithic `rl_utils.py` (1,669 lines) | Dedicated `sequence_packing_utils.py` (1,169 lines) |
| Data Structures | Implicit packing logic | Explicit `PackingInfo` and `PackingContext` classes |
| Bin Size Config | Manual `--rl-sequence-packing-bin-size` required | Automatic derivation from sequence length |

**New Data Structures:**
- `PackingInfo`: Contains bin assignments, sequence indices, and metadata
- `PackingContext`: Holds pre-computed packed sequences, position IDs, attention masks, loss masks, and cached PackedSeqParams

**Loss Function Integration Fix:**
```python
# Now supports loss_func=None for inference-only paths
if loss_func is None:
    forward_data_store.append(output_tensor)
elif not collect_non_loss_data:
    outputs = loss_func(output_tensor)
```

**Model Inference Wrapper Fix:**
- Changed from direct `self.model.set_input_tensor()` calls
- Now uses `get_attr_wrapped_model()` for proper access through wrapper layers

#### Results

- 2,408 insertions, 2,103 deletions (net +305 lines)
- Enhanced test coverage with 409-line `test_sequence_packing_utils.py`
- Eliminated manual bin size configuration requirement

---

### 2.2 Rollouts Distributed Over Regular Data Parallel Group

#### Motivation

When Expert Parallelism (EP) coexists with Data Parallelism (DP), the system was incorrectly using Expert Data Parallel groups for rollout distribution instead of regular DP groups. This caused incorrect rollout partitioning across ranks and synchronization issues.

#### Technical Details

**PR #2634** (December 17, 2025) - Critical bug fix in `/megatron/rl/rl_utils.py`:

**Before (Incorrect):**
```python
n_prompts % mpu.get_expert_data_parallel_world_size() == 0
data_split_range = (mpu.get_expert_data_parallel_rank() * data_split_size, ...)
torch.distributed.all_gather(trajs_list, trajs, group=mpu.get_expert_data_parallel_group())
```

**After (Correct):**
```python
n_prompts % mpu.get_data_parallel_world_size() == 0
data_split_range = (mpu.get_data_parallel_rank() * data_split_size, ...)
torch.distributed.all_gather(trajs_list, trajs, group=mpu.get_data_parallel_group())
```

#### Results

- 18 insertions, 16 deletions (minimal but critical)
- Fixes critical bug affecting EP configurations
- Enables proper use of RL with Expert Parallelism

---

### 2.3 Parameter for Parallel Generation Tasks

#### Motivation

A new command-line parameter was needed to control the number of parallel generation tasks during RL inference, allowing fine-tuning of inference performance, GPU memory utilization, and throughput vs. latency trade-offs.

#### Technical Details

**PR #2712** (December 18, 2025):

**New Command-Line Argument:**
```python
group.add_argument('--rl-parallel-generation-tasks', type=int, default=512,
                  help='Number of parallel generation tasks for RL inference.')
```

**Integration Points:**
- `GroupedRolloutGenerator`: Added `parallel_generation_tasks: int = 512` attribute
- `WeightedMultiTask.from_config()`: Propagates parameter to all agents
- Dynamic inference engine: Configures `max_requests` based on parameter

#### Results

- 20 insertions, 5 deletions
- Flexible configuration for inference parallelism
- Seamless integration with weighted multi-task environments

---

### 2.4 RL Performance Benchmarks

**Configuration:** GPT 583M parameters, 24 layers, 1152 hidden size, 16 attention heads

| Metric | Value |
|--------|-------|
| Training Config | TP=2, PP=4, DP=8 |
| Inference Config | TP=1, PP=2 |
| Peak Memory (H100) | ~49.0-49.1 GB |
| Max Allocated | ~49.9 GB |
| Step 1 (with init) | 63.08s |
| Steady State | 3.48-4.35s |
| Average Throughput | ~3.73s/iteration |

---

## 3. Hybrid Parallelism for Inference

### 3.1 Motivation

Deploying large-scale Mixture-of-Experts (MoE) models for inference faces several challenges:

- **CUDA Graph Incompatibility**: MoE models with dynamic expert routing produce variable-sized tensors during prefill, making CUDA graph capture impossible
- **Expert Parallelism Requirements**: Models trained with EP > 1 need matching inference parallelism
- **Flexible Deployment Needs**: Organizations need various parallelism configurations without retraining

### 3.2 Technical Details

**Core Mechanism: Expert Padding for Decode**

The `set_decode_expert_padding()` function (`megatron/core/inference/utils.py`) toggles MoE drop-and-pad behavior:

| Phase | Expert Padding | Behavior |
|-------|----------------|----------|
| Prefill | Disabled | Variable sizes, dropless MoE, optimal utilization |
| Decode | Enabled | Fixed shapes, CUDA graph-safe, padding with capacity_factor |

**Capacity Factor Calculation:**
```python
capacity_factor = num_moe_experts / moe_router_topk
```

**Supported Parallelism Configurations:**

| Configuration | Support Status |
|---------------|---------------|
| TP Only | ✅ Fully supported |
| EP Only | ✅ Fully supported |
| TP + EP + DP | ✅ Fully supported |
| TP + EP + PP | ✅ Fully supported |
| TP + EP + DP + PP | ✅ Fully supported |

**Token Dispatcher Support:**
- **AllGather**: For small EP configurations
- **All-to-All**: Standard EP > 1 via NCCL
- **FlexDispatcher with DeepEP**: Removes redundant tokens during cross-node communication
- **FlexDispatcher with HybridEP**: Optimized for GB200/Multi-Node setups

### 3.3 Results

- **CUDA Graph Support**: Enables graph capture during decode for MoE models
- **No Token Dropping**: Proper capacity factor ensures all tokens processed
- **Memory Efficiency**: Variable-size metadata cleared when entering decode
- **Stop Word Support**: Added early termination for text generation
- **Request Tracking**: Improved lifecycle management via `InferenceRequest` status tracking

---

## 4. Documentation Overhaul

### 4.1 Motivation

The documentation infrastructure needed modernization:

- **Format Transition**: RST to Markdown for better developer experience
- **API Documentation**: Old `sphinx.ext.automodule` inadequate for Google-style docstrings
- **Organization**: Scattered structure lacking clear categorization
- **Developer Onboarding**: Contributing guidelines buried in root files

### 4.2 Technical Details

**Key PRs:**

| PR | Date | Description |
|----|------|-------------|
| #2489 | Dec 12 | Migrate docs to new Sphinx |
| #2542 | Dec 12 | Use autodoc2 and remove automodule |
| #2671 | Dec 18 | Improve documentation organization |
| #2717 | Dec 19 | Add developer section to docs |

**New Sphinx Configuration (`docs/conf.py`):**
```python
extensions = [
    "myst_parser",         # Markdown support
    "sphinx.ext.viewcode", # Source code links
    "sphinx.ext.napoleon", # Google-style docstrings
    "sphinx_copybutton",   # Copy buttons for code blocks
    "autodoc2"             # Modern API documentation
]
```

**autodoc2 Configuration:**
```python
autodoc2_packages = [{"path": "../megatron/core", "exclude_dirs": ["converters"]}]
autodoc2_render_plugin = "myst"
autodoc2_output_dir = "apidocs"
```

**Custom Google Docstring Parser** (`docs/autodoc2_docstrings_parser.py`):
- Bridges Google-style docstrings with autodoc2
- Uses Napoleon for parsing transformation

**New Documentation Structure:**
```
docs/
├── api-guide/
│   ├── core/           # Core Megatron components
│   ├── models/         # Model-specific APIs
│   └── internal/       # Internal utilities
├── user-guide/
│   └── features/       # MoE, Context Parallel, etc.
├── get-started/        # Quick start guides
├── models/             # LLMs, Multimodal
├── developer/          # NEW: Contributing guides
└── advanced/           # Complex topics
```

**Developer Section Contents:**
- `contribute.md` - Issue policies, code standards, commit guidelines
- `submit.md` - PR workflow, expert review, auto-assignment
- `oncall.md` - On-call rotation procedures
- `generate_docs.md` - Documentation generation instructions

### 4.3 Results

- **1,192 lines** of new documentation content
- **Markdown Support**: More intuitive for GitHub-familiar developers
- **MyST Features**: Math support, code blocks, task lists, flexible formatting
- **Build Optimization**: `SKIP_AUTODOC=true` for faster local builds
- **Live Building**: `sphinx-autobuild` for real-time documentation updates

---

## 5. Checkpointing & Compatibility

### 5.1 Backward Compatibility for mcore 0.15 Checkpoints

#### Motivation

Production systems with existing checkpoints needed seamless loading into newer Megatron Core versions without manual migration.

#### Technical Details

**PR #2648** (December 12, 2025):

Added two dummy placeholder classes in `megatron/core/dist_checkpointing/strategies/torch.py`:
- `MCoreMetadata`: Compatibility shim for deserializing optimizer states
- `MCoreSavePlan`: Compatibility shim for loading old checkpoint weights

#### Results

- Minimal 7-line implementation
- Zero runtime overhead
- Seamless checkpoint loading from version 0.15

---

### 5.2 Removed Flattened_Range Code Paths

#### Motivation

The `flattened_range` feature was deprecated and no longer needed for PyTorch distributed checkpoints. Removing dead code simplifies maintenance and reduces technical debt.

#### Technical Details

**PR #2126** (December 11, 2025):

- Eliminated all `flattened_range` code paths
- Removed related unit and functional tests
- Cleaned up distributed optimizer support code

#### Results

- Simplified checkpoint codebase
- Reduced maintenance burden
- **Note**: Prevents loading optimizer states from mcore < 0.14 checkpoints (model weights still loadable)

---

### 5.3 Simplified Parameter Sync for Checkpoint Save

#### Motivation

The original checkpoint workflow had unnecessary overhead from the disable/save/re-enable hook lifecycle.

#### Technical Details

**PR #2344** (December 15, 2025):

The optimization directly forces parameter synchronization during saves:
- Calls `start_param_sync()` directly
- Avoids hook disable/re-enable overhead
- Maintains correctness with simplified state management

#### Results

- Reduced overhead during distributed checkpoint operations
- Faster checkpoint saves in large-scale training
- Simplified DDP module logic

---

## 6. FSDP Improvements

### 6.1 Built Default FSDP DeviceMesh

#### Motivation

Users were required to manually construct DeviceMesh objects, adding complexity and boilerplate to FSDP applications.

#### Technical Details

**PR #2471** (December 16, 2025):

**Automatic DeviceMesh Construction:**
```python
if device_mesh is None:
    device_mesh = init_device_mesh(
        device_type="cuda",
        mesh_shape=(torch.distributed.get_world_size(), 1),
        mesh_dim_names=("fsdp", "tp"),
    )
```

**API Simplification:**
- Made `device_mesh` and `dp_shard_dim` parameters optional
- Separated `fully_shard_model()` and `fully_shard_optimizer()`
- Trivial TP dimension (size 1) for TransformerEngine compatibility

#### Results

- Simplified user onboarding (2-3 lines vs. manual mesh setup)
- Maintains backward compatibility for advanced users
- Users can now use: `model = fully_shard_model(model, fsdp_unit_modules=[...])`

---

### 6.2 Fixed HSDP Register Submesh Issues

#### Motivation

HSDP submeshes weren't properly registered with the main device mesh when using FSDP with Expert Parallelism, causing crashes in MoE model configurations.

#### Technical Details

**PR #2388** (December 2, 2025):

**Registration Fix:**
```python
# Register HSDP submeshes first
register_submesh(self.device_mesh, hsdp_submesh, True)
register_submesh(self.device_mesh, hsdp_tp_submesh, True)

# Then register EP submeshes
register_submesh(self.expt_device_mesh, tp_submesh, True)
register_submesh(self.expt_device_mesh, fsdp_tp_submesh, True)
```

#### Results

- Resolves crashes when combining HSDP with Expert Parallelism
- Enables MoE models with hybrid sharding strategies
- Clean separation between HSDP and EP mesh registrations

---

### 6.3 Support for Old and New DeviceMesh APIs

#### Motivation

PyTorch's DeviceMesh API underwent significant changes. Supporting both APIs ensures compatibility across PyTorch versions.

#### Technical Details

**PR #2575** (December 15, 2025):

**Dual API Pattern:**
```python
try:
    # New API
    flatten_mesh_names = [
        flat_dim for flat_dim, flat_mesh
        in device_mesh._get_root_mesh()._flatten_mapping.items()
    ]
except AttributeError:
    # Fallback to old API
    from torch.distributed.device_mesh import _mesh_resources
    flatten_mesh_names = [...]
```

#### Results

- Full compatibility with both old and new PyTorch DeviceMesh APIs
- Eliminates crashes when PyTorch's global mesh state is cleared
- Clean deprecation path for future PyTorch transitions

---

## 7. Features & Fixes

### 7.1 Stop Word Support for Inference

**PR #2685** | December 24, 2025

#### Motivation
Enable early termination of text generation when user-specified keywords appear, reducing unnecessary computation.

#### Technical Details
- Modified `sampling_params.py`, `dynamic_engine.py`, `text_generation_controller.py`
- Multi-token stop word support (all tokens must appear consecutively)
- Pre-tokenized stop words passed as arguments

#### Results
- Early stopping during text generation
- Integrated into Core 0.16 milestone

---

### 7.2 QK Logits Clipping (Non-Split Version)

**PR #1929** | December 10, 2025

#### Motivation
Improve training stability by preventing numerically unstable or extreme attention logits values.

#### Technical Details
- Implements clipping across MHA, GQA, and MLA attention types
- Operates on raw QK logits before softmax normalization
- Requires TransformerEngine 2.9.0+

#### Results
- Numerical stability improvements for attention computations
- Broad compatibility across attention mechanism variants

---

### 7.3 Batch Invariance for Consistent Training

**PR #2308** | December 10, 2025

#### Motivation
RL workflows require consistency between training and inference regardless of batch size variations.

#### Technical Details
- Achieves "complete match between Megatron inference and training"
- Batch-independent computation throughout the pipeline
- 26 commits with comprehensive test coverage

#### Results
- Consistent behavior across different batch sizes
- Critical for RL pipelines requiring deterministic behavior

---

### 7.4 Optimized TE CUDA Graph Input Memory

**PR #2392** | December 18, 2025

#### Motivation
Reduce memory overhead when preparing static input tensor buffers for CUDA graph captures across many microbatches.

#### Technical Details
- Reuses static input tensor memory buffers across microbatches
- Targets memory usage during input tensor preparation phase
- Unit tests for TP, PP > 1, VP > 1 configurations

#### Results
- Meaningful memory footprint reduction during graph initialization
- Particularly beneficial for long-training scenarios

---

### 7.5 Improved Data Loader Initialization Time at Scale

**PR #2445** | December 22, 2025

#### Motivation
Large-scale distributed training creates data loader initialization bottlenecks with massive datasets.

#### Technical Details

**Six Optimization Strategies:**

| Strategy | Description |
|----------|-------------|
| Filesystem Check Skipping | Removes redundant `os.path.exists` calls for `.idx`/`.bin` files |
| Array Generation Deferral | Skips `np.arange` arrays for available documents per split |
| Cache Existence Checks | Eliminates `os.path.isfile` verification for cached indexes |
| Synchronization Optimization | Removes barriers from BlendedMegatronDatasetBuilder |
| Memory Mapping Deferral | Defers `.npy` file memory mapping until first access |
| Metadata Consolidation | Single JSON file for all prefix metadata |

#### Results

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Initialization Time (4 nodes) | ~150 seconds | <15 seconds | **10x speedup** |

Further optimization possible by increasing `--num-dataset-builder-threads`.

---

## 8. Model Support

### 8.1 Kitchen Extensions' SDPA and Flash Attention

**PR #2232** | December 16, 2025

#### Motivation
Integrate NVIDIA's optimized Kitchen attention implementations for performance improvements over standard PyTorch.

#### Technical Details
- **1,732 lines** of new code across 8 files
- `KitchenDotProductAttention` and `KitchenFlashAttention` classes
- Quantization support via `QAttentionParams` and `QuantizedBMM`
- New config: `use_kitchen_attention`, `kitchen_attention_backend`

#### Results
- Unified attention interface with SDPA/Flash Attention selection
- Integration with Kitchen's quantization framework
- Access to highly optimized attention kernels

---

### 8.2 Nemotron Nano v2 VL Changes for Megatron Bridge

**PR #2078** | December 12, 2025

#### Motivation
Support FP8 quantization of vision backbone and optimize model initialization during CPU-based conversion.

#### Technical Details
- Added `use_vision_backbone_fp8_arch` parameter to LLaVAModel
- Conditional initialization guard in Mamba mixer
- Respects `perform_initialization` flag throughout architecture

#### Results
- Faster model conversion by eliminating redundant CPU-side initialization
- Flexible FP8 quantization for vision encoders
- Clean Megatron Bridge export compatibility

---

### 8.3 Non-Decode CUDA Graphs for Mamba Models

**PR #2474** | December 23, 2025

#### Motivation
Extend CUDA graph support from decode-only to include prefill operations for full inference pipeline optimization.

#### Technical Details
- **11 files, 1,159 insertions, 194 deletions**
- Separate buffers for decode, prefill, and chunked prefill requests
- New tracking: `_seq_idx_buffer`, `_cu_seqlens_buffer`
- Device-side counters for graph-conditional code paths

**New Triton Kernels** (`tensor_ops.py` - 462 lines):
- `_tensor_get_slice_after_kernel`: Slices tensor rows from device-resident position
- `_tensor_merge_kernel`: Merges two tensors at position boundary
- `_tensor_masked_update_kernel_2d/3d/4d`: Updates state tensors with index-based masking

#### Results
- CUDA graphs cover both prefill and decode phases
- Chunked prefill support for very long sequences
- Proper expert parallelism synchronization prevents hangs
- Variable-length sequence handling for mixed-batch scenarios

---

## Infrastructure Updates

| Category | Update |
|----------|--------|
| Documentation | Migration to Sphinx with MyST and autodoc2 |
| CI/CD | GitHub Actions upgraded to latest versions |
| Code Quality | API backwards compatibility checks marked as optional |
| Configuration | Model configs moved to GitHub |
| Operations | Oncall rotation system added |
| Checkpointing | Documentation updated for new workflows |

---

## Summary Statistics

| Metric | Value |
|--------|-------|
| Total December Commits | 639 |
| Target Milestone | Core 0.16 |
| Key Focus Areas | 8 major feature categories |
| Performance Highlights | 10x data loader init speedup, 4→1 kernel fusion |
| New Documentation | 1,192+ lines |

---

## Contributors

Special thanks to all contributors who made December 2025 a milestone month for Megatron-LM development:

- NVLS Kernels: jaredcasper, kvareddy, shanmugamr1992
- RL Enhancements: mathemakitten, tdene, jbarker
- FSDP Improvements: Cory Ye, Lifu Zhang, jianbinc
- Documentation: NVIDIA Documentation Team
- Checkpointing: dimapihtar, ananthsub, deepakn94
- Model Support: Frank Sun, Chen Cui, Keshav Santhanam, Kan Zhu

---

*Report generated: January 2026*
*Megatron-LM Repository: https://github.com/NVIDIA/Megatron-LM*
