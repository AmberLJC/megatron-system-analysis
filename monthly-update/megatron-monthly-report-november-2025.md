# Megatron-LM Monthly Progress Report: November 2025

## Overview

November 2025 completed the **Megatron-RL integration**, introduced **API backwards compatibility checks**, and advanced **MoE and hybrid model capabilities**.

---

## 1. Megatron-RL Integration Complete

### Motivation

Foundation models require reinforcement learning-based post-training to align outputs with human preferences and improve task-specific performance. Existing RL solutions lacked deep integration with Megatron's distributed training infrastructure, creating performance bottlenecks and synchronization overhead between disconnected inference and training pipelines. Organizations needed native RL capabilities within Megatron for seamless RLHF workflows, efficient data parallel coordination for distributed inference-training workflows, and support for emerging techniques like Group Reward Policy Optimization (GRPO).

### Technical Details

The integration was completed in four phases (August-November 2025):

**Architecture Components:**

- **Decoupled Agent/Environment Design**: Agents and environments accept an inference interface handle supporting `.generate(prompt, **generation_args)` and return experience rollouts with rewards. Supports multiple backends: Megatron (native), OpenAI, HuggingFace.

- **Data Parallel Coordinator**: Central ZMQ-based server handling worker registration, client connections with handshake protocol, request forwarding with round-robin scheduling across DP ranks, response routing, and control signal broadcasting (PAUSE, STOP, UNPAUSE). Implements ZMQ ROUTER socket pattern with msgpack serialization.

- **Control Logic for RL Compatibility**: Updated coordinator control flow with explicit state transitions (`PAUSE_ACK`/`STOP_ACK` acknowledgments), proper synchronization barriers ensuring all DP ranks acknowledge before proceeding, and engine state management using `asyncio.Event()` objects.

- **GRPO Loss Implementation**: Policy gradient optimization with reward clipping, KL divergence penalty against reference model, entropy regularization, and importance sampling support with optional truncation.

- **Sequence Packing Support**: Enables multiple sequences per training batch for efficiency, tracks sequence boundaries within packed bins, and maintains compatibility with tensor parallelism and pipeline parallelism.

### Results

- **Working GRPO Example**: Trains Qwen 2.5 32B on DAPO17k dataset, evaluates on AIME 2024, achieves ~0.7 pass@32 after 300 steps with 0.6 average reward
- **Full feature set**: Sequence packing, CUDA graph support, KV cache offloading, weight swapping for efficient model updates
- **Comprehensive testing**: Unit tests for RL utilities, functional tests for DP coordinator throughput, batch invariant kernel testing

---

## 2. API Stability Initiative

### Motivation

The API Backwards Compatibility verification framework protects critical downstream projects like NeMo from unexpected breaking changes when upgrading Megatron-LM versions. It provides stable API guarantees for users upgrading between versions, catches unintended breaking changes before they reach production, and establishes clear boundaries for the public API (only `megatron/core` module is checked).

### Technical Details

The verification framework uses **Griffe**, a Python API documentation generator that analyzes code structure and detects breaking changes:

**Workflow:**
1. GitHub Actions triggers CI/workflow on PRs
2. Python script loads baseline (latest release tag) and current (PR branch) via Griffe
3. Applies filtering: skips `@internal_api`, `@experimental_api`, `@deprecated` decorated objects, private functions, test/experimental/legacy paths
4. Griffe compares signatures: parameters, defaults, parameter order, type annotations, return types
5. Reports breaking changes with exit code (0=pass, 1=fail)

**Exempt Decorators:**
- `@internal_api`: Marks internal implementation details not part of public API
- `@experimental_api`: Marks experimental features under active development
- `@deprecated(version, removal_version, alternative)`: Enables gradual migration path with DeprecationWarning

**Breaking Changes Detected:** Parameter removed, parameter added without default, parameter order changed, optional→required, function removed, return type changed

### Results

- **Signature Safety**: No unintended breaking changes in public API signatures slip through to release
- **Deliberate Breaking Changes**: Properly flagged with decorators for gradual transition
- **Type Safety**: Return type changes detected as warnings
- **Three-version deprecation timeline**: Version N (add decorator), Version N+1 (warnings), Version N+2 (remove)
- **CI Enforcement**: Breaking changes automatically fail CI/PR checks

---

## 3. MoE Advancements

### 3.1 NVFP4 MOE with Proper Padding

**Motivation:** NVFP4 (NVIDIA FP4) training on MoE models requires specific alignment constraints for efficient GEMM operations on Blackwell GPUs. MoE routing produces variable token counts per expert which may not align with NVFP4's hardware requirements, causing either padding overhead or CUDA graph incompatibility.

**Technical Details:**
- Implements "proper padding" at the routing map level (not token level) to align expert token counts
- Alignment size = 128 for MoE (to accommodate Hadamard transform operations and grouped quantization kernel efficiency)
- Added `get_fp4_align_size()` utility returning 64-128 alignment size depending on context
- Renamed `--moe-router-padding-for-fp8` to `--moe-router-padding-for-quantization` for unified FP8/FP4 support

**Results:** Eliminates explicit per-token padding in GroupedMLP layer, enables CUDA graph capture for MoE+NVFP4 training, maintains efficient NVFP4 GEMM execution with proper tensor alignment.

### 3.2 Hybrid-EP Backend for Flex Dispatcher

**Motivation:** DeepEP optimizes cross-node token dispatching but has suboptimal performance for intra-node scenarios. Large MoE models on GB200/H100 clusters need efficient all-to-all communication leveraging NVIDIA hardware capabilities (TMA, IBGDA).

**Technical Details:**
- Implements HybridEP backend alongside existing DeepEP backend in Flex Dispatcher
- Replaced `--moe-enable-deepep` with flexible `--moe-flex-dispatcher-backend` parameter supporting "deepep" and "hybridep"
- HybridEP leverages TMA (Tensor Memory Accelerator) for efficient memory access patterns and IBGDA for NVLink optimization
- Configurable SM count for dispatch and combine operations

**Results:** Significantly reduces SM resource usage, improves all-to-all communication bandwidth utilization on GB200/B200 GPUs, supports both intra-node (NVLink) and multi-node scenarios.

### 3.3 MoE Layer Type for Hybrid Models

**Motivation:** Megatron's hybrid models (combining Mamba, Attention, and MLP layers) could not previously include MoE layers, limiting exploration of hybrid expert/dense layer combinations.

**Technical Details:**
- Added MOE layer symbol 'E' to the `Symbols` class in hybrid layer allocation
- Extended layer mapping functions to track MoE layer indices alongside attention/mamba/mlp layers
- Updated `get_layer_maps_from_layer_type_list()` to return 4-tuple to accommodate MoE layer mapping
- Implemented MoE layer construction using same factory pattern as other hybrid layer types

**Results:** Users can now specify hybrid architectures like `"M*E-E*-"` (Mamba, Attention, MoE, MoE, Attention, dense MLP), enabling exploration of mixed-expert and mixed-layer-density architectures.

### 3.4 JIT for MoE Router and Preprocess

**Motivation:** MoE router and token preprocessing involve multiple small kernel launches with low arithmetic intensity and high CPU-to-GPU overhead. CUDA graph capture requires statically defined code paths.

**Technical Details:**
- Applied `@jit_fuser` decorator to MoE-critical operations
- Router operations JIT'd: `TopKRouter._apply_expert_bias()` with fused expert bias accumulation
- Preprocess operations JIT'd: `MoEFlexTokenDispatcher.dispatch_preprocess()`, `fused_pad_routing_map()`
- Implemented `RandomSTE` caching mechanism for CUDA graph compatibility

**Results:** Reduced kernel launch overhead, improved CUDA graph compatibility, better GPU utilization, reduced latency in inference scenarios with small batch sizes.

---

## 4. Dynamic Inference

### 4.1 Dynamic Engine Suspend/Resume via Prefill

**Motivation:** Long-running inference jobs need to pause/resume without losing in-flight request state, enabling GPU resource sharing across multiple inference jobs and checkpointing for fault tolerance.

**Technical Details:**
- **Suspend Phase**: Deallocates all GPU tensors (KV cache), saves request state via `checkpoint()`, clears CUDA graphs
- **Resume Phase**: Reallocates GPU tensors with `is_init=False`, recreates CUDA graphs if needed, re-adds all requests in original order
- Request state uses `DynamicInferenceRequestRecord.checkpoint()` to save token counts, sequence positions, and sampling parameters
- Works with unified memory levels (0, 1, 2) to control persistence strategy

**Results:** 100% lossless state preservation, suspend time 1-3 seconds, resume time 2-5 seconds, frees 60-80% of GPU memory (entire KV cache + metadata).

### 4.2 Graph Config Implementation

**Motivation:** CUDA graphs require fixed batch dimensions (token count, prefill/decode request counts), but dynamic batching produces variable request mixes each step. Without matching graphs, engine falls back to eager mode (10x slower).

**Technical Details:**
- Core data structure `InferenceBatchDimensions` captures (token_count, prefill_req_count, decode_req_count, has_explicit_chunked_prefill_req)
- `is_applicable_for_batch_dim()` checks if a graph config can handle actual batch dimensions using capacity matching
- `CUDAGraphBatchDimensionBuilder` generates optimal configs (decode-only, mixed prefill/decode)
- Expert parallelism synchronizes batch dimensions across EP ranks using AllReduce MAX

**Results:** 95%+ of inference steps match a pre-generated graph, 5-10x speedup for graphed steps vs. eager mode, 9x end-to-end throughput improvement in typical jobs.

### 4.3 PP KV Cache Allocation Fix

**Motivation:** In pipeline parallelism, different PP ranks have different layer counts causing different KV cache requirements. Without synchronization, ranks diverge in scheduling behavior leading to NCCL deadlock.

**Technical Details:**
- Problem: Each PP rank independently computed block_count from available GPU memory, resulting in different values
- Solution: AllReduce with MIN operation across pipeline_parallel_group to synchronize block counts
- All ranks adopt the lowest block count, ensuring none exceed memory
- Synchronization happens at context initialization (~1-5ms overhead)

**Results:** Eliminates hung inference jobs with pipeline parallelism, all PP ranks make identical scheduling decisions, works with uneven layer distributions.

### 4.4 Multi-Node Pipeline Parallel Inference

**Motivation:** Large models (175B+) require multiple nodes to hold all parameters. Pipeline stages can be distributed across different nodes requiring efficient inter-node token/embedding routing.

**Technical Details:**
- Each PP stage can be on different node (e.g., 175B with PP=8 could be distributed across 4 nodes)
- Uses NCCL P2P send/recv for inter-stage communication (not collective operations)
- Context allocated only for this rank's layers: `num_layers = total_layers // pipeline_model_parallel_size`
- Inference wrapper tracks `model_is_pipeline_parallel` flag for different forward paths

**Results:** Enables inference on 175B+ models split across multiple nodes, intra-node P2P (NVLink) <1ms per stage, inter-node P2P 25-100ms per stage, compatible with dynamic batching.

---

## 5. Knowledge Distillation

### 5.1 Refactored KD to use ModelOpt Plugins File

**Motivation:** The original KD implementation contained 600+ lines of custom distillation code duplicating functionality already available in the ModelOpt library. This created code duplication, maintenance burden, and inconsistency risk between Megatron's custom implementation and NVIDIA's standard ModelOpt plugins.

**Technical Details:**
- Deleted `/megatron/post_training/algos/distillation.py` (601 lines)
- Removed custom implementations of loss classes (LogitsKLLoss, HiddenStateCosineLoss, MSELoss), loss balancers, and projection layers
- Imported from ModelOpt: `modelopt.torch.distill.plugins.megatron` as `mtd_mcore`
- Replaced local functions with ModelOpt API: `mtd_mcore.setup_distillation_config()`, `mtd_mcore.adjust_distillation_model_for_mcore()`

**Results:** Reduced code footprint by 600+ lines, unified architecture with official ModelOpt plugins, single source of truth for loss functions, improved maintainability.

### 5.2 Create Separate Teacher Layer Spec in KD Mode

**Motivation:** The original KD implementation used a factory function that caused pickling issues with ModelOpt's distillation API. The teacher model was created with default layer specs inherited from the student without accounting for potentially different architecture. Heterogeneous model support was broken because custom hybrid patterns weren't properly applied to the teacher.

**Technical Details:**
- Replaced factory function approach with direct model instantiation, eliminating pickling requirements
- Teacher-specific layer specs now created based on teacher's configuration, not inherited from student
- Hybrid model parameters (attention ratio, MLP ratio, override patterns) properly extracted from teacher config
- Teacher checkpoint loaded immediately, ensuring weights are available

**Results:** Fixed pickling issues, correct teacher architecture with matching layer specs, proper heterogeneous model support, more direct and explicit model creation flow.

---

## 6. Mamba Model Improvements

### 6.1 Fixed Mamba Tensor Parallelism Issues

**Motivation:** The Mamba implementation had legacy weight initialization code that conflicted with tensor parallelism (TP) handling. The custom `_init_weights()` function interfered with Megatron's proper tensor parallel attribute tracking and initialization semantics.

**Technical Details:**
- Removed 50+ lines of legacy `_init_weights()` function that manually initialized layers using custom initialization scheme
- Removed confusing marker attributes (`_no_reinit`, `_no_weight_decay`) on parameters
- Added proper `tensor_model_parallel=True` attribute to critical parameters (`dt_bias`, `A_log`, `D`, `norm.weight`)
- Fixed fallback initialization: conv1d weight now uses `kaiming_uniform_` when `conv_init` is None

**Results:** Eliminated initialization conflicts, Mamba layers properly respect Megatron's distributed weight initialization framework, parameters correctly sharded across tensor parallel ranks, cleaner code.

### 6.2 Added MambaInferenceStateConfig Dataclass

**Motivation:** The inference context initialization required passing multiple separate arguments (`layer_type_list`, `mamba_conv_states_shape`, `mamba_ssm_states_shape`) across the codebase. This was error-prone and made the API unclear.

**Technical Details:**
- `MambaInferenceStateConfig` dataclass encapsulates three key pieces of information: layer_type_list, mamba_conv_states_shape, mamba_ssm_states_shape
- Added utility function `get_mamba_inference_state_config_from_model()` to automatically extract configuration from the model
- Updated `DynamicInferenceContext` to accept the config object
- Updated all inference examples and tools to use the helper function

**Results:** Simplified API with single parameter instead of three, reduced boilerplate by ~25 lines across multiple files, type safety with clear type hints.

### 6.3 Fixed Mamba with Chunked-Prefill

**Motivation:** During dynamic inference, when mixing decode requests with both regular prefill and chunked prefill requests, the code was incorrectly indexing into the batch indices array, causing incorrect state mapping for chunked prefill requests.

**Technical Details:**
- Bug: Direct indexing using the request ID didn't account for active requests having entries in the batch_indices array
- Fix: Uses `torch.where(context.request_ids == context.chunked_prefill_request_id)[0][0]` to properly locate the position of the chunked prefill request
- Then uses that position to index into `batch_indices`

**Results:** Chunked prefill requests now correctly map to their SSM state slots, enables mixed inference (decode + regular prefill + chunked prefill), single-line fix with high impact on inference correctness.

---

## 7. Data & Training

### 7.1 FIM (Fill-In-the-Middle) Dataset Support

**Motivation:** FIM is a training technique that improves code understanding models by training them to predict missing code segments in various positions. Standard left-to-right training doesn't expose models to predicting code at different positions (beginning, middle, end). This enables multi-directional code completion capabilities.

**Technical Details:**
- `GPTFIMDataset` extends `GPTDataset` with probabilistic text permutation
- Two FIM patterns supported:
  - **PSM (Prefix-Suffix-Middle)**: `[PREFIX_TOKEN] prefix [SUFFIX_TOKEN] suffix [MIDDLE_TOKEN] middle`
  - **SPM (Suffix-Prefix-Middle)**: `[PREFIX_TOKEN, SUFFIX_TOKEN] suffix [MIDDLE_TOKEN] prefix middle`
- Configuration: `fim_rate` (probability to convert sample), `fim_spm_rate` (SPM vs PSM selection), `fim_split_sample` (fragment splitting delimiter), `fim_fragment_rate` (per-fragment FIM probability)
- Algorithm: Bernoulli sampling controls FIM application, tokenizes to text, selects random boundaries, re-tokenizes segments independently, truncates/pads to maintain length

**Results:** Full integration with Megatron's argument parsing and data loading, supports both repository-level and file-level FIM augmentation strategies, length-preserving output ensures compatibility with sequence-length assumptions.

### 7.2 Removed Dependency on megatron.training within megatron.core

**Motivation:** The dependency created circular import risks (`megatron.core` is a foundation module that `megatron.training` depends on), limited module reusability for users integrating `megatron.core` independently, and caused problems for FSDP checkpoint compatibility.

**Technical Details:**
- Refactored functions to accept explicit parameters instead of calling global variables
- `get_ep_layer_offset(num_experts: int | None = None)`: Takes optional parameter instead of accessing `args.num_experts`
- `handle_experts_in_state_dict(state_dict, num_experts: int | None = None)`: Accepts `num_experts` explicitly
- Callers now pass `num_experts` from model config: `self.model_chunks[0].config.num_moe_experts`

**Results:** `megatron.core` no longer depends on `megatron.training`, core library can be used independently, cleaner API with explicit parameters and proper type hints.

### 7.3 Timer Overhead Reduction

**Motivation:** Large-scale training runs require detailed timing analysis, but timer overhead can skew measurements. Program startup time was untracked, first iteration has different characteristics (warm-up, caching) that shouldn't be averaged with subsequent iterations.

**Technical Details:**
- Added `set_elapsed()` method to Timer class for injecting pre-computed timing values
- Startup timestamp tracking: `program_start`, `main_entry`, `pretrain_entry` phases
- First iteration special handling with `is_first_iteration` flag
- Global time synchronization: computes minimum startup time across all ranks for fair comparison

**Results:** Can measure and report program initialization phases, first iteration excluded from averaging reducing bias, non-invasive timestamp injection via `set_elapsed()`, distributed awareness with global min startup time.

---

## 8. Compatibility

### 8.1 Fixed UVM Compatibility with CUDA 13

**Motivation:** Unified Virtual Memory (UVM) in CUDA has suffered from performance bottlenecks due to page faults and memory migration overhead. CUDA 13 introduced significant architectural improvements and new memory management features (HMM - Heterogeneous Memory Management) that Megatron needed to leverage.

**Technical Details:**
- Two UVM optimization levels: Level 0 (disabled), Level 1 (UVM allocation with CPU prefetching for idle weights)
- Key optimization techniques: async prefetching using `cudaMemPrefetchAsync()`, memory advising via `cudaMemAdvise()`, careful stream management, batch page fault processing
- CUDA 13 integration: HMM for seamless host-device sharing, managed memory discard capability, improved batched memcpy API, support for host-side `cuMemCreate` and `cudaMallocAsync`

**Results:** Up to 50% performance improvement with async prefetching, reduced geometric slowdown from 95.8% to 0.7%, better distributed training support, reduced kernel stalling with prefetched data.

### 8.2 Added asyncio Queue Like in Python 3.13

**Motivation:** Megatron-LM's dynamic inference engine uses asyncio-based queue systems for managing concurrent requests and batching. The original `asyncio.Queue` lacked proper shutdown semantics, risking deadlocks when both producers and consumers could be blocked waiting on the queue.

**Technical Details:**
- Python 3.13's `Queue.shutdown()` method with two operation modes:
  - **Default Mode (immediate=False)**: Future `put()` raises `QueueShutDown`, currently blocked callers unblocked, queue winds down normally with `get()` calls
  - **Immediate Mode (immediate=True)**: Queue terminates immediately, drained completely, blocked callers unblocked with `QueueShutDown`
- Integration with Megatron's async inference: requests managed through asyncio queues with dynamic batching, `async_step()` uses asyncio for continuous token generation, proper handling of control signals (PAUSE, UNPAUSE, SUSPEND, RESUME, STOP)

**Results:** Graceful worker termination without deadlocks, cleaner shutdown semantics, full Python 3.13 support, improved reliability in distributed inference with proper cleanup of async request queues.

---

## Notable Commits

| Area | Description |
|------|-------------|
| RL | Merge Megatron-RL into LM (4/4) - Complete |
| RL | Clean up DP coord code & unit test |
| RL | Update coordinator control logic for RL |
| API | Initialize API backward compatibility verification |
| API | API compat check workflow |
| MoE | NVFP4 MOE with Proper Padding |
| MoE | Add Hybrid-EP backend to Flex Dispatcher |
| MoE | Add MoE layer type to hybrid models |
| MoE | JIT for MoE router and preprocess |
| Inference | Dynamic engine suspend/resume via prefill |
| Inference | Implement graph config |
| Inference | Fix PP KV cache allocation |
| Inference | Enable multi-node PP inference |
| KD | Refactor KD to use ModelOpt plugins file |
| KD | Create separate teacher Layer Spec in KD mode |
| Mamba | Fix Mamba TP and remove legacy initialization |
| Mamba | Add MambaInferenceStateConfig dataclass |
| Mamba | Bugfix for Mamba with Chunked-Prefill |
| Data | Add FIM dataset support |
| Core | Remove dependency on megatron.training in megatron.core |
| Performance | Reduce Overhead in Timers |
| Compatibility | Fix UVM compatibility with CUDA 13 |
| Compatibility | Add asyncio Queue like in Python 3.13 |

---

## Infrastructure Updates

- API backwards compatibility check baseline established
- Flaky test markers for LTS tests
- Install test improvements
- Merge queue skip for install tests
- Updated backwards compat check baseline commits
