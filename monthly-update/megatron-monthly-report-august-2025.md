# Megatron-LM Monthly Progress Report: August 2025

## Overview

August 2025 was a foundational month with major work on **Mixture of Experts (MoE) optimizations**, the beginning of **Megatron-RL integration**, and significant **inference engine improvements**.

---

## Key Highlights

### 1. MoE (Mixture of Experts) Enhancements

#### 1.1 Expert Parallel A2A Overlap for Interleaved PP and MTP (PR #3074)

**Motivation:**
All-to-All (A2A) communication in MoE models creates significant GPU idle time during training, especially with large expert counts. The standard approach serializes computation and communication, leaving GPUs waiting during token dispatch and combination phases. This feature enables overlapping A2A communication with computation to maximize GPU utilization.

**Technical Details:**
- Implements a pipelined approach where A2A dispatch for the next layer overlaps with expert computation of the current layer
- Extends support to interleaved Pipeline Parallelism (PP) configurations where layers are distributed across pipeline stages in an interleaved pattern
- Adds Multi-Token Prediction (MTP) support, allowing the overlap optimization to work with speculative decoding training
- Configuration: `--moe-expert-parallel-a2a-overlap` flag enables this optimization
- Works in conjunction with existing expert parallelism configurations

**Results:**
- Reduced A2A communication overhead by overlapping with computation
- Improved training throughput for large MoE models
- Enables efficient scaling with both pipeline parallelism and MTP training

---

#### 1.2 MoE Recomputation for FP8 layernorm/moe_act/shared_experts (PR #3465)

**Motivation:**
Full activation checkpointing introduces ~30% computational overhead, making it impractical for large-scale MoE training. Memory constraints with FP8 training require fine-grained optimization that selectively recomputes high-cost components rather than entire layers.

**Technical Details:**
- Implements output-discarding checkpointing that discards submodule outputs during forward pass and recomputes during backward pass
- Three key recompute targets:
  - `moe_act`: Recomputes GroupedMLP activation function
  - `layernorm`: Recomputes input_layernorm and pre_mlp_layernorm
  - `shared_experts`: Recomputes shared expert computations
- Configuration: `--recompute-granularity selective --recompute-modules [module_names]`
- Integrates with TransformerEngine v1.9+ for Gradient Accumulation Fusion and FP8 training

**Results:**
- Enables larger batch sizes by freeing activation memory
- Low computational overhead compared to full recomputation
- Supports FP8 precision training without prohibitive memory costs
- Allows selective optimization targeting identified memory bottlenecks

---

#### 1.3 Fused Weighted Squared ReLU Activation (PR #3471)

**Motivation:**
Separate probability scaling operations during MoE token routing create unnecessary memory allocations and kernel launch overhead. Kernel fusion reduces GPU overhead by combining multiple operations into a single optimized kernel.

**Technical Details:**
- Implements `fused_weighted_squared_relu` as a core activation function
- Moves probability multiplication from unpermutation stage to GroupedMLP activation function
- Eliminates separate scaling operations by incorporating weighting within the kernel itself
- Integrated into `megatron.core.activations` module for framework-wide availability
- Configurable through model configuration via `squared_relu` parameter

**Results:**
- Reduced memory footprint during token routing phase
- Improved computational efficiency through kernel fusion
- Better scalability for large-scale MoE models
- Lower latency for token permutation operations

---

#### 1.4 MoE Router Fusion (PR #3809)

**Motivation:**
Router operations and auxiliary loss computation are hot paths in MoE training. Separate kernel launches for TopK routing and auxiliary loss calculations introduce CPU scheduling overhead and create GPU pipeline bubbles.

**Technical Details:**
- Fuses MoE TopK routing and auxiliary loss computation into single optimized kernel
- Requires TransformerEngine 2.7.0 or above
- Supports multiple auxiliary loss strategies:
  - GShard/SwitchTransformer style aux_loss (global load balancing)
  - DeepSeekV2/DeepSeekV3 style seq_aux_loss (per-sample load balancing)
- Configuration flag: `--moe-router-fusion` enables this optimization
- Complements `--moe-permute-fusion` and `--cross-entropy-loss-fusion`

**Results:**
- Reduced CPU overhead in MoE router execution
- Improved throughput by minimizing kernel launch latency
- Better GPU utilization with fewer pipeline stalls
- Router throughput/memory impact becomes negligible when combined with permute fusion

---

#### 1.5 Context Parallelism and Recompute for MTP (PR #3330)

**Motivation:**
Long-sequence training causes linear growth in activation memory, frequently leading to OOM errors. Standard full recomputation avoids OOM but introduces ~30% computational overhead. Multi-Token Prediction (MTP) further exacerbates memory requirements by expanding sequence context.

**Technical Details:**
- Context Parallelism (CP) partitions network inputs and activations along sequence dimension
- Total GPUs: TP × CP × PP × DP configuration
- Each GPU processes only its sequence chunk, reducing per-GPU activation memory by CP factor
- Attention computation requires all-gather of Key-Value (KV) pairs across GPUs
- Selective recomputation of KV pairs during backward pass avoids storing full sequence KV
- Works with multiple attention variants: MHA, MQA, GQA, uni-directional and bi-directional masking

**Results:**
- Activation memory per GPU reduced by CP factor (linear scaling)
- Computation and communication reduced by CP factor
- Eliminates OOM for long-sequence training
- ~30% overhead of full recomputation avoided through selective KV recomputation
- Enables MTP training with longer contexts and larger batch sizes

---

### 2. Megatron-RL Integration

#### 2.1 Megatron-RL Integration Phase 1/4 (PR #3646)

**Motivation:**
Enable native reinforcement learning (RL) based post-training within Megatron-LM, supporting RLHF and other RL algorithms at scale. This unifies training and reinforcement learning capabilities in a single framework.

**Technical Details:**
- **Refactored model provider system**: Created separate `gpt_builders.py` and `mamba_builders.py` to decouple model building from training/inference entry points
- **RL utilities integration**: Added comprehensive `rl_utils.py` (872+ lines) for rollout generation and reward computation
- **Optimizer enhancements**: Implemented `offload_to_cpu()` and `restore_from_cpu()` methods to free GPU memory during inference phases
- **Training loop modifications**: Updated `training.py` to support RL-specific workflows including agent environment integration
- Supports both in-place Megatron inference and external OpenAI/HuggingFace inference interfaces

**Results:**
- Enables distributed RL training on large models with dynamic inference integration
- Successfully integrated GRPO (Group Relative Policy Optimization) with math reasoning benchmarks
- Achieves approximately 0.7 pass@32 on AIME 2024 after 300 training steps
- Average training reward of 0.6 on DAPO17k mathematics dataset

---

### 3. Dynamic Inference Engine

#### 3.1 Dynamic Backend Inference MLA (PR #3569)

**Motivation:**
Extend dynamic batching inference to support Multi-Latent Attention (MLA) architectures, reducing KV cache memory requirements through compressed representation. This enables efficient inference for models like Qwen and DeepSeek with MLA layers.

**Technical Details:**
- **MLA-specific KV cache compression**: Modified memory buffer to store compressed KV latents instead of full KV tensors
- Memory reduction from `(2, layers, chunks, tokens, heads, head_dim)` to `(layers, chunks, tokens, kv_reduced_dim)` where `kv_reduced_dim = kv_lora_rank + qk_pos_emb_head_dim`
- **Flash MLA kernel integration**: Added support for `flash_mla_with_kvcache` kernel
- **Dynamic context enhancements**: Updated `DynamicInferenceContext` with `cache_mla_latent`, `kv_lora_rank`, and `qk_pos_emb_head_dim` parameters
- Added `split_te_layernorm_column_parallel_linear()` utility for MLA layer processing

**Results:**
- Reduced memory footprint for MLA models during dynamic batching
- Block size constraint: requires 64-token chunks for Flash MLA kernel compatibility
- Seamless integration with existing dynamic inference engine

---

#### 3.2 Non-decode CUDA Graphs for Dynamic Inference (PR #3688)

**Motivation:**
Extend CUDA graph acceleration beyond decode-only phases to include prefill and mixed-phase inference. This improves end-to-end latency by capturing both prefill (variable-length) and decode operations.

**Technical Details:**
- **Warmup mode enumeration**: Introduced `WarmupEngineMode` enum with `DECODE` and `NON_DECODE` modes
- **Token-count based graph configuration**: Changed from request-based to token-count based (`cuda_graph_token_counts`)
- **Dynamic graph selection**: Implemented `using_cuda_graph_this_step()` method for runtime CUDA graph selection
- Added `use_cuda_graphs_for_non_decode_steps` parameter to control non-decode phase capture
- Backward compatibility maintained via legacy `cuda_graph_request_counts` attribute

**Results:**
- CUDA graph acceleration for prefill phases with variable sequence lengths
- Improved latency for mixed-mode inference (blend of prefill and decode)
- Better hardware utilization during prefill-heavy workloads

---

#### 3.3 ZMQ-based Communication for Parallel Inference (PR #3757)

**Motivation:**
Support distributed inference coordination across data-parallel ranks using asynchronous message passing. Enable scalable parallel inference with round-robin load balancing across multiple model instances.

**Technical Details:**
- **DataParallelInferenceCoordinator**: Central coordination server using ZMQ ROUTER socket for:
  - Worker registration from data-parallel ranks
  - Client connection handling with handshakes
  - Request forwarding via round-robin scheduling
  - Control signal broadcasting (PAUSE, STOP)
- **InferenceClient**: Asynchronous client for submitting requests through ZMQ
- **Message protocols**: Uses msgpack for efficient binary serialization
- Built on asyncio for non-blocking request handling
- Added `gpt_dynamic_inference_with_coordinator.py` example

**Results:**
- Horizontal scaling of inference across multiple data-parallel ranks
- Clients submit requests asynchronously without blocking inference pipeline
- Clean separation between request routing and model inference
- Supports throughput-optimized batching strategies

---

#### 3.4 Log Probability Calculation Fix for PP and SP (PR #3718)

**Motivation:**
Fix incorrect log probability calculations in dynamic inference with pipeline parallelism (PP) and sequence parallelism (SP). Ensure correct token selection for log probability extraction across all parallelism configurations.

**Technical Details:**
- **Token selection logic**: Sophisticated tracking across distributed ranks:
  - Decode-only: directly select logits for last token in each request
  - Prefill/mixed: shift active token window left by one and set newly generated tokens
- **Two-phase log probability tracking**: Separated `prompt_log_probs` and `generated_log_probs`
- **Pipeline parallel fix**: Updated logits shape validation from batch-based to token-based
- **Sequence parallel support**: Added receiver buffer adjustment for SP with dynamic batching
- Added `unwrap_model()` calls to handle Float16Module wrappers

**Results:**
- Correct log probabilities for both prompt and generated tokens
- Accurate token probability tracking across multi-GPU setups
- Enables reliable likelihood-based beam search in distributed settings
- 405+ lines of test improvements covering dynamic context scenarios

---

### 4. CUDA Graph Improvements

#### 4.1 CUDA Graph Capture Move to Core Module (PR #3782)

**Motivation:**
Refactor CUDA graph capture from the training module into the core transformer module, improving code modularity and enabling reuse across training and inference scenarios.

**Technical Details:**
- **New `TECudaGraphHelper` class**: Captures CUDA Graphs using Transformer Engine's `make_graphed_callables()` API
- Configured via `--external-cuda-graph` flag and `--cuda-graph-scope` (full or attn)
- Works with both model chunks and microbatches
- Supports MTP layers alongside decoder layers
- Key additions:
  - `_layer_is_graphable()`: Checks layer eligibility for capture
  - `get_cuda_graph_input_data()`: Creates static input data per-chunk/microbatch/layer
  - `cuda_graph_set_manual_hooks()`: Sets pre-forward hooks for captured parameters
- Integration with `FP8GlobalStateManager` for quantized training

**Results:**
- External CUDA graph capture using TE's optimized API
- Reduced code duplication between training and inference paths
- Variable scope configuration (full layers vs attention blocks only)
- Better debugging with per-layer granularity logging

---

#### 4.2 FullCudaGraphWrapper Implementation (PR #3473)

**Motivation:**
Enable full-iteration CUDA graph capture for maximum performance when shapes are completely static. Capture the entire forward-backward pass as a monolithic graph for single kernel replay per iteration.

**Technical Details:**
- **FullCudaGraphWrapper class**: Wraps forward-backward function with graph capture/replay logic
- **StaticBufferLoader**: Copies dataloader outputs to static CUDA tensors with non-blocking copy
- **Three-phase execution**:
  - Phase 1 (Warmup): First N iterations run normally
  - Phase 2 (Capture): Captures entire forward-backward pass
  - Phase 3 (Replay): Subsequent iterations replay captured graph
- RNG state registration for reproducibility
- Supports single and multi-model chunk configurations

**Results:**
- Eliminates 500+ individual kernel launches per iteration
- Single graph replay call per iteration
- Maximum performance for static workloads
- Verified with FP8 training through golden value tests

---

#### 4.3 CUDA Graphs Fix for VPP and First/Last Layer BF16 (PR #3824, PR #3746)

**Motivation:**
Fix correctness issues when using CUDA graphs with Virtual Pipeline Parallelism (VPP) and mixed precision configurations where first/last layers use BF16 while interior layers use FP8.

**Technical Details:**

**VPP Fix (PR #3824):**
- Fixed `_determine_if_first_last_layer_of_this_vp_chunk()` to use correct VPP size
- Added `vp_stage` parameter to CudaGraphManager constructor
- Proper memory pool allocation per virtual pipeline stage

**BF16 Fix (PR #3746):**
- Updated `_CudagraphRunner.get_fp8_context()` to use layer-aware FP8 context
- Calls `get_fp8_context()` from `fp8_utils` with layer-specific awareness
- Determines FP8 application based on layer position and configuration flags

**Results:**
- Fixed training failures with CUDA graphs + VPP
- Correct first/last layer identification across VPP stages
- Proper FP8 recipe application only to intended layers
- 1.6% TPOT improvement for 583M model FP8 with CUDA graphs

---

### 5. FSDP (Fully Sharded Data Parallel)

#### 5.1 FSDP Decoupling for Independent Installation (PR #3443)

**Motivation:**
Decouple Megatron's custom FSDP implementation from the monolithic framework to enable independent installation and use. Allow integration with HuggingFace Transformers, TransformerEngine, and other frameworks.

**Technical Details:**
- **Directory restructuring**: Moved from `megatron/core/distributed/custom_fsdp/` to standalone `megatron_fsdp/` package
- **Adapter pattern**: `mcore_fsdp_adapter.py` bridges Megatron-Core and standalone library
- Key implementation components:
  - `megatron_fsdp.py` - Core FSDP implementation (1,107 lines)
  - `fully_shard.py` - High-level API (387 lines)
  - `param_and_grad_buffer.py` - Optimized parameter/gradient management (2,183+ lines)
- **DTensor integration**: Refactored to use PyTorch DTensor-based distributed checkpointing
- API change: `--use-custom-fsdp` → `--use-megatron-fsdp`

**Results:**
- **25% speed improvement** and **23% memory savings** vs PyTorch FSDP2
- Available as standalone PyPI package: `pip install megatron-fsdp`
- Works with multiple frameworks: PyTorch, HuggingFace, TransformerEngine
- Advanced optimizations: bucketing, zero-copy communication, SHARP support

---

#### 5.2 FSDP Distributed Parameter Weight Shapes Fix (PR #3877)

**Motivation:**
Fix a critical bug in handling multi-dimensional (>2D) parameters in FSDP's distributed tensor handling, which caused shape mismatches during collective operations.

**Technical Details:**
- **Bug location**: Lines ~2381 and ~3657 in `param_and_grad_buffer.py`
- **Original code**: `local_shape = (-1, orig_param.shape[1:].numel())` - incorrectly flattens all dimensions
- **Fixed code**: `local_shape = (-1, *orig_param.shape[1:])` - preserves all dimensions using tuple unpacking
- Example: Shape `(64, 32, 16)` now correctly creates `(-1, 32, 16)` instead of `(-1, 512)`

**Results:**
- Correctly handles multi-dimensional parameters (3D, 4D, and higher)
- Proper shape preservation throughout distributed tensor lifecycle
- Critical for tensor parallelism combined with FSDP
- Cherry-picked into `core_r0.14.0` release branch

---

### 6. Model Support

#### 6.1 VLM (Vision-Language Model) FP8 Support (PR #3432)

**Motivation:**
Enable FP8 quantization for Vision-Language Models to reduce memory consumption and computational overhead while maintaining training efficiency for multimodal models.

**Technical Details:**
- Enables FP8 quantization for VLM training pipeline
- Selective precision in vision backbone and language model components:
  - Vision projection keeps FP8 when enabled
  - Token handling uses FP8 format (32 tokens for MXFP8, 16 for others)
- `use_vision_backbone_fp8_arch` flag controls FP8 in vision encoder
- Integrates with LLaVA-style VLM models (CLIP ViT encoder + Language Model decoder)
- Compatible with tensor and pipeline parallelism

**Results:**
- Efficient multimodal training on resource-constrained hardware
- Reduced memory footprint without sacrificing convergence
- Maintains numerical stability through FP8 delayed/current scaling
- Enables large VLM training on GPUs with compute capability >= 80

---

#### 6.2 Llama4 HuggingFace Checkpoint Import (PR #3731)

**Motivation:**
Enable seamless integration of Meta's Llama4 models into Megatron-LM for efficient distributed training and inference, leveraging Llama4's advanced capabilities within the Megatron framework.

**Technical Details:**
- Supports `meta-llama/Llama-4-Scout-17B-16E-Instruct` and `Llama-4-Maverick-17B-128E-Instruct`
- Llama4 Scout configuration:
  - 48 transformer layers, 5120 hidden size
  - Group-Query Attention: 40 heads, 8 query groups
  - 16 MoE experts with sigmoid routing
  - YaRN RoPE scaling (factor 8.0, base 500000)
  - SwiGLU activation, RMSNorm
- Automatic HuggingFace checkpoint detection and conversion
- Integrates with ModelOpt for quantization workflows

**Results:**
- Full Llama4 training and inference support
- Distributed training with TP, PP, EP parallelism
- Quantization support (NVFP4, NVFP8)
- Export compatibility with TensorRT-LLM

---

#### 6.3 Kimi-K2-Instruct HF Import, PTQ, and EAGLE3 Training (PR #3678)

**Motivation:**
Provide end-to-end support for importing, quantizing, and training EAGLE3 speculative decoding models based on Moonshot AI's Kimi-K2-Instruct with Multi-Latent Attention.

**Technical Details:**

**HuggingFace Import:**
- Automatic conversion from HuggingFace to Megatron distributed checkpoint format
- Configuration: TP=8, EP=64, ETP=1

**Post-Training Quantization (PTQ):**
- Integrates with NVIDIA ModelOpt
- Supports NVFP4, FP8-Static, W4A8-FP8 quantization
- Quantization-Aware Training (QAT) support

**EAGLE3 Training:**
- **Online EAGLE3**: Both target and draft models in memory
- **Offline EAGLE3**: Precomputes target hidden states to disk
- Acceptance length (AL) evaluation on MT-Bench
- Training: Magpie-Align dataset, 8 nodes DGX H100 (64 GPUs)

**Results:**
- Improved inference speed through EAGLE3 draft models
- Reduced model size through 4-8 bit quantization
- End-to-end pipeline: HF import → PTQ/QAT → EAGLE3 → export
- Production-ready EAGLE3 checkpoints for speculative decoding

---

#### 6.4 Transformer Engine Activation Functions in MLP (PR #2452)

**Motivation:**
Enable using Transformer Engine's optimized fused activation kernels directly in MLP layers for better training performance and reduced memory consumption.

**Technical Details:**
- Supported activations: GELU, SwiGLU, GEGLU, ReLU, Quick GELU, ReGLU, SiLU (TE 2.8+)
- `use_te_activation_func` configuration flag in TransformerConfig
- Fused bias + activation computation in single kernel
- Recomputation support for activation memory savings in FP8
- Constraints: Cannot use with `bias_activation_fusion`, not compatible with Kitchen Linear

**Results:**
- Faster MLP computation through fused kernels
- Lower activation memory footprint
- Improved numerical stability
- Works with all Megatron parallelism strategies

---

### 7. Checkpointing & Optimization

#### 7.1 New Optimizer Checkpoint Formats for Distributed Optimizer (PR #3532)

**Motivation:**
Address limitations in optimizer state checkpointing that prevented changing model parallelism configurations when resuming training (resharding).

**Technical Details:**
- **Two new checkpoint formats**:
  1. `dp_reshardable` (Default): Fast save/load, not reshardable
  2. `fully_reshardable`: Supports arbitrary parallelism changes, slower
- Configuration: `--dist-ckpt-optim-fully-reshardable` flag
- Added `CheckpointableShardedTensor` and `LocalShardsContainer` classes
- Enhanced validation with richer mismatch debugging data
- Workflow: Train with `dp_reshardable`, switch to `fully_reshardable` when changing parallelism

**Results:**
- Flexibility in parallelism changes while maintaining performance
- 624 lines added to test_optimizer.py
- 1506 total insertions across 19 files

---

#### 7.2 Singleton Local Shards for Checkpointing (PR #3320)

**Motivation:**
Address scalability challenges in checkpointing MoE models by breaking down experts into individual tensors stored under separate global keys, improving checkpoint structure and enabling granular control.

**Technical Details:**
- Added `singleton_local_shards` metadata flag
- Modified `GroupedMLP.sharded_state_dict()` with `_break_into_individual_experts()` function
- Each expert gets separate global key: `experts.{global_expert_idx}.weight`
- GLU layers additionally split into `w` (weights) and `v` (values) tensors
- Maintains compatibility with sequential MLP checkpoints

**Results:**
- Fine-tuning compatibility between GroupedMLP and SequentialMLP
- Improved checkpoint flexibility with granular expert-level control
- 364 insertions across 11 files

---

#### 7.3 Ability to Abort Persistent Checkpoint Worker (PR #3719)

**Motivation:**
Enable graceful abort of asynchronous checkpoint worker processes during training failures, unexpected terminations, or in-process restarts to prevent resource leaks.

**Technical Details:**
- Modified `AsyncCaller.close()` to accept `abort: bool = False` parameter
- Process termination strategies:
  - **TemporalAsyncCaller**: Uses `process.kill()` for immediate termination
  - **PersistentAsyncCaller**: Uses `process.kill()` instead of `queue.join()`
- Added `reset_persistent_async_worker()` function in `async_utils.py`
- Integrates with inprocess restart framework via `AbortCheckpoint` class

**Results:**
- Graceful fault recovery with proper async worker cleanup
- Parametrized testing with `abort=[True, False]`
- Supports both synchronous and asynchronous termination paths
- Integrates with Megatron's in-process restart mechanism

---

## Notable Commits

| Area | Description | PR |
|------|-------------|-----|
| MoE | EP A2A overlap for interleaved PP and MTP | #3074 |
| MoE | Recomputation for FP8 layernorm/moe_act/shared_experts | #3465 |
| MoE | Fused weighted squared ReLU | #3471 |
| MoE | MoE router fusion | #3809 |
| MoE | Context Parallelism and recompute for MTP | #3330 |
| RL | Merge Megatron-RL into LM (Phase 1/4) | #3646 |
| Inference | Dynamic Backend Inference MLA | #3569 |
| Inference | Non-decode CUDA graphs for dynamic inference | #3688 |
| Inference | ZMQ based communication for parallel inference | #3757 |
| Inference | Log probability fix for PP and SP | #3718 |
| CUDA Graphs | Move cuda graph capture to core | #3782 |
| CUDA Graphs | FullCudaGraphWrapper implementation | #3473 |
| CUDA Graphs | Fix CUDA graph with VPP | #3824 |
| CUDA Graphs | Fix CUDA graph with first/last layer BF16 | #3746 |
| FSDP | Decouple Custom FSDP for independent installation | #3443 |
| FSDP | Fix distributed parameter weight shapes (>2D) | #3877 |
| Models | VLM FP8 enablement | #3432 |
| Models | Llama4 HF checkpoint support | #3731 |
| Models | Kimi-K2-Instruct HF import, PTQ, EAGLE3 | #3678 |
| Models | TE activation functions in MLP | #2452 |
| Checkpointing | New optimizer checkpoint formats for DistOpt | #3532 |
| Checkpointing | Singleton local shards for checkpointing | #3320 |
| Checkpointing | Ability to abort persistent checkpoint worker | #3719 |

---

## Infrastructure Updates

- Upgraded Transformer Engine to 2.7
- Updated PyTorch to 25.06 container
- Added MLA-flash dependency
- Auto-publish megatron-fsdp package
- Multiple CI improvements for stability

---

## Performance Summary

| Feature Category | Key Metric |
|-----------------|------------|
| MoE Optimizations | 468 TFLOPS for Mixtral 8X7B bf16 training |
| FSDP | 25% speed improvement, 23% memory savings vs PyTorch FSDP2 |
| CUDA Graphs | 3-9% end-to-end training speedup |
| Megatron-RL | 0.7 pass@32 on AIME 2024 benchmarks |
