# Megatron-LM Monthly Progress Report: August 2025

## Overview

August 2025 was a foundational month with major work on **Mixture of Experts (MoE) optimizations**, the beginning of **Megatron-RL integration**, and significant **inference engine improvements**.

---

## Key Highlights

### 1. MoE (Mixture of Experts) Enhancements

- Added Expert Parallel All-to-All (A2A) overlap support for interleaved Pipeline Parallelism and Multi-Token Prediction (MTP)
- Enabled recomputation for FP8 layernorm, MoE activations, and shared experts
- Introduced fused weighted squared ReLU activation
- Added MoE router fusion for improved performance
- Enabled Context Parallelism and recompute for MTP

### 2. Megatron-RL Integration Begins

- Started merging Megatron-RL into the main LM codebase (Phase 1/4)
- This marks the beginning of unified training and reinforcement learning capabilities

### 3. Dynamic Inference Engine

- Added Multi-Latent Attention (MLA) support for the dynamic backend
- Implemented non-decode CUDA graphs for faster inference
- Fixed log probability calculation with pipeline and sequence parallelism
- Added ZMQ-based communication for parallel inference requests

### 4. CUDA Graph Improvements

- Moved CUDA graph capture to core module
- Added `FullCudaGraphWrapper` implementation
- Fixed CUDA graphs with VPP (Virtual Pipeline Parallelism) and first/last layer BF16

### 5. FSDP (Fully Sharded Data Parallel)

- Decoupled Custom FSDP to make it independently installable
- Fixed distributed parameter weight shapes for >2D parameters

### 6. Model Support

- Enabled VLM (Vision-Language Model) FP8 support
- Added Llama4 HuggingFace checkpoint import
- Added Kimi-K2-Instruct HF import, PTQ, and EAGLE3 training support
- Enabled using Transformer Engine activation functions in MLP

### 7. Checkpointing & Optimization

- Implemented new optimizer checkpoint formats for distributed optimizer
- Added singleton local shards for checkpointing
- Added ability to abort persistent checkpoint worker

---

## Notable Commits

| Area | Description |
|------|-------------|
| MoE | EP A2A overlap for interleaved PP and MTP |
| MoE | Recomputation for FP8 layernorm/moe_act/shared_experts |
| MoE | Fused weighted squared ReLU |
| MoE | MoE router fusion |
| RL | Merge Megatron-RL into LM (1/4) |
| Inference | Dynamic Backend Inference MLA |
| Inference | Non-decode CUDA graphs for dynamic inference |
| Inference | ZMQ based communication for parallel inference |
| CUDA Graphs | Move cuda graph capture to core |
| CUDA Graphs | FullCudaGraphWrapper implementation |
| FSDP | Decouple Custom FSDP for independent installation |
| Models | VLM FP8 enablement |
| Models | Llama4 HF checkpoint support |
| Models | Kimi-K2-Instruct HF import, PTQ, EAGLE3 |
| Checkpointing | New optimizer checkpoint formats for DistOpt |
| Dependencies | Upgrade TE to 2.7 |
| Dependencies | Update PyTorch to 25.06 and TE 2.6 |

---

## Infrastructure Updates

- Upgraded Transformer Engine to 2.7
- Updated PyTorch to 25.06 container
- Added MLA-flash dependency
- Auto-publish megatron-fsdp package
- Multiple CI improvements for stability
