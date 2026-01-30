# Megatron-LM Monthly Progress Report: September 2025

## Overview

September 2025 focused on **RL integration continuation**, **attention mechanism innovations**, **dynamic inference maturation**, and **training optimizations**.

---

## Key Highlights

### 1. Megatron-RL Integration Progress

- Completed phases 2/4 and 3/4 of Megatron-RL merge
- The RL subsystem is becoming deeply integrated with the core training loop

### 2. Attention Mechanism Innovations

- **Sliding Window Attention (SWA) mixing**: Enabled mixing SWA with full attention for more flexible architectures
- **Sink Attention**: Added support for sink attention patterns (gpt-oss feature)
- **YaRN support**: Added Yet another RoPE extension for extended context length

### 3. Dynamic Inference Maturation

- Implemented chunked prefill for efficient long-context inference
- Added unified memory support for inference context
- Introduced event-based coordination for the dynamic engine
- Added functional tests for CUDA graphs in inference
- Optimized attention preprocessing

### 4. Training Features

- Added support for both Adam and AdamW optimizer selection
- Enabled gradient accumulation fusion for TE (Transformer Engine)
- Introduced Bridge Communicator for joint training of independently parallel modules
- Added quick GeGLU activation support

### 5. Expert/MoE Enhancements

- Enabled bias in expert MLP layers
- Fixed router input jitter dtype issues

### 6. Checkpointing & Validation

- Enabled simplified checkpointing workflow
- Fixed BERT + virtual pipeline parallelism compatibility

### 7. Knowledge Distillation

- Enabled KD support with Hybrid model training loop
- Added ModelOpt EAGLE refactorization and offline implementation

### 8. Infrastructure

- Upgraded to Transformer Engine 2.7
- Upgraded dependencies for 25.09 container
- Added gradient comparison test framework
- Major CUDA graph code refactoring

---

## Notable Commits

| Area | Description |
|------|-------------|
| RL | Merge Megatron-RL into LM (2/4) |
| RL | Merge Megatron-RL into LM (3/4) |
| Attention | Enabling mixing SWA with full attention |
| Attention | Sink Attention (gpt-oss) |
| Attention | YaRN support for gpt-oss |
| Inference | Add chunked prefill |
| Inference | Dynamic inference context - Unified memory |
| Inference | Dynamic inference engine - Events |
| Inference | Optimize attention preproc |
| Training | Add setting to support Adam or AdamW |
| Training | Add support for gradient accumulation fusion |
| Training | Bridge Communicator for joint training |
| MoE | Enable bias in expert mlp |
| MoE | Fix router input jitter dtype |
| Activation | Add quick geglu activation for gpt-oss |
| Checkpointing | Enable simplified checkpointing |
| KD | Enable KD support with Hybrid model train loop |
| KD | Support ModelOpt EAGLE refactorization |
| CUDA Graphs | Cudagraph code refactor |
| Fixes | Fix BERT + virtual pipeline parallelism |

---

## Infrastructure Updates

- Upgraded to Transformer Engine 2.7 wheel testing
- Added post-training review group
- GitHub workflows and CI automation
- Dependabot integration for GitHub CI
- Dev branch CI enablement
