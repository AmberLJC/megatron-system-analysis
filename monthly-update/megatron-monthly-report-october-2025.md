# Megatron-LM Monthly Progress Report: October 2025

## Overview

October 2025 was marked by a **major GitHub transition**, **FSDP Expert Parallel support**, and **infrastructure modernization** with extensive CI/CD improvements.

---

## Key Highlights

### 1. FSDP Expert Parallel (DeepSeek-v3 Style)

- Major feature: Megatron-FSDP now supports Expert Parallel
- This enables efficient training of MoE models with FSDP, similar to DeepSeek-v3 architecture

### 2. Memory & Performance Optimizations

- Added CPU offloading interface for memory-constrained scenarios
- Implemented reduce-scatter with FP32 accumulation for improved numerical stability
- Added NSys NVTX context tracking and cleanup

### 3. Reinforcement Learning

- Added sequence packing support for RL training
- Improved efficiency of RL training with packed sequences

### 4. Dynamic Inference Cleanup

- Cleaned up dynamic inference step implementation
- Deduplicated dynamic engine and coordinator code
- Enabled mixed-batch sampling
- Made `get_asyncio_loop` safe for repeated use

### 5. GitHub Migration

- Massive investment in GitHub CI infrastructure
- Added multi-approval action workflow
- Implemented approval bots for code review automation
- Added copyright checking for GitHub CI
- Set up branch synchronization between internal and external repos
- Enabled integration tests on GitHub

### 6. Deprecations & Cleanup

- Soft deprecated Zarr checkpoint format (moving toward newer formats)
- Removed FP8 calibration script (superseded by newer approaches)
- Cleaned up GC (garbage collection) around CUDA graph creation

### 7. Compatibility

- Updated symmetric registration interface to sync with upstream PyTorch changes
- Fixed datasets to account for tokenizers with incorrect PAD token definitions

---

## Notable Commits

| Area | Description |
|------|-------------|
| FSDP | Megatron-FSDP Expert Parallel (DeepSeek-v3) Support |
| Memory | Add CPU offloading interface |
| Performance | Reduce-scatter implementation with FP32 accumulation |
| Profiling | Track and cleanup NSys NVTX context |
| RL | Add sequence packing to RL |
| Inference | Clean up dynamic inference step |
| Inference | Deduplicate dynamic engine + coordinator |
| Inference | Allow mixed-batch sampling in dynamic inference |
| CI | Multi-approval action |
| CI | Enable integration tests |
| CI | Auto-update copy-pr-bot vetters |
| CI | Add copyright checker for GitHub CI |
| CI | Sync branches workflow |
| Deprecation | Zarr soft deprecation |
| Cleanup | Delete utils_object_storage.py |
| Compatibility | Update symmetric registration interface |
| Data | Handle tokenizers with incorrect PAD definition |

---

## Infrastructure Updates

- Bumped ModelOpt version
- Extensive GitHub Actions setup:
  - Multi-approval workflow
  - Approval bots for dev/main branches
  - Copyright checker
  - Branch synchronization
  - Milestone automation
  - PR template community bot
- More granular unit test buckets
- Queue manager for dev branch
- Container image tagging with GitHub SHA
