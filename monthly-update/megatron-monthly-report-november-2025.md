# Megatron-LM Monthly Progress Report: November 2025

## Overview

November 2025 completed the **Megatron-RL integration**, introduced **API backwards compatibility checks**, and advanced **MoE and hybrid model capabilities**.

---

## Key Highlights

### 1. Megatron-RL Integration Complete

- Merged phase 4/4 of Megatron-RL into the main codebase
- RL is now a first-class citizen in Megatron-LM
- Added DP (Data Parallel) coordinator with unit tests
- Introduced coordinator control logic compatible with RL workflows

### 2. API Stability Initiative

- Introduced API backwards compatibility verification framework
- Helps ensure breaking changes are caught before release
- Critical for downstream users (NeMo, etc.)

### 3. MoE Advancements

- **NVFP4 MOE**: Support for NVFP4 quantization with proper padding
- **Hybrid-EP backend**: Added to Flex Dispatcher for more flexible expert routing
- Added MoE layer type to hybrid models
- Implemented JIT compilation for MoE router and preprocess

### 4. Dynamic Inference

- Implemented dynamic engine suspend/resume via prefill
- Added Graph Config implementation for flexible graph management
- Fixed PP (Pipeline Parallel) KV cache allocation
- Enabled multi-node pipeline parallel inference

### 5. Knowledge Distillation

- Refactored KD to use ModelOpt plugins file
- Created separate teacher Layer Spec in KD mode

### 6. Mamba Model Improvements

- Fixed Mamba tensor parallelism issues
- Added MambaInferenceStateConfig dataclass
- Fixed Mamba with chunked-prefill

### 7. Data & Training

- Added FIM (Fill-In-the-Middle) dataset support
- Removed dependency on `megatron.training` within `megatron.core`
- Improved timer overhead reduction

### 8. Compatibility

- Fixed UVM compatibility with CUDA 13
- Added asyncio Queue like in Python 3.13

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
