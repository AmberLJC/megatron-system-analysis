# Megatron-LM Monthly Progress Report: December 2025

## Overview

December 2025 focused on **high-performance kernels**, **RL production readiness**, **documentation overhaul**, and **hybrid parallelism** for inference.

---

## Key Highlights

### 1. High-Performance Fused Kernels

- **NVLS Fused Kernel**: Added fused reduce-scatter + residual + RMS-norm + all-gather kernel
- This is a significant communication-computation overlap optimization

### 2. Reinforcement Learning Enhancements

- Fixed pipeline parallelism in RL with sequence packing rewriting
- Rollouts now distributed over regular data parallel group
- Added parameter for setting parallel generation tasks

### 3. Hybrid Parallelism for Inference

- Enabled hybrid tensor + expert + data parallelism in Megatron Core inference
- This allows for more flexible deployment configurations

### 4. Documentation Overhaul

- Migrated docs to new Sphinx framework
- Added autodoc2 for better API documentation
- Added developer section to documentation
- Improved documentation organization with additional guides

### 5. Checkpointing & Compatibility

- Added backward compatibility support for loading mcore 0.15 checkpoints
- Removed flattened_range code paths for distributed optimizer checkpointing
- Simplified parameter sync for checkpoint save

### 6. FSDP Improvements

- Built default FSDP DeviceMesh
- Fixed HSDP register submesh issues
- Support for both old and new DeviceMesh APIs

### 7. Features & Fixes

- Added stop word support for inference
- Added QK logits clipping (non-split version)
- Implemented batch invariance for consistent training
- Optimized TE CUDA graph input memory
- Improved data loader initialization time at scale

### 8. Model Support

- Added Kitchen extensions' SDPA and Flash Attention implementations
- Improved Nemotron nano v2 VL changes for Megatron Bridge
- Added support for non-decode CUDA graphs for Mamba models

---

## Notable Commits

| Area | Description |
|------|-------------|
| Kernels | NVLS fused reduce-scatter + residual + rms-norm + all-gather |
| RL | Pipeline parallelism fix with sequence packing rewriting |
| RL | Rollouts distributed over regular data parallel group |
| RL | Add parameter for parallel generation tasks |
| Inference | Enable hybrid tensor + expert + data parallelism |
| Inference | Adding stop word support |
| Inference | Fix entangled request generations |
| Docs | Migrate docs to new Sphinx |
| Docs | Use autodoc2 and remove automodule |
| Docs | Add developer section to docs |
| Docs | Improve documentation organization |
| Checkpointing | Backward compatibility for mcore 0.15 checkpoints |
| Checkpointing | Remove flattened_range code paths |
| Checkpointing | Simplify parameter sync for save |
| FSDP | Build default FSDP DeviceMesh |
| FSDP | HSDP register submesh fix |
| FSDP | Support old and new DeviceMesh APIs |
| Training | Batch Invariance |
| Training | QK logits clipping (non-split version) |
| Performance | Optimize TE cudagraph input memory |
| Performance | Improve data loader initialization time |
| Models | Kitchen extensions' SDPA and FA implementations |
| Models | Nemotron nano v2 VL changes |
| Models | Non-decode CUDA graphs for Mamba |
| Distributed | M4 + Dist Checkpoint: Replace global parallel state |
| Core | RNG sharding to include EP rank |

---

## Infrastructure Updates

- Documentation migration to Sphinx with autodoc2
- Oncall rotation system added
- GitHub Actions upgraded to latest versions
- API backwards compatibility checks marked as optional
- Model configs moved to GitHub
- Checkpointing documentation updated
