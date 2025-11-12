# Interleaved 1F1B (Virtual Pipeline)

## Context

Standard 1F1B still has bubble time proportional to number of stages. Can we reduce bubbles further?

**Solution:** Split each stage's layers into V "virtual stages", alternate execution between chunks to fill bubbles.

## Implementation

Each pipeline stage manages V model chunks. Schedule alternates between chunks to keep stage busy during what would be bubbles.

```python
# Virtual pipeline with V=2
# Stage 0 has layers [1-12] and [49-60]
# Stage 1 has layers [13-24] and [61-72]
# etc.

# Schedule alternates: chunk0 forward, chunk1 forward, chunk0 backward, ...
# This fills bubbles with useful work!
```

## Code Location

- **Implementation:** `megatron/core/pipeline_parallel/schedules.py` lines 809-1400
- **Schedule table:** Lines 900-980
- **Chunk switching:** Lines 1050-1150

## Performance Impact

### Bubble Reduction

Formula: `bubble_fraction = (P-1) / (2MV)`
- V = virtual pipeline size

| Config | Bubble % | Improvement |
|--------|----------|-------------|
| 1F1B (P=4, M=16, V=1) | 9.4% | Baseline |
| Interleaved (P=4, M=16, V=2) | 4.7% | **2x better** |
| Interleaved (P=4, M=16, V=4) | 2.4% | **4x better** |

### Cost

- Memory: Need V× model chunks (2-4x memory per stage)
- Complexity: More complex scheduling

## When to Use

**Use when:**
- Pipeline stages ≥ 4
- Pipeline bubbles > 5% of time
- Memory allows (need 2-4x model chunks)

**Configuration:**

```python
pipeline_model_parallel_size = 4
virtual_pipeline_model_parallel_size = 2  # or 4
num_microbatches = 16
```

**Skip if:**
- PP ≤ 2 (minimal benefit)
- Memory constrained (needs more memory)

## Related Optimizations

- [1F1B Scheduling](30_parallelism_1f1b.md) - Base schedule
- [Dynamic Checkpointing](16_memory_dynamic_checkpointing.md) - Manages memory for virtual stages

## References

- Paper: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- [Megatron-LM Virtual Pipeline](https://github.com/NVIDIA/Megatron-LM)

