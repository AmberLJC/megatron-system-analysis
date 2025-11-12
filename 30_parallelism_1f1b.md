# 1F1B Pipeline Scheduling

## Context

Naive pipeline parallelism (GPipe) has massive bubbles. With 4 stages, bubble time is ~43% of total time.

**1F1B solution:** Interleave forward and backward to keep all stages busy → reduces bubbles to ~18%.

## Implementation

Three phases:
1. **Warmup:** Fill pipeline with (stages-1) forwards
2. **Steady state:** Alternate 1 forward, 1 backward
3. **Cooldown:** Drain pipeline with (stages-1) backwards

```python
# Conceptual 1F1B schedule
def forward_backward_1f1b(model, data, num_microbatches, pipeline_stages):
    # Phase 1: Warmup (fill pipeline)
    for i in range(pipeline_stages - 1):
        forward(microbatch=i)
    
    # Phase 2: Steady state (1 forward, 1 backward)
    for i in range(pipeline_stages - 1, num_microbatches):
        forward(microbatch=i)
        backward(microbatch=i - (pipeline_stages - 1))
    
    # Phase 3: Cooldown (drain pipeline)
    for i in range(num_microbatches - (pipeline_stages - 1), num_microbatches):
        backward(microbatch=i)
```

## Code Location

- **Implementation:** `megatron/core/pipeline_parallel/schedules.py` lines 1949-2304
- **Warmup:** Lines 1990-2020
- **Steady state:** Lines 2022-2156
- **Cooldown:** Lines 2158-2200

## Performance Impact

### Bubble Fraction

Formula: `bubble_fraction = (P-1) / (2M)`
- P = pipeline stages
- M = num microbatches

| Config | Bubble % | Notes |
|--------|----------|-------|
| GPipe (P=4) | 43% | Naive |
| 1F1B (P=4, M=16) | 9.4% | (3)/(32) |
| 1F1B (P=8, M=64) | 5.4% | (7)/(128) |

### vs GPipe

- **2-4x better** GPU utilization
- **Key:** Keep all stages busy during steady state

## When to Use

**Always use with pipeline parallelism:**
- PP size > 1
- Set `num_microbatches = 4-8 × pipeline_stages`

**Configuration:**

```python
# Automatically used in Megatron
pipeline_model_parallel_size = 4
num_microbatches = 16  # 4 × PP for good efficiency
```

## Related Optimizations

- [Interleaved 1F1B](31_parallelism_interleaved_1f1b.md) - Further reduces bubbles
- [Gradient Sync in Bubbles](37_parallelism_gradient_sync_bubbles.md) - Hide communication
- [Activation Deallocation](12_memory_activation_deallocation.md) - Reduces memory per microbatch

## References

- Paper: [GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- Paper: [PipeDream: Generalized Pipeline Parallelism](https://arxiv.org/abs/1806.03377)
- [Megatron-LM Paper](https://arxiv.org/abs/1909.08053)

