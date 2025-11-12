# Gradient Sync in Pipeline Bubbles

## Context

Pipeline cooldown creates bubbles where some ranks idle. Use this time for gradient synchronization instead of adding it to critical path.

**Key insight:** Gradient all-reduce can happen during cooldown → "free" communication!

## Implementation

Disable gradient sync during pipeline execution. Enable only on last backward, which happens during cooldown bubble.

```python
# Conceptual implementation
def forward_backward_1f1b_with_grad_sync():
    # Warmup + Steady state: Disable grad sync
    with model.no_sync():
        for microbatch in range(num_microbatches - 1):
            forward(microbatch)
            backward(microbatch)
    
    # Last microbatch: Enable grad sync (happens during cooldown bubble!)
    backward(last_microbatch)  # Gradient all-reduce here!
    
    # Cooldown continues while all-reduce happens
```

## Code Location

- **Disable/enable:** `megatron/core/pipeline_parallel/schedules.py` lines 2046-2065
- **Sync in cooldown:** Lines 2246-2274

## Performance Impact

- **Communication: Fully hidden** (0% exposed)
- **Critical path:** No gradient communication overhead
- **"Free" gradient synchronization** during otherwise idle time

## When to Use

**Always use with pipeline parallelism:**
- PP > 1
- Fully hides gradient all-reduce in bubbles

## Related Optimizations

- [1F1B Scheduling](30_parallelism_1f1b.md) - Creates bubbles to use
- [Gradient Bucketing](01_communication_gradient_bucketing.md) - What gets synchronized

## References

- Paper: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- [Megatron-LM Pipeline Parallelism](https://github.com/NVIDIA/Megatron-LM)

