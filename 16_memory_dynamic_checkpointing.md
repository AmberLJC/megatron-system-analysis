# Dynamic Activation Checkpointing (Pipeline)

## Context

During pipeline warmup, many microbatches are in-flight simultaneously. Older microbatches need more aggressive checkpointing since they'll sit in memory longer during the warmup phase.

**Key insight:** Use selective checkpointing for recent microbatches, full checkpointing for older ones.

## Implementation

```python
def forward_step_helper(microbatch_id, num_microbatches, checkpoint_config):
    """Dynamically adjust checkpointing based on microbatch age"""
    
    # Recent microbatches: lightweight checkpointing
    if microbatch_id >= num_microbatches - checkpoint_config.partial_checkpoint_mb:
        checkpoint_mode = 'selective'
    else:
        # Older microbatches: aggressive checkpointing
        checkpoint_mode = 'full'
    
    # Forward with appropriate checkpointing
    output = model.forward(input, checkpoint_mode=checkpoint_mode)
    return output
```

## Configuration

```python
# In pipeline schedule
num_microbatches = 16
num_microbatches_with_partial_activation_checkpoints = 4

# Last 4 microbatches: selective checkpointing
# First 12 microbatches: full checkpointing
```

## Code Location

- **Implementation:** `megatron/core/pipeline_parallel/schedules.py` lines 2083-2174
- **Config:** `num_microbatches_with_partial_activation_checkpoints` argument

## Performance Impact

- **Reduces peak memory** during pipeline warmup by 20-30%
- **Enables more microbatches** (better bubble reduction)
- **Minimal compute overhead** (only old microbatches fully checkpointed)

## When to Use

**Use when:**
- Pipeline parallelism with many microbatches (> 8)
- Large pipeline stages (many layers per stage)
- Memory-constrained training

**Configuration:**
```python
num_microbatches = 16
num_microbatches_with_partial_activation_checkpoints = 4
```

## Related Optimizations

- [Activation Checkpointing](15_memory_activation_checkpointing.md) - Base checkpointing strategy
- [1F1B Scheduling](30_parallelism_1f1b.md) - Pipeline schedule that uses this
- [Activation Deallocation](12_memory_activation_deallocation.md) - Complementary optimization

## References

- Paper: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)

