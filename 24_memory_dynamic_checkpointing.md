# 24. Dynamic Activation Checkpointing (Pipeline)

## Context

During pipeline warmup, many microbatches are in-flight. Older microbatches need more aggressive checkpointing since they'll sit in memory longer.

## Implementation

Newer microbatches use selective checkpointing, older ones use full checkpointing. Balances memory for outstanding microbatches.

## Core Code

- `megatron/core/pipeline_parallel/schedules.py:2083-2174` - Dynamic checkpointing
- Config: `num_microbatches_with_partial_activation_checkpoints`

## Code Snippet

```python
# Dynamic checkpointing based on microbatch age
def forward_step_with_dynamic_checkpointing(mb_id, num_warmup_mb):
    """Adjust checkpointing granularity based on microbatch age"""
    
    # Newer microbatches (in warmup): selective checkpointing
    if mb_id < num_microbatches_with_partial_checkpointing:
        recompute_granularity = 'selective'  # Less aggressive
    else:
        # Older microbatches (will wait long): full checkpointing
        recompute_granularity = 'full'  # More aggressive
    
    # Forward with appropriate checkpointing
    output = forward_step(input, recompute_granularity)
    return output
```

## When to Use

- Pipeline parallelism with many microbatches
- Large pipeline stages (many layers per stage)

```python
num_microbatches = 64
num_microbatches_with_partial_activation_checkpoints = 4
# First 4 microbatches: selective checkpointing
# Remaining 60: full checkpointing
```

## Performance Impact

- Reduces peak memory during pipeline warmup
- Enables more microbatches (better bubble reduction)
- Trade-off: More recomputation for older microbatches

## References

- Implementation: `megatron/core/pipeline_parallel/schedules.py`

