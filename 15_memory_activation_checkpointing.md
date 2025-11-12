# Activation Checkpointing

## Context

Storing all activations for backward pass requires O(layers) memory. For 96-layer transformer, this is ~40-60% of total memory. Selective checkpointing trades memory for recomputation.

**Key trade-off:** Save 40-60% memory at cost of 15-25% more compute.

## Implementation

Three strategies:

### 1. Selective Checkpointing
- Checkpoint attention blocks only
- Recompute attention in backward
- Less memory, more recompute

### 2. Full Checkpointing
- Checkpoint all transformer layers
- Recompute everything in backward
- More memory savings, most recompute

### 3. Distributed Checkpointing
- Split activations across TP dimension
- Combines with sequence parallelism
- Best memory/compute trade-off

## Configuration

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    # Selective checkpointing (recommended)
    recompute_granularity='selective',  # or 'full'
    recompute_method='uniform',         # or 'block'
    recompute_num_layers=None,          # None = all layers
    
    # Distributed checkpointing (with TP)
    distribute_saved_activations=True,  # Split across TP
)
```

## Code Location

- **Config:** `megatron/core/transformer/transformer_config.py` lines 296-310
- **Implementation:** `megatron/core/transformer/transformer_block.py`

## Performance Impact

| Strategy | Memory Saved | Compute Overhead | Best For |
|----------|--------------|------------------|----------|
| Selective | 30-40% | 10-15% | Balanced |
| Full | 50-60% | 20-25% | Memory-constrained |
| Distributed | 40-50% | 12-18% | With TP |

### Example: 175B Model

**Without checkpointing:**
- Activations: 80 GB per GPU
- Training time: 100s/step

**With selective checkpointing:**
- Activations: 50 GB per GPU (38% savings)
- Training time: 112s/step (12% overhead)
- **Net benefit:** Can fit model that otherwise wouldn't

## When to Use

**Use checkpointing when:**
- OOM errors during training
- Want larger batch sizes
- Memory-constrained hardware

**Skip if:**
- Plenty of GPU memory
- Compute-bound (not memory-bound)
- Want maximum throughput

## Related Optimizations

- [Dynamic Checkpointing](16_memory_dynamic_checkpointing.md) - Adaptive strategy for pipeline
- [Activation Deallocation](12_memory_activation_deallocation.md) - Complementary for pipeline
- [Sequence Parallelism](03_communication_sequence_parallel.md) - Reduces checkpoint memory

## References

- Paper: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- [Gradient Checkpointing in PyTorch](https://pytorch.org/docs/stable/checkpoint.html)

