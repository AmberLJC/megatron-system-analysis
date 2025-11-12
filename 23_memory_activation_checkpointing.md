# 23. Activation Checkpointing Strategies

## Context

Storing all activations for backward pass requires O(layers) memory. For 96-layer model, that's 96 × activation_size memory!

Selective checkpointing trades memory for recomputation: save only some activations, recompute others during backward.

## Implementation

Three strategies:
- **Selective:** Checkpoint attention only (less memory, more recompute)
- **Full:** Checkpoint all layers (more memory, less recompute)
- **Distributed:** Split activations across sequence dimension

## Core Code

- `megatron/core/transformer/transformer_config.py:296-310` - Strategy configuration
- Config options: `recompute_granularity`, `recompute_method`, `recompute_num_layers`

## Code Snippet

```python
# Configuration
config = TransformerConfig(
    # Checkpointing strategy
    recompute_granularity='selective',  # or 'full'
    recompute_method='uniform',         # or 'block'
    recompute_num_layers=None,          # None = all layers
)

# Selective checkpointing (typical)
# - Save: LayerNorm, dropout, linear outputs
# - Recompute: Attention, MLP activation functions
# - Memory saved: 30-40%
# - Compute overhead: 10-15%

# Full checkpointing (maximum memory saving)
# - Save: Only inputs to transformer blocks
# - Recompute: Everything in forward pass
# - Memory saved: 50-60%
# - Compute overhead: 20-25%
```

## Performance Impact

| Strategy | Memory Saved | Compute Overhead |
|----------|--------------|------------------|
| Selective | 30-40% | 10-15% |
| Full | 50-60% | 20-25% |
| Distributed | 40-50% | 12-18% |

**Trade-off:** Memory vs compute time
- Use selective for balanced approach
- Use full when memory-constrained
- Use distributed with long sequences

## Configuration Example

```python
config = TransformerConfig(
    recompute_granularity='selective',  # Good default
    recompute_method='uniform',
    
    # Combine with other memory optimizations
    deallocate_pipeline_outputs=True,
    use_distributed_optimizer=True,
)
```

## References

- Gradient checkpointing paper: [Training Deep Nets with Sublinear Memory Cost](https://arxiv.org/abs/1604.06174)
- Implementation: `megatron/core/transformer/`

