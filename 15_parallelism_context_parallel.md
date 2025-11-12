# 15. Context Parallelism (SSM Models)

## Context

SSM models (Mamba) with long contexts need to split sequence dimension differently than attention models. Standard sequence parallelism (#03) is designed for attention mechanisms.

## Implementation

Splits sequence dimension for SSM-specific operations. Different from sequence parallelism (which is for attention).

Works with Mamba/SSM layers to enable longer contexts by distributing sequence processing across GPUs.

## Core Code

- `megatron/core/ssm/mamba_context_parallel.py` - Context parallelism for Mamba
- SSM-specific communication patterns

## Code Snippet

```python
# Context parallelism for SSM models
class MambaContextParallel:
    def __init__(self, context_parallel_size):
        self.cp_size = context_parallel_size
        self.cp_group = get_context_parallel_group()
    
    def forward(self, hidden_states, ssm_state):
        # Split sequence across CP group
        seq_len = hidden_states.size(0)
        local_seq_len = seq_len // self.cp_size
        
        # Each rank processes portion of sequence
        local_hidden = hidden_states[:local_seq_len]
        
        # SSM-specific processing with state passing
        # Different from attention (no Q/K/V all-gather)
        output = self.ssm_layer(local_hidden, ssm_state)
        
        # Gather results if needed
        if self.gather_output:
            output = all_gather_along_sequence_dim(output, self.cp_group)
        
        return output
```

## When to Use

**Mamba/SSM models only!**
- Long contexts (>4096)
- `context_parallel_size > 1`

```python
context_parallel_size = 4  # For SSM models
sequence_parallel = False  # Don't use standard SP with CP
```

**Skip if:**
- Not using SSM models
- Standard attention (use sequence parallelism #03 instead)

## Performance Impact

**Memory savings:** Similar to sequence parallelism
- Activations: `1/CP` size per rank
- Enables longer contexts for SSM models

**Example:** Mamba 7B, context=32K, CP=4
- Per-rank context: 32K / 4 = 8K tokens
- Memory saved: 3/4 of activation memory

## Configuration Example

```python
# Context parallelism for Mamba
config = TransformerConfig(
    # Mamba-specific
    use_mamba_ssm=True,
    context_parallel_size=4,
    
    # Don't mix with standard SP
    sequence_parallel=False,
)
```

## References

- Mamba paper: [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- Implementation: `megatron/core/ssm/`

