# 27. Distributed Optimizer (ZeRO)

## Context

Standard optimizer stores full optimizer state on each data-parallel rank. For 175B model with Adam, that's 350GB × DP ranks of redundant memory!

## Implementation

Shards optimizer state across data-parallel ranks. Each rank only stores its shard, all-gathers parameters when needed.

**ZeRO stages:**
- **Stage 1:** Shard optimizer states
- **Stage 2:** + Shard gradients (reduce-scatter)
- **Stage 3:** + Shard parameters (all-gather on demand)

Megatron implements Stage 2 (shard optimizer + gradients).

## Core Code

- `megatron/core/optimizer/distrib_optimizer.py` (3500+ lines) - Main implementation
- Features: Parameter sharding, gradient reduce-scatter, optimizer state sharding
- `megatron/core/distributed/param_and_grad_buffer.py:221-329` - Parameter all-gather overlap

## Code Snippet

```python
# Distributed optimizer (ZeRO Stage 2)
class DistributedOptimizer:
    def __init__(self, optimizer, model_chunks, config):
        self.optimizer = optimizer
        self.dp_size = get_data_parallel_world_size()
        self.dp_rank = get_data_parallel_rank()
        
        # Shard parameters across DP group
        self.param_shards = self._shard_parameters(model_chunks)
        
        # Each rank stores only its shard of optimizer states
        # Memory: (params + states) / DP instead of (params + states)
    
    def _shard_parameters(self, model_chunks):
        """Shard parameters across DP ranks"""
        all_params = []
        for chunk in model_chunks:
            all_params.extend(chunk.parameters())
        
        # Divide parameters among DP ranks
        shard_size = len(all_params) // self.dp_size
        start = self.dp_rank * shard_size
        end = start + shard_size
        
        return all_params[start:end]  # This rank's shard
    
    def step(self):
        """Optimizer step with parameter all-gather"""
        
        # Step 1: Update local parameter shard
        self.optimizer.step()  # Updates only this rank's shard
        
        # Step 2: All-gather updated parameters (for next forward)
        # Can overlap with forward computation!
        for bucket in self.param_buckets:
            torch.distributed.all_gather_into_tensor(
                bucket.param_data,      # Output: full parameters
                bucket.local_shard,     # Input: this rank's shard
                group=self.dp_group,
                async_op=True  # Overlap with forward!
            )


# Memory breakdown
# Standard DDP (no sharding):
# - Parameters: 350GB (175B × 2 bytes)
# - Gradients: 350GB
# - Optimizer states: 700GB (2x for Adam)
# Total per rank: 1.4TB

# Distributed optimizer (DP=8):
# - Parameters: 350GB (replicated for forward)
# - Gradients: 44GB (sharded via reduce-scatter)
# - Optimizer states: 88GB (sharded, 700GB / 8)
# Total per rank: 482GB
# Saved: 918GB per rank (66% reduction!)
```

## When to Use

**Always enable with DP > 1!**

```python
ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,  # Enable ZeRO-style sharding
    overlap_param_gather=True,       # Overlap all-gather with forward
)
```

**Skip if:**
- DP = 1 (no benefit)
- Already have sufficient memory

## Performance Impact

**Memory saved:** O(DP) reduction in optimizer state
- DP=8: 8x reduction → 87.5% memory saved
- DP=64: 64x reduction → 98.4% memory saved

**Communication:** Same cost as standard DDP
- Reduce-scatter (distributed) ≈ All-reduce (standard)
- Parameter all-gather adds cost but can be overlapped

**Example:** GPT-3 175B, DP=8
- Standard: 1.4TB per rank
- Distributed: 482GB per rank
- **Saved:** 918GB per rank (enables training on smaller GPUs!)

## Troubleshooting

### OOM Even with DistOpt

**Causes:**
- Activations dominating memory
- Parameter all-gather not overlapped

**Fix:**
1. Enable activation checkpointing (#23)
2. Verify `overlap_param_gather=True`
3. Reduce microbatch size

### Slower Than Expected

**Causes:**
- Parameter all-gather on critical path
- Communication not overlapped

**Fix:**
1. Enable overlap: `overlap_param_gather=True`
2. Profile to verify overlap
3. Check network bandwidth

## Related Optimizations

- **#01 Gradient Bucketing:** Works with reduce-scatter
- **#18 Gradient Buffer Padding:** Applied to sharded gradients
- **#23 Activation Checkpointing:** Combine for maximum memory savings

## Configuration Example

```python
from megatron.core.distributed import DistributedDataParallelConfig

# Distributed optimizer configuration
ddp_config = DistributedDataParallelConfig(
    # ZeRO-style sharding
    use_distributed_optimizer=True,
    overlap_param_gather=True,       # Overlap all-gather with forward
    
    # Gradient communication
    overlap_grad_reduce=True,        # Overlap reduce-scatter with backward
    bucket_size=40000000,
    
    # Memory optimizations
    reduce_scatter_with_fp32_accumulation=True,  # Better precision
)
```

## Memory Scaling

As model size increases, distributed optimizer becomes critical:

| Model Size | Standard DDP | Distributed Opt (DP=8) | Savings |
|------------|--------------|------------------------|---------|
| 7B | 70 GB | 22 GB | 69% |
| 70B | 560 GB | 140 GB | 75% |
| 175B | 1.4 TB | 482 GB | 66% |
| 540B | 4.3 TB | 1.3 TB | 70% |

Without distributed optimizer, many models simply don't fit!

## References

- ZeRO paper: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- DeepSpeed ZeRO: [DeepSpeed Blog](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)
- Implementation: `megatron/core/optimizer/distrib_optimizer.py`

