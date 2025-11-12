# Distributed Optimizer (ZeRO)

## Context

Standard optimizer stores full optimizer state on each data-parallel rank. For 175B model with Adam:
- Parameters: 350 GB × DP ranks
- Gradients: 350 GB × DP ranks  
- Optimizer states: 700 GB × DP ranks (momentum + variance)

**Total redundancy:** Each rank stores everything!

## Implementation

Shards optimizer state across data-parallel ranks. Each rank only stores 1/DP of optimizer state, all-gathers parameters when needed.

```python
class DistributedOptimizer:
    """ZeRO-style optimizer with state sharding"""
    
    def __init__(self, optimizer, model_params, dp_group):
        self.optimizer = optimizer
        self.dp_group = dp_group
        self.dp_size = dp_group.size()
        self.dp_rank = dp_group.rank()
        
        # Shard parameters across DP ranks
        params_per_rank = len(model_params) // self.dp_size
        start = self.dp_rank * params_per_rank
        end = (self.dp_rank + 1) * params_per_rank
        
        # Each rank only optimizes its shard
        self.local_params = model_params[start:end]
        
        # Optimizer states only for local params (not all!)
        self.optimizer = optimizer(self.local_params)
    
    def step(self):
        """Optimizer step with all-gather"""
        # 1. Optimizer step on local shard
        self.optimizer.step()
        
        # 2. All-gather updated parameters
        for param in self.local_params:
            torch.distributed.all_gather_into_tensor(
                full_param, param, group=self.dp_group
            )
    
    def backward_hook(self, param, grad):
        """Reduce-scatter gradients instead of all-reduce"""
        # Reduce-scatter: Each rank gets its shard
        torch.distributed.reduce_scatter_tensor(
            local_grad_shard, grad, group=self.dp_group
        )
        return local_grad_shard
```

## Code Location

- **Main implementation:** `megatron/core/optimizer/distrib_optimizer.py` (3500+ lines)
- **Parameter all-gather:** `megatron/core/distributed/param_and_grad_buffer.py` lines 221-329

## Performance Impact

### Memory Savings

For 175B model, DP=8:

**Standard optimizer:**
```
Per rank:
- Parameters: 350 GB
- Gradients: 350 GB
- Optimizer states: 700 GB
Total: 1400 GB per rank
```

**Distributed optimizer:**
```
Per rank:
- Parameters: 350 GB (sharded, all-gathered on demand)
- Gradients: 350 GB (sharded after reduce-scatter)
- Optimizer states: 700 / 8 = 87.5 GB (sharded!)
Total: 787.5 GB per rank
Savings: 612.5 GB (44%!)
```

### Scaling Formula

```
Memory per rank = (params + grads + opt_states/DP)
Savings = opt_states × (DP-1) / DP

For DP=8: Savings = 87.5% of optimizer state memory
```

## When to Use

**Use when:**
- Data-parallel size ≥ 4
- Memory-constrained scenarios
- Want to maximize batch size

**Configuration:**

```python
ddp_config = DistributedDataParallelConfig(
    use_distributed_optimizer=True,  # Enable ZeRO
    overlap_param_gather=True,       # Overlap all-gather with forward
)
```

**Skip if:**
- DP size = 1 (no benefit)
- Already have sufficient memory
- Want simplest setup (regular DDP)

## Communication Pattern

**Regular DDP:**
```
Backward: All-reduce gradients (full tensors)
Optimizer: Local step (no communication)
```

**Distributed Optimizer:**
```
Backward: Reduce-scatter gradients (sharded)
Optimizer: Local step on shard
Forward: All-gather parameters (sharded → full)
```

**Same communication volume, different pattern!**

## Related Optimizations

- [Gradient Bucketing](01_communication_gradient_bucketing.md) - Uses reduce-scatter with DistOpt
- [NCCL Symmetric Memory](02_communication_nccl_symmetric_memory.md) - Speeds up all-gather
- [Hierarchical Communication](05_communication_hierarchical.md) - Multi-instance DistOpt

## References

- Paper: [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)
- [PyTorch FSDP](https://pytorch.org/docs/stable/fsdp.html)

