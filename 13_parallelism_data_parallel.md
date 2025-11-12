# 13. Data Parallelism

## Context

After exhausting tensor and pipeline parallelism, scale further by replicating the model and sharding data across remaining GPUs.

Each replica processes different batch, gradients synchronized via all-reduce or reduce-scatter.

## Implementation

**Standard DDP:** All-reduce gradients (each rank gets full gradients)
**Distributed Optimizer (ZeRO):** Reduce-scatter gradients (each rank gets its shard)

## Core Code

- `megatron/core/distributed/distributed_data_parallel.py` - DDP wrapper
- `megatron/core/distributed/param_and_grad_buffer.py` - Gradient synchronization
- See optimization #01 for bucketing details
- See optimization #27 for distributed optimizer

## Code Snippet

```python
# Data parallelism with distributed optimizer
class DistributedDataParallel:
    def __init__(self, module, ddp_config):
        self.module = module
        self.ddp_config = ddp_config
        
        # Create gradient buffers with bucketing
        self.buffers = create_param_and_grad_buffers(
            module.parameters(),
            ddp_config
        )
    
    def forward(self, *inputs):
        # Disable gradient sync during forward/backward
        # Will sync in pipeline cooldown bubble
        return self.module(*inputs)
    
    def finish_grad_sync(self):
        # Gradient synchronization
        if self.ddp_config.use_distributed_optimizer:
            # Reduce-scatter: Each rank gets gradient shard
            for buffer in self.buffers:
                buffer.start_grad_sync()  # Async reduce-scatter
        else:
            # All-reduce: Each rank gets full gradients  
            for buffer in self.buffers:
                buffer.start_grad_sync()  # Async all-reduce
```

## When to Use

**Always** - Final dimension for scaling!

```python
# Total GPUs = TP × PP × DP
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 8  
data_parallel_size = 8  # DP fills remaining GPUs
# Total: 8 × 8 × 8 = 512 GPUs
```

## Performance Impact

**Scaling efficiency:**
- With overlap: 80-95% efficient (#01)
- Linear scaling up to thousands of GPUs

**Example:** GPT-3 175B, DP=64
- Per-GPU batch: 2M tokens
- Total batch: 128M tokens
- Gradient sync: ~3s (90% overlapped → 0.3s exposed)
- **Result:** 64x throughput at 95% efficiency!

## Configuration Example

```python
# Data parallelism (automatic based on total GPUs)
total_gpus = 512
tensor_model_parallel_size = 8
pipeline_model_parallel_size = 8
data_parallel_size = total_gpus // (8 * 8)  # = 8

ddp_config = DistributedDataParallelConfig(
    overlap_grad_reduce=True,            # Enable async (#01)
    use_distributed_optimizer=True,      # ZeRO-style (#27)
)
```

## References

- PyTorch DDP: [DistributedDataParallel](https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)
- ZeRO paper: [ZeRO](https://arxiv.org/abs/1910.02054)

