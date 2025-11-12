# CPU Offloading (Optimizer States)

## Context

Optimizer states (Adam: momentum + variance) require ~2x model memory. For 175B model, that's 350GB+ just for optimizer. GPU memory is expensive—CPU memory is cheaper and more abundant.

**Trade-off:** 5-10% training slowdown for ability to train models that otherwise wouldn't fit.

## Implementation

`HybridDeviceOptimizer` wrapper moves optimizer states to CPU with async transfers overlapped with computation.

```python
class HybridDeviceOptimizer:
    """Optimizer with CPU-offloaded states"""
    
    def __init__(self, optimizer, overlap_transfers=True):
        self.optimizer = optimizer
        self.cpu_states = {}
        
        # Move optimizer states to CPU
        for param_group in optimizer.param_groups:
            for param in param_group['params']:
                state = optimizer.state[param]
                # Move momentum and variance to CPU
                self.cpu_states[param] = {
                    'exp_avg': state['exp_avg'].to('cpu', non_blocking=True),
                    'exp_avg_sq': state['exp_avg_sq'].to('cpu', non_blocking=True)
                }
    
    def step(self):
        # Transfer states from CPU to GPU (async)
        for param, cpu_state in self.cpu_states.items():
            param.exp_avg = cpu_state['exp_avg'].to('cuda', non_blocking=True)
            param.exp_avg_sq = cpu_state['exp_avg_sq'].to('cuda', non_blocking=True)
        
        # Optimizer step on GPU
        self.optimizer.step()
        
        # Transfer back to CPU (async)
        for param, cpu_state in self.cpu_states.items():
            cpu_state['exp_avg'].copy_(param.exp_avg, non_blocking=True)
            cpu_state['exp_avg_sq'].copy_(param.exp_avg_sq, non_blocking=True)
```

## Code Location

- **Implementation:** `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py`
- **Flag:** `--use-cpu-offloading`

## Performance Impact

### Memory Savings

For 175B model (Adam optimizer):
- Parameters: 350 GB (FP16)
- Optimizer states: 700 GB (FP32 momentum + variance)
- **With CPU offloading:** Optimizer states moved to CPU
- **GPU memory freed:** 700 GB → can train on fewer GPUs!

### Cost

- CPU-GPU transfer bandwidth: ~20-30 GB/s (PCIe)
- Transfer time: ~30-35s for 700 GB per step
- Training slowdown: **5-10%**

## When to Use

**Use when:**
- Extreme memory constraints
- Can't fit model + optimizer on GPU
- Have sufficient CPU memory (typically cheaper)
- 5-10% slowdown acceptable

**Skip if:**
- Have enough GPU memory
- Need maximum throughput
- CPU-GPU bandwidth limited

**Configuration:**

```bash
# Enable CPU offloading
--use-cpu-offloading
```

## Related Optimizations

- [Distributed Optimizer](19_memory_distributed_optimizer.md) - Shard optimizer states first
- [Activation Checkpointing](15_memory_activation_checkpointing.md) - Reduce activation memory
- [FP8 Training](28_compute_fp8_training.md) - Reduce parameter/gradient memory

## References

- Paper: [ZeRO-Offload](https://arxiv.org/abs/2101.06840)
- [DeepSpeed ZeRO-Infinity](https://arxiv.org/abs/2104.07857)

