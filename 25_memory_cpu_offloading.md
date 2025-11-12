# 25. CPU Offloading (Optimizer States)

## Context

Optimizer states (Adam: momentum + variance) require ~2x model memory. For 175B model, that's 350GB+ just for optimizer!

## Implementation

`HybridDeviceOptimizer` wrapper moves optimizer states to CPU memory with async transfers overlapped with computation.

## Core Code

- `megatron/core/optimizer/cpu_offloading/hybrid_optimizer.py` - Main implementation
- Flag: `--use-cpu-offloading`

## Code Snippet

```python
class HybridDeviceOptimizer:
    """Offload optimizer states to CPU"""
    
    def __init__(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        
        # Move optimizer states to CPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cpu()  # Offload to CPU
    
    def step(self):
        # Copy gradients to CPU (async)
        self._async_copy_grads_to_cpu()
        
        # Optimizer step on CPU
        self.optimizer.step()
        
        # Copy parameters back to GPU (async)
        self._async_copy_params_to_gpu()
```

## When to Use

**Extreme memory constraints** only!

```bash
# Enable CPU offloading
python train.py --use-cpu-offloading
```

**Skip if:**
- Have enough GPU memory
- CPU-GPU bandwidth is limited

## Performance Impact

**Memory saved:** ~2x model size moved to CPU
- 175B model: ~350GB optimizer states offloaded

**Cost:** 5-10% training slowdown from CPU-GPU transfers

**Trade-off:** Memory vs speed
- Enables training models that otherwise wouldn't fit
- Only use as last resort

## References

- ZeRO-Offload paper: [ZeRO-Offload](https://arxiv.org/abs/2101.06840)
- Implementation: `megatron/core/optimizer/cpu_offloading/`

