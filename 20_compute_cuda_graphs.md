# CUDA Graphs

## Context

Each CUDA kernel launch has ~5-10μs CPU overhead. For 96-layer transformer with ~500 kernels per forward-backward, that's 2.5-5ms of pure overhead (5-10% of step time).

**Solution:** Capture entire forward/backward as static graph, replay with single ~100μs call.

## Implementation

Three-phase process:
1. **Recording** (first step): Execute normally, record graph intent
2. **Capture** (end of first step): Capture all kernels into static graph
3. **Replay** (subsequent steps): Replay with single CPU call

```python
class CUDAGraphModule:
    """Module wrapper with CUDA graph capture"""
    
    def __init__(self, module):
        self.module = module
        self.graph = None
        self.static_inputs = None
        self.static_outputs = None
    
    def capture(self, example_inputs):
        """Capture forward pass as CUDA graph"""
        # Warmup
        for _ in range(3):
            _ = self.module(*example_inputs)
        
        # Allocate static I/O buffers
        self.static_inputs = [x.clone() for x in example_inputs]
        
        # Capture
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_outputs = self.module(*self.static_inputs)
    
    def forward(self, *inputs):
        """Replay captured graph"""
        # Copy inputs to static buffers
        for static, dynamic in zip(self.static_inputs, inputs):
            static.copy_(dynamic)
        
        # Replay graph (single API call for 500+ kernels!)
        self.graph.replay()
        
        return self.static_outputs
```

## Code Location

- **Implementation:** `megatron/core/transformer/cuda_graphs.py` lines 177-1024
- **Global tracking:** Lines 177-352
- **Per-layer management:** Lines 522-1024

## Performance Impact

### Kernel Launch Overhead

| Configuration | Overhead | Notes |
|---------------|----------|-------|
| No graphs | 24ms | 500 kernels × 50μs |
| With graphs | 0.8ms | Single replay call |
| **Savings** | **23.2ms** | **30x reduction** |

### End-to-End

For 96-layer model:
- Step time: 250ms → 242ms (3% faster)
- Smaller models see bigger gains (8% for 7B model)

## When to Use

**Use when:**
- Static shapes (fixed batch size, sequence length)
- Long training runs (amortize capture cost)
- CPU overhead visible in profiling

**Skip if:**
- Variable sequence lengths
- Frequent model changes
- Debugging (graphs harder to debug)

**Configuration:**

```python
config = TransformerConfig(
    cuda_graph=True,
    cuda_graph_scope='per_layer',  # or 'full_iteration'
    cuda_graph_warmup_steps=2,
)
```

## Constraints

**Requirements:**
- Static tensor shapes
- No CPU synchronization in captured region
- No dynamic control flow

## Related Optimizations

- [Kernel Fusion](21_compute_bias_activation_fusion.md) - Fewer kernels → more graph benefit
- [Gradient Accumulation Fusion](26_compute_grad_accumulation_fusion.md) - Works with graphs

## References

- [CUDA Graphs Documentation](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [PyTorch CUDA Graphs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs)

