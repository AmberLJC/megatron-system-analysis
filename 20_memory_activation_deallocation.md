# 20. Activation Deallocation (Pipeline)

## Context

After sending activations to next pipeline stage, the output tensor sits in memory unused until backward pass. For 96-layer model with 8 microbatches, this wastes GBs!

**Problem:** Pipeline fills with activations during warmup, all held in memory simultaneously.

## Implementation

Replaces activation `.data` with 1-element tensor after sending, keeping `.grad_fn` for autograd. Uses custom C++ autograd engine call to bypass shape checks during backward.

**Key trick:** PyTorch backward checks `output.shape == grad_output.shape`, but we deallocated output! Solution: Call C++ engine directly (doesn't check shapes).

## Core Code

- `megatron/core/pipeline_parallel/schedules.py:135-179` - Deallocation system
- Line 146: `deallocate_output_tensor()` - Replace .data
- Lines 149-178: `custom_backward()` - Call C++ engine

## Code Snippet

```python
# From schedules.py:135-179
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    """Pseudo-deallocate by replacing .data with tiny tensor"""
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    
    # Save autograd metadata
    grad_fn = out.grad_fn
    requires_grad = out.requires_grad
    
    # Replace huge activation tensor with 1-element tensor
    # Keep .grad_fn for autograd graph!
    out.data = torch.empty(
        (1,), 
        device=out.device,
        dtype=out.dtype,
        requires_grad=requires_grad
    )
    out.grad_fn = grad_fn  # Restore autograd connection
    # ^ Frees GBs of memory per tensor while keeping autograd intact!


def custom_backward(output, grad_output):
    """Bypass PyTorch's shape check by calling C++ engine directly"""
    
    # PyTorch's backward() checks: output.shape == grad_output.shape
    # But we deallocated output.data to shape (1,)!
    # Solution: Call C++ autograd engine directly (no shape check)
    
    import torch.autograd
    
    # C++ engine only needs .grad_fn, not .data
    torch.autograd.Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),  # Full shape gradient
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True
    )
    # ^ Works because autograd graph is intact, only .data was freed!


# Usage in pipeline schedule
def forward_backward_with_deallocation():
    # Forward pass
    output = forward_step(...)
    
    # Send to next stage
    send_forward(output)
    
    # Deallocate activation (keep autograd)
    deallocate_output_tensor(output, deallocate_pipeline_outputs=True)
    # ^ Memory freed immediately!
    
    # ... later, in backward pass ...
    
    # Receive gradient from next stage
    output_grad = recv_backward(...)
    
    # Custom backward (bypasses shape check)
    custom_backward(output, output_grad)
```

## When to Use

**Always with pipeline parallelism!**

```python
# Enable activation deallocation
deallocate_pipeline_outputs = True

# Automatic in Megatron's pipeline schedules
config = TransformerConfig(
    deallocate_pipeline_outputs=True,
)
```

## Performance Impact

**Memory saved:** ~50MB per layer per microbatch = GBs total
- 96-layer model, 8 microbatches: 96 × 8 × 50MB = 38.4GB saved
- Enables larger microbatch sizes or more microbatches

**Example:** GPT-3 175B, PP=8, 16 microbatches
- Activation size per stage: 12 layers × 50MB = 600MB
- Without deallocation: 16 × 600MB = 9.6GB held
- With deallocation: ~600MB peak (only current microbatch)
- **Saved:** 9GB per pipeline stage!

## Troubleshooting

### Crashes in Backward

**Symptoms:**
- Crashes during backward pass
- Shape mismatch errors

**Causes:**
- Custom backward not working
- Autograd graph corrupted

**Fix:**
1. Disable deallocation to verify issue
2. Check PyTorch version (needs ≥1.10)
3. Verify grad_fn preserved

## Related Optimizations

- **#10 1F1B Pipeline:** Deallocation works with 1F1B scheduling
- **#23 Activation Checkpointing:** Combine for even more memory savings

## Configuration Example

```python
# Pipeline with activation deallocation
config = TransformerConfig(
    pipeline_model_parallel_size=8,
    num_microbatches=64,
    deallocate_pipeline_outputs=True,  # Enable deallocation
)
```

## References

- PyTorch autograd internals: [Autograd Engine](https://pytorch.org/docs/stable/notes/extending.html)
- Implementation: `megatron/core/pipeline_parallel/schedules.py`

