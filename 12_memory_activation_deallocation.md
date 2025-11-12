# Activation Deallocation (Pipeline)

## Context

After sending activations to the next pipeline stage, the output tensor sits in memory unused until backward pass. For 96-layer model with 8 microbatches, this wastes GBs of memory.

**Key insight:** Replace activation `.data` with tiny tensor after sending, keeping `.grad_fn` for autograd. Use custom C++ engine call during backward to bypass shape checks.

## Implementation

1. **After send:** Replace `out.data` with 1-element tensor
2. **Keep:** `out.grad_fn` intact (autograd graph)  
3. **During backward:** Call C++ engine directly (bypasses shape check)
4. **Result:** Frees GBs, autograd still works!

## Core Code

Located in `megatron/core/pipeline_parallel/schedules.py`:

```python
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    """Pseudo-deallocate by replacing .data with tiny tensor"""
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    
    # KEY TRICK: Replace huge activation with 1-element tensor
    # Keep .grad_fn so autograd graph intact!
    original_shape = out.shape
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
    # ^ Frees GBs of memory!
    
    return original_shape


def custom_backward(output, grad_output):
    """Bypass PyTorch's shape check by calling C++ engine directly"""
    # PyTorch backward() checks: output.shape == grad_output.shape
    # But we deallocated output.data to shape (1,)!
    
    # Solution: Call C++ autograd engine directly
    # C++ engine only needs .grad_fn, not .data!
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),  # Full shape from recv
        keep_graph=False,
        create_graph=False,
        allow_unreachable=True,
        accumulate_grad=True
    )
    # ^ Works because autograd graph is intact, only .data was freed
```

### Usage in Pipeline Schedule

```python
# In forward_step_helper
def forward_step_helper(microbatch_id, ...):
    output_tensor = model.forward(input_tensor)
    
    # Send to next stage
    send_forward(output_tensor)
    
    # Deallocate activation memory
    original_shape = deallocate_output_tensor(
        output_tensor,
        deallocate_pipeline_outputs=True
    )
    
    return output_tensor  # Now only 1 element!


# In backward_step_helper
def backward_step_helper(microbatch_id, ...):
    # Receive gradient from next stage
    grad_output = recv_backward(shape=original_shape)
    
    # Custom backward (bypasses shape check)
    custom_backward(output_tensor, grad_output)
```

## Code Location

- **Deallocation:** `megatron/core/pipeline_parallel/schedules.py` lines 135-147
- **Custom backward:** `megatron/core/pipeline_parallel/schedules.py` lines 149-178
- **Usage:** Integrated in 1F1B schedule (lines 1949-2304)

## When to Use

**Always use with pipeline parallelism:**
- PP size > 1
- Any number of microbatches
- Frees memory for more microbatches or larger batch sizes

**Configuration:**

```python
config = TransformerConfig(
    pipeline_model_parallel_size=4,      # PP > 1
    deallocate_pipeline_outputs=True,    # Enable!
)
```

## Performance Impact

### Memory Savings

For typical transformer layer:
- Activation size: ~50 MB per layer per microbatch
- 8 microbatches in flight: 8 × 50 MB = 400 MB per layer
- 24 layers per stage: 24 × 400 MB = **9.6 GB saved per stage**

For 175B model, PP=8:
- Per stage savings: ~10 GB
- Enables 2-4x more microbatches → better pipeline efficiency

## Why This Works

### PyTorch Autograd Architecture

```
Tensor object:
├─ .data: Storage (the actual values) ← We free this!
├─ .grad_fn: Autograd node (computation graph) ← Keep this!
└─ .grad: Gradient storage

During backward:
1. PyTorch backward() checks .data shape (unnecessary!)
2. C++ engine only needs .grad_fn (necessary)
3. Solution: Skip Python backward(), call C++ directly
```

### Memory Timeline

```
Without deallocation:
Forward  → |====act0====|====act1====|====act2====|...
Backward ← |====act0====|====act1====|====act2====|... (all in memory!)
Peak memory: Sum of all activations

With deallocation:
Forward  → |====act0====| (freed after send)
           |====act1====| (freed after send)
           |====act2====| (freed after send)
Backward ← (activations not needed, only gradients)
Peak memory: 1 activation at a time
```

## Troubleshooting

**Symptom:** "Shape mismatch" error in backward

**Cause:** Not using `custom_backward`, using standard `backward()`

**Fix:**
```python
# DON'T do this after deallocation:
output_tensor.backward(grad_output)  # Will fail!

# DO this:
custom_backward(output_tensor, grad_output)  # Works!
```

**Symptom:** Still running out of memory

**Cause:** Other memory bottlenecks

**Fix:**
```python
# Combine with other optimizations:
config.deallocate_pipeline_outputs = True          # This one
config.recompute_granularity = 'selective'         # + checkpointing
config.use_distributed_optimizer = True            # + param sharding
```

## Related Optimizations

- [1F1B Scheduling](30_parallelism_1f1b.md) - Uses activation deallocation
- [Dynamic Checkpointing](16_memory_dynamic_checkpointing.md) - Complements deallocation
- [Activation Checkpointing](15_memory_activation_checkpointing.md) - Alternative/complementary

## References

- PyTorch Autograd: [Automatic Differentiation](https://pytorch.org/docs/stable/notes/autograd.html)
- [Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

