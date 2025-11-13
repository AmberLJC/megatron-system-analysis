# 10. 1F1B Pipeline Scheduling

## Context

Naive pipeline parallelism (GPipe) has massive bubbles where GPUs sit idle. With 4 pipeline stages:
- **GPipe:** Stage 1 does all forwards, then all backwards → other stages wait
- **Bubble time:** ~43% of total time wasted!

1F1B (One-Forward-One-Backward) interleaves forward and backward passes to keep all stages busy.

## Implementation

**Three phases:**
1. **Warmup:** Fill pipeline with (P-1) forwards where P = number of stages
2. **Steady state:** Alternate 1 forward, 1 backward for remaining microbatches
3. **Cooldown:** Drain pipeline with (P-1) backwards

**Key benefit:** During steady state, all stages are busy simultaneously!

### How It Works

```
Time →
Stage 0: F0 F1 F2 F3 B0 F4 B1 F5 B2 ... Bn-3 Bn-2 Bn-1
Stage 1:    F0 F1 F2 B0 F3 B1 F4 B2 ... Bn-4 Bn-3 Bn-2 Bn-1
Stage 2:       F0 F1 B0 F2 B1 F3 B2 ... Bn-5 Bn-4 Bn-3 Bn-2 Bn-1
Stage 3:          F0 B0 F1 B1 F2 B2 ... Bn-6 Bn-5 Bn-4 Bn-3 Bn-2 Bn-1
         |warmup|      steady state      |cooldown|
```

## Core Code

- `megatron/core/pipeline_parallel/schedules.py:1949-2304` - 1F1B implementation
- Lines 1990-2020: Warmup phase
- Lines 2022-2156: Steady state (1F1B loop)
- Lines 2158-2200: Cooldown phase

## Code Snippet

```python
# From schedules.py:1949-2304 (simplified)
def forward_backward_pipelining_with_interleaving(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    forward_only=False
):
    """
    1F1B pipeline scheduling with minimal bubbles.
    """
    pipeline_parallel_size = get_pipeline_model_parallel_world_size()
    pipeline_parallel_rank = get_pipeline_model_parallel_rank()
    
    # --- PHASE 1: WARMUP ---
    # Fill pipeline with (P-1) forward passes
    # This creates "in-flight" microbatches for pipeline
    num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
    
    input_tensors = []
    output_tensors = []
    
    for i in range(num_warmup_microbatches):
        # Get input from previous stage (or data iterator for first stage)
        input_tensor = recv_forward(tensor_shape, dtype, config)
        
        # Forward pass for this microbatch
        output_tensor = forward_step_func(data_iterator, model)
        
        # Send to next stage
        send_forward(output_tensor, tensor_shape, dtype, config)
        
        # Save for backward pass later
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
    
    # --- PHASE 2: STEADY STATE (1F1B) ---
    # Alternate: 1 forward, 1 backward
    # This is THE KEY - keeps all stages busy!
    num_microbatches_remaining = num_microbatches - num_warmup_microbatches
    
    for i in range(num_microbatches_remaining):
        # Get last forward output for backward
        last_iteration = (i == (num_microbatches_remaining - 1))
        
        # --- 1 FORWARD ---
        input_tensor = recv_forward(tensor_shape, dtype, config)
        output_tensor = forward_step_func(data_iterator, model)
        send_forward(output_tensor, tensor_shape, dtype, config)
        
        # --- 1 BACKWARD (for earlier microbatch) ---
        # While forward happens on microbatch N,
        # backward happens on microbatch (N - warmup_size)
        input_tensor_grad = recv_backward(tensor_shape, dtype, config)
        output_tensor_grad = backward_step_func(
            input_tensors.pop(0),   # Input from warmup
            output_tensors.pop(0),  # Output from warmup
            input_tensor_grad
        )
        send_backward(output_tensor_grad, tensor_shape, dtype, config)
        
        # Save new tensors for later backward
        input_tensors.append(input_tensor)
        output_tensors.append(output_tensor)
        
        # ^ At this point:
        #   - Forward for microbatch i is done
        #   - Backward for microbatch (i - warmup) is done
        #   - Next iteration will do forward i+1 and backward i-warmup+1
        #   - All stages are doing useful work simultaneously!
    
    # --- PHASE 3: COOLDOWN ---
    # Drain pipeline with remaining backwards
    for i in range(num_warmup_microbatches):
        input_tensor = input_tensors.pop(0)
        output_tensor = output_tensors.pop(0)
        
        input_tensor_grad = recv_backward(tensor_shape, dtype, config)
        output_tensor_grad = backward_step_func(
            input_tensor, output_tensor, input_tensor_grad
        )
        send_backward(output_tensor_grad, tensor_shape, dtype, config)
    
    # IMPORTANT: Synchronize gradients during cooldown bubble!
    # This hides gradient all-reduce in the bubble (see optimization #16)
    if not forward_only:
        enable_grad_sync()
        # Gradient all-reduce happens NOW (during cooldown bubble)
        # Result: "Free" communication!
```

## When to Use

**Always use with pipeline parallelism!** Don't use GPipe.

```python
# Pipeline parallelism configuration
pipeline_model_parallel_size = 8       # Number of pipeline stages
num_microbatches = 64                  # IMPORTANT: set to 4-8 × pipeline_stages

# 1F1B is default in Megatron-LM
# No special flag needed
```

### Microbatch Count

**Critical:** Set `num_microbatches = 4-8 × pipeline_stages`

- Too few microbatches: Large bubbles
- Too many: Memory overhead, diminishing returns
- Sweet spot: 4-8x pipeline stages

## Performance Impact

### Bubble Time Reduction

**Bubble fraction formula:**
- GPipe: `(P - 1) / P` where P = pipeline stages
- 1F1B: `(P - 1) / (2M)` where M = microbatches

**Example: P=4 stages, M=16 microbatches**
- GPipe bubble: (4-1)/4 = **75% wasted!**
- 1F1B bubble: (4-1)/(2×16) = **9.4%** 
- **Improvement:** 8x reduction in bubble time

### GPU Utilization

**GPipe:**
- Stage 0: 25% utilized (busy 1/4 of time)
- Stage 3: 25% utilized
- Average: 25%

**1F1B:**
- Stage 0: 90.6% utilized
- Stage 3: 90.6% utilized
- Average: 90.6%

**Result:** 3.6x better GPU utilization!

### End-to-End Measurements

**GPT-3 175B with PP=8, M=64:**
- GPipe bubble: (8-1)/8 = 87.5%
- 1F1B bubble: (8-1)/(2×64) = 5.5%
- Step time: GPipe 1800ms → 1F1B 520ms
- **Speedup:** 3.46x

## Troubleshooting

### High Bubble Time

**Symptoms:**
- Low GPU utilization
- Profiler shows long idle periods

**Causes:**
1. Too few microbatches (M < 4×P)
2. Imbalanced pipeline stages
3. Communication bottleneck

**Fix priority:**
1. Increase `num_microbatches` to 4-8 × PP
2. Balance layers across stages
3. Profile to find bottleneck

### OOM Errors

**Symptoms:**
- Out of memory during 1F1B
- Works with fewer microbatches

**Causes:**
- Too many microbatches (activations accumulate)
- Memory leak in activations

**Fix priority:**
1. Reduce `num_microbatches`
2. Enable activation deallocation (#20)
3. Reduce microbatch size

### Wrong Gradients

**Symptoms:**
- Training diverges with 1F1B
- Different results than data parallel

**Causes:**
- Gradient accumulation bug
- Wrong microbatch handling

**Fix priority:**
1. Verify gradient accumulation enabled
2. Check microbatch data loading
3. Test with PP=1 to isolate

## Related Optimizations

- **#11 Interleaved 1F1B:** Further reduces bubbles (half the bubble time!)
- **#16 Gradient Sync in Bubbles:** Hides gradient all-reduce in cooldown
- **#20 Activation Deallocation:** Reduces memory for more microbatches
- **#06 P2P Communication Modes:** Optimizes forward/backward communication

## Configuration Example

```python
# 1F1B pipeline configuration
training_args = {
    # Pipeline parallelism
    'pipeline_model_parallel_size': 8,  # Number of stages
    'num_microbatches': 64,             # 8 × 8 = 64 (good ratio)
    
    # Memory optimizations for more microbatches
    'deallocate_pipeline_outputs': True,  # Free activations (#20)
    'recompute_granularity': 'selective',  # Activation checkpointing
    
    # Communication optimizations
    'overlap_p2p_comm': True,  # For NVLink (#06)
}

# Verify microbatch ratio
pipeline_stages = 8
num_microbatches = 64
ratio = num_microbatches / pipeline_stages
print(f"Microbatch ratio: {ratio}x")  # Should be 4-8x
assert 4 <= ratio <= 8, "Adjust num_microbatches for optimal bubble time"
```

## Bubble Analysis

Calculate expected bubble time:

```python
def analyze_bubble_time(pipeline_stages, num_microbatches):
    """Calculate 1F1B bubble time"""
    bubble_fraction = (pipeline_stages - 1) / (2 * num_microbatches)
    efficiency = 1 - bubble_fraction
    
    print(f"Pipeline stages: {pipeline_stages}")
    print(f"Microbatches: {num_microbatches}")
    print(f"Bubble time: {bubble_fraction:.1%}")
    print(f"Efficiency: {efficiency:.1%}")
    
    # Recommendation
    if bubble_fraction > 0.10:
        recommended_mb = (pipeline_stages - 1) / (2 * 0.05)  # Target 5% bubble
        print(f"⚠ Consider increasing microbatches to {int(recommended_mb)}")
    else:
        print(f"✓ Good configuration!")

# Example
analyze_bubble_time(pipeline_stages=8, num_microbatches=64)
# Output:
# Pipeline stages: 8
# Microbatches: 64
# Bubble time: 5.5%
# Efficiency: 94.5%
# ✓ Good configuration!
```

## References

- GPipe paper: [GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- PipeDream (1F1B): [PipeDream: Generalized Pipeline Parallelism](https://arxiv.org/abs/1806.03377)
- Megatron-LM: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- Implementation: `megatron/core/pipeline_parallel/schedules.py`



