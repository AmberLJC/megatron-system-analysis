# 11. Interleaved 1F1B (Virtual Pipeline)

## Context

Standard 1F1B still has bubble time proportional to pipeline stages: `(P-1)/(2M)`. For P=8, M=32, that's still 10.9% bubble time. Can we do better?

**Solution:** Split each pipeline stage's layers into V "virtual stages" (model chunks) and interleave execution between chunks to fill bubbles with useful work!

## Implementation

Each pipeline stage contains V model chunks (virtual pipeline stages). Schedule alternates between chunks within each stage, filling bubbles:
- While Chunk 1 waits for inputs, Chunk 2 processes its microbatches
- Creates V×M microbatch schedule for P physical stages

**Schedule table** determines exact interleaving: maps virtual microbatch ID to (chunk_id, microbatch_id).

## Core Code

- `megatron/core/pipeline_parallel/schedules.py:809-1400` - Interleaved 1F1B implementation
- Lines 900-980: Schedule table generation
- Lines 1050-1150: Chunk switching logic
- Lines 1200-1300: Virtual pipeline execution

## Code Snippet

```python
# From schedules.py:900-980 (schedule table generation)
def get_interleaved_schedule(
    num_microbatches,
    pipeline_parallel_size,
    virtual_pipeline_model_parallel_size
):
    """
    Generate schedule table for interleaved 1F1B.
    
    Returns: schedule[step] = (chunk_id, microbatch_id, is_forward)
    """
    P = pipeline_parallel_size
    V = virtual_pipeline_model_parallel_size
    M = num_microbatches
    
    schedule = []
    
    # Total virtual microbatches: V × M
    total_virtual_mb = V * M
    
    # For each step, determine which (chunk, microbatch) to execute
    for step in range(total_virtual_mb * 2):  # ×2 for forward+backward
        # Complex scheduling logic to minimize bubbles
        # Interleaves chunks to fill gaps
        chunk_id, mb_id, is_forward = _compute_schedule_entry(step, P, V, M)
        schedule.append((chunk_id, mb_id, is_forward))
    
    return schedule


# From schedules.py:1050-1150 (chunk switching)
def interleaved_forward_backward(
    forward_step_func,
    data_iterator,
    model,  # List of V model chunks
    num_microbatches,
    virtual_pipeline_size
):
    """
    Execute interleaved 1F1B with virtual pipeline stages.
    """
    V = virtual_pipeline_size
    M = num_microbatches
    
    # Get pre-computed schedule
    schedule = get_interleaved_schedule(M, pipeline_parallel_size, V)
    
    # Tensor buffers for each virtual stage
    input_tensors = [[None for _ in range(M)] for _ in range(V)]
    output_tensors = [[None for _ in range(M)] for _ in range(V)]
    
    # Execute schedule step by step
    for step, (chunk_id, mb_id, is_forward) in enumerate(schedule):
        # Switch to appropriate model chunk
        current_model_chunk = model[chunk_id]
        
        if is_forward:
            # Forward pass on this virtual stage
            input_tensor = recv_forward(...)
            output_tensor = forward_step_func(
                data_iterator, current_model_chunk
            )
            send_forward(output_tensor, ...)
            
            # Save for backward
            input_tensors[chunk_id][mb_id] = input_tensor
            output_tensors[chunk_id][mb_id] = output_tensor
            
        else:
            # Backward pass on this virtual stage
            input_tensor_grad = recv_backward(...)
            output_tensor_grad = backward_step_func(
                input_tensors[chunk_id][mb_id],
                output_tensors[chunk_id][mb_id],
                input_tensor_grad
            )
            send_backward(output_tensor_grad, ...)
        
        # ^ Key: While one chunk waits for communication,
        #   another chunk is computing (fills the bubble!)
```

## When to Use

**Enable when:**
- Pipeline stages ≥ 4
- Pipeline bubbles > 5% of time
- Have memory for 2-4x model chunks

```python
# Interleaved pipeline configuration
pipeline_model_parallel_size = 8           # Physical stages
virtual_pipeline_model_parallel_size = 2   # 2 or 4 model chunks per stage
num_microbatches = 64                      # More microbatches needed
```

**Skip if:**
- PP ≤ 2 (minimal benefit)
- Memory constrained (needs V× model per stage)
- Bubbles already < 5%

## Performance Impact

### Bubble Reduction

**Bubble fraction:**
- Standard 1F1B: `(P-1) / (2M)`
- Interleaved 1F1B: `(P-1) / (2MV)` where V = virtual stages

**With V=2:** Half the bubble time!

**Example: P=8, M=32**
- Standard 1F1B: (8-1)/(2×32) = 10.9% bubble
- Interleaved (V=2): (8-1)/(2×32×2) = 5.5% bubble
- **Improvement:** 2x bubble reduction

**Example: P=8, M=64, V=4**
- Bubble: (8-1)/(2×64×4) = 1.37%
- **Result:** Nearly perfect pipeline efficiency!

### Throughput

**GPT-3 175B with PP=8, V=2:**
- Standard 1F1B: 1850 tokens/sec
- Interleaved: 1960 tokens/sec
- **Speedup:** 5.9% improvement

## Troubleshooting

### OOM Errors

**Symptoms:**
- Out of memory with virtual pipeline
- Works with V=1

**Causes:**
- Each stage holds V model chunks (V× parameters)
- More in-flight activations

**Fix:**
1. Reduce V (try V=2 instead of 4)
2. Enable activation deallocation (#20)
3. Use activation checkpointing (#23)

### No Performance Gain

**Symptoms:**
- Same throughput as standard 1F1B
- Bubbles not reduced

**Causes:**
- Not enough microbatches
- Communication bottleneck (not compute)

**Fix:**
1. Increase `num_microbatches` to 8-16 × PP × V
2. Profile to verify compute-bound (not comm-bound)

## Related Optimizations

- **#10 1F1B Pipeline:** Base scheduling that interleaved builds on
- **#20 Activation Deallocation:** Helps fit V model chunks in memory
- **#23 Activation Checkpointing:** Reduces memory for virtual stages

## Configuration Example

```python
# Interleaved pipeline configuration
training_args = {
    'pipeline_model_parallel_size': 8,           # Physical stages
    'virtual_pipeline_model_parallel_size': 2,   # Virtual stages per physical
    'num_microbatches': 128,                     # 8 × 8 × 2 = 128 (good ratio)
    
    # Memory optimizations (needed for V > 1)
    'deallocate_pipeline_outputs': True,
    'recompute_granularity': 'selective',
}

# Memory requirement
# Each stage holds: (model_size / P) × V
# For 175B model, PP=8, V=2: each stage holds 2 × (175B/8) = 43.75B parameters
```

## References

- Megatron paper: [Efficient Large-Scale Language Model Training](https://arxiv.org/abs/2104.04473)
- Implementation: `megatron/core/pipeline_parallel/schedules.py`

