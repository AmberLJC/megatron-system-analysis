# Data Parallelism

## Context

After exhausting TP and PP, replicate model across remaining GPUs and shard data.

## Implementation

Each DP replica processes different data batch. Gradients synchronized via all-reduce or reduce-scatter.

```python
# Simple data parallelism
for rank in range(dp_size):
    # Each rank gets different batch
    batch = data[rank * batch_size : (rank + 1) * batch_size]
    
    # Forward + backward (independent)
    loss = model(batch)
    loss.backward()
    
    # Synchronize gradients
    all_reduce(gradients, group=dp_group)
    
    # Optimizer step (same on all ranks)
    optimizer.step()
```

## Code Location

- **DDP wrapper:** `megatron/core/distributed/distributed_data_parallel.py`
- **Gradient sync:** `megatron/core/distributed/param_and_grad_buffer.py`

## Performance Impact

- **Linear scaling** if communication overlapped
- **80-95% efficient** with gradient bucketing + overlap
- Enables scaling to thousands of GPUs

## When to Use

**Always** - Final dimension for scaling:
- After applying TP and PP
- **Total GPUs = TP × PP × DP**

**Configuration:**

```python
# Calculated automatically:
# data_parallel_size = world_size / (TP × PP)

tensor_model_parallel_size = 4
pipeline_model_parallel_size = 4
# DP size = 64 / (4 × 4) = 4
```

## Related Optimizations

- [Gradient Bucketing](01_communication_gradient_bucketing.md) - Enables efficient DP
- [Distributed Optimizer](19_memory_distributed_optimizer.md) - ZeRO-style DP

## References

- Paper: [Accurate, Large Minibatch SGD](https://arxiv.org/abs/1706.02677)
- [PyTorch DDP](https://pytorch.org/docs/stable/notes/ddp.html)

