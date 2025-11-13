# 09. Expert Parallelism Communication (MoE)

## Context

MoE (Mixture of Experts) models need to route tokens to different experts across GPUs. A token on GPU 0 might need Expert 5 on GPU 3, requiring specialized communication patterns.

Traditional approach would all-gather all tokens to all GPUs (massive memory + communication), then mask unused tokens. Very inefficient!

## Implementation

Three dispatcher modes optimized for different expert parallelism (EP) sizes:
1. **AllGather Dispatcher:** For small EP (≤4) - gather tokens, mask locally
2. **AlltoAll Dispatcher:** For larger EP (>4) - direct token routing
3. **Fused A2A (DeepEP):** Custom CUDA buffers for maximum efficiency

### How They Work

**AllGather (EP ≤ 4):**
1. Router determines which expert each token needs
2. All-gather all tokens to all ranks
3. Each rank masks to extract its expert's tokens
4. Process expert computation
5. Reduce-scatter results back

**AlltoAll (EP > 4):**
1. Router determines token distribution
2. All-to-all sends tokens directly to target ranks
3. Process expert computation on received tokens
4. All-to-all sends results back to origin ranks

**Fused A2A (DeepEP):**
1. Custom CUDA kernels for routing decision
2. Pre-allocated buffers for all-to-all
3. Fused computation with communication
4. Minimal memory overhead

## Core Code

- `megatron/core/transformer/moe/token_dispatcher.py:197-330` - AllGather dispatcher
- `megatron/core/transformer/moe/token_dispatcher.py:333-400` - AlltoAll dispatcher
- `megatron/core/transformer/moe/fused_a2a.py` (200 lines) - Fused implementation
- `megatron/core/transformer/moe/router.py` - Token routing logic

## Code Snippet

```python
# From token_dispatcher.py:197-330
class AllGatherTokenDispatcher:
    """
    Token dispatcher using all-gather for small EP sizes.
    Best for EP ≤ 4.
    """
    
    def token_permutation(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, hidden_size]
        routing_weights: torch.Tensor,  # [num_tokens, num_experts]
    ):
        """
        Route tokens to experts using all-gather.
        """
        # Step 1: Router determines expert assignments
        # routing_weights[i, j] = probability token i goes to expert j
        selected_experts = torch.argmax(routing_weights, dim=-1)
        
        # Step 2: All-gather tokens to all ranks
        # Each rank gets ALL tokens (memory intensive for large EP!)
        gathered_hidden_states = torch.empty(
            (self.ep_size, *hidden_states.shape),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        torch.distributed.all_gather_into_tensor(
            gathered_hidden_states,
            hidden_states,
            group=self.ep_group
        )
        # Now every rank has all tokens!
        
        # Step 3: Mask to get tokens for THIS rank's expert
        # Rank 0 processes Expert 0, Rank 1 processes Expert 1, etc.
        my_expert_id = self.ep_rank
        mask = (selected_experts == my_expert_id)
        my_tokens = gathered_hidden_states[:, mask, :]
        
        # Step 4: Process expert (happens outside dispatcher)
        # expert_output = self.experts[my_expert_id](my_tokens)
        
        return my_tokens, mask


# From token_dispatcher.py:333-400
class AlltoAllTokenDispatcher:
    """
    Token dispatcher using all-to-all for larger EP sizes.
    Best for EP > 4. More efficient - direct routing!
    """
    
    def token_permutation(
        self,
        hidden_states: torch.Tensor,  # [num_tokens, hidden_size]
        routing_weights: torch.Tensor,  # [num_tokens, num_experts]
    ):
        """
        Route tokens to experts using all-to-all communication.
        """
        # Step 1: Determine token counts per expert
        selected_experts = torch.argmax(routing_weights, dim=-1)
        
        # Count tokens for each expert
        tokens_per_expert = torch.zeros(
            self.num_experts, dtype=torch.long, device='cuda'
        )
        for expert_id in range(self.num_experts):
            tokens_per_expert[expert_id] = (selected_experts == expert_id).sum()
        
        # Step 2: All-to-all to exchange token counts
        # Each rank learns how many tokens to send/receive from each other rank
        send_counts = tokens_per_expert  # What I send to each rank
        recv_counts = torch.zeros_like(send_counts)
        
        torch.distributed.all_to_all_single(
            recv_counts, send_counts, group=self.ep_group
        )
        
        # Step 3: Prepare send buffer (sort tokens by target expert)
        send_buffer = torch.empty(
            (tokens_per_expert.sum(), hidden_states.size(-1)),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        offset = 0
        for expert_id in range(self.num_experts):
            mask = (selected_experts == expert_id)
            count = mask.sum()
            send_buffer[offset:offset+count] = hidden_states[mask]
            offset += count
        
        # Step 4: All-to-all to route tokens directly to target ranks
        # This is THE KEY - tokens go directly where needed!
        recv_buffer = torch.empty(
            (recv_counts.sum(), hidden_states.size(-1)),
            dtype=hidden_states.dtype,
            device=hidden_states.device
        )
        
        torch.distributed.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=recv_counts.tolist(),
            input_split_sizes=send_counts.tolist(),
            group=self.ep_group
        )
        # ^ Each rank now has exactly the tokens it needs!
        #   No masking, no wasted memory
        
        # Step 5: Process expert on received tokens
        return recv_buffer


# From fused_a2a.py (DeepEP implementation)
class FusedAlltoAllDispatcher:
    """
    Fused all-to-all with custom CUDA kernels.
    Maximum efficiency for large EP sizes.
    """
    
    def __init__(self, ...):
        # Pre-allocate buffers to avoid dynamic allocation
        self.send_buffer = torch.empty(
            (self.max_tokens, self.hidden_size),
            dtype=torch.float16,
            device='cuda'
        )
        self.recv_buffer = torch.empty(
            (self.max_tokens, self.hidden_size),
            dtype=torch.float16,
            device='cuda'
        )
        
        # Custom CUDA kernels for routing
        self.routing_kernel = _load_fused_routing_kernel()
    
    def token_permutation(self, hidden_states, routing_weights):
        """
        Fused routing + all-to-all with custom CUDA kernels.
        """
        # Use custom CUDA kernel for routing decision
        # Faster than PyTorch operations
        send_counts, recv_counts = self.routing_kernel(
            routing_weights, self.send_buffer, hidden_states
        )
        
        # All-to-all with pre-allocated buffers
        torch.distributed.all_to_all_single(
            self.recv_buffer,
            self.send_buffer,
            output_split_sizes=recv_counts,
            input_split_sizes=send_counts,
            group=self.ep_group
        )
        
        return self.recv_buffer[:recv_counts.sum()]
```

## When to Use

**MoE models only!** Choose dispatcher based on EP size:

```python
from megatron.core.transformer.moe import MoEConfig

# For small EP (≤4)
moe_config = MoEConfig(
    expert_model_parallel_size=4,
    token_dispatcher_type='allgather',  # AllGather dispatcher
)

# For larger EP (>4)
moe_config = MoEConfig(
    expert_model_parallel_size=8,
    token_dispatcher_type='alltoall',  # AlltoAll dispatcher
)

# For maximum performance (requires custom installation)
moe_config = MoEConfig(
    expert_model_parallel_size=8,
    token_dispatcher_type='fused',  # Fused A2A (DeepEP)
)
```

### Skip If

- Not using MoE models
- Dense models (no experts)
- Single expert (EP = 1)

## Performance Impact

### Communication Efficiency

**AllGather Dispatcher (EP=4):**
- Communication: 2 all-gathers per MoE layer
- Memory: O(EP × tokens) temporary buffers
- Best for: EP ≤ 4

**AlltoAll Dispatcher (EP=8):**
- Communication: 2 all-to-alls per MoE layer
- Memory: O(tokens) temporary buffers (much better!)
- Best for: EP > 4

**Fused A2A (EP=8+):**
- Communication: Same as AlltoAll
- Memory: Pre-allocated buffers (no dynamic allocation)
- Additional: Custom CUDA kernels for routing
- Best for: EP ≥ 8, maximum performance

### Throughput Improvement

**vs. Naive All-Gather:**
- AlltoAll dispatcher: **2-3x faster** for EP > 4
- Fused A2A: **3-4x faster** for EP ≥ 8

**End-to-End MoE Training:**
- 10-30% speedup in MoE training vs naive implementation
- Critical for scaling to many experts (64, 128+)

### Example Measurements

**Mixtral 8×7B (8 experts, EP=8):**
- Naive all-gather: 45ms per MoE layer
- AlltoAll dispatcher: 18ms per MoE layer
- Fused A2A: 12ms per MoE layer
- **Result:** 3.75x speedup with Fused A2A

## Troubleshooting

### Load Imbalance

**Symptoms:**
- Some experts get many more tokens than others
- Some ranks idle while others compute

**Causes:**
- Router not balanced
- Auxiliary loss not configured

**Fix priority:**
1. Enable load balancing auxiliary loss
2. Tune router temperature
3. Use capacity factor to limit tokens per expert

### Communication Bottleneck

**Symptoms:**
- MoE layers much slower than dense layers
- High communication time in profiler

**Causes:**
- Wrong dispatcher for EP size
- Inefficient routing
- Network bottleneck

**Fix priority:**
1. Use AlltoAll for EP > 4
2. Enable Fused A2A if available
3. Profile communication patterns

### Memory Issues

**Symptoms:**
- OOM in MoE layers
- Crashes during token routing

**Causes:**
- AllGather with large EP (too much memory)
- Imbalanced token distribution

**Fix priority:**
1. Switch to AlltoAll dispatcher
2. Reduce capacity factor
3. Enable dropless MoE

## Related Optimizations

- **#14 Expert Parallelism:** Strategy for parallelizing experts
- **#20 MoE Batch-Level Overlapping:** Hide EP All-to-All behind dense computation (15-25% speedup)
- **#01 Gradient Bucketing:** Also applies to expert gradients
- **#36 Grouped GEMM:** Optimizes expert computation itself

## Configuration Example

```python
from megatron.core.transformer.moe import MoEConfig

# MoE configuration with optimal dispatcher
moe_config = MoEConfig(
    # Expert parallelism
    expert_model_parallel_size=8,      # EP size
    num_experts=64,                    # Total experts
    
    # Token routing
    moe_router_topk=2,                 # Top-k routing (top 2 experts)
    moe_router_load_balancing_type='sinkhorn',  # Load balancing
    moe_aux_loss_coeff=1e-2,           # Auxiliary loss weight
    
    # Token dispatcher (CHOOSE ONE)
    token_dispatcher_type='alltoall',  # For EP > 4
    # token_dispatcher_type='fused',   # For maximum performance
    
    # Capacity and dropless
    moe_token_capacity_factor=1.25,    # 1.25x average tokens per expert
    moe_pad_expert_input_to_capacity=False,
    moe_expert_capacity_factor=None,   # Dynamic capacity
)
```

## Load Balancing

Critical for MoE performance:

```python
# Enable load balancing auxiliary loss
moe_config = MoEConfig(
    # Sinkhorn routing for better balance
    moe_router_load_balancing_type='sinkhorn',
    moe_aux_loss_coeff=1e-2,  # Weight for load balancing loss
    
    # Or use auxiliary loss (simpler)
    # moe_router_load_balancing_type='aux_loss',
    # moe_aux_loss_coeff=1e-2,
)

# Monitor expert utilization
# Should be roughly equal across experts:
# Expert 0: 12.5% tokens
# Expert 1: 12.3% tokens
# ...
# Expert 7: 12.8% tokens
```

## References

- Switch Transformers: [Switch Transformers Paper](https://arxiv.org/abs/2101.03961)
- DeepSpeed MoE: [DeepSpeed-MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/)
- Implementation: `megatron/core/transformer/moe/`
