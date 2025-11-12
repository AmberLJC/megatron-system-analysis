# 14. Expert Parallelism (MoE)

## Context

MoE models have many experts (8, 64, 128+). Not all experts needed for all tokens. Parallelize across expert dimension rather than replicating all experts.

## Implementation

Each EP rank holds subset of experts. Tokens routed to appropriate rank via all-to-all communication.

**Three dispatcher modes:**
- AllGather: For EP ≤ 4 (see #09)
- AlltoAll: For EP > 4 (see #09)
- Fused A2A: Maximum efficiency (see #09)

## Core Code

- `megatron/core/transformer/moe/router.py` - Token routing
- `megatron/core/transformer/moe/token_dispatcher.py` - Dispatchers (see #09)
- `megatron/core/transformer/moe/moe_layer.py` - MoE layer implementation

## Code Snippet

```python
class MoELayer(torch.nn.Module):
    def __init__(self, config):
        self.num_local_experts = config.num_experts // config.expert_model_parallel_size
        
        # Create local experts (only subset on this rank)
        self.experts = torch.nn.ModuleList([
            ExpertModule(config) for _ in range(self.num_local_experts)
        ])
        
        # Router decides which tokens go to which experts
        self.router = Router(config)
        
        # Dispatcher handles communication
        self.token_dispatcher = create_token_dispatcher(config)
    
    def forward(self, hidden_states):
        # Route tokens to experts
        router_output = self.router(hidden_states)
        
        # Dispatch tokens to appropriate ranks
        dispatched_input = self.token_dispatcher.token_permutation(
            hidden_states, router_output
        )
        
        # Process with local experts
        expert_output = self.expert_computation(dispatched_input)
        
        # Return tokens to original positions
        output = self.token_dispatcher.token_unpermutation(expert_output)
        
        return output
```

## When to Use

**MoE models only!**

```python
# MoE configuration
num_experts = 64
expert_model_parallel_size = 8  # Each rank holds 64/8 = 8 experts
moe_router_topk = 2             # Each token goes to top-2 experts

# Total GPUs: TP × PP × EP × DP
# Example: 4 × 4 × 8 × 8 = 1024 GPUs
```

## Performance Impact

**vs. Dense models:**
- 2-5x speedup for similar quality
- More parameters without proportional compute increase

**Communication:** All-to-all per MoE layer (see #09)

## Configuration Example

```python
from megatron.core.transformer.moe import MoEConfig

moe_config = MoEConfig(
    num_experts=64,
    expert_model_parallel_size=8,
    moe_router_topk=2,
    token_dispatcher_type='alltoall',  # Or 'fused'
    moe_router_load_balancing_type='sinkhorn',
    moe_aux_loss_coeff=1e-2,
)
```

## References

- Switch Transformers: [Switch Transformers](https://arxiv.org/abs/2101.03961)
- Detailed communication: See optimization #09

