# 14. Expert Parallelism (MoE)

## Overview

Expert Parallelism (EP) is a critical parallelism strategy for Mixture-of-Experts (MoE) models that distributes different experts across multiple GPUs. Unlike traditional model parallelism that replicates all experts on every device, EP partitions experts across the expert parallel dimension, enabling efficient scaling of models with large numbers of experts (8, 64, 128+). Each device hosts only a subset of experts, and tokens are dynamically routed to the appropriate devices during training and inference.

## Core Architecture

Megatron-LM implements a sophisticated three-layer architecture for expert parallelism:

### 1. MoE Layer Implementation

The `MoELayer` class (megatron/core/transformer/moe/moe_layer.py:92-295) orchestrates the entire expert parallel workflow:

```python
class MoELayer(BaseMoELayer):
    def __init__(self, config, submodules=None, layer_number=None, pg_collection=None):
        super(MoELayer, self).__init__(config, layer_number, pg_collection)

        # Partition experts across EP ranks
        ep_size = utils.get_pg_size(self.ep_group)
        ep_rank = utils.get_pg_rank(self.ep_group)
        self.num_local_experts = self.config.num_moe_experts // ep_size
        local_expert_indices_offset = ep_rank * self.num_local_experts

        # Each rank holds subset of experts
        self.local_expert_indices = [
            local_expert_indices_offset + i for i in range(self.num_local_experts)
        ]

        # Initialize router for token assignment
        self.router = TopKRouter(config=self.config, pg_collection=pg_collection)

        # Initialize token dispatcher (AllGather/AlltoAll/Flex)
        if config.moe_token_dispatcher_type == "allgather":
            self.token_dispatcher = MoEAllGatherTokenDispatcher(...)
        elif config.moe_token_dispatcher_type == "alltoall":
            self.token_dispatcher = MoEAlltoAllTokenDispatcher(...)
        elif config.moe_token_dispatcher_type == "flex":
            self.token_dispatcher = MoEFlexTokenDispatcher(...)

        # Build local experts only
        self.experts = build_module(
            self.submodules.experts,
            self.num_local_experts,
            self.config,
            pg_collection=pg_collection,
        )
```

The forward pass implements a four-stage pipeline:

```python
def forward(self, hidden_states):
    # Stage 1: Route tokens to experts and preprocess
    hidden_states, probs, residual = self.router_and_preprocess(hidden_states)

    # Stage 2: Dispatch tokens via communication (AllGather/AlltoAll)
    dispatched_input, probs = self.dispatch(hidden_states, probs)

    # Stage 3: Compute with local experts
    output, mlp_bias = self.routed_experts_compute(dispatched_input, probs, residual)

    # Stage 4: Combine expert outputs and return
    output = self.combine(output, shared_expert_output)

    return output, mlp_bias
```

### 2. Token Routing with TopKRouter

The `TopKRouter` (megatron/core/transformer/moe/router.py:129-573) determines which experts process which tokens:

```python
class TopKRouter(Router):
    def __init__(self, config, pg_collection=None):
        super().__init__(config=config, pg_collection=pg_collection)
        self.topk = self.config.moe_router_topk
        self.routing_type = self.config.moe_router_load_balancing_type

        # Initialize gating network weights [num_experts, hidden_size]
        self.weight = torch.nn.Parameter(
            torch.empty((self.config.num_moe_experts, self.config.hidden_size),
                       dtype=torch.float32)
        )

    def gating(self, input):
        # Compute routing logits
        logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
        return logits

    def routing(self, logits):
        # Apply Z-Loss for stability
        logits = self.apply_z_loss(logits)

        # Top-k selection with score function
        if self.routing_type == "sinkhorn":
            probs, routing_map = self.sinkhorn_load_balancing(logits)
        else:
            probs, routing_map = topk_routing_with_score_function(
                logits, self.topk,
                use_pre_softmax=self.config.moe_router_pre_softmax,
                score_function=self.score_function,
                expert_bias=self.expert_bias
            )

        # Apply token dropping if capacity is set
        if self.config.moe_expert_capacity_factor is not None:
            probs, routing_map = apply_router_token_dropping(
                probs, routing_map,
                router_topk=self.topk,
                capacity_factor=self.config.moe_expert_capacity_factor
            )

        return probs, routing_map
```

**Load Balancing Loss**: The framework implements the Switch Transformer auxiliary loss to encourage balanced expert utilization:

```python
def switch_load_balancing_loss_func(probs, tokens_per_expert, total_num_tokens,
                                     topk, num_experts, moe_aux_loss_coeff):
    """
    Auxiliary loss formula:
        loss = E * Σ_{i=1}^{E} (f_i * P_i)
    where:
        f_i = fraction of tokens dispatched to expert i
        P_i = averaged router probability for expert i
        E = number of experts
    """
    aggregated_probs_per_expert = probs.sum(dim=0)
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss
```

### 3. Token Dispatcher Communication Strategies

Megatron-LM supports three token dispatcher implementations with different communication patterns:

#### **AllGather Dispatcher** (for EP ≤ 4)
Uses AllGather to replicate tokens across all ranks, suitable for small expert parallel sizes:

```python
class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    def token_dispatch(self, hidden_states, probs):
        # AllGather tokens across TP×EP group
        if self.tp_size > 1 or self.ep_size > 1:
            # [num_local_tokens, H] -> [num_global_tokens, H]
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group, use_global_buffer=True
            )
            probs = gather_from_sequence_parallel_region(probs, group=self.tp_ep_group)
        return hidden_states, probs

    def token_combine(self, hidden_states):
        # ReduceScatter to return tokens to original ranks
        if self.tp_size > 1 or self.ep_size > 1:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )
        return hidden_states
```

#### **AlltoAll Dispatcher** (for EP > 4)
Uses AlltoAll for efficient point-to-point communication with variable token counts:

```python
class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    def preprocess(self, routing_map):
        # Calculate how many tokens each rank sends/receives
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        # [ep_size]: tokens sent to each EP rank
        self.input_splits = num_local_tokens_per_expert.reshape(
            self.ep_size, self.num_local_experts
        ).sum(axis=1)

        # Gather global token distribution
        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, group=self.tp_ep_group
        ).reshape(self.ep_size, self.tp_size, self.num_experts).transpose(0, 1)

        # [ep_size]: tokens received from each EP rank
        self.output_splits = num_global_tokens_per_rank[self.tp_rank]

        return num_tokens_per_local_expert

    def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
        # AlltoAll communication with variable split sizes
        global_input_tokens = all_to_all(
            self.ep_group, permutated_local_input_tokens,
            self.output_splits, self.input_splits
        )
        global_probs = all_to_all(
            self.ep_group, permuted_probs,
            self.output_splits, self.input_splits
        )
        return global_input_tokens, global_probs
```

#### **Flex Dispatcher** (with DeepEP integration)
Leverages fused kernels that combine permutation and communication for maximum efficiency:

```python
class MoEFlexTokenDispatcher(MoETokenDispatcher):
    def token_dispatch(self, hidden_states, probs=None,
                       async_finish=True, allocate_on_comm_stream=True):
        """
        Fused permutation + AlltoAll using DeepEP kernels.
        This reduces memory bandwidth and enables better overlap.
        """
        return (
            self._comm_manager.dispatch(hidden_states, async_finish,
                                       allocate_on_comm_stream),
            self._comm_manager.dispatched_probs,
        )

    def token_combine(self, hidden_states, async_finish=True,
                      allocate_on_comm_stream=True):
        """Fused un-permutation + AlltoAll for combining expert outputs"""
        return self._comm_manager.combine(hidden_states, async_finish,
                                         allocate_on_comm_stream)
```

## Expert Implementations

Megatron-LM provides three expert implementations with different performance characteristics:

### **GroupedMLP** (with GroupedGEMM)
Processes multiple experts in parallel using batched GEMM operations:

```python
class GroupedMLP(MegatronModule):
    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        # Reshape weights for grouped execution [num_local_experts, hidden, ffn_size]
        w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
        w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

        # Grouped matrix multiplication across all experts simultaneously
        fc1_output = gg.ops.gmm(permuted_local_hidden_states, w1,
                                tokens_per_expert, trans_b=False)

        # Apply activation and routing probabilities
        intermediate_parallel = self.activation_func_with_probs(
            fc1_output, permuted_probs.unsqueeze(-1)
        )

        # Second grouped GEMM
        fc2_output = gg.ops.gmm(intermediate_parallel, w2,
                                tokens_per_expert, trans_b=False)

        return fc2_output, None
```

### **TEGroupedMLP** (with TransformerEngine)
Leverages TransformerEngine's GroupedLinear for FP8 and optimized performance:

```python
class TEGroupedMLP(MegatronModule):
    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        tokens_per_expert = tokens_per_expert.tolist()

        # FP8 padding for alignment
        if self.config.fp8:
            permuted_local_hidden_states, tokens_per_expert = self.fp8_padding(
                permuted_local_hidden_states, tokens_per_expert
            )

        # TE GroupedLinear forward with per-expert token counts
        intermediate_parallel, bias_parallel = self.linear_fc1(
            permuted_local_hidden_states, tokens_per_expert
        )

        # Fused bias-activation-weighting
        if self.config.bias_activation_fusion:
            if self.activation_func == F.silu and self.config.gated_linear_unit:
                intermediate_parallel = weighted_bias_swiglu_impl(
                    intermediate_parallel, bias_parallel, permuted_probs
                )

        output, output_bias = self.linear_fc2(intermediate_parallel, tokens_per_expert)

        # Unpad if FP8 was used
        if self.config.fp8:
            output = self.fp8_unpadding(output, actual_tokens_per_expert)

        return output, output_bias
```

### **SequentialMLP**
Processes experts sequentially, used for compatibility or when GroupedGEMM is unavailable:

```python
class SequentialMLP(MegatronModule):
    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        tokens_per_expert = tokens_per_expert.tolist()
        tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert)
        probs_list = torch.split(permuted_probs, tokens_per_expert)

        output_local_list = []
        # Execute each expert sequentially
        for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
            output, output_bias = expert(tokens, probs)
            output_local_list.append(output)

        output_local = torch.cat(output_local_list, dim=0)
        return output_local, None
```

## Advanced Features

### **1. Shared Experts**
Shared experts process all tokens in addition to routed experts, useful for capturing common patterns:

```python
class SharedExpertMLP(MLP):
    def __init__(self, config, submodules, gate, pg_collection=None):
        config.ffn_hidden_size = config.moe_shared_expert_intermediate_size
        super().__init__(config=config, submodules=submodules)

        # Optional gating mechanism
        self.use_shared_expert_gate = gate
        if self.use_shared_expert_gate:
            self.gate_weight = torch.nn.Parameter(
                torch.empty((1, self.config.hidden_size))
            )

    def forward(self, hidden_states):
        output, _ = super().forward(hidden_states)
        if self.use_shared_expert_gate:
            logits = torch.nn.functional.linear(hidden_states, self.gate_weight)
            gate_score = torch.nn.functional.sigmoid(logits)
            output = output * gate_score
        return output
```

Shared experts can be overlapped with routed expert communication when `--moe-shared-expert-overlap` is enabled, processing in parallel streams.

### **2. Expert Tensor Parallelism (ETP)**
Experts can have a different tensor parallel size than attention layers:

```python
# Configuration example
self.ep_group = pg_collection.ep           # Expert parallel group
self.expt_tp_group = pg_collection.expt_tp # Expert tensor parallel group
self.attn_tp_group = pg_collection.tp      # Attention tensor parallel group

# Experts can use expt_tp_group with size different from attn_tp_group
expert = MLP(config, submodules, tp_group=pg_collection.expt_tp)
```

This enables optimal resource allocation where experts may benefit from different TP configurations than attention layers.

### **3. Load Balancing Strategies**

The framework supports multiple load balancing approaches:

- **Auxiliary Loss**: Standard Switch Transformer loss encouraging balanced token distribution
- **Sequence-level Auxiliary Loss**: Balances experts within each sequence independently
- **Global Auxiliary Loss**: Tracks global statistics across training for long-term balance
- **Sinkhorn Routing**: Uses Sinkhorn-Knopp algorithm for balanced assignment without auxiliary loss

```python
# Sinkhorn load balancing
def sinkhorn_load_balancing(self, logits):
    if self.training:
        with torch.no_grad():
            # Sinkhorn algorithm for balanced routing
            norm_logits = sinkhorn(logits.to(dtype=torch.float32))
            _, indices = torch.topk(norm_logits, k=self.topk, dim=1)
        logits = torch.sigmoid(logits) if self.topk == 1 else torch.softmax(logits, dim=-1)
    else:
        logits = torch.sigmoid(logits) if self.topk == 1 else torch.softmax(logits, dim=-1)
        _, indices = torch.topk(logits, k=self.topk, dim=1)

    map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
    scores = logits * map
    return scores, map
```

### **4. Token Capacity Management**

Expert capacity limits tokens per expert to prevent load imbalance and enable static memory allocation:

```python
def get_capacity(num_tokens, num_experts, capacity_factor, min_capacity=None):
    """
    capacity = ceil((num_tokens / num_experts) * capacity_factor)

    capacity_factor > 1.0: Buffer room to handle imbalance
    capacity_factor = 1.0: Perfect balance required (may drop tokens)
    """
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity
```

## Configuration and Usage

### Basic MoE Configuration
```python
# Training script arguments
--num-experts 64                           # Total experts in model
--expert-model-parallel-size 8             # EP size (64/8 = 8 experts per GPU)
--moe-router-topk 2                        # Route each token to top-2 experts
--moe-token-dispatcher-type alltoall       # AlltoAll for EP > 4
--moe-ffn-hidden-size 14336               # Expert FFN intermediate size
--disable-bias-linear                      # Required for GroupedMLP

# Load balancing
--moe-router-load-balancing-type aux_loss  # Standard auxiliary loss
--moe-aux-loss-coeff 0.01                 # Loss coefficient

# Expert implementation
--moe-grouped-gemm                         # Use GroupedGEMM (fastest)
# or use TEGroupedMLP (for FP8 support)

# Optional features
--moe-expert-capacity-factor 1.25          # 25% buffer capacity
--moe-shared-expert-intermediate-size 2048 # Add shared experts
--expert-tensor-parallel-size 2            # ETP for experts
```

### Multi-dimensional Parallelism
Expert parallelism composes with other parallelism strategies:

```
Total GPUs = TP × PP × EP × DP
Example: 4 (TP) × 4 (PP) × 8 (EP) × 8 (DP) = 1024 GPUs

Per-GPU computation:
- Attention layers: TP=4 across GPUs
- Experts: 64 total experts / 8 EP ranks = 8 local experts per GPU
- Each expert can use ETP=2 internally
```

### Performance Characteristics

**AllGather vs AlltoAll**:
- AllGather: Simple, good for EP ≤ 4, uniform communication volume
- AlltoAll: Efficient for EP > 4, handles variable token counts, scales to large EP

**GroupedMLP vs TEGroupedMLP vs SequentialMLP**:
- GroupedMLP: ~3-5x faster than Sequential, uses CUTLASS kernels
- TEGroupedMLP: Best for FP8 training, supports advanced fusions
- SequentialMLP: Fallback, compatible with all configurations

## References

- Switch Transformers: https://arxiv.org/abs/2101.03961
- ST-MoE (Z-Loss): https://arxiv.org/pdf/2202.08906.pdf
- Global Load Balancing: https://arxiv.org/abs/2501.11873
- DeepEP (Fused Dispatcher): https://github.com/deepseek-ai/deepep
- Grouped GEMM: https://github.com/fanshiqing/grouped_gemm
