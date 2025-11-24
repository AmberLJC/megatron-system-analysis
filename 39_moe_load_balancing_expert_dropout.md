# MoE Load Balancing & Expert Dropout

## High-Level Overview

Mixture of Experts (MoE) load balancing is critical for training efficient sparse models. In Megatron-LM, load balancing prevents the common problem of token imbalance—where some experts receive many tokens while others receive few, causing compute waste and throughput degradation. Expert dropout (via token dropping) complements this by enforcing capacity limits on experts to prevent memory overflows.

The system employs **three complementary load balancing strategies**:
1. **Auxiliary Loss**: Added to training loss to penalize imbalanced token distribution
2. **Token Dropping**: Limits expert capacity, dropping or padding tokens that exceed capacity
3. **Expert Bias**: Dynamic bias adjustment for auxiliary-loss-free load balancing

## Load Balancing Mechanism

### Auxiliary Loss Function (Switch Transformer Style)

The primary load balancing technique uses the Switch Transformer auxiliary loss:

```
loss = E × Σ(f_i × P_i)
```

Where:
- `f_i` = fraction of tokens routed to expert i
- `P_i` = averaged router probability for expert i
- `E` = number of experts
- Coefficient scaled by `num_experts / (topk × total_tokens²)`

**Three Auxiliary Loss Types**:
- **aux_loss**: Micro-batch level (GShard style)
- **seq_aux_loss**: Per-sequence level (DeepSeekV2 style)
- **global_aux_loss**: Global batch level with running average over multiple steps

These can be combined by setting `moe_router_load_balancing_type = ["aux_loss", "seq_aux_loss", "global_aux_loss"]`.

**Core Implementation** (`megatron/core/transformer/moe/moe_utils.py:35-112`):

```python
def switch_load_balancing_loss_func(
    probs: torch.Tensor,
    tokens_per_expert: torch.Tensor,
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
    fused: bool = False,
):
    """Calculate auxiliary loss for load balancing.

    Args:
        probs: Router probabilities [num_tokens, num_experts]
        tokens_per_expert: Token counts per expert [num_experts]
        total_num_tokens: Total tokens in batch
        topk: Number of experts per token
        num_experts: Total number of experts
        moe_aux_loss_coeff: Loss coefficient
    """
    if fused:
        return fused_moe_aux_loss(probs, tokens_per_expert, total_num_tokens,
                                  topk, num_experts, moe_aux_loss_coeff)

    # Sum probabilities across tokens for each expert
    aggregated_probs_per_expert = probs.sum(dim=0)

    # Compute auxiliary loss:
    # loss = E × Σ(f_i × P_i) where:
    #   f_i = tokens_per_expert[i] / (topk × total_num_tokens)
    #   P_i = aggregated_probs_per_expert[i]
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss
```

### Z-Loss Regularization

Prevents router logits from growing unbounded:

```python
z_loss = mean(square(logsumexp(logits))) × z_loss_coeff
```

Typically set to `moe_z_loss_coeff = 1e-3`.

**Core Implementation** (`megatron/core/transformer/moe/moe_utils.py:115-127`):

```python
def z_loss_func(logits, z_loss_coeff):
    """Encourages router logits to remain small for training stability.
    From ST-MoE paper: https://arxiv.org/pdf/2202.08906.pdf

    Args:
        logits: Router logits [num_tokens, num_experts]
        z_loss_coeff: Loss coefficient (recommended: 1e-3)

    Returns:
        Scalar loss tensor
    """
    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss
```

## Token Dropping & Expert Capacity

Expert capacity is calculated as:

```python
capacity = ceil((num_tokens / num_experts) × capacity_factor)
```

When `moe_expert_capacity_factor` is set (e.g., 1.0), tokens exceeding capacity are:
- **Dropped** by policy (lowest probability or position-based)
- **Padded** if `moe_pad_expert_input_to_capacity=True`

Default (no capacity factor) passes all tokens without dropping.

**Core Implementation** (`megatron/core/transformer/moe/moe_utils.py:147-163`):

```python
def get_capacity(num_tokens: int, num_experts: int,
                 capacity_factor: float, min_capacity=None):
    """Calculate expert capacity for token dropping.

    Args:
        num_tokens: Number of tokens in batch
        num_experts: Number of experts
        capacity_factor: Multiplier for base capacity
                        (e.g., 1.0 = GShard, 1.25 = Switch Transformer)
        min_capacity: Minimum capacity to enforce

    Returns:
        Expert capacity (max tokens per expert)
    """
    capacity = math.ceil((num_tokens / num_experts) * capacity_factor)
    if min_capacity is not None and capacity < min_capacity:
        capacity = min_capacity
    return capacity
```

**Token Dropping Mechanism** (`megatron/core/transformer/moe/moe_utils.py:656-712`):

```python
def apply_router_token_dropping(
    routing_probs: torch.Tensor,
    routing_map: torch.Tensor,
    router_topk: int,
    capacity_factor: float,
    drop_policy: str = "probs",
    pad_to_capacity: bool = False,
):
    """Apply capacity constraints via token dropping.

    Args:
        routing_probs: [num_tokens, num_experts] - routing probabilities
        routing_map: [num_tokens, num_experts] - boolean routing mask
        router_topk: Number of experts per token
        capacity_factor: Expert capacity factor (None = no dropping)
        drop_policy: "probs" = drop lowest prob tokens, "position" = position-based
        pad_to_capacity: Pad dropped positions with zeros

    Returns:
        final_probs: Routing probabilities after capacity constraints
        final_map: Boolean mask after capacity constraints
    """
    num_tokens, num_experts = routing_probs.shape

    # Calculate expert capacity: ceil((num_tokens * topk) / num_experts * capacity_factor)
    expert_capacity = get_capacity(
        num_tokens=num_tokens * router_topk,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    # Create capacity mask - keep top-k tokens by policy
    if drop_policy == "probs":
        # Drop tokens with lowest routing probabilities
        _, capacity_indices = torch.topk(routing_probs, k=expert_capacity,
                                        dim=0, sorted=False)
        capacity_mask = torch.zeros_like(routing_probs).scatter(
            0, capacity_indices, 1).bool()
    elif drop_policy == "position":
        # Drop tokens at end of sequence
        _, capacity_indices = torch.topk(routing_map.int(),
                                        k=expert_capacity, dim=0, sorted=False)
        capacity_mask = torch.zeros_like(routing_probs).scatter(
            0, capacity_indices, 1).bool()

    # Apply mask to enforce capacity
    if pad_to_capacity:
        # Pad dropped tokens (mask becomes 0 at dropped positions)
        final_map = capacity_mask
        final_probs = routing_probs * final_map
    else:
        # Drop tokens that exceed capacity
        final_map = torch.logical_and(routing_map, capacity_mask)
        final_probs = routing_probs * final_map

    return final_probs, final_map
```

## Expert Bias & Dynamic Load Balancing

**DeepSeekV3-style auxiliary-loss-free approach**:

Increases bias for underutilized experts, decreases for overutilized ones, enabling load balancing without auxiliary loss overhead.

**Core Implementation** (`megatron/core/transformer/moe/moe_utils.py:854-872`):

```python
def get_updated_expert_bias(tokens_per_expert, expert_bias,
                            expert_bias_update_rate):
    """Update expert bias for load balancing (DeepSeekV3 style).
    Ref: https://arxiv.org/abs/2408.15664v1

    Args:
        tokens_per_expert: [num_experts] - token counts per expert
        expert_bias: [num_experts] - current expert bias
        expert_bias_update_rate: Update step size (recommended: 1e-3)

    Returns:
        updated_expert_bias: [num_experts] - updated bias for next iteration
    """
    with torch.no_grad():
        # All-Reduce across TP×CP×DP group for distributed training
        torch.distributed.all_reduce(
            tokens_per_expert,
            group=parallel_state.get_tensor_and_data_parallel_group(
                with_context_parallel=True),
        )

        # Calculate ideal average tokens per expert
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]

        # Compute deviation: positive means underutilized, negative means overutilized
        offset = average_tokens - tokens_per_expert

        # Update bias: increase for underutilized, decrease for overutilized
        # sign(positive_offset) = +1, sign(negative_offset) = -1
        updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate

        return updated_expert_bias
```

**Expert Bias Initialization in Router** (`megatron/core/transformer/moe/router.py:160-181`):

```python
if self.enable_expert_bias:
    # Register buffer for tracking local token counts
    self.register_buffer(
        'local_tokens_per_expert',
        torch.zeros(self.config.num_moe_experts,
                   dtype=torch.float32,
                   device=torch.cuda.current_device()),
        persistent=False,
    )
    # Register learnable expert bias
    self.register_buffer(
        'expert_bias',
        torch.zeros(self.config.num_moe_experts,
                   dtype=torch.float32,
                   device=torch.cuda.current_device()),
    )
else:
    self.local_tokens_per_expert = None
    self.expert_bias = None
```

**Bias Update in Routing** (`megatron/core/transformer/moe/router.py:529-533`):

```python
# Update expert bias and tokens_per_expert
# Prevent extra local tokens accumulation on evaluation or activation recomputation
if self.enable_expert_bias and torch.is_grad_enabled():
    with torch.no_grad():
        self.local_tokens_per_expert += routing_map.sum(dim=0)
```

## Top-K Routing with Score Functions

Supports **group-limited routing** for communication efficiency by restricting token-to-expert assignments to subsets of devices.

**Core Implementation** (`megatron/core/transformer/moe/moe_utils.py:522-620`):

```python
def topk_routing_with_score_function(
    logits: torch.Tensor,
    topk: int,
    use_pre_softmax: bool = False,
    num_groups: Optional[int] = None,
    group_topk: Optional[int] = None,
    scaling_factor: Optional[float] = None,
    score_function: str = "softmax",
    expert_bias: Optional[torch.Tensor] = None,
    fused: bool = False,
):
    """Compute routing probabilities and assignment map for top-k selection.

    Args:
        logits: [num_tokens, num_experts] - router output logits
        topk: Number of experts to select per token
        use_pre_softmax: Apply softmax before topk (alternative: apply after)
        num_groups: Number of expert groups (for group-limited routing)
        group_topk: Number of groups to select per token
        scaling_factor: Optional scaling on routing probabilities
        score_function: "softmax" or "sigmoid"
        expert_bias: [num_experts] - bias to add to logits (for dynamic load balancing)
        fused: Use fused kernel from Transformer Engine

    Returns:
        routing_probs: [num_tokens, num_experts] - routing probabilities
        routing_map: [num_tokens, num_experts] - boolean routing mask
    """
    if fused:
        return fused_topk_with_score_function(
            logits, topk, use_pre_softmax, num_groups, group_topk,
            scaling_factor, score_function, expert_bias)

    num_tokens, num_experts = logits.shape

    def compute_topk(scores, topk, num_groups=None, group_topk=None):
        if group_topk:  # Group-limited routing (DeepSeekV2/V3 style)
            return group_limited_topk(scores, topk, num_tokens,
                                     num_experts, num_groups, group_topk)
        else:
            return torch.topk(scores, k=topk, dim=1)

    # Compute scores based on score function
    if score_function == "softmax":
        if use_pre_softmax:
            # Apply softmax first, then topk
            scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
            probs, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        else:
            # Apply topk first, then softmax (saves computation)
            scores, top_indices = compute_topk(logits, topk, num_groups, group_topk)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).type_as(logits)

    elif score_function == "sigmoid":
        scores = torch.sigmoid(logits.float()).type_as(logits)
        if expert_bias is not None:
            # Add expert bias to logits for dynamic load balancing
            scores_for_routing = scores + expert_bias
            _, top_indices = compute_topk(scores_for_routing, topk, num_groups, group_topk)
            scores = torch.gather(scores, dim=1, index=top_indices).type_as(logits)
        else:
            scores, top_indices = compute_topk(scores, topk, num_groups, group_topk)
        # Normalize sigmoid scores by their sum
        probs = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if topk > 1 else scores

    if scaling_factor:
        probs = probs * scaling_factor

    # Build sparse routing map [num_tokens, num_experts]
    if torch.are_deterministic_algorithms_enabled():
        # Deterministic version using index_put
        routing_probs = torch.zeros_like(logits)
        rows = torch.arange(num_tokens, device=logits.device).unsqueeze(1)
        routing_probs.index_put_((rows, top_indices), probs, accumulate=False)

        routing_map = torch.zeros_like(logits, dtype=logits.dtype)
        routing_map.index_put_((rows, top_indices),
                              torch.ones_like(probs, dtype=routing_map.dtype),
                              accumulate=False)
        routing_map = routing_map.bool()
    else:
        # Fast scatter version
        routing_probs = torch.zeros_like(logits).scatter(1, top_indices, probs)
        routing_map = torch.zeros_like(logits).int().scatter(1, top_indices, 1).bool()

    return routing_probs, routing_map
```

**Router Gating Network** (`megatron/core/transformer/moe/router.py:77-99`):

```python
def gating(self, input: torch.Tensor):
    """Forward pass of router gate linear layer.

    Args:
        input: [num_tokens, hidden_size] - input embeddings

    Returns:
        logits: [num_tokens, num_experts] - unnormalized routing scores
    """
    if self.weight.device.type == 'cpu':
        self.weight.data = self.weight.data.to(device=torch.cuda.current_device())
    if self.bias is not None and self.bias.device.type == 'cpu':
        self.bias.data = self.bias.data.to(device=torch.cuda.current_device())

    # Use specified precision for routing computation
    router_dtype = input.dtype
    if self.config.moe_router_dtype == 'fp32':
        router_dtype = torch.float32
    elif self.config.moe_router_dtype == 'fp64':
        router_dtype = torch.float64

    logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
    return logits
```

## Auxiliary Loss Auto-Scaler

Ensures gradient scaling is appropriate for auxiliary losses relative to main loss.

**Core Implementation** (`megatron/core/transformer/moe/moe_utils.py:166-216`):

```python
class MoEAuxLossAutoScaler(torch.autograd.Function):
    """Custom autograd function for scaling auxiliary loss gradients.

    This prevents auxiliary loss from dominating training by scaling gradients
    relative to main loss scale.
    """
    main_loss_backward_scale: torch.Tensor = None

    @staticmethod
    def forward(ctx, output: torch.Tensor, aux_loss: torch.Tensor):
        """Forward pass - pass through output, save aux loss for backward.

        Args:
            output: Model output tensor
            aux_loss: Auxiliary loss tensor (not used in forward)

        Returns:
            output: Unchanged output tensor
        """
        ctx.save_for_backward(aux_loss)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """Backward pass - scale auxiliary loss gradient appropriately.

        Args:
            grad_output: Gradient of output from loss.backward()

        Returns:
            (grad_output, scaled_aux_loss_grad) where scaled_aux_loss_grad
            matches the backward scale of main loss
        """
        (aux_loss,) = ctx.saved_tensors
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = torch.tensor(
                1.0, device=aux_loss.device)

        # Scale auxiliary loss gradient by same factor as main loss
        aux_loss_backward_scale = MoEAuxLossAutoScaler.main_loss_backward_scale
        scaled_aux_loss_grad = torch.ones_like(aux_loss) * aux_loss_backward_scale
        return grad_output, scaled_aux_loss_grad

    @staticmethod
    def set_loss_scale(scale: torch.Tensor):
        """Set the gradient scale for auxiliary loss.

        Called with same scale as main loss to ensure consistent gradient scaling.
        """
        if MoEAuxLossAutoScaler.main_loss_backward_scale is None:
            MoEAuxLossAutoScaler.main_loss_backward_scale = scale
        else:
            MoEAuxLossAutoScaler.main_loss_backward_scale.copy_(scale)
```

**Usage in Router** (`megatron/core/transformer/moe/router.py:402-412`):

```python
if self.calculate_per_token_loss:
    # Scale aux loss by number of tokens in micro-batch
    # to match per-token loss scaling in finalize_model_grads
    activation = MoEAuxLossAutoScaler.apply(
        activation, aux_loss * activation.shape[0])
else:
    activation = MoEAuxLossAutoScaler.apply(activation, aux_loss)
return activation
```

## Permute & Unpermute Operations

Token permutation groups tokens by assigned expert for efficient batched processing.

**Permute** (`megatron/core/transformer/moe/moe_utils.py:219-303`):

```python
def permute(
    tokens,
    routing_map,
    probs: Optional[torch.Tensor] = None,
    num_out_tokens: Optional[int] = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """Permute tokens: group by expert for efficient expert computation.

    Converts sparse [num_tokens, num_experts] routing to dense
    [num_expert_tokens, hidden] by grouping tokens assigned to same expert.

    Args:
        tokens: [num_tokens, hidden] - input tokens
        routing_map: [num_tokens, num_experts] - sparse routing assignment
        probs: [num_tokens, num_experts] - routing probabilities (optional)
        num_out_tokens: Output token count (if not provided, equals num_tokens)
        fused: Use fused kernel from Transformer Engine
        drop_and_pad: Tokens already padded to capacity

    Returns:
        permuted_input: [num_out_tokens, hidden] - permuted tokens
        permuted_probs: [num_out_tokens] or None - permuted probabilities
        sorted_indices: Indices used for permutation
    """
    if fused and probs is None:
        return fused_permute(tokens, routing_map, num_out_tokens=num_out_tokens)

    if fused and probs is not None:
        return fused_permute_with_probs(tokens, probs, routing_map,
                                       num_out_tokens=num_out_tokens)

    num_tokens, hidden = tokens.shape
    num_experts = routing_map.shape[1]
    permuted_probs = None

    # Transpose routing_map from [num_tokens, num_experts] to [num_experts, num_tokens]
    routing_map = routing_map.bool().T.contiguous()

    # Create expert-to-token mapping by collecting token indices
    token_indices = torch.arange(num_tokens, device=routing_map.device).unsqueeze(0).expand(num_experts, -1)
    sorted_indices = token_indices.masked_select(routing_map)

    # Gather probabilities if provided
    if probs is not None:
        permuted_probs = probs.T.contiguous().masked_select(routing_map)

    # Use indices to permute tokens
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices
```

**Unpermute** (`megatron/core/transformer/moe/moe_utils.py:306-393`):

```python
def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    restore_shape: torch.Size,
    probs: torch.Tensor = None,
    routing_map: torch.Tensor = None,
    fused: bool = False,
    drop_and_pad: bool = False,
):
    """Restore original token order after expert computation.

    Inverse operation of permute: converts expert-grouped tokens back to
    original sequence order, optionally weighted by routing probabilities.

    Args:
        permuted_tokens: [num_permuted_tokens, hidden] - tokens after expert computation
        sorted_indices: Indices from permute operation
        restore_shape: Original shape to restore to [num_tokens, hidden]
        probs: [num_tokens, num_experts] - routing probabilities (optional)
        routing_map: [num_tokens, num_experts] - routing mask (optional)
        fused: Use fused kernel from Transformer Engine
        drop_and_pad: Tokens were padded to capacity

    Returns:
        output_tokens: [num_tokens, hidden] - unpermuted tokens, original order
    """
    if fused:
        return fused_unpermute(permuted_tokens, sorted_indices,
                              merging_probs=probs, restore_shape=restore_shape)

    _, hidden = restore_shape
    input_dtype = permuted_tokens.dtype

    # Apply routing probabilities if provided (weighted combination of expert outputs)
    if probs is not None:
        assert routing_map is not None, "Routing map required for probability weighting"
        # Extract permuted probabilities from routing_map
        permuted_probs = probs.T.contiguous().masked_select(routing_map.T.contiguous())
        # Weight expert outputs by their routing probability
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)

    # Create output tensor and scatter tokens back to original positions
    output_tokens = torch.zeros(
        restore_shape, dtype=permuted_tokens.dtype, device=permuted_tokens.device)

    if torch.are_deterministic_algorithms_enabled():
        # Use deterministic index_add for CUDA graph compatibility
        output_tokens.index_add_(0, sorted_indices, permuted_tokens)
    else:
        # Use scatter_add for speed
        output_tokens.scatter_add_(
            0, sorted_indices.unsqueeze(1).expand(-1, hidden), permuted_tokens)

    return output_tokens.to(dtype=input_dtype)
```

## Core Implementation Classes

**Router** (`megatron/core/transformer/moe/router.py:129-562`):

Main routing orchestration class. Complete forward workflow:

```python
class TopKRouter(Router):
    def forward(self, input: torch.Tensor):
        """Main router forward pass.

        Args:
            input: [num_tokens, hidden_size] - input embeddings

        Returns:
            probs: Routing probabilities
            routing_map: Routing assignment mask
        """
        self._maintain_float32_expert_bias()

        # 1. Add input noise (optional regularization)
        input = self.apply_input_jitter(input)

        # 2. Compute router logits via gating network
        logits = self.gating(input)

        # 3. Apply benchmark mode (optional)
        if self.config.moe_router_force_load_balancing:
            logits = apply_random_logits(logits)

        # 4. Main routing: z-loss, topk, token dropping, aux losses
        probs, routing_map = self.routing(logits)

        return probs, routing_map
```

**MoE Layer** (`megatron/core/transformer/moe/moe_layer.py:250-294`):

Orchestrates complete forward pass: shared experts → routing → token dispatch → expert computation → combine

```python
def forward(self, hidden_states: torch.Tensor):
    """MoE layer forward pass with token routing and expert computation.

    Args:
        hidden_states: [num_tokens, hidden_size] - input

    Returns:
        output: [num_tokens, hidden_size] - output with routed expert contributions
        mlp_bias: Optional bias term from experts
    """
    # 1. Optional: compute shared experts (non-routed dense experts)
    shared_expert_output = self.shared_experts_compute(hidden_states)

    # 2. Preprocess and route to sparse experts
    hidden_states, probs, residual = self.router_and_preprocess(hidden_states)

    # 3. Dispatch tokens to experts via all-to-all communication
    dispatched_input, probs = self.dispatch(hidden_states, probs)

    # 4. Compute expert outputs (each expert processes its assigned tokens)
    expert_output, mlp_bias = self.routed_experts_compute(
        dispatched_input, probs, residual)

    # 5. Combine expert outputs via all-to-all or reduce-scatter communication
    output = self.combine(expert_output, shared_expert_output)

    return output, mlp_bias
```

## Auxiliary Loss Computation - Three Types

**Micro-batch Level** (`megatron/core/transformer/moe/router.py:269-295`):

```python
def _apply_aux_loss(
    self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor,
    routing_map: torch.Tensor
):
    """Micro-batch level auxiliary loss (GShard style)."""
    aux_loss_coeff = self.get_aux_loss_coeff("aux_loss")
    if aux_loss_coeff == 0:
        return probs

    # Count tokens assigned to each expert in this micro-batch
    tokens_per_expert = routing_map.sum(dim=0)

    # All-reduce across tensor-parallel and context-parallel groups
    tokens_per_expert = reduce_from_tensor_model_parallel_region(
        tokens_per_expert, self.tp_cp_group)

    # Global batch size
    num_tokens = routing_map.shape[0]
    total_num_tokens = num_tokens * self.tp_cp_group.size()

    # Compute auxiliary loss
    aux_loss = switch_load_balancing_loss_func(
        probs=scores_for_aux_loss,
        tokens_per_expert=tokens_per_expert,
        total_num_tokens=total_num_tokens,
        topk=self.topk,
        num_experts=self.config.num_moe_experts,
        moe_aux_loss_coeff=aux_loss_coeff,
        fused=self.config.moe_router_fusion,
    )

    # Attach loss to computation graph
    probs = self.attach_and_log_load_balancing_loss(
        probs, aux_loss_coeff, aux_loss, "load_balancing_loss",
        self.tp_cp_group)
    return probs
```

**Sequence Level** (`megatron/core/transformer/moe/router.py:297-339`):

```python
def _apply_seq_aux_loss(
    self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor,
    routing_map: torch.Tensor, seq_length: int, bsz: int
):
    """Per-sequence auxiliary loss (DeepSeekV2 style).

    Reshapes batch dimension to sequence dimension for finer-grained
    load balancing control at sequence level.
    """
    seq_aux_loss_coeff = self.get_aux_loss_coeff("seq_aux_loss")
    if seq_aux_loss_coeff == 0:
        return probs

    # Reshape to [seq_length, -1] treating each sequence separately
    scores_for_aux_loss = scores_for_aux_loss.reshape(seq_length, -1)
    tokens_per_expert = routing_map.reshape(seq_length, -1).sum(dim=0)

    # All-reduce for distributed training
    tokens_per_expert = reduce_from_tensor_model_parallel_region(
        tokens_per_expert, self.tp_cp_group)

    total_num_tokens = seq_length * self.tp_cp_group.size()

    # Compute sequence-level auxiliary loss
    aux_loss = (
        switch_load_balancing_loss_func(
            probs=scores_for_aux_loss,
            tokens_per_expert=tokens_per_expert,
            total_num_tokens=total_num_tokens,
            topk=self.topk,
            num_experts=self.config.num_moe_experts,
            moe_aux_loss_coeff=seq_aux_loss_coeff,
            fused=self.config.moe_router_fusion,
        )
        / bsz  # Average over batch size
    )

    probs = self.attach_and_log_load_balancing_loss(
        probs, seq_aux_loss_coeff, aux_loss, "seq_load_balancing_loss",
        self.tp_cp_group)
    return probs
```

**Global Level** (`megatron/core/transformer/moe/router.py:341-377`):

```python
def _apply_global_aux_loss(
    self, probs: torch.Tensor, scores_for_aux_loss: torch.Tensor,
    routing_map: torch.Tensor
):
    """Global batch level auxiliary loss with running average.

    Uses all-reduce across DP×TP×CP to compute global statistics,
    then maintains running average over gradient accumulation steps.
    """
    global_aux_loss_coeff = self.get_aux_loss_coeff("global_aux_loss")
    if global_aux_loss_coeff == 0:
        return probs

    tokens_per_expert = routing_map.sum(dim=0)

    # All-reduce across DP×TP×CP (wider group than micro-batch)
    tokens_per_expert = reduce_from_tensor_model_parallel_region(
        tokens_per_expert, self.tp_dp_cp_group)

    # Accumulate tokens over multiple steps
    self.global_tokens_per_expert += tokens_per_expert
    self.ga_steps += 1

    # Use running average (smoother load balancing signal)
    averaged_tokens_per_expert = self.global_tokens_per_expert / self.ga_steps

    num_tokens = scores_for_aux_loss.shape[0]
    total_num_tokens = num_tokens * self.tp_dp_cp_group.size()

    global_aux_loss = switch_load_balancing_loss_func(
        probs=scores_for_aux_loss,
        tokens_per_expert=averaged_tokens_per_expert,
        total_num_tokens=total_num_tokens,
        topk=self.topk,
        num_experts=self.config.num_moe_experts,
        moe_aux_loss_coeff=global_aux_loss_coeff,
        fused=self.config.moe_router_fusion,
    )

    probs = self.attach_and_log_load_balancing_loss(
        probs, global_aux_loss_coeff, global_aux_loss,
        "global_load_balancing_loss", self.tp_dp_cp_group)
    return probs
```

## Configuration Flags

Key parameters in `TransformerConfig` (`megatron/core/transformer/transformer_config.py:425-588`):

```python
# === MoE Basic Configuration ===
num_moe_experts: Optional[int] = None
moe_ffn_hidden_size: Optional[int] = None
moe_layer_freq: Union[int, List[int]] = 1  # Hybrid MoE/Dense

# === Router Configuration ===
moe_router_topk: int = 2
moe_router_load_balancing_type: Union[str, List[str]] = "aux_loss"
moe_router_score_function: str = "softmax"  # "sigmoid"
moe_router_pre_softmax: bool = False
moe_router_topk_scaling_factor: Optional[float] = None
moe_router_fusion: bool = False  # Fused kernels
moe_router_dtype: Optional[str] = None  # "fp32", "fp64"

# === Group-Limited Routing (DeepSeekV2/V3) ===
moe_router_num_groups: Optional[int] = None
moe_router_group_topk: Optional[int] = None

# === Load Balancing Loss ===
moe_aux_loss_coeff: Union[float, List[float]] = 0.0
moe_z_loss_coeff: Optional[float] = None

# === Token Dropping (Expert Capacity) ===
moe_expert_capacity_factor: Optional[float] = None
moe_pad_expert_input_to_capacity: bool = False
moe_token_drop_policy: str = "probs"  # "position"

# === Dynamic Expert Bias ===
moe_router_enable_expert_bias: bool = False
moe_router_bias_update_rate: float = 1e-3

# === Regularization ===
moe_input_jitter_eps: Optional[float] = None

# === Token Dispatching ===
moe_token_dispatcher_type: str = "allgather"  # "alltoall", "flex"

# === Optimization ===
moe_grouped_gemm: bool = False
moe_permute_fusion: bool = False
moe_layer_recompute: bool = False
```

## Performance Optimizations

- **Fused Operations**: Fused topk, permutation, unpermutation via Transformer Engine
- **Grouped GEMM**: Expert computation optimization for multiple experts
- **Deterministic Algorithms**: `index_add` when deterministic mode enabled
- **Activation Recomputation**: Layer-wise recomputation to reduce memory

## Critical Insights

1. **No Traditional Expert Dropout**: Megatron-LM doesn't randomly drop experts. Instead, it uses deterministic token dropping based on capacity factors.

2. **Distributed Coordination**: All load balancing mechanisms properly handle all-reduce operations across TP, DP, EP, and CP groups for distributed training.

3. **Gradient Scaling**: `MoEAuxLossAutoScaler` ensures auxiliary loss gradients scale appropriately relative to main loss, preventing imbalance.

4. **Flexible Load Balancing**: Support for multiple simultaneous loss types enables fine-tuned control over token distribution at different granularities.

## References

- Switch Transformer: https://arxiv.org/abs/2101.03961
- ST-MoE: https://arxiv.org/abs/2202.08906
- DeepSeekV3: https://arxiv.org/abs/2408.15664
- Global Load Balancing Loss: https://arxiv.org/abs/2501.11873

**Files**:
- `megatron/core/transformer/moe/moe_utils.py`: Core utilities (35-712)
- `megatron/core/transformer/moe/router.py`: Router implementation (129-572)
- `megatron/core/transformer/moe/moe_layer.py`: MoE layer orchestration
- `megatron/core/transformer/transformer_config.py`: Configuration flags
