# 42. Expert Parallel Optimizations (Comprehensive Guide)

## Overview

Expert Parallelism (EP) is a distributed training strategy specifically designed for Mixture-of-Experts (MoE) models. Unlike data or tensor parallelism that replicate or shard all model components uniformly, EP partitions only the expert layers across devices while keeping other components (attention, embeddings) replicated or sharded via other parallelism strategies.

This report consolidates all expert parallel optimizations in Megatron-LM, covering:
1. **Token Dispatching Strategies** - How tokens are routed to experts across devices
2. **Expert Computation Optimizations** - Grouped GEMM and fused kernels
3. **Communication Optimizations** - Overlapping and hiding EP communication latency
4. **Load Balancing** - Ensuring balanced expert utilization
5. **Multi-dimensional Parallelism** - Composing EP with TP, PP, DP, and CP

---

## 1. Token Dispatching Strategies

### The Core Problem

In MoE models, each token must be routed to its assigned expert(s), which may reside on different GPUs. A token on GPU 0 might need Expert 5 on GPU 3, requiring inter-device communication. The token dispatcher handles this routing efficiently.

### Three Dispatcher Implementations

Megatron-LM provides three token dispatcher strategies optimized for different scenarios:

| Dispatcher | Best For | Communication Pattern | Memory Overhead |
|------------|----------|----------------------|-----------------|
| **AllGather** | EP ≤ 4 | AllGather + ReduceScatter | O(EP × tokens) |
| **AlltoAll** | EP > 4 | Point-to-point AlltoAll | O(tokens) |
| **Flex (DeepEP)** | EP ≥ 8, maximum perf | Fused permute + AlltoAll | Pre-allocated |

### 1.1 AllGather Token Dispatcher

**When to use:** Small expert parallel sizes (EP ≤ 4)

**How it works:**
1. Router determines which expert each token needs
2. AllGather all tokens to all EP ranks (each rank sees all tokens)
3. Each rank masks to extract tokens for its local experts
4. Process expert computation locally
5. ReduceScatter results back to original ranks

**Implementation:** `megatron/core/transformer/moe/token_dispatcher.py:197-331`

```python
class MoEAllGatherTokenDispatcher(MoETokenDispatcher):
    """
    Token dispatcher using AllGather for small EP sizes.
    Communication: AllGather across TP*EP group, then ReduceScatter back.
    """

    def token_dispatch(self, hidden_states, probs):
        # AllGather tokens across TP*EP group
        # [num_local_tokens, H] -> [num_global_tokens, H]
        if self.tp_size > 1 or self.ep_size > 1:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                group=self.tp_ep_group,
                use_global_buffer=True  # Memory optimization
            )
            probs = gather_from_sequence_parallel_region(
                probs, group=self.tp_ep_group
            )
        return hidden_states, probs

    def dispatch_postprocess(self, hidden_states, probs):
        # Extract tokens for THIS rank's local experts
        # Uses routing_map to mask and permute tokens
        permuted_input, permuted_probs, _ = permute(
            hidden_states,
            self.local_routing_map,  # Only tokens for local experts
            probs,
            fused=self.config.moe_permute_fusion,
        )
        return permuted_input, permuted_probs

    def token_combine(self, hidden_states):
        # ReduceScatter to return combined outputs to original ranks
        if self.tp_size > 1 or self.ep_size > 1:
            hidden_states = reduce_scatter_to_sequence_parallel_region(
                hidden_states, group=self.tp_ep_group
            )
        return hidden_states
```

**Advantages:**
- Simple implementation
- Uniform communication volume (predictable)
- Good for small EP where AllGather overhead is acceptable

**Disadvantages:**
- Memory scales as O(EP × tokens) - each rank stores all tokens
- Inefficient for large EP sizes

### 1.2 AlltoAll Token Dispatcher

**When to use:** Larger expert parallel sizes (EP > 4)

**How it works:**
1. Router determines token-to-expert assignments
2. Compute send/receive counts for each rank pair
3. AlltoAll sends tokens directly to target EP ranks
4. Process expert computation on received tokens
5. AlltoAll sends results back to origin ranks

**Implementation:** `megatron/core/transformer/moe/token_dispatcher.py:333-862`

```python
class MoEAlltoAllTokenDispatcher(MoETokenDispatcher):
    """
    Token dispatcher using AlltoAll for larger EP sizes.
    More efficient than AllGather - direct point-to-point routing.
    """

    def preprocess(self, routing_map):
        # Calculate how many tokens each rank sends/receives
        num_local_tokens_per_expert = routing_map.sum(dim=0).long()

        # [ep_size]: tokens sent to each EP rank
        self.input_splits = num_local_tokens_per_expert.reshape(
            self.ep_size, self.num_local_experts
        ).sum(axis=1)

        # Gather global token distribution across TP*EP
        num_global_tokens_per_expert = gather_from_sequence_parallel_region(
            num_local_tokens_per_expert, group=self.tp_ep_group
        ).reshape(self.ep_size, self.tp_size, self.num_experts).transpose(0, 1)

        # [ep_size]: tokens received from each EP rank
        self.output_splits = num_global_tokens_per_rank[self.tp_rank]

        return num_tokens_per_local_expert

    def token_dispatch(self, permutated_local_input_tokens, permuted_probs):
        # AlltoAll: send tokens directly to target experts
        # Each rank sends/receives variable number of tokens
        global_input_tokens = all_to_all(
            self.ep_group,
            permutated_local_input_tokens,
            self.output_splits,  # How many to receive from each rank
            self.input_splits    # How many to send to each rank
        )
        global_probs = all_to_all(
            self.ep_group, permuted_probs,
            self.output_splits, self.input_splits
        )
        return global_input_tokens, global_probs

    def token_combine(self, hidden_states):
        # Reverse AlltoAll: send expert outputs back to origin ranks
        output = all_to_all(
            self.ep_group, hidden_states,
            self.input_splits,   # Now receiving what we originally sent
            self.output_splits   # Now sending what we originally received
        )
        return output
```

**Key Optimizations:**

1. **CUDA Synchronization Control:**
```python
# Configurable sync points for better overlap
cuda_sync_point: str = "before_ep_alltoall"  # Options:
# - "before_permutation_1": Sync before first permutation
# - "before_ep_alltoall": Sync before AlltoAll (default)
# - "before_permutation_2": Sync before second permutation
# - "before_finish": Sync at end
# - "no_sync": No explicit sync (async)
```

2. **GPU-to-CPU Transfer Overlap:**
```python
# Metadata transfers on side stream to overlap with computation
def maybe_move_tensor_to_cpu(tensor, cuda_dtoh_stream):
    with torch.cuda.stream(cuda_dtoh_stream):
        return tensor.cpu()  # Non-blocking transfer
```

3. **Fused Chunk Sorting:**
```python
# Sort tokens by local expert for better memory access
if self.sort_input_by_local_experts:
    hidden_states = fused_sort_chunks_by_index(
        hidden_states, tokens_per_expert, sorted_indices
    )
```

**Advantages:**
- Memory efficient: O(tokens) not O(EP × tokens)
- Scales well to large EP sizes
- Supports variable token counts per expert

**Disadvantages:**
- More complex implementation
- Multiple communication rounds
- Requires careful synchronization

### 1.3 Flex Token Dispatcher (DeepEP)

**When to use:** Maximum performance for EP ≥ 8, requires DeepEP library

**How it works:**
1. Fuses token permutation with AlltoAll communication
2. Uses custom CUDA kernels optimized for MoE routing
3. Pre-allocated buffers eliminate dynamic memory allocation
4. Supports async communication for better overlap

**Implementation:** `megatron/core/transformer/moe/token_dispatcher.py:1138-1328`

```python
class MoEFlexTokenDispatcher(MoETokenDispatcher):
    """
    Token dispatcher using DeepEP fused kernels.
    Combines permutation + AlltoAll in single optimized operation.
    """

    def __init__(self, config, ...):
        # Initialize DeepEP communication manager
        self._comm_manager = _DeepepManager(
            num_experts=config.num_moe_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=config.hidden_size,
            params_dtype=config.params_dtype,
            ep_group=self.ep_group,
        )

    def token_dispatch(self, hidden_states, probs=None,
                       async_finish=True, allocate_on_comm_stream=True):
        """
        Fused permutation + AlltoAll using DeepEP kernels.
        Single kernel call replaces multiple operations.
        """
        return (
            self._comm_manager.dispatch(
                hidden_states,
                async_finish,           # Non-blocking for overlap
                allocate_on_comm_stream # Allocate on comm stream
            ),
            self._comm_manager.dispatched_probs,
        )

    def token_combine(self, hidden_states, async_finish=True,
                      allocate_on_comm_stream=True):
        """Fused un-permutation + AlltoAll for combining expert outputs"""
        return self._comm_manager.combine(
            hidden_states, async_finish, allocate_on_comm_stream
        )
```

**DeepEP Manager Implementation:** `megatron/core/transformer/moe/token_dispatcher.py:907-1136`

```python
class _DeepepManager:
    """Manages DeepEP fused dispatch/combine operations."""

    def dispatch(self, hidden_states, async_finish, allocate_on_comm_stream):
        # Compute dispatch layout (token counts per rank/expert)
        # Single fused kernel: permutation + AlltoAll + re-routing
        dispatched_tokens, handle = deepep.fused_dispatch(
            hidden_states,
            self.routing_indices,
            self.routing_probs,
            self.send_buffer,      # Pre-allocated
            self.recv_buffer,      # Pre-allocated
            async_finish=async_finish,
        )
        self._handle = handle  # Save for combine operation
        return dispatched_tokens

    def combine(self, expert_output, async_finish, allocate_on_comm_stream):
        # Use saved handle for inverse operation
        # Single fused kernel: combine + AlltoAll + un-permutation
        return deepep.fused_combine(
            expert_output,
            self._handle,
            async_finish=async_finish,
        )
```

**Fused A2A Implementation:** `megatron/core/transformer/moe/fused_a2a.py:68-265`

```python
class FusedDispatch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden_states, routing_map, probs, config):
        # Fused forward: permute + AlltoAll in single kernel
        dispatched, indices, probs_out, tokens_per_expert, handle = (
            deepep_dispatch(
                hidden_states,
                routing_map,
                probs,
                config.nvl_buffer,   # NVLink buffer
                config.rdma_buffer,  # RDMA buffer
                async_finish=config.async_finish,
            )
        )
        ctx.save_for_backward(handle)
        return dispatched, indices, probs_out, tokens_per_expert

    @staticmethod
    def backward(ctx, grad_output, ...):
        # Backward uses saved handle for inverse combine
        handle = ctx.saved_tensors[0]
        return deepep_combine(grad_output, handle)
```

**Advantages:**
- Maximum throughput (3-4x faster than naive AllGather)
- Single kernel eliminates overhead
- Pre-allocated buffers reduce memory fragmentation
- Async support enables computation-communication overlap

**Disadvantages:**
- Requires DeepEP library installation
- Hardware-specific optimizations (NVLink, RDMA)
- More complex debugging

---

## 2. Expert Computation Optimizations

### The Challenge

After tokens are dispatched to their target experts, we need to compute expert MLPs efficiently. With many experts (64, 128, 256+), naive sequential execution is extremely slow due to:
- Kernel launch overhead (one launch per expert)
- Poor GPU utilization (small matrices per expert)
- Memory bandwidth inefficiency

### 2.1 Grouped GEMM (GroupedMLP)

**Key Idea:** Batch all expert computations into a single kernel call using grouped/batched GEMM operations.

**Implementation:** `megatron/core/transformer/moe/experts.py:100-307`

```python
class GroupedMLP(MegatronModule):
    """
    Expert MLP using Grouped GEMM for efficient multi-expert computation.
    Single kernel call processes all local experts simultaneously.
    """

    def __init__(self, num_local_experts, config, ...):
        # Weight layout: [hidden_size, ffn_hidden_size * num_local_experts]
        # All expert weights concatenated for grouped GEMM
        self.weight1 = Parameter(torch.empty(
            config.hidden_size,
            config.ffn_hidden_size * num_local_experts * glu_factor,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        ))

        self.weight2 = Parameter(torch.empty(
            config.ffn_hidden_size * num_local_experts,
            config.hidden_size,
            device=torch.cuda.current_device(),
            dtype=config.params_dtype,
        ))

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        # Reshape weights for grouped execution
        # [hidden_size, ffn_hidden_size * num_experts] -> [num_experts, hidden_size, ffn_hidden_size]
        w1 = self.weight1.view(self.num_local_experts, self.config.hidden_size, -1)
        w2 = self.weight2.view(self.num_local_experts, -1, self.config.hidden_size)

        # GROUPED GEMM: Single kernel for all experts!
        # tokens_per_expert tells kernel how many tokens each expert has
        fc1_output = grouped_gemm.ops.gmm(
            permuted_local_hidden_states,  # [total_tokens, hidden]
            w1,                             # [num_experts, hidden, ffn]
            tokens_per_expert,              # [num_experts] - token counts
            trans_b=False
        )
        # ^ This is THE KEY optimization - one kernel call, all experts

        # Fused activation with probability weighting
        intermediate = self.activation_func_with_probs(
            fc1_output, permuted_probs.unsqueeze(-1)
        )

        # Second grouped GEMM
        fc2_output = grouped_gemm.ops.gmm(
            intermediate,
            w2,
            tokens_per_expert,
            trans_b=False
        )

        return fc2_output, None
```

**Fused Activation with Probability Scaling:**

```python
# JIT-compiled fused kernel for activation + probability weighting
@torch.jit.script
def activation_func_with_probs(x, probs):
    # Fuses: activation(x) * probs in single pass
    # Reduces memory bandwidth by avoiding intermediate tensor
    return F.silu(x) * probs
```

**Performance Impact:**
- **3-5x faster** than sequential expert execution
- Single kernel launch vs N kernel launches (N = num_local_experts)
- Better GPU utilization through batching

### 2.2 TEGroupedMLP (TransformerEngine Integration)

**Key Idea:** Leverage TransformerEngine's GroupedLinear for FP8 support and advanced fusions.

**Implementation:** `megatron/core/transformer/moe/experts.py:746-1012`

```python
class TEGroupedMLP(MegatronModule):
    """
    Expert MLP using TransformerEngine's GroupedLinear.
    Supports FP8 training and advanced kernel fusions.
    """

    def __init__(self, num_local_experts, config, ...):
        # TE GroupedLinear with FP8 support
        self.linear_fc1 = TEGroupedLinear(
            num_local_experts,
            config.hidden_size,
            config.ffn_hidden_size * glu_factor,
            config=config,
            init_method=config.init_method,
            bias=config.add_bias_linear,
            skip_bias_add=True,
            tp_group=self.expt_tp_group,  # Expert-specific TP group
        )

        self.linear_fc2 = TEGroupedLinear(
            num_local_experts,
            config.ffn_hidden_size,
            config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=config.add_bias_linear,
            skip_bias_add=True,
            tp_group=self.expt_tp_group,
        )

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        tokens_per_expert_list = tokens_per_expert.tolist()

        # FP8 padding for alignment (if using FP8)
        if self.config.fp8:
            permuted_local_hidden_states, tokens_per_expert_list = self.fp8_padding(
                permuted_local_hidden_states, tokens_per_expert_list
            )

        # TE GroupedLinear forward with per-expert token counts
        intermediate, bias = self.linear_fc1(
            permuted_local_hidden_states, tokens_per_expert_list
        )

        # Fused bias-activation-weighting (if supported)
        if self.config.bias_activation_fusion:
            if self.activation_func == F.silu and self.config.gated_linear_unit:
                # Fused SwiGLU + bias + probability weighting
                intermediate = weighted_bias_swiglu_impl(
                    intermediate, bias, permuted_probs
                )
            else:
                intermediate = self.activation_func(intermediate + bias)
                intermediate = intermediate * permuted_probs.unsqueeze(-1)
        else:
            intermediate = self.activation_func(intermediate + bias)
            intermediate = intermediate * permuted_probs.unsqueeze(-1)

        # Second layer
        output, output_bias = self.linear_fc2(intermediate, tokens_per_expert_list)

        # FP8 unpadding
        if self.config.fp8:
            output = self.fp8_unpadding(output, actual_tokens_per_expert)

        return output, output_bias
```

**FP8 Padding/Unpadding for Alignment:**

```python
def fp8_padding(self, hidden_states, tokens_per_expert):
    """Pad tokens to FP8 alignment boundaries for efficient quantization."""
    padded_tokens = []
    padded_counts = []
    for i, count in enumerate(tokens_per_expert):
        # Pad to multiple of 16 for FP8 tensor cores
        padded_count = ((count + 15) // 16) * 16
        if padded_count > count:
            padding = torch.zeros(padded_count - count, hidden_states.size(-1),
                                  device=hidden_states.device, dtype=hidden_states.dtype)
            # ... append padding
        padded_counts.append(padded_count)
    return padded_hidden_states, padded_counts
```

**Advantages over GroupedMLP:**
- FP8 training support (significant memory and compute savings)
- Advanced fusions (bias + activation + GLU + probability)
- Better integration with TE's FP8 recipe system

### 2.3 SequentialMLP (Fallback)

**When to use:** Debugging, compatibility, or when GroupedGEMM unavailable

**Implementation:** `megatron/core/transformer/moe/experts.py:1014-1167`

```python
class SequentialMLP(MegatronModule):
    """
    Sequential expert execution - processes one expert at a time.
    Used for debugging or when GroupedGEMM is unavailable.
    """

    def __init__(self, num_local_experts, config, ...):
        # Create separate MLP for each local expert
        self.local_experts = torch.nn.ModuleList([
            MLP(config, submodules, is_expert=True)
            for _ in range(num_local_experts)
        ])

    def forward(self, permuted_local_hidden_states, tokens_per_expert, permuted_probs):
        tokens_per_expert_list = tokens_per_expert.tolist()

        # Split tokens by expert
        tokens_list = torch.split(permuted_local_hidden_states, tokens_per_expert_list)
        probs_list = torch.split(permuted_probs, tokens_per_expert_list)

        output_list = []
        # Process each expert SEQUENTIALLY
        for expert, tokens, probs in zip(self.local_experts, tokens_list, probs_list):
            if tokens.numel() > 0:
                output, _ = expert(tokens)
                output = output * probs.unsqueeze(-1)
                output_list.append(output)

        output = torch.cat(output_list, dim=0)
        return output, None
```

**Performance Comparison:**

| Implementation | Kernel Launches | FP8 Support | Relative Speed |
|---------------|-----------------|-------------|----------------|
| SequentialMLP | N (num_experts) | No | 1x (baseline) |
| GroupedMLP | 2 | No | 3-5x |
| TEGroupedMLP | 2 | Yes | 3-5x (+ FP8 benefits) |

---

## 3. Communication Optimizations

### 3.1 Batch-Level Overlapping (Dense-Expert Overlap)

**The Problem:** EP All-to-All communication is a significant bottleneck. GPUs idle during communication, and the network idles during computation.

**The Solution:** Overlap dense computation from one micro-batch with expert communication from another micro-batch using separate CUDA streams.

**Implementation:** `megatron/core/pipeline_parallel/combined_1f1b.py`

```
Micro-batch flow (overlapped):
comm_stream: [dispatch_fwd]──────────────[combine_fwd]────────────[dispatch_bwd]
comp_stream: [attn_fwd]→[post_attn_fwd]→[mlp_fwd]→[attn_bwd]→[post_attn_bwd]→[mlp_bwd]
             └─────── Microbatch N ──────┴─────── Microbatch N+1 ───────┘

Key insight: EP A2A from microbatch N overlaps with attention/MLP of microbatch N+1
```

**Fine-Grained Layer Decomposition:**

The optimization decomposes each transformer layer into 5 independently schedulable sub-modules:

```python
# From megatron/core/models/gpt/fine_grained_callables.py
class TransformerLayerCallable:
    """Decomposes transformer layer for fine-grained scheduling."""

    def __init__(self, layer, delay_wgrad_compute=False):
        # Sub-modules for attention (compute stream)
        self.attention_callable = AttentionCallable(layer.self_attention)
        self.post_attention_callable = PostAttentionCallable(layer.pre_mlp_layernorm)

        # Sub-modules for MoE (mixed streams)
        if isinstance(layer.mlp, MoELayer):
            self.moe_callable = MoELayerCallable(layer.mlp, delay_wgrad_compute)


class MoELayerCallable:
    """Decomposes MoE into dispatch/expert/combine for stream scheduling."""

    def __init__(self, moe_layer, delay_wgrad_compute=False):
        self.dispatch_callable = MoEDispatchCallable(moe_layer.token_dispatcher)
        self.expert_callable = ExpertComputeCallable(moe_layer.experts)
        self.combine_callable = MoECombineCallable(moe_layer.token_dispatcher)

    def forward(self, hidden_states):
        # 1. Router and preprocessing (compute stream)
        routing_probs, routing_map = self.moe_layer.router(hidden_states)

        # 2. Dispatch tokens (COMMUNICATION STREAM)
        with torch.cuda.stream(get_comm_stream()):
            dispatched_tokens = self.dispatch_callable.forward(hidden_states, routing_map)

        # 3. Expert computation (compute stream)
        get_compute_stream().wait_stream(get_comm_stream())
        expert_output = self.expert_callable.forward(dispatched_tokens, routing_probs)

        # 4. Combine expert outputs (COMMUNICATION STREAM)
        with torch.cuda.stream(get_comm_stream()):
            get_comm_stream().wait_stream(get_compute_stream())
            output = self.combine_callable.forward(expert_output, routing_map)

        get_compute_stream().wait_stream(get_comm_stream())
        return output
```

**Combined 1F1B Schedule:**

```python
# From megatron/core/pipeline_parallel/combined_1f1b.py:18-108
def combined_1f1b_schedule_for_no_pipelining(
    forward_step_func, data_iterator, model, num_microbatches, ...
):
    """
    Combined 1F1B schedule with batch-level overlapping.

    Schedule (4 microbatches):
    Phase 0: Microbatch 0 forward only
    Phase 1: Microbatch 0 backward + Microbatch 1 forward  ← Overlap!
    Phase 2: Microbatch 1 backward + Microbatch 2 forward  ← Overlap!
    Phase 3: Microbatch 2 backward + Microbatch 3 forward  ← Overlap!
    Phase 4: Microbatch 3 backward only
    """
    # Phase 0: First microbatch forward
    output_tensor = forward_step_helper(model, ...)

    # Phases 1 to N-1: Overlapped forward + backward
    for i in range(num_microbatches - 1):
        # Co-schedule: backward(i) overlaps with forward(i+1)
        # They run on different streams!
        output_tensor = forward_step_helper(model, ...)  # Next forward
        backward_step_helper(model, ...)                  # Current backward

    # Final phase: Last backward only
    backward_step_helper(model, ...)
```

**Performance Impact:**

```
Without overlap (sequential):
Timeline: [Attention 800μs][idle 800μs][Dispatch A2A 800μs][Expert 900μs][idle 800μs][Combine A2A 800μs]
Total: ~4.9ms per MoE layer

With overlap:
Timeline: [Attn_N][Attn_N+1][Expert_N overlapped with Dispatch_N+1][Expert_N+1]...
Effective: ~3.2ms per MoE layer (35% faster)
```

**Configuration:**

```bash
# Enable batch-level overlapping
--overlap-moe-expert-parallel-comm

# Optional: split dgrad and wgrad for finer overlap
--delay-wgrad-compute

# Required environment
export CUDA_DEVICE_MAX_CONNECTIONS=1
```

**Requirements:**
- PyTorch >= 2.6.0 (earlier versions have stream sync bugs)
- AlltoAll or Flex dispatcher (not AllGather)
- bf16 or fp16 (not fp32)
- Cannot use with `--recompute-granularity=full`

### 3.2 Shared Expert Overlap

**Key Idea:** Shared experts (non-routed dense experts) can execute in parallel with token dispatcher communication.

**Implementation:** `megatron/core/transformer/moe/shared_experts.py:30-287`

```python
class SharedExpertMLP(MLP):
    """
    Shared expert that processes ALL tokens (not routed).
    Runs on separate CUDA stream to overlap with dispatcher communication.
    """

    def __init__(self, config, submodules, gate, ...):
        # Dedicated CUDA stream for shared expert computation
        self._shared_expert_stream = torch.cuda.Stream()

        # Optional gating mechanism
        self.use_shared_expert_gate = gate
        if self.use_shared_expert_gate:
            self.gate_weight = torch.nn.Parameter(
                torch.empty((1, config.hidden_size))
            )

    def pre_forward_comm(self, hidden_states):
        """AllGather for sequence parallel + cache input."""
        if self.sequence_parallel:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states, use_global_buffer=True
            )
        self._cached_input = hidden_states
        return hidden_states

    def linear_fc1_forward_and_act(self):
        """FC1 + activation on dedicated stream."""
        with torch.cuda.stream(self._shared_expert_stream):
            intermediate, _ = self.linear_fc1(self._cached_input)
            if self.config.gated_linear_unit:
                intermediate = self.activation_func(intermediate)
            else:
                intermediate = self.activation_func(intermediate)
            self._cached_intermediate = intermediate

    def linear_fc2_forward(self):
        """FC2 on dedicated stream."""
        with torch.cuda.stream(self._shared_expert_stream):
            self._cached_output, _ = self.linear_fc2(self._cached_intermediate)

    def post_forward_comm(self):
        """ReduceScatter for sequence parallel."""
        with torch.cuda.stream(self._shared_expert_stream):
            if self.sequence_parallel:
                self._cached_output = reduce_scatter_to_sequence_parallel_region(
                    self._cached_output
                )

    def get_output(self):
        """Synchronize and apply optional gate."""
        # Wait for shared expert stream to complete
        torch.cuda.current_stream().wait_stream(self._shared_expert_stream)

        output = self._cached_output
        if self.use_shared_expert_gate:
            gate_score = torch.sigmoid(F.linear(self._cached_input, self.gate_weight))
            output = output * gate_score

        return output
```

**Overlap Timeline:**

```
Main stream:    [Router][Dispatch preprocess]──────────────────[Dispatch A2A]────[Expert compute]
Shared stream:  ──────────[Shared FC1+Act]──[Shared FC2]──[ReduceScatter]────────────────────
                └─────── Shared expert overlaps with dispatch prep ─────────┘
```

**Configuration:**

```bash
# Enable shared expert overlap
--moe-shared-expert-overlap

# Configure shared expert size
--moe-shared-expert-intermediate-size 2048
```

**Note:** Cannot use with `--overlap-moe-expert-parallel-comm` (batch-level overlap)

### 3.3 Token Permutation Optimizations

**Fused Permutation:**

```python
# From megatron/core/transformer/moe/moe_utils.py:219-303
def permute(tokens, routing_map, probs=None, num_out_tokens=None, fused=False):
    """
    Permute tokens: group by expert for efficient computation.

    Fused version uses single kernel for:
    - Transpose routing_map
    - Masked select for indices
    - Gather for token reordering
    """
    if fused and probs is None:
        return fused_permute(tokens, routing_map, num_out_tokens=num_out_tokens)

    if fused and probs is not None:
        return fused_permute_with_probs(tokens, probs, routing_map, num_out_tokens)

    # Non-fused fallback
    routing_map = routing_map.bool().T.contiguous()
    token_indices = torch.arange(num_tokens, device=routing_map.device)
    sorted_indices = token_indices.unsqueeze(0).expand(num_experts, -1).masked_select(routing_map)
    permuted_input = tokens.index_select(0, sorted_indices)

    return permuted_input, permuted_probs, sorted_indices
```

**Fused Chunk Sorting:**

```python
# Sort tokens by local expert for better memory access patterns
def fused_sort_chunks_by_index(hidden_states, tokens_per_expert, sorted_indices):
    """
    Reorder chunks to group tokens by expert.
    Improves memory locality for grouped GEMM.
    """
    # Single kernel: reorders based on expert assignment
    return sorted_hidden_states
```

---

## 4. Load Balancing

### The Problem

Without load balancing, routers often collapse to using only a few experts, causing:
- Wasted compute (unused experts)
- Memory imbalance (some GPUs overloaded)
- Poor model quality (limited capacity)

### 4.1 Auxiliary Loss (Switch Transformer Style)

**Formula:**
```
loss = E × Σ(f_i × P_i)
```
Where:
- `f_i` = fraction of tokens routed to expert i
- `P_i` = averaged router probability for expert i
- `E` = number of experts

**Implementation:** `megatron/core/transformer/moe/moe_utils.py:35-112`

```python
def switch_load_balancing_loss_func(
    probs: torch.Tensor,           # [num_tokens, num_experts]
    tokens_per_expert: torch.Tensor,  # [num_experts]
    total_num_tokens: int,
    topk: int,
    num_experts: int,
    moe_aux_loss_coeff: float,
):
    """
    Auxiliary loss encouraging balanced expert utilization.
    Added to main loss during training.
    """
    # Sum probabilities across tokens for each expert
    aggregated_probs_per_expert = probs.sum(dim=0)

    # Compute loss: penalizes when high-prob experts also get many tokens
    aux_loss = torch.sum(aggregated_probs_per_expert * tokens_per_expert) * (
        num_experts * moe_aux_loss_coeff / (topk * total_num_tokens * total_num_tokens)
    )
    return aux_loss
```

**Three Auxiliary Loss Types:**

1. **aux_loss** - Micro-batch level (GShard style)
2. **seq_aux_loss** - Per-sequence level (DeepSeekV2 style)
3. **global_aux_loss** - Global batch level with running average

```python
# Can combine multiple loss types
moe_router_load_balancing_type = ["aux_loss", "seq_aux_loss", "global_aux_loss"]
```

### 4.2 Z-Loss Regularization

**Purpose:** Prevents router logits from growing unbounded (training stability).

**Formula:**
```
z_loss = mean(square(logsumexp(logits))) × z_loss_coeff
```

**Implementation:** `megatron/core/transformer/moe/moe_utils.py:115-127`

```python
def z_loss_func(logits, z_loss_coeff):
    """
    Encourages router logits to remain small.
    From ST-MoE paper: https://arxiv.org/pdf/2202.08906.pdf
    """
    z_loss = torch.mean(torch.square(torch.logsumexp(logits, dim=-1))) * z_loss_coeff
    return z_loss
```

### 4.3 Expert Capacity and Token Dropping

**Expert Capacity:** Maximum tokens each expert can process.

```python
capacity = ceil((num_tokens / num_experts) × capacity_factor)
```

**Implementation:** `megatron/core/transformer/moe/moe_utils.py:656-712`

```python
def apply_router_token_dropping(
    routing_probs: torch.Tensor,
    routing_map: torch.Tensor,
    router_topk: int,
    capacity_factor: float,
    drop_policy: str = "probs",  # or "position"
    pad_to_capacity: bool = False,
):
    """
    Apply capacity constraints via token dropping.

    drop_policy options:
    - "probs": Drop tokens with lowest routing probabilities
    - "position": Drop tokens at end of sequence
    """
    expert_capacity = get_capacity(
        num_tokens=num_tokens * router_topk,
        num_experts=num_experts,
        capacity_factor=capacity_factor,
    )

    if drop_policy == "probs":
        # Keep top-capacity tokens by probability
        _, capacity_indices = torch.topk(routing_probs, k=expert_capacity, dim=0)
        capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()
    elif drop_policy == "position":
        # Keep first-capacity tokens by position
        _, capacity_indices = torch.topk(routing_map.int(), k=expert_capacity, dim=0)
        capacity_mask = torch.zeros_like(routing_probs).scatter(0, capacity_indices, 1).bool()

    # Apply mask
    final_map = torch.logical_and(routing_map, capacity_mask)
    final_probs = routing_probs * final_map

    return final_probs, final_map
```

### 4.4 Dynamic Expert Bias (DeepSeekV3 Style)

**Key Idea:** Auxiliary-loss-free load balancing by dynamically adjusting expert biases.

**Implementation:** `megatron/core/transformer/moe/moe_utils.py:854-872`

```python
def get_updated_expert_bias(tokens_per_expert, expert_bias, expert_bias_update_rate):
    """
    Update expert bias for load balancing without auxiliary loss.
    Ref: https://arxiv.org/abs/2408.15664v1

    - Underutilized experts: INCREASE bias (attract more tokens)
    - Overutilized experts: DECREASE bias (repel tokens)
    """
    with torch.no_grad():
        # All-Reduce across TP×CP×DP group
        torch.distributed.all_reduce(
            tokens_per_expert,
            group=parallel_state.get_tensor_and_data_parallel_group(with_context_parallel=True),
        )

        # Calculate ideal average
        average_tokens = tokens_per_expert.sum(dim=-1, keepdim=True) / tokens_per_expert.shape[-1]

        # Compute deviation
        offset = average_tokens - tokens_per_expert

        # Update bias: sign(offset) gives direction
        updated_expert_bias = expert_bias + torch.sign(offset) * expert_bias_update_rate

        return updated_expert_bias
```

**Router Integration:**

```python
class TopKRouter(Router):
    def __init__(self, config, ...):
        if self.enable_expert_bias:
            # Track token counts per expert
            self.register_buffer('local_tokens_per_expert',
                torch.zeros(config.num_moe_experts, dtype=torch.float32))
            # Learnable expert bias
            self.register_buffer('expert_bias',
                torch.zeros(config.num_moe_experts, dtype=torch.float32))

    def routing(self, logits):
        # Apply expert bias to logits before topk selection
        if self.expert_bias is not None:
            scores_for_routing = scores + self.expert_bias
            _, top_indices = torch.topk(scores_for_routing, k=self.topk, dim=1)
        # ...
```

### 4.5 Sinkhorn Load Balancing

**Key Idea:** Use Sinkhorn-Knopp algorithm for optimal balanced assignment.

```python
def sinkhorn_load_balancing(self, logits):
    """
    Sinkhorn routing for balanced expert assignment.
    Iteratively normalizes rows and columns of routing matrix.
    """
    if self.training:
        with torch.no_grad():
            # Sinkhorn algorithm produces balanced routing
            norm_logits = sinkhorn(logits.to(dtype=torch.float32))
            _, indices = torch.topk(norm_logits, k=self.topk, dim=1)

        # Use original logits for probability computation
        logits = torch.sigmoid(logits) if self.topk == 1 else torch.softmax(logits, dim=-1)
    else:
        logits = torch.sigmoid(logits) if self.topk == 1 else torch.softmax(logits, dim=-1)
        _, indices = torch.topk(logits, k=self.topk, dim=1)

    # Build routing map
    routing_map = torch.zeros_like(logits).int().scatter(1, indices, 1).bool()
    routing_probs = logits * routing_map

    return routing_probs, routing_map
```

---

## 5. Multi-dimensional Parallelism

### Process Group Architecture

Expert parallelism composes with other parallelism strategies. Megatron-LM manages multiple process groups:

**Implementation:** `megatron/core/transformer/moe/moe_utils.py:1011-1031`

```python
# ProcessGroupCollection structure for MoE
pg_collection.ep = get_expert_model_parallel_group()          # EP ranks
pg_collection.tp = get_tensor_model_parallel_group()          # TP ranks
pg_collection.cp = get_context_parallel_group()               # CP ranks
pg_collection.expt_tp = get_expert_tensor_parallel_group()    # Expert-specific TP
pg_collection.expt_dp = get_expert_data_parallel_group()      # Expert-specific DP
pg_collection.tp_ep = get_expert_tensor_and_model_parallel_group()  # Combined TP*EP
pg_collection.tp_cp = get_tensor_and_context_parallel_group()
pg_collection.tp_dp_cp = get_tensor_and_data_parallel_group(with_context_parallel=True)
```

### GPU Layout Example

```
Total GPUs = TP × PP × EP × DP
Example: 4 (TP) × 4 (PP) × 8 (EP) × 8 (DP) = 1024 GPUs

Per-GPU computation:
- Attention layers: TP=4 shards weights across 4 GPUs
- Experts: 64 total experts / 8 EP ranks = 8 local experts per GPU
- Each expert can use ETP=2 internally
```

### Expert Tensor Parallelism (ETP)

Experts can have a different tensor parallel size than attention layers:

```python
# From experts.py
class GroupedMLP(MegatronModule):
    def __init__(self, num_local_experts, config, ...):
        # Experts use expert-specific TP group (may differ from attention TP)
        self.expt_tp_group = pg_collection.expt_tp

        # Attention layers use standard TP group
        # self.tp_group = pg_collection.tp  (in attention module)

# Configuration
--expert-tensor-parallel-size 2  # ETP for experts
--tensor-model-parallel-size 8   # TP for attention
```

### Process Group Interactions

```
Token Flow in MoE Layer:

1. Input arrives (distributed across TP)
   [tokens sharded across TP=4 ranks]

2. Router computes (replicated across TP, EP)
   [each rank computes full routing]

3. Token dispatch (AllGather/AlltoAll across TP*EP)
   [tokens redistributed across EP ranks]

4. Expert compute (within EP rank, optionally ETP sharded)
   [each EP rank processes its local experts]

5. Token combine (AlltoAll/ReduceScatter back)
   [results returned to original TP distribution]

6. Output (distributed across TP, same as input)
```

---

## 6. Configuration Reference

### Basic MoE Configuration

```bash
# Expert configuration
--num-experts 64                           # Total experts in model
--expert-model-parallel-size 8             # EP size (64/8 = 8 experts per GPU)
--moe-router-topk 2                        # Route each token to top-2 experts
--moe-ffn-hidden-size 14336               # Expert FFN intermediate size

# Token dispatcher
--moe-token-dispatcher-type alltoall       # allgather, alltoall, or flex

# Expert computation
--moe-grouped-gemm                         # Use GroupedGEMM (fastest)
--disable-bias-linear                      # Required for GroupedMLP
```

### Load Balancing Configuration

```bash
# Auxiliary loss
--moe-router-load-balancing-type aux_loss  # aux_loss, seq_aux_loss, global_aux_loss
--moe-aux-loss-coeff 0.01                 # Loss coefficient

# Z-loss
--moe-z-loss-coeff 1e-3                   # Z-loss coefficient

# Token capacity
--moe-expert-capacity-factor 1.25          # 25% buffer capacity
--moe-token-drop-policy probs              # probs or position

# Dynamic expert bias
--moe-router-enable-expert-bias           # Enable bias-based balancing
--moe-router-bias-update-rate 1e-3        # Bias update rate
```

### Communication Optimization Configuration

```bash
# Batch-level overlapping
--overlap-moe-expert-parallel-comm        # Enable dense-expert overlap
--delay-wgrad-compute                     # Split dgrad/wgrad for finer overlap

# Shared expert overlap
--moe-shared-expert-overlap               # Enable shared expert overlap
--moe-shared-expert-intermediate-size 2048

# Environment
export CUDA_DEVICE_MAX_CONNECTIONS=1      # Required for overlap
```

### Full Training Script Example

```bash
python pretrain_gpt.py \
    --num-layers 64 \
    --hidden-size 7168 \
    --num-attention-heads 128 \
    --seq-length 4096 \
    \
    # MoE configuration
    --num-experts 256 \
    --expert-model-parallel-size 8 \
    --moe-router-topk 6 \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --disable-bias-linear \
    \
    # Communication overlap
    --overlap-moe-expert-parallel-comm \
    \
    # Load balancing
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    --moe-z-loss-coeff 1e-3 \
    \
    # Parallelism
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --virtual-pipeline-model-parallel-size 4 \
    \
    # Other optimizations
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --bf16
```

---

## 7. Performance Summary

### Token Dispatcher Comparison

| Dispatcher | EP ≤ 4 | EP 4-8 | EP > 8 | Memory | Complexity |
|------------|--------|--------|--------|--------|------------|
| AllGather | Best | OK | Poor | High | Low |
| AlltoAll | OK | Best | Good | Low | Medium |
| Flex (DeepEP) | OK | Good | Best | Low | High |

### Expert Computation Comparison

| Implementation | Relative Speed | FP8 | Memory | Use Case |
|---------------|---------------|-----|--------|----------|
| SequentialMLP | 1x | No | High | Debug |
| GroupedMLP | 3-5x | No | Low | Default |
| TEGroupedMLP | 3-5x | Yes | Low | FP8 training |

### Communication Overlap Impact

| Optimization | Speedup | Requirements |
|-------------|---------|--------------|
| Batch-level overlap | 15-30% | PyTorch ≥ 2.6, AlltoAll/Flex |
| Shared expert overlap | 5-15% | Shared experts enabled |
| Fused permutation | 5-10% | moe_permute_fusion=True |

### Load Balancing Comparison

| Strategy | Pros | Cons |
|----------|------|------|
| Auxiliary loss | Simple, effective | Adds to loss |
| Z-loss | Training stability | Extra computation |
| Token dropping | Memory control | May drop important tokens |
| Expert bias | No loss overhead | Requires tuning |
| Sinkhorn | Optimal balance | Higher compute cost |

---

## 8. Troubleshooting

### Common Issues

**1. Load Imbalance**
- Symptoms: Some experts get many more tokens, idle GPUs
- Fix: Enable auxiliary loss, tune coefficient, use Sinkhorn routing

**2. OOM in MoE Layers**
- Symptoms: Crashes during token routing
- Fix: Switch to AlltoAll, reduce capacity factor, enable token dropping

**3. No Speedup from Overlap**
- Symptoms: Same iteration time with overlap enabled
- Fix: Verify PyTorch ≥ 2.6, check EP > 1, profile A2A vs compute ratio

**4. Training Hangs**
- Symptoms: Freezes after enabling overlap
- Fix: Upgrade PyTorch, check CUDA_DEVICE_MAX_CONNECTIONS, disable delay_wgrad_compute

**5. Numerical Divergence**
- Symptoms: Loss NaN after enabling overlap
- Fix: Verify PyTorch ≥ 2.6, check stream sync, enable CUDA_LAUNCH_BLOCKING=1 for debug

---

## 9. References

### Papers
- Switch Transformers: https://arxiv.org/abs/2101.03961
- ST-MoE (Z-Loss): https://arxiv.org/abs/2202.08906
- DeepSeekV3: https://arxiv.org/abs/2412.19437
- Global Load Balancing: https://arxiv.org/abs/2501.11873

### Implementation Files
- Token Dispatcher: `megatron/core/transformer/moe/token_dispatcher.py`
- Expert MLPs: `megatron/core/transformer/moe/experts.py`
- Router: `megatron/core/transformer/moe/router.py`
- MoE Utils: `megatron/core/transformer/moe/moe_utils.py`
- Fused A2A: `megatron/core/transformer/moe/fused_a2a.py`
- Shared Experts: `megatron/core/transformer/moe/shared_experts.py`
- Combined 1F1B: `megatron/core/pipeline_parallel/combined_1f1b.py`
- Fine-grained Callables: `megatron/core/models/gpt/fine_grained_callables.py`

### Related Reports
- #09 Expert Parallelism Communication
- #14 Expert Parallelism (MoE)
- #20 MoE Batch-Level Overlapping
- #36 Grouped GEMM
- #39 MoE Load Balancing & Expert Dropout
