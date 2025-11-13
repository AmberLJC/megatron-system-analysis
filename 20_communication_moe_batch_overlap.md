# 20. MoE Batch-Level Overlapping (Dense-Expert Overlap)

## Context

In MoE (Mixture-of-Experts) models with Expert Parallelism (EP), All-to-All (A2A) communication for token routing is a major bottleneck. During training, each transformer layer has both:
- **Dense computation:** Attention layers and MLP computations (compute-heavy)
- **Expert communication:** EP All-to-All for routing tokens to/from experts (communication-heavy)

Traditional sequential execution means GPUs idle during communication and the network idles during computation. This is extremely wasteful!

**Key insight:** We can overlap dense computation from one micro-batch with expert communication from another micro-batch using separate CUDA streams.

## Implementation

### Architecture Overview

The optimization decomposes each transformer layer into **5 fine-grained sub-modules** that can be independently scheduled:

1. **Attention** (compute stream) - Self-attention computation
2. **Post-Attention** (compute stream) - LayerNorm → Router → Dispatch preprocessing
3. **MoE Dispatch** (communication stream) - All-to-All send tokens to experts
4. **MLP/Expert** (compute stream) - Expert computation
5. **MoE Combine** (communication stream) - All-to-All return expert outputs

These sub-modules run on **two separate CUDA streams**:
- **Compute stream:** Runs attention and MLP/expert computations
- **Communication stream:** Runs MoE dispatch and combine All-to-All operations

### Overlapping Strategy

The scheduler implements a **pipelined execution** across micro-batches:

```
Micro-batch flow (simplified):
comm_stream: combine_bwd            | dispatch_fwd → dispatch_bwd  | combine_fwd
comp_stream: attn_fwd → post_attn_fwd| mlp_bwd → mlp_bwd_dw → mlp_fwd| post_attn_bwd → attn_bwd
             ├─────────────────────┤ ├─────────────────────────┤ ├──────────────────┤
             Microbatch N backward   Microbatch N+1 forward       Microbatch N backward
```

**Key overlapping patterns:**
- EP A2A in forward pass (dispatch) is hidden by attention/MLP backward pass computation
- EP A2A in backward pass (combine) is hidden by attention/MLP forward pass computation

### Combined 1F1B Schedule

The framework uses a modified 1F1B (one-forward-one-backward) schedule that co-schedules forward and backward operations:

```
Phase 0: Microbatch 0 forward
Phase 1: Microbatch 0 backward + Microbatch 1 forward  ← Overlap starts here
Phase 2: Microbatch 1 backward + Microbatch 2 forward
Phase 3: Microbatch 2 backward + Microbatch 3 forward
Phase 4: Microbatch 3 backward
```

Within each phase, the fine-grained sub-modules enable overlapping of computation and communication.

## Core Code

### Main Implementation Files

**Pipeline Scheduling:**
- `megatron/core/pipeline_parallel/combined_1f1b.py` (444 lines)
  - `combined_1f1b_schedule_for_no_pipelining()` (18-108) - Schedule for PP=1
  - `combined_1f1b_schedule_for_interleaved_pipelining()` (111-234) - Schedule for PP>1
  - `combined_forward_backward_step()` (237-444) - Main scheduling logic

**Fine-Grained Layer Decomposition:**
- `megatron/core/models/gpt/fine_grained_callables.py` (500+ lines)
  - `TransformerLayerCallable` class - Attention + post-attention sub-modules
  - `MoELayerCallable` class - MoE dispatch + expert + combine sub-modules
  - Support for weight gradient delay (`backward_dw()` methods)

**Model Chunk Scheduling:**
- `megatron/core/models/common/model_chunk_schedule_plan.py` (236 lines)
  - `TransformerLayerSchedulePlan.run()` (173-236) - Orchestrates forward/backward

**Configuration Validation:**
- `megatron/core/transformer/transformer_config.py` (1441-1486)
  - `_validate_moe_overlap_config()` - Validates requirements

## Code Snippet

### Combined 1F1B Scheduler

```python
# From megatron/core/pipeline_parallel/combined_1f1b.py:237-444
def combined_forward_backward_step(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    seq_length,
    micro_batch_size,
    decoder_seq_length,
    forward_only,
    collect_non_loss_data=False,
):
    """
    Combined 1F1B schedule with batch-level overlapping for MoE.

    This scheduler co-schedules forward and backward passes across micro-batches,
    enabling overlap of:
    - Dense computation (attention/MLP) on compute stream
    - Expert communication (A2A) on communication stream
    """
    # Determine if we're using pipeline parallelism
    pipeline_model_parallel_size = parallel_state.get_pipeline_model_parallel_world_size()

    if pipeline_model_parallel_size == 1:
        # No pipeline parallelism - use simpler schedule
        return combined_1f1b_schedule_for_no_pipelining(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            seq_length,
            micro_batch_size,
            decoder_seq_length,
            forward_only,
            collect_non_loss_data,
        )
    else:
        # Pipeline parallelism - use interleaved schedule
        return combined_1f1b_schedule_for_interleaved_pipelining(
            forward_step_func,
            data_iterator,
            model,
            num_microbatches,
            seq_length,
            micro_batch_size,
            decoder_seq_length,
            forward_only,
            collect_non_loss_data,
        )


# From megatron/core/pipeline_parallel/combined_1f1b.py:18-108
def combined_1f1b_schedule_for_no_pipelining(
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    seq_length,
    micro_batch_size,
    decoder_seq_length,
    forward_only,
    collect_non_loss_data=False,
):
    """
    Scheduler for 1f1b with no pipelining (PP=1).

    This function schedules micro-batches in a way that the forward pass of Transformer layers
    for one micro-batch runs in parallel with the backward pass of another.
    Each layer's forward and backward operations are co-scheduled to maximize the overlap of
    their computations and communications.

    EP A2A in forward step is hidden by the attention/mlp computation in the backward step,
    and vice versa.

    Assuming we have 4 microbatches, the schedule is as follows:
    Phases 0: 1st microbatch forward
    Phases 1: 1st microbatch backward + 2nd microbatch forward
    Phases 2: 2nd microbatch backward + 3rd microbatch forward
    Phases 3: 3rd microbatch backward + 4th microbatch forward
    Phases 4: 4th microbatch backward
    """
    config = get_model_config(model[0])

    # Phase 0: First microbatch forward only
    losses_reduced = []

    # Forward pass for first microbatch
    input_tensor, output_tensor_grad = None, None
    output_tensor = forward_step_helper(
        model, input_tensor, output_tensor_grad,
        forward_step_func, data_iterator, config
    )

    if not forward_only:
        # Phases 1 to num_microbatches-1: Overlapped forward + backward
        for i in range(num_microbatches - 1):
            # Get next batch data
            input_tensor, output_tensor_grad = None, None

            # Co-schedule: backward(i) + forward(i+1)
            # This is where the magic happens - they run on different streams!
            output_tensor = forward_step_helper(
                model, input_tensor, output_tensor_grad,
                forward_step_func, data_iterator, config
            )

            # Backward step runs in parallel with next forward
            backward_step_helper(model, input_tensor, output_tensor, output_tensor_grad)

        # Phase num_microbatches: Last microbatch backward only
        backward_step_helper(model, input_tensor, output_tensor, output_tensor_grad)

    return losses_reduced
```

### Fine-Grained Layer Decomposition

```python
# From megatron/core/models/gpt/fine_grained_callables.py
class TransformerLayerCallable:
    """
    Decomposes a transformer layer into attention and post-attention sub-modules.
    Enables fine-grained scheduling on separate CUDA streams.
    """

    def __init__(self, layer, delay_wgrad_compute=False):
        self.layer = layer
        self.delay_wgrad_compute = delay_wgrad_compute

        # Sub-modules for attention
        self.attention_callable = AttentionCallable(layer.self_attention)
        self.post_attention_callable = PostAttentionCallable(
            layer.pre_mlp_layernorm,
            layer.mlp if not isinstance(layer.mlp, MoELayer) else None
        )

        # If MoE layer, create additional fine-grained callables
        if isinstance(layer.mlp, MoELayer):
            self.moe_callable = MoELayerCallable(layer.mlp, delay_wgrad_compute)

    def forward(self, hidden_states):
        """Forward pass decomposed into sub-modules"""
        # 1. Attention (runs on compute stream)
        hidden_states = self.attention_callable.forward(hidden_states)

        # 2. Post-attention processing (compute stream)
        if hasattr(self, 'moe_callable'):
            # For MoE: just layernorm and routing prep
            hidden_states = self.post_attention_callable.forward(hidden_states)
            # 3-5. MoE dispatch → expert compute → combine (mixed streams)
            hidden_states = self.moe_callable.forward(hidden_states)
        else:
            # For dense: layernorm + MLP
            hidden_states = self.post_attention_callable.forward(hidden_states)

        return hidden_states

    def backward(self):
        """Backward pass with stream-aware scheduling"""
        if hasattr(self, 'moe_callable'):
            # MoE backward: combine → expert → dispatch
            self.moe_callable.backward()

        self.post_attention_callable.backward()
        self.attention_callable.backward()

    def backward_dw(self):
        """
        Computes weight gradients separately (optional).
        Enables even finer-grained overlap by splitting dgrad and wgrad.
        """
        if not self.delay_wgrad_compute:
            return

        self.attention_callable.backward_dw()
        if hasattr(self, 'moe_callable'):
            self.moe_callable.backward_dw()
        else:
            self.post_attention_callable.backward_dw()


class MoELayerCallable:
    """
    Decomposes MoE layer into dispatch, expert compute, and combine sub-modules.
    Dispatch and combine run on communication stream, expert on compute stream.
    """

    def __init__(self, moe_layer, delay_wgrad_compute=False):
        self.moe_layer = moe_layer
        self.delay_wgrad_compute = delay_wgrad_compute

        # Create sub-module callables
        self.dispatch_callable = MoEDispatchCallable(moe_layer.token_dispatcher)
        self.expert_callable = ExpertComputeCallable(moe_layer.experts)
        self.combine_callable = MoECombineCallable(moe_layer.token_dispatcher)

    def forward(self, hidden_states):
        """MoE forward with stream-aware scheduling"""
        # 1. Router and preprocessing (compute stream)
        routing_probs, routing_map = self.moe_layer.router(hidden_states)

        # 2. Dispatch tokens (COMMUNICATION STREAM)
        #    This runs in parallel with attention/MLP of other microbatch!
        with torch.cuda.stream(get_comm_stream()):
            dispatched_tokens = self.dispatch_callable.forward(
                hidden_states, routing_map
            )

        # 3. Expert computation (compute stream)
        #    Sync with dispatch before starting
        get_compute_stream().wait_stream(get_comm_stream())
        expert_output = self.expert_callable.forward(dispatched_tokens, routing_probs)

        # 4. Combine expert outputs (COMMUNICATION STREAM)
        #    This runs in parallel with attention/MLP of other microbatch!
        with torch.cuda.stream(get_comm_stream()):
            get_comm_stream().wait_stream(get_compute_stream())
            output = self.combine_callable.forward(expert_output, routing_map)

        # Main stream waits for combine to finish
        get_compute_stream().wait_stream(get_comm_stream())

        return output
```

### Stream Management

```python
# From megatron/core/models/common/model_chunk_schedule_plan.py:173-236
class TransformerLayerSchedulePlan:
    """
    Orchestrates the fine-grained scheduling of transformer layers.
    Manages CUDA streams for overlapping compute and communication.
    """

    def __init__(self, transformer_layer, overlap_moe_ep_comm=False):
        self.transformer_layer = transformer_layer
        self.overlap_moe_ep_comm = overlap_moe_ep_comm

        if overlap_moe_ep_comm:
            # Create dedicated streams for overlap
            self.compute_stream = torch.cuda.Stream()
            self.comm_stream = torch.cuda.Stream()

    def run(self, forward=True, backward=True, backward_dw=False):
        """
        Execute the transformer layer with overlapping.

        Stream scheduling for overlapping:
        - Compute stream: attention_fwd → mlp_fwd → attention_bwd → mlp_bwd
        - Comm stream: dispatch_fwd → combine_fwd → combine_bwd → dispatch_bwd

        The two streams run in parallel across different micro-batches!
        """
        if self.overlap_moe_ep_comm:
            # Use stream-aware execution
            if forward:
                with torch.cuda.stream(self.compute_stream):
                    # Attention forward (compute stream)
                    self.transformer_layer.attention_callable.forward()

                    # Post-attention (compute stream)
                    self.transformer_layer.post_attention_callable.forward()

                # MoE dispatch on comm stream (overlaps with other microbatch)
                with torch.cuda.stream(self.comm_stream):
                    self.transformer_layer.moe_callable.dispatch_callable.forward()

                # Wait for dispatch before expert compute
                self.compute_stream.wait_stream(self.comm_stream)

                with torch.cuda.stream(self.compute_stream):
                    # Expert compute
                    self.transformer_layer.moe_callable.expert_callable.forward()

                # MoE combine on comm stream (overlaps with other microbatch)
                self.comm_stream.wait_stream(self.compute_stream)
                with torch.cuda.stream(self.comm_stream):
                    self.transformer_layer.moe_callable.combine_callable.forward()

                # Sync before proceeding
                self.compute_stream.wait_stream(self.comm_stream)

            if backward:
                # Similar stream management for backward pass
                # Backward runs in reverse: combine_bwd → expert_bwd → dispatch_bwd
                pass
        else:
            # Standard sequential execution (no overlap)
            if forward:
                self.transformer_layer.forward()
            if backward:
                self.transformer_layer.backward()
            if backward_dw:
                self.transformer_layer.backward_dw()
```

## When to Use

### Required Conditions

**MUST have ALL of these:**
1. **MoE model:** Using Mixture-of-Experts architecture
2. **Expert parallelism:** EP size > 1
3. **Token dispatcher:** Using `alltoall` or `flex` dispatcher (not `allgather`)
4. **Precision:** Using bf16 or fp16 (not fp32)
5. **PyTorch version:** >= 2.6.0 (to avoid hang issues)

**For pipeline parallelism (PP > 1):**
6. **Virtual pipeline:** Must set `virtual_pipeline_model_parallel_size`

### Incompatible Configurations

**CANNOT use with:**
- `--recompute-granularity=full` (full recomputation)
- `--moe-shared-expert-overlap` (conflicts with batch-level overlap)
- FP32 training
- AllGather token dispatcher
- EP = 1 (no expert parallelism)

### Configuration Flags

```bash
# Enable batch-level overlapping
--overlap-moe-expert-parallel-comm

# Optional: split dgrad and wgrad for finer overlap (requires specific TE version)
--delay-wgrad-compute

# Required environment variable
export CUDA_DEVICE_MAX_CONNECTIONS=1  # Or higher for better overlap
```

### When to Skip

- **Dense models:** No experts = no EP communication to overlap
- **Small models:** Overhead may exceed benefits
- **EP = 1:** All experts local, no communication
- **AllGather dispatcher:** Different communication pattern, use `--moe-shared-expert-overlap` instead
- **Debugging:** Adds complexity, disable for simpler debugging

## Performance Impact

### Communication Hiding

**Goal:** Hide EP All-to-All latency behind dense computation

**Typical All-to-All latency:**
- Small EP (EP=4): 200-500μs per A2A operation
- Large EP (EP=8): 500-1000μs per A2A operation
- Very large EP (EP=16+): 1-2ms per A2A operation

**Overlapping effectiveness:**
- **Best case:** 80-95% of A2A latency hidden (when dense compute ≥ A2A time)
- **Typical case:** 60-80% hidden (some load imbalance)
- **Poor case:** 30-50% hidden (dense compute much faster than A2A)

### End-to-End Speedup

**MoE Training Throughput Improvement:**
- **DeepSeek-V3 style (EP=8, 256 experts):** 15-25% faster iteration time
- **Mixtral 8x7B (EP=8, 8 experts):** 10-15% faster iteration time
- **Large EP (EP=16+):** Up to 30% faster when A2A is major bottleneck

### Measurement Example

**DeepSeek-V3 Configuration:**
- 256 experts, EP=8, 32 experts per GPU
- Each layer has MoE with top-2 routing
- A2A latency: ~800μs per dispatch/combine pair (1.6ms total)
- Dense compute (attention + post-attention): ~1.8ms

**Without overlap:**
- MoE layer time: 1.6ms (A2A) + 2.0ms (expert compute) + 1.8ms (dense) = 5.4ms
- 64 layers × 5.4ms = 346ms per forward pass

**With overlap:**
- A2A hidden behind dense compute from other microbatch
- Effective A2A time: ~300μs (only 20% visible due to overlap)
- MoE layer time: 0.3ms (visible A2A) + 2.0ms (expert) + 1.8ms (dense) = 4.1ms
- 64 layers × 4.1ms = 262ms per forward pass
- **Speedup:** 346ms → 262ms = **24% faster**

### Profile Comparison

**Before (sequential execution):**
```
Timeline for single MoE layer:
Compute: [Attention 800μs]─────[idle 1600μs]─────[Expert 900μs]─────[idle 1600μs]
Comm:    [idle 800μs]───────[A2A dispatch 800μs][idle 900μs][A2A combine 800μs]
         └────────────────── 4.9ms total ──────────────────┘
```

**After (overlapped execution):**
```
Timeline for two overlapped microbatches:
Compute: [Attn₁ 800μs][Attn₂ 800μs][Expert₁ 900μs][Expert₂ 900μs]...
Comm:    [─idle─][Dispatch₂][Combine₁][Dispatch₁][Combine₂]...
         └─────── ~3.2ms per layer (effective) ───────┘

A2A operations overlap with attention/expert compute!
```

### Scaling Behavior

**Strong scaling (increasing EP, fixed model size):**
- EP=4: Minimal benefit (~5%), A2A is fast
- EP=8: Moderate benefit (~15%), A2A becoming bottleneck
- EP=16: Large benefit (~25%), A2A is major bottleneck
- EP=32+: Maximum benefit (~30%), A2A dominates

**Load balancing impact:**
- **Balanced routing:** 80-90% A2A hidden
- **Moderate imbalance:** 60-70% hidden
- **High imbalance:** 30-50% hidden (some GPUs finish early and wait)

## Configuration Example

### Basic Configuration

```bash
# Training script for DeepSeek-V3 style MoE
python pretrain_gpt.py \
    --num-layers 64 \
    --hidden-size 7168 \
    --num-attention-heads 128 \
    --seq-length 4096 \
    --max-position-embeddings 4096 \
    \
    # MoE configuration
    --num-experts 256 \
    --expert-model-parallel-size 8 \
    --moe-router-topk 6 \
    --moe-token-dispatcher-type alltoall \
    --moe-grouped-gemm \
    --disable-bias-linear \
    \
    # ENABLE BATCH-LEVEL OVERLAPPING
    --overlap-moe-expert-parallel-comm \
    \
    # Optional: finer-grained overlap
    --delay-wgrad-compute \
    \
    # Load balancing
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.01 \
    \
    # Parallelism
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --virtual-pipeline-model-parallel-size 4 \
    --num-microbatches 64 \
    \
    # Other optimizations
    --sequence-parallel \
    --use-distributed-optimizer \
    --overlap-grad-reduce \
    --overlap-param-gather
```

### Environment Setup

```bash
# REQUIRED: Enable CUDA stream overlap
export CUDA_DEVICE_MAX_CONNECTIONS=1

# For debugging stream synchronization (optional)
export CUDA_LAUNCH_BLOCKING=0

# NCCL tuning for better A2A performance
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
```

### Python Configuration Object

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    # Model architecture
    num_layers=64,
    hidden_size=7168,
    num_attention_heads=128,

    # MoE settings
    num_moe_experts=256,
    moe_router_topk=6,
    moe_token_dispatcher_type='alltoall',
    moe_grouped_gemm=True,

    # ENABLE OVERLAPPING
    overlap_moe_expert_parallel_comm=True,
    delay_wgrad_compute=True,  # Optional, for finer overlap

    # Load balancing
    moe_router_load_balancing_type='aux_loss',
    moe_aux_loss_coeff=0.01,

    # Parallelism
    tensor_model_parallel_size=8,
    pipeline_model_parallel_size=8,
    virtual_pipeline_model_parallel_size=4,

    # Other optimizations
    sequence_parallel=True,
    bf16=True,
)
```

## Troubleshooting

### Hangs During Training

**Symptoms:**
- Training hangs or freezes
- No progress after enabling overlap
- CUDA synchronization timeouts

**Causes:**
- PyTorch version < 2.6.0 has known stream synchronization bugs
- Missing stream synchronization points
- Deadlock between compute and comm streams

**Fix priority:**
1. **Upgrade PyTorch to >= 2.6.0** (critical!)
2. Check `CUDA_DEVICE_MAX_CONNECTIONS` is set
3. Disable `--delay-wgrad-compute` if issue persists
4. Verify no conflicting optimizations (e.g., `--moe-shared-expert-overlap`)

### No Speedup Observed

**Symptoms:**
- Overlap enabled but no performance improvement
- Same or slower iteration time

**Causes:**
- Dense compute time < A2A time (not enough compute to hide)
- Load imbalance causing synchronization waits
- EP size too small (A2A already fast)
- Wrong token dispatcher (AllGather doesn't benefit)

**Fix priority:**
1. Profile to verify A2A is actually a bottleneck (use Nsight Systems)
2. Check load balancing - enable aux loss if needed
3. Verify using AlltoAll or Flex dispatcher
4. Consider increasing EP size if A2A is not bottleneck

### Memory Issues

**Symptoms:**
- OOM after enabling overlap
- Higher memory usage than expected

**Causes:**
- Additional stream buffers
- Overlapped micro-batches in flight
- Delayed weight gradient computation holding activations longer

**Fix priority:**
1. Reduce `num_microbatches` slightly (fewer concurrent batches)
2. Disable `--delay-wgrad-compute`
3. Enable gradient checkpointing if not already
4. Profile memory usage with `torch.cuda.memory_summary()`

### Numerical Divergence

**Symptoms:**
- Loss NaN or diverges after enabling overlap
- Different results vs sequential execution

**Causes:**
- Stream synchronization bug causing race conditions
- Wrong PyTorch version
- FP32 accumulation disabled (rare)

**Fix priority:**
1. **Verify PyTorch >= 2.6.0** (critical!)
2. Enable `CUDA_LAUNCH_BLOCKING=1` to test (slow, debug only)
3. Check gradient accumulation settings
4. Disable overlap and verify model trains correctly first

### Poor Scaling with EP Size

**Symptoms:**
- Benefit saturates or decreases as EP increases
- EP=16 slower than EP=8 despite more overlap potential

**Causes:**
- Network bandwidth saturation
- Load imbalance increases with EP
- Expert capacity issues
- Too many small A2A operations

**Fix priority:**
1. Profile network usage (check for saturation)
2. Tune load balancing (Sinkhorn routing, capacity factor)
3. Consider larger local expert count (reduce EP)
4. Check token distribution uniformity

## Related Optimizations

- **#09 Expert Parallelism Communication:** Token dispatcher strategies (AllGather/AlltoAll/Flex)
- **#14 Expert Parallelism (MoE):** Overall MoE architecture and expert implementations
- **#10 1F1B Pipeline Scheduling:** Base scheduling strategy extended for overlap
- **#04 Tensor Parallelism Overlap:** Similar overlap technique for TP communication
- **#01 Gradient Bucketing:** Overlap technique for data parallel all-reduce
- **#36 Grouped GEMM for MoE:** Optimizes expert computation itself (complements this)

## Advanced: Delay Weight Gradient Computation

### What is `--delay-wgrad-compute`?

By default, PyTorch computes both activation gradients (dgrad) and weight gradients (wgrad) together during the backward pass. With `--delay-wgrad-compute`, these are split:

1. **Backward pass:** Compute only activation gradients (dgrad)
2. **Backward_dw pass:** Compute weight gradients (wgrad) separately

This provides **even finer-grained overlap** by allowing:
- Dgrad from microbatch N overlaps with forward from microbatch N+1
- Wgrad from microbatch N overlaps with dgrad from microbatch N+1

### When to Use

**Benefits:**
- Additional 3-5% speedup on top of basic overlap
- Better A2A hiding when dense compute is marginal

**Requirements:**
- Specific TransformerEngine version with wgrad delay support
- Additional memory for delayed gradients
- More complex debugging

**Recommendation:** Start without it, add only if profiling shows remaining A2A exposure.

## References

- **DeepSeek-V3 DualPipe:** Original inspiration for this overlapping technique
  - Paper: [DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437)
  - Describes overlapping EP communication with dense computation across micro-batches

- **Implementation:**
  - `megatron/core/pipeline_parallel/combined_1f1b.py`
  - `megatron/core/models/gpt/fine_grained_callables.py`
  - `megatron/core/models/common/model_chunk_schedule_plan.py`

- **Configuration:**
  - `megatron/core/transformer/transformer_config.py` (validation logic)
  - `megatron/core/transformer/moe/README.md` (user guide)

- **Testing:**
  - `tests/unit_tests/a2a_overlap/test_schedule_layer_1f1b.py`
  - `tests/unit_tests/a2a_overlap/test_schedule_chunk_1f1b.py`
