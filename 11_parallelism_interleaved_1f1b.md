# 11. Interleaved 1F1B (Virtual Pipeline Parallelism)

## Overview

Interleaved 1F1B is an advanced pipeline parallelism technique in Megatron-LM that significantly reduces pipeline bubbles by splitting each physical pipeline stage into multiple virtual pipeline stages, also called model chunks. Instead of having each GPU hold one contiguous portion of the model, each GPU holds multiple non-contiguous chunks, allowing for better overlap of computation and communication during the warmup and cooldown phases of pipeline execution.

The key innovation is that while regular 1F1B has bubble time proportional to `(P-1)/(2M)` where P is the number of pipeline stages and M is the number of microbatches, interleaved 1F1B reduces this to `(P-1)/(2MV)` where V is the number of virtual pipeline stages per physical stage. This V× improvement in bubble reduction comes at the cost of increased memory usage, as each stage must hold V model chunks instead of one. For large-scale distributed training with deep pipeline parallelism, this technique can provide significant throughput improvements.

## Core Implementation Architecture

The main implementation resides in `megatron/core/pipeline_parallel/schedules.py` in the function `forward_backward_pipelining_with_interleaving` spanning lines 809-1920. This is a complex scheduler that orchestrates the entire interleaved pipeline execution across multiple model chunks.

### Main Entry Point

```python
def forward_backward_pipelining_with_interleaving(
    *,
    forward_step_func,
    data_iterator: Union[Iterator, List[Iterator]],
    model: Union[torch.nn.Module, List[torch.nn.Module]],
    num_microbatches: int,
    seq_length: int,
    micro_batch_size: int,
    decoder_seq_length: Optional[int] = None,
    forward_only: bool = False,
    collect_non_loss_data: bool = False,
    first_val_step: Optional[bool] = None,
    adjust_tensor_shapes_fn: Optional[Callable] = None,
    p2p_communicator: Optional[P2PCommunicator] = None,
    pg_collection: Optional[ProcessGroupCollection] = None,
):
    """Run interleaved 1F1B schedule (model split into model chunks), with
    communication between pipeline stages as needed.

    Returns dictionary with losses if the last stage, empty dict otherwise."""
```

The critical difference from standard pipeline parallelism is that the `model` parameter is a list of model chunks rather than a single model. Each chunk represents a virtual pipeline stage, and the scheduler interleaves execution between these chunks to maximize GPU utilization and minimize idle time.

## Virtual Pipeline Stage Management

### Schedule Table Generation

The heart of interleaved scheduling is the schedule table, which maps each virtual microbatch iteration to a specific (microbatch_id, model_chunk_id) pair. This is computed by the `get_schedule_table` function at lines 753-780:

```python
def get_schedule_table(num_microbatches, num_model_chunks, microbatch_group_size_per_vp_stage):
    """Get the schedule table for PP scheduling."""
    schedule_table = []
    for min_microbatch_id_in_group in range(
        0, num_microbatches, microbatch_group_size_per_vp_stage
    ):
        if min_microbatch_id_in_group + microbatch_group_size_per_vp_stage >= num_microbatches:
            # Construct schedule for the last microbatch group
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(min_microbatch_id_in_group, num_microbatches)
                ]
            )
        else:
            # Construct schedule for other microbatch groups
            schedule_table.extend(
                [
                    (microbatch_id, model_chunk_id)
                    for model_chunk_id in range(num_model_chunks)
                    for microbatch_id in range(
                        min_microbatch_id_in_group,
                        min_microbatch_id_in_group + microbatch_group_size_per_vp_stage,
                    )
                ]
            )
    return schedule_table
```

This schedule table groups microbatches and assigns them to model chunks in a round-robin fashion controlled by `microbatch_group_size_per_vp_stage`. For example, with pipeline parallelism size PP=2, virtual pipeline size VP=2, and M=5 microbatches, the schedule table creates this mapping:

```
virtual_microbatch_id | 0 1 2 3 4 5 6 7 8 9
microbatch_id         | 0 1 2 0 1 2 3 4 3 4
model_chunk_id        | 0 0 0 1 1 1 0 0 1 1
```

The `microbatch_group_size_per_vp_stage` parameter controls how many contiguous microbatches are processed on the same model chunk before switching. This grouping amortizes the cost of context switching between model chunks and can be tuned for different hardware configurations.

### Helper Functions for Schedule Navigation

Several helper functions navigate this schedule table and are defined at lines 1059-1147:

```python
def get_model_chunk_id(virtual_microbatch_id, forward):
    """Helper method to get the model chunk ID given the iteration number."""
    model_chunk_id = model_chunk_id_table[virtual_microbatch_id % total_num_microbatches]
    if not forward:
        model_chunk_id = num_model_chunks - model_chunk_id - 1
    return model_chunk_id

def get_microbatch_id_in_model_chunk(iteration_id, forward):
    """Helper method to get the microbatch_id within model chunk given the iteration number."""
    assert forward
    microbatch_id_in_model_chunk = microbatch_id_table[iteration_id]
    return microbatch_id_in_model_chunk

def is_first_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
    """Check if an iteration is the first for a model chunk."""
    if virtual_microbatch_id < total_num_microbatches:
        return microbatch_id_table[virtual_microbatch_id] == 0
    else:
        return False

def is_last_microbatch_for_model_chunk(virtual_microbatch_id: int) -> bool:
    """Check if an iteration is the last for a model chunk."""
    if virtual_microbatch_id < total_num_microbatches:
        return microbatch_id_table[virtual_microbatch_id] == num_microbatches - 1
    else:
        return False
```

A key aspect is that forward passes process chunks in order (0, 1, 2, ...), while backward passes reverse this order (model_chunk_id = num_model_chunks - model_chunk_id - 1) to maintain the correct dependency chain for gradient computation. The first/last microbatch checks are crucial for managing gradient synchronization and parameter updates, ensuring all-reduce operations occur at the appropriate times.

## Three-Phase Execution Pipeline

The interleaved 1F1B schedule divides execution into three phases: warmup, steady state (1F1B), and cooldown.

### Phase 1: Warmup Phase

The warmup phase at lines 1390-1541 fills the pipeline with forward passes across all virtual stages:

```python
for k in range(num_warmup_microbatches):
    cur_model_chunk_id = get_model_chunk_id(k, forward=True)

    # Determine if tensor should be received from previous stage
    recv_prev, next_forward_model_chunk_id = recv_tensor_from_previous_stage(k, forward=True)

    # Forward step for this virtual microbatch
    output_tensor, _ = forward_backward_helper_wrapper(
        f_virtual_microbatch_id=k,
        checkpoint_activations_microbatch=checkpoint_activations_microbatch,
    )

    # Send to next stage and receive for next iteration
    if not config.overlap_p2p_comm_warmup_flush:
        input_tensor = p2p_communicator.send_forward_recv_forward(
            output_tensor, recv_prev=recv_prev, tensor_shape=tensor_shape
        )

        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
```

During warmup, each pipeline stage performs forward passes on different model chunks, gradually filling the pipeline. The number of warmup microbatches for interleaved schedules is calculated at lines 693-750 as:

```python
if virtual_pipeline_parallel_size is None:
    # forward_backward_pipelining_without_interleaving
    num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
else:
    # forward_backward_pipelining_with_interleaving
    num_warmup_microbatches = (pipeline_parallel_size - pipeline_parallel_rank - 1) * 2
    num_warmup_microbatches += (num_model_chunks - 1) * microbatch_group_size_per_vp_stage
    if overlap_moe_expert_parallel_comm:
        num_warmup_microbatches = num_warmup_microbatches + 1
```

This is significantly larger than the non-interleaved case, which uses `pipeline_parallel_size - pipeline_parallel_rank - 1`, reflecting the additional overhead of managing multiple model chunks that must be warmed up sequentially.

### Phase 2: Steady State (1F1B Phase)

The steady-state phase at lines 1545-1759 performs one forward and one backward pass per iteration, alternating between different model chunks:

```python
for k in range(num_microbatches_remaining):
    forward_k = k + num_warmup_microbatches
    backward_k = k

    if config.overlap_p2p_comm:
        # Overlapped communication version with pre/post hooks
        output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=forward_k,
            b_virtual_microbatch_id=backward_k,
            pre_forward=pp_pre_forward,
            pre_backward=pp_pre_backward,
            post_forward=pp_post_forward,
            post_backward=pp_post_backward,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )
    else:
        # Standard version without overlap
        output_tensor, input_tensor_grad = forward_backward_helper_wrapper(
            f_virtual_microbatch_id=forward_k,
            b_virtual_microbatch_id=backward_k,
            checkpoint_activations_microbatch=checkpoint_activations_microbatch,
        )

        # Communicate tensors
        (input_tensor, output_tensor_grad) = (
            p2p_communicator.send_forward_backward_recv_forward_backward(
                output_tensor,
                input_tensor_grad,
                recv_prev=recv_prev,
                recv_next=recv_next,
                tensor_shape=tensor_shape,
            )
        )

        if recv_prev:
            input_tensors[next_forward_model_chunk_id].append(input_tensor)
        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
```

This is the most compute-efficient phase, where each iteration performs one forward and one backward pass, keeping the GPUs maximally utilized. The interleaving allows the scheduler to switch between different model chunks, ensuring that while one chunk waits for communication with neighboring pipeline stages, another chunk can perform computation. This is the key to filling the pipeline bubbles.

### Phase 3: Cooldown Phase

The cooldown phase at lines 1761-1877 drains remaining backward passes from the pipeline:

```python
for k in range(num_microbatches_remaining, total_num_microbatches):
    cur_model_chunk_id = get_model_chunk_id(k, forward=False)

    # Backward step helper
    _, input_tensor_grad = forward_backward_helper_wrapper(b_virtual_microbatch_id=k)

    # Determine communication requirements
    _, next_backward_model_chunk_id = recv_tensor_from_next_stage(k)

    # Send gradient backward
    if not config.overlap_p2p_comm_warmup_flush:
        output_tensor_grad = p2p_communicator.send_backward_recv_backward(
            input_tensor_grad, recv_next=recv_next, tensor_shape=tensor_shape
        )

        if recv_next:
            output_tensor_grads[next_backward_model_chunk_id].append(output_tensor_grad)
```

This phase completes all remaining backward passes that were initiated during the warmup and steady-state phases. Like warmup, the cooldown phase is longer with interleaving due to the multiple model chunks that must each complete their backward passes.

## Configuration and Parameters

### Command-Line Arguments

The interleaved 1F1B feature is configured through command-line arguments defined in `megatron/training/arguments.py`:

```python
# Virtual pipeline configuration
group.add_argument('--num-layers-per-virtual-pipeline-stage', type=int, default=None,
                   help='Number of layers per virtual pipeline stage')
group.add_argument('--num-virtual-stages-per-pipeline-rank', type=int, default=None,
                   help='Number of virtual pipeline stages per pipeline parallelism rank')
group.add_argument('--microbatch-group-size-per-virtual-pipeline-stage', type=int, default=None,
                   help='Number of contiguous microbatches per virtual pipeline stage',
                   dest='microbatch_group_size_per_vp_stage')
```

Users can specify either `--num-layers-per-virtual-pipeline-stage` (which specifies how many transformer layers go in each virtual stage) or `--num-virtual-stages-per-pipeline-rank` (which directly specifies the number of virtual stages). The framework computes the virtual pipeline model parallel size from these parameters at lines 554-583:

```python
if args.num_layers_per_virtual_pipeline_stage is not None:
    args.virtual_pipeline_model_parallel_size = num_layers_per_pipeline_stage // \
        args.num_layers_per_virtual_pipeline_stage
elif args.num_virtual_stages_per_pipeline_rank is not None:
    args.virtual_pipeline_model_parallel_size = args.num_virtual_stages_per_pipeline_rank
```

For example, if you have 96 layers with PP=8 (12 layers per stage), and you set `--num-layers-per-virtual-pipeline-stage=6`, you get VP=2 (two 6-layer chunks per physical stage).

### Model Parallel Configuration

The `ModelParallelConfig` dataclass in `megatron/core/model_parallel_config.py` contains configuration options for interleaved pipeline:

```python
@dataclass
class ModelParallelConfig:
    virtual_pipeline_model_parallel_size: Optional[int] = None

    overlap_p2p_comm: bool = False
    """If True, overlap pipeline parallel communication with forward/backward compute."""

    overlap_p2p_comm_warmup_flush: bool = False
    """If True, overlap p2p communication in warmup and flush phases."""

    microbatch_group_size_per_vp_stage: Optional[int] = None
    """Number of contiguous microbatches to process per virtual pipeline stage."""

    deallocate_pipeline_outputs: bool = False
    """If True, output data is deallocated after sending to next pipeline stage."""
```

These configuration options allow fine-tuning of the interleaved pipeline behavior. The `overlap_p2p_comm` options enable asynchronous communication to hide latency, while `microbatch_group_size_per_vp_stage` controls the granularity of chunk switching. Setting `deallocate_pipeline_outputs=True` helps reduce memory pressure by freeing activations as soon as they're sent to the next stage.

## Buffer Management and Memory Optimization

### Input/Output Tensor Buffers

One of the complexities of interleaved 1F1B is managing separate tensor buffers for each model chunk. The implementation maintains separate lists for each virtual stage at lines 961-968:

```python
# Input/output tensor buffers - one list per model chunk
input_tensors = [[] for _ in range(len(model))]
output_tensors = [[] for _ in range(len(model))]
if not forward_only:
    output_tensor_grads = [[] for _ in range(len(model))]
```

These buffers store the input and output activations for each microbatch processed by each model chunk. During the forward pass, tensors are appended to these buffers. During the backward pass, they're retrieved to compute gradients.

### P2P Communication Buffers

For pipeline parallel stages that receive tensors from neighbors, the implementation uses circular buffers to handle multiple in-flight receives at lines 1372-1384:

```python
if is_pp_first_stage(p2p_communicator.pp_group):
    fwd_recv_buffer_size = (
        config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
    )
else:
    fwd_recv_buffer_size = 1

fwd_recv_buffer = [None] * fwd_recv_buffer_size
bwd_recv_buffer = [None] * bwd_recv_buffer_size
```

This buffering is necessary because the first pipeline stage may receive multiple activation tensors before it begins processing them, especially when using microbatch grouping. The buffer size calculation ensures sufficient space for all potentially in-flight tensors.

### Input Buffering with Offset Tracking

A subtle but crucial aspect of the implementation is tracking which tensors have been released (popped from input buffers). The `num_released_microbatches` helper at lines 1072-1084 counts how many microbatches for a given model chunk have already been processed and released:

```python
def num_released_microbatches(virtual_microbatch_id, model_chunk_id):
    """Helper method to count number of released (i.e. popped from input_tensors)
    microbatches for a model chunk."""
    if forward_only:  # Micro-batch is released after forward prop.
        return model_chunk_id_table[:virtual_microbatch_id].count(model_chunk_id)
    else:  # Micro-batch is released after backward prop.
        if virtual_microbatch_id < num_warmup_microbatches:
            return 0
        else:
            backward_microbatch_id = virtual_microbatch_id - num_warmup_microbatches
            model_chunk_id = num_model_chunks - model_chunk_id - 1
            return model_chunk_id_table[:backward_microbatch_id].count(model_chunk_id)
```

This offset is used to correctly index into the input tensors list when accessing activations at lines 1175-1183:

```python
# For non-depth-first pipeline schedules, the first rank buffers multiple received
# activation tensors for a model chunk until accessed during warmup
offset = num_released_microbatches(virtual_microbatch_id, model_chunk_id)
input_tensor = input_tensors[model_chunk_id][microbatch_id - offset]
```

This mechanism ensures that even with complex interleaved schedules, where microbatches for the same model chunk may arrive out of order or be processed with delays, the correct activation tensors are retrieved for each forward and backward pass.

## Communication Overlap Optimizations

### Asynchronous P2P Communication

When `overlap_p2p_comm` is enabled, the implementation uses pre/post hooks to overlap communication with computation at lines 1559-1695:

```python
# Sync forward recv
def pp_pre_forward(vp_stage=None):
    if not (_is_vp_first_stage(vp_stage=vp_stage) and is_pp_first_stage(pp_group)):
        recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
        recv_prev_wait_handle.wait()
    deallocate_output_tensor(output_tensor, config.deallocate_pipeline_outputs)

# Async forward send / receive
def pp_post_forward(output_tensor, vp_stage=None):
    fwd_recv_buffer[forward_k % fwd_recv_buffer_size], fwd_wait_handles = (
        p2p_communicator.send_forward_recv_forward(
            output_tensor,
            recv_prev=recv_prev,
            tensor_shape=tensor_shape,
            overlap_p2p_comm=True,
        )
    )
    if recv_prev:
        input_tensors[next_forward_model_chunk_id].append(
            fwd_recv_buffer[forward_k % fwd_recv_buffer_size]
        )
    if fwd_wait_handles is not None:
        recv_prev_wait_handles.extend(fwd_wait_handles)
```

These hooks implement double buffering: the communication for the next iteration is initiated asynchronously after computation completes (`pp_post_forward`), with the wait happening just before the tensor is needed (`pp_pre_forward` of the next iteration). This overlap is critical for hiding communication latency, which can be substantial in deep pipeline configurations.

### Parameter and Gradient Synchronization

For distributed optimizer and data parallel training, parameters and gradients must be synchronized across data parallel ranks. The interleaved schedule carefully manages when these synchronizations occur at lines 1149-1168 and 1234-1271:

```python
# Launch param synchronization for next model chunk asynchronously
if config.param_sync_func is not None:
    param_sync_virtual_microbatch_id = virtual_microbatch_id + pipeline_parallel_rank
    if (
        param_sync_virtual_microbatch_id < total_num_microbatches
        and is_first_microbatch_for_model_chunk(param_sync_virtual_microbatch_id)
    ):
        param_sync_chunk_id = (
            get_model_chunk_id(param_sync_virtual_microbatch_id, forward=True) + 1
        )
        if 1 < param_sync_chunk_id < num_model_chunks:
            config.param_sync_func[param_sync_chunk_id](
                model[param_sync_chunk_id].parameters()
            )

# Gradient synchronization (default)
if config.grad_sync_func is None and is_last_microbatch_for_model_chunk(
    virtual_microbatch_id
):
    enable_grad_sync()
    synchronized_model_chunks.add(model_chunk_id)
```

Parameters are synchronized at the beginning of each model chunk's first microbatch (to ensure updated weights are available), while gradients are synchronized after the last microbatch completes its backward pass (when all gradients for that chunk have been computed). This ensures that all-reduce operations for data parallelism don't block the critical path of pipeline execution.

## Advanced Optimizations

### Combined 1F1B for MoE Communication Overlap

For models with Mixture of Experts (MoE) layers, Megatron-LM provides a specialized interleaved schedule in `megatron/core/pipeline_parallel/combined_1f1b.py`:

```python
def combined_1f1b_schedule_for_interleaved_pipelining(
    config,
    forward_step_func,
    data_iterator,
    model,
    num_microbatches,
    forward_data_store,
    forward_step_helper_preprocess,
    forward_step_helper_postprocess,
    backward_step_helper_preprocess,
    backward_step_helper_postprocess,
    get_microbatch_id_in_model_chunk,
    get_model_chunk_id,
    check_first_val_step,
    is_first_microbatch_for_model_chunk,
    collect_non_loss_data,
    ...
):
    """Helper method to run combined forward and backward step for A2A communication hiding.
    This method merges forward_step_helper and backward_step_helper and eventually calls
    combined_forward_backward_step method."""
```

When `overlap_moe_expert_parallel_comm=True`, this scheduler merges forward and backward operations at the layer level, allowing All-to-All communication for expert parallelism to be overlapped with computation from other layers. This is particularly important for sparse MoE models where expert communication can dominate execution time.

### Fine-Grained Model Chunk Scheduling

The `TransformerModelChunkSchedulePlan` class in `megatron/core/models/common/model_chunk_schedule_plan.py` provides even finer-grained control over layer execution within each model chunk:

```python
class TransformerModelChunkSchedulePlan(AbstractSchedulePlan):
    """Schedule the executing plan of the sub-modules in a model chunk.

    This class organizes computation nodes for a model chunk,
    including preprocessing, transformer layers, and postprocessing."""

    @staticmethod
    def run(
        f_schedule_plan,
        b_schedule_plan,
        b_grad=None,
        pre_forward=None,
        pre_backward=None,
        post_forward=None,
        post_backward=None,
    ):
        """Model Chunk level 1f1b fine-grained scheduler.

        Interleaves forward and backward functions of multiple Transformer layers
        within a model chunk to overlap submodules between individual passes."""
```

This layer-level scheduling overlaps MoE dispatch/combine operations (which use the communication stream) with attention and MLP computation (which use the compute stream), further improving GPU utilization by ensuring that communication and computation can proceed simultaneously on different CUDA streams.

## Performance Impact and Trade-offs

### Bubble Time Reduction

The primary benefit of interleaved 1F1B is reduced pipeline bubble time. For standard 1F1B, the bubble fraction is:

**Bubble = (P-1) / (2M)**

For interleaved 1F1B with V virtual stages per physical stage:

**Bubble = (P-1) / (2MV)**

For example, with P=8 pipeline stages, M=32 microbatches:
- Standard 1F1B: (8-1)/(2×32) = 10.9% bubble
- Interleaved with V=2: (8-1)/(2×32×2) = 5.5% bubble
- Interleaved with V=4: (8-1)/(2×32×4) = 2.7% bubble

This represents a direct 2× or 4× reduction in wasted computation time, translating to proportional throughput improvements when the system is compute-bound.

### Memory Requirements

The trade-off is increased memory usage. Each pipeline stage must hold V model chunks instead of one:

**Memory per stage = (total_model_size / P) × V**

For a 175B parameter model with PP=8 and V=2:
- Standard: 175B / 8 = 21.9B parameters per stage
- Interleaved: (175B / 8) × 2 = 43.8B parameters per stage

This doubling of model parameters per stage also increases activation memory proportionally, as each model chunk maintains its own activation buffers during execution. The increased memory pressure can be mitigated with activation checkpointing and output deallocation strategies.

### Microbatch Group Size Tuning

The `microbatch_group_size_per_vp_stage` parameter allows trading off between context switch overhead and bubble reduction. Larger values reduce the number of model chunk switches (lowering overhead from cache effects and kernel launch latency) but may increase bubbles by reducing scheduling flexibility. Smaller values provide finer-grained interleaving and better bubble filling but incur more switching cost. Typical values range from 1-4 depending on the model size and hardware.

### When to Use Interleaved 1F1B

Interleaved 1F1B is most beneficial when:
1. Pipeline parallelism degree (P) is large (typically P ≥ 4) relative to the number of microbatches (M)
2. GPU memory allows holding multiple model chunks per stage (HBM capacity is sufficient)
3. The model architecture supports clean splitting into chunks (e.g., Transformer layers are divisible)
4. Communication bandwidth is high enough that increased communication frequency doesn't bottleneck
5. The system is compute-bound rather than communication-bound

In these scenarios, the bubble reduction directly translates to higher throughput, often yielding 5-10% improvements for typical large language model training configurations.

## Comparison with Standard 1F1B

The non-interleaved 1F1B scheduler (`forward_backward_pipelining_without_interleaving` at lines 1949-2304) provides a simpler alternative:

- Single model chunk per pipeline stage
- Simpler scheduling logic: warmup → 1F1B → cooldown
- Lower memory usage (no chunk replication)
- Higher bubble time: `(P-1)/(2M)` vs `(P-1)/(2MV)`
- Simpler buffer management (no per-chunk tracking needed)
- No chunk switching overhead

The warmup calculation for standard 1F1B is much simpler:

```python
num_warmup_microbatches = pipeline_parallel_size - pipeline_parallel_rank - 1
```

This reflects the fact that each stage only needs to complete one forward pass per microbatch to fill the pipeline, rather than V forward passes as in the interleaved case. For smaller pipeline depths (P ≤ 2) or when memory is constrained, standard 1F1B may be preferable.

## Configuration Example

```bash
# Training script configuration for interleaved 1F1B
python pretrain_gpt.py \
    --num-layers 96 \
    --hidden-size 12288 \
    --num-attention-heads 96 \
    --micro-batch-size 1 \
    --global-batch-size 1536 \
    --seq-length 2048 \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 8 \
    --num-layers-per-virtual-pipeline-stage 6 \
    --microbatch-group-size-per-virtual-pipeline-stage 2 \
    --overlap-p2p-comm \
    --deallocate-pipeline-outputs \
    --recompute-granularity selective \
    --recompute-method uniform \
    --recompute-num-layers 1
```

In this configuration:
- 96 layers with 6 layers per virtual stage = 16 total virtual stages
- With PP=8, each physical stage holds 2 model chunks (VP=2)
- Microbatch group size of 2 provides balanced switching overhead
- P2P communication overlap enabled for latency hiding
- Activation deallocation reduces memory footprint

## Summary

Megatron-LM's interleaved 1F1B pipeline parallelism represents a sophisticated optimization of the standard 1F1B schedule. By splitting each physical pipeline stage into multiple virtual stages (model chunks), it achieves V× reduction in pipeline bubble time at the cost of increased memory usage and implementation complexity. The framework provides extensive configuration options to tune the interleaving behavior, including microbatch grouping, communication overlap, and fine-grained layer-level scheduling. For large-scale distributed training with deep pipeline parallelism, interleaved 1F1B can provide significant throughput improvements (5-10% or more), making it a critical technique for training massive language models efficiently. The implementation carefully manages tensor buffers, communication patterns, and synchronization points to ensure correct execution while maximizing GPU utilization across all pipeline stages and model chunks.
