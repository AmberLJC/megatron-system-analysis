# Pipeline Parallelism Optimizations in Megatron-LM

## Overview

Pipeline parallelism (PP) partitions model layers across multiple GPUs, enabling training of models too large for single-device memory. Megatron-LM implements several optimizations to minimize communication overhead, reduce memory footprint, and improve load balancing across pipeline stages.

---

## 1. Pipeline Output Deallocation

**Purpose:** Reduce GPU memory consumption by immediately freeing activation tensors after P2P transmission.

**Mechanism:** After sending activations to the next pipeline stage, the tensor's `.data` field is replaced with a scalar tensor `torch.empty((1,))`, preserving only the `.grad_fn` needed for backward computation. This "pseudo-deallocation" enables the autograd graph to remain intact while releasing activation memory.

**Key Insight:** Standard PyTorch backward requires matching tensor shapes. Megatron bypasses this by directly invoking the C++ autograd engine (`Variable._execution_engine.run_backward`), which doesn't enforce shape consistency.

| Config Parameter | Default | File |
|-----------------|---------|------|
| `deallocate_pipeline_outputs` | `False` | `model_parallel_config.py:281-284` |

### Core Implementation

**Pseudo-Deallocation Function** (`schedules.py:135-146`):
```python
def deallocate_output_tensor(out, deallocate_pipeline_outputs=False):
    '''Pseudo-deallocate (i.e., set to scalar) the output tensor's '.data' field.

    This method should be called right after the output tensor has been
    sent to the next pipeline stage. At this point, the output tensor is
    only useful for its '.grad_fn' field, and not its '.data'.
    '''
    if (out is None) or (not deallocate_pipeline_outputs):
        return
    assert isinstance(out, torch.Tensor), "expected Tensor, found %s." % type(out).__name__
    assert out._base is None, "counter-productive to free a view of another tensor."
    out.data = torch.empty((1,), device=out.device, dtype=out.dtype)
```

**Custom Backward for Deallocated Tensors** (`schedules.py:149-178`):
```python
def custom_backward(output, grad_output):
    '''Directly call C++ autograd engine.

    To make the 'deallocate_output_tensor' (above) optimization work, the C++
    autograd engine must be called directly, bypassing Pytorch's
    torch.autograd.backward. Pytorch's 'backward' checks that the output and
    grad have the same shape, while C++'s 'backward' does not.
    '''
    assert output.numel() == 1, "output should be pseudo-'freed' in schedule"

    if grad_output is None:
        grad_output = torch.ones_like(output, memory_format=torch.preserve_format)

    # Call C++ engine directly [see torch/csrc/autograd/python_engine.cpp]
    Variable._execution_engine.run_backward(
        tensors=(output,),
        grad_tensors=(grad_output,),
        keep_graph=False,
        create_graph=False,
        inputs=tuple(),
        allow_unreachable=True,
        accumulate_grad=True,
    )
```

**Usage in Pipeline Schedule** (`schedules.py:2213`):
```python
# After forward pass completes and tensor is sent
output_tensor, num_tokens = forward_step(...)
input_tensors.append(input_tensor)
output_tensors.append(output_tensor)
deallocate_output_tensor(output_tensor[0], config.deallocate_pipeline_outputs)
```

**Backward Pass Dispatch** (`schedules.py:465-468`):
```python
if config.deallocate_pipeline_outputs:
    custom_backward(output_tensor[0], output_tensor_grad[0])
else:
    torch.autograd.backward(output_tensor[0], grad_tensors=output_tensor_grad[0])
```

---

## 2. Ring Exchange P2P Communication

**Purpose:** Optimize P2P communication for specialized network topologies using a custom ring-based kernel.

**Mechanism:** Replaces `torch.distributed.batch_isend_irecv()` with `torch.distributed.ring_exchange()`, a custom kernel that performs bidirectional send/recv in a ring pattern. This is synchronous (no request handles) and requires a custom PyTorch build.

**Use Case:** Clusters with ring-topology interconnects where standard P2P patterns are suboptimal.

| Config Parameter | Default | File |
|-----------------|---------|------|
| `use_ring_exchange_p2p` | `False` | `model_parallel_config.py:276-279` |

### Core Implementation

**P2P Mode Dispatcher** (`p2p_communication.py:346-357`):
```python
# Send tensors in both the forward and backward directions as appropriate.
if config.use_ring_exchange_p2p:

    def _ring_exchange_wrapper(**kwargs):
        torch.distributed.ring_exchange(**kwargs)
        return []  # Synchronous - no request handles

    p2p_func = _ring_exchange_wrapper
elif config.batch_p2p_comm:
    assert wait_on_reqs
    p2p_func = _batched_p2p_ops
else:
    p2p_func = _p2p_ops
```

**Ring Exchange for Shape Communication** (`p2p_communication.py:205-212`):
```python
if config.use_ring_exchange_p2p:
    torch.distributed.ring_exchange(
        tensor_send_prev=send_prev_shape_tensor,
        tensor_recv_prev=recv_prev_shape_tensor,
        tensor_send_next=send_next_shape_tensor,
        tensor_recv_next=recv_next_shape_tensor,
        group=self.pp_group,
    )
```

**Request Handle Management** (`p2p_communication.py:363-366`):
```python
# Ring exchange is synchronous, returns empty list
if config.use_ring_exchange_p2p or config.batch_p2p_comm:
    reqs = []
else:
    reqs = {}  # Overlapped P2P returns dict of handles
```

---

## 3. P2P Warmup/Flush Overlap

**Purpose:** Hide communication latency during pipeline warmup and cooldown phases by overlapping with computation.

**Mechanism:** Prefetches recv operations for iteration k+1 before completing iteration k's forward pass. Uses circular buffers sized by `microbatch_group_size_per_vp_stage` to pipeline communications without conflicts.

**Pattern:**
```
Iteration k: Prefetch recv(k+1) → Wait recv(k) → Compute → Send(k)
Iteration k+1: recv(k+1) already in flight → Wait → Compute → ...
```

| Config Parameter | Default | File |
|-----------------|---------|------|
| `overlap_p2p_comm_warmup_flush` | `False` | `model_parallel_config.py:298-302` |
| `microbatch_group_size_per_vp_stage` | `pipeline_parallel_size` | `model_parallel_config.py:304-317` |

**Requirements:** `overlap_p2p_comm=True` and `batch_p2p_comm=False`

### Core Implementation

**Buffer Initialization** (`schedules.py:1371-1388`):
```python
# Forward direction receive buffers
if is_pp_first_stage(p2p_communicator.pp_group):
    fwd_recv_buffer_size = (
        config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
    )
else:
    fwd_recv_buffer_size = 1

# Backward direction receive buffers
if is_pp_last_stage(p2p_communicator.pp_group):
    bwd_recv_buffer_size = (
        config.microbatch_group_size_per_vp_stage - pipeline_parallel_size + 1
    )
else:
    bwd_recv_buffer_size = 1

fwd_recv_buffer = [None] * fwd_recv_buffer_size
bwd_recv_buffer = [None] * bwd_recv_buffer_size
recv_prev_wait_handles = []      # Queue of forward recv handles
send_next_wait_handle = None     # Latest forward send handle
send_prev_wait_handle = None     # Latest backward send handle
recv_next_wait_handles = []      # Queue of backward recv handles
```

**Warmup Phase Prefetch** (`schedules.py:1393-1428`):
```python
for k in range(num_warmup_microbatches):
    cur_model_chunk_id = get_model_chunk_id(k, forward=True)

    # Wait for previous recv before starting compute
    if config.overlap_p2p_comm_warmup_flush:
        if (
            not (_is_vp_first_stage(vp_stage=cur_model_chunk_id)
                 and is_pp_first_stage(pp_group))
            and k != 0
        ):
            recv_prev_wait_handle = recv_prev_wait_handles.pop(0)
            recv_prev_wait_handle.wait()

    # Prefetch recv for iteration k+1 (non-first ranks)
    if config.overlap_p2p_comm_warmup_flush and not is_pp_first_stage(pp_group):
        fwd_recv_buffer[k % fwd_recv_buffer_size], fwd_wait_recv_handles = (
            p2p_communicator.send_forward_recv_forward(
                output_tensor=None,  # No output to send yet
                recv_prev=recv_prev,
                tensor_shape=tensor_shape,
                overlap_p2p_comm=True,  # Asynchronous!
            )
        )
        if fwd_wait_recv_handles:
            recv_prev_wait_handles.append(fwd_wait_recv_handles.pop("recv_prev"))
```

**Overlapped Send/Recv in P2P Communicator** (`p2p_communication.py:572-595`):
```python
def send_forward_recv_forward(
    self,
    output_tensor: torch.Tensor,
    recv_prev: bool,
    tensor_shape: Shape,
    overlap_p2p_comm: bool = False,
) -> torch.Tensor:
    """Batched recv from previous rank and send to next rank in pipeline."""
    input_tensor, _, wait_handles = self._communicate(
        tensor_send_next=output_tensor,
        tensor_send_prev=None,
        recv_prev=recv_prev,
        recv_next=False,
        tensor_shape=tensor_shape,
        wait_on_reqs=(not overlap_p2p_comm),  # Return handles when overlap=True
    )
    if overlap_p2p_comm:
        return input_tensor, wait_handles  # Dict: {'recv_prev': handle, 'send_next': handle}
    return input_tensor
```

---

## 4. Variable Sequence Lengths

**Purpose:** Support microbatches with different sequence lengths within a global batch.

**Mechanism:** Before tensor transmission, communicates tensor shapes (3-element int64 tensors) between pipeline stages. Receiving stages allocate appropriately-sized buffers based on communicated shapes rather than assuming fixed dimensions.

**Trade-off:** Adds communication overhead; only enable when sequence lengths genuinely vary.

| Config Parameter | Default | File |
|-----------------|---------|------|
| `variable_seq_lengths` | `False` | `model_parallel_config.py:255-259` |

### Core Implementation

**Shape Communication Method** (`p2p_communication.py:165-252`):
```python
def _communicate_shapes(self, tensor_send_next, tensor_send_prev, recv_prev, recv_next):
    """Communicate tensor shapes between stages. Used to communicate
    tensor shapes before the actual tensor communication happens.
    This is required when the sequence lengths across micro batches
    are not uniform.
    """
    # Create shape tensors (3 elements: seq_len, batch_size, hidden_size)
    recv_prev_shape_tensor = torch.empty(
        (3,), device=torch.cuda.current_device(), dtype=torch.int64
    ) if recv_prev else None

    recv_next_shape_tensor = torch.empty(
        (3,), device=torch.cuda.current_device(), dtype=torch.int64
    ) if recv_next else None

    if tensor_send_prev is not None:
        send_prev_shape_tensor = torch.tensor(
            tensor_send_prev.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )
    if tensor_send_next is not None:
        send_next_shape_tensor = torch.tensor(
            tensor_send_next.size(), device=torch.cuda.current_device(), dtype=torch.int64
        )

    # Exchange shapes via P2P operations
    ops = []
    if send_prev_shape_tensor is not None:
        ops.append(torch.distributed.P2POp(torch.distributed.isend,
                                           send_prev_shape_tensor, self.prev_rank))
    if recv_prev_shape_tensor is not None:
        ops.append(torch.distributed.P2POp(torch.distributed.irecv,
                                           recv_prev_shape_tensor, self.prev_rank))
    # ... similar for next rank

    if len(ops) > 0:
        reqs = torch.distributed.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()

    return recv_prev_shape.tolist(), recv_next_shape.tolist()
```

**Conditional Shape Communication in `_communicate()`** (`p2p_communication.py:301-323`):
```python
# Determine recv shapes - either fixed or communicated
if not config.variable_seq_lengths:
    recv_prev_shape = tensor_shape  # Use provided fixed shape
    recv_next_shape = tensor_shape
else:
    # Communicate actual shapes before tensor transfer
    recv_prev_shape, recv_next_shape = self._communicate_shapes(
        tensor_send_next, tensor_send_prev, recv_prev, recv_next
    )

# Create receive buffers with correct (possibly dynamic) shapes
def create_tensor_recv_prev():
    return torch.empty(
        recv_prev_shape,
        requires_grad=True,
        device=torch.cuda.current_device(),
        dtype=config.pipeline_dtype,
    )

def create_tensor_recv_next():
    return torch.empty(
        recv_next_shape,
        requires_grad=True,
        device=torch.cuda.current_device(),
        dtype=config.pipeline_dtype,
    )
```

---

## 5. Custom PP Layer Layouts

**Purpose:** Enable heterogeneous layer distribution across pipeline stages for load balancing.

**Mechanism:** Supports flexible layer assignment via string or list format specifications. Layers can be distributed non-uniformly to account for embedding/loss overhead or varying layer complexity.

**Format Examples:**
- String: `"Et*3|(tt|)*29,m|L"` (DeepSeek-V3 style)
- List: `[['embedding', 'decoder'], ['decoder', 'decoder', 'decoder', 'loss']]`

**Layer Types:** `E`=embedding, `t`=decoder, `L`=loss, `m`=MTP

| Config Parameter | Default | File |
|-----------------|---------|------|
| `pipeline_model_parallel_layout` | `None` | `transformer_config.py:62-86` |
| `num_layers_in_first_pipeline_stage` | `None` | `transformer_config.py:54-56` |
| `num_layers_in_last_pipeline_stage` | `None` | `transformer_config.py:58-60` |

### Core Implementation

**PipelineParallelLayerLayout Class** (`pipeline_parallel_layer_layout.py:15-83`):
```python
class PipelineParallelLayerLayout:
    """Configuration of custom pipeline parallel layer partitioning."""

    def __init__(self, layout: str | list, pipeline_model_parallel_size: int):
        """Initialize from a list or a string format."""
        self.input_data = layout
        if isinstance(layout, str):
            layout = PipelineParallelLayerLayout.parse_str_to_list(layout)

        # Convert 1D flattened layout to 2D [pp_rank][vpp_rank]
        virtual_pipeline_model_parallel_size = len(layout) // pipeline_model_parallel_size
        layout = [
            [
                layout[vpp_rank * pipeline_model_parallel_size + pp_rank]
                for vpp_rank in range(virtual_pipeline_model_parallel_size)
            ]
            for pp_rank in range(pipeline_model_parallel_size)
        ]

        # Convert string layer types to LayerType enum
        for pp_rank in range(pipeline_model_parallel_size):
            for vpp_rank in range(virtual_pipeline_model_parallel_size):
                transferred_layout = []
                for layer_type in layout[pp_rank][vpp_rank]:
                    if isinstance(layer_type, str):
                        layer_type = LayerType[layer_type.strip().lower()]
                    transferred_layout.append(layer_type)
                layout[pp_rank][vpp_rank] = transferred_layout

        self.layout = layout  # 2D array: layout[pp_rank][vpp_rank] = [LayerType, ...]
```

**String Parser with Multiplication Support** (`pipeline_parallel_layer_layout.py:270-308`):
```python
@staticmethod
def parse_str_to_list(layout_str: str):
    """Parse a layout string to a list of lists.
    Example: "Ettt|(tt|)*29,m|L" -> [["E","t","t","t"]] + [["t","t"]]*29 + [["m"],["L"]]
    """
    layout_str = layout_str.replace(",", "")  # Remove cosmetic commas

    # Unroll multiplications: (ab|cd)*2 -> ab|cdab|cd, t*3 -> ttt
    patterns = [
        r'\(([^)]+)\)\*(\d+)',  # (group)*n
        r'(.)\*(\d+)',          # char*n
    ]
    for pattern in patterns:
        layout_str = re.sub(pattern, lambda x: x.group(1) * int(x.group(2)), layout_str)

    char2layer_type = {
        "E": LayerType.embedding,
        "L": LayerType.loss,
        "t": LayerType.decoder,  # transformer
        "m": LayerType.mtp,
    }

    # Split by '|' for pipeline stages
    layout_list = []
    for stage in layout_str.split('|'):
        layout_list.append([char2layer_type[c] for c in stage])
    return layout_list
```

**Layer Count and Offset Methods** (`pipeline_parallel_layer_layout.py:139-192`):
```python
def get_num_layers_to_build(
    self,
    layer_type: LayerType = LayerType.decoder,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    """Get the number of layers to build in the pipeline stage."""
    if pp_rank is None:
        pp_rank = parallel_state.get_pipeline_model_parallel_rank()
    if vp_stage is None:
        vp_stage = 0
    # Count layer occurrences in this stage
    return self.layout[pp_rank][vp_stage].count(layer_type)

def get_layer_offset(
    self,
    layer_type: LayerType = LayerType.decoder,
    vp_stage: Optional[int] = None,
    pp_rank: Optional[int] = None,
):
    """Get the layer offset (global layer index) for this pipeline stage."""
    # Sum layers in all previous stages
    offset = 0
    for _vpp_rank in range(vp_stage + 1):
        for _pp_rank in range(
            self.pipeline_model_parallel_size if _vpp_rank < vp_stage else pp_rank
        ):
            offset += self.layout[_pp_rank][_vpp_rank].count(layer_type)
    return offset

def get_layer_id_list(self, layer_type, vp_stage, pp_rank):
    """Get list of global layer IDs for this stage."""
    offset = self.get_layer_offset(layer_type, vp_stage, pp_rank)
    num_layers = self.get_num_layers_to_build(layer_type, vp_stage, pp_rank)
    return list(range(offset, offset + num_layers))
```

**Usage in Transformer Block** (`transformer_block.py:75-90`):
```python
def get_num_layers_to_build(config: TransformerConfig, vp_stage=None, pp_rank=None) -> int:
    """Determine number of transformer layers for current pipeline stage."""
    # If custom layout is provided, use it directly
    if config.pipeline_model_parallel_layout is not None:
        return config.pipeline_model_parallel_layout.get_num_layers_to_build(
            layer_type=LayerType.decoder, vp_stage=vp_stage
        )
    # Otherwise fall back to uniform or first/last stage overrides
    # ...
```

### Example: DeepSeek-V3 Layout (PP=16, VPP=2)

```bash
--pipeline-model-parallel-layout "Et*3|(tt|)*29,m|L"
```

| PP Rank | VPP 0 | VPP 1 |
|---------|-------|-------|
| 0 | embedding + 3×decoder | 2×decoder |
| 1-13 | 2×decoder | 2×decoder |
| 14 | 2×decoder | mtp |
| 15 | 2×decoder | loss |

---

## Configuration Summary

| Optimization | Config Flag | Memory | Compute | Communication |
|-------------|-------------|--------|---------|---------------|
| Output Deallocation | `deallocate_pipeline_outputs` | ✓ Reduced | - | - |
| Ring Exchange P2P | `use_ring_exchange_p2p` | - | - | ✓ Topology-optimized |
| Warmup/Flush Overlap | `overlap_p2p_comm_warmup_flush` | - | ✓ Hidden latency | ✓ Overlapped |
| Variable Seq Lengths | `variable_seq_lengths` | ✓ Dynamic alloc | - | + Shape overhead |
| Custom Layouts | `pipeline_model_parallel_layout` | - | ✓ Load balanced | - |

---

## Implementation Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Pipeline Parallel Training                        │
├─────────────────────────────────────────────────────────────────────┤
│  schedules.py                                                        │
│  ├── forward_backward_pipelining_with_interleaving()                │
│  ├── forward_backward_pipelining_without_interleaving()             │
│  ├── deallocate_output_tensor()         [Output Deallocation]       │
│  └── custom_backward()                  [C++ Autograd Bypass]       │
├─────────────────────────────────────────────────────────────────────┤
│  p2p_communication.py                                                │
│  ├── P2PCommunicator._communicate()     [Core P2P Dispatcher]       │
│  ├── P2PCommunicator._communicate_shapes() [Variable Seq Support]   │
│  └── ring_exchange_wrapper()            [Ring Exchange P2P]         │
├─────────────────────────────────────────────────────────────────────┤
│  pipeline_parallel_layer_layout.py                                   │
│  ├── PipelineParallelLayerLayout        [Custom Layout Class]       │
│  ├── get_num_layers_to_build()          [Layer Count per Stage]     │
│  └── get_layer_offset()                 [Global Layer Indexing]     │
├─────────────────────────────────────────────────────────────────────┤
│  model_parallel_config.py                                            │
│  └── All PP optimization flags                                       │
└─────────────────────────────────────────────────────────────────────┘
```
