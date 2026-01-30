# FP8 Training Infrastructure

## Overview

FP8 (8-bit floating point) training is one of the most significant performance optimizations in Megatron-LM, providing 1.5-2x training speedup on NVIDIA H100 GPUs and newer architectures. This optimization reduces memory bandwidth requirements and leverages specialized tensor cores designed for FP8 computation. The framework provides a comprehensive FP8 infrastructure that supports multiple quantization recipes, dynamic scaling, parameter quantization, and sophisticated context management to maintain model accuracy while achieving substantial performance gains.

The FP8 implementation is tightly integrated with NVIDIA's Transformer Engine library, which provides hardware-optimized kernels for FP8 operations. Megatron-LM extends this with distributed training support, including FP8-aware gradient communication, parameter gathering, and checkpoint management.

## Technical Background

### FP8 Number Formats

FP8 training uses two complementary 8-bit floating-point formats:

**E4M3 Format (4 exponent bits, 3 mantissa bits)**
- Range: 2^-6 to 448 (approximately)
- Better precision, narrower range
- Typically used for forward pass activations and weights
- Suitable for values with predictable magnitude

**E5M2 Format (5 exponent bits, 2 mantissa bits)**
- Range: 2^-14 to 57344 (approximately)
- Wider range, less precision
- Typically used for backward pass gradients
- Better for values with unpredictable magnitude

### Why FP8 Training Accelerates Training

1. **Memory Bandwidth Reduction**: Moving data from HBM to compute units is often the bottleneck. FP8 halves bandwidth vs BF16.
2. **Tensor Core Acceleration**: H100 tensor cores provide 2x throughput for FP8 vs BF16 matrix multiplication.
3. **Reduced Communication**: Gradient communication in distributed training is faster with smaller tensors.

## Core Implementation

### 1. FP8 Recipe Management

Megatron-LM supports four FP8 quantization recipes, each with different trade-offs between accuracy, performance, and hardware requirements:

```python
# From megatron/core/enums.py:22-28
class Fp8Recipe(str, enum.Enum):
    """FP8 recipe names: delayed, tensorwise, mxfp8, blockwise."""

    delayed = "delayed"
    tensorwise = "tensorwise"
    mxfp8 = "mxfp8"
    blockwise = "blockwise"
```

The recipe selection logic in `megatron/core/fp8_utils.py:459-514`:

```python
def get_fp8_recipe(config: TransformerConfig):
    """Return fp8 recipe based on configuration.

    Arguments:
        config (TransformerConfig): Configuration object.

    Returns:
        FP8 recipe appropriate for the specified format.
    """
    if config.fp8 == "e4m3":
        fp8_format = transformer_engine.common.recipe.Format.E4M3
    elif config.fp8 == "hybrid":
        fp8_format = transformer_engine.common.recipe.Format.HYBRID
    else:
        raise ValueError("E4M3 and HYBRID are the only supported FP8 formats.")

    # Select fp8 recipe (TE version >= 2.1.0).
    fp8_recipe = None
    if is_te_min_version("2.1.0"):
        if config.fp8_recipe == Fp8Recipe.delayed:
            fp8_recipe = TEDelayedScaling(
                config=config,
                fp8_format=fp8_format,
                override_linear_precision=(False, False, not config.fp8_wgrad),
            )
        elif config.fp8_recipe == Fp8Recipe.tensorwise and is_te_min_version("2.2.0.dev0"):
            fp8_recipe = transformer_engine.common.recipe.Float8CurrentScaling(
                fp8_format=fp8_format, fp8_dpa=config.fp8_dot_product_attention
            )
        elif config.fp8_recipe == Fp8Recipe.blockwise and is_te_min_version("2.3.0.dev0"):
            fp8_recipe = transformer_engine.common.recipe.Float8BlockScaling(
                fp8_format=fp8_format
            )
        elif config.fp8_recipe == Fp8Recipe.mxfp8:
            fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
                fp8_format=fp8_format
            )
        else:
            raise ValueError(
                "Float8CurrentScaling, MXFP8BlockScaling, Float8BlockwiseScaling and "
                "DelayedScaling are the only supported FP8 recipes. Please also make sure "
                "you are using a compatible TE version."
            )
    return fp8_recipe
```

**Recipe Comparison:**

| Recipe | Scaling Strategy | Accuracy | Performance | TE Version Required |
|--------|------------------|----------|-------------|---------------------|
| `delayed` | Per-tensor, updated periodically | Good | Fast | >= 2.1.0 |
| `tensorwise` | Per-tensor, updated every iteration | Better | Slightly slower | >= 2.2.0 |
| `blockwise` | Per-block (32 elements) | Best | Fast | >= 2.3.0 |
| `mxfp8` | Microscaling per-block | Best | Fast | >= 2.1.0 |

### 2. Delayed Scaling Recipe

The delayed scaling recipe wraps Transformer Engine's DelayedScaling with Megatron-specific configuration:

```python
# From megatron/core/extensions/transformer_engine.py:1791-1824
class TEDelayedScaling(te.common.recipe.DelayedScaling):
    """
    Wrapper for the Transformer-Engine's `DelayedScaling` layer.
    """

    def __init__(
        self,
        config: ModelParallelConfig,
        fp8_format: int,
        override_linear_precision: tuple = (False, False, False),
    ):
        if not HAVE_TE:
            raise ImportError(
                "Transformer Engine is not installed. "
                "Please install it with `pip install transformer-engine`."
            )

        extra_kwargs = _get_extra_te_kwargs(config)
        if is_te_min_version("1.6.0.dev0"):
            extra_kwargs["fp8_dpa"] = config.fp8_dot_product_attention
            extra_kwargs["fp8_mha"] = config.fp8_multi_head_attention
        if get_te_version() < PkgVersion("1.8.0"):
            extra_kwargs["interval"] = config.fp8_interval
        elif config.fp8_interval != 1:
            warnings.warn("fp8_interval is deprecated and ignored from Transformer-Engine v1.8.0.")

        super().__init__(
            margin=config.fp8_margin,
            fp8_format=fp8_format,
            amax_compute_algo=config.fp8_amax_compute_algo,
            amax_history_len=config.fp8_amax_history_len,
            override_linear_precision=override_linear_precision,
            **extra_kwargs,
        )
```

Key configuration parameters from `megatron/core/transformer/transformer_config.py:362-373`:

```python
fp8_margin: int = 0
"""Margin for the scaling factor computation."""

fp8_interval: int = 1
"""DEPRECATED from TransformerEngine v1.8.0. This flag is ignored.
Controls how often the scaling factor is recomputed.
"""

fp8_amax_history_len: int = 1
"""The length of the amax history window used for scaling factor computation."""

fp8_amax_compute_algo: str = "most_recent"
"""Algorithm used for choosing the `amax` value for the scaling factor computation. There are 2
predefined choices: `max` chooses the largest `amax` in the history window, while `most_recent`
always chooses the most recently seen value.
"""
```

### 3. FP8 Context Management

The framework provides sophisticated context management to control when FP8 is applied:

```python
# From megatron/core/fp8_utils.py:516-574
def get_fp8_context(config: TransformerConfig, layer_no: int = -1, is_init: bool = False):
    """Return fp8 context manager.

    Arguments:
        config (TransformerConfig): Configuration object.
        layer_no (int): *Global* layer index (including layers on other
            pipeline-parallel ranks).
        is_init (bool): Whether the context is fp8_model_init (True) or fp8_autocast (False).

    Returns:
        FP8 context.
        If layer_no < 0, we return a fp8 context for all layers regardless of layer_no.
        We return nullcontext() when: a) not using fp8 to train, b) layer_no is a layer
        that needs to be trained in bf16.
    """

    need_fp8_context = config.fp8 if not is_init else config.fp8_param

    if not need_fp8_context or is_first_last_bf16_layer(config, layer_no):
        # bf16 training or bf16 layer in fp8 training
        fp8_context = nullcontext()
    else:
        # fp8 training and this layer_no is in fp8
        fp8_recipe = get_fp8_recipe(config)

        fp8_group = None
        if parallel_state.model_parallel_is_initialized():
            fp8_group = parallel_state.get_amax_reduction_group(
                with_context_parallel=True, tp_only_amax_red=config.tp_only_amax_red
            )

        if not is_init:
            fp8_context = transformer_engine.pytorch.fp8_autocast(
                enabled=True, fp8_recipe=fp8_recipe, fp8_group=fp8_group
            )
        else:
            import inspect

            context_args = {"enabled": True}
            # Check if fp8_model_init supports setting recipe
            if "recipe" in (
                inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
            ):
                context_args["recipe"] = fp8_recipe
            # Check if fp8_model_init supports preserve_high_precision_init_val
            if "preserve_high_precision_init_val" in (
                inspect.signature(transformer_engine.pytorch.fp8_model_init).parameters
            ):
                context_args["preserve_high_precision_init_val"] = torch.is_grad_enabled()
            fp8_context = transformer_engine.pytorch.fp8_model_init(**context_args)

        # First / last layer in bf16 isn't supported with delayed scaling since it
        # requires entering/exiting fp8 context per layer, causing incorrect amax
        # reduction behavior.
        assert not (
            config.first_last_layers_bf16 and isinstance(fp8_recipe, TEDelayedScaling)
        ), "Delayed scaling does not support first / last layer in BF16."

    return fp8_context
```

### 4. First/Last Layer BF16 Override

Critical for model stability, the first and last layers can remain in BF16:

```python
# From megatron/core/fp8_utils.py:436-452
def is_first_last_bf16_layer(config: TransformerConfig, layer_no: int):
    """Check if the layer is in bf16."""
    num_bf16_layers_at_start = (
        config.num_layers_at_start_in_bf16 if config.first_last_layers_bf16 else 0
    )
    num_bf16_layers_at_end = (
        config.num_layers_at_end_in_bf16 if config.first_last_layers_bf16 else 0
    )
    # Since layer_no is a global layer index, additional checks on whether
    # we are in the first or last pipeline-parallel rank are not needed.
    is_first_layer = layer_no < num_bf16_layers_at_start
    is_last_layer = layer_no >= config.num_layers - num_bf16_layers_at_end

    if layer_no >= 0 and config.first_last_layers_bf16 and (is_first_layer or is_last_layer):
        return True
    else:
        return False
```

Configuration options from `megatron/core/transformer/transformer_config.py:393-402`:

```python
first_last_layers_bf16: bool = False
"""If True, retains first and last N TransformerBlocks in BF16 as opposed to FP8."""

num_layers_at_start_in_bf16: int = 1
"""Number of layers at the start of the model to keep in BF16 precision when
first_last_layers_bf16 is True."""

num_layers_at_end_in_bf16: int = 1
"""Number of layers at the end of the model to keep in BF16 precision when
first_last_layers_bf16 is True."""
```

### 5. FP8 Parameter Quantization

For distributed optimizer, parameters can be stored and gathered in FP8:

```python
# From megatron/core/fp8_utils.py:221-294 (simplified for clarity)
def _quantize_param_shard_impl(
    model_params: List[QuantizedTensor],
    main_params: List[torch.Tensor],
    start_offsets: List[int],
    data_parallel_group: torch.distributed.ProcessGroup,
    fsdp_shard_model_params: Optional[List[torch.Tensor]] = None,
) -> None:
    """Cast fp32 main params to fp8 model params."""
    if len(model_params) == 0:
        return

    if fsdp_shard_model_params is None:
        fsdp_shard_model_params = [None] * len(model_params)

    for model_param, main_param, start_offset, fsdp_shard_model_param in zip(
        model_params, main_params, start_offsets, fsdp_shard_model_params
    ):
        if main_param is None:
            continue

        if fsdp_shard_model_param is not None:
            shard_model_param = fsdp_shard_model_param
        else:
            shard_model_param = model_param._data.view(-1)[
                start_offset : start_offset + main_param.numel()
            ]

        quantizer = model_param._quantizer
        # Cast main_param to model_param.dtype for numerical consistency
        main_param = main_param.to(model_param.dtype)
        out = Float8Tensor(
            shape=main_param.size(),
            dtype=model_param.dtype,
            requires_grad=False,
            data=shard_model_param,
            fp8_scale_inv=model_param._scale_inv,
            fp8_dtype=model_param._fp8_dtype,
            quantizer=quantizer,
        )
        quantizer.update_quantized(main_param, out)

    # Collect scaling factors and amaxes
    amaxes = []
    scales = []
    scale_invs = []
    for model_param in model_params:
        quantizer = model_param._quantizer
        amaxes.append(quantizer.amax.view(1))
        scales.append(quantizer.scale.view(1))
        scale_invs.append(model_param._scale_inv.view(1))
        model_param._reset_caches()

    # Reduce amaxes across data parallel group
    packed_amaxes = torch.empty(len(amaxes), dtype=torch.float32, device=amaxes[0].device)
    packed_amax_views = [packed_amaxes[i].view(1) for i in range(len(amaxes))]
    _multi_tensor_copy_this_to_that(amaxes, packed_amax_views, dummy_overflow_buf)
    torch.distributed.all_reduce(
        packed_amaxes, op=torch.distributed.ReduceOp.MAX, group=data_parallel_group
    )
    _multi_tensor_copy_this_to_that(packed_amax_views, amaxes, dummy_overflow_buf)
```

### 6. FP8 Inference Padding

For FP8 inference, sequences need proper alignment. The framework automatically wraps linear layers:

```python
# From megatron/core/fp8_utils.py:593-668 (simplified)
def _wrap_te_linear_for_padding(module: torch.nn.Module):
    """Wrap a TE linear module to automatically pad sequences for FP8 inference.

    Modifies the module's forward method to:
    1. Pad input sequences to FP8 alignment requirements
    2. Run the original forward pass
    3. Unpad outputs to original sequence length
    """
    if module in _fp8_inference_wrapped_modules:
        return
    _pad_func = Fp8Padding(1)
    _unpad_func = Fp8Unpadding(1)

    original_forward = module.forward

    @wraps(original_forward)
    def padded_forward(input_tensor, *args, **kwargs):
        # Only do padding for fp8 if we are in fp8 context
        if not FP8GlobalStateManager.is_fp8_enabled():
            return original_forward(input_tensor, *args, **kwargs)

        # Handle sequence parallelism if needed
        if is_sequence_parallel := getattr(module, "sequence_parallel", False):
            if is_column_parallel_linear(module):
                input_tensor = gather_from_sequence_parallel_region(
                    input_tensor, group=module.tp_group
                )
            module.sequence_parallel = False

        seq_len, batch_size, hidden_size = input_tensor.shape
        # Reshape to (S, B*H) to pad sequence dimension
        input_2d = input_tensor.reshape(seq_len, -1)
        # Pad the sequence dimension
        padded_input_2d, _ = _pad_func(input_2d, [seq_len])
        padded_seq_len = padded_input_2d.shape[0]

        # Reshape back to (padded_S, B, H)
        padded_input_3d = padded_input_2d.view(padded_seq_len, batch_size, hidden_size)
        output = original_forward(padded_input_3d, *args, **kwargs)

        # Unpad output
        if isinstance(output, tuple):
            output_tensor = output[0]
            other_outputs = output[1:]
        else:
            output_tensor = output
            other_outputs = ()

        _, _, output_hidden_size = output_tensor.shape
        output_2d = output_tensor.reshape(padded_seq_len, -1)
        unpadded_output_2d = _unpad_func(output_2d, [seq_len])
        unpadded_output = unpadded_output_2d.reshape(seq_len, batch_size, output_hidden_size)

        if is_sequence_parallel:
            if is_row_parallel_linear(module):
                unpadded_output = reduce_scatter_to_sequence_parallel_region(
                    unpadded_output, group=module.tp_group
                )
            module.sequence_parallel = True

        if other_outputs:
            return (unpadded_output,) + other_outputs
        else:
            return unpadded_output

    module.forward = padded_forward
    _fp8_inference_wrapped_modules.add(module)
```

## Configuration and Usage

### Command-Line Arguments

From `megatron/training/arguments.py:1316-1345`:

```bash
# Enable FP8 with hybrid format (E4M3 for forward, E5M2 for backward gradients)
--fp8-format hybrid

# Choose FP8 recipe
--fp8-recipe delayed       # Default, good balance
--fp8-recipe tensorwise    # More frequent scaling updates
--fp8-recipe blockwise     # Block-level scaling
--fp8-recipe mxfp8         # Microscaling (best accuracy)

# Scaling configuration
--fp8-margin 0                      # Margin for scaling factor
--fp8-interval 1                    # Deprecated in TE 1.8+
--fp8-amax-history-len 1024         # Length of amax history window
--fp8-amax-compute-algo most_recent # 'max' or 'most_recent'

# Enable FP8 parameter storage (saves memory with distributed optimizer)
--fp8-param-gather

# Disable weight gradient in FP8 (use higher precision)
--no-fp8-wgrad
```

### Python Configuration

```python
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.enums import Fp8Recipe

config = TransformerConfig(
    # Basic FP8 settings
    fp8='hybrid',  # or 'e4m3'
    fp8_recipe=Fp8Recipe.mxfp8,  # Best accuracy

    # Scaling parameters
    fp8_margin=0,
    fp8_amax_history_len=1024,
    fp8_amax_compute_algo='most_recent',

    # Keep first/last layers in BF16 for stability
    first_last_layers_bf16=True,
    num_layers_at_start_in_bf16=1,
    num_layers_at_end_in_bf16=1,

    # FP8 attention (optional, requires newer TE)
    fp8_dot_product_attention=True,
    fp8_multi_head_attention=True,

    # Parameter storage in FP8
    fp8_param=True,  # Requires distributed optimizer

    # Weight gradient precision
    fp8_wgrad=True,  # False = higher precision for wgrad

    # Reduce FP8 amax only in TP domain (not across DP)
    tp_only_amax_red=False,
)
```

## Performance Impact

### Training Speedup

On H100 GPUs with properly configured FP8:
- **1.5-2x** training throughput increase
- **50%** reduction in activation memory bandwidth
- **25-30%** faster gradient communication in distributed training

### Memory Savings

With `--fp8-param-gather`:
- Parameters stored in FP8 instead of BF16
- ~40% reduction in parameter memory
- Enables training larger models on same hardware

### Accuracy Considerations

Recipe accuracy comparison on large language models:
- `delayed`: ~0.5-1% accuracy degradation vs BF16
- `tensorwise`: ~0.3-0.7% accuracy degradation
- `blockwise`: ~0.1-0.4% accuracy degradation
- `mxfp8`: ~0.1-0.3% accuracy degradation (best)

## Hardware Requirements

**Supported:**
- NVIDIA H100 (optimal)
- NVIDIA H200
- NVIDIA Blackwell (B100, B200)
- Future NVIDIA architectures with FP8 tensor cores

**Not Supported:**
- NVIDIA A100 (no FP8 hardware)
- NVIDIA V100
- Consumer GPUs (RTX series)

## Best Practices

1. **Start with MXFP8**: Best accuracy-performance trade-off
2. **Monitor training stability**: Watch for loss spikes or NaN
3. **Keep first/last layers in BF16**: Improves stability
4. **Use long amax history**: `fp8_amax_history_len=1024` or higher
5. **Enable FP8 parameter storage**: With distributed optimizer for memory savings
6. **Test accuracy**: Validate model quality matches BF16 baseline

## Troubleshooting

**Issue**: Training diverges or produces NaN

**Solutions**:
- Switch from `delayed` to `mxfp8` recipe
- Enable `first_last_layers_bf16=True`
- Increase `fp8_amax_history_len` to 2048 or 4096
- Set `fp8_wgrad=False` for higher precision weight gradients
- Use `fp8_amax_compute_algo='max'` instead of `'most_recent'`

**Issue**: No speedup observed

**Solutions**:
- Verify H100+ GPU: `nvidia-smi` should show H100/H200/B100
- Check TransformerEngine version: `pip show transformer-engine` (need >= 2.1.0)
- Ensure FP8 is actually enabled: Check logs for "FP8 enabled" messages

**Issue**: Accuracy degradation too high

**Solutions**:
- Use `mxfp8` recipe for best accuracy
- Enable `first_last_layers_bf16=True` with more BF16 layers
- Disable FP8 for attention: `fp8_dot_product_attention=False`
- Consider hybrid approach: Some layers in FP8, critical layers in BF16

## Related Features

- **MXFP8 Blockwise Scaling**: Enhanced accuracy with microscaling
- **FP8 Inference Padding**: Automatic sequence padding for FP8 inference
- **Distributed Optimizer**: Required for `--fp8-param-gather`
- **Gradient Buffer Reuse**: Can reuse gradient buffers for MXFP8 all-gather

## References

1. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) - Original FP8 paper
2. [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine) - FP8 implementation
3. [H100 Tensor Core Architecture](https://www.nvidia.com/en-us/data-center/h100/) - Hardware capabilities
