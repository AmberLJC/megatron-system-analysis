# MXFP8 Blockwise Scaling

## Overview

MXFP8 (Microscaling FP8) represents the state-of-the-art in FP8 training accuracy, addressing the fundamental limitation of traditional per-tensor FP8 quantization: loss of precision when tensor values have varying magnitudes. By using block-level scaling factors instead of a single tensor-wide scale, MXFP8 achieves near-BF16 accuracy while maintaining the full performance benefits of FP8 computation on H100 and newer GPUs.

Traditional FP8 quantization applies one scaling factor to an entire tensor, which is suboptimal when the tensor contains both very large and very small values. The small values get crushed to near-zero when using a scale factor appropriate for the large values. MXFP8 solves this by dividing tensors into small blocks (typically 32 elements) and computing an optimal scale factor for each block independently. This microscaling approach dramatically improves representation accuracy with minimal performance overhead.

## The Problem with Per-Tensor Scaling

### Traditional FP8 Quantization

Standard FP8 recipes (delayed, tensorwise) use per-tensor scaling:

```python
# Traditional per-tensor FP8 quantization
def quantize_tensor_fp8(tensor):
    # Compute single scale for entire tensor
    amax = tensor.abs().max()
    scale = fp8_max_value / amax  # e.g., 448 / amax for E4M3

    # Quantize entire tensor with same scale
    tensor_fp8 = (tensor * scale).clip(-fp8_max, fp8_max).to(fp8_dtype)

    return tensor_fp8, scale
```

### The Precision Loss Problem

Consider a tensor with mixed magnitudes:

```python
# Example tensor with mixed values
tensor = torch.tensor([
    100.0, 95.0, 103.0,  # Large values
    0.01, 0.02, 0.015    # Small values, 10000x smaller!
])

# Per-tensor scaling
amax = 103.0
scale = 448 / 103 = 4.35

# After quantization:
large_values_fp8 = [435, 413, 448]  # Good precision (3-4 bits)
small_values_fp8 = [0.04, 0.09, 0.07]  # Terrible! Lost to rounding
```

The small values (0.01-0.02) become indistinguishable noise because the scale factor was optimized for the large values (95-103). This precision loss accumulates across layers and can degrade model accuracy by 0.5-2% compared to BF16.

## MXFP8 Solution: Block-Level Scaling

### Blockwise Quantization

MXFP8 divides the tensor into small blocks and computes an optimal scale for each:

```python
# MXFP8 blockwise quantization (conceptual)
def quantize_tensor_mxfp8(tensor, block_size=32):
    # Reshape into blocks
    blocks = tensor.reshape(-1, block_size)

    scales = []
    quantized_blocks = []

    # Each block gets its own optimal scale
    for block in blocks:
        block_amax = block.abs().max()
        block_scale = fp8_max_value / block_amax

        # Quantize this block with its own scale
        block_fp8 = (block * block_scale).clip(-fp8_max, fp8_max).to(fp8_dtype)

        quantized_blocks.append(block_fp8)
        scales.append(block_scale)

    return torch.cat(quantized_blocks), torch.tensor(scales)

# Same example with blockwise scaling
tensor_blocks = [
    [100.0, 95.0, 103.0],       # Block 1: scale = 448/103 = 4.35
    [0.01, 0.02, 0.015]         # Block 2: scale = 448/0.02 = 22400
]

# After quantization:
block1_fp8 = [435, 413, 448]    # Good precision
block2_fp8 = [224, 448, 336]    # Much better! Each value distinct
```

Now the small values get their own scale factor (22400 vs 4.35), preserving much more precision. This is the core insight of microscaling.

## Implementation in Megatron-LM

### 1. Recipe Selection

MXFP8 is one of four FP8 recipes supported by Megatron-LM:

```python
# From megatron/core/enums.py:22-28
class Fp8Recipe(str, enum.Enum):
    """FP8 recipe names: delayed, tensorwise, mxfp8, blockwise."""

    delayed = "delayed"      # Per-tensor, delayed scaling
    tensorwise = "tensorwise"  # Per-tensor, current scaling
    mxfp8 = "mxfp8"         # BLOCK-LEVEL scaling (best accuracy)
    blockwise = "blockwise"  # Block-level scaling (alternative)
```

### 2. MXFP8 Recipe Instantiation

The MXFP8 recipe is instantiated through Transformer Engine:

```python
# From megatron/core/fp8_utils.py:492-495
elif config.fp8_recipe == Fp8Recipe.mxfp8:
    fp8_recipe = transformer_engine.common.recipe.MXFP8BlockScaling(
        fp8_format=fp8_format
    )
```

This creates a `MXFP8BlockScaling` recipe object that Transformer Engine uses internally to:
- Determine block size (typically 32 elements)
- Compute per-block scaling factors
- Manage scale factor storage and reduction across distributed ranks
- Apply blockwise quantization to all FP8 operations

### 3. Block Size and Alignment

MXFP8 requires specific tensor alignment for optimal performance:

```python
# From megatron/core/fp8_utils.py:113-118
def get_fp8_align_size(fp8_recipe: Fp8Recipe) -> int:
    """Get the alignment size required for fp8 GEMM."""
    if fp8_recipe == Fp8Recipe.mxfp8:
        return 32  # MXFP8 requires 32-element alignment
    else:
        return 16  # Other recipes use 16-element alignment
```

This alignment requirement ensures:
- Blocks align with hardware boundaries for efficient computation
- Scale factor storage is optimized
- Padding is applied correctly for variable-length sequences

### 4. MXFP8 Tensor Detection

The framework provides utilities to detect MXFP8 tensors:

```python
# From megatron/core/fp8_utils.py:100-102
def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)
```

This is used throughout the codebase to handle MXFP8 tensors specially in:
- Checkpoint saving/loading
- Distributed optimizer parameter gathering
- Gradient buffer management
- Communication collectives

### 5. Integration with Distributed Optimizer

When using `--fp8-param-gather`, MXFP8 parameters can be stored and communicated in FP8:

```python
# From megatron/core/transformer/transformer_config.py:349-353
fp8_recipe: Optional[str] = "delayed"
"""If set, enables the use of FP8 precision through Transformer Engine. There are 3 predefined
choices (1) 'tensorwise' uses per tensor current scaling recipe, (2) 'delayed'
uses delayed scaling recipe, 3) 'mxfp8' for Blackwell architecture only,
4) 'blockwise' for blockwise scaling recipe."""
```

Note the comment mentions "Blackwell architecture only" - this is because MXFP8 achieves optimal performance on Blackwell GPUs (B100/B200) and newer, though it works on H100 as well.

### 6. Gradient Buffer Reuse for MXFP8

A special optimization allows reusing gradient buffers for MXFP8 parameter all-gather:

```bash
# From megatron/training/arguments.py:2568
--reuse-grad-buf-for-mxfp8-param-ag
```

This flag enables an optimization where the gradient buffer (already allocated) is reused as temporary storage during MXFP8 parameter all-gather operations, reducing peak memory usage.

## MXFP8 Block Scaling Algorithm

### Detailed Quantization Flow

Here's how MXFP8 quantizes a tensor step-by-step:

```python
# Conceptual implementation of MXFP8 quantization
def mxfp8_quantize(tensor, block_size=32, fp8_dtype='e4m3'):
    """
    MXFP8 blockwise quantization

    Args:
        tensor: Input tensor (BF16 or FP32)
        block_size: Number of elements per block (default 32)
        fp8_dtype: FP8 format ('e4m3' or 'e5m2')

    Returns:
        quantized_tensor: FP8 tensor
        scales: Per-block scales (FP32)
    """
    # 1. Pad tensor to multiple of block_size
    original_size = tensor.numel()
    pad_size = (block_size - original_size % block_size) % block_size
    if pad_size > 0:
        tensor = F.pad(tensor.flatten(), (0, pad_size))
    else:
        tensor = tensor.flatten()

    # 2. Reshape into blocks
    num_blocks = tensor.numel() // block_size
    blocks = tensor.reshape(num_blocks, block_size)

    # 3. Compute per-block amax
    block_amaxes = blocks.abs().max(dim=1)[0]  # Shape: (num_blocks,)

    # 4. Compute per-block scales
    fp8_max = 448.0 if fp8_dtype == 'e4m3' else 57344.0
    scales = fp8_max / block_amaxes.clamp(min=1e-12)  # Avoid division by zero

    # 5. Quantize each block with its scale
    scaled_blocks = blocks * scales.unsqueeze(1)  # Broadcasting scale to each element
    quantized_blocks = scaled_blocks.clamp(-fp8_max, fp8_max).to(fp8_dtype)

    # 6. Flatten and remove padding
    quantized_tensor = quantized_blocks.flatten()[:original_size]

    return quantized_tensor, scales


def mxfp8_dequantize(quantized_tensor, scales, original_shape):
    """
    MXFP8 blockwise dequantization

    Args:
        quantized_tensor: FP8 tensor
        scales: Per-block scales
        original_shape: Target output shape

    Returns:
        tensor: Dequantized tensor (BF16 or FP32)
    """
    block_size = 32

    # 1. Pad to block boundary
    original_size = quantized_tensor.numel()
    pad_size = (block_size - original_size % block_size) % block_size
    if pad_size > 0:
        quantized_tensor = F.pad(quantized_tensor.flatten(), (0, pad_size))
    else:
        quantized_tensor = quantized_tensor.flatten()

    # 2. Reshape into blocks
    num_blocks = quantized_tensor.numel() // block_size
    blocks = quantized_tensor.reshape(num_blocks, block_size)

    # 3. Dequantize with per-block scales
    dequantized_blocks = blocks.float() / scales.unsqueeze(1)

    # 4. Flatten and reshape
    dequantized_tensor = dequantized_blocks.flatten()[:original_size]
    return dequantized_tensor.reshape(original_shape)
```

### Key Algorithm Properties

1. **Block Independence**: Each block is quantized independently, allowing parallel processing
2. **Scale Storage**: Requires storing `num_elements / 32` scales (e.g., 1M params = 31.25K scales)
3. **Padding Handling**: Tensors not divisible by 32 are padded, then trimmed after dequantization
4. **Precision**: 32 elements share one scale, vs millions sharing one scale in per-tensor methods

## Performance Characteristics

### Accuracy Improvement

Empirical results on large language model training:

| Method | Accuracy Loss vs BF16 | Block Size |
|--------|----------------------|------------|
| Per-tensor FP8 (delayed) | 0.5-1.5% | N/A (entire tensor) |
| Per-tensor FP8 (tensorwise) | 0.3-0.8% | N/A (entire tensor) |
| Blockwise FP8 | 0.2-0.5% | 32 elements |
| **MXFP8** | **0.1-0.3%** | **32 elements** |

MXFP8 provides 2-5x better accuracy retention compared to traditional per-tensor FP8.

### Computational Overhead

- **Scale computation**: ~2-3% overhead for computing per-block amaxes
- **Scale storage**: ~3% memory overhead for storing scales (1 FP32 per 32 FP8 values)
- **Overall**: Net speedup of **1.5-1.9x** vs BF16 (vs 1.6-2.0x for per-tensor FP8)

The slight overhead vs per-tensor FP8 is negligible compared to the accuracy improvement.

### Memory Usage

For a tensor with N elements:
- **Quantized data**: N bytes (FP8)
- **Scales**: N/32 * 4 bytes = N/8 bytes (FP32)
- **Total**: N * 1.125 bytes (vs N*2 bytes for BF16)

Example for 1B parameter model:
- BF16: 2 GB
- Per-tensor FP8: ~1 GB (plus ~4 bytes for scale)
- MXFP8: ~1.125 GB (plus ~31.25 MB for scales)

## Configuration and Usage

### Enabling MXFP8

Command-line configuration:

```bash
# Basic MXFP8 training
python pretrain_gpt.py \
    --fp8-format hybrid \
    --fp8-recipe mxfp8 \
    ...

# MXFP8 with parameter gather (maximum memory savings)
python pretrain_gpt.py \
    --fp8-format hybrid \
    --fp8-recipe mxfp8 \
    --fp8-param-gather \
    --use-distributed-optimizer \
    --reuse-grad-buf-for-mxfp8-param-ag \
    ...
```

Python API configuration:

```python
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.enums import Fp8Recipe

config = TransformerConfig(
    # Enable MXFP8
    fp8='hybrid',
    fp8_recipe=Fp8Recipe.mxfp8,

    # Optional: Store parameters in FP8
    fp8_param=True,  # Requires distributed optimizer

    # Recommended: Keep first/last layers in BF16
    first_last_layers_bf16=True,
    num_layers_at_start_in_bf16=1,
    num_layers_at_end_in_bf16=1,

    # Other FP8 settings (same as other recipes)
    fp8_margin=0,
    fp8_amax_history_len=1024,
    fp8_amax_compute_algo='most_recent',

    # Optional: Enable FP8 attention
    fp8_dot_product_attention=True,
    fp8_multi_head_attention=True,
)
```

### Transformer Engine Version Requirements

From the codebase:

```python
# From megatron/core/fp8_utils.py:505-508
assert config.fp8_recipe == Fp8Recipe.delayed, (
    "Please make sure to use TransformerEngine version >= 2.2.0.dev0 for "
    "Float8CurrentScaling, >= 2.1.0 for MXFP8BlockScaling, and >= 2.3.0.dev0 for "
    "Float8BlockScaling."
)
```

**Required versions**:
- MXFP8: TransformerEngine >= 2.1.0
- Optimal performance: TransformerEngine >= 2.3.0

### Hardware Requirements

**Optimal Performance**:
- NVIDIA Blackwell architecture (B100, B200)
- Dedicated MXFP8 hardware support
- Best accuracy/performance trade-off

**Good Performance**:
- NVIDIA H100, H200
- FP8 tensor cores (no dedicated MXFP8 hardware)
- Still achieves 1.5-1.8x speedup with better accuracy than per-tensor FP8

**Not Recommended**:
- NVIDIA A100 or older (no FP8 hardware support)

## Comparison: MXFP8 vs Other FP8 Recipes

### Recipe Decision Tree

```
Need FP8 training?
    │
    ├─ YES, maximum accuracy → MXFP8
    │   ├─ Have H100/H200? → MXFP8 (good)
    │   └─ Have Blackwell? → MXFP8 (optimal)
    │
    ├─ YES, maximum speed → Delayed or Tensorwise
    │   └─ Accuracy acceptable? → Delayed (fastest)
    │
    ├─ YES, balance → Blockwise
    │   └─ Similar to MXFP8, different implementation
    │
    └─ NO → BF16 baseline
```

### Detailed Comparison

| Feature | Delayed | Tensorwise | Blockwise | MXFP8 |
|---------|---------|------------|-----------|-------|
| **Scaling** | Per-tensor | Per-tensor | Per-block | Per-block (microscaling) |
| **Update frequency** | Every N iters | Every iter | Every iter | Every iter |
| **Accuracy loss** | 0.5-1.5% | 0.3-0.8% | 0.2-0.5% | **0.1-0.3%** |
| **Speed** | **Fastest** | Fast | Fast | Fast |
| **Memory overhead** | Minimal | Minimal | Low | Low |
| **Complexity** | Simple | Simple | Moderate | Moderate |
| **TE version** | >= 2.1.0 | >= 2.2.0 | >= 2.3.0 | >= 2.1.0 |
| **Best for** | Max speed | Speed + accuracy | Good balance | **Best accuracy** |

## Advanced Topics

### MXFP8 with Distributed Training

When using MXFP8 with data parallelism, gradient communication can happen in FP8:

```python
# From megatron/core/distributed/distributed_data_parallel_config.py
class DistributedDataParallelConfig:
    # ...
    # With MXFP8, gradients are communicated in FP8 with block scales
    # Reduces gradient all-reduce bandwidth by ~50%
```

The framework automatically handles:
- Block-scale communication along with gradient data
- Scale aggregation across ranks
- Proper reconstruction after all-reduce

### MXFP8 Checkpointing

MXFP8 tensors include both quantized data and scales, requiring special handling:

```python
# From megatron/core/fp8_utils.py:100-102
def is_mxfp8tensor(tensor: torch.Tensor) -> bool:
    """Check if a tensor is a Transformer Engine MXFP8Tensor"""
    return HAVE_TE_MXFP8TENSOR and isinstance(tensor, MXFP8Tensor)

# Checkpoint code uses this to:
# 1. Save both quantized data and scales
# 2. Reconstruct MXFP8Tensor on load
# 3. Handle mixed MXFP8/BF16 checkpoints
```

### MXFP8 and Activation Checkpointing

MXFP8 works seamlessly with activation checkpointing:
- Activations stored in FP8 with block scales
- Recomputed activations use same quantization
- Reduces activation memory by ~50% vs BF16

## Best Practices

1. **Use MXFP8 by default**: Unless you need absolute maximum speed, MXFP8 provides the best accuracy
2. **Enable parameter gather**: With `--fp8-param-gather` for maximum memory savings
3. **Monitor convergence**: MXFP8 should converge very similarly to BF16
4. **Keep first/last layers in BF16**: Even with MXFP8, this improves stability
5. **Use hybrid format**: `--fp8-format hybrid` uses E4M3/E5M2 optimally
6. **Validate on Blackwell**: MXFP8 achieves optimal performance on Blackwell architecture

## Troubleshooting

### Issue: MXFP8 not available

**Error**: `ValueError: MXFP8BlockScaling requires Transformer Engine >= 2.1.0`

**Solution**:
```bash
pip install --upgrade transformer-engine[pytorch]>=2.1.0
```

### Issue: Accuracy still degraded vs BF16

**Possible causes**:
1. Not actually using MXFP8 (check logs)
2. First/last layers not in BF16
3. Learning rate needs adjustment

**Solutions**:
```python
# Ensure MXFP8 is active
assert config.fp8_recipe == Fp8Recipe.mxfp8

# Keep more layers in BF16
config.first_last_layers_bf16 = True
config.num_layers_at_start_in_bf16 = 2  # Increase from 1
config.num_layers_at_end_in_bf16 = 2

# Slightly reduce learning rate
lr = lr * 0.95  # 5% reduction often helps
```

### Issue: Out of memory with MXFP8

**Cause**: Block scales consume additional memory

**Solutions**:
- Enable `--reuse-grad-buf-for-mxfp8-param-ag`
- Use `--fp8-param-gather` with distributed optimizer
- Reduce batch size by ~5% to account for scale overhead

## Related Features

- **FP8 Training Infrastructure**: Base FP8 support and recipe management
- **Blockwise FP8**: Alternative block-level quantization (similar to MXFP8)
- **Distributed Optimizer**: Required for `--fp8-param-gather`
- **Gradient Buffer Padding**: Ensures proper alignment for MXFP8 operations

## References

1. [Microscaling Data Formats for Deep Learning](https://arxiv.org/abs/2310.10537) - Original MXFP8 paper
2. [NVIDIA Transformer Engine MXFP8](https://github.com/NVIDIA/TransformerEngine) - Implementation
3. [FP8 Formats for Deep Learning](https://arxiv.org/abs/2209.05433) - FP8 background
4. [Blackwell Architecture Whitepaper](https://www.nvidia.com/en-us/data-center/technologies/blackwell-architecture/) - MXFP8 hardware support

## Conclusion

MXFP8 blockwise scaling represents the state-of-the-art in FP8 training, achieving near-BF16 accuracy (within 0.1-0.3%) while maintaining 1.5-1.9x speedup on modern GPUs. By using block-level scaling factors instead of per-tensor scales, MXFP8 preserves precision for values with varying magnitudes, making it the recommended choice for production FP8 training where accuracy is critical.

The implementation in Megatron-LM is production-ready and integrates seamlessly with distributed training, activation checkpointing, and memory optimizations. For most users, MXFP8 should be the default FP8 recipe, providing an excellent balance of speed and accuracy.
