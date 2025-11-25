# 42. Long-Context Training Optimizations

## Overview

Long-context training enables language models to process sequences of 32K, 64K, 128K tokens or more. This capability is essential for applications requiring extended reasoning, document understanding, and complex multi-turn conversations. However, training with long sequences presents significant challenges: attention complexity scales quadratically O(n²) with sequence length, activation memory grows linearly, and communication overhead increases substantially in distributed settings.

Megatron-LM addresses these challenges through a comprehensive suite of optimizations spanning parallelism strategies, memory management, efficient attention mechanisms, and advanced positional encodings. This report details each optimization technique, its implementation, and configuration options.

**Key Optimizations Covered:**
1. **Context Parallelism (CP)**: Splits sequences across GPUs with ring attention communication
2. **Ring Attention**: Asynchronous KV exchange enabling computation-communication overlap
3. **Positional Encodings**: RoPE interpolation and YARN for context extension
4. **Flash Attention**: Memory-efficient attention with linear memory complexity
5. **Variable Sequence Length**: Packed sequence formats eliminating padding overhead
6. **Activation Checkpointing**: Selective recomputation for memory efficiency

---

## 1. Context Parallelism

### Problem Statement

When training with long sequences, activation memory becomes the primary bottleneck. For a transformer layer processing a 32K token sequence with hidden dimension 8192 in BF16:

```
Activation per layer = seq_len × batch × hidden × 2 bytes
                     = 32768 × 1 × 8192 × 2
                     = 512 MB per activation tensor
```

With multiple activation tensors per layer (Q, K, V, attention output, MLP intermediates), a single layer can require several gigabytes. Across 80+ layers, this quickly exceeds GPU memory capacity.

### Solution: Sequence Dimension Partitioning

Context Parallelism (CP) partitions the sequence dimension across multiple GPUs. Each GPU processes only `seq_len / cp_size` tokens, reducing activation memory proportionally.

```python
# Standard attention (full sequence per GPU)
# Each GPU: [seq_len, batch, hidden] = [32768, 1, 8192]
activation_per_gpu = 32768 * 1 * 8192 * 2  # 512 MB

# With Context Parallelism (CP=4)
# Each GPU: [seq_len/CP, batch, hidden] = [8192, 1, 8192]
activation_per_gpu = 8192 * 1 * 8192 * 2   # 128 MB (75% reduction)
```

### Implementation Architecture

Context Parallelism is initialized through process group configuration in `megatron/core/parallel_state.py`:

```python
# Global CP group variables (parallel_state.py:106-127)
_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GLOBAL_RANKS = None
_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = None

def get_context_parallel_group(check_initialized=True):
    """Get the context-parallel group the caller rank belongs to."""
    if check_initialized:
        assert _CONTEXT_PARALLEL_GROUP is not None
    return _CONTEXT_PARALLEL_GROUP

def get_context_parallel_world_size():
    """Return world size for the context parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(group=get_context_parallel_group())
    return 0

def get_context_parallel_rank():
    """Return caller's rank in the context-parallel group."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank(group=get_context_parallel_group())
    return 0
```

### Configuration Parameters

Core CP configuration resides in `megatron/core/model_parallel_config.py`:

```python
@dataclass
class ModelParallelConfig:
    context_parallel_size: int = 1
    """Splits network input along sequence dimension across GPU ranks."""

    hierarchical_context_parallel_sizes: Optional[list[int]] = None
    """Degrees of hierarchical context parallelism.
    For a2a+p2p mode: [a2a_group_size, p2p_group_size]
    Enables multi-level communication optimization for complex network topologies."""
```

Extended CP configuration in `megatron/core/transformer/transformer_config.py`:

```python
@dataclass
class TransformerConfig(ModelParallelConfig):
    cp_comm_type: Optional[Union[str, List[str]]] = None
    """Inter-GPU communication type for context parallelism.
    Options:
    - "p2p": Point-to-point ring attention (lowest latency for NVLink)
    - "all_gather": Gather full KV before attention (simplest)
    - "a2a": All-to-all DeepSpeed Ulysses style (best bandwidth)
    - "a2a+p2p": Hierarchical combining both (multi-level networks)
    Can specify per-layer as list: ["p2p", "a2a", "p2p", ...]
    """

    overlap_p2p_comm: bool = False
    """Overlap P2P communication with attention computation."""

    batch_p2p_comm: bool = True
    """Batch multiple P2P operations together."""

    batch_p2p_sync: bool = True
    """Synchronize batched P2P operations."""

    use_ring_exchange_p2p: bool = False
    """Use ring exchange pattern for P2P communication."""
```

### Communication Patterns

Megatron-LM supports four distinct CP communication patterns optimized for different network topologies:

| Pattern | Description | Best For | Latency | Bandwidth |
|---------|-------------|----------|---------|-----------|
| **P2P** | Ring-based KV exchange between consecutive ranks | NVLink, small CP (2-4) | Lowest | Medium |
| **All-Gather** | Synchronous full KV gathering | Debugging, simple cases | High | Low |
| **A2A** | All-to-all head distribution (Ulysses) | Large CP, IB networks | Medium | High |
| **A2A+P2P** | Hierarchical multi-level | Multi-tier networks | Adaptive | Optimal |

### Memory Savings Calculation

For a GPT-3 175B configuration with 32K context:

| Configuration | Activation Memory/GPU | Savings |
|---------------|----------------------|---------|
| No CP (baseline) | 18.4 GB | - |
| CP=2 | 9.2 GB | 50% |
| CP=4 | 4.6 GB | 75% |
| CP=8 | 2.3 GB | 87.5% |

---

## 2. Ring Attention

### Concept

Ring Attention is the communication mechanism that enables Context Parallelism. Each GPU holds a portion of the query (Q) tokens but needs access to all key-value (KV) tokens to compute attention. Rather than gathering all KV pairs upfront (expensive), ring attention circulates KV chunks through a ring of GPUs while overlapping communication with computation.

### How Ring Attention Works

```
Step 1: Initial State
┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│  GPU 0  │  │  GPU 1  │  │  GPU 2  │  │  GPU 3  │
│ Q[0:N/4]│  │Q[N/4:N/2]│ │Q[N/2:3N/4]│ │Q[3N/4:N]│
│KV[0:N/4]│  │KV[N/4:N/2]│KV[N/2:3N/4]│KV[3N/4:N]│
└─────────┘  └─────────┘  └─────────┘  └─────────┘

Step 2: Compute local attention + async send KV to next rank
GPU 0: Attn(Q[0:N/4], KV[0:N/4]) + send KV[0:N/4] → GPU 1
GPU 1: Attn(Q[N/4:N/2], KV[N/4:N/2]) + send KV[N/4:N/2] → GPU 2
...

Step 3: Receive KV from previous rank, compute, send
GPU 0: recv KV[3N/4:N] from GPU 3, compute Attn(Q[0:N/4], KV[3N/4:N])
...

After CP-1 steps: All GPUs have computed full attention
```

### Implementation Details

The ring attention mechanism is integrated through Transformer Engine's `TEDotProductAttention` class in `megatron/core/extensions/transformer_engine.py`:

```python
class TEDotProductAttention(te.pytorch.DotProductAttention):
    """TE DotProductAttention with Context Parallelism support."""

    # Class-level CUDA stream for async CP communication
    cp_stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
        ...
    ):
        # CP-specific initialization (lines 913-937)
        if self.config.context_parallel_size > 1:
            # Requires Transformer-Engine >= 1.0.0 for CP support
            if getattr(TEDotProductAttention, "cp_stream") is None:
                TEDotProductAttention.cp_stream = torch.cuda.Stream()

            extra_kwargs["cp_group"] = pg_collection.cp
            extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
                pg_collection.cp
            )
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream

            # Configure communication pattern based on TE version
            if is_te_min_version("1.10.0"):
                if cp_comm_type == "a2a+p2p":
                    # Hierarchical mode: multi-level groups (requires TE >= 1.12.0)
                    extra_kwargs["cp_comm_type"] = "a2a+p2p"
                    extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups()
                else:
                    extra_kwargs["cp_comm_type"] = cp_comm_type  # p2p, all_gather, or a2a
```

### Key Implementation Features

1. **Dedicated CUDA Stream**: A separate stream (`cp_stream`) handles KV communication, enabling overlap with attention computation on the default stream.

2. **Asynchronous P2P Operations**: Send/receive operations are non-blocking, allowing computation to proceed while data transfers occur in the background.

3. **Ring Topology**: Each rank communicates only with immediate neighbors (rank±1), minimizing network congestion and providing predictable latency.

4. **Hierarchical Communication (A2A+P2P)**: For multi-tier networks, combines all-to-all within high-bandwidth groups (NVLink) with P2P across lower-bandwidth connections (InfiniBand).

### Communication Volume Analysis

For sequence length N and CP size P:

| Pattern | Data Transferred per GPU | Phases |
|---------|-------------------------|--------|
| **All-Gather** | (P-1)/P × N × d | 1 |
| **P2P Ring** | 2 × (P-1) × N/P × d | P-1 |
| **A2A** | (P-1)/P × N × d | 1 |

Where d = head_dim × num_kv_heads × 2 (K and V).

For P2P, although total data transferred is similar, the incremental nature allows for better overlap:

```
Overlap efficiency:
- All-Gather: ~0% (must wait for full gather before compute)
- P2P Ring: ~80-90% (compute overlaps with next chunk transfer)
- A2A+P2P: ~70-85% (adaptive based on network topology)
```

---

## 3. Positional Encodings for Long Contexts

### The Challenge

Standard positional encodings are trained for a specific maximum sequence length. When extending beyond this length, the model encounters unseen position indices, leading to degraded performance. Long-context training requires positional encoding strategies that generalize beyond the training length.

### RoPE (Rotary Position Embedding)

RoPE encodes positions through rotation matrices applied to query and key vectors, providing relative position information that naturally extends to longer sequences.

#### Implementation

Located in `megatron/core/models/common/embeddings/rotary_pos_embedding.py`:

```python
class RotaryEmbedding(nn.Module):
    """Rotary Embedding for language model.

    Args:
        kv_channels: Projection weights dimension in multi-head attention
        rotary_percent: Percent of rotary dimension to use (default 1.0)
        rotary_interleaved: If True, use interleaved rotary position embeddings
        seq_len_interpolation_factor: Scale for linear interpolation to longer sequences
        rotary_base: Base period for rotary embeddings (default 10000)
        rope_scaling: Apply LLaMA 3.x style rope scaling
        rope_scaling_factor: Scaling factor for LLaMA 3.x (default 8.0)
        cp_group: Process group for context parallel
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float,
        rotary_interleaved: bool = False,
        seq_len_interpolation_factor: float = None,
        rotary_base: int = 10000,
        rope_scaling: bool = False,
        rope_scaling_factor: float = 8.0,
        cp_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> None:
        super().__init__()

        dim = kv_channels
        if rotary_percent < 1.0:
            dim = int(dim * rotary_percent)

        self.seq_len_interpolation_factor = seq_len_interpolation_factor

        # Inverse frequency computation
        self.inv_freq = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        )

        # Apply LLaMA 3.x scaling if enabled
        if rope_scaling:
            self.inv_freq = self._apply_scaling(self.inv_freq, factor=rope_scaling_factor)

        self.cp_group = cp_group
```

#### Sequence Length Interpolation

For extending to longer sequences without retraining:

```python
def get_freqs_non_repeated(self, max_seq_len: int, offset: int = 0) -> Tensor:
    """Generate position frequencies with optional interpolation."""
    seq = torch.arange(max_seq_len, device=self.inv_freq.device) + offset

    # Key: Scale positions for longer sequences
    if self.seq_len_interpolation_factor is not None:
        seq *= 1 / self.seq_len_interpolation_factor

    freqs = torch.outer(seq, self.inv_freq)
    return freqs
```

With `seq_len_interpolation_factor=2.0`, a model trained on 4K context can handle 8K sequences by scaling position indices to fit within the original training range.

#### LLaMA 3.x Rope Scaling

A more sophisticated approach that applies frequency-dependent scaling:

```python
def _apply_scaling(self, freqs, factor=8, low_freq_factor=1,
                   high_freq_factor=4, original_max_position_embeddings=8192):
    """Apply LLaMA 3.x style rope scaling with frequency-aware adjustment."""
    old_context_len = original_max_position_embeddings

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * math.pi / freqs

    # Different scaling for different frequency bands:
    # - High frequencies (short wavelengths): unchanged
    # - Low frequencies (long wavelengths): scaled by factor
    # - Medium frequencies: smooth interpolation
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, freqs / factor, freqs)

    smooth_factor = (old_context_len / wavelen - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    smoothed_inv_freq = (1 - smooth_factor) * inv_freq_llama / factor + smooth_factor * inv_freq_llama

    is_medium_freq = ~(wavelen < high_freq_wavelen) * ~(wavelen > low_freq_wavelen)
    inv_freq_llama = torch.where(is_medium_freq, smoothed_inv_freq, inv_freq_llama)

    return inv_freq_llama
```

#### Context Parallel Integration

RoPE automatically handles sequence splitting for Context Parallelism:

```python
def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
    freqs = self.get_freqs_non_repeated(max_seq_len, offset)

    if not self.rotary_interleaved:
        emb = torch.cat((freqs, freqs), dim=-1)
    else:
        emb = torch.stack((freqs.view(-1, 1), freqs.view(-1, 1)), dim=-1).view(
            freqs.shape[0], -1
        )

    emb = emb[:, None, None, :]

    # Slice for Context Parallelism - each rank gets its portion
    if self.cp_group is not None and self.cp_group.size() > 1 and not packed_seq:
        emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)

    return emb
```

### YARN (Yet Another RoPE eNhancement)

YARN provides superior context extension through dimension-aware frequency correction, located in `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py`:

```python
class YarnRotaryEmbedding(RotaryEmbedding):
    """YARN RoPE for improved long-context generalization.

    Args:
        scaling_factor: Context extension factor (e.g., 4.0 for 4x)
        original_max_position_embeddings: Training context length
        beta_fast: Fast rotation boundary (default 32.0)
        beta_slow: Slow rotation boundary (default 1.0)
        mscale: Concentration factor scale (default 1.0)
        mscale_all_dim: Apply mscale across all dimensions (default 0.0)
    """

    def __init__(
        self,
        kv_channels: int,
        rotary_percent: float = 1.0,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: float = 32.0,
        beta_slow: float = 1.0,
        mscale: float = 1.0,
        mscale_all_dim: float = 0.0,
        ...
    ):
        # Two frequency sets: extrapolation and interpolation
        self.inv_freq_extra = 1.0 / (
            rotary_base ** (torch.arange(0, dim, 2) / dim)
        )
        self.inv_freq_inter = 1.0 / (
            scaling_factor * rotary_base ** (torch.arange(0, dim, 2) / dim)
        )
```

#### YARN Forward Pass

```python
def forward(self, max_seq_len: int, offset: int = 0, packed_seq: bool = False) -> Tensor:
    # Find correction range based on rotation counts
    low, high = _yarn_find_correction_range(
        self.beta_fast, self.beta_slow, self.dim,
        self.rotary_base, self.original_max_position_embeddings
    )

    # Create mask for blending extrapolation and interpolation
    inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, self.dim // 2)

    # Blend frequencies: interpolate for high dims, extrapolate for low dims
    inv_freq = self.inv_freq_inter * (1 - inv_freq_mask) + self.inv_freq_extra * inv_freq_mask

    seq = torch.arange(max_seq_len) + offset
    freqs = torch.outer(seq, inv_freq)

    # Apply concentration factor (attention scaling)
    _mscale = _yarn_get_concentration_factor(
        self.scaling_factor, self.mscale, self.mscale_all_dim
    )

    emb = torch.cat((freqs, freqs), dim=-1)

    if self.cp_group is not None and self.cp_group.size() > 1 and not packed_seq:
        emb = get_pos_emb_on_this_cp_rank(emb, 0, self.cp_group)

    return emb, _mscale
```

### Positional Encoding Configuration

```python
# Standard RoPE
config = TransformerConfig(
    rotary_percent=1.0,
    rotary_interleaved=False,
    apply_rope_fusion=True,  # Use fused kernel for 1.5-2x speedup
)

# RoPE with linear interpolation (2x context extension)
config = TransformerConfig(
    rotary_percent=1.0,
    seq_len_interpolation_factor=2.0,
)

# RoPE with LLaMA 3.x scaling (8x context extension)
config = TransformerConfig(
    rotary_percent=1.0,
    rope_scaling=True,
    rope_scaling_factor=8.0,
)

# YARN for superior context extension
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import YarnRotaryEmbedding

yarn_rope = YarnRotaryEmbedding(
    kv_channels=128,
    rotary_percent=1.0,
    scaling_factor=4.0,  # 4x context extension
    original_max_position_embeddings=4096,
    beta_fast=32.0,
    beta_slow=1.0,
    mscale=1.0,
)
```

### Comparison of RoPE Variants

| Method | Context Extension | Quality | Compute Overhead |
|--------|------------------|---------|------------------|
| **Linear Interpolation** | 2-4x | Good | None |
| **LLaMA 3.x Scaling** | 4-8x | Very Good | None |
| **YARN** | 4-16x | Excellent | Minimal |

---

## 4. Flash Attention Integration

### The Problem with Standard Attention

Standard attention has O(N²) memory complexity due to materializing the full attention matrix:

```
Standard Attention Memory:
- Attention scores: [batch, heads, seq_len, seq_len]
- For seq_len=32K, heads=32: 32 × 32 × 32768 × 32768 × 2 bytes = 128 GB per batch!
```

This makes long-context training impossible without specialized algorithms.

### Flash Attention Solution

Flash Attention computes attention in a memory-efficient, tile-based manner:
- **O(N) memory complexity** instead of O(N²)
- **IO-aware algorithm** that minimizes HBM accesses
- **Kernel fusion** combining Q×K, softmax, and ×V operations

### Implementation in Megatron-LM

Flash Attention integration is in `megatron/core/transformer/attention.py`:

```python
# Flash Attention 3 support (lines 51-69)
try:
    from flash_attn_3.flash_attn_interface import _flash_attn_forward
    from flash_attn_3.flash_attn_interface import (
        flash_attn_with_kvcache as flash_attn3_with_kvcache,
    )
    HAVE_FA3 = True
except ImportError:
    HAVE_FA3 = False

# Fallback to Hopper-specific implementation
if not HAVE_FA3:
    try:
        from flashattn_hopper.flash_attn_interface import _flash_attn_forward
        from flashattn_hopper.flash_attn_interface import (
            flash_attn_with_kvcache as flash_attn3_with_kvcache,
        )
        HAVE_FA3 = True
    except ImportError:
        pass

# Flash Attention 2 variable-length support
try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
except:
    flash_attn_varlen_func = None
    flash_attn_with_kvcache = None
```

### Fused QKV RoPE

A key optimization for long sequences combines RoPE application with QKV operations:

```python
# Fused QKV RoPE support (lines 102-106)
try:
    from transformer_engine.pytorch.attention.rope import apply_fused_qkv_rotary_pos_emb
    HAVE_FUSED_QKV_ROPE = True
except ImportError:
    HAVE_FUSED_QKV_ROPE = False
```

Benefits:
- Avoids splitting QKV tensor before RoPE
- Avoids concatenating gradients after RoPE backward
- Reduces kernel launches and memory traffic by ~30%

### Flash MLA (Multi-Latent Attention)

For models using MLA (Multi-Latent Attention), specialized kernels provide additional efficiency:

```python
# Flash MLA support (lines 72-78)
try:
    from flash_mla import flash_mla_with_kvcache, get_mla_metadata
    HAVE_FMLA = True
except ImportError:
    flash_mla_with_kvcache = None
    get_mla_metadata = None
    HAVE_FMLA = False
```

### Memory Comparison

For a 32K sequence with 32 attention heads and head_dim=128:

| Method | Attention Memory | Speedup |
|--------|-----------------|---------|
| **Standard** | 128 GB | 1.0x |
| **Flash Attention 2** | 64 MB | 2-3x |
| **Flash Attention 3** | 64 MB | 3-4x |

Flash Attention reduces attention memory from O(N²) to O(N), making 32K+ sequences feasible.

### Configuration

```python
config = TransformerConfig(
    # Enable fused RoPE with Flash Attention
    apply_rope_fusion=True,
    fused_single_qkv_rope=True,

    # Attention configuration
    attention_softmax_in_fp32=True,  # Numerical stability
)
```

---

## 5. Variable Sequence Length Handling

### The Padding Problem

When batching sequences of different lengths, traditional approaches pad all sequences to the maximum length:

```
Batch with padding:
  Sequence 1: [tok, tok, tok, tok, PAD, PAD, PAD, PAD]  # 50% wasted
  Sequence 2: [tok, tok, tok, tok, tok, tok, tok, tok]
  Sequence 3: [tok, tok, PAD, PAD, PAD, PAD, PAD, PAD]  # 75% wasted

Average padding waste: ~40% of compute and memory
```

For long sequences, this waste becomes prohibitive.

### Packed Sequence Format (THD)

Megatron-LM supports packed sequences that concatenate variable-length sequences without padding.

#### PackedSeqParams Data Structure

Defined in `megatron/core/packed_seq_params.py`:

```python
@dataclass
class PackedSeqParams:
    """Parameters for THD (packed) sequence format attention."""

    qkv_format: str = None
    cu_seqlens_q: Tensor = None       # Cumulative query sequence lengths
    cu_seqlens_kv: Tensor = None      # Cumulative KV sequence lengths
    cu_seqlens_q_padded: Tensor = None
    cu_seqlens_kv_padded: Tensor = None
    max_seqlen_q: int = None          # Maximum query sequence length
    max_seqlen_kv: int = None         # Maximum KV sequence length
```

#### How Packed Sequences Work

```
Traditional (BSHD) format - with padding:
  Batch 0: [tok0, tok1, tok2, PAD, PAD, PAD, PAD, PAD]
  Batch 1: [tok0, tok1, tok2, tok3, tok4, tok5, tok6, tok7]
  Batch 2: [tok0, tok1, PAD, PAD, PAD, PAD, PAD, PAD]
  Shape: [3, 8, heads, dim]

Packed (THD) format - no padding:
  All tokens: [tok0, tok1, tok2, tok0, tok1, tok2, tok3, tok4, tok5, tok6, tok7, tok0, tok1]
  cu_seqlens: [0, 3, 11, 13]  # Cumulative lengths
  Shape: [13, heads, dim]  # Total tokens only
```

#### Sequence Format Comparison

| Format | Shape | Description | Use Case |
|--------|-------|-------------|----------|
| **BSHD** | [batch, seq, heads, dim] | Padded, batch-first | Standard training |
| **SBHD** | [seq, batch, heads, dim] | Padded, sequence-first | TP-friendly |
| **THD** | [total, heads, dim] | Packed, no padding | Variable lengths |

### RoPE with Packed Sequences

The RoPE implementation handles packed sequences correctly through the `get_rotary_seq_len` method:

```python
def get_rotary_seq_len(
    self,
    inference_context: BaseInferenceContext,
    transformer: TransformerBlock,
    transformer_input: Tensor,
    transformer_config: TransformerConfig,
    packed_seq_params: Optional[PackedSeqParams] = None,
) -> int:
    """Determine rotary sequence length accounting for packing."""

    if packed_seq_params is not None:
        # For packed sequences, use max sequence length in batch
        return max(packed_seq_params.max_seqlen_q, packed_seq_params.max_seqlen_kv)

    elif inference_context is not None:
        rotary_seq_len = inference_context.max_sequence_length

    else:
        if transformer is not None and transformer.input_tensor is not None:
            rotary_seq_len = transformer.input_tensor.size(0)
        else:
            rotary_seq_len = transformer_input.size(0)

        # Account for sequence parallelism
        if transformer_config.sequence_parallel:
            rotary_seq_len *= transformer_config.tensor_model_parallel_size

    # Account for context parallelism
    rotary_seq_len *= transformer_config.context_parallel_size

    return rotary_seq_len
```

### Memory Savings with Packing

For a batch with sequences of lengths [1024, 8192, 2048, 4096]:

```
Padded approach:
  Memory = 4 × 8192 × hidden × 2 bytes = 4 × max_len
  Wasted = (8192-1024) + (8192-2048) + (8192-4096) = 17408 tokens (53% waste)

Packed approach:
  Memory = (1024 + 8192 + 2048 + 4096) × hidden × 2 bytes = sum(lengths)
  Wasted = 0 tokens
```

**Memory savings: 53% for this example batch**

### Integration with Flash Attention

Packed sequences require Flash Attention's variable-length support:

```python
# Using flash_attn_varlen_func for packed sequences
from flash_attn import flash_attn_varlen_func

output = flash_attn_varlen_func(
    q=query,           # [total_q, heads, dim]
    k=key,             # [total_kv, heads, dim]
    v=value,           # [total_kv, heads, dim]
    cu_seqlens_q=packed_seq_params.cu_seqlens_q,
    cu_seqlens_k=packed_seq_params.cu_seqlens_kv,
    max_seqlen_q=packed_seq_params.max_seqlen_q,
    max_seqlen_k=packed_seq_params.max_seqlen_kv,
    causal=True,
)
```

---

## 6. Activation Checkpointing for Long Contexts

### The Memory-Compute Tradeoff

Long sequences dramatically increase activation memory. Activation checkpointing trades compute for memory by recomputing activations during the backward pass instead of storing them.

### Configuration Options

Defined in `megatron/core/transformer/transformer_config.py` (lines 298-339):

```python
@dataclass
class TransformerConfig:
    recompute_granularity: Optional[str] = None
    """Activation recompute strategy:
    - 'selective': Checkpoint specific submodules (default: core_attn)
    - 'full': Checkpoint entire transformer layer
    - None: No recomputation, store all activations
    """

    recompute_method: Optional[str] = None
    """Layer selection for recomputation:
    - 'uniform': Evenly divide layers into recompute chunks
    - 'block': Recompute first N layers per pipeline stage
    - None: Recompute all layers (when recompute_granularity is set)
    """

    recompute_num_layers: Optional[int] = None
    """Number of layers to recompute:
    - For 'uniform': Layers per recompute chunk
    - For 'block': Number of layers to recompute per stage
    """

    distribute_saved_activations: Optional[bool] = None
    """Distribute recomputed activations across model parallel group."""

    recompute_modules: Optional[List[str]] = None
    """Submodules to recompute when using 'selective' granularity:
    - 'core_attn': Core attention (most memory-intensive, recommended)
    - 'moe_act': MoE activation function
    - 'layernorm': Input and pre-MLP layer normalization
    - 'mla_up_proj': MLA up projection and RoPE
    - 'mlp': Dense MLP submodule
    - 'moe': Entire MoE layer
    - 'shared_experts': Shared experts in MoE
    Default: ['core_attn']
    """
```

### Selective vs Full Recomputation

#### Selective Recomputation (Recommended for Long Contexts)

Targets the most memory-intensive but compute-efficient operations:

```python
config = TransformerConfig(
    recompute_granularity="selective",
    recompute_modules=["core_attn"],  # Only recompute attention
)
```

**Why attention?**
- Attention activations are large: O(batch × heads × seq² × dim)
- Attention computation is memory-bound, not compute-bound
- Recomputing attention adds minimal overhead (~10-15%)

#### Full Recomputation

Checkpoints entire transformer layers:

```python
config = TransformerConfig(
    recompute_granularity="full",
    recompute_method="block",
    recompute_num_layers=4,  # Recompute 4 layers per pipeline stage
)
```

### Memory Savings Analysis

For a 32K sequence, 80-layer model:

| Strategy | Activation Memory | Compute Overhead |
|----------|------------------|------------------|
| **None (baseline)** | 100% | 0% |
| **Selective (core_attn)** | 40-50% | 10-15% |
| **Full (all layers)** | 10-15% | 30-40% |
| **Full + Distributed** | 5-10% | 35-45% |

### Combining with Other Optimizations

For maximum long-context capability, combine all strategies:

```python
config = TransformerConfig(
    # Context Parallelism
    context_parallel_size=4,
    cp_comm_type="p2p",
    overlap_p2p_comm=True,

    # Sequence Parallelism
    sequence_parallel=True,
    tensor_model_parallel_size=4,

    # Selective Recomputation
    recompute_granularity="selective",
    recompute_modules=["core_attn"],
    distribute_saved_activations=True,
)
```

Combined memory reduction:
- CP=4: 75% reduction
- SP with TP=4: Additional 75% reduction
- Selective recompute: Additional 50% reduction
- **Total: ~97% activation memory reduction**

---

## 7. Complete Configuration Example

### Production Configuration for 128K Context Training

```python
from megatron.core.transformer import TransformerConfig

config = TransformerConfig(
    # Model Architecture
    num_layers=80,
    hidden_size=8192,
    num_attention_heads=64,
    num_query_groups=8,          # Group Query Attention for efficiency
    ffn_hidden_size=28672,
    kv_channels=128,

    # Parallelism Configuration
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    context_parallel_size=8,      # Key for 128K sequences
    sequence_parallel=True,        # Required with TP + CP

    # Context Parallelism Settings
    cp_comm_type="p2p",           # Ring attention for NVLink
    overlap_p2p_comm=True,        # Async overlap
    batch_p2p_comm=True,

    # For multi-tier networks (optional)
    # hierarchical_context_parallel_sizes=[4, 2],  # 4-way A2A, 2-way P2P
    # cp_comm_type="a2a+p2p",

    # Positional Encoding
    rotary_percent=1.0,
    rotary_interleaved=False,
    apply_rope_fusion=True,       # 1.5-2x RoPE speedup
    # seq_len_interpolation_factor=4.0,  # For context extension

    # Memory Optimizations
    recompute_granularity="selective",
    recompute_modules=["core_attn"],
    distribute_saved_activations=True,

    # Precision
    bf16=True,
    attention_softmax_in_fp32=True,
)
```

### Command-Line Arguments

```bash
python train.py \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 4 \
    --context-parallel-size 8 \
    --sequence-parallel \
    --cp-comm-type p2p \
    --overlap-p2p-comm \
    --batch-p2p-comm \
    --recompute-granularity selective \
    --recompute-modules core_attn \
    --apply-rope-fusion \
    --seq-length 131072 \
    --bf16
```

---

## 8. Performance Summary

### Optimization Impact Matrix

| Optimization | Memory Saved | Compute Overhead | Communication Overhead |
|--------------|-------------|------------------|----------------------|
| **Context Parallelism (CP=8)** | 87.5% activations | 0% | 15-20% |
| **Sequence Parallelism (TP=4)** | 75% activations | 0% | 5-10% |
| **Flash Attention** | O(N²) → O(N) | -40% (speedup!) | 0% |
| **Selective Recompute** | 50-60% activations | 10-15% | 0% |
| **Packed Sequences** | 15-40% (varies) | 0% | 0% |
| **RoPE Fusion** | 0% | -30% (speedup!) | 0% |
| **YARN Scaling** | 0% | 0% | 0% |

### Recommended Configurations by Context Length

| Context Length | Recommended Configuration |
|----------------|--------------------------|
| **4K-8K** | Flash Attention + Selective Recompute |
| **8K-32K** | + Context Parallel (CP=2-4) + Sequence Parallel |
| **32K-64K** | + Context Parallel (CP=4-8) + RoPE Interpolation |
| **64K-128K** | + Context Parallel (CP=8) + YARN + Full Optimization Stack |
| **128K+** | + Hierarchical CP (A2A+P2P) + Pipeline Parallel |

### Hardware Requirements

For 128K context training on 70B parameter model:

| Configuration | GPUs Required | Per-GPU Memory |
|---------------|--------------|----------------|
| **Baseline** | Infeasible | >1TB |
| **+ Flash Attention** | 64 × A100 80GB | ~75GB |
| **+ CP=8** | 64 × A100 80GB | ~45GB |
| **+ All Optimizations** | 32 × A100 80GB | ~60GB |

---

## 9. Troubleshooting

### Common Issues and Solutions

**Issue: OOM with long sequences**
```
Solution: Enable context parallelism and/or increase CP size
--context-parallel-size 4  # or 8
```

**Issue: Sequence length not divisible by CP × TP**
```
Error: seq_length must be divisible by context_parallel_size × tensor_model_parallel_size
Solution: Pad sequence length to nearest multiple
padded_len = math.ceil(seq_len / (cp_size * tp_size)) * (cp_size * tp_size)
```

**Issue: Slow ring attention communication**
```
Solution:
1. Use P2P for NVLink-connected GPUs
2. Use A2A for InfiniBand networks
3. Enable overlap: --overlap-p2p-comm
```

**Issue: Numerical instability with very long sequences**
```
Solution:
1. Use attention_softmax_in_fp32=True
2. Consider YARN over linear RoPE interpolation
3. Reduce learning rate for long-context fine-tuning
```

---

## 10. References

### Implementation Files

| Component | File Path |
|-----------|-----------|
| Context Parallelism | `megatron/core/parallel_state.py` |
| CP Communication | `megatron/core/extensions/transformer_engine.py:913-937` |
| RoPE | `megatron/core/models/common/embeddings/rotary_pos_embedding.py` |
| YARN | `megatron/core/models/common/embeddings/yarn_rotary_pos_embedding.py` |
| Flash Attention | `megatron/core/transformer/attention.py:45-86` |
| Packed Sequences | `megatron/core/packed_seq_params.py` |
| Recomputation Config | `megatron/core/transformer/transformer_config.py:298-339` |
| CP Config | `megatron/core/model_parallel_config.py:45-54` |

### Related Reports

- **#03 Sequence Parallel Communication**: Reduce-scatter for sequence parallelism
- **#15 Context Parallelism**: Detailed CP architecture
- **#18 Sequence Parallelism**: SP integration with attention
- **#34 Fused RoPE**: RoPE kernel optimization

### External References

- Ring Attention Paper: Liu et al., "Ring Attention with Blockwise Transformers"
- Flash Attention: Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention"
- YARN: Peng et al., "YaRN: Efficient Context Window Extension of Large Language Models"
- RoPE: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"

