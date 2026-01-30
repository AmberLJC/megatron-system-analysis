# 15. Context Parallelism

## Context

Context Parallelism (CP) is a sophisticated parallelization strategy that splits the sequence dimension across multiple GPUs to enable training with extremely long sequences. Unlike standard sequence parallelism which targets specific attention operations, Context Parallelism provides a comprehensive framework supporting multiple communication patterns (point-to-point, all-gather, all-to-all) and works across both Transformer attention layers and SSM (Mamba) architectures. This feature becomes critical when training models with context lengths exceeding 32K tokens, where activation memory consumption becomes the primary bottleneck.

## Implementation Architecture

Megatron-LM implements Context Parallelism through a multi-layered architecture consisting of parallel group initialization, configuration management, attention mechanism integration, and SSM-specific utilities. The implementation resides primarily in megatron/core with extensions for Transformer Engine integration.

### Process Group Initialization

The core CP group initialization happens in `megatron/core/parallel_state.py` where several global process groups are created during `initialize_model_parallel()`:

```python
# Global CP group variables (lines 106-127)
_CONTEXT_PARALLEL_GROUP = None
_CONTEXT_PARALLEL_GLOBAL_RANKS = None
_HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = None
_DATA_PARALLEL_GROUP_WITH_CP = None
_TENSOR_AND_CONTEXT_PARALLEL_GROUP = None

# Getter functions providing access to CP groups
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

def get_hierarchical_context_parallel_groups(check_initialized=True):
    """Get the hierarchical context parallel groups for a2a+p2p communication."""
    if check_initialized:
        assert _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS is not None
    return _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS
```

The initialization process creates CP groups using a `RankGenerator` that computes rank mappings based on the parallelism order (typically "tp-cp-ep-dp-pp"). CP groups are created with configurable NCCL options specified in `cp_comm_cfgs` for performance tuning.

### Configuration Parameters

Context Parallelism configuration spans multiple config classes to provide flexibility:

```python
# In megatron/core/model_parallel_config.py (lines 45-54)
@dataclass
class ModelParallelConfig:
    context_parallel_size: int = 1
    """Splits network input along sequence dimension across GPU ranks.
    When > 1, each GPU processes seq_length/context_parallel_size tokens."""

    hierarchical_context_parallel_sizes: Optional[list[int]] = None
    """Degrees of hierarchical context parallelism.
    For a2a+p2p mode: [a2a_group_size, p2p_group_size]
    Enables multi-level communication optimization."""

# In megatron/core/transformer/transformer_config.py (lines 594-608)
@dataclass
class TransformerConfig(ModelParallelConfig):
    cp_comm_type: Optional[Union[str, List[str]]] = None
    """Inter-GPU communication type for context parallelism.
    Options:
    - "p2p": Point-to-point ring attention (lowest latency)
    - "all_gather": Gather full KV before attention (simplest)
    - "a2a": All-to-all, DeepSpeed Ulysses style (best bandwidth)
    - "a2a+p2p": Hierarchical with both (multi-level networks)
    Can specify per-layer as list: ["p2p", "a2a", "p2p", ...]
    """

    # P2P-specific configuration
    overlap_p2p_comm: bool = False
    """Overlap P2P communication with attention computation."""

    batch_p2p_comm: bool = True
    """Batch multiple P2P operations together."""

    batch_p2p_sync: bool = True
    """Synchronize batched P2P operations."""

    use_ring_exchange_p2p: bool = False
    """Use ring exchange pattern for P2P."""
```

### Attention Implementation with Context Parallelism

The `TEDotProductAttention` class in `megatron/core/extensions/transformer_engine.py` provides the primary CP-enabled attention mechanism, integrating seamlessly with NVIDIA Transformer Engine:

```python
class TEDotProductAttention(torch.nn.Module):
    """
    Transformer Engine Dot Product Attention with Context Parallelism support.
    Supports multiple CP communication patterns and asynchronous execution.
    """

    # Class-level CUDA stream for async CP communication
    cp_stream = None

    def __init__(
        self,
        config: TransformerConfig,
        layer_number: int,
        attn_mask_type: AttnMaskType,
        attention_type: str,
        attention_dropout: Optional[float] = None,
        softmax_scale: Optional[float] = None,
        k_channels: Optional[int] = None,
        v_channels: Optional[int] = None,
        cp_comm_type: str = "p2p",
        pg_collection: ProcessGroupCollection = None,
    ):
        super().__init__()
        self.config = config
        self.cp_comm_type = cp_comm_type

        # Prepare TE attention kwargs
        extra_kwargs = {}

        # CP-specific setup (lines 913-937)
        if self.config.context_parallel_size > 1:
            # Requires Transformer-Engine >= 1.0.0 for CP support
            extra_kwargs["cp_group"] = pg_collection.cp
            extra_kwargs["cp_global_ranks"] = torch.distributed.get_process_group_ranks(
                pg_collection.cp
            )
            # Use dedicated stream for async communication
            extra_kwargs["cp_stream"] = TEDotProductAttention.cp_stream

            # Configure communication pattern based on TE version
            if is_te_min_version("1.10.0"):
                if cp_comm_type == "a2a+p2p":
                    # Hierarchical mode: use multi-level groups
                    extra_kwargs["cp_comm_type"] = "a2a+p2p"
                    extra_kwargs["cp_group"] = get_hierarchical_context_parallel_groups()
                else:
                    # Single-level mode: p2p, all_gather, or a2a
                    extra_kwargs["cp_comm_type"] = cp_comm_type

        # Initialize Transformer Engine attention with CP config
        self.te_core_attention = te.pytorch.DotProductAttention(
            config.num_attention_heads,
            config.kv_channels,
            attention_dropout=attention_dropout,
            attn_mask_type=attn_mask_type.name,
            **extra_kwargs
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass with CP communication.

        With CP enabled:
        1. Query tensors are local (seq_len/cp_size per rank)
        2. KV tensors are communicated via cp_comm_type pattern
        3. Attention computed over full sequence
        4. Output local (seq_len/cp_size per rank)
        """
        # TE handles all CP communication internally based on cp_comm_type
        output = self.te_core_attention(
            query, key, value, attention_mask
        )
        return output
```

The beauty of this integration is that Transformer Engine handles all the complex communication patterns internally. When `cp_comm_type="p2p"`, it performs ring attention with asynchronous KV exchange on a dedicated CUDA stream, enabling overlap with attention computation. When using `"all_gather"`, it performs a synchronous gather operation. The `"a2a"` mode implements DeepSpeed Ulysses-style all-to-all communication that scatters attention heads across CP ranks.

### SSM Context Parallelism for Mamba

SSM models like Mamba require a fundamentally different CP approach because they don't have the key/value structure of attention. The implementation in `megatron/core/ssm/mamba_context_parallel.py` provides specialized utilities:

```python
class MambaContextParallel:
    """
    Provides context parallelism functionality for Mamba/SSM layers.
    Uses all-to-all communication to rearrange sequence and feature dimensions.

    Key difference from attention CP: No KV gathering needed.
    Instead, rearranges [seq//cp, batch, features] ↔ [seq, batch, features//cp]
    """

    def __init__(
        self,
        cp_group: torch.distributed.ProcessGroup,
        d_inner_local_tp: int,
        nheads_local_tp: int,
        ngroups_local_tp: int,
        d_state: int,
        conv1d_cp1: nn.Conv1d,
        dt_bias_cp1: torch.Tensor,
        A_log_cp1: torch.Tensor,
        D_cp1: torch.Tensor,
        D_has_hdim: bool,
    ):
        """
        Initialize Mamba CP with local slices of parameters.

        Args:
            cp_group: Context parallel process group
            d_inner_local_tp: Inner dimension after tensor parallelism
            nheads_local_tp: Number of attention heads (local)
            ngroups_local_tp: Number of groups (local)
            d_state: SSM state dimension
            conv1d_cp1: 1D convolution layer
            dt_bias_cp1, A_log_cp1, D_cp1: SSM parameters
        """
        self.cp_group = cp_group
        self.cp_size = torch.distributed.get_world_size(cp_group)
        self.cp_rank = torch.distributed.get_rank(cp_group)

        # Store layer components
        self.conv1d_cp1 = conv1d_cp1
        self.dt_bias_cp1 = dt_bias_cp1
        self.A_log_cp1 = A_log_cp1
        self.D_cp1 = D_cp1

        # Compute dimensions
        self.d_inner_local_tp = d_inner_local_tp
        self.nheads_local_tp = nheads_local_tp
        self.ngroups_local_tp = ngroups_local_tp
        self.d_state = d_state
        self.D_has_hdim = D_has_hdim

    def pre_conv_ssm(self, input_):
        """
        Prepare input before convolution and SSM operations.

        Transforms from CP layout to hidden-parallel layout:
        [seq//cp, batch, features] → [seq, batch, features//cp]

        This enables each rank to process the full sequence but only
        a slice of the feature dimension, which is more suitable for
        depthwise convolution operations.
        """
        # Split input along feature dimension for all-to-all
        # Each rank gets features // cp_size
        input_split = torch.split(
            input_,
            input_.shape[-1] // self.cp_size,
            dim=-1
        )

        # All-to-all: exchange sequence chunks for feature slices
        input_hp = _all_to_all_cp2hp(input_, self.cp_group)
        # Now: [seq, batch, features//cp]

        # Undo attention load balancing for sequential processing
        input_hp = _undo_attention_load_balancing(input_hp, self.cp_size)

        return input_hp

    def post_conv_ssm(self, input_):
        """
        Transform output back after conv/SSM operations.

        Reverse of pre_conv_ssm:
        [seq, batch, features//cp] → [seq//cp, batch, features]
        """
        # Redo attention load balancing
        input_hp = _redo_attention_load_balancing(input_, self.cp_size)

        # All-to-all: exchange feature slices for sequence chunks
        output = _all_to_all_hp2cp(input_hp, self.cp_group)
        # Now: [seq//cp, batch, features]

        return output

    def conv1d(self, input_):
        """
        Execute depthwise 1D convolution on local slice.
        Each rank processes full sequence but partial features.
        """
        return self.conv1d_cp1(input_)

# Helper functions for all-to-all communication patterns

def _all_to_all_cp2hp(input_, cp_group):
    """
    All-to-all from context-parallel to hidden-parallel layout.
    Input:  [seq//cp, batch, hidden]
    Output: [seq, batch, hidden//cp]
    """
    cp_size = torch.distributed.get_world_size(cp_group)
    seq_len_cp, batch_size, hidden = input_.shape
    seq_len = seq_len_cp * cp_size

    # Reshape for all-to-all: [seq//cp, batch, cp, hidden//cp]
    input_reshaped = input_.reshape(seq_len_cp, batch_size, cp_size, hidden // cp_size)

    # Transpose to group by target rank: [cp, seq//cp, batch, hidden//cp]
    input_transposed = input_reshaped.permute(2, 0, 1, 3).contiguous()

    # All-to-all exchange
    output_transposed = torch.empty_like(input_transposed)
    torch.distributed.all_to_all_single(
        output_transposed, input_transposed, group=cp_group
    )

    # Reshape to final form: [seq, batch, hidden//cp]
    output = output_transposed.permute(1, 0, 2, 3).contiguous()
    output = output.reshape(seq_len, batch_size, hidden // cp_size)

    return output

def _all_to_all_hp2cp(input_, cp_group):
    """
    All-to-all from hidden-parallel to context-parallel layout.
    Input:  [seq, batch, hidden//cp]
    Output: [seq//cp, batch, hidden]
    """
    # Inverse of _all_to_all_cp2hp
    cp_size = torch.distributed.get_world_size(cp_group)
    seq_len, batch_size, hidden_cp = input_.shape
    seq_len_cp = seq_len // cp_size
    hidden = hidden_cp * cp_size

    # Reshape: [seq//cp, cp, batch, hidden//cp]
    input_reshaped = input_.reshape(seq_len_cp, cp_size, batch_size, hidden_cp)

    # Transpose: [cp, seq//cp, batch, hidden//cp]
    input_transposed = input_reshaped.permute(1, 0, 2, 3).contiguous()

    # All-to-all exchange
    output_transposed = torch.empty_like(input_transposed)
    torch.distributed.all_to_all_single(
        output_transposed, input_transposed, group=cp_group
    )

    # Reshape to final: [seq//cp, batch, hidden]
    output = output_transposed.permute(1, 0, 2, 3).contiguous()
    output = output.reshape(seq_len_cp, batch_size, hidden)

    return output
```

### Communication Patterns Comparison

Megatron-LM supports four distinct CP communication patterns, each optimized for different network topologies:

**1. Point-to-Point (P2P) Ring Attention:**
- Implements ring-based KV exchange between consecutive CP ranks
- Asynchronous execution on dedicated CUDA stream enables overlap with attention
- Lowest latency for high-bandwidth interconnects like NVLink
- Best for small CP degrees (2-4) with fast node-local communication

**2. All-Gather:**
- Performs synchronous all-gather of full KV sequence before attention
- Simplest implementation but highest communication volume
- No overlap capability with computation
- Useful for debugging or when overlap doesn't help

**3. All-to-All (A2A):**
- Inspired by DeepSpeed Ulysses approach
- Scatters attention heads across CP group
- Each rank computes full QKV for local head subset
- Better bandwidth utilization than all-gather
- Final all-gather reconstructs full sequence output

**4. Hierarchical A2A+P2P:**
- Uses all-to-all within dense, high-bandwidth groups (NVLink)
- Uses P2P between sparse, lower-bandwidth groups (InfiniBand)
- Requires `hierarchical_context_parallel_sizes` configuration
- Optimal for multi-level network topologies

### Integration with Other Parallelism Strategies

Context Parallelism integrates orthogonally with other parallelism dimensions. The process group configuration in `megatron/core/process_groups_config.py` manages these interactions:

```python
@dataclass
class ProcessGroupCollection:
    """
    Container for all parallel process groups.
    Provides unified interface for model components.
    """
    # Context Parallelism groups
    cp: torch.distributed.ProcessGroup
    """Context parallel group for sequence splitting."""

    tp_cp: torch.distributed.ProcessGroup
    """Combined tensor and context parallel group."""

    hcp: List[torch.distributed.ProcessGroup]
    """Hierarchical context parallel groups for a2a+p2p."""

    dp_cp: torch.distributed.ProcessGroup
    """Data parallel group including context parallel dimension.
    Used for gradient all-reduce since CP duplicates weights."""

    intra_dp_cp: torch.distributed.ProcessGroup
    """Partial data parallel with CP for distributed optimizer."""
```

The key insight for multi-dimensional parallelism with CP is that **CP duplicates model weights** across the CP group (similar to DP), but **splits activations** along the sequence dimension. Therefore:

- **With Tensor Parallelism (TP):** CP and TP are orthogonal. Total model parallelism = TP × CP. The combined `tp_cp` group is used for FP8 amax reduction operations.

- **With Pipeline Parallelism (PP):** CP operates independently within each pipeline stage. Each stage has its own CP groups with no cross-stage CP communication.

- **With Data Parallelism (DP):** Weight gradients must be all-reduced across the `dp_cp` group (size = DP × CP) since weights are duplicated. Activation gradients use CP-specific communication.

- **With Expert Parallelism (EP):** Currently, CP and EP cannot be combined in the same rank generator due to complexity, though they can exist in separate decoder/expert generators.

## Performance Impact

Context Parallelism provides substantial memory savings enabling much longer sequence training:

**Activation Memory Reduction:**
- Activations reduced by factor of CP size
- Each rank stores only `seq_length / cp_size` activations
- Enables 4× longer sequences with CP=4

**Communication Overhead:**
- P2P mode: 2 × (CP-1) / CP × message size (ring communication)
- All-gather mode: (CP-1) / CP × message size (gather operation)
- A2A mode: Depends on head distribution, generally efficient
- Overlap capability in P2P mode can hide most communication

**Example:** GPT-3 175B with 8K context, CP=4
- Per-rank sequence: 8192 / 4 = 2048 tokens
- Activation memory saved: 3/4 = 75%
- Enables training with 32K context using same memory

## When to Use

**Enable Context Parallelism when:**
- Training with long sequences (>8K tokens for attention, >4K for Mamba)
- Activation memory is the bottleneck (check with profiler)
- Have sufficient inter-node or inter-GPU bandwidth
- Can afford small communication overhead for large memory savings

**CP Configuration Guidelines:**
- Start with CP=2, increase to 4 or 8 for very long sequences
- Use P2P mode for NVLink-connected GPUs
- Use A2A mode for larger CP degrees or IB networks
- Use hierarchical A2A+P2P for multi-level network topologies
- Always enable `sequence_parallel=True` when using CP with TP

**Avoid Context Parallelism when:**
- Sequences are short (<4K tokens) - overhead exceeds benefit
- Network bandwidth is limited (1GbE Ethernet)
- Batch size is very small (less than CP size)
- Memory is not the primary constraint

## Training Arguments

```bash
# Basic CP configuration
--context-parallel-size 4

# Hierarchical CP for multi-level networks
--hierarchical-context-parallel-sizes 2 2  # [level1_size, level2_size]

# Communication pattern selection
--cp-comm-type p2p  # or all_gather, a2a, a2a+p2p

# P2P optimizations
--overlap-p2p-comm  # Enable async overlap
--use-ring-exchange-p2p  # Ring exchange pattern

# Must enable with tensor parallelism
--sequence-parallel  # Required with TP + CP
```

## Complete Configuration Example

```python
from megatron.core.transformer import TransformerConfig

# Configure transformer with context parallelism
config = TransformerConfig(
    # Model dimensions
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,

    # Parallelism configuration
    tensor_model_parallel_size=2,
    pipeline_model_parallel_size=2,
    context_parallel_size=4,

    # CP communication settings
    cp_comm_type="p2p",  # Ring attention
    overlap_p2p_comm=True,  # Async overlap
    batch_p2p_comm=True,

    # Must enable with TP + CP
    sequence_parallel=True,

    # Memory optimizations
    recompute_granularity="selective",
    recompute_method="block",
)

# Total GPUs: TP × PP × CP × DP = 2 × 2 × 4 × DP
# For 64 GPUs: DP = 64 / (2 × 2 × 4) = 4
```

## References

- Megatron-LM Context Parallel documentation: `docs/source/api-guide/context_parallel.rst`
- Ring Attention paper: Liu et al., "Ring Attention with Blockwise Transformers"
- DeepSpeed Ulysses: Microsoft DeepSpeed documentation
- Implementation: `megatron/core/extensions/transformer_engine.py`, `megatron/core/ssm/mamba_context_parallel.py`

