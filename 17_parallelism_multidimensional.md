# 17. Multi-Dimensional Parallelism

## Context

Multi-dimensional parallelism is the cornerstone of large-scale distributed training, enabling models with hundreds of billions of parameters to be trained across thousands of GPUs. Megatron-LM implements a sophisticated orchestration system that combines Tensor Parallelism (TP), Pipeline Parallelism (PP), Data Parallelism (DP), Context Parallelism (CP), and Expert Parallelism (EP) into cohesive training configurations. This system provides 3D parallelism (TP×PP×DP), 4D parallelism (TP×PP×DP×CP or TP×PP×DP×EP), and even 5D parallelism (TP×PP×DP×CP×EP) support. The key innovation lies in the careful orchestration of process groups, rank mappings, and communication patterns that ensure each parallelism strategy operates independently while maintaining consistency across dimensions.

## Architecture Overview

The multi-dimensional parallelism implementation centers around three core components: the `RankGenerator` class that computes rank mappings, the `initialize_model_parallel()` function that creates process groups, and the `ProcessGroupCollection` dataclass that provides unified access to all groups. This architecture enables deterministic rank-to-group mapping based on configurable ordering, ensuring reproducibility and flexibility in parallelism configurations.

## Core Implementation: RankGenerator

The `RankGenerator` class in `megatron/core/parallel_state.py` (lines 413-488) provides the mathematical foundation for multi-dimensional rank mapping:

```python
class RankGenerator(object):
    """
    A class for generating rank groups for different modes of parallelism.

    Computes rank-to-group mappings based on parallelism dimensions and ordering.
    Enables orthogonal communication groups for TP, PP, DP, CP, and EP.
    """

    def __init__(
        self,
        tp: int,   # Tensor parallel size
        ep: int,   # Expert parallel size
        dp: int,   # Data parallel size
        pp: int,   # Pipeline parallel size
        cp: int,   # Context parallel size
        order: str,  # Rank ordering like "tp-cp-ep-dp-pp"
        rank_offset: int = 0
    ) -> None:
        """
        Initialize rank generator with parallelism dimensions.

        Args:
            tp: Tensor model parallel size
            ep: Expert model parallel size
            dp: Data parallel size
            pp: Pipeline model parallel size
            cp: Context parallel size
            order: String specifying dimension ordering
            rank_offset: Offset for global rank calculation
        """
        self.tp = tp
        self.ep = ep
        self.dp = dp
        self.pp = pp
        self.cp = cp
        self.rank_offset = rank_offset
        self.world_size = tp * dp * pp * cp * ep

        # Parse and validate ordering
        self.order = order
        self.name_to_size = {
            "tp": self.tp,
            "pp": self.pp,
            "dp": self.dp,
            "ep": self.ep,
            "cp": self.cp,
        }

        # Validate: all dimensions > 1 must be in order
        self.dims = order.split("-")
        for name, size in self.name_to_size.items():
            if size > 1 and name not in self.dims:
                raise ValueError(
                    f"Dimension '{name}' with size {size} > 1 must be in order '{order}'"
                )

    def get_ranks(self, token: str, independent_ep: bool = False):
        """
        Get ranks for specified parallelism dimensions.

        Args:
            token: Dimension specification like "tp", "dp-cp", "tp-pp"
            independent_ep: Whether to compute EP ranks independently

        Returns:
            List of rank lists, where each inner list contains ranks
            forming one process group.

        Example:
            With 16 GPUs, TP=2, PP=2, DP=4, order="tp-pp-dp":
            get_ranks("tp") returns:
            [[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]]

            get_ranks("dp") returns:
            [[0,2,4,6], [1,3,5,7], [8,10,12,14], [9,11,13,15]]
        """
        # Implementation calls generate_masked_orthogonal_rank_groups()
        # with appropriate masks for requested dimensions
        return self._generate_ranks_for_token(token, independent_ep)
```

The `RankGenerator` computes rank groups using a sophisticated masking algorithm in `generate_masked_orthogonal_rank_groups()` (lines 242-348):

```python
def generate_masked_orthogonal_rank_groups(
    world_size: int,
    parallel_size: List[int],
    mask: List[bool],
    start_rank: int = 0,
) -> List[List[int]]:
    """
    Generate orthogonal rank groups based on masking.

    This is the core algorithm for computing rank-to-group mappings.
    Uses bit manipulation and modular arithmetic to generate groups.

    Args:
        world_size: Total number of ranks
        parallel_size: List of sizes for each dimension [tp, cp, ep, dp, pp]
        mask: Boolean mask indicating which dimensions to include
        start_rank: Starting rank offset

    Returns:
        List of rank groups, each group is a list of ranks

    Example:
        world_size=16, parallel_size=[2,1,1,2,4], mask=[True,False,False,True,False]
        Creates TP×DP groups: [[0,2], [1,3], [4,6], [5,7], ...]
    """

    # Compute group size from masked dimensions
    rank_group_size = 1
    for i, include in enumerate(mask):
        if include:
            rank_group_size *= parallel_size[i]

    # Number of groups
    num_groups = world_size // rank_group_size

    # Generate groups using mathematical mapping
    rank_groups = []
    for group_idx in range(num_groups):
        ranks_in_group = []
        for rank_in_group in range(rank_group_size):
            # Compute global rank from group structure
            global_rank = start_rank

            # Apply ordering based on masked dimensions
            remaining = rank_in_group
            remaining_group = group_idx

            for dim_idx, (size, include) in enumerate(zip(parallel_size, mask)):
                if include:
                    # This dimension contributes to group
                    coord = remaining % size
                    remaining //= size
                else:
                    # This dimension distinguishes groups
                    coord = remaining_group % size
                    remaining_group //= size

                # Accumulate global rank based on order
                global_rank += coord * compute_stride(parallel_size, dim_idx)

            ranks_in_group.append(global_rank)

        rank_groups.append(ranks_in_group)

    return rank_groups
```

## Process Group Initialization

The `initialize_model_parallel()` function in `megatron/core/parallel_state.py` (lines 615-1344) orchestrates the creation of all process groups:

```python
def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    context_parallel_size: int = 1,
    hierarchical_context_parallel_sizes: Optional[List[int]] = None,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: Optional[int] = None,
    num_distributed_optimizer_instances: int = 1,
    order: str = "tp-cp-ep-dp-pp",
    nccl_communicator_config_path: Optional[str] = None,
    distributed_timeout_minutes: int = 30,
    get_embedding_ranks: Optional[Callable] = None,
    get_position_embedding_ranks: Optional[Callable] = None,
    create_gloo_process_groups: bool = True,
    high_priority_stream_groups: Optional[List[str]] = None,
) -> None:
    """
    Initialize model parallel groups for multi-dimensional parallelism.

    This is the central initialization function that:
    1. Validates parallelism configuration
    2. Creates RankGenerators for decoder and expert layers
    3. Generates all process groups
    4. Stores groups in global variables

    Args:
        tensor_model_parallel_size: TP dimension
        pipeline_model_parallel_size: PP dimension
        virtual_pipeline_model_parallel_size: Virtual pipeline stages
        context_parallel_size: CP dimension
        hierarchical_context_parallel_sizes: Multi-level CP
        expert_model_parallel_size: EP dimension
        expert_tensor_parallel_size: TP for expert layers
        num_distributed_optimizer_instances: Optimizer sharding
        order: Dimension ordering string
        ... [other config parameters]
    """

    # Step 1: Validate world size decomposition (lines 695-700)
    world_size = torch.distributed.get_world_size()
    model_size = (
        tensor_model_parallel_size
        * pipeline_model_parallel_size
        * context_parallel_size
    )

    if world_size % model_size != 0:
        raise RuntimeError(
            f"world_size ({world_size}) is not divisible by "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) × "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size}) × "
            f"context_parallel_size ({context_parallel_size})"
        )

    # Compute data parallel size automatically
    data_parallel_size = world_size // model_size

    # Step 2: Create decoder RankGenerator (lines 720-730)
    decoder_rank_generator = RankGenerator(
        tp=tensor_model_parallel_size,
        ep=1,  # Standard layers don't use EP
        dp=data_parallel_size,
        pp=pipeline_model_parallel_size,
        cp=context_parallel_size,
        order=order,
    )

    # Step 3: Create expert RankGenerator if EP > 1 (lines 742-774)
    if expert_model_parallel_size > 1:
        expert_tensor_parallel_size = expert_tensor_parallel_size or tensor_model_parallel_size

        # Compute expert data parallel size
        expert_model_size = (
            expert_tensor_parallel_size
            * expert_model_parallel_size
            * pipeline_model_parallel_size
        )
        expert_data_parallel_size = world_size // expert_model_size

        expert_decoder_rank_generator = RankGenerator(
            tp=expert_tensor_parallel_size,
            ep=expert_model_parallel_size,
            dp=expert_data_parallel_size,
            pp=pipeline_model_parallel_size,
            cp=1,  # CP and EP cannot be combined
            order=order,
        )

        # Validate: PP groups must match between decoders
        assert (
            decoder_rank_generator.get_ranks("pp")
            == expert_decoder_rank_generator.get_ranks("pp")
        ), "Pipeline parallel groups must be identical for decoder and expert"

    # Step 4: Validate distributed optimizer sharding (lines 790-796)
    if num_distributed_optimizer_instances > 1:
        dp_cp_size = data_parallel_size * context_parallel_size
        assert dp_cp_size % num_distributed_optimizer_instances == 0, (
            f"DP×CP size ({dp_cp_size}) must be divisible by "
            f"num_distributed_optimizer_instances ({num_distributed_optimizer_instances})"
        )
        intra_partial_data_parallel_size = dp_cp_size // num_distributed_optimizer_instances

    # Step 5: Create Data Parallel groups (lines 807-890)
    global _DATA_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP_WITH_CP
    global _DATA_PARALLEL_GLOBAL_RANKS

    # DP with CP (for SHARP optimization)
    all_dp_cp_ranks = decoder_rank_generator.get_ranks('dp-cp')
    for ranks in all_dp_cp_ranks:
        group = torch.distributed.new_group(
            ranks,
            timeout=timeout,
            pg_options=nccl_options
        )
        if rank in ranks:
            _DATA_PARALLEL_GROUP_WITH_CP = group
            _DATA_PARALLEL_GLOBAL_RANKS = ranks

    # DP only (standard data parallelism)
    all_dp_ranks = decoder_rank_generator.get_ranks('dp')
    for ranks in all_dp_ranks:
        group = torch.distributed.new_group(ranks, timeout=timeout)
        if rank in ranks:
            _DATA_PARALLEL_GROUP = group

    # Step 6: Create Context Parallel groups (lines 902-929)
    global _CONTEXT_PARALLEL_GROUP
    global _CONTEXT_PARALLEL_GLOBAL_RANKS
    global _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS

    all_cp_ranks = decoder_rank_generator.get_ranks('cp')
    for ranks in all_cp_ranks:
        group = torch.distributed.new_group(
            ranks,
            pg_options=cp_comm_cfgs  # NCCL tuning options
        )
        if rank in ranks:
            _CONTEXT_PARALLEL_GROUP = group
            _CONTEXT_PARALLEL_GLOBAL_RANKS = ranks

    # Hierarchical CP for multi-level communication
    if hierarchical_context_parallel_sizes is not None:
        # Use einops to create nested groups
        _HIERARCHICAL_CONTEXT_PARALLEL_GROUPS = create_hierarchical_groups(
            _CONTEXT_PARALLEL_GLOBAL_RANKS,
            hierarchical_context_parallel_sizes
        )

    # Step 7: Create Tensor Parallel groups (lines 935-970)
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS

    all_tp_ranks = decoder_rank_generator.get_ranks('tp')
    for ranks in all_tp_ranks:
        group = torch.distributed.new_group(ranks, timeout=timeout)
        if rank in ranks:
            _TENSOR_MODEL_PARALLEL_GROUP = group
            _TENSOR_MODEL_PARALLEL_GLOBAL_RANKS = ranks

    # Step 8: Create Pipeline Parallel groups (lines 1028-1044)
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _PIPELINE_GLOBAL_RANKS

    all_pp_ranks = decoder_rank_generator.get_ranks('pp')
    for ranks in all_pp_ranks:
        group = torch.distributed.new_group(
            ranks,
            backend=pipeline_model_parallel_comm_backend or 'nccl'
        )
        if rank in ranks:
            _PIPELINE_MODEL_PARALLEL_GROUP = group
            _PIPELINE_GLOBAL_RANKS = ranks

    # Step 9: Create Model Parallel (TP+PP) groups (lines 972-1002)
    global _MODEL_PARALLEL_GROUP

    all_mp_ranks = decoder_rank_generator.get_ranks('tp-pp')
    for ranks in all_mp_ranks:
        group = torch.distributed.new_group(ranks, timeout=timeout)
        if rank in ranks:
            _MODEL_PARALLEL_GROUP = group

    # Step 10: Create combined groups for FP8 (lines 1085-1120)
    global _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP
    global _TENSOR_AND_CONTEXT_PARALLEL_GROUP

    # TP + DP + CP (for FP8 amax reduction)
    all_tp_dp_cp_ranks = decoder_rank_generator.get_ranks('tp-dp-cp')
    for ranks in all_tp_dp_cp_ranks:
        group = torch.distributed.new_group(ranks, timeout=timeout)
        if rank in ranks:
            _TENSOR_AND_DATA_PARALLEL_GROUP_WITH_CP = group

    # TP + CP (for FP8 within model parallel)
    all_tp_cp_ranks = decoder_rank_generator.get_ranks('tp-cp')
    for ranks in all_tp_cp_ranks:
        group = torch.distributed.new_group(ranks, timeout=timeout)
        if rank in ranks:
            _TENSOR_AND_CONTEXT_PARALLEL_GROUP = group

    # Step 11: Create Expert Parallel groups (lines 1122-1287)
    if expert_model_parallel_size > 1:
        global _EXPERT_MODEL_PARALLEL_GROUP
        global _EXPERT_TENSOR_PARALLEL_GROUP
        global _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP
        global _EXPERT_DATA_PARALLEL_GROUP

        # EP group (distributes experts)
        all_ep_ranks = expert_decoder_rank_generator.get_ranks('ep')
        for ranks in all_ep_ranks:
            group = torch.distributed.new_group(ranks, timeout=timeout)
            if rank in ranks:
                _EXPERT_MODEL_PARALLEL_GROUP = group

        # Expert TP group
        all_expert_tp_ranks = expert_decoder_rank_generator.get_ranks('tp')
        for ranks in all_expert_tp_ranks:
            group = torch.distributed.new_group(ranks, timeout=timeout)
            if rank in ranks:
                _EXPERT_TENSOR_PARALLEL_GROUP = group

        # Expert TP + EP combined
        all_expert_tp_ep_ranks = expert_decoder_rank_generator.get_ranks('tp-ep')
        for ranks in all_expert_tp_ep_ranks:
            group = torch.distributed.new_group(ranks, timeout=timeout)
            if rank in ranks:
                _EXPERT_TENSOR_AND_MODEL_PARALLEL_GROUP = group

        # Expert DP group
        all_expert_dp_ranks = expert_decoder_rank_generator.get_ranks('dp')
        for ranks in all_expert_dp_ranks:
            group = torch.distributed.new_group(ranks, timeout=timeout)
            if rank in ranks:
                _EXPERT_DATA_PARALLEL_GROUP = group

    # Step 12: Create Embedding groups (lines 1058-1080)
    global _EMBEDDING_GROUP
    global _POSITION_EMBEDDING_GROUP

    # Use custom functions or defaults (first and last PP stages)
    if get_embedding_ranks is None:
        get_embedding_ranks = default_embedding_ranks  # First and last stage
    if get_position_embedding_ranks is None:
        get_position_embedding_ranks = default_position_embedding_ranks  # First stage

    embedding_ranks = get_embedding_ranks()
    _EMBEDDING_GROUP = torch.distributed.new_group(embedding_ranks, timeout=timeout)

    position_embedding_ranks = get_position_embedding_ranks()
    _POSITION_EMBEDDING_GROUP = torch.distributed.new_group(
        position_embedding_ranks, timeout=timeout
    )
```

## Process Group Collection

The `ProcessGroupCollection` in `megatron/core/process_groups_config.py` provides a unified interface to all process groups:

```python
@dataclass
class ProcessGroupCollection:
    """
    Container for all parallel process groups.

    Organizes 30+ different process groups into a single dataclass.
    Enables passing groups to model components with clean interface.
    """

    # Basic Model Parallelism
    tp: torch.distributed.ProcessGroup  # Tensor parallel
    pp: torch.distributed.ProcessGroup  # Pipeline parallel
    mp: torch.distributed.ProcessGroup  # Model parallel (TP+PP)

    # Data Parallelism
    dp: torch.distributed.ProcessGroup  # Data parallel
    dp_cp: torch.distributed.ProcessGroup  # Data + Context parallel

    # Context Parallelism
    cp: torch.distributed.ProcessGroup  # Context parallel
    tp_cp: torch.distributed.ProcessGroup  # Tensor + Context parallel
    hcp: List[torch.distributed.ProcessGroup]  # Hierarchical CP

    # Expert Parallelism (MoE)
    ep: torch.distributed.ProcessGroup  # Expert model parallel
    expt_tp: torch.distributed.ProcessGroup  # Expert tensor parallel
    tp_ep: torch.distributed.ProcessGroup  # TP + EP
    tp_ep_pp: torch.distributed.ProcessGroup  # TP + EP + PP
    expt_dp: torch.distributed.ProcessGroup  # Expert data parallel

    # Combined groups for FP8
    tp_dp_cp: torch.distributed.ProcessGroup  # TP + DP + CP

    # Distributed Optimizer
    intra_dp_cp: torch.distributed.ProcessGroup  # Partial DP+CP
    intra_expt_dp: torch.distributed.ProcessGroup  # Partial expert DP
    intra_dist_opt: torch.distributed.ProcessGroup  # Optimizer shard
    inter_dist_opt: torch.distributed.ProcessGroup  # Inter-shard

    # Embeddings
    embd: torch.distributed.ProcessGroup  # Embedding group
    pos_embd: torch.distributed.ProcessGroup  # Position embedding

    @classmethod
    def use_mpu_process_groups(
        cls,
        required_pgs: Optional[List[str]] = None
    ) -> "ProcessGroupCollection":
        """
        Initialize ProcessGroupCollection from global parallel_state groups.

        Args:
            required_pgs: List of required group names. If None, includes all.
                         Useful for testing or limited parallelism setups.

        Returns:
            ProcessGroupCollection with requested groups populated.
        """
        # Mapping from field names to parallel_state getter functions
        pg_to_func = {
            'tp': partial(parallel_state.get_tensor_model_parallel_group,
                         check_initialized=False),
            'pp': partial(parallel_state.get_pipeline_model_parallel_group,
                         check_initialized=False),
            'mp': partial(parallel_state.get_model_parallel_group,
                         check_initialized=False),
            'dp': partial(parallel_state.get_data_parallel_group,
                         with_context_parallel=False),
            'dp_cp': partial(parallel_state.get_data_parallel_group,
                            with_context_parallel=True),
            'cp': partial(parallel_state.get_context_parallel_group,
                         check_initialized=False),
            'tp_cp': partial(parallel_state.get_tensor_and_context_parallel_group,
                            check_initialized=False),
            'hcp': partial(parallel_state.get_hierarchical_context_parallel_groups,
                          check_initialized=False),
            'ep': partial(parallel_state.get_expert_model_parallel_group,
                         check_initialized=False),
            'expt_tp': partial(parallel_state.get_expert_tensor_parallel_group,
                              check_initialized=False),
            'tp_ep': partial(parallel_state.get_expert_tensor_and_model_parallel_group,
                            check_initialized=False),
            'tp_ep_pp': partial(parallel_state.get_expert_tensor_model_pipeline_parallel_group,
                               check_initialized=False),
            'expt_dp': partial(parallel_state.get_expert_data_parallel_group,
                              check_initialized=False),
            # ... [additional mappings for all 30+ groups]
        }

        # Populate only requested groups
        kwargs = {}
        for pg_name in (required_pgs or pg_to_func.keys()):
            if pg_name in pg_to_func:
                kwargs[pg_name] = pg_to_func[pg_name]()

        return cls(**kwargs)

    @staticmethod
    def setup_process_groups_for_optimizer(
        pg_collection: "ProcessGroupCollection",
        model_chunks: List[torch.nn.Module],
        use_gloo: bool = False,
    ) -> List[torch.distributed.ProcessGroup]:
        """
        Setup process groups for distributed optimizer.

        Creates appropriate groups for gradient sharding based on whether
        model has expert parameters.

        Args:
            pg_collection: The process group collection
            model_chunks: List of model chunks (for pipeline parallel)
            use_gloo: Whether to use Gloo backend

        Returns:
            List of process groups for each model chunk
        """
        # Check if any chunk has expert parameters
        has_expert_params = any(
            hasattr(chunk, 'has_expert_parameters') and chunk.has_expert_parameters
            for chunk in model_chunks
        )

        # Select appropriate DP group
        if has_expert_params:
            # Use expert DP group for MoE layers
            dp_groups = [pg_collection.intra_expt_dp] * len(model_chunks)
        else:
            # Use standard DP+CP group for regular layers
            dp_groups = [pg_collection.intra_dp_cp] * len(model_chunks)

        return dp_groups
```

## Multi-Dimensional Configuration Examples

### 3D Parallelism: TP × PP × DP

```python
# 512 GPUs training GPT-3 175B
# Goal: Fit model with TP+PP, maximize throughput with DP

world_size = 512
tensor_model_parallel_size = 8      # NVLink domain (single node)
pipeline_model_parallel_size = 8    # 96 layers / 8 = 12 layers per stage
context_parallel_size = 1           # No CP for standard contexts
expert_model_parallel_size = 1      # No experts

# Data parallel size computed automatically:
# data_parallel_size = 512 / (8 × 8 × 1) = 8

# This creates:
# - 64 TP groups of 8 GPUs each (NVLink domains)
# - 64 PP groups of 8 GPUs each (across nodes)
# - 64 DP groups of 8 GPUs each (gradient sync)
# - Total: 8 (TP) × 8 (PP) × 8 (DP) = 512 GPUs

# Each rank belongs to exactly one group of each type
```

### 4D Parallelism with Context Parallel: TP × PP × DP × CP

```python
# 256 GPUs training with very long context (32K tokens)
# Goal: Enable long context with CP, maintain throughput

world_size = 256
tensor_model_parallel_size = 4
pipeline_model_parallel_size = 4
context_parallel_size = 4           # Split 32K context 4 ways
expert_model_parallel_size = 1

# data_parallel_size = 256 / (4 × 4 × 4) = 4

# Process groups created:
# - TP groups: 64 groups of 4 GPUs
# - PP groups: 64 groups of 4 GPUs
# - CP groups: 64 groups of 4 GPUs
# - DP groups: 64 groups of 4 GPUs
# - DP+CP groups: 16 groups of 16 GPUs (for weight gradient sync)

# Weight gradients all-reduced across DP×CP = 16 GPUs
# Activation memory reduced by 4× due to sequence splitting
```

### 4D Parallelism with Expert Parallel: TP × PP × DP × EP

```python
# 1024 GPUs training MoE model with 64 experts
# Goal: Distribute experts across GPUs

world_size = 1024
tensor_model_parallel_size = 4      # Standard TP
pipeline_model_parallel_size = 4
context_parallel_size = 1
expert_model_parallel_size = 8      # 64 experts / 8 = 8 experts per GPU
expert_tensor_parallel_size = 4     # Same TP for experts

# Standard layers: TP=4, PP=4, CP=1, EP=1
# standard_data_parallel_size = 1024 / (4 × 4 × 1) = 64

# Expert layers: TP=4, PP=4, EP=8
# expert_data_parallel_size = 1024 / (4 × 4 × 8) = 8

# Creates TWO rank generators:
# 1. decoder_rank_generator (standard layers)
# 2. expert_decoder_rank_generator (expert layers)

# PP groups must be identical between both generators
```

### 5D Parallelism: TP × PP × DP × CP × EP

```python
# Complex setup requiring separate rank generators
# Current limitation: CP and EP cannot both be >1 in same generator

# Option 1: Use CP in decoder, EP in experts (most common)
decoder_config = {
    'tensor_model_parallel_size': 4,
    'pipeline_model_parallel_size': 4,
    'context_parallel_size': 4,
    'expert_model_parallel_size': 1,  # No EP in decoder
}

expert_config = {
    'expert_tensor_parallel_size': 4,
    'pipeline_model_parallel_size': 4,  # Must match decoder
    'expert_model_parallel_size': 8,    # EP for experts
    'context_parallel_size': 1,         # No CP in experts
}

# Creates two independent rank generators with shared PP groups
```

## Rank Ordering and Its Impact

The `order` parameter determines how ranks map to parallelism dimensions. Common orderings:

**Default: "tp-cp-ep-dp-pp"**
- TP most inner (consecutive ranks in TP group)
- PP most outer (largest stride between ranks)
- Optimizes for NVLink-based TP groups

**Example with 16 GPUs, TP=2, PP=2, DP=4, order="tp-pp-dp":**
```
TP groups (stride 1):
[0,1], [2,3], [4,5], [6,7], [8,9], [10,11], [12,13], [14,15]

PP groups (stride 2):
[0,2,4,6,8,10,12,14], [1,3,5,7,9,11,13,15]

DP groups (stride 4):
[0,2,4,6], [1,3,5,7], [8,10,12,14], [9,11,13,15]
```

**Alternative: "tp-cp-pp-ep-dp"**
- PP inner than EP
- Useful for certain network topologies
- Must ensure PP groups match between decoder and expert generators

## Communication Hierarchy and Frequency

Different parallelism dimensions operate at different timescales:

| Dimension | Frequency | Message Size | Latency Tolerance | Network Requirement |
|-----------|-----------|--------------|-------------------|---------------------|
| **Tensor Parallel (TP)** | Every layer (~100 μs) | ~MB per op | Very low | NVLink (300+ GB/s) |
| **Pipeline Parallel (PP)** | Every microbatch (~10 ms) | ~MB per stage | Medium | InfiniBand (100 GB/s) |
| **Expert Parallel (EP)** | Per MoE layer (~10 ms) | Variable | Medium | InfiniBand (100 GB/s) |
| **Data Parallel (DP)** | Once per batch (~seconds) | ~GB total | High | Ethernet OK (10-100 GB/s) |
| **Context Parallel (CP)** | Every attention (~1 ms) | ~MB per rank | Low-Medium | NVLink or IB preferred |

This hierarchy explains key design decisions:
- **TP requires NVLink** due to high frequency and low latency needs
- **PP works on InfiniBand** as microbatch intervals tolerate medium latency
- **DP can use Ethernet** since batch intervals amortize high latency
- **CP needs good bandwidth** but can tolerate more latency than TP

## Constraints and Validation

The implementation enforces several critical constraints:

```python
# 1. World size divisibility
assert world_size % (tp_size * pp_size * cp_size) == 0

# 2. EP and CP mutual exclusion in same rank generator
if ep_size > 1 and cp_size > 1:
    raise ValueError("Cannot combine EP and CP in same rank generator")

# 3. PP group consistency between decoder and expert
assert decoder_rank_generator.get_ranks("pp") == expert_decoder_rank_generator.get_ranks("pp")

# 4. Distributed optimizer sharding divisibility
assert (dp_size * cp_size) % num_distributed_optimizer_instances == 0
assert expert_dp_size % num_distributed_optimizer_instances == 0

# 5. Sequence parallel requires TP
if sequence_parallel and tp_size == 1:
    raise ValueError("Sequence parallelism requires tensor_model_parallel_size > 1")

# 6. TP + EP requires sequence parallel
if tp_size > 1 and ep_size > 1:
    assert sequence_parallel, "TP with EP requires sequence_parallel=True"

# 7. Pipeline requires dtype specification
if pp_size > 1 and pipeline_dtype is None:
    raise ValueError("Must specify pipeline_dtype when using pipeline parallelism")

# 8. Virtual pipeline requires pipeline
if virtual_pipeline_model_parallel_size is not None and pp_size == 1:
    raise ValueError("Virtual pipeline requires pipeline_model_parallel_size > 1")
```

## Complete Configuration Example

```python
from megatron.core import parallel_state
from megatron.core.transformer import TransformerConfig
from megatron.training import get_args

# Initialize distributed
torch.distributed.init_process_group(backend='nccl')

# Initialize 4D parallelism: TP × PP × DP × CP
parallel_state.initialize_model_parallel(
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    virtual_pipeline_model_parallel_size=2,  # Interleaved schedule
    context_parallel_size=2,
    expert_model_parallel_size=1,
    num_distributed_optimizer_instances=1,
    order="tp-cp-ep-dp-pp",  # Default ordering
    distributed_timeout_minutes=30,
)

# Total GPUs: 4 (TP) × 4 (PP) × 2 (CP) × DP = 128 GPUs
# For 128 GPUs: DP = 128 / (4 × 4 × 2) = 4

# Configure model with all parallelism settings
config = TransformerConfig(
    # Model architecture
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    ffn_hidden_size=16384,

    # Parallelism configuration
    tensor_model_parallel_size=4,
    pipeline_model_parallel_size=4,
    context_parallel_size=2,

    # Sequence handling
    sequence_parallel=True,  # Required with TP

    # Context parallel communication
    cp_comm_type="p2p",  # Ring attention
    overlap_p2p_comm=True,  # Async overlap

    # Pipeline configuration
    pipeline_dtype=torch.bfloat16,
    virtual_pipeline_model_parallel_size=2,

    # Memory optimizations
    recompute_granularity="selective",
    recompute_method="block",
    recompute_num_layers=1,

    # Communication optimizations
    overlap_param_gather=True,
    overlap_grad_reduce=True,
    use_distributed_optimizer=True,
)

# Query parallelism information
print(f"TP rank: {parallel_state.get_tensor_model_parallel_rank()}")
print(f"PP rank: {parallel_state.get_pipeline_model_parallel_rank()}")
print(f"DP rank: {parallel_state.get_data_parallel_rank()}")
print(f"CP rank: {parallel_state.get_context_parallel_rank()}")
print(f"TP world size: {parallel_state.get_tensor_model_parallel_world_size()}")
print(f"DP world size: {parallel_state.get_data_parallel_world_size()}")
```

## Performance Tuning Guidelines

### Choosing Parallelism Dimensions

**Step 1: Start with Data Parallelism Only**
- If model fits on single GPU → Use only DP
- Maximizes simplicity and often best throughput

**Step 2: Add Tensor Parallelism if Needed**
- Model doesn't fit on single GPU → Add TP=2, 4, or 8
- Use NVLink domain size (usually 8 GPUs per node)
- Always enable `sequence_parallel=True` with TP

**Step 3: Add Pipeline Parallelism for Large Models**
- Model still doesn't fit → Add PP=2, 4, 8, ...
- Balance stages: ~equal memory per stage
- Set microbatches: `num_microbatches = 4-8 × PP × DP`
- Use virtual pipeline if PP ≥ 4: `virtual_pp_size=2`

**Step 4: Add Context Parallelism for Long Sequences**
- Context > 8K and activation memory is bottleneck → Add CP=2, 4
- Use P2P mode for NVLink, A2A for larger CP or IB

**Step 5: Add Expert Parallelism for MoE**
- Using MoE architecture → Set EP = num_experts / experts_per_rank
- Balance between EP and DP based on expert count

### Expected Performance

| Scale | Configuration | Expected MFU | Notes |
|-------|---------------|--------------|-------|
| 8 GPUs | TP=2, DP=4 | 40-50% | Single-node, excellent overlap |
| 64 GPUs | TP=4, PP=2, DP=8 | 45-55% | Multi-node, good balance |
| 512 GPUs | TP=8, PP=8, DP=8 | 48-58% | Large scale, well-tuned |
| 2048+ GPUs | TP=8, PP=16, DP=16+ | 50-60% | Very large scale, expert tuning |

MFU (Model FLOPs Utilization) improves with scale due to better batching but communication overhead increases.

## References

- Megatron-LM paper: "Efficient Large-Scale Language Model Training on GPU Clusters" (Narayanan et al., 2021)
- Implementation: `megatron/core/parallel_state.py`, `megatron/core/process_groups_config.py`
- 3D Parallelism blog: https://developer.nvidia.com/blog/scaling-language-model-training-to-a-trillion-parameters-using-megatron/
- Process group documentation: `docs/source/api-guide/parallelism.rst`
