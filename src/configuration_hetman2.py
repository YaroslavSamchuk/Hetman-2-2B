"""
================================================================================
HETMAN-2-NCM 2.0B: PRODUCTION CONFIGURATION MODULE
================================================================================
Architecture: Block-Causal Next-Concept Modeling (NCM) + Ternary BitNet b1.58 +
              Gated DeltaNet + Speculative Latent-Hopfield Core v2 (SLH-Core v2)
Specification Reference: spec/01_MATHEMATICAL_FOUNDATIONS_AND_TENSOR_ALGEBRA.md
                         spec/02_BITNET_TERNARY_AND_FWHT_INT8_ENGINE.md
                         spec/03_GATED_DELTANET_RECURRENT_STATE_DYNAMICS.md
                         spec/04_SLH_CORE_V2_HOPFIELD_FACTUAL_MEMORY.md
                         spec/05_PHASE1_CAUSAL_BLUEPRINT_DRAFTER.md
                         spec/06_PHASE2_BIDIRECTIONAL_NCM_REFINER_AND_DEDUP.md
                         spec/07_PHASE3_BLOCK_CAUSAL_NCM_DECODER.md
                         spec/08_TRAINING_RECIPE_LOSSES_AND_CURRICULUM.md
                         spec/09_EDGE_RUNTIME_RADA_CPP_AND_MEMORY_AUDIT.md
                         spec/10_INPAINTING_VERIFICATION_AND_ERROR_HANDLING.md
                         spec/11_HIDDEN_PITFALLS_FAILURE_MODES_AND_MITIGATIONS.md
                         spec/12_TMAC_LUT_AVX2_FEASIBILITY_AND_PITFALLS.md
                         spec/13_RADA_CPP_HOPFIELD_KV_CACHE_ARCHITECTURE.md
                         spec/14_HOPFIELD_KV_CACHE_TRAINING_AND_NO_DEGRADATION.md
                         spec/15_DYNAMIC_BLUEPRINT_EVOLUTION_MECHANICS.md
License: Apache License 2.0
================================================================================
"""

import math
import struct
from typing import Any, ClassVar, Dict, List, Optional, Set, Tuple, Union

from transformers.configuration_utils import PretrainedConfig
from transformers import AutoConfig


class Hetman2Config(PretrainedConfig):
    """
    Configuration class for the Hetman-2-NCM 2.0B foundational neural architecture.

    Compatible with:
      - PyTorch 2.x
      - Hugging Face Transformers (`PretrainedConfig`, `AutoConfig`, `AutoModel`, `AutoModelForCausalLM`)
      - Hugging Face Dynamic AutoClass (`auto_map` remote code / dynamic module execution)
      - Rada.cpp C++20 Edge Inference Engine (Container format v2)

    Key Hyperparameters Invariants:
      - `d_model`: 1536 (3 x 512 FWHT blocks)
      - `layers`: 34 (28 Gated DeltaNet + 6 Global Softmax GQA 12:2)
      - `intermediate_dim`: 6144 (4x SwiGLU FFN expansion)
      - `vocab_size`: 65536 (Cyrillic-optimized Byte-Pair Encoding)
      - `hopfield_slots`: 16777216 (16.78M factual memory attractor slots)
      - `block_size`: 400 (Tokens per macro-block, with 16-token overlap)
      - `base_passes`: 8 (Multi-scale decoder schedule: 2 syntax, 2 morphology, 4 lexicon)
      - `emergency_passes`: 3 (Inpainting retry budget before micro-AR fallback)
      - `rvq_codebooks`: 4 (4-level Residual Vector Quantization ladder)
      - `version`: 2 (Rada NCM v2 binary protocol)
    """

    model_type = "hetman2_ncm"

    # Hugging Face Dynamic AutoClass mapping (for remote code & dynamic loading)
    auto_map: ClassVar[Dict[str, str]] = {
        "AutoConfig": "configuration_hetman2.Hetman2Config",
        "AutoModel": "modeling_hetman2.Hetman2Model",
        "AutoModelForCausalLM": "modeling_hetman2.Hetman2ForCausalLM",
    }

    # Canonical Hugging Face attribute map (standard HF aliases)
    attribute_map: ClassVar[Dict[str, str]] = {
        "hidden_size": "d_model",
        "num_hidden_layers": "layers",
        "num_layers": "layers",
        "intermediate_size": "intermediate_dim",
        "tokens_per_block": "block_size",
        "num_rvq_codebooks": "rvq_codebooks",
        "rvq_levels": "rvq_codebooks",
    }

    # Full bidirectional alias registry for synchronized parameter tracking
    ALIAS_MAP: ClassVar[Dict[str, List[str]]] = {
        "d_model": ["hidden_size"],
        "layers": ["num_hidden_layers", "num_layers"],
        "intermediate_dim": ["intermediate_size"],
        "block_size": ["tokens_per_block", "block_len"],
        "overlap_size": ["overlap_len"],
        "rvq_codebooks": ["num_rvq_codebooks", "rvq_levels"],
        "deltanet_num_heads": ["deltanet_heads", "num_attention_heads"],
        "global_attention_heads": ["global_heads_q"],
        "global_kv_heads": ["global_heads_kv"],
        "hopfield_num_groups": ["hopfield_groups"],
        "refiner_iterations": ["refine_passes"],
        "refiner_dedup_threshold": ["dedup_tau"],
        "enable_dual_loop_deliberation": ["enable_deliberation", "use_deliberation"],
        "deliberation_max_passes": ["max_deliberation_passes"],
    }

    # Reverse alias lookup: alias -> canonical attribute name
    REVERSE_ALIAS_MAP: ClassVar[Dict[str, str]] = {
        alias: canonical
        for canonical, aliases in ALIAS_MAP.items()
        for alias in aliases
    }

    # Strict integer attributes for preventing int -> float type loss
    _INT_ATTRS: ClassVar[Set[str]] = {
        "d_model", "layers", "intermediate_dim", "vocab_size", "hopfield_slots",
        "block_size", "base_passes", "emergency_passes", "rvq_codebooks", "version",
        "hidden_size", "num_hidden_layers", "num_layers", "intermediate_size",
        "tokens_per_block", "block_len", "overlap_len", "num_rvq_codebooks", "rvq_levels",
        "num_deltanet_layers", "num_global_layers", "deltanet_num_heads", "deltanet_head_dim",
        "deltanet_inner_dim", "deltanet_conv_kernel", "num_attention_heads", "deltanet_heads",
        "global_attention_heads", "global_heads_q", "global_kv_heads", "global_heads_kv",
        "global_head_dim", "hopfield_num_groups", "hopfield_groups", "hopfield_codebook_size",
        "hopfield_code_dim", "hopfield_val_dim", "hopfield_expanded_dim", "hopfield_subquery_dim",
        "hopfield_topk_1", "hopfield_topk_2", "hopfield_shared_attractors", "d_drafter",
        "drafter_heads", "dim_c_sem", "dim_idx_hop", "dim_tau_style", "dim_ell_budget",
        "min_blueprint_steps", "max_blueprint_steps", "default_blueprint_steps",
        "refiner_iterations", "refine_passes", "overlap_size", "max_blocks", "latents_per_block",
        "rvq_codebook_size", "rvq_dim", "hadamard_block_size", "int8_clip_bound",
        "max_position_embeddings", "rada_magic", "rada_version", "rada_alignment",
        "dim_intent", "dim_style", "seam_window",
    }

    # Strict floating point attributes
    _FLOAT_ATTRS: ClassVar[Set[str]] = {
        "rope_theta", "yarn_cooling", "qk_norm_bound", "hopfield_beta",
        "hopfield_pid_ki", "hopfield_pid_kd", "vmf_kappa", "gumbel_temperature",
        "refiner_dedup_threshold", "dedup_tau", "refiner_lipschitz_bound",
        "rvq_commitment_weight", "rvq_entropy_weight", "gamma_min", "gamma_max",
        "rho_thresh", "ternary_bits", "swiglu_limit", "logit_softcap",
        "rms_norm_eps", "initializer_range", "rada_memory_arena_mb",
        "alpha_residual_anchor",
    }

    def __init__(
        self,
        # Mandatory primary parameters (Hetman-2-NCM 2.0B canonical)
        d_model: int = 1536,
        layers: int = 34,
        intermediate_dim: int = 6144,
        vocab_size: int = 65536,
        hopfield_slots: int = 16777216,
        block_size: int = 400,
        base_passes: int = 8,
        emergency_passes: int = 3,
        rvq_codebooks: int = 4,
        version: int = 2,
        # HuggingFace & Legacy parameter aliases (for seamless cross-compatibility)
        hidden_size: Optional[int] = None,
        num_hidden_layers: Optional[int] = None,
        num_layers: Optional[int] = None,
        intermediate_size: Optional[int] = None,
        tokens_per_block: Optional[int] = None,
        block_len: Optional[int] = None,
        overlap_len: Optional[int] = None,
        num_rvq_codebooks: Optional[int] = None,
        rvq_levels: Optional[int] = None,
        deltanet_heads: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        global_heads_q: Optional[int] = None,
        global_heads_kv: Optional[int] = None,
        hopfield_groups: Optional[int] = None,
        refine_passes: Optional[int] = None,
        dedup_tau: Optional[float] = None,
        # Hybrid Transformer Layer Topology (28 DeltaNet + 6 Global GQA 12:2)
        num_deltanet_layers: int = 28,
        num_global_layers: int = 6,
        deltanet_num_heads: int = 16,
        deltanet_head_dim: int = 64,
        deltanet_inner_dim: Optional[int] = None,
        deltanet_conv_kernel: int = 4,
        deltanet_state_dtype: str = "float32",
        global_attention_heads: int = 12,
        global_kv_heads: int = 2,
        global_head_dim: int = 128,
        global_layer_indices: Optional[List[int]] = None,
        rope_theta: float = 10000000.0,
        yarn_cooling: float = 1.6,
        qk_norm_bound: float = 11.3137085,
        use_qk_norm: bool = True,
        # Speculative Latent-Hopfield Core v2 (SLH-Core v2)
        hopfield_num_groups: int = 16,
        hopfield_codebook_size: int = 1024,
        hopfield_code_dim: int = 64,
        hopfield_val_dim: int = 32,
        hopfield_expanded_dim: int = 128,
        hopfield_subquery_dim: int = 2048,
        hopfield_topk_1: int = 8,
        hopfield_topk_2: int = 4,
        hopfield_shared_attractors: int = 1,
        hopfield_beta: float = 0.176776695,  # 1.0 / sqrt(32)
        hopfield_pid_ki: float = 0.01,
        hopfield_pid_kd: float = 0.005,
        # Phase 1: Causal AR Blueprint Drafter
        d_drafter: int = 512,
        drafter_heads: int = 8,
        dim_c_sem: int = 768,
        dim_idx_hop: int = 384,
        dim_tau_style: int = 256,
        dim_ell_budget: int = 128,
        min_blueprint_steps: int = 1,
        max_blueprint_steps: int = 8,
        default_blueprint_steps: int = 4,
        vmf_kappa: float = 16.0,
        gumbel_temperature: float = 1.0,
        length_bins: Tuple[int, ...] = (200, 300, 400, 500),
        # Phase 2: Global Bidirectional Refiner & Deduplicator
        refiner_iterations: int = 4,
        refiner_dedup_threshold: float = 0.85,
        refiner_lipschitz_bound: float = 0.957,
        # Dual-Loop Deliberation (Variant C)
        enable_dual_loop_deliberation: bool = True,
        deliberation_max_passes: int = 32,
        deliberation_banach_tol: float = 1e-4,
        explain_thought_default: bool = False,
        think_tag_default: bool = False,
        # Phase 3: Block-Causal NCM Decoder & Dynamic Intent Evolution
        overlap_size: int = 16,
        max_blocks: int = 8,
        latents_per_block: int = 16,
        rvq_codebook_size: int = 4096,
        rvq_dim: int = 768,
        rvq_commitment_weight: float = 0.25,
        rvq_entropy_weight: float = 0.01,
        gamma_min: float = 0.10,
        gamma_max: float = 0.80,
        rho_thresh: float = 0.25,
        # Semantic Intent, Style & Seam Smoothing parameters
        dim_intent: int = 256,
        dim_style: int = 64,
        seam_window: int = 4,
        alpha_residual_anchor: float = 0.25,
        # BitNet b1.58 & Fast Walsh-Hadamard INT8 Engine
        ternary_bits: float = 1.58,
        use_hadamard_rotation: bool = True,
        hadamard_block_size: int = 512,
        int8_clip_bound: int = 127,
        swiglu_limit: float = 10.0,
        logit_softcap: float = 30.0,
        # Numerical Bounds & Context Scaling
        max_position_embeddings: int = 262144,  # 256k context
        rms_norm_eps: float = 1e-6,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        # Rada Edge Runtime Binary Protocol Constants
        rada_magic: int = 0x4D434E52,  # 'RNCM' in little-endian = 1296256594
        rada_version: int = 2,
        rada_alignment: int = 64,
        rada_memory_arena_mb: float = 717.67,
        auto_map: Optional[Dict[str, str]] = None,
        **kwargs: Any,
    ) -> None:
        # Resolve aliases with strict type conversions
        resolved_d_model = int(hidden_size if hidden_size is not None else d_model)
        resolved_layers = int(
            num_hidden_layers
            if num_hidden_layers is not None
            else (num_layers if num_layers is not None else layers)
        )
        resolved_intermediate = int(
            intermediate_size if intermediate_size is not None else intermediate_dim
        )
        resolved_block_size = int(
            tokens_per_block
            if tokens_per_block is not None
            else (block_len if block_len is not None else block_size)
        )
        resolved_overlap_size = int(
            overlap_len if overlap_len is not None else overlap_size
        )
        resolved_rvq_codebooks = int(
            num_rvq_codebooks
            if num_rvq_codebooks is not None
            else (rvq_levels if rvq_levels is not None else rvq_codebooks)
        )
        resolved_deltanet_heads = int(
            num_attention_heads
            if num_attention_heads is not None
            else (deltanet_heads if deltanet_heads is not None else deltanet_num_heads)
        )
        resolved_global_heads_q = int(
            global_heads_q if global_heads_q is not None else global_attention_heads
        )
        resolved_global_heads_kv = int(
            global_heads_kv if global_heads_kv is not None else global_kv_heads
        )
        resolved_hopfield_groups = int(
            hopfield_groups if hopfield_groups is not None else hopfield_num_groups
        )
        resolved_refiner_iterations = int(
            refine_passes if refine_passes is not None else refiner_iterations
        )
        resolved_refiner_dedup_threshold = float(
            dedup_tau if dedup_tau is not None else refiner_dedup_threshold
        )

        if deltanet_inner_dim is None:
            resolved_deltanet_inner_dim = resolved_deltanet_heads * int(deltanet_head_dim)
        else:
            resolved_deltanet_inner_dim = int(deltanet_inner_dim)

        # Call PretrainedConfig super constructor
        super().__init__(
            vocab_size=int(vocab_size),
            hidden_size=resolved_d_model,
            num_hidden_layers=resolved_layers,
            intermediate_size=resolved_intermediate,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

        # Ensure auto_map is populated for Hugging Face Dynamic AutoClass
        if auto_map is not None:
            self.auto_map = auto_map
        else:
            self.auto_map = dict(self.__class__.auto_map)

        # Core model identity and dimensions (strictly cast)
        self.d_model = resolved_d_model
        self.layers = resolved_layers
        self.intermediate_dim = resolved_intermediate
        self.vocab_size = int(vocab_size)
        self.hopfield_slots = int(hopfield_slots)
        self.block_size = resolved_block_size
        self.base_passes = int(base_passes)
        self.emergency_passes = int(emergency_passes)
        self.rvq_codebooks = resolved_rvq_codebooks
        self.version = int(version)

        # Hybrid Transformer Topology
        self.num_deltanet_layers = int(num_deltanet_layers)
        self.num_global_layers = int(num_global_layers)

        # DeltaNet recurrence settings
        self.deltanet_num_heads = resolved_deltanet_heads
        self.deltanet_head_dim = int(deltanet_head_dim)
        self.deltanet_inner_dim = resolved_deltanet_inner_dim
        self.deltanet_conv_kernel = int(deltanet_conv_kernel)
        self.deltanet_state_dtype = str(deltanet_state_dtype)

        # Global Attention GQA settings
        self.global_attention_heads = resolved_global_heads_q
        self.global_kv_heads = resolved_global_heads_kv
        self.global_head_dim = int(global_head_dim)
        if global_layer_indices is None:
            self.global_layer_indices = [7, 13, 17, 21, 25, 29]
        else:
            self.global_layer_indices = [int(idx) for idx in global_layer_indices]

        self.rope_theta = float(rope_theta)
        self.yarn_cooling = float(yarn_cooling)
        self.qk_norm_bound = float(qk_norm_bound)
        self.use_qk_norm = bool(use_qk_norm)

        # SLH-Core v2 Hopfield memory
        self.hopfield_num_groups = resolved_hopfield_groups
        self.hopfield_codebook_size = int(hopfield_codebook_size)
        self.hopfield_code_dim = int(hopfield_code_dim)
        self.hopfield_val_dim = int(hopfield_val_dim)
        self.hopfield_expanded_dim = int(hopfield_expanded_dim)
        self.hopfield_subquery_dim = int(hopfield_subquery_dim)
        self.hopfield_topk_1 = int(hopfield_topk_1)
        self.hopfield_topk_2 = int(hopfield_topk_2)
        self.hopfield_shared_attractors = int(hopfield_shared_attractors)
        self.hopfield_beta = float(hopfield_beta)
        self.hopfield_pid_ki = float(hopfield_pid_ki)
        self.hopfield_pid_kd = float(hopfield_pid_kd)

        # Phase 1: Blueprint Drafter
        self.d_drafter = int(d_drafter)
        self.drafter_heads = int(drafter_heads)
        self.dim_c_sem = int(dim_c_sem)
        self.dim_idx_hop = int(dim_idx_hop)
        self.dim_tau_style = int(dim_tau_style)
        self.dim_ell_budget = int(dim_ell_budget)
        self.min_blueprint_steps = int(min_blueprint_steps)
        self.max_blueprint_steps = int(max_blueprint_steps)
        self.default_blueprint_steps = int(default_blueprint_steps)
        self.vmf_kappa = float(vmf_kappa)
        self.gumbel_temperature = float(gumbel_temperature)
        self.length_bins = [int(b) for b in length_bins]

        # Phase 2: Refiner & Deduplicator
        self.refiner_iterations = resolved_refiner_iterations
        self.refiner_dedup_threshold = resolved_refiner_dedup_threshold
        self.refiner_lipschitz_bound = float(refiner_lipschitz_bound)
        # Dual-Loop Deliberation (Variant C)
        self.enable_dual_loop_deliberation = bool(enable_dual_loop_deliberation)
        self.deliberation_max_passes = int(deliberation_max_passes)
        self.deliberation_banach_tol = float(deliberation_banach_tol)
        self.explain_thought_default = bool(explain_thought_default)
        self.think_tag_default = bool(think_tag_default)

        # Phase 3: Block Decoder & Dynamic Blueprint Evolution
        self.overlap_size = resolved_overlap_size
        self.max_blocks = int(max_blocks)
        self.latents_per_block = int(latents_per_block)
        self.rvq_codebook_size = int(rvq_codebook_size)
        self.rvq_dim = int(rvq_dim)
        self.rvq_commitment_weight = float(rvq_commitment_weight)
        self.rvq_entropy_weight = float(rvq_entropy_weight)
        self.gamma_min = float(gamma_min)
        self.gamma_max = float(gamma_max)
        self.rho_thresh = float(rho_thresh)

        # Intent, Style & Boundary Seam Smoothing
        self.dim_intent = int(dim_intent)
        self.dim_style = int(dim_style)
        self.seam_window = int(seam_window)
        self.alpha_residual_anchor = float(alpha_residual_anchor)

        # BitNet & FWHT Compute Engine
        self.ternary_bits = float(ternary_bits)
        self.use_hadamard_rotation = bool(use_hadamard_rotation)
        self.hadamard_block_size = int(hadamard_block_size)
        self.int8_clip_bound = int(int8_clip_bound)
        self.swiglu_limit = float(swiglu_limit)
        self.logit_softcap = float(logit_softcap)

        # Long Context & Normalization
        self.max_position_embeddings = int(max_position_embeddings)
        self.rms_norm_eps = float(rms_norm_eps)
        self.initializer_range = float(initializer_range)

        # Rada Runtime Binary Container Format
        self.rada_magic = int(rada_magic)
        self.rada_version = int(rada_version)
        self.rada_alignment = int(rada_alignment)
        self.rada_memory_arena_mb = float(rada_memory_arena_mb)

        # Initialize alias properties
        for canonical, aliases in self.ALIAS_MAP.items():
            val = getattr(self, canonical, None)
            if val is not None:
                for alias in aliases:
                    super().__setattr__(alias, val)

        # Mark initialization complete
        self._is_initialized = True

        # Perform strict architectural validation
        self.validate()

    def __setattr__(self, name: str, value: Any) -> None:
        """
        Custom attribute setter providing:
          1. Strict type preservation (no int -> float degradation).
          2. Bidirectional alias synchronization (updating an alias updates canonical and vice versa).
        """
        # Strict type casting
        if name in self._INT_ATTRS and value is not None:
            value = int(value)
        elif name in self._FLOAT_ATTRS and value is not None:
            value = float(value)

        # Check if setting an alias -> resolve to canonical and sync all
        if name in self.REVERSE_ALIAS_MAP:
            canonical = self.REVERSE_ALIAS_MAP[name]
            super().__setattr__(name, value)
            super().__setattr__(canonical, value)
            for other_alias in self.ALIAS_MAP[canonical]:
                if other_alias != name:
                    super().__setattr__(other_alias, value)
            return

        super().__setattr__(name, value)
        # Check if setting canonical -> update all registered aliases
        if name in self.ALIAS_MAP:
            for alias in self.ALIAS_MAP[name]:
                super().__setattr__(alias, value)

    def validate(self) -> None:
        """
        Validates the configuration invariants against the 15 specifications.
        Raises AssertionError if any mathematical or architectural constraint is violated.
        """
        # 1. Layer distribution constraint
        assert (
            self.layers == self.num_deltanet_layers + self.num_global_layers
        ), f"Total layers ({self.layers}) must equal num_deltanet_layers ({self.num_deltanet_layers}) + num_global_layers ({self.num_global_layers})"

        assert (
            len(self.global_layer_indices) == self.num_global_layers
        ), f"global_layer_indices length ({len(self.global_layer_indices)}) must match num_global_layers ({self.num_global_layers})"

        # 2. Fast Walsh-Hadamard block divisibility
        assert (
            self.d_model % self.hadamard_block_size == 0
        ), f"d_model ({self.d_model}) must be divisible by hadamard_block_size ({self.hadamard_block_size})"

        # 3. Global Attention dimension constraint
        assert (
            self.d_model == self.global_attention_heads * self.global_head_dim
        ), f"d_model ({self.d_model}) must equal global_attention_heads ({self.global_attention_heads}) * global_head_dim ({self.global_head_dim})"

        # 4. DeltaNet spatial bottleneck dimension constraint
        assert (
            self.deltanet_inner_dim == self.deltanet_num_heads * self.deltanet_head_dim
        ), f"deltanet_inner_dim ({self.deltanet_inner_dim}) must equal deltanet_num_heads ({self.deltanet_num_heads}) * deltanet_head_dim ({self.deltanet_head_dim})"

        # 5. Hopfield memory slot count
        expected_hopfield_slots = (
            self.hopfield_num_groups
            * self.hopfield_codebook_size
            * self.hopfield_codebook_size
        )
        assert (
            self.hopfield_slots == expected_hopfield_slots
        ), f"hopfield_slots ({self.hopfield_slots}) must equal groups * C1 * C2 ({expected_hopfield_slots})"

        # 6. Blueprint partitioning integrity (dim_c_sem + dim_idx_hop + dim_tau_style + dim_ell_budget == d_model)
        blueprint_sum = (
            self.dim_c_sem
            + self.dim_idx_hop
            + self.dim_tau_style
            + self.dim_ell_budget
        )
        assert (
            blueprint_sum == self.d_model
        ), f"Blueprint channels sum ({blueprint_sum}) must equal d_model ({self.d_model})"

        # 7. Block size & Overlap constraints
        assert (
            self.block_size > self.overlap_size
        ), f"block_size ({self.block_size}) must be greater than overlap_size ({self.overlap_size})"

        # 8. Generation passes constraints
        assert self.base_passes >= 1, f"base_passes ({self.base_passes}) must be >= 1"
        assert self.emergency_passes >= 1, f"emergency_passes ({self.emergency_passes}) must be >= 1"
        assert self.rvq_codebooks >= 1, f"rvq_codebooks ({self.rvq_codebooks}) must be >= 1"

        # 9. Strict FP32 state invariant for DeltaNet
        assert (
            self.deltanet_state_dtype.lower() == "float32"
        ), f"DeltaNet state dtype must be float32, got {self.deltanet_state_dtype}"

    def calculate_memory_budget(self) -> Dict[str, Union[int, float]]:
        """
        Calculates exact memory allocations across static weights, recurrent state,
        scratchpads, and KV cache, verifying that peak RAM usage is strictly < 750 MiB.
        Validates configuration invariants before calculating allocations.
        """
        self.validate()

        # 1. DeltaNet recurrent state (FP32)
        bytes_per_deltanet_layer = (
            self.deltanet_num_heads
            * self.deltanet_head_dim
            * self.deltanet_head_dim
            * 4  # FP32
        )
        bytes_deltanet_state = self.num_deltanet_layers * bytes_per_deltanet_layer

        # 2. Static weights:
        # Embeddings: vocab_size * d_model * 2 (BF16, tied LM head)
        bytes_static_embeddings = self.vocab_size * self.d_model * 2

        # BitNet b1.58 ternary weights (2 bits = 0.25 bytes per weight)
        deltanet_weights_per_layer = (
            4 * self.d_model * self.deltanet_inner_dim
            + self.deltanet_inner_dim * self.d_model
        )
        ffn_weights_per_layer = 3 * self.d_model * self.intermediate_dim
        global_qkv_dim = (self.global_attention_heads + 2 * self.global_kv_heads) * self.global_head_dim
        global_weights_per_layer = (global_qkv_dim + self.global_attention_heads * self.global_head_dim) * self.d_model

        total_ternary_weights_count = (
            self.num_deltanet_layers * deltanet_weights_per_layer
            + self.layers * ffn_weights_per_layer
            + self.num_global_layers * global_weights_per_layer
        )
        bytes_static_ternary_weights = int(math.ceil(total_ternary_weights_count * 0.25))

        # Hopfield Aux: Codebooks C1, C2 (BF16) + W_expand (BF16)
        bytes_static_hopfield_aux = int(
            2 * self.hopfield_num_groups * self.hopfield_codebook_size * self.hopfield_code_dim * 2
            + self.hopfield_val_dim * self.hopfield_expanded_dim * 2
        )

        # 3. Dynamic execution scratchpads (Rada.cpp Zero-Allocation)
        bytes_phase1_draft_scratch = 4194304      # 4.00 MiB
        bytes_phase2_refine_scratch = 8388608     # 8.00 MiB
        bytes_phase3_decode_workspace = 13417216  # 12.80 MiB
        kv_context_tokens = min(self.max_position_embeddings, 2048)
        bytes_global_kv_cache = 2 * self.num_global_layers * self.global_kv_heads * self.global_head_dim * kv_context_tokens * 1
        bytes_metadata_and_bpe = 2162688          # 2.06 MiB

        # 4. Total monolithic arena
        bytes_total_arena = (
            bytes_static_ternary_weights
            + bytes_static_embeddings
            + bytes_static_hopfield_aux
            + bytes_deltanet_state
            + bytes_phase1_draft_scratch
            + bytes_phase2_refine_scratch
            + bytes_phase3_decode_workspace
            + bytes_global_kv_cache
            + bytes_metadata_and_bpe
        )
        total_arena_mib = bytes_total_arena / (1024.0 * 1024.0)

        return {
            "bytes_deltanet_state": bytes_deltanet_state,
            "mib_deltanet_state": bytes_deltanet_state / (1024.0 * 1024.0),
            "bytes_static_ternary_weights": bytes_static_ternary_weights,
            "bytes_static_embeddings": bytes_static_embeddings,
            "bytes_static_hopfield_aux": bytes_static_hopfield_aux,
            "bytes_phase1_draft_scratch": bytes_phase1_draft_scratch,
            "bytes_phase2_refine_scratch": bytes_phase2_refine_scratch,
            "bytes_phase3_decode_workspace": bytes_phase3_decode_workspace,
            "bytes_global_kv_cache": bytes_global_kv_cache,
            "bytes_total_monolithic_arena": bytes_total_arena,
            "mib_total_monolithic_arena": total_arena_mib,
            "target_budget_limit_mib": 750.0,
            "budget_passed": total_arena_mib < 750.0,
        }

    def validate_memory_allocation(self, max_budget_mib: float = 750.0) -> Dict[str, Union[int, float]]:
        """
        Validates that the configuration memory footprint strictly fits within the
        specified edge memory budget (default: 750.0 MiB) before any tensor allocation
        is initiated. Raises ValueError if the budget is exceeded or invariants fail.
        """
        budget = self.calculate_memory_budget()
        total_mib = budget["mib_total_monolithic_arena"]
        if total_mib >= max_budget_mib:
            raise ValueError(
                f"Edge memory allocation budget exceeded: Required monolithic arena is "
                f"{total_mib:.2f} MiB, which exceeds the strict {max_budget_mib:.2f} MiB limit. "
                f"Aborting tensor allocation to prevent OOM/hardware degradation."
            )
        return budget

    @classmethod
    def create_mini(
        cls,
        vocab_size: int = 1024,
        d_model: int = 512,
        intermediate_dim: int = 1024,
        layers: int = 34,
        num_deltanet_layers: int = 28,
        num_global_layers: int = 6,
        deltanet_num_heads: int = 8,
        deltanet_head_dim: int = 32,
        global_attention_heads: int = 4,
        global_kv_heads: int = 2,
        global_head_dim: int = 128,  # 4 * 128 = 512 == d_model
        hadamard_block_size: int = 512,
        hopfield_num_groups: int = 4,
        hopfield_codebook_size: int = 64,
        hopfield_val_dim: int = 16,
        block_size: int = 32,
        overlap_size: int = 8,
        dim_intent: int = 64,
        dim_style: int = 32,
        rvq_codebook_size: int = 128,
        rvq_dim: int = 64,
        refiner_iterations: int = 3,
        **kwargs: Any,
    ) -> "Hetman2Config":
        """
        Creates a lightweight valid mini configuration for testing that passes all
        mathematical invariants and validation checks.
        """
        hopfield_slots = hopfield_num_groups * hopfield_codebook_size * hopfield_codebook_size
        dim_c_sem = d_model // 2
        dim_idx_hop = d_model // 4
        dim_tau_style = d_model // 8
        dim_ell_budget = d_model - (dim_c_sem + dim_idx_hop + dim_tau_style)

        return cls(
            vocab_size=vocab_size,
            d_model=d_model,
            intermediate_dim=intermediate_dim,
            layers=layers,
            num_deltanet_layers=num_deltanet_layers,
            num_global_layers=num_global_layers,
            deltanet_num_heads=deltanet_num_heads,
            deltanet_head_dim=deltanet_head_dim,
            deltanet_inner_dim=deltanet_num_heads * deltanet_head_dim,
            global_attention_heads=global_attention_heads,
            global_kv_heads=global_kv_heads,
            global_head_dim=global_head_dim,
            hadamard_block_size=min(hadamard_block_size, d_model),
            hopfield_num_groups=hopfield_num_groups,
            hopfield_codebook_size=hopfield_codebook_size,
            hopfield_slots=hopfield_slots,
            hopfield_val_dim=hopfield_val_dim,
            hopfield_expanded_dim=hopfield_val_dim * 2,
            hopfield_subquery_dim=hopfield_num_groups * 2 * 64,
            hopfield_topk_1=min(4, hopfield_codebook_size),
            hopfield_topk_2=min(2, hopfield_codebook_size),
            dim_c_sem=dim_c_sem,
            dim_idx_hop=dim_idx_hop,
            dim_tau_style=dim_tau_style,
            dim_ell_budget=dim_ell_budget,
            block_size=block_size,
            overlap_size=overlap_size,
            dim_intent=dim_intent,
            dim_style=dim_style,
            rvq_codebook_size=rvq_codebook_size,
            rvq_dim=rvq_dim,
            refiner_iterations=refiner_iterations,
            **kwargs,
        )

    def pack_rada_header_bytes(
        self,
        tensor_table_offset: int = 128,
        tensor_data_offset: int = 0,
        bpe_vocab_offset: int = 0,
        bpe_vocab_size_bytes: int = 0,
        header_crc32: int = 0,
        table_crc32: int = 0,
        flags: int = 1,
    ) -> bytes:
        """
        Packs the configuration into an exact 128-byte binary header conforming
        to `RadaNcmFileHeader` defined in `spec/rada_ncm_engine.hpp` and `src/export_to_rada.py`.

        Struct Layout:
          18 x uint32_t (72 bytes: architectural parameters)
           4 x uint64_t (32 bytes: section offsets and sizes)
           3 x uint32_t (12 bytes: header_crc32, table_crc32, flags)
          12 x uint8_t  (12 bytes: padding to 128 bytes)
          Total: exactly 128 bytes (strictly alignas(64)).
        """
        self.validate()
        header_bytes = struct.pack(
            "<18I4Q3I12s",
            int(self.rada_magic),
            int(self.rada_version),
            int(self.vocab_size),
            int(self.hidden_size),
            int(self.intermediate_size),
            int(self.num_hidden_layers),
            int(self.num_deltanet_layers),
            int(self.num_global_layers),
            int(self.deltanet_num_heads),
            int(self.deltanet_head_dim),
            int(self.global_attention_heads),
            int(self.global_kv_heads),
            int(self.global_head_dim),
            int(self.hopfield_num_groups),
            int(self.hopfield_codebook_size),
            int(self.hopfield_val_dim),
            int(self.block_size),
            int(self.max_blocks),
            int(tensor_table_offset),
            int(tensor_data_offset),
            int(bpe_vocab_offset),
            int(bpe_vocab_size_bytes),
            int(header_crc32),
            int(table_crc32),
            int(flags),
            b"\x00" * 12,
        )
        assert len(header_bytes) == 128, f"Rada header must be 128 bytes, got {len(header_bytes)}"
        return header_bytes

    def to_rada_header_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the header parameters for Rada export tools.
        """
        return {
            "magic": hex(self.rada_magic),
            "version": self.rada_version,
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "num_layers": self.num_hidden_layers,
            "num_deltanet_layers": self.num_deltanet_layers,
            "num_global_layers": self.num_global_layers,
            "deltanet_heads": self.deltanet_num_heads,
            "deltanet_head_dim": self.deltanet_head_dim,
            "global_heads_q": self.global_attention_heads,
            "global_heads_kv": self.global_kv_heads,
            "global_head_dim": self.global_head_dim,
            "hopfield_groups": self.hopfield_num_groups,
            "hopfield_codebook_size": self.hopfield_codebook_size,
            "hopfield_val_dim": self.hopfield_val_dim,
            "tokens_per_block": self.block_size,
            "max_blocks": self.max_blocks,
            "hopfield_slots": self.hopfield_slots,
            "base_passes": self.base_passes,
            "emergency_passes": self.emergency_passes,
            "rvq_codebooks": self.rvq_codebooks,
        }

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes this configuration instance to a dictionary, ensuring all canonical
        and dynamic parameters are included, and private runtime flags are omitted.
        """
        output = super().to_dict()
        output.pop("_is_initialized", None)
        return output


# Register model type with Hugging Face AutoConfig
AutoConfig.register("hetman2_ncm", Hetman2Config)
