"""
Hetman-2-NCM Source Package
"""
from .configuration_hetman2 import Hetman2Config

from .deltanet import (
    DeltaNetConfig,
    DeltaNetBridgeConfig,
    LayerRecurrentState,
    DeltaNetStateBundle,
    BlockCausalStateBundle,
    StateSnapshotManager,
    BitLinear,
    GatedDeltaNetLayer,
    GatedDeltaNet,
    GatedDeltaNetBlockCausalBridge,
    DeltaNetBlockCausalBridge,
    DeltaNetStack,
    compute_deltanet_memory_footprint,
)
from .concept_codebook import (
    SemanticScale,
    CodebookConfig,
    RVQLevelOutput,
    MultiScaleRVQOutput,
    VectorQuantizerEMA,
    MultiScaleResidualVectorQuantizer,
    ConceptCodebook,
)

from .block_ncm_decoder import (
    DecoderConfig,
    RMSNorm,
    BitLinear158,
    VectorQuantizer,
    DeltaNetRecurrentState,
    ConfidenceGate,
    BlockDecoderLayer,
    BlockNCMDecoder,
    seamless_bpe_stitch,
    run_phase3_autonomous_test,
)

from .hopfield_core import (
    SLHCoreConfig,
    SLHCoreNCMAdapter,
    SpeculativeLatentHopfieldCore,
)

from .modeling_hetman2 import (
    Hetman2Config as MonolithicHetman2Config,
    Hetman2PreTrainedModel,
    Hetman2Model,
    Hetman2ForCausalLM,
    MODE_DRAFT,
    MODE_REFINE,
    MODE_DECODE,
    run_end_to_end_verification,
)
from .export_to_rada import (
    RadaExporter,
    export_to_rada,
    verify_rada_file,
    pack_ternary_weights,
    unpack_ternary_weights,
    RADA_VERSION,
    RADA_NCM_MAGIC,
    CACHE_LINE_ALIGNMENT,
)
from .blueprint_drafter import (
    BlueprintConfig,
    BlueprintComponents,
    BlockBlueprint,
    GatedDeltaNetCell,
    BlueprintIngestion,
    BlueprintPredictionHeads,
    CausalBlueprintDrafter,
    run_blueprint_drafter_benchmarks,
)

__all__ = [
    "Hetman2Config",
    "DeltaNetConfig",
    "DeltaNetBridgeConfig",
    "LayerRecurrentState",
    "DeltaNetStateBundle",
    "BlockCausalStateBundle",
    "StateSnapshotManager",
    "BitLinear",
    "GatedDeltaNetLayer",
    "GatedDeltaNet",
    "GatedDeltaNetBlockCausalBridge",
    "DeltaNetBlockCausalBridge",
    "DeltaNetStack",
    "compute_deltanet_memory_footprint",
    "SemanticScale",
    "CodebookConfig",
    "RVQLevelOutput",
    "MultiScaleRVQOutput",
    "VectorQuantizerEMA",
    "MultiScaleResidualVectorQuantizer",
    "ConceptCodebook",
    "DecoderConfig",
    "RMSNorm",
    "BitLinear158",
    "VectorQuantizer",
    "DeltaNetRecurrentState",
    "ConfidenceGate",
    "BlockDecoderLayer",
    "BlockNCMDecoder",
    "seamless_bpe_stitch",
    "run_phase3_autonomous_test",
    "SLHCoreConfig",
    "SLHCoreNCMAdapter",
    "SpeculativeLatentHopfieldCore",
    "MonolithicHetman2Config",
    "Hetman2PreTrainedModel",
    "Hetman2Model",
    "Hetman2ForCausalLM",
    "MODE_DRAFT",
    "MODE_REFINE",
    "MODE_DECODE",
    "run_end_to_end_verification",
    "RadaExporter",
    "export_to_rada",
    "verify_rada_file",
    "pack_ternary_weights",
    "unpack_ternary_weights",
    "RADA_VERSION",
    "RADA_NCM_MAGIC",
    "CACHE_LINE_ALIGNMENT",
    "BlueprintConfig",
    "BlueprintComponents",
    "BlockBlueprint",
    "GatedDeltaNetCell",
    "BlueprintIngestion",
    "BlueprintPredictionHeads",
    "CausalBlueprintDrafter",
    "run_blueprint_drafter_benchmarks",
]



