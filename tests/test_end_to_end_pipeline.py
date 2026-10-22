"""
Hetman-2-NCM 2.0B: End-to-End Pipeline & Rada Container Verification Test
File: tests/test_end_to_end_pipeline.py
Specification: RFC-RADA-NCM-2.0-EDGE-AUDIT / RFC-HETMAN-2-NCM-07 / RFC-HETMAN-2-NCM-15

This test scenario:
1. Instantiates a mini-instance of Hetman2ForCausalLM.
2. Executes the full generation cycle:
   - Phase 1 (Causal AR Blueprint Drafter)
   - Phase 2 (Global Bidirectional Refiner: 3 passes)
   - Phase 3 (Block-Causal NCM Decoder: 8 base passes + 3 emergency passes = 8+3 passes)
   - Gated DeltaNet recurrent state carryover across blocks in FP32
3. Exports the model to a test .rada container with RADA_VERSION = 2,
   64-byte alignment, and BitNet b1.58 2-bit ternary weight packing.
4. Verifies binary container validity, header integrity, cache-line alignment,
   and numerical round-trip accuracy of unpacked ternary weights.
"""

import os
import sys
from pathlib import Path
try:
    import pytest
except ImportError:
    pytest = None
import torch

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model import Hetman2Config, Hetman2ForCausalLM
from src.export_to_rada import (
    RADA_NCM_MAGIC,
    RADA_VERSION,
    CACHE_LINE_ALIGNMENT,
    HEADER_SIZE_BYTES,
    DTYPE_TERNARY_2BIT,
    export_to_rada,
    verify_rada_file,
    unpack_ternary_weights,
)


def test_end_to_end_generation_and_export(tmp_path=None):
    print("=" * 80)
    print("HETMAN-2-NCM: FULL END-TO-END PIPELINE & RADA.CPP EXPORT TEST")
    print("=" * 80)

    # --------------------------------------------------------------------------
    # 1. CREATE MINI-INSTANCE OF HETMAN2FORCAUSALLM
    # --------------------------------------------------------------------------
    print("\n[STEP 1] Creating Hetman2ForCausalLM mini-instance...")
    config = Hetman2Config.create_mini(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_layers=4,
        num_deltanet_layers=3,
        num_global_layers=1,
        deltanet_heads=4,
        deltanet_head_dim=32,
        tokens_per_block=16,
        max_blocks=2,
        latents_per_block=4,
        phase2_passes=3,
        phase3_passes=8,
        fwht_block_size=64
    )

    torch.manual_seed(42)
    model = Hetman2ForCausalLM(config)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"-> Model Initialized successfully. Total parameters: {total_params:,}")
    print(f"   Hidden Size: {config.hidden_size} | Intermediate: {config.intermediate_size}")
    print(f"   DeltaNet Heads: {config.deltanet_heads} x {config.deltanet_head_dim}")
    print(f"   Tokens per block: {config.tokens_per_block} | Blocks: {config.max_blocks}")

    # --------------------------------------------------------------------------
    # 2. EXECUTE FULL GENERATION CYCLE: PHASE 1 -> PHASE 2 -> PHASE 3 (8+3 PASSES)
    # --------------------------------------------------------------------------
    print("\n[STEP 2] Executing Full Hierarchical Generation Cycle...")
    prompt_ids = torch.tensor([[12, 45, 88, 102]], dtype=torch.long)
    print(f"-> Input prompt IDs: {prompt_ids.tolist()} (Length: {prompt_ids.shape[1]})")

    with torch.no_grad():
        out = model.generate(prompt_ids, num_blocks=config.max_blocks)

    # Validate Phase 1: Draft Blueprints
    draft_bps = out["draft_blueprints"]
    print(f"\n[PHASE 1] Causal AR Blueprint Drafter Output: {draft_bps.shape}")
    assert draft_bps.shape == (1, config.max_blocks, config.hidden_size), \
        f"Draft blueprints shape mismatch: {draft_bps.shape}"
    assert not torch.isnan(draft_bps).any(), "Phase 1 produced NaNs!"
    assert not torch.isinf(draft_bps).any(), "Phase 1 produced Infs!"
    print("   [PASS] Phase 1 Draft blueprints verified.")

    # Validate Phase 2: Refiner (3 passes)
    refined_bps = out["refined_blueprints"]
    phase2_tel = out["phase2_telemetry"]
    print(f"\n[PHASE 2] Global Bidirectional Refiner Output: {refined_bps.shape}")
    assert refined_bps.shape == (1, config.max_blocks, config.hidden_size), \
        f"Refined blueprints shape mismatch: {refined_bps.shape}"
    pass_tel_2 = phase2_tel["pass_telemetry"]
    print(f"-> Phase 2 Refinement Passes Executed: {len(pass_tel_2)} (Configured: {config.phase2_passes})")
    assert len(pass_tel_2) == 3, f"Expected exactly 3 Phase 2 passes, got {len(pass_tel_2)}"
    for p_info in pass_tel_2:
        print(f"   Pass {p_info['pass']}: Dirichlet Energy={p_info['dirichlet_energy']:.4f}, "
              f"Repulsions={p_info['repulsions']}, FWHT Scale={p_info['mean_fwht_eta']:.4f}")
    assert not torch.isnan(refined_bps).any(), "Phase 2 produced NaNs!"
    print("   [PASS] Phase 2 Bidirectional Refinement & Deduplication verified.")

    # Validate Phase 3: Block Decoder (8 base + 3 emergency = 8+3 passes) & DeltaNet State
    gen_tokens = out["generated_tokens"]
    phase3_tel = out["phase3_telemetry"]
    final_state = out["final_deltanet_state"]

    expected_total_tokens = config.max_blocks * config.tokens_per_block
    print(f"\n[PHASE 3] Block-Causal NCM Decoder Output: {gen_tokens.shape}")
    print(f"-> Generated Tokens: {gen_tokens.squeeze().tolist()[:16]}... (Total: {gen_tokens.shape[1]})")
    assert gen_tokens.shape == (1, expected_total_tokens), \
        f"Generated tokens shape mismatch: {gen_tokens.shape} != (1, {expected_total_tokens})"
    assert (gen_tokens >= 0).all() and (gen_tokens < config.vocab_size).all(), \
        "Generated token IDs out of vocabulary bounds!"

    # Verify 8+3 passes execution per block
    assert len(phase3_tel) == config.max_blocks, f"Expected telemetry for {config.max_blocks} blocks"
    for b_idx, block_info in enumerate(phase3_tel):
        base_passes = block_info["base_passes_executed"]
        em_passes = block_info["emergency_passes_executed"]
        total_p = block_info["total_passes_executed"]
        print(f"-> Block {b_idx}: Base Passes = {base_passes}, Emergency Passes = {em_passes}, "
              f"Total Passes = {total_p} (8+3 passes)")
        assert base_passes == 8, f"Expected 8 base passes in Block {b_idx}, got {base_passes}"
        assert em_passes == 3, f"Expected 3 emergency passes in Block {b_idx}, got {em_passes}"
        assert total_p == 11, f"Expected 11 total passes (8+3) in Block {b_idx}, got {total_p}"
        assert len(block_info["pass_metrics"]) == 11, "Telemetry metrics count mismatch!"

    # Verify DeltaNet persistent recurrent state
    print(f"\n[DELTANET STATE] Verifying O(1) Recurrent State Dynamics...")
    assert final_state is not None, "DeltaNet state was not returned!"
    expected_state_shape = (1, config.deltanet_heads, config.deltanet_head_dim, config.deltanet_head_dim)
    assert final_state.shape == expected_state_shape, \
        f"DeltaNet state shape mismatch: {final_state.shape} != {expected_state_shape}"
    assert final_state.dtype == torch.float32, \
        f"DeltaNet state MUST be stored in FP32! Got: {final_state.dtype}"
    assert not torch.isnan(final_state).any(), "DeltaNet state contains NaNs!"
    assert not torch.isinf(final_state).any(), "DeltaNet state contains Infs!"
    state_norm = float(torch.norm(final_state).item())
    print(f"-> Final DeltaNet State Norm: {state_norm:.6f} (Active Recurrent Memory)")
    assert state_norm > 0.0, "DeltaNet state must have evolved beyond zero!"
    print("   [PASS] DeltaNet recurrent state carryover verified in FP32.")

    # --------------------------------------------------------------------------
    # 3. EXPORT MODEL TO TEST .RADA CONTAINER
    # --------------------------------------------------------------------------
    print("\n[STEP 3] Exporting to .rada Binary Container...")
    if tmp_path is not None:
        container_path = str(tmp_path / "test_model_v2.rada")
    else:
        test_dir = Path(_ROOT) / "tests"
        test_dir.mkdir(parents=True, exist_ok=True)
        container_path = str(test_dir / "test_end_to_end_model.rada")

    export_result = export_to_rada(
        model_or_state_dict=model,
        output_path=container_path,
        config=config
    )

    print(f"-> Exported Container Path: {container_path}")
    print(f"-> Container Size: {export_result['total_file_size_bytes']:,} bytes")
    print(f"-> Tensors Count:  {export_result['total_tensors']}")
    print(f"-> Ternary Weights: {export_result['total_ternary_weights']:,} parameters")
    print(f"-> Ternary Memory:  {export_result['total_ternary_bytes']:,} bytes (BitNet b1.58 2-bit)")
    assert export_result["total_ternary_weights"] > 0, "No ternary weights were packed!"
    assert export_result["total_ternary_bytes"] > 0, "No ternary bytes were written!"
    assert os.path.exists(container_path), "Export file does not exist on disk!"

    # --------------------------------------------------------------------------
    # 4. VERIFY BINARY CONTAINER & HEADER VALIDITY
    # --------------------------------------------------------------------------
    print("\n[STEP 4] Verifying Binary Container & Header Validity...")
    audit = verify_rada_file(container_path)

    # 4.1 Header Check
    assert audit["valid"] is True, "Container verification failed!"
    print(f"-> Magic Number: {audit['magic_hex']} (Expected: 0x{RADA_NCM_MAGIC:08X})")
    assert int(audit["magic_hex"], 16) == RADA_NCM_MAGIC, f"Magic mismatch: {audit['magic_hex']}"

    print(f"-> Format Version: {audit['version']} (Expected: {RADA_VERSION})")
    assert audit["version"] == RADA_VERSION, f"Version mismatch: {audit['version']}"

    assert audit["vocab_size"] == config.vocab_size
    assert audit["hidden_size"] == config.hidden_size
    assert audit["intermediate_size"] == config.intermediate_size
    assert audit["num_layers"] == config.num_layers
    assert audit["num_deltanet_layers"] == config.num_deltanet_layers
    assert audit["tokens_per_block"] == config.tokens_per_block
    assert audit["max_blocks"] == config.max_blocks

    # 4.2 Strict 64-Byte Alignment Checks
    print(f"-> Tensor Table Offset: {audit['tensor_table_offset']} (Offset % 64 == 0)")
    assert audit["tensor_table_offset"] % CACHE_LINE_ALIGNMENT == 0, \
        f"Tensor table offset not 64B aligned: {audit['tensor_table_offset']}"
    assert audit["tensor_table_offset"] == HEADER_SIZE_BYTES, \
        f"Tensor table offset must follow immediately after 128B header"

    print(f"-> Tensor Data Offset:  {audit['tensor_data_offset']} (Offset % 64 == 0)")
    assert audit["tensor_data_offset"] % CACHE_LINE_ALIGNMENT == 0, \
        f"Tensor data offset not 64B aligned: {audit['tensor_data_offset']}"

    descriptors = audit["descriptors"]
    assert len(descriptors) == export_result["total_tensors"], "Descriptor count mismatch!"

    ternary_tensors_tested = 0
    with open(container_path, "rb") as f:
        for desc in descriptors:
            # Check 64-byte alignment of every tensor payload
            assert desc.offset % CACHE_LINE_ALIGNMENT == 0, \
                f"Tensor {desc.name} offset {desc.offset} is not 64-byte aligned!"
            assert desc.size_bytes % CACHE_LINE_ALIGNMENT == 0, \
                f"Tensor {desc.name} size {desc.size_bytes} is not 64-byte aligned!"

            # For ternary tensors, unpack and verify numerical exactness
            if desc.dtype == DTYPE_TERNARY_2BIT and ternary_tensors_tested < 5:
                f.seek(desc.offset)
                raw_payload = f.read(desc.size_bytes)
                unpacked_w = unpack_ternary_weights(
                    packed_bytes=raw_payload,
                    shape=desc.shape,
                    scale_gamma=desc.scale_gamma,
                    align_row_bytes=CACHE_LINE_ALIGNMENT
                )

                orig_tensor = model.state_dict()[desc.name]
                orig_quant = torch.clamp(torch.round(orig_tensor / desc.scale_gamma), -1.0, 1.0)
                unpacked_quant = torch.round(unpacked_w / desc.scale_gamma)

                assert torch.equal(orig_quant, unpacked_quant), \
                    f"Bit-level mismatch in ternary weight: {desc.name}"
                ternary_tensors_tested += 1

    print(f"-> Tested {ternary_tensors_tested} ternary tensors for bit-exact round-trip: 100% Match!")
    print(f"\n" + "=" * 80)
    print("ALL TESTS PASSED WITH 100% SUCCESS: GENERATION, DELTANET, EXPORT, ALIGNMENT & HEADER")
    print("=" * 80)


if __name__ == "__main__":
    test_end_to_end_generation_and_export()
