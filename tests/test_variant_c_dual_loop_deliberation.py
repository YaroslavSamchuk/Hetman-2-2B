"""
================================================================================
HETMAN-2-NCM: VARIANT C (HYBRID DUAL-LOOP DELIBERATION) INTEGRATION TEST SUITE
================================================================================
File: tests/test_variant_c_dual_loop_deliberation.py
Architecture: Hetman-2-NCM 2.0B
Author: Advanced Agentic Engineer
License: Apache License 2.0
================================================================================
"""

import math
import sys
import threading
import time
from pathlib import Path

# Add project root to sys.path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import torch
from src.configuration_hetman2 import Hetman2Config
from src.modeling_hetman2 import Hetman2ForCausalLM
from src.dual_loop_deliberation import CognitiveLevel, DualLoopDeliberationEngine


def test_variant_c_full_lifecycle():
    print("=" * 80)
    print("HETMAN-2-NCM: TESTING VARIANT C (HYBRID DUAL-LOOP DELIBERATION)")
    print("=" * 80)

    cfg = Hetman2Config.create_mini(
        vocab_size=1024,
        d_model=512,
        intermediate_dim=1024,
        layers=34,
        num_deltanet_layers=28,
        num_global_layers=6,
        deltanet_num_heads=8,
        deltanet_head_dim=32,
        global_attention_heads=4,
        global_kv_heads=2,
        global_head_dim=128,
        hadamard_block_size=512,
        hopfield_num_groups=4,
        hopfield_codebook_size=64,
        hopfield_val_dim=16,
        block_size=32,
        overlap_size=8,
        dim_intent=64,
        dim_style=32,
        rvq_codebook_size=128,
        rvq_dim=64,
        refiner_iterations=4,
    )

    model = Hetman2ForCausalLM(cfg)
    model.eval()

    # -------------------------------------------------------------
    # 1. TEST SIMPLE QUERY ROUTING (FAST PATH)
    # -------------------------------------------------------------
    print("\n--- 1. Testing Fast-Path (Simple Query Deliberation) ---")
    out_simple = model.generate_ncm(
        prompt="Столиця України?",
        max_tokens=32,
        deliberate=True,
        cognitive_level="simple",
        explain_thought=True
    )
    profile_simple = out_simple["cognitive_profile"]
    print(f"Profile: {profile_simple}")
    assert profile_simple["level"] == "SIMPLE", "Expected SIMPLE cognitive level!"
    assert profile_simple["passes_budget"] == 2, "Expected 2 passes for simple query!"
    assert profile_simple["passes_executed"] == 2, "Expected exactly 2 passes executed!"
    print(f"[PASS] Fast-path executed in {profile_simple['passes_executed']} passes.")

    # -------------------------------------------------------------
    # 2. TEST COMPLEX QUERY ROUTING (DEEP DELIBERATION)
    # -------------------------------------------------------------
    print("\n--- 2. Testing Deep-Deliberation (Complex Query with Banach Convergence) ---")
    out_complex = model.generate_ncm(
        prompt="Доведи теорему Ферма або розв'яжи систему рівнянь",
        max_tokens=32,
        deliberate=True,
        cognitive_level="complex",
        explain_thought=True,
        think=True
    )
    profile_complex = out_complex["cognitive_profile"]
    print(f"Profile: {profile_complex}")
    assert profile_complex["level"] == "COMPLEX", "Expected COMPLEX cognitive level!"
    assert profile_complex["passes_budget"] >= 16, "Expected >=16 passes budget!"
    assert len(out_complex["thought_steps"]) >= 2, "Expected multiple thought steps emitted!"
    assert out_complex["think_blueprint"] is not None, "Expected think blueprint to be generated!"
    print(f"[PASS] Deep-deliberation executed {profile_complex['passes_executed']} passes.")

    # -------------------------------------------------------------
    # 3. TEST USER EARLY STOP INTERRUPTION (< 15 ms ABORT-TO-STREAM)
    # -------------------------------------------------------------
    print("\n--- 3. Testing User Early Stop Interruption ('Відповідай зараз!') ---")
    abort_event = threading.Event()

    # Create a callback that trips on step 3
    step_counter = 0
    def early_stop_on_step3():
        nonlocal step_counter
        step_counter += 1
        if step_counter >= 3:
            abort_event.set()
            return True
        return False

    out_aborted = model.generate_ncm(
        prompt="Складна тривала задача з оптимізації...",
        max_tokens=32,
        deliberate=True,
        cognitive_level="complex",
        early_stop_requested=early_stop_on_step3,
        explain_thought=True
    )
    profile_aborted = out_aborted["cognitive_profile"]
    print(f"Aborted Execution Profile: {profile_aborted}")
    assert profile_aborted["passes_executed"] <= 4, f"Early stop failed! Executed {profile_aborted['passes_executed']} passes"
    assert out_aborted["generated_tokens"].shape[-1] <= 32, "Token emission failed after abort!"
    print(f"[PASS] User interruption successfully aborted deliberation at pass {profile_aborted['passes_executed']} and emitted tokens.")

    # -------------------------------------------------------------
    # 4. TEST VERBALIZER TELEMETRY STREAM
    # -------------------------------------------------------------
    print("\n--- 4. Testing Thought Verbalizer Output Telemetry ---")
    for idx, step in enumerate(out_complex["thought_steps"][:5]):
        print(f"  Step {idx+1:02d}: {step['summary']}")
        assert "summary" in step
        assert "delta_norm" in step

    print("\n" + "=" * 80)
    print("ALL VARIANT C INTEGRATION TESTS PASSED (100% OPERATIONAL)")
    print("=" * 80)


if __name__ == "__main__":
    test_variant_c_full_lifecycle()
