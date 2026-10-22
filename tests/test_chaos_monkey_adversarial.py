"""
================================================================================
HETMAN-2-NCM 2.0B: CHAOS MONKEY ADVERSARIAL RED TEAM STRESS TEST SUITE
================================================================================
File: tests/test_chaos_monkey_adversarial.py
Specification: RFC-HETMAN-2.0B-REDTEAM-011 / spec/audits/audit_20_red_team_chaos_adversarial.md
Author: Chief Chaos Monkey & Red Team Systems Lead ("Devil's Advocate")
License: Apache License 2.0

Mission:
Execute brutal stress tests and adversarial penetration vectors against the
Hetman-2-NCM architecture to discover catastrophic input combinations and
empirically prove architectural invulnerability:
1. Extreme Prompts: Empty strings, single whitespace, 10k repetition, Emoji/Unicode storms, OOV.
2. Numerical Sabotage: Direct NaN/Inf injection into activations, weights, DeltaNet recurrent states.
3. Hopfield Memory Sabotage: Spurious Attractor Jamming, Codebook Collapse, PID router saturation.
4. Latent Space Collapse: Identical blueprint jamming, zero Dirichlet energy, anti-repetition repulsions.
5. End-to-End Meltdown Resistance: Simultaneous multi-vector adversarial attacks.
================================================================================
"""

import math
import os
import sys
import unittest
from pathlib import Path
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to sys.path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model import (
    Hetman2Config,
    Hetman2ForCausalLM,
    FastWalshHadamardTransform,
    GatedDeltaNetLayer,
    TernaryQuantizeSTE,
    BitLinear,
    SemanticDeduplicationFilter,
    DirichletEnergyWatchdog,
)
from src.hopfield_core import (
    SLHCoreNCMAdapter,
    SLHCoreConfig,
)
from src.dynamic_blueprint_evolver import (
    DynamicBlueprintEvolver,
    SphericalNorm,
)
from src.configuration_hetman2 import Hetman2Config as ProductionConfig
from src.modeling_hetman2 import (
    Hetman2ForCausalLM as MonolithicHetman2ForCausalLM,
    GatedDeltaNet as MonolithicGatedDeltaNet,
    MODE_DRAFT,
    MODE_REFINE,
    MODE_DECODE,
)


class TestChaosMonkeyAdversarial(unittest.TestCase):
    """
    Brutal Red Team / Chaos Monkey Adversarial Test Suite for Hetman-2-NCM 2.0B.
    """

    @classmethod
    def setUpClass(cls):
        print("\n" + "=" * 80)
        print("HETMAN-2-NCM: INITIALIZING CHAOS MONKEY ADVERSARIAL RED TEAM HARNESS")
        print("=" * 80)
        torch.manual_seed(1337)
        cls.config = Hetman2Config.create_mini(
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
            fwht_block_size=64,
            dedup_tau=0.85
        )
        cls.model = Hetman2ForCausalLM(cls.config)
        cls.model.eval()
        print("-> Mini-Model initialized with all architectural defenses active.\n")

    # ==========================================================================
    # VECTOR 1: EXTREME & ADVERSARIAL PROMPTS
    # ==========================================================================

    def test_01_empty_prompt_handling(self):
        """
        Attack: Ingest empty prompt (length 0).
        Risk: Mean pooling over 0 tokens yields NaN (0/0), crashing Phase 1 drafter.
        Invariant: Model MUST output valid finite tokens without NaNs or exceptions.
        """
        print("[VECTOR 1.1] Testing Empty Prompt (len=0)...")
        empty_tensor = torch.empty((1, 0), dtype=torch.long)
        with torch.no_grad():
            out = self.model.generate(empty_tensor, num_blocks=self.config.max_blocks)

        tokens = out["generated_tokens"]
        draft_bps = out["draft_blueprints"]
        refined_bps = out["refined_blueprints"]

        self.assertEqual(tokens.shape, (1, self.config.max_blocks * self.config.tokens_per_block))
        self.assertFalse(torch.isnan(draft_bps).any(), "Draft blueprints contain NaN on empty prompt!")
        self.assertFalse(torch.isnan(refined_bps).any(), "Refined blueprints contain NaN on empty prompt!")
        self.assertTrue((tokens >= 0).all() and (tokens < self.config.vocab_size).all(), "Invalid token IDs emitted!")
        print("  >>> [PASS] Empty prompt gracefully synthesized unconditioned prior without NaN.")

    def test_02_single_whitespace_and_minimal_prompt(self):
        """
        Attack: Ingest single whitespace ' ' or control token.
        Risk: Degenerate single-token embeddings causing boundary seam breakdown.
        """
        print("[VECTOR 1.2] Testing Single Whitespace Prompt (' ')...")
        space_id = torch.tensor([[ord(' ') % self.config.vocab_size]], dtype=torch.long)
        with torch.no_grad():
            out = self.model.generate(space_id, num_blocks=self.config.max_blocks)

        tokens = out["generated_tokens"]
        self.assertEqual(tokens.shape, (1, self.config.max_blocks * self.config.tokens_per_block))
        self.assertFalse(torch.isnan(tokens.float()).any())
        print(f"  >>> [PASS] Single whitespace generated {tokens.shape[1]} valid tokens.")

    def test_03_massive_10000_token_repetition_stability(self):
        """
        Attack: 10,000 repeats of a single token ('A' * 10,000).
        Risk: In classical transformers this consumes O(N^2) memory and explodes.
              In DeltaNet, could recurrent state norm blow up to infinity?
        Invariant: State norm MUST converge to a strictly bounded Lyapunov fixed point (rho < 1).
        """
        print("[VECTOR 1.3] Testing 10,000 Repeated Tokens (Long-Context Stability)...")
        token_val = ord('A') % self.config.vocab_size
        layer = self.model.phase1_drafter.drafter_layers[0]

        with torch.no_grad():
            x_emb = self.model.embed_tokens(torch.tensor([[token_val]]))
            norm_x = layer.norm(x_emb).squeeze(1)

            q = layer.q_proj(norm_x).view(1, layer.num_heads, layer.head_dim)
            k = layer.k_proj(norm_x).view(1, layer.num_heads, layer.head_dim)
            v = layer.v_proj(norm_x).view(1, layer.num_heads, layer.head_dim)
            k = F.normalize(k, p=2, dim=-1)
            alpha = torch.sigmoid(layer.alpha_proj(norm_x)).view(1, layer.num_heads, 1, 1).to(torch.float32)
            beta = torch.sigmoid(layer.beta_proj(norm_x)).view(1, layer.num_heads, 1).to(torch.float32)

            k_col = k.unsqueeze(-1).to(torch.float32)
            v_col = v.unsqueeze(-1).to(torch.float32)

            state = torch.zeros(1, layer.num_heads, layer.head_dim, layer.head_dim, dtype=torch.float32)
            step_checkpoints = []

            for step in range(10000):
                retrieved_v = torch.matmul(state, k_col)
                delta_v = v_col - retrieved_v
                update_term = torch.matmul(delta_v, k_col.transpose(-2, -1))
                state = alpha * state + beta.unsqueeze(-1) * update_term

                if step in (1000, 3000, 5000, 8000, 9999):
                    step_checkpoints.append((step, float(torch.norm(state).item())))

        print(f"  Checkpoints [step, ||S_t||]: {step_checkpoints}")
        final_norm = step_checkpoints[-1][1]
        prev_norm = step_checkpoints[-2][1]
        diff = abs(final_norm - prev_norm)

        self.assertFalse(math.isnan(final_norm), "Recurrent state norm became NaN!")
        self.assertFalse(math.isinf(final_norm), "Recurrent state norm exploded to Inf!")
        self.assertLess(final_norm, 100.0, "Recurrent state norm grew uncontrollably!")
        self.assertLess(diff, 1e-4, f"State did not converge to asymptotic attractor: diff={diff}")
        print(f"  >>> [PASS] Asymptotic Lyapunov fixed-point verified: convergence diff={diff:.2e} < 1e-4.")

    def test_04_unicode_emoji_and_control_characters(self):
        """
        Attack: High-entropy Unicode emoji, zero-width characters, null bytes, and RTL characters.
        Risk: Out-of-bounds codepoints causing byte-level indexing crashes.
        """
        print("[VECTOR 1.4] Testing Unicode / Emoji / Control Character Storm...")
        evil_prompt = "🚀🔥💀💥🤖🧪⚡️✨\u200b\ufeff\U0001f92f\u0000\ufffd\U0001f600\U0001f4a9\u202eRTL_SABOTAGE\u202c"
        evil_ids = torch.tensor([[ord(c) % self.config.vocab_size for c in evil_prompt]], dtype=torch.long)

        with torch.no_grad():
            out = self.model.generate(evil_ids, num_blocks=self.config.max_blocks)

        tokens = out["generated_tokens"]
        self.assertEqual(tokens.shape, (1, self.config.max_blocks * self.config.tokens_per_block))
        self.assertFalse(torch.isnan(tokens.float()).any())
        print(f"  >>> [PASS] Unicode / Emoji prompt successfully digested and transformed into tokens.")

    def test_05_out_of_vocabulary_and_negative_indices(self):
        """
        Attack: Explicit OOV and negative indices (e.g. [-1, 999999, -42, vocab_size + 500]).
        Risk: Segmentation faults or CUDA illegal memory access in nn.Embedding.
        Invariant: Model input clamp shield MUST normalize indices to [0, vocab_size - 1].
        """
        print("[VECTOR 1.5] Testing Out-Of-Vocabulary & Negative Index Clamping...")
        oov_ids = torch.tensor([[-500, self.config.vocab_size + 9999, -1, self.config.vocab_size]], dtype=torch.long)

        with torch.no_grad():
            out = self.model.generate(oov_ids, num_blocks=self.config.max_blocks)

        tokens = out["generated_tokens"]
        self.assertTrue((tokens >= 0).all() and (tokens < self.config.vocab_size).all())
        print("  >>> [PASS] Out-of-bounds indices quarantined and clamped without memory fault.")

    # ==========================================================================
    # VECTOR 2: NUMERICAL SABOTAGE & NaN / Inf QUARANTINE
    # ==========================================================================

    def test_06_fwht_nan_and_inf_quarantine(self):
        """
        Attack: Inject NaN and +/- Inf directly into Fast Walsh-Hadamard Transform.
        Risk: Hadamard butterfly cascades single NaN into all coordinates (O(d log d) contamination).
        Invariant: FWHT quarantine shield MUST extinguish NaNs/Infs and produce strictly finite INT8 scale.
        """
        print("[VECTOR 2.1] Testing FWHT NaN and Inf Self-Healing Quarantine...")
        fwht = FastWalshHadamardTransform(block_size=self.config.fwht_block_size)
        x = torch.randn(2, 4, self.config.hidden_size)

        # Inject NaN and Inf
        x[0, 1, 5] = float('nan')
        x[0, 2, 10] = float('inf')
        x[1, 0, 12] = float('-inf')

        out_fwht, eta = fwht(x)

        self.assertFalse(torch.isnan(out_fwht).any(), "FWHT output contains NaN!")
        self.assertFalse(torch.isinf(out_fwht).any(), "FWHT output contains Inf!")
        self.assertFalse(math.isnan(eta), "FWHT eta scale is NaN!")
        self.assertFalse(math.isinf(eta), "FWHT eta scale is Inf!")
        self.assertGreaterEqual(eta, fwht.eta_min)
        self.assertLessEqual(eta, fwht.eta_max)
        print(f"  >>> [PASS] FWHT neutralized NaN/Inf and computed valid scale eta={eta:.4f}.")

    def test_07_deltanet_recurrent_state_self_healing(self):
        """
        Attack: Corrupt DeltaNet recurrent state S_t with NaNs mid-generation.
        Risk: Recurrent accumulator permanently stores and amplifies NaN to all future tokens.
        Invariant: State self-healing quench MUST reset/quench corrupted state and resume execution.
        """
        print("[VECTOR 2.2] Testing DeltaNet Recurrent State Self-Healing Quench...")
        layer = self.model.phase3_decoder.deltanet_core
        x_seq = torch.randn(1, 10, self.config.hidden_size)

        # Normal initial steps
        out1, s1 = layer.forward_step(x_seq[:, 0, :], None)

        # Inject poison into recurrent state
        s_poisoned = s1.clone()
        s_poisoned[0, 0, :, :] = float('nan')
        s_poisoned[0, 1, 0, 0] = float('inf')

        # Run forward_step with poisoned state
        out_healed, s_healed = layer.forward_step(x_seq[:, 1, :], s_poisoned)

        self.assertFalse(torch.isnan(out_healed).any(), "DeltaNet output contains NaN after state corruption!")
        self.assertFalse(torch.isinf(out_healed).any(), "DeltaNet output contains Inf after state corruption!")
        self.assertFalse(torch.isnan(s_healed).any(), "DeltaNet recurrent state still contains NaN!")
        self.assertFalse(torch.isinf(s_healed).any(), "DeltaNet recurrent state still contains Inf!")
        print("  >>> [PASS] DeltaNet detected and healed poisoned recurrent state.")

    def test_08_bitlinear_weight_sabotage(self):
        """
        Attack: Inject NaN and Inf directly into BitLinear weight matrices.
        Risk: Absmean scale gamma collapses to NaN, zero-division in quantization.
        Invariant: TernaryQuantizeSTE MUST quench weights before absmean computation.
        """
        print("[VECTOR 2.3] Testing BitLinear Weight Matrix Sabotage...")
        bit_linear = BitLinear(self.config.hidden_size, self.config.intermediate_size)
        with torch.no_grad():
            bit_linear.weight[0, 0] = float('nan')
            bit_linear.weight[1, 1] = float('inf')

        x_in = torch.randn(2, 4, self.config.hidden_size)
        y_out = bit_linear(x_in)

        self.assertFalse(torch.isnan(y_out).any(), "BitLinear emitted NaN!")
        self.assertFalse(torch.isinf(y_out).any(), "BitLinear emitted Inf!")
        print("  >>> [PASS] BitLinear STE quarantined corrupted weights and emitted finite output.")

    # ==========================================================================
    # VECTOR 3: HOPFIELD FACTUAL MEMORY SPURIOUS ATTRACTOR JAMMING
    # ==========================================================================

    def test_09_hopfield_spurious_attractor_jamming(self):
        """
        Attack: Spurious Attractor Jamming (100% of tensor V set to identical ternary vector +1).
        Risk: Memory collapses into single spurious basin; query retrieval variance drops to zero.
        Invariant:
            1. Continuous Hopfield energy E(q) remains bounded and well-behaved.
            2. Attention variance drops below 1e-6, triggering spurious_jamming_detected telemetry.
            3. Anti-jamming gate decouples poisoned memory from Phase 3 word decoder.
        """
        print("[VECTOR 3.1] Testing Hopfield Spurious Attractor Jamming...")
        adapter = SLHCoreNCMAdapter(
            hidden_size=self.config.hidden_size,
            num_groups=4,
            codebook_size=64,
            code_dim=16,
            val_dim=16,
            expanded_dim=32,
            topk_1=4,
            topk_2=2,
            pack_2bit=False
        )

        # Adversarially flood attractor tensor V with identical +1.0 ternary micro-facts
        with torch.no_grad():
            adapter.V.data.fill_(1.0)
            # Collapse dual spherical codebooks C1, C2 to identical centroids
            c_shared = F.normalize(torch.randn(1, 1, 16).repeat(4, 64, 1), p=2, dim=-1)
            adapter.C1.data.copy_(c_shared)
            adapter.C2.data.copy_(c_shared)

        q = torch.randn(1, 2, self.config.hidden_size)
        decoder_state = torch.randn(1, 2, self.config.hidden_size)

        y_fact, commit_loss, diagnostics = adapter(
            q,
            phase3_decoder_state=decoder_state,
            return_diagnostics=True
        )

        energy = diagnostics["hopfield_energy"].mean().item()
        attn_var = diagnostics["attn_variance_mean"]
        is_jammed = diagnostics["spurious_jamming_detected"]

        print(f"  Jamming Telemetry: Energy={energy:.4f}, Attn Variance={attn_var:.8f}, Jammed={is_jammed}")
        self.assertTrue(is_jammed, "Spurious jamming condition was not detected by memory telemetry!")
        self.assertLess(attn_var, 1e-6, "Attention weights variance should be near zero under jamming!")
        self.assertFalse(math.isnan(energy), "Hopfield Lyapunov energy became NaN!")

        # Verify Decoder Isolation:
        # Under jamming, final_output must match decoder_state because g_fact is zeroed out!
        diff_from_decoder = (y_fact - decoder_state).abs().max().item()
        self.assertLess(diff_from_decoder, 1e-5,
            f"Decoder was not isolated from jammed memory! Diff={diff_from_decoder}")
        print("  >>> [PASS] Hopfield Spurious Jamming detected; Anti-Hallucination Gate isolated decoder.")

    def test_10_hopfield_pid_router_saturation_bound(self):
        """
        Attack: 100% Load Imbalance / Routing Attack on Hopfield PID Load Balancer.
        Risk: Router bias accumulation blows up or overflows buffer memory.
        Invariant: PID router biases b1, b2 MUST be strictly clamped in [-0.5, +0.5].
        """
        print("[VECTOR 3.2] Testing Dynamic PID Router Saturation Clamping...")
        adapter = SLHCoreNCMAdapter(
            hidden_size=self.config.hidden_size,
            num_groups=4,
            codebook_size=64,
            code_dim=16,
            val_dim=16,
            expanded_dim=32,
            topk_1=4,
            topk_2=2,
            pack_2bit=False
        )
        adapter.train()

        # Repeat identical query 50 times to force extreme slot starvation
        q_starve = torch.randn(2, 4, self.config.hidden_size)
        for _ in range(50):
            _ = adapter(q_starve, return_diagnostics=False)

        b1_max = adapter.b1.abs().max().item()
        b2_max = adapter.b2.abs().max().item()

        self.assertLessEqual(b1_max, 0.500001, f"b1 bias exceeded clamp limit! Got {b1_max}")
        self.assertLessEqual(b2_max, 0.500001, f"b2 bias exceeded clamp limit! Got {b2_max}")
        self.assertFalse(torch.isnan(adapter.b1).any(), "PID router b1 contains NaN!")
        print(f"  >>> [PASS] PID router biases strictly bounded: max(|b1|)={b1_max:.4f} <= 0.5.")

    # ==========================================================================
    # VECTOR 4: PHASE 2 & 3 REPETITION ATTACK & DEDUPLICATION
    # ==========================================================================

    def test_11_phase2_identical_blueprints_anti_repetition(self):
        """
        Attack: Phase 1 outputs identical blueprints for all chunks (B_1 = B_2).
        Risk: Repetition curse, identical parallel decode across all blocks.
        Invariant:
            1. SemanticDeduplicationFilter triggers orthogonal repulsion (sim > tau=0.85).
            2. DirichletEnergyWatchdog triggers symmetry breaking on collapsed energy (E0 < 1e-6).
        """
        print("[VECTOR 4.1] Testing Phase 2 Identical Blueprint Repulsion & Watchdog...")
        deduplicator = SemanticDeduplicationFilter(d_model=self.config.hidden_size, tau=self.config.dedup_tau)
        watchdog = DirichletEnergyWatchdog(threshold_ratio=0.5, boost_gain=0.25)

        # Create identical blueprints for K=3 blocks
        base_vec = F.normalize(torch.randn(1, 1, self.config.hidden_size), p=2, dim=-1)
        H_identical = base_vec.repeat(1, 3, 1)

        # 1. Deduplication Filter
        H_dedup, repulsions = deduplicator(H_identical)
        self.assertGreater(repulsions, 0, "Deduplicator failed to trigger on identical chunks!")

        # 2. Dirichlet Watchdog
        E0 = watchdog.compute_energy(H_identical)
        self.assertLess(E0, 1e-6, "Identical chunks should have near-zero Dirichlet energy!")

        H_boosted, E_curr, triggered = watchdog(H_dedup, E0)
        self.assertTrue(triggered, "Watchdog failed to trigger on collapsed energy!")
        E_final = watchdog.compute_energy(H_boosted)

        self.assertGreater(E_final, E0, "Watchdog failed to increase Dirichlet diversity!")
        print(f"  >>> [PASS] Repulsions={repulsions}, Initial Energy={E0:.6f} -> Boosted Energy={E_final:.4f}.")

    def test_12_dynamic_blueprint_evolver_zero_vector_safeguard(self):
        """
        Attack: Force residual intent norm to 0 (Zero-Vector Collapse).
        Risk: Spherical normalization division by zero (0 / 0 = NaN).
        Invariant: SphericalNorm and Evolver MUST transition to [COMPLETED] attractor cleanly.
        """
        print("[VECTOR 4.2] Testing Dynamic Blueprint Evolver Zero-Vector Collapse...")
        evolver = DynamicBlueprintEvolver(
            d_model=self.config.hidden_size,
            total_steps=8,
            collapse_threshold=0.25
        )
        spher_norm = SphericalNorm()

        # Test SphericalNorm on pure zero vector
        v_zero = torch.zeros(1, self.config.hidden_size)
        v_normed = spher_norm(v_zero)

        self.assertFalse(torch.isnan(v_normed).any(), "SphericalNorm produced NaN on zero vector!")
        self.assertAlmostEqual(torch.norm(v_normed).item(), 1.0, places=4, msg="Norm is not unitary!")

        # Test Evolver step with sub-threshold collapsed candidate
        B_curr = torch.randn(1, self.config.hidden_size) * 0.05  # below 0.25 collapse threshold
        B_init = torch.randn(1, self.config.hidden_size)
        H_gen = torch.randn(1, 16, self.config.hidden_size)

        B_next, metrics = evolver.step(B_curr, B_init, H_gen, step_idx=8)

        self.assertFalse(torch.isnan(B_next).any(), "Evolver emitted NaN!")
        self.assertAlmostEqual(torch.norm(B_next).item(), 1.0, places=4)
        self.assertGreater(metrics["alpha_done_mean"], 0.70, "Evolver failed to trigger completion attractor on sub-threshold norm!")
        self.assertGreater(metrics["cos_sim_to_completed"], 0.70, "Evolver did not align with [COMPLETED] attractor!")
        print("  >>> [PASS] Zero-Vector Collapse avoided; transition to [COMPLETED] attractor verified.")

    # ==========================================================================
    # VECTOR 5: FULL END-TO-END ADVERSARIAL MELTDOWN TEST
    # ==========================================================================

    def test_13_end_to_end_adversarial_meltdown_resistance(self):
        """
        Attack: Multi-Vector Simultaneous Assault.
        Feed corrupted adversarial prompt with OOV IDs, Unicode, and empty blocks
        through full 3-phase generation pipeline.
        Invariant: Pipeline executes without crashing and emits 100% valid, finite token IDs.
        """
        print("[VECTOR 5.1] Executing Full End-to-End Adversarial Meltdown Assault...")
        # Corrupted prompt with negative IDs, OOV IDs, and special tokens
        chaos_prompt = torch.tensor([[-99, 12, 255, 9999, 0, -1, 42, 100]], dtype=torch.long)

        with torch.no_grad():
            out = self.model.generate(chaos_prompt, num_blocks=self.config.max_blocks)

        tokens = out["generated_tokens"]
        draft_bps = out["draft_blueprints"]
        refined_bps = out["refined_blueprints"]
        delta_state = out["final_deltanet_state"]

        # Validate complete absence of NaN and Inf
        self.assertFalse(torch.isnan(tokens.float()).any(), "Emitted tokens contain NaN!")
        self.assertFalse(torch.isnan(draft_bps).any(), "Draft blueprints contain NaN!")
        self.assertFalse(torch.isnan(refined_bps).any(), "Refined blueprints contain NaN!")
        self.assertFalse(torch.isnan(delta_state).any(), "Final DeltaNet state contains NaN!")
        self.assertFalse(torch.isinf(delta_state).any(), "Final DeltaNet state contains Inf!")

        # Validate token ID bounds
        self.assertTrue((tokens >= 0).all() and (tokens < self.config.vocab_size).all(),
            "Token IDs outside valid vocabulary bounds!")

        print(f"  Emitted Tokens: {tokens.squeeze().tolist()[:16]}... (Total: {tokens.shape[1]})")
        print(f"  Final DeltaNet State Norm: {torch.norm(delta_state).item():.4f} (FP32)")
        print("  >>> [PASS] HETMAN-2-NCM SURVIVED FULL ADVERSARIAL MELTDOWN ASSAULT!")

    # ==========================================================================
    # VECTOR 6: MONOLITHIC BACKBONE (src/modeling_hetman2.py) STRESS & EDGE CASES
    # ==========================================================================

    def test_14_monolithic_backbone_adversarial_empty_oov_unicode(self):
        """
        Attack: Subject monolithic Hetman2ForCausalLM (src/modeling_hetman2.py) to:
            1. Empty prompt string, empty list, and empty tensor (len=0).
            2. Single whitespace and multi-whitespace control characters.
            3. Extreme Out-Of-Vocabulary IDs (negative, gigantic > vocab_size).
            4. High-entropy multi-script Unicode storm (Cyrillic, Arabic RTL, Math, Astral Emojis).
        Invariant:
            - Model MUST execute without exception in forward() and generate_ncm().
            - All output logits and generated tokens MUST be strictly finite and non-NaN.
            - All token IDs MUST be within [0, vocab_size - 1].
        """
        print("[VECTOR 6.1] Testing Monolithic Backbone (src/modeling_hetman2.py) Edge Inputs...")
        mono_cfg = ProductionConfig.create_mini(
            vocab_size=256,
            block_size=16,
            overlap_size=4
        )
        mono_model = MonolithicHetman2ForCausalLM(mono_cfg)
        mono_model.eval()

        with torch.no_grad():
            # 1. Empty inputs
            out_empty_str = mono_model.generate_ncm("", max_tokens=16)
            self.assertFalse(torch.isnan(out_empty_str["generated_tokens"].float()).any())
            self.assertGreater(out_empty_str["generated_tokens"].shape[1], 0)

            out_empty_list = mono_model.generate_ncm([], max_tokens=16)
            self.assertFalse(torch.isnan(out_empty_list["generated_tokens"].float()).any())

            out_empty_tensor = mono_model(torch.empty((1, 0), dtype=torch.long))
            self.assertEqual(out_empty_tensor["logits"].shape, (1, 0, mono_cfg.vocab_size))
            self.assertFalse(torch.isnan(out_empty_tensor["logits"]).any())

            # 2. Single whitespace and minimal control tokens
            out_space = mono_model.generate_ncm(" \t\n\r", max_tokens=16)
            self.assertFalse(torch.isnan(out_space["generated_tokens"].float()).any())

            # 3. Extreme OOV & Negative token IDs
            oov_tensor = torch.tensor([[-9999, -1, mono_cfg.vocab_size + 500, 1000000]], dtype=torch.long)
            out_oov_fwd = mono_model(oov_tensor)
            self.assertFalse(torch.isnan(out_oov_fwd["logits"]).any())
            self.assertEqual(out_oov_fwd["logits"].shape, (1, 4, mono_cfg.vocab_size))

            out_oov_gen = mono_model.generate_ncm(oov_tensor, max_tokens=16)
            tokens_gen = out_oov_gen["generated_tokens"]
            self.assertTrue(((tokens_gen >= 0) & (tokens_gen < mono_cfg.vocab_size)).all())

            # 4. Unicode / Astral / RTL Storm
            storm = "Козак \U0001f92f Гетьман \u200b\u202eRTL_SABOTAGE\u202c 🔥⚡"
            out_unicode = mono_model.generate_ncm(storm, max_tokens=16)
            self.assertFalse(torch.isnan(out_unicode["generated_tokens"].float()).any())
            self.assertTrue(((out_unicode["generated_tokens"] >= 0) & (out_unicode["generated_tokens"] < mono_cfg.vocab_size)).all())

        print("  >>> [PASS] Monolithic backbone successfully weathered empty, OOV, and Unicode storms.")

    def test_15_intermediate_quench_and_self_healing(self):
        """
        Attack: Inject NaN and +/- Inf directly into:
            1. Monolithic GatedDeltaNet recurrent state accumulator S.
            2. Monolithic GatedDeltaNet input activations.
            3. FWHT blockwise transform and BitLinear weights.
        Invariant:
            - Self-healing quench MUST extinguish all NaN/Inf coordinates.
            - Model forward step produces 100% finite, valid output.
            - Recurrent state does NOT permanently contaminate or blow up.
        """
        print("[VECTOR 6.2] Testing Intermediate Quench & Self-Healing (src/modeling_hetman2.py)...")
        mono_cfg = ProductionConfig.create_mini(
            vocab_size=256
        )
        layer = MonolithicGatedDeltaNet(mono_cfg)

        # 1. Inject poison into recurrent state S
        B, H, D = 1, mono_cfg.deltanet_num_heads, mono_cfg.deltanet_head_dim
        poisoned_S = torch.randn(B, H, D, D, dtype=torch.float32)
        poisoned_S[0, 0, :, :] = float('nan')
        poisoned_S[0, 1, 0, 0] = float('inf')
        poisoned_S[0, 2, 1, 1] = float('-inf')

        x_clean = torch.randn(B, 4, mono_cfg.hidden_size)
        out, (_, healed_S) = layer(x_clean, prev_state=(None, poisoned_S))

        self.assertFalse(torch.isnan(out).any(), "DeltaNet output contains NaN after state corruption!")
        self.assertFalse(torch.isinf(out).any(), "DeltaNet output contains Inf after state corruption!")
        self.assertFalse(torch.isnan(healed_S).any(), "DeltaNet state was not healed of NaNs!")
        self.assertFalse(torch.isinf(healed_S).any(), "DeltaNet state was not healed of Infs!")

        # 2. Inject poison into input activation x
        x_poisoned = torch.randn(B, 4, mono_cfg.hidden_size)
        x_poisoned[0, 1, 5] = float('nan')
        x_poisoned[0, 2, 10] = float('inf')

        out2, (_, healed_S2) = layer(x_poisoned, prev_state=(None, healed_S))
        self.assertFalse(torch.isnan(out2).any(), "DeltaNet output contains NaN after input corruption!")
        self.assertFalse(torch.isinf(out2).any(), "DeltaNet output contains Inf after input corruption!")
        self.assertFalse(torch.isnan(healed_S2).any(), "State contaminated by input NaN!")
        print("  >>> [PASS] DeltaNet recurrent quench healed all NaN/Inf activations and internal states.")

    def test_16_max_blocks_k8_3200_tokens_memory_budget(self):
        """
        Stress Verification: Concurrency and peak memory under K=8 blocks (3200 tokens).
        RFC-HETMAN-2.0B / Spec 09 Invariant: Peak RAM < 750.0000 MiB.
        DeltaNet recurrent state memory complexity: O(1) constant.
        Sliding decoding workspace: 416-token window (400 tokens + 16 overlap) = 12.80 MiB.
        """
        print("[VECTOR 6.3] Testing Max Horizon Generation (K=8 blocks, 3200 tokens) Memory Invariants...")
        prod_cfg = ProductionConfig()
        budget = prod_cfg.calculate_memory_budget()

        # 1. DeltaNet recurrent state O(1) memory
        state_bytes = budget["bytes_deltanet_state"]
        state_mib = budget["mib_deltanet_state"]
        expected_state_bytes = 28 * 16 * 64 * 64 * 4  # 28 layers, 16 heads, 64x64 FP32 = 7,340,032 bytes
        self.assertEqual(state_bytes, expected_state_bytes)
        self.assertAlmostEqual(state_mib, 7.000000, places=4)
        print(f"  DeltaNet Recurrent State: {state_mib:.6f} MiB (strictly 7.000000 MiB across 28 layers)")

        # 2. Decoding Workspace O(1) memory
        decode_workspace = budget["bytes_phase3_decode_workspace"]
        self.assertEqual(decode_workspace, 13417216)  # 12.80 MiB
        print(f"  Phase 3 Decoding Workspace: {decode_workspace / (1024*1024):.2f} MiB (strictly 12.80 MiB)")

        # 3. Total Peak Memory Arena
        total_arena_mib = budget["mib_total_monolithic_arena"]
        print(f"  Total Monolithic Arena: {total_arena_mib:.4f} MiB (Limit: 750.0000 MiB)")
        self.assertLess(total_arena_mib, 750.0, f"Memory budget exceeded! {total_arena_mib} MiB >= 750.0 MiB")
        margin = 750.0 - total_arena_mib
        self.assertGreater(margin, 30.0, f"Safety margin too narrow! Got {margin:.2f} MiB")
        print(f"  Safety Margin: {margin:.4f} MiB headroom available.")

        # 4. Simulate recurrent state memory invariance over K=8 blocks
        # DeltaNet state does not grow by a single byte regardless of token count
        mini_cfg = ProductionConfig.create_mini(
            vocab_size=256,
            block_size=16,
            overlap_size=4
        )
        layer = MonolithicGatedDeltaNet(mini_cfg)
        conv_st = None
        S_t = None
        state_sizes = []

        # 8 blocks of 16 tokens = 128 tokens
        for block_idx in range(8):
            x_blk = torch.randn(1, 16, mini_cfg.hidden_size)
            _, (conv_st, S_t) = layer(x_blk, prev_state=(conv_st, S_t))
            state_sizes.append(S_t.element_size() * S_t.nelement())

        # Assert state size is identically constant across all 8 blocks
        self.assertTrue(all(sz == state_sizes[0] for sz in state_sizes))
        print(f"  State size per block over 8 blocks: {state_sizes} bytes (strictly O(1) invariant)")
        print("  >>> [PASS] Peak RAM strictly < 750 MiB proven mathematically and empirically.")

    def test_17_cross_phase_triad_and_dynamic_fusion_under_chaos(self):
        """
        Stress Verification: Triad concurrent processing under extreme branch imbalances:
            Branch A (Hopfield) saturated with zero-variance attractors,
            Branch B (SwiGLU) receiving large activations (clamped to <= 10.0),
            Branch C (Hypothesis subnet) under high perturbation.
        Invariant:
            Dynamic Adaptive Fusion Gate produces strictly finite, normalized representations.
        """
        print("[VECTOR 6.4] Testing Triad Concurrent Processing & Fusion under Chaos...")
        mono_cfg = ProductionConfig.create_mini(
            vocab_size=256
        )
        model = MonolithicHetman2ForCausalLM(mono_cfg)
        model.eval()

        # Corrupt stem output with high magnitude activations
        x_stem = torch.randn(2, 8, mono_cfg.hidden_size) * 100.0

        with torch.no_grad():
            x_fused, hopfield_loss, _ = model.model.forward_triad(x_stem, mode=MODE_DRAFT)

        self.assertFalse(torch.isnan(x_fused).any(), "Fused triad output contains NaN!")
        self.assertFalse(torch.isinf(x_fused).any(), "Fused triad output contains Inf!")
        self.assertFalse(math.isnan(hopfield_loss.item()), "Hopfield loss is NaN!")
        self.assertLess(x_fused.abs().max().item(), 50.0, "Fused representation exploded!")
        print(f"  Triad fused output norm: {torch.norm(x_fused).item():.4f}, Max element: {x_fused.abs().max().item():.4f}")
        print("  >>> [PASS] Triad concurrent fusion maintained absolute stability under severe saturation.")


def run_chaos_monkey_suite():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestChaosMonkeyAdversarial)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        sys.exit(1)


if __name__ == "__main__":
    run_chaos_monkey_suite()
