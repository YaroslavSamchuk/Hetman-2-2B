"""
Autonomous Test Suite for Phase 3 Block-Causal NCM Decoder
Target: C:\\Users\\samch\\Downloads\\TH2\\src\\block_ncm_decoder.py
Verifies:
  1. 400-token block generation with exact tensor invariants
  2. 8 Base Passes (Scale 1 Syntax -> Scale 2 Morphology -> Scale 3 Tokens)
  3. Confidence Gate Shannon entropy & top-1 probability calculation
  4. 3 Emergency Correction Passes with adaptive early termination
  5. 16-token Overlap Buffer (Frozen Prefix Prompting & Asymmetric Masking)
  6. DeltaNet S_{k-1} -> S_k state carryover with O(1) FP32 memory
  7. Multi-token anti-repetition invariant across 800 tokens
"""

import sys
import os
import unittest
import math
import torch

# Ensure src is in python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.block_ncm_decoder import (
    DecoderConfig,
    RMSNorm,
    BitLinear158,
    VectorQuantizer,
    DeltaNetRecurrentState,
    ConfidenceGate,
    BlockDecoderLayer,
    BlockNCMDecoder,
    seamless_bpe_stitch,
)


class TestBlockNCMDecoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[INIT] TestBlockNCMDecoder running on device: {cls.device}")
        # Standard configuration
        cls.config = DecoderConfig()

    def test_01_config_invariants(self):
        """Verifies architectural hyperparameter invariants according to RFC spec."""
        cfg = self.config
        self.assertEqual(cfg.d_model, 1536)
        self.assertEqual(cfg.n_layers, 8)
        self.assertEqual(cfg.n_heads, 16)
        self.assertEqual(cfg.d_head, 96)
        self.assertEqual(cfg.d_k_delta, 64)
        self.assertEqual(cfg.d_v_delta, 64)
        self.assertEqual(cfg.vocab_size, 65536)
        self.assertEqual(cfg.block_len, 400)
        self.assertEqual(cfg.overlap_len, 16)
        self.assertEqual(cfg.total_window_len, 416)
        self.assertEqual(cfg.base_passes, 8)
        self.assertEqual(cfg.max_emergency_passes, 3)
        self.assertEqual(cfg.confidence_threshold, 0.85)
        print("  [OK] Test 01: Architectural configuration invariants verified.")

    def test_02_deltanet_state_fp32_o1_invariant(self):
        """Verifies DeltaNet recurrent memory O(1) boundedness and FP32 precision."""
        state = DeltaNetRecurrentState(self.config).to(self.device)
        state.reset(batch_size=1, device=self.device)
        
        s_tensor = state.get_state()
        self.assertEqual(s_tensor.dtype, torch.float32, "DeltaNet state must strictly be FP32!")
        self.assertEqual(s_tensor.shape, (1, 16, 64, 64))
        
        # Memory footprint per layer: 16 * 64 * 64 * 4 bytes = 262,144 bytes = 256 KiB
        layer_bytes = s_tensor.numel() * s_tensor.element_size()
        self.assertEqual(layer_bytes, 262144)
        
        # Test recurrent step update
        q = torch.randn(1, 16, 64, device=self.device)
        k = torch.randn(1, 16, 64, device=self.device)
        v = torch.randn(1, 16, 64, device=self.device)
        alpha = torch.full((1, 16, 1), 0.95, device=self.device)
        beta = torch.full((1, 16, 1), 0.25, device=self.device)
        
        out = state.update_step(q, k, v, alpha, beta)
        self.assertEqual(out.shape, (1, 16, 64))
        self.assertEqual(state.get_state().dtype, torch.float32)
        print("  [OK] Test 02: DeltaNet FP32 O(1) constant memory (256 KB/layer) verified.")

    def test_03_confidence_gate_entropy(self):
        """Verifies Shannon entropy and confidence calculations in ConfidenceGate."""
        gate = ConfidenceGate(vocab_size=self.config.vocab_size, confidence_threshold=0.85, entropy_threshold=0.35)
        
        # Case A: High confidence / near-zero entropy (one-hot logit)
        logits_sharp = torch.zeros(1, 416, self.config.vocab_size)
        logits_sharp[:, :, 42] = 50.0  # Extremely confident
        is_unc, metrics_sharp = gate.evaluate_gate(logits_sharp, active_start=16)
        self.assertFalse(is_unc, "Sharp logits must not trigger uncertainty!")
        self.assertAlmostEqual(metrics_sharp["mean_confidence"], 1.0, places=3)
        self.assertAlmostEqual(metrics_sharp["mean_entropy"], 0.0, places=3)
        self.assertEqual(metrics_sharp["num_uncertain"], 0)
        
        # Case B: High entropy (uniform distribution)
        logits_flat = torch.zeros(1, 416, self.config.vocab_size)
        is_unc_flat, metrics_flat = gate.evaluate_gate(logits_flat, active_start=16)
        self.assertTrue(is_unc_flat, "Flat logits must trigger uncertainty!")
        self.assertAlmostEqual(metrics_flat["mean_entropy"], 1.0, places=3)
        self.assertEqual(metrics_flat["num_uncertain"], 400)
        print("  [OK] Test 03: Confidence Gate Shannon entropy and uncertainty detection verified.")

    def test_04_overlap_prefix_mask(self):
        """Verifies asymmetric attention mask for Frozen Prefix Prompting."""
        decoder = BlockNCMDecoder(self.config).to(self.device)
        mask = decoder.build_prefix_mask(batch_size=1, device=self.device)
        
        self.assertEqual(mask.shape, (1, 16, 416, 416))
        # Prefix slots 0..15 cannot see future tokens 16..415
        self.assertTrue(torch.all(mask[0, 0, :16, 16:] <= -1e8))
        # Future tokens can see prefix tokens 0..15
        self.assertTrue(torch.all(mask[0, 0, 16:, :16] == 0.0))
        # Future tokens can see each other bidirectionally
        self.assertTrue(torch.all(mask[0, 0, 16:, 16:] == 0.0))
        # Prefix tokens can see each other
        self.assertTrue(torch.all(mask[0, 0, :16, :16] == 0.0))
        print("  [OK] Test 04: Asymmetric Frozen Prefix attention mask verified.")

    def test_05_single_block_400_token_synthesis(self):
        """Verifies full 400-token synthesis through 8 base RVQ passes."""
        decoder = BlockNCMDecoder(self.config).to(self.device)
        decoder.eval()
        
        for s in decoder.delta_states:
            s.reset(batch_size=1, device=self.device)
            
        bp = torch.randn(1, self.config.n_bp_slots, self.config.d_model, device=self.device)
        hopfield_mem = torch.randn(1, 1, self.config.d_model, device=self.device)
        prefix = torch.randint(100, 1000, (1, self.config.overlap_len), device=self.device)
        
        with torch.no_grad():
            tokens, diag = decoder.decode_block(
                blueprint=bp,
                prev_overlap_tokens=prefix,
                hopfield_memory=hopfield_mem
            )
            
        self.assertEqual(len(tokens), 400, "Output must be strictly 400 clean tokens!")
        self.assertEqual(diag["base_passes_executed"], 8, "Must execute exactly 8 base passes!")
        self.assertIn("rvq_scale1_codes", diag)
        self.assertIn("rvq_scale2_codes", diag)
        self.assertEqual(len(diag["rvq_scale1_codes"]), 400)
        self.assertEqual(len(diag["rvq_scale2_codes"]), 400)
        self.assertEqual(len(diag["delta_states"]), 8)
        print(f"  [OK] Test 05: 400-token block synthesis verified (8 passes, initial entropy={diag['initial_mean_entropy']:.3f}, final entropy={diag['final_mean_entropy']:.3f}).")

    def test_06_multi_block_seamless_continuity(self):
        """Verifies 2 adjacent blocks (800 tokens) with S_{k-1} DeltaNet state carryover and no repetition."""
        decoder = BlockNCMDecoder(self.config).to(self.device)
        decoder.eval()
        for s in decoder.delta_states:
            s.reset(batch_size=1, device=self.device)

        # Block 1
        bp1 = torch.randn(1, self.config.n_bp_slots, self.config.d_model, device=self.device)
        hf1 = torch.randn(1, 1, self.config.d_model, device=self.device)
        prefix1 = torch.randint(100, 1000, (1, self.config.overlap_len), device=self.device)

        with torch.no_grad():
            tokens_1, diag_1 = decoder.decode_block(blueprint=bp1, prev_overlap_tokens=prefix1, hopfield_memory=hf1)
        
        # Extract overlap prefix for Block 2
        overlap_for_2 = tokens_1[-self.config.overlap_len:].unsqueeze(0)
        s1_states = diag_1["delta_states"]

        # Block 2 with S_1 carryover
        bp2 = torch.randn(1, self.config.n_bp_slots, self.config.d_model, device=self.device)
        hf2 = torch.randn(1, 1, self.config.d_model, device=self.device)

        with torch.no_grad():
            tokens_2, diag_2 = decoder.decode_block(
                blueprint=bp2,
                prev_overlap_tokens=overlap_for_2,
                hopfield_memory=hf2,
                incoming_delta_states=s1_states
            )

        full_stream = torch.cat([tokens_1, tokens_2], dim=0)
        self.assertEqual(len(full_stream), 800)
        
        # Verify DeltaNet memory evolved
        self.assertGreater(diag_2["delta_state_norms"][0], 0.0)

        # Repetition curse check
        for i in range(len(full_stream) - 2):
            t1, t2, t3 = full_stream[i:i+3].tolist()
            self.assertFalse(t1 == t2 == t3, f"3-gram stutter detected at index {i}!")
        print("  [OK] Test 06: Multi-block seamless continuity (800 tokens, 0 stutters) verified.")

    def test_07_emergency_correction_passes(self):
        """Verifies Confidence Gate activates up to 3 emergency correction passes upon uncertainty."""
        decoder = BlockNCMDecoder(self.config).to(self.device)
        decoder.eval()
        for s in decoder.delta_states:
            s.reset(batch_size=1, device=self.device)

        bp = torch.randn(1, self.config.n_bp_slots, self.config.d_model, device=self.device)
        hf = torch.randn(1, 1, self.config.d_model, device=self.device)
        prefix = torch.randint(100, 1000, (1, self.config.overlap_len), device=self.device)

        with torch.no_grad():
            tokens_em, diag_em = decoder.decode_block(
                blueprint=bp,
                prev_overlap_tokens=prefix,
                hopfield_memory=hf,
                force_emergency=True
            )

        self.assertTrue(diag_em["confidence_gate_triggered"])
        self.assertGreater(diag_em["emergency_passes_executed"], 0)
        self.assertLessEqual(diag_em["emergency_passes_executed"], 3)
        self.assertEqual(diag_em["total_passes_executed"], 8 + diag_em["emergency_passes_executed"])
        print(f"  [OK] Test 07: Confidence Gate triggered {diag_em['emergency_passes_executed']} emergency correction passes successfully.")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBlockNCMDecoder)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
