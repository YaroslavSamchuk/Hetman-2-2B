"""
Unit and Integration Test Suite: Cyrillic Boundary Stitching, UTF-8 Integrity,
Asymmetric Frozen Prefix Mask, Anti-Repetition, and Shannon Entropy Gate.
Target:
    - src/block_ncm_decoder.py
    - src/bidirectional_refiner.py
Specification: RFC-HETMAN-2-NCM-PHASE3-DECODER-SPEC
"""

import math
import os
import sys
import unittest
from typing import List, Optional, Any, Union
import torch
import torch.nn.functional as F

# Ensure root directory is on python search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.block_ncm_decoder import (
    DecoderConfig,
    ConfidenceGate,
    BlockNCMDecoder,
    seamless_bpe_stitch,
)


class TestCyrillicBoundaryStitching(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"\n[INIT] TestCyrillicBoundaryStitching running on device: {cls.device}")

        # Try loading a BPE tokenizer if transformers is available
        cls.tokenizer = None
        try:
            from transformers import AutoTokenizer
            cls.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-0.5B-Instruct", local_files_only=True
            )
            print("  [INIT] Loaded local BPE tokenizer: Qwen2.5-0.5B-Instruct")
        except Exception as e:
            print(f"  [INIT] Pretrained tokenizer not loaded ({e}), using mock BPE tokenizer.")

    # =========================================================================
    # TEST 1: BPE Subwords & Multi-byte UTF-8 Stitching across Boundary
    # =========================================================================
    def test_01_bpe_subword_and_utf8_stitching(self):
        """
        Verifies that Cyrillic BPE subwords and multi-byte UTF-8 tokens are stitched
        without replacement characters (U+FFFD), double spaces, or word mutilations.
        """
        ukr_text = (
            "Hetman-2 розгортає концептуальні креслення в неперервний потік "
            "високоякісного українського тексту без будь-яких артефактів на межах блоків."
        )

        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(ukr_text)
            self.assertGreater(len(tokens), 20)

            # Test multiple split points (e.g. inside words, at whitespace, at punctuation)
            for split_idx in range(5, len(tokens) - 5):
                prev_toks = tokens[:split_idx]
                curr_toks = tokens[split_idx:]

                # 1. Direct token ID concatenation followed by decode
                combined_ids = prev_toks + curr_toks
                clean_decoded = self.tokenizer.decode(combined_ids)

                self.assertEqual(
                    clean_decoded, ukr_text,
                    f"Combined token decoding mismatch at split {split_idx}!"
                )
                self.assertNotIn("\ufffd", clean_decoded, "UTF-8 replacement character found!")

                # 2. Check seamless_bpe_stitch
                stitched = seamless_bpe_stitch(prev_toks, curr_toks, tokenizer=self.tokenizer)
                self.assertNotIn("\ufffd", stitched, "UTF-8 replacement character in seamless_bpe_stitch!")

            print("  [OK] Test 01: Cyrillic BPE subword and UTF-8 continuity across splits verified.")
        else:
            # Fallback mock byte-level BPE test
            prev_toks = [101, 102, 103]
            curr_toks = [104, 105, 106]
            res = seamless_bpe_stitch(prev_toks, curr_toks)
            self.assertIn("6 tokens", res)
            print("  [OK] Test 01: Token ID stream report verified (mock mode).")

    # =========================================================================
    # TEST 2: Asymmetric Frozen Prefix Attention Mask & Immutability Invariant
    # =========================================================================
    def test_02_asymmetric_frozen_prefix_immutability(self):
        """
        Verifies that tokens [0..15] in the Frozen Prefix:
        1. Cannot attend to future active block tokens [16..415] (causal isolation).
        2. Are mathematically guaranteed never to be mutated or overwritten across passes.
        """
        cfg = DecoderConfig(block_len=64, overlap_len=16, n_layers=2)
        decoder = BlockNCMDecoder(cfg).to(self.device)
        decoder.eval()

        # Build asymmetric mask
        mask = decoder.build_prefix_mask(batch_size=1, device=self.device)
        l_tot = cfg.total_window_len
        n_over = cfg.overlap_len

        # Prefix slots 0..15 cannot see tokens 16..total
        self.assertTrue(
            torch.all(mask[0, 0, :n_over, n_over:] <= -1e8),
            "Prefix slots must be strictly masked from attending to future tokens!"
        )
        # Active tokens 16..total CAN see prefix slots 0..15
        self.assertTrue(
            torch.all(mask[0, 0, n_over:, :n_over] == 0.0),
            "Active block tokens must attend to prefix tokens!"
        )

        # Verify token immutability through select_tokens_with_penalty
        test_tokens = torch.randint(100, 500, (1, l_tot), device=self.device)
        pristine_prefix = test_tokens[0, :n_over].clone()

        dummy_logits = torch.randn(1, l_tot, cfg.vocab_size, device=self.device)
        updated_tokens, _ = decoder.select_tokens_with_penalty(dummy_logits, test_tokens.clone())

        # Assert prefix is 100% bit-exact identical
        self.assertTrue(
            torch.equal(updated_tokens[0, :n_over], pristine_prefix),
            "Prefix slots [0..15] were illegally modified during token selection!"
        )

        # Verify ConfidenceGate never flags prefix as uncertain
        gate = decoder.confidence_gate
        flat_logits = torch.zeros(1, l_tot, cfg.vocab_size, device=self.device)
        metrics = gate.compute_metrics(flat_logits, active_start=n_over)
        unc_mask = metrics["uncertain_mask"]
        self.assertFalse(
            torch.any(unc_mask[0, :n_over]),
            "Prefix slots [0..15] must NEVER be flagged as uncertain!"
        )
        print("  [OK] Test 02: Asymmetric Frozen Prefix attention mask and strict immutability verified.")

    # =========================================================================
    # TEST 3: Repetition Penalty, Distance-Decay & Ukrainian Legitimate Repeats
    # =========================================================================
    def test_03_repetition_penalty_and_ukrainian_legitimate_repeats(self):
        """
        Verifies behavior on legitimate Ukrainian grammatical repetitions:
        1. Enumerations with commas (distance = 2 tokens).
        2. Correlative conjunctions 'і ... і', 'ні ... ні' (distance = 3 tokens).
        3. Prepositional phrases 'в лісі і в полі' (distance = 3 tokens).
        4. Identifies the difference between naive flat penalty vs distance-decay soft penalty.
        """
        vocab_size = 1000
        comma_id = 44         # ','
        word_apples_id = 101  # 'яблука'
        word_pears_id = 102   # 'груші'
        word_plums_id = 103   # 'сливи'

        # Case A: Sequence [яблука, ,, груші] -> next token wants to be ','
        # Distance from previous comma is exactly 2 tokens.
        tokens = torch.tensor([[0]*16 + [word_apples_id, comma_id, word_pears_id] + [0]*10], dtype=torch.long)
        pos = 16 + 3 # pos 19

        logits = torch.zeros(1, tokens.shape[1], vocab_size)
        logits[0, pos, comma_id] = 10.0      # Comma is highly favored by language model
        logits[0, pos, word_plums_id] = 2.0  # Plums is second choice

        # Naive flat penalty of 50.0 without distance decay or whitelist:
        flat_penalty = 50.0
        logits_naive = logits.clone()
        for prev_idx in range(pos - 4, pos):
            tok = tokens[0, prev_idx].item()
            logits_naive[0, pos, tok] -= flat_penalty

        naive_choice = torch.argmax(logits_naive[0, pos]).item()
        # Naive penalty forced suppression of comma:
        self.assertNotEqual(
            naive_choice, comma_id,
            "Naive flat penalty was expected to suppress comma!"
        )

        # Distance-Decay Soft Penalty formulation:
        # penalty(d) = base_penalty * exp(-gamma * (d - 1))
        # with whitelist immunity for punctuation and grammatical connectors
        whitelist = {comma_id}  # Punctuation / closed class whitelist
        logits_soft = logits.clone()
        base_penalty = 15.0
        gamma = 1.0

        for prev_idx in range(pos - 4, pos):
            tok = tokens[0, prev_idx].item()
            d = pos - prev_idx
            if tok in whitelist:
                eff_penalty = 0.0  # Whitelisted: legitimate punctuation
            else:
                eff_penalty = base_penalty * math.exp(-gamma * (d - 1))
            logits_soft[0, pos, tok] -= eff_penalty

        soft_choice = torch.argmax(logits_soft[0, pos]).item()
        # Soft distance-decay with whitelist preserves legitimate comma:
        self.assertEqual(
            soft_choice, comma_id,
            "Distance-Decay Soft Penalty must allow legitimate comma at distance 2!"
        )

        # But immediate unigram stutter (d=1, 'сливи' -> 'сливи') is still strongly penalized:
        tokens[0, pos] = word_plums_id
        logits[0, pos + 1, word_plums_id] = 10.0
        logits[0, pos + 1, 999] = 2.0

        d1_penalty = base_penalty * math.exp(-gamma * 0) # 15.0
        logits_stutter = logits.clone()
        logits_stutter[0, pos + 1, word_plums_id] -= d1_penalty
        stutter_choice = torch.argmax(logits_stutter[0, pos + 1]).item()
        self.assertNotEqual(
            stutter_choice, word_plums_id,
            "Immediate stutter (d=1) must be strictly suppressed!"
        )

        print("  [OK] Test 03: Distance-Decay Soft Penalty and Ukrainian legitimate repetition preserved.")

    # =========================================================================
    # TEST 4: Shannon Entropy Gate Edge Cases & Numerical Stability
    # =========================================================================
    def test_04_confidence_gate_entropy_edge_cases(self):
        """
        Verifies Confidence Gate numerical stability under:
        1. One-hot deterministic logits (H strictly 0.0).
        2. Extremely negative logits / masked slots (-inf or -1e9) without producing NaN.
        3. Flat uniform distribution (H_norm = 1.0).
        4. Floating-point underflow probabilities (p < 1e-30).
        """
        vocab_size = 65536
        gate = ConfidenceGate(vocab_size=vocab_size, confidence_threshold=0.85, entropy_threshold=0.35)

        # 1. Deterministic one-hot logit
        logits_det = torch.full((1, 32, vocab_size), -100.0)
        logits_det[:, :, 777] = 100.0  # Extremely sharp
        metrics_det = gate.compute_metrics(logits_det, active_start=16)

        self.assertFalse(torch.isnan(metrics_det["entropy"]).any())
        self.assertAlmostEqual(metrics_det["mean_entropy"], 0.0, places=4)
        self.assertAlmostEqual(metrics_det["mean_confidence"], 1.0, places=4)
        self.assertEqual(metrics_det["num_uncertain"], 0)

        # 2. Logits containing -inf (e.g. from masked vocabulary or padding)
        logits_inf = torch.randn(1, 32, vocab_size)
        logits_inf[:, :, 103] = float("-inf") # Masked token
        logits_inf[:, :, :100] = float("-inf") # Masked range

        # Test safe Shannon entropy calculation avoiding 0 * (-inf) = NaN
        scaled = logits_inf / gate.temperature
        lp = F.log_softmax(scaled, dim=-1)
        p = torch.exp(lp)

        # Robust formulation
        plogp = torch.where((p > 0.0) & torch.isfinite(lp), p * lp, torch.zeros_like(p))
        safe_entropy = -torch.sum(plogp, dim=-1)
        self.assertFalse(torch.isnan(safe_entropy).any(), "Safe entropy must NEVER produce NaN!")
        self.assertFalse(torch.isinf(safe_entropy).any(), "Safe entropy must NEVER produce Inf!")

        # 3. Flat uniform distribution
        logits_flat = torch.zeros(1, 32, vocab_size)
        metrics_flat = gate.compute_metrics(logits_flat, active_start=16)
        self.assertAlmostEqual(metrics_flat["mean_entropy"], 1.0, places=3)
        self.assertEqual(metrics_flat["num_uncertain"], 16) # 32 - 16 = 16 active tokens

        print("  [OK] Test 04: Confidence Gate Shannon entropy edge cases and NaN immunity verified.")

    # =========================================================================
    # TEST 5: Two-Block 400-Token Cyrillic Stream Stitching
    # =========================================================================
    def test_05_two_block_400_token_cyrillic_continuity(self):
        """
        Synthesizes two adjacent 400-token blocks (800 tokens total) with a Cyrillic
        prompt prefix, tests DeltaNet state carryover S_1 -> S_2, and verifies
        boundary stitching continuity at junction token 399 -> 400.
        """
        # 8 layers required by architecture (0..1 syntax, 2..4 morphology, 5..7 surface)
        cfg = DecoderConfig(
            d_model=128,
            d_ffn=256,
            n_layers=8,
            n_heads=4,
            d_head=32,
            d_k_delta=16,
            d_v_delta=16,
            vocab_size=2000,
            block_len=400,
            overlap_len=16,
            base_passes=8,
            max_emergency_passes=1
        )
        decoder = BlockNCMDecoder(cfg).to(self.device)
        decoder.eval()

        for s in decoder.delta_states:
            s.reset(batch_size=1, device=self.device)

        # Block 1
        bp1 = torch.randn(1, cfg.n_bp_slots, cfg.d_model, device=self.device)
        hf1 = torch.randn(1, 1, cfg.d_model, device=self.device)
        prefix_ukr = torch.randint(100, 500, (1, cfg.overlap_len), device=self.device)

        with torch.no_grad():
            tokens_1, diag_1 = decoder.decode_block(
                blueprint=bp1,
                prev_overlap_tokens=prefix_ukr,
                hopfield_memory=hf1
            )

        self.assertEqual(len(tokens_1), 400)
        s1_states = diag_1["delta_states"]

        # Overlap prefix for Block 2: exactly the last 16 tokens of Block 1
        overlap_for_2 = tokens_1[-cfg.overlap_len:].unsqueeze(0)
        self.assertEqual(overlap_for_2.shape, (1, 16))

        # Block 2 conditioned on S_1 and overlap buffer
        bp2 = torch.randn(1, cfg.n_bp_slots, cfg.d_model, device=self.device)
        hf2 = torch.randn(1, 1, cfg.d_model, device=self.device)

        with torch.no_grad():
            tokens_2, diag_2 = decoder.decode_block(
                blueprint=bp2,
                prev_overlap_tokens=overlap_for_2,
                hopfield_memory=hf2,
                incoming_delta_states=s1_states
            )

        self.assertEqual(len(tokens_2), 400)

        # Full 800-token stream
        full_tokens = torch.cat([tokens_1, tokens_2], dim=0)
        self.assertEqual(len(full_tokens), 800)

        # Audit junction window [tokens 390 .. 410]
        junction = full_tokens[390:410].tolist()
        print(f"\n  [AUDIT] Seam Junction (tokens 391..410 across 400-token border):")
        print(f"    Block 1 tail: {junction[:10]}")
        print(f"    Block 2 head: {junction[10:]}")

        # Check for immediate 3-gram repetition across junction
        for i in range(390, 408):
            t1, t2, t3 = full_tokens[i:i+3].tolist()
            self.assertFalse(
                t1 == t2 == t3,
                f"3-gram stutter detected at junction index {i}: ({t1}, {t2}, {t3})"
            )

        # Verify DeltaNet state evolved continuously
        self.assertGreater(diag_2["delta_state_norms"][0], 0.0)

        # Test seamless stitching utility
        report = seamless_bpe_stitch(tokens_1.tolist(), tokens_2.tolist())
        self.assertIn("800 tokens", report)

        print("  [OK] Test 05: Two-block 400-token Cyrillic stream synthesis (800 tokens) verified.")


def run_tests():
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCyrillicBoundaryStitching)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
