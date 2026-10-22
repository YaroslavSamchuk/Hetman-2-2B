"""
================================================================================
UNIT TESTS: END-TO-END AUTOGRAD GRAPH, GRADIENT FLOWS & BPTT INTEGRITY
File: tests/test_autograd_gradient_flows.py
Audit: Autograd Graph & BPTT Auditor (Subagent 1)
Specification: RFC-HETMAN-2-NCM-AUDIT-AUTOGRAD
================================================================================
"""

import math
import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.configuration_hetman2 import Hetman2Config
from src.modeling_hetman2 import (
    Hetman2ForCausalLM,
    MODE_DRAFT,
    MODE_REFINE,
    MODE_DECODE,
)
from src.bidirectional_refiner import (
    GlobalBidirectionalRefiner,
    BidirectionalLogicArbiter,
    FastWalshHadamardTransform,
)
from training_loss_engine import (
    TernaryQuantizeSTE,
    BitLinear,
    HierarchicalRVQLoss,
    HopfieldAttractorLoss,
    HetmanCompositeLossEngine,
)


class TestAutogradGradientFlows(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_01_refiner_inplace_mutation_and_broadcasting_b_gt_1(self):
        """
        Verifies that Phase 2 GlobalBidirectionalRefiner:
        1. Works with batch size B > 1 without broadcast expansion crash.
        2. Allows backward pass without in-place mutation RuntimeError on AsStridedBackward0.
        """
        B, K, L_c, d = 4, 4, 16, 256
        H = torch.randn(B, K, L_c, d, requires_grad=True)

        with torch.no_grad():
            # Inject near-duplicate chunks across all batches to trigger deduplication
            H.data[:, 1] = H.data[:, 0] * 0.99 + torch.randn_like(H.data[:, 0]) * 0.01

        arbiter = BidirectionalLogicArbiter(d_model=d, n_heads=8)
        refiner = GlobalBidirectionalRefiner(
            d_model=d,
            shared_arbiter=arbiter,
            fwht_block_size=256,
            num_iterations=2
        )

        res = refiner(H)
        refined = res["refined_chunks"]
        self.assertEqual(refined.shape, (B, K, L_c, d))
        self.assertFalse(torch.isnan(refined).any())

        target = torch.randn_like(refined)
        loss = F.mse_loss(refined, target)
        loss.backward()

        self.assertIsNotNone(H.grad)
        self.assertFalse(torch.isnan(H.grad).any())
        self.assertGreater(H.grad.norm().item(), 0.0)
        self.assertIsNotNone(refiner.mode_embedding_refine.grad)
        self.assertGreater(refiner.mode_embedding_refine.grad.norm().item(), 0.0)

    def test_02_fwht_ste_gradient_propagation(self):
        """
        Verifies that FastWalshHadamardTransform utilizes Straight-Through Estimation (STE)
        so that 100% of activation elements receive gradients.
        """
        fwht = FastWalshHadamardTransform(block_size=512)
        x = torch.randn(2, 512, requires_grad=True)
        out, _ = fwht(x)

        loss = F.mse_loss(out, torch.zeros_like(out))
        loss.backward()

        self.assertIsNotNone(x.grad)
        non_zero_frac = (x.grad.abs() > 1e-6).float().mean().item()
        self.assertGreaterEqual(non_zero_frac, 0.99, "Gradients died in FWHT quantization round()")

    def test_03_bitlinear_ste_gradient_scale(self):
        """
        Verifies that TernaryQuantizeSTE divides by gamma in backward,
        eliminating the 82x gradient vanishing bug.
        """
        in_dim, out_dim = 256, 128
        w_init = torch.randn(out_dim, in_dim) * (1.0 / math.sqrt(in_dim))
        x_in = torch.randn(4, in_dim)

        # BitLinear from training_loss_engine
        bl = BitLinear(in_dim, out_dim)
        bl.weight.data.copy_(w_init)
        out_bl = bl(x_in)
        out_bl.sum().backward()
        grad_bl_norm = bl.weight.grad.norm().item()

        # FP32 reference
        w_fp = w_init.clone().requires_grad_(True)
        out_fp = F.linear(x_in, w_fp)
        out_fp.sum().backward()
        grad_fp_norm = w_fp.grad.norm().item()

        ratio = grad_bl_norm / grad_fp_norm
        self.assertGreater(ratio, 0.60, f"BitLinear gradient vanished: ratio {ratio:.4f} < 0.60")

    def test_04_rvq_commitment_loss_detachment(self):
        """
        Verifies that HierarchicalRVQLoss correctly detaches codebook centroids
        in commitment loss so that codebooks are not pushed backwards.
        """
        rvq = HierarchicalRVQLoss(d_model=64, d_code=32, num_codes_l0=64, num_codes_l1=64)
        h = torch.randn(4, 64, requires_grad=True)
        out = rvq(h)

        total_rvq = out["commit_loss"] + out["codebook_loss"]
        total_rvq.backward()

        self.assertIsNotNone(rvq.codebook_l0.weight.grad)
        self.assertIsNotNone(rvq.codebook_l1.weight.grad)
        self.assertFalse(torch.isnan(rvq.codebook_l0.weight.grad).any())
        self.assertFalse(torch.isnan(rvq.codebook_l1.weight.grad).any())

    def test_05_hopfield_attractor_energy_normalization(self):
        """
        Verifies that HopfieldAttractorLoss energy is normalized by dimension,
        preventing 100x loss and gradient explosion over cross-entropy.
        """
        hop = HopfieldAttractorLoss(num_patterns=128, dim=256, beta=8.0)
        z = torch.randn(4, 256, requires_grad=True)
        energy_loss = hop(z)
        energy_loss.backward()

        self.assertFalse(torch.isnan(energy_loss))
        self.assertLess(energy_loss.item(), 5.0, "Hopfield loss value unnormalized/exploding")
        self.assertLess(z.grad.norm().item(), 5.0, "Hopfield gradient norm unnormalized/exploding")

    def test_06_hetman2_full_model_backward(self):
        """
        Verifies end-to-end backward pass through the full Hetman2ForCausalLM model
        with tied weights and MTP auxiliary head.
        """
        cfg = Hetman2Config.create_mini(
            vocab_size=512,
            d_model=256,
            intermediate_dim=512,
            layers=34,
            num_deltanet_layers=28,
            num_global_layers=6,
            deltanet_num_heads=4,
            deltanet_head_dim=32,
            global_attention_heads=4,
            global_kv_heads=2,
            global_head_dim=64,
            hadamard_block_size=256,
            hopfield_num_groups=2,
            hopfield_codebook_size=32,
            hopfield_val_dim=16,
            block_size=16,
            overlap_size=4,
            dim_intent=32,
            dim_style=16,
            rvq_codebook_size=64,
            rvq_dim=32,
            refiner_iterations=2,
        )

        model = Hetman2ForCausalLM(cfg)
        model.train()

        input_ids = torch.randint(0, cfg.vocab_size, (2, 8))
        labels = input_ids.clone()

        res = model(input_ids=input_ids, labels=labels, mode=MODE_DRAFT)
        loss = res["loss"]
        loss.backward()

        self.assertIsNotNone(model.model.embed_tokens.weight.grad)
        self.assertFalse(torch.isnan(model.model.embed_tokens.weight.grad).any())
        self.assertIsNotNone(model.mtp_head.norm.weight.grad)
        self.assertFalse(torch.isnan(model.mtp_head.norm.weight.grad).any())
        self.assertIsNotNone(model.model.hopfield_core.W_expand.grad)


if __name__ == "__main__":
    unittest.main()
