"""
================================================================================
UNIT TESTS: GLOBAL BIDIRECTIONAL NCM REFINER & SEMANTIC DEDUPLICATION
File: tests/test_bidirectional_refiner.py
Specification: RFC-HETMAN-2-NCM-06 (Phase 2)
Authors: Yaroslav Samchuk & Antigravity Systems Architecture Team
License: Apache License 2.0
================================================================================
"""

import math
import os
import sys
import unittest
from pathlib import Path

# Add project root to sys.path for standalone script execution
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.bidirectional_refiner import (
    FastWalshHadamardTransform,
    SemanticDeduplicationFilter,
    BoundarySeamSmoother,
    DirichletEnergyWatchdog,
    BidirectionalLogicArbiter,
    GlobalBidirectionalRefiner,
    compute_repulsion_loss,
    compute_seam_loss,
    compute_dirichlet_loss,
    compute_phase2_auxiliary_loss,
)


class TestSemanticDeduplication(unittest.TestCase):
    """
    Tests verifying the Inhibitory Semantic Deduplication Filter (INV-P2-01).
    Specifically tests the drop in cosine similarity of duplicated blocks.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.B = 1
        self.K = 4
        self.L_c = 16
        self.d = 1536
        self.tau = 0.85

    def test_duplicate_blocks_similarity_drop(self):
        """
        Primary test: Verifies that when two blocks are highly duplicated (sim > 0.85),
        the inhibitory repulsion operator projects the redundant block onto the
        orthogonal subspace, causing a steep drop in similarity below tau=0.85.
        """
        # 1. Construct random chunk representations
        H = torch.randn(self.B, self.K, self.L_c, self.d)

        # 2. Intentionally inject duplicate thought: Chunk 2 is a near-exact replica of Chunk 1
        # Similarity before will be > 0.99
        H[0, 2] = H[0, 1] * 0.98 + torch.randn(self.L_c, self.d) * 0.02

        dedup = SemanticDeduplicationFilter(d_model=self.d, tau=self.tau)
        H_deduped, M_orig, rep_count = dedup(H)

        # 3. Calculate post-repulsion similarity matrix
        c_after, M_after = dedup.compute_similarity_matrix(H_deduped)

        sim_before = M_orig[0, 1, 2].item()
        sim_after = M_after[0, 1, 2].item()

        print(f"\n[TestSemanticDeduplication] Chunk 1 vs Chunk 2:")
        print(f"  Pre-repulsion Cosine Similarity:  {sim_before:.4f}")
        print(f"  Post-repulsion Cosine Similarity: {sim_after:.4f}")
        print(f"  Repulsion events triggered:       {rep_count}")

        # Verification criteria:
        # A. Pre-repulsion must exceed the threshold
        self.assertGreater(sim_before, self.tau, "Setup error: pre-repulsion similarity must be > tau")
        # B. Post-repulsion similarity must drop strictly below tau=0.85
        self.assertLess(sim_after, self.tau, f"Deduplication failed: post-repulsion {sim_after} is not < {self.tau}")
        # C. Substantial drop in similarity
        self.assertLess(sim_after, sim_before - 0.50, "Similarity did not drop substantially")
        # D. Repulsion event was recorded
        self.assertGreaterEqual(rep_count, 1, "At least one repulsion event must be triggered")

    def test_exact_clone_blocks_orthogonalized(self):
        """
        Tests extreme edge case: Chunk 2 is a 100% identical copy of Chunk 1 (sim = 1.0).
        Verifies that no zero-collapse occurs and orthogonal dispersion is applied.
        """
        H = torch.randn(self.B, self.K, self.L_c, self.d)
        H[0, 2] = H[0, 1].clone()  # Exact 1.0 similarity

        dedup = SemanticDeduplicationFilter(d_model=self.d, tau=self.tau)
        H_deduped, M_orig, rep_count = dedup(H)

        c_after, M_after = dedup.compute_similarity_matrix(H_deduped)
        sim_before = M_orig[0, 1, 2].item()
        sim_after = M_after[0, 1, 2].item()

        self.assertAlmostEqual(sim_before, 1.0, places=4)
        self.assertLess(sim_after, self.tau)
        self.assertGreater(torch.norm(H_deduped[0, 2]).item(), 1e-3, "Chunk 2 must not collapse to zero")

    def test_non_duplicate_blocks_unmodified(self):
        """
        Verifies that independent blocks with low cosine similarity (< 0.85)
        are completely untouched by the deduplication filter (zero interventions).
        """
        H = torch.randn(self.B, self.K, self.L_c, self.d)
        dedup = SemanticDeduplicationFilter(d_model=self.d, tau=self.tau)
        H_deduped, M_orig, rep_count = dedup(H)

        self.assertEqual(rep_count, 0, "No repulsion should fire for random independent blocks")
        max_sim = torch.triu(M_orig[0], diagonal=1).max().item()
        self.assertLess(max_sim, self.tau)
        self.assertTrue(torch.allclose(H, H_deduped, atol=1e-6))


class TestBoundarySeamSmoothing(unittest.TestCase):
    """
    Tests verifying Boundary Seam Smoothing (INV-P2-02).
    """

    def setUp(self):
        torch.manual_seed(42)
        self.B = 1
        self.K = 4
        self.L_c = 16
        self.d = 1536

    def test_discontinuity_jump_reduction(self):
        """
        Verifies that smoothing reduces the Euclidean metric jump across chunk boundaries.
        """
        H = torch.randn(self.B, self.K, self.L_c, self.d)
        smoother = BoundarySeamSmoother(d_model=self.d, window_size=4)

        jump_before = torch.norm(H[0, 0, -1] - H[0, 1, 0], p=2).item()
        H_smoothed = smoother(H)
        jump_after = torch.norm(H_smoothed[0, 0, -1] - H_smoothed[0, 1, 0], p=2).item()

        print(f"\n[TestBoundarySeamSmoothing]")
        print(f"  Boundary Jump Before: {jump_before:.4f}")
        print(f"  Boundary Jump After:  {jump_after:.4f}")

        self.assertLess(jump_after, jump_before, "Seam smoothing failed to soften boundary discontinuity")


class TestDirichletEnergyWatchdog(unittest.TestCase):
    """
    Tests verifying the Dirichlet Energy Watchdog against Oversmoothing (INV-P2-03).
    """

    def setUp(self):
        torch.manual_seed(42)
        self.B = 1
        self.K = 4
        self.L_c = 16
        self.d = 1536

    def test_watchdog_dispersion_trigger(self):
        """
        Forces artificial oversmoothing and verifies that the watchdog triggers
        contrastive dispersion to expand energy.
        """
        watchdog = DirichletEnergyWatchdog(energy_threshold_ratio=0.5, boost_gain=0.25)
        H_init = torch.randn(self.B, self.K, self.L_c, self.d)
        e0 = watchdog.compute_energy(H_init).item()

        # Artificially homogenize chunks towards the common barycenter
        barycenter = H_init.mean(dim=1, keepdim=True)
        H_collapsed = H_init * 0.1 + barycenter * 0.9

        e_collapsed = watchdog.compute_energy(H_collapsed).item()
        self.assertLess(e_collapsed, 0.5 * e0, "Setup error: collapsed energy should be < 0.5 * e0")

        H_restored, energy_reported, triggered = watchdog(H_collapsed, initial_energy=e0)
        e_restored = watchdog.compute_energy(H_restored).item()

        print(f"\n[TestDirichletEnergyWatchdog]")
        print(f"  Initial Energy:   {e0:.4f}")
        print(f"  Collapsed Energy: {e_collapsed:.4f}")
        print(f"  Restored Energy:  {e_restored:.4f}")
        print(f"  Watchdog Fired:   {triggered}")

        self.assertTrue(triggered, "Watchdog must trigger when energy drops below threshold")
        self.assertGreater(e_restored, e_collapsed, "Contrastive dispersion must increase Dirichlet energy")


class TestFastWalshHadamardTransform(unittest.TestCase):
    """
    Tests verifying FWHT INT8 normalization and self-inversion.
    """

    def test_fwht_self_inverse(self):
        """
        Verifies that FWHT is strictly self-inverse: FWHT(FWHT(x)) == x.
        """
        fwht = FastWalshHadamardTransform(block_size=512)
        x = torch.randn(2, 4, 512)
        # Check pure butterfly transform
        x_rot = fwht._fwht_butterfly(x)
        x_back = fwht._fwht_butterfly(x_rot)
        max_diff = torch.max(torch.abs(x - x_back)).item()
        self.assertLess(max_diff, 1e-5, f"FWHT is not self-inverse (max diff {max_diff})")

    def test_fwht_quantization_stability(self):
        """
        Verifies that INT8 simulated quantization with gain clamping produces finite values.
        """
        fwht = FastWalshHadamardTransform(block_size=512, eta_min=0.05, eta_max=40.0)
        x = torch.randn(1, 1536)
        x_quant, eta = fwht(x)
        self.assertFalse(torch.isnan(x_quant).any())
        self.assertFalse(torch.isinf(x_quant).any())
        self.assertTrue(0.05 <= eta <= 40.0)


class TestGlobalBidirectionalRefiner(unittest.TestCase):
    """
    Tests verifying the full Phase 2 Global Bidirectional Refiner pipeline.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.B = 1
        self.K = 4
        self.L_c = 16
        self.d = 1536

    def test_full_refiner_3_iterations(self):
        """
        Verifies 3 iterations of bidirectional refinement.
        """
        refiner = GlobalBidirectionalRefiner(d_model=self.d, num_iterations=3)
        H_raw = torch.randn(self.B, self.K, self.L_c, self.d)
        result = refiner(H_raw)

        refined = result["refined_chunks"]
        telemetry = result["telemetry"]

        self.assertEqual(refined.shape, H_raw.shape)
        self.assertEqual(len(telemetry["pass_telemetry"]), 3)
        self.assertFalse(torch.isnan(refined).any())

    def test_full_refiner_4_iterations_with_duplicate(self):
        """
        Verifies 4 iterations of bidirectional refinement with an intentional duplicate block.
        Checks that deduplication fires, similarity drops, and energy is preserved.
        """
        refiner = GlobalBidirectionalRefiner(d_model=self.d, num_iterations=4)
        H_raw = torch.randn(self.B, self.K, self.L_c, self.d)
        # Duplicate block 1 into block 2
        H_raw[0, 2] = H_raw[0, 1] * 0.99 + torch.randn(self.L_c, self.d) * 0.01

        result = refiner(H_raw)
        refined = result["refined_chunks"]
        telemetry = result["telemetry"]

        e0 = telemetry["initial_dirichlet_energy"]
        e_final = telemetry["final_dirichlet_energy"]
        retention = e_final / (e0 + 1e-8)

        print(f"\n[TestGlobalBidirectionalRefiner (4 passes)]")
        print(f"  Initial Dirichlet Energy: {e0:.4f}")
        print(f"  Final Dirichlet Energy:   {e_final:.4f} (Retention: {retention * 100:.2f}%)")

        total_repulsions = sum(p["repulsion_events"] for p in telemetry["pass_telemetry"])
        print(f"  Total Repulsion Events:   {total_repulsions}")

        # Check invariants
        self.assertGreaterEqual(total_repulsions, 1, "Deduplication must fire on duplicate block")
        self.assertGreater(retention, 0.35, "INV-P2-03 violated: Dirichlet energy retention must be >= 0.35")

        # Verify post-refinement similarity of duplicate block
        c_final = refined.mean(dim=2)
        c_final_norm = torch.norm(c_final, p=2, dim=-1, keepdim=True) + 1e-8
        c_final_hat = c_final / c_final_norm
        final_sim = torch.dot(c_final_hat[0, 1], c_final_hat[0, 2]).item()
        print(f"  Final Cosine Similarity (Chunk 1 vs Chunk 2): {final_sim:.4f}")
        self.assertLess(final_sim, 0.85, "Duplicate block similarity must remain below tau=0.85")

    def test_3d_input_tensor_handling(self):
        """
        Verifies that refiner accepts 3D tensors [B, K, d] (macro-blueprint vectors).
        """
        refiner = GlobalBidirectionalRefiner(d_model=self.d, num_iterations=3)
        H_3d = torch.randn(self.B, self.K, self.d)
        result = refiner(H_3d)

        refined = result["refined_chunks"]
        self.assertEqual(refined.shape, (self.B, self.K, self.d))
        self.assertFalse(torch.isnan(refined).any())

    def test_loss_engine_computations(self):
        """
        Verifies that Phase 2 regularization loss functions produce valid gradients.
        """
        H = torch.randn(self.B, self.K, self.L_c, self.d, requires_grad=True)
        losses = compute_phase2_auxiliary_loss(H)

        total_loss = losses["total_aux_loss"]
        self.assertFalse(torch.isnan(total_loss))
        self.assertGreaterEqual(total_loss.item(), 0.0)

        total_loss.backward()
        self.assertIsNotNone(H.grad)
        self.assertFalse(torch.isnan(H.grad).any())

    def test_audit_dedup_antiparallel_and_orthogonal_preservation(self):
        """
        [AUDIT-08 Item 1] Verifies that antiparallel (cos=-1.0) and orthogonal (cos=0.0)
        blocks are strictly preserved without triggering false repulsion.
        """
        tau = 0.85
        dedup = SemanticDeduplicationFilter(d_model=self.d, tau=tau)
        c_1 = torch.randn(1, 1, self.L_c, self.d)
        c_anti = -c_1.clone()  # Strictly antiparallel
        c_ortho = torch.randn(1, 1, self.L_c, self.d)
        c_ortho = c_ortho - (torch.sum(c_ortho * c_1) / torch.sum(c_1 * c_1)) * c_1

        H_test = torch.cat([c_1, c_anti, c_ortho], dim=1)
        H_out, M_orig, rep_count = dedup(H_test)

        self.assertEqual(rep_count, 0, "No repulsion should fire for antiparallel or orthogonal concepts")
        self.assertTrue(torch.allclose(H_test, H_out, atol=1e-5), "Antiparallel & orthogonal blocks must remain untouched")

    def test_audit_contrast_preserving_seam_smoothing(self):
        """
        [AUDIT-08 Item 3] Verifies that contrast-preserving smoother protects section boundaries
        (e.g., transition from Advantages to Disadvantages) from being washed out.
        """
        smoother = BoundarySeamSmoother(d_model=self.d, window_size=4)
        v = torch.randn(1, 1, self.d)
        v = v / torch.norm(v, p=2, dim=-1, keepdim=True) * 39.2

        # Chunk 0: positive theme ('Advantages'), Chunk 1: negative theme ('Disadvantages')
        chunk_adv = v.expand(1, 1, self.L_c, self.d).clone()
        chunk_dis = (-v).expand(1, 1, self.L_c, self.d).clone()
        H_contrast = torch.cat([chunk_adv, chunk_dis], dim=1)

        sim_before = torch.cosine_similarity(H_contrast[0, 0, -1], H_contrast[0, 1, 0], dim=-1).item()
        H_smoothed = smoother(H_contrast)
        sim_after = torch.cosine_similarity(H_smoothed[0, 0, -1], H_smoothed[0, 1, 0], dim=-1).item()

        print(f"\n[TestContrastPreservingSmoothing] Boundary Cosine Before: {sim_before:.4f}, After: {sim_after:.4f}")
        # Cosine similarity must remain negative (contrast preserved, not washed into positive)
        self.assertLess(sim_after, 0.0, "Contrast preservation failed: opposing concepts were washed out to positive correlation")

    def test_audit_watchdog_anti_oscillation_convergence(self):
        """
        [AUDIT-08 Item 2] Verifies that watchdog dampens boost across passes and
        freezes dispersion on pass 4, guaranteeing Banach contraction convergence.
        """
        watchdog = DirichletEnergyWatchdog(energy_threshold_ratio=0.5, boost_gain=0.25)
        H_init = torch.randn(self.B, self.K, self.L_c, self.d)
        e0 = watchdog.compute_energy(H_init).item()

        # Artificially oversmoothed state
        barycenter = H_init.mean(dim=1, keepdim=True)
        H_low = H_init * 0.4 + barycenter * 0.6

        # Pass 3: soft boost allowed
        H_p3, _, trig_p3 = watchdog(H_low, initial_energy=e0, current_pass=3, total_passes=4)
        pert_p3 = torch.norm(H_p3 - H_low).item()

        # Pass 4: dispersion frozen to guarantee fixed-point convergence
        H_p4, _, trig_p4 = watchdog(H_low, initial_energy=e0, current_pass=4, total_passes=4)
        pert_p4 = torch.norm(H_p4 - H_low).item()

        print(f"\n[TestWatchdogAntiOscillation] Pass 3 Perturbation: {pert_p3:.4f}, Pass 4 Perturbation: {pert_p4:.4f}")
        self.assertTrue(trig_p3)
        self.assertFalse(trig_p4, "Watchdog must not fire on final pass (guaranteeing Banach convergence)")
        self.assertAlmostEqual(pert_p4, 0.0, places=5)

    def test_audit_block_sparse_attention_efficiency(self):
        """
        [AUDIT-08 Item 4] Verifies that block-sparse logic arbiter produces finite,
        numerically valid representations while enforcing block sparsity.
        """
        arbiter = BidirectionalLogicArbiter(d_model=self.d, block_window=1)
        K = 8
        S = K * self.L_c
        x = torch.randn(1, S, self.d)

        # Check mask generation
        mask = arbiter.create_block_sparse_mask(K=K, L_c=self.L_c, device=x.device)
        total_elements = S * S
        active_elements = mask.sum().item()
        sparsity = 1.0 - (active_elements / total_elements)
        print(f"\n[TestBlockSparseAttention] K={K}, L_c={self.L_c}: Sparsity = {sparsity * 100:.1f}%")
        self.assertGreater(sparsity, 0.50, "Block sparsity must eliminate at least 50% of connections")

        out_sparse = arbiter(x, bidirectional=True, K=K, L_c=self.L_c, sparse=True)
        self.assertEqual(out_sparse.shape, x.shape)
        self.assertFalse(torch.isnan(out_sparse).any())


if __name__ == "__main__":
    unittest.main()

