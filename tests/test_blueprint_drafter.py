"""
Unit & Integration Tests for Hetman-2-NCM Phase 1: Causal AR Blueprint Drafter
==============================================================================
Tests tensor invariants, hyper-sphere S^255 normalization, vMF sampling,
Gumbel-Softmax length bin classification, [END_PLAN] stop-token logic,
Fallback Re-Drafting on semantic drift, and CPU latency (<30 ms).
"""

import unittest
import torch
import torch.nn.functional as F

from src.blueprint_drafter import (
    BlueprintConfig,
    BlockBlueprint,
    BlueprintComponents,
    GatedDeltaNetCell,
    BlueprintIngestion,
    BlueprintPredictionHeads,
    CausalBlueprintDrafter,
    run_blueprint_drafter_benchmarks,
)


class TestBlueprintDrafter(unittest.TestCase):
    """Набір модульних тестів архітектурних інваріантів Фази 1."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.config = BlueprintConfig()
        self.drafter = CausalBlueprintDrafter(self.config).to(self.device)
        self.drafter.eval()

    def test_config_dimensions_and_invariants(self):
        """Перевірка розбивки 1536-вимірного простору моделі."""
        cfg = self.config
        self.assertEqual(cfg.d_model, 1536)
        self.assertEqual(cfg.dim_c_sem, 768)
        self.assertEqual(cfg.dim_idx_hop, 384)
        self.assertEqual(cfg.dim_tau_style, 256)
        self.assertEqual(cfg.dim_ell_budget, 128)
        self.assertEqual(
            cfg.dim_c_sem + cfg.dim_idx_hop + cfg.dim_tau_style + cfg.dim_ell_budget,
            1536
        )

    def test_block_blueprint_structure_and_packing(self):
        """Перевірка пакування та розпакування BlockBlueprint у просторі R^1536."""
        batch_size = 3
        dummy_query = torch.randn(batch_size, self.config.d_model, device=self.device)
        
        with torch.no_grad():
            plan = self.drafter.draft_plan(dummy_query, temperature=0.7)
            
        self.assertGreaterEqual(len(plan), 1)
        self.assertLessEqual(len(plan), self.config.max_blueprint_steps)
        
        for bp in plan:
            full_t = bp.tensor
            # Форма єдиного вектора B_k
            self.assertEqual(full_t.shape, (batch_size, 1536))
            # Форми підкомпонентів
            self.assertEqual(bp.c_sem.shape, (batch_size, 768))
            self.assertEqual(bp.idx_hop.shape, (batch_size, 384))
            self.assertEqual(bp.tau_style.shape, (batch_size, 256))
            self.assertEqual(bp.ell_budget.shape, (batch_size, 128))
            
            # Перевірка розпакування unpack_blueprint
            unpacked = self.drafter.unpack_blueprint(full_t)
            self.assertTrue(torch.equal(unpacked.c_sem, bp.c_sem))
            self.assertTrue(torch.equal(unpacked.idx_hop, bp.idx_hop))
            self.assertTrue(torch.equal(unpacked.tau_style, bp.tau_style))
            self.assertTrue(torch.equal(unpacked.ell_budget, bp.ell_budget))

    def test_vmf_spherical_sampling_on_s255(self):
        """
        Перевірка математичного інваріанта фон Мізеса-Фішера:
        Вектор tau_style строго належить гіперсфері S^255 (||tau||_2 == 1.0).
        """
        batch_size = 16
        h = torch.randn(batch_size, self.config.d_drafter)
        
        # Тест стохастичного режиму
        c_sem, idx_hop, tau, ell, coords, budget, bin_idx, stop_p = self.drafter.heads(
            h, temperature=0.8, deterministic=False
        )
        norms = torch.norm(tau, p=2, dim=-1)
        self.assertTrue(
            torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
            f"Порушення норми на S^255 у стохастичному режимі: {norms}"
        )
        
        # Тест детерміністичного режиму
        _, _, tau_det, _, _, _, _, _ = self.drafter.heads(h, deterministic=True)
        norms_det = torch.norm(tau_det, p=2, dim=-1)
        self.assertTrue(
            torch.allclose(norms_det, torch.ones_like(norms_det), atol=1e-5),
            f"Порушення норми на S^255 у детерміністичному режимі: {norms_det}"
        )

    def test_gumbel_softmax_length_bins(self):
        """Перевірка відбору бінів довжини (200, 300, 400, 500) через Gumbel-Softmax."""
        batch_size = 8
        h = torch.randn(batch_size, self.config.d_drafter)
        
        _, _, _, ell_budget, _, budget, bin_idx, _ = self.drafter.heads(h, temperature=1.0)
        
        self.assertEqual(ell_budget.shape, (batch_size, 128))
        for b_val, b_idx in zip(budget, bin_idx):
            self.assertIn(b_val.item(), self.config.length_bins)
            self.assertEqual(self.config.length_bins[b_idx.item()], b_val.item())

    def test_deltanet_o1_memory_invariance(self):
        """Перевірка константної пам'яті DeltaNet O(1) під час генерації."""
        cell = GatedDeltaNetCell(self.config)
        x = torch.randn(2, self.config.d_drafter)
        state = None
        
        for _ in range(10):
            out, state = cell(x, state)
            self.assertEqual(out.shape, (2, self.config.d_drafter))
            # Розмірність пам'яті залишається строго [B, H, dk, dk]
            self.assertEqual(
                state.shape, 
                (2, self.config.num_heads, self.config.d_state, self.config.d_state)
            )

    def test_fallback_redrafting_on_semantic_drift(self):
        """Перевірка спрацьовування та коректного відновлення при Fallback Re-Drafting."""
        # Штучно виставляємо поріг повторення 0.0, щоб кожен блок викликав відкат
        drift_cfg = BlueprintConfig(repetition_margin=0.0)
        drift_drafter = CausalBlueprintDrafter(drift_cfg)
        
        q = torch.randn(1, 1536)
        plan = drift_drafter.draft_plan(q, temperature=0.7, enable_fallback=True)
        
        # Перевіряємо, що хоча б на одному блоці було зафіксовано відкат (redraft_count > 0)
        total_redrafts = sum(bp.redraft_count for bp in plan)
        self.assertGreater(
            total_redrafts, 0, 
            "Fallback Re-Drafting не зареєстрував жодного відкату при екстремальному дрейфі!"
        )

    def test_cpu_generation_speed_target(self):
        """Перевірка цільової швидкості планування: < 30 мс на CPU."""
        results = run_blueprint_drafter_benchmarks(num_runs=30)
        self.assertTrue(results["target_met"], "Затримка генерації на CPU перевищила 30 мс!")
        self.assertLess(results["mean_latency_ms"], 30.0)


if __name__ == "__main__":
    unittest.main()
