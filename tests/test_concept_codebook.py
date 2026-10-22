"""
================================================================================
HETMAN-2-NCM 2.0B: MULTI-SCALE RVQ CONCEPT CODEBOOK UNIT TESTS
================================================================================
Test Suite: tests/test_concept_codebook.py
Specification Reference: spec/01_MATHEMATICAL_FOUNDATIONS_AND_TENSOR_ALGEBRA.md
                         spec/07_PHASE3_BLOCK_CAUSAL_NCM_DECODER.md

Verifications:
1. Tensor Shape Contracts across arbitrary batch/sequence dimensions.
2. Straight-Through Estimator (STE) end-to-end gradient backpropagation.
3. EMA Centroid Dynamics (monotonic update in train, strictly frozen in eval).
4. Laplace Smoothing (eps=1e-5) numerical stability against division-by-zero & NaN.
5. K-Means++ Dead Codebook Reinitialization and revival of orphan clusters.
6. Multi-Scale Hierarchy: L0 (Discourse) -> L1 (Syntax) -> L2 (Surface Tokens).
7. Theorem 2.1: Exponential Decay of Quantization Error (||r_m|| <= R * rho^m, rho < 1).
================================================================================
"""

import math
import sys
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

# Ensure repository root is on sys.path for direct execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.concept_codebook import (
    SemanticScale,
    CodebookConfig,
    RVQLevelOutput,
    MultiScaleRVQOutput,
    VectorQuantizerEMA,
    MultiScaleResidualVectorQuantizer,
    ConceptCodebook,
)


class TestVectorQuantizerEMA(unittest.TestCase):
    """
    Тестування базового однорівневого квантувальника з EMA та K-Means++.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.codebook_size = 32
        self.quantizer = VectorQuantizerEMA(
            dim=self.dim,
            codebook_size=self.codebook_size,
            decay=0.9,
            commitment_beta=0.25,
            entropy_lambda=0.01,
            temperature=1.0,
            laplace_eps=1e-5,
            dead_threshold=1.0,
            reinit_dead_codes=True,
            spherical=False,
            scale=SemanticScale.L0_GLOBAL_DISCOURSE,
        )

    def test_forward_shapes(self):
        """Перевірка розмірностей вихідних тензорів."""
        # 2D тензор: [B, D]
        x_2d = torch.randn(8, self.dim)
        z_2d, idx_2d, out_2d = self.quantizer(x_2d)
        self.assertEqual(z_2d.shape, (8, self.dim))
        self.assertEqual(idx_2d.shape, (8,))
        self.assertEqual(out_2d.quantized.shape, (8, self.dim))

        # 3D тензор: [B, K, D]
        x_3d = torch.randn(4, 16, self.dim)
        z_3d, idx_3d, out_3d = self.quantizer(x_3d)
        self.assertEqual(z_3d.shape, (4, 16, self.dim))
        self.assertEqual(idx_3d.shape, (4, 16))
        self.assertEqual(out_3d.indices.shape, (4, 16))

    def test_ste_gradient_propagation(self):
        """Перевірка наскрізного проходження градієнта через Straight-Through Estimator (STE)."""
        x = torch.randn(8, self.dim, requires_grad=True)
        z_ste, indices, out = self.quantizer(x)

        # Симуляція зовнішнього цільового функціоналу
        dummy_target = torch.randn_like(z_ste)
        task_loss = F.mse_loss(z_ste, dummy_target)
        total_loss = task_loss + out.total_loss

        total_loss.backward()

        self.assertIsNotNone(x.grad, "Градієнт вхідного тензора x є None!")
        self.assertFalse(torch.isnan(x.grad).any(), "Градієнт містить NaN!")
        self.assertFalse(torch.isinf(x.grad).any(), "Градієнт містить Inf!")
        self.assertGreater(x.grad.abs().sum().item(), 0.0, "Градієнт занулився через дискретизацію!")

    def test_ema_centroid_updating_and_eval_freeze(self):
        """Перевірка: EMA оновлює центроїди при train, і заморожує при eval."""
        initial_centroids = self.quantizer.embedding.data.clone()

        # 1. Режим train(): центроїди мають оновитися
        self.quantizer.train()
        x_train = torch.randn(64, self.dim)
        self.quantizer(x_train)
        updated_centroids = self.quantizer.embedding.data.clone()

        diff = (updated_centroids - initial_centroids).abs().sum().item()
        self.assertGreater(diff, 1e-4, "Центроїди не оновилися в режимі train()!")

        # 2. Режим eval(): центроїди повинні бути строго незмінними
        self.quantizer.eval()
        snapshot_centroids = self.quantizer.embedding.data.clone()
        x_eval = torch.randn(64, self.dim)
        self.quantizer(x_eval)
        after_eval_centroids = self.quantizer.embedding.data.clone()

        eval_diff = (after_eval_centroids - snapshot_centroids).abs().sum().item()
        self.assertEqual(eval_diff, 0.0, "Центроїди змінилися в режимі eval()!")

    def test_laplace_smoothing_numerical_stability(self):
        """Перевірка стійкості Лапласового згладжування (eps=1e-5) при вироджених вибірках."""
        self.quantizer.train()

        # Постійний однаковий вектор (всі точки потрапляють в один кластер, 31 кластер порожній)
        degenerate_x = torch.ones(16, self.dim) * 5.0
        z_ste, indices, out = self.quantizer(degenerate_x)

        self.assertFalse(torch.isnan(out.total_loss), "total_loss містить NaN при виродженому батчі!")
        self.assertFalse(torch.isnan(out.loss_entropy), "loss_entropy містить NaN при ln(0)!")
        self.assertFalse(torch.isinf(out.loss_entropy), "loss_entropy містить Inf!")
        self.assertFalse(torch.isnan(self.quantizer.embedding.data).any(), "Embedding містить NaN після згладжування!")

    def test_kmeans_pp_dead_code_reinitialization(self):
        """Перевірка відродження мертвих центроїдів через K-Means++ реініціалізацію."""
        self.quantizer.train()

        # Штучно створюємо мертві центроїди, зсуваючи їх у нескінченність
        with torch.no_grad():
            self.quantizer.embedding.data[16:] += 1000.0
            self.quantizer.ema_cluster_size[16:] = 0.0

        stats_before = self.quantizer.get_codebook_usage()
        self.assertGreaterEqual(stats_before["dead_codes"], 16)

        # Викликаємо ручний перезапуск K-Means++
        candidate_pool = torch.randn(128, self.dim)
        revived_count = self.quantizer.reinitialize_dead_codes(candidate_pool)

        self.assertGreaterEqual(revived_count, 16, "K-Means++ не реанімував очікувану кількість кодів!")

        # Перевірка: тепер усі центроїди знаходяться в межах норми кандидатів
        max_norm = self.quantizer.embedding.data.norm(dim=-1).max().item()
        self.assertLess(max_norm, 100.0, "Мертві центроїди не були перенесені до розподілу даних!")


class TestMultiScaleRVQ(unittest.TestCase):
    """
    Тестування багаторівневого квантувальника MultiScaleResidualVectorQuantizer
    ієрархії L0 (семантичний дискурс) -> L1 (синтаксис) -> L2 (поверхневі токени).
    """

    def setUp(self):
        torch.manual_seed(42)
        self.dim = 64
        self.rvq = MultiScaleResidualVectorQuantizer(
            dim=self.dim,
            num_levels=3,
            codebook_size=32,
            decay=0.9,
            commitment_beta=0.25,
            entropy_lambda=0.01,
            laplace_eps=1e-5,
            spherical=False,
        )

    def test_multiscale_levels_contract(self):
        """Перевірка відповідності рівнів L0, L1, L2 та розбору індексів."""
        self.assertEqual(len(self.rvq.quantizers), 3)
        self.assertEqual(self.rvq.scales[0], SemanticScale.L0_GLOBAL_DISCOURSE)
        self.assertEqual(self.rvq.scales[1], SemanticScale.L1_SYNTAX)
        self.assertEqual(self.rvq.scales[2], SemanticScale.L2_SURFACE_TOKENS)

        x = torch.randn(4, 8, self.dim)
        out = self.rvq(x)

        self.assertEqual(out.quantized.shape, (4, 8, self.dim))
        self.assertEqual(out.indices.shape, (4, 8, 3))
        self.assertIn("L0_GLOBAL_DISCOURSE", out.scale_indices)
        self.assertIn("L1_SYNTAX", out.scale_indices)
        self.assertIn("L2_SURFACE_TOKENS", out.scale_indices)
        self.assertEqual(out.scale_indices["L0_GLOBAL_DISCOURSE"].shape, (4, 8))

    def test_partial_and_full_decoding(self):
        """Перевірка деквантування з дискретних кодів (повного та часткового)."""
        self.rvq.eval()
        x = torch.randn(4, self.dim)
        out = self.rvq(x)
        indices = out.indices  # [4, 3]

        # Повне деквантування
        x_recon_full = self.rvq.decode_indices(indices)
        diff_full = (x_recon_full - out.quantized).abs().max().item()
        self.assertLess(diff_full, 1e-5, "Повне деквантування не збігається з прямим проходом!")

        # Часткове деквантування L0
        x_recon_l0 = self.rvq.decode_partial(indices, up_to_level=0)
        self.assertEqual(x_recon_l0.shape, (4, self.dim))

        # Часткове деквантування L0 + L1
        x_recon_l1 = self.rvq.decode_partial(indices, up_to_level=1)
        self.assertEqual(x_recon_l1.shape, (4, self.dim))

    def test_exponential_quantization_error_decay(self):
        """
        Тестування Теореми 2.1 (Експоненційне спадання похибки RVQ):
        ||r_0|| > ||r_1|| > ||r_2|| > ||r_3||
        та коефіцієнт стискання rho_m < 1.0 на кожному кроці.
        """
        # Попереднє тренування центроїдів на потоці векторів для вирівнювання залишків
        self.rvq.train()
        for _ in range(60):
            train_batch = torch.randn(64, self.dim)
            self.rvq(train_batch)

        # Тестування на незалежній вибірці
        self.rvq.eval()
        test_x = torch.randn(128, self.dim)
        eval_result = self.rvq.verify_exponential_decay(test_x)

        norms = eval_result["residual_norms"]
        contraction_rates = eval_result["contraction_rates"]

        # 1. Повинно бути рівно 4 норми: r_0 (вхід), r_1 (після L0), r_2 (після L1), r_3 (після L2)
        self.assertEqual(len(norms), 4)
        self.assertEqual(len(contraction_rates), 3)

        # 2. Суворе монотонне спадання норм залишків: E_0 > E_1 > E_2 > E_3
        for m in range(len(norms) - 1):
            self.assertLess(
                norms[m + 1],
                norms[m],
                f"Похибка не зменшилась на рівні {m} -> {m+1}: "
                f"norm[{m}]={norms[m]:.4f}, norm[{m+1}]={norms[m+1]:.4f}"
            )
            # Коефіцієнт стискання rho < 1.0
            self.assertLess(
                contraction_rates[m],
                1.0,
                f"Коефіцієнт стискання rho_{m} >= 1.0 ({contraction_rates[m]:.4f})!"
            )

        # 3. Середній геометричний коефіцієнт стискання rho_mean < 1.0
        self.assertLess(eval_result["empirical_mean_rho"], 1.0)
        self.assertLess(eval_result["mean_log_slope"], 0.0)
        self.assertTrue(eval_result["verified"])


class TestConceptCodebookProductionIntegration(unittest.TestCase):
    """
    Тестування виробничої інтеграції ConceptCodebook за контрактами Hetman-2-NCM:
    Семантичне ядро c_sem [B=2, K=16, d=768] на рівнях L0 -> L1 -> L2.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.codebook = ConceptCodebook(
            dim=768,
            codebook_size=128,  # Компактний розмір для швидкого тестування
            decay=0.95,
            commitment_beta=0.25,
            entropy_lambda=0.01,
            laplace_eps=1e-5,
            spherical=False,
        )

    def test_production_hetman_contract(self):
        """Перевірка відповідності формам тензорів RFC-HETMAN-2-NCM-MATH-SPEC-01."""
        batch_size = 2
        k_blocks = 16
        d_sem = 768

        c_sem_raw = torch.randn(batch_size, k_blocks, d_sem)

        out = self.codebook(c_sem_raw)

        # Перевірка форми квантованого семантичного ядра
        self.assertEqual(out.quantized.shape, (batch_size, k_blocks, d_sem))
        # Перевірка форми індексів: [B, K, 3] (L0, L1, L2)
        self.assertEqual(out.indices.shape, (batch_size, k_blocks, 3))
        # Перевірка наявності лосів
        self.assertTrue(torch.is_tensor(out.total_loss))
        self.assertFalse(torch.isnan(out.total_loss))

        summary = self.codebook.get_summary()
        self.assertEqual(summary["dim"], 768)
        self.assertEqual(summary["num_levels"], 3)
        self.assertEqual(len(summary["levels"]), 3)


if __name__ == "__main__":
    print("=================================================================")
    print("Running Hetman-2-NCM Concept Codebook Autonomous Verification Suite")
    print("=================================================================")
    unittest.main(verbosity=2)
