"""
================================================================================
HETMAN-2-NCM 2.0B: RVQ & CODEBOOK STABILITY AUDIT SUITE (1000 ITERATIONS)
================================================================================
Test Suite: tests/test_audit_rvq_1000_iterations.py
Auditor: Subagent 6 (RVQ & Codebook Stability Auditor)
Target: src/concept_codebook.py

Verifications:
1. FP16/BF16 underflow mitigation in Laplace smoothed entropy (no NaN).
2. K-Means++ duplicate clustering protection under identical inputs.
3. Monotonic Quantization Error Contraction (Theorem 2.1) on low-norm/noisy inputs.
4. Support for M=4 levels (L0, L1, L2, L3) without Enum ValueError crash.
5. 1000-iteration multi-scale RVQ ladder stability test.
================================================================================
"""

import math
import sys
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.concept_codebook import (
    SemanticScale,
    VectorQuantizerEMA,
    MultiScaleResidualVectorQuantizer,
    ConceptCodebook,
)


class TestRVQStabilityAudit(unittest.TestCase):
    """
    Аудиторський тестовий набір для перевірки усунення прихованих вразливостей RVQ.
    """

    def setUp(self):
        torch.manual_seed(42)

    def test_01_fp16_entropy_underflow_mitigation(self):
        """
        Тест 1: Перевірка захисту від піднормального underflow у FP16.
        Підтверджує, що обчислення ентропії в FP32 з clamp(min=1e-12)
        повністю запобігає появі NaN при неактивних/мертвих центроїдах.
        """
        dim = 128
        codebook_size = 4096
        vq = VectorQuantizerEMA(
            dim=dim,
            codebook_size=codebook_size,
            decay=0.99,
            laplace_eps=1e-5,
        ).half()

        # Штучно відсуваємо половину центроїдів, роблячи їх мертвими
        with torch.no_grad():
            vq.embedding.data[2048:] += 50.0

        x_fp16 = torch.randn(2, 16, dim, dtype=torch.float16)
        z, idx, out = vq(x_fp16)

        self.assertFalse(torch.isnan(out.loss_entropy).item(), "loss_entropy не повинно бути NaN у FP16!")
        self.assertFalse(torch.isnan(out.total_loss).item(), "total_loss не повинно бути NaN у FP16!")
        self.assertFalse(torch.isinf(out.loss_entropy).item(), "loss_entropy не повинно бути Inf у FP16!")

    def test_02_kmeans_pp_near_identical_batch_clustering(self):
        """
        Тест 2: Перевірка K-Means++ при подачі однакових залишків у батчі.
        Адаптивний шум batch_std гарантує, що центроїди не колапсують у нульову відстань.
        """
        dim = 128
        codebook_size = 256
        vq = VectorQuantizerEMA(
            dim=dim,
            codebook_size=codebook_size,
            decay=0.99,
            reinit_check_interval=1,
            dead_threshold=1.0,
        )
        vq.train()

        with torch.no_grad():
            vq.ema_cluster_size[50:150] = 0.0

        identical_batch = torch.ones(8, dim) * 2.5 + torch.randn(8, dim) * 1e-5
        vq.reinitialize_dead_codes(identical_batch)

        revived_cb = vq.embedding.data[50:150]
        pdist = torch.cdist(revived_cb, revived_cb)
        pdist.fill_diagonal_(999.0)
        min_dist = pdist.min().item()

        # Перевірка: жодні два центроїди не є абсолютно однаковими (min_dist > 0)
        self.assertGreater(min_dist, 0.0, "Центроїди не повинні бути абсолютно ідентичними!")

    def test_03_theorem_2_1_monotonic_contraction_guarantee(self):
        """
        Тест 3: Верифікація монотонного незростання похибки квантування (Теорема 2.1).
        Завдяки проекційному калібруванню, навіть на чистому шумі або малих нормах
        норма залишку гарантовано не зростає: ||r_M|| <= ||r_0||.
        """
        dim = 128
        codebook = MultiScaleResidualVectorQuantizer(
            dim=dim,
            num_levels=3,
            codebook_size=64,
            spherical=False,
        )
        codebook.eval()

        # Вхід з малим шумом
        small_input = torch.randn(4, dim) * 0.01
        out = codebook(small_input)

        norms = out.residual_norms
        # Перевіряємо монотонне незростання похибки
        for m in range(len(norms) - 1):
            self.assertLessEqual(
                norms[m + 1],
                norms[m] + 1e-5,
                f"Похибка квантування не повинна зростати: r_{m}={norms[m]:.4f}, r_{m+1}={norms[m+1]:.4f}"
            )

    def test_04_num_levels_4_support(self):
        """
        Тест 4: Перевірка коректної ініціалізації та прямого проходу для 4 рівнів RVQ (L0-L3).
        """
        dim = 128
        rvq4 = MultiScaleResidualVectorQuantizer(
            dim=dim,
            num_levels=4,
            codebook_size=64,
        )
        self.assertEqual(len(rvq4.quantizers), 4)
        self.assertEqual(rvq4.scales[0], SemanticScale.L0_GLOBAL_DISCOURSE)
        self.assertEqual(rvq4.scales[1], SemanticScale.L1_SYNTAX)
        self.assertEqual(rvq4.scales[2], SemanticScale.L2_SURFACE_TOKENS)
        self.assertEqual(rvq4.scales[3], SemanticScale.L3_SUB_SCALE)

        x = torch.randn(2, 4, dim)
        out = rvq4(x)
        self.assertEqual(out.indices.shape, (2, 4, 4))
        self.assertIn("L3_SUB_SCALE", out.scale_indices)

    def test_05_1000_iterations_stability_audit(self):
        """
        Тест 5: 1000-ітераційний стрес-тест RVQ на відсутність NaN/Inf та деградації норм.
        """
        dim = 64
        codebook_size = 128
        vq = VectorQuantizerEMA(
            dim=dim,
            codebook_size=codebook_size,
            decay=0.99,
            reinit_check_interval=50,
            dead_threshold=1.0,
            laplace_eps=1e-5,
            spherical=False,
        )
        vq.train()

        for step in range(1, 1001):
            x = torch.randn(8, dim)
            z, idx, out = vq(x)

        # Перевірка відсутності вибуху норм або NaN
        norms = vq.embedding.data.norm(dim=-1)
        self.assertFalse(torch.isnan(norms).any(), "Центроїди містять NaN після 1000 ітерацій!")
        self.assertFalse(torch.isinf(norms).any(), "Центроїди містять Inf після 1000 ітерацій!")
        self.assertLess(norms.max().item(), 50.0, "Норми центроїдів вибухнули після 1000 ітерацій!")


if __name__ == "__main__":
    unittest.main(verbosity=2)
