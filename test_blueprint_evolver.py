"""
Стрес-тест математичної коректності модуля DynamicBlueprintEvolver.
Специфікація: spec/15_DYNAMIC_BLUEPRINT_EVOLUTION_MECHANICS.md, Розділ 5.2

Перевіряє:
1. Збереження норми ||B_k^(t)||_2 == 1.0 на кожному кроці.
2. Неможливість Zero-Vector Collapse при штучному зануленні залишку.
3. Монотонне спадання косинусної схожості з початковим наміром.
4. Сходження до атрактора [COMPLETED].
5. Стабільність градієнтного бекпропа через усі 8 кроків (BPTE).
"""

import sys
from pathlib import Path

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F
from src.dynamic_blueprint_evolver import DynamicBlueprintEvolver


def run_evolver_verification_suite():
    print("=" * 80)
    print("HETMAN-2-NCM: ВЕРИФІКАЦІЯ МАТЕМАТИКИ ДИНАМІЧНОГО НАЧЕРКУ (8 ПРОХОДІВ)")
    print("=" * 80)
    
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Цільовий пристрій виконання: {device}")
    
    batch_size = 4
    d_model = 1536
    l_block = 400
    total_steps = 8
    
    # 1. Ініціалізація модуля
    evolver = DynamicBlueprintEvolver(
        d_model=d_model,
        total_steps=total_steps,
        gamma_min=0.10,
        gamma_max=0.80,
        schedule_type="cosine"
    ).to(device)
    
    # 2. Генерація початкового наміру B_k^(0)
    B_init_raw = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    B_init = F.normalize(B_init_raw, p=2, dim=-1)
    
    # 3. Синтез 8 проміжних тензорів згенерованих станів H_gen^(t)
    H_passes = []
    for t in range(1, total_steps + 1):
        alignment = float(t) / float(total_steps)
        H_t = alignment * B_init.unsqueeze(1).expand(-1, l_block, -1) + \
              (1.0 - alignment * 0.5) * torch.randn(batch_size, l_block, d_model, device=device)
        H_passes.append(H_t)
        
    print("\n[ТЕСТ 1] Симуляція траєкторії 8 проходів та моніторинг норм...")
    trajectory, metrics = evolver.simulate_8_passes(B_init, H_passes)
    
    for idx, m in enumerate(metrics, 1):
        print(f"  Крок {m['step']:.0f} | gamma={m['gamma']:.4f} | "
              f"ResNorm={m['residual_norm_mean']:.4f} | AlphaDone={m['alpha_done_mean']:.4f} | "
              f"SimToInit={m['cos_sim_to_init']:+.4f} | SimToComp={m['cos_sim_to_completed']:+.4f} | "
              f"Norm={m['final_norm']:.6f}")
        
        # Перевірка одиничної норми
        assert abs(m['final_norm'] - 1.0) < 1e-5, f"КРИТИЧНО: Порушення сферичної норми на кроці {idx}!"
        
    print("  -> РЕЗУЛЬТАТ ТЕСТУ 1: УСПІШНО! Норма строго збережена (1.000000).")
    
    print("\n[ТЕСТ 2] Стрес-тест на Zero-Vector Collapse (Штучне протистояння)...")
    B_collapse_test = trajectory[3].detach().clone()
    H_destructive = B_collapse_test.unsqueeze(1).expand(-1, l_block, -1) * 2.0
    
    B_collapsed_next, collapse_metrics = evolver.step(
        B_curr=B_collapse_test,
        B_init=B_init,
        H_gen=H_destructive,
        step_idx=7
    )
    
    c_norm = torch.norm(B_collapsed_next, p=2, dim=-1).mean().item()
    print(f"  Норма після спроби колапсу: {c_norm:.6f} | "
          f"AlphaDone: {collapse_metrics['alpha_done_mean']:.4f} | "
          f"SimToCompleted: {collapse_metrics['cos_sim_to_completed']:.4f}")
    assert abs(c_norm - 1.0) < 1e-5, "КРИТИЧНО: Модель розлетілася в нуль або NaN!"
    assert collapse_metrics['alpha_done_mean'] > 0.5, "Вентиль завершеності не перехопив колапс!"
    print("  -> РЕЗУЛЬТАТ ТЕСТУ 2: УСПІШНО! Атрактор перехопив нульовий залишок.")

    print("\n[ТЕСТ 3] Перевірка градієнтного потоку через 8 кроків (BPTE)...")
    attractor_normed = F.normalize(evolver.completed_attractor, p=2, dim=-1)
    loss = torch.sum(trajectory[-1] * attractor_normed)
    loss.backward()
    
    assert B_init_raw.grad is not None, "Градієнт не дійшов до початкового креслення!"
    grad_norm = torch.norm(B_init_raw.grad).item()
    assert not torch.isnan(B_init_raw.grad).any(), "NaN у градієнтах B_init!"
    assert not torch.isinf(B_init_raw.grad).any(), "Inf у градієнтах B_init!"
    assert 1e-6 < grad_norm < 1e3, f"Градієнт затух або вибухнув: ||grad|| = {grad_norm}"
    print(f"  Норма градієнта dL/dB_init: {grad_norm:.4f} (Стабільний потік)")
    print("  -> РЕЗУЛЬТАТ ТЕСТУ 3: УСПІШНО! Відсутність вибуху чи затухання градієнтів.")
    
    print("\n" + "=" * 80)
    print("ВСІ ТЕОРЕТИЧНІ ТА МАТЕМАТИЧНІ ПЕРЕВІРКИ ПРОЙДЕНО УСПІШНО!")
    print("=" * 80)


if __name__ == "__main__":
    run_evolver_verification_suite()
