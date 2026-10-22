"""
================================================================================
HETMAN-2-NCM: ВЕРИФІКАЦІЙНИЙ ТЕСТ-МОДУЛЬ ДЛЯ DYNAMIC BLUEPRINT EVOLVER
Документ: RFC-HETMAN-2-NCM-DYNAMIC-BLUEPRINT-EVOLUTION
Специфікація: spec/15_DYNAMIC_BLUEPRINT_EVOLUTION_MECHANICS.md
================================================================================

Перевіряє:
1. Збереження норми векторів ||B_k^(t)||_2 == 1.0 на кожному з 8 кроків.
2. Захист від Zero-Vector Collapse через автоматичний перехід до атрактора [COMPLETED].
3. Ортогоналізація Грама-Шмідта: теорема 1 про захист нерозкритих тез від Over-Pruning.
4. Математична точність косинусного розкладу gamma_t in [0.10, 0.80].
5. Стабільність градієнтного бекпропа через усі 8 кроків (BPTE).
6. Багатошкальна субпросторова факторизація (Scale 1: Синтаксис -> Scale 2: Морфологія -> Scale 3: Словник).
7. Модулятор AdaLN-Zero: нульова ініціалізація та коректність розмірностей.
"""

import math
import os
import sys
from pathlib import Path

# Add project root to sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    import pytest
except ImportError:
    pytest = None
import torch
import torch.nn.functional as F

from src.dynamic_blueprint_evolver import (
    SphericalNormFunction,
    SphericalNorm,
    GramSchmidtSubspaceFilter,
    SemanticContentExtractor,
    MultiScaleSubspaceFactorization,
    AdaLNZeroModulator,
    DynamicBlueprintEvolver,
)


# ==============================================================================
# 1. Тест розкладу коефіцієнта gamma_t
# ==============================================================================

def test_cosine_schedule_exact_values():
    """
    Перевіряє точність значень косинусного розкладу gamma_t для t=1..8
    відповідно до таблиці 2.3.1 специфікації:
        t=1: 0.1000
        t=8: 0.8000
        Монотонне зростання.
    """
    evolver = DynamicBlueprintEvolver(
        d_model=1536,
        total_steps=8,
        gamma_min=0.10,
        gamma_max=0.80,
        schedule_type="cosine"
    )
    
    # Перевірка крайових точок
    assert abs(evolver.get_gamma(1) - 0.10) < 1e-6, "t=1 має бути точно 0.10"
    assert abs(evolver.get_gamma(8) - 0.80) < 1e-6, "t=8 має бути точно 0.80"
    
    # Перевірка згідно з формулою (Equation 258):
    for t in range(1, 9):
        progress = (t - 1) / 7.0
        expected = 0.10 + 0.5 * (0.80 - 0.10) * (1.0 - math.cos(math.pi * progress))
        actual = evolver.get_gamma(t)
        assert abs(actual - expected) < 1e-6, f"Крок t={t}: очікувалось {expected:.6f}, отримано {actual:.6f}"
        
    # Перевірка строгого монотонного зростання
    gammas = [evolver.get_gamma(t) for t in range(1, 9)]
    for i in range(len(gammas) - 1):
        assert gammas[i] < gammas[i + 1], f"Косинусний розклад не є строго монотонним: {gammas}"


# ==============================================================================
# 2. Тест збереження одиничної норми ||B_k^(t)||_2 == 1.0 на всіх 8 кроках
# ==============================================================================

def test_vector_norm_preservation_8_passes():
    """
    Симулює 8 проходів декодера для батчу з різними вхідними векторами
    та верифікує строге збереження евклідової норми ||B_k^(t)||_2 = 1.000000.
    """
    torch.manual_seed(42)
    batch_size = 4
    d_model = 1536
    l_block = 400
    total_steps = 8
    
    evolver = DynamicBlueprintEvolver(
        d_model=d_model,
        total_steps=total_steps,
        gamma_min=0.10,
        gamma_max=0.80,
        schedule_type="cosine"
    )
    evolver.eval()
    
    # Початковий начерк B_k^(0)
    B_init_raw = torch.randn(batch_size, d_model)
    B_init = F.normalize(B_init_raw, p=2, dim=-1)
    
    # 8 проміжних тензорів згенерованого тексту
    H_passes = []
    for t in range(1, total_steps + 1):
        alignment = float(t) / float(total_steps)
        H_t = (
            alignment * B_init.unsqueeze(1).expand(-1, l_block, -1)
            + (1.0 - alignment * 0.5) * torch.randn(batch_size, l_block, d_model)
        )
        H_passes.append(H_t)
        
    with torch.no_grad():
        trajectory, metrics_history = evolver.simulate_8_passes(B_init, H_passes)
        
    assert len(trajectory) == 9, "Траєкторія повинна містити B_0..B_8 (9 станів)"
    assert len(metrics_history) == 8, "Історія метрик повинна містити 8 кроків"
    
    for idx, B_t in enumerate(trajectory):
        norms = torch.norm(B_t, p=2, dim=-1)
        for b in range(batch_size):
            assert abs(norms[b].item() - 1.0) < 1e-5, (
                f"Крок {idx}, батч {b}: норма {norms[b].item():.7f} відрізняється від 1.0!"
            )


# ==============================================================================
# 3. Тест захисту від Zero-Vector Collapse та сходження до атрактора [COMPLETED]
# ==============================================================================

def test_zero_vector_collapse_protection():
    """
    Стрес-тест на колапс у нуль:
    Штучно конструюється стан, коли дельта віднімання повністю анігілює
    поточний намір (B_curr - gamma * delta_h -> 0).
    Модуль повинен плавно активувати атрактор [COMPLETED] та утримати норму 1.0
    без NaN, Inf або хаотичного шуму.
    """
    torch.manual_seed(1337)
    batch_size = 3
    d_model = 1536
    l_block = 400
    
    evolver = DynamicBlueprintEvolver(
        d_model=d_model,
        total_steps=8,
        collapse_threshold=0.25,
        schedule_type="cosine"
    )
    evolver.eval()
    
    B_curr = F.normalize(torch.randn(batch_size, d_model), p=2, dim=-1)
    B_init = B_curr.clone()
    
    # Створюємо агресивний деструктивний вхідний тензор
    H_destructive = B_curr.unsqueeze(1).expand(-1, l_block, -1) * 5.0
    
    with torch.no_grad():
        B_next, metrics = evolver.step(
            B_curr=B_curr,
            B_init=B_init,
            H_gen=H_destructive,
            step_idx=7
        )
        
    norms = torch.norm(B_next, p=2, dim=-1)
    for b in range(batch_size):
        assert abs(norms[b].item() - 1.0) < 1e-5, f"Батч {b}: Норма порушена при колапсі: {norms[b].item()}"
        assert not torch.isnan(B_next[b]).any(), f"Батч {b}: Виявлено NaN після захисту від колапсу!"
        assert not torch.isinf(B_next[b]).any(), f"Батч {b}: Виявлено Inf після захисту від колапсу!"
        
    # Вентиль завершеності повинен бути активований
    assert metrics["alpha_done_mean"] > 0.5, (
        f"AlphaDone {metrics['alpha_done_mean']} недостатньо високий при зануленні залишку!"
    )
    
    # На фінальному кроці (t=8) захоплення атрактором повинно бути >= 0.85
    with torch.no_grad():
        B_final, metrics_final = evolver.step(
            B_curr=B_curr,
            B_init=B_init,
            H_gen=H_destructive,
            step_idx=8
        )
    assert metrics_final["alpha_done_mean"] >= 0.85, (
        f"Фінальний крок t=8: alpha_done {metrics_final['alpha_done_mean']} має бути >= 0.85"
    )
    assert metrics_final["cos_sim_to_completed"] > 0.80, (
        "На кроці 8 вектор наміру не зійшовся до атрактора [COMPLETED]"
    )


# ==============================================================================
# 4. Тест ортогоналізації Грама-Шмідта (Теорема 1: Захист майбутніх тез)
# ==============================================================================

def test_gram_schmidt_future_theses_protection():
    """
    Математична перевірка Теореми 1 (Subspace Protection):
    Маємо M=4 ортонормованих векторів тез: u_1, u_2, u_3, u_4.
    Нехай u_1 - активна теза, а u_3, u_4 - майбутні нерозкриті тези (c_j < 0.15).
    Подаємо паразитний вектор delta_h, що містить шум у напрямках u_3 та u_4.
    Фільтр Грама-Шмідта повинен придушити паразитичну проекцію на u_3, u_4,
    зберігаючи майбутні тези недоторканими.
    """
    torch.manual_seed(777)
    d_model = 128
    batch_size = 2
    num_slots = 4
    
    # Генеруємо ортонормований базис за допомогою QR
    rand_mat = torch.randn(d_model, num_slots)
    q_mat, _ = torch.linalg.qr(rand_mat) # [d_model, num_slots]
    slots = q_mat.t().unsqueeze(0).expand(batch_size, -1, -1) # [B, 4, d_model]
    
    u1 = slots[:, 0, :]
    u2 = slots[:, 1, :]
    u3 = slots[:, 2, :]
    u4 = slots[:, 3, :]
    
    # Поточний намір реалізує u1 (активний), u3 та u4 мають нульовий намір
    B_curr = u1 # c_1 = 1.0, c_2 = 0.0, c_3 = 0.0, c_4 = 0.0
    
    # Паразитний вектор delta_h, що має компоненти u1 та шкідливі u3, u4
    delta_h_corrupted = 0.8 * u1 + 0.9 * u3 + 0.7 * u4
    
    gs_filter = GramSchmidtSubspaceFilter(
        d_model=d_model,
        num_slots=num_slots,
        active_threshold=0.15,
        orthogonal_leak=0.0 # Strict isolation for mathematical verification
    )
    
    filtered_delta = gs_filter.project_slots(
        delta_h=delta_h_corrupted,
        slots=slots,
        B_current=B_curr,
        active_threshold=0.15
    )
    
    # Перевірка:
    # 1. Проекція на активну тезу u1 збережена
    proj_u1 = torch.sum(filtered_delta * u1, dim=-1)
    assert torch.allclose(proj_u1, torch.tensor([0.8, 0.8]), atol=1e-5), (
        f"Активна теза u1 спотворена: {proj_u1}"
    )
    
    # 2. Проекція на захищені майбутні тези u3, u4 строго нульова!
    proj_u3 = torch.sum(filtered_delta * u3, dim=-1)
    proj_u4 = torch.sum(filtered_delta * u4, dim=-1)
    
    assert torch.allclose(proj_u3, torch.zeros_like(proj_u3), atol=1e-6), (
        f"Майбутня теза u3 не захищена: проекція {proj_u3}"
    )
    assert torch.allclose(proj_u4, torch.zeros_like(proj_u4), atol=1e-6), (
        f"Майбутня теза u4 не захищена: проекція {proj_u4}"
    )


# ==============================================================================
# 5. Тест стабільності градієнтного бекпропа через 8 кроків (BPTE)
# ==============================================================================

def test_backpropagation_through_evolution():
    """
    Перевіряє, що градієнти протікають через усі 8 кроків еволюції
    до початкового начерку B_init_raw без згасання (vanishing) та без вибуху (explosion).
    """
    torch.manual_seed(999)
    device = torch.device("cpu")
    batch_size = 2
    d_model = 256
    l_block = 32
    total_steps = 8
    
    evolver = DynamicBlueprintEvolver(
        d_model=d_model,
        total_steps=total_steps,
        gamma_min=0.10,
        gamma_max=0.80,
        schedule_type="cosine"
    ).to(device)
    
    B_init_raw = torch.randn(batch_size, d_model, device=device, requires_grad=True)
    B_init = F.normalize(B_init_raw, p=2, dim=-1)
    
    H_passes = [
        torch.randn(batch_size, l_block, d_model, device=device)
        for _ in range(total_steps)
    ]
    
    trajectory, _ = evolver.simulate_8_passes(B_init, H_passes)
    
    # Втрати на фінальному кроці: максимізація проекції на атрактор
    target_attractor = F.normalize(evolver.completed_attractor, p=2, dim=-1)
    loss = torch.sum(trajectory[-1] * target_attractor)
    loss.backward()
    
    assert B_init_raw.grad is not None, "Градієнт dL/dB_init_raw відсутній!"
    grad_norm = torch.norm(B_init_raw.grad).item()
    
    assert not torch.isnan(B_init_raw.grad).any(), "NaN у градієнтах B_init_raw!"
    assert not torch.isinf(B_init_raw.grad).any(), "Inf у градієнтах B_init_raw!"
    assert 1e-5 < grad_norm < 1e3, f"Градієнт затух або вибухнув: ||grad|| = {grad_norm}"


# ==============================================================================
# 6. Тест багатошкальної субпросторової факторизації (Scales 1, 2, 3)
# ==============================================================================

def test_multiscale_subspace_factorization():
    """
    Перевіряє коректність розподілу ваг за субпросторами:
    Синтаксис (384), Морфологія (384), Словник (768).
    """
    factorizer = MultiScaleSubspaceFactorization(
        d_model=1536,
        d_syn=384,
        d_morph=384,
        d_lex=768
    )
    
    # Scale 1: t=1
    w_syn, w_morph, w_lex = factorizer.get_scale_weights(1)
    assert w_syn == 0.80 and w_morph == 0.15 and w_lex == 0.05
    
    # Scale 2: t=3
    w_syn, w_morph, w_lex = factorizer.get_scale_weights(3)
    assert w_syn == 0.10 and w_morph == 0.70 and w_lex == 0.20
    
    # Scale 3: t=6
    w_syn, w_morph, w_lex = factorizer.get_scale_weights(6)
    assert w_syn == 0.05 and w_morph == 0.05 and w_lex == 0.90
    
    # Тест застосування маски
    ones = torch.ones(2, 1536)
    filtered = factorizer.apply_filter(ones, step_idx=1)
    
    assert torch.allclose(filtered[:, :384], torch.tensor(0.80))
    assert torch.allclose(filtered[:, 384:768], torch.tensor(0.15))
    assert torch.allclose(filtered[:, 768:], torch.tensor(0.05))


# ==============================================================================
# 7. Тест модулятора AdaLN-Zero (Section 3.3)
# ==============================================================================

def test_adaln_zero_modulator():
    """
    Перевіряє нульову ініціалізацію та розмірності 6 параметрів модуляції.
    """
    d_model = 1536
    num_layers = 8
    batch_size = 2
    
    modulator = AdaLNZeroModulator(d_model=d_model, num_layers=num_layers)
    B_intent = F.normalize(torch.randn(batch_size, d_model), p=2, dim=-1)
    
    for layer_idx in range(num_layers):
        g1, b1, a1, g2, b2, a2 = modulator(B_intent, layer_idx)
        for param in (g1, b1, a1, g2, b2, a2):
            assert param.shape == (batch_size, d_model)
            # При нульовій ініціалізації вихід повинен бути нулем
            assert torch.allclose(param, torch.zeros_like(param)), (
                f"AdaLN-Zero шар {layer_idx} не дорівнює нулю на старті!"
            )


# ==============================================================================
# 8. Запуск повного верифікаційного набору при прямому виконанні
# ==============================================================================

def run_all_tests():
    print("=" * 80)
    print("HETMAN-2-NCM: ЗАПУСК ВЕРИФІКАЦІЙНОГО ТЕСТУВАННЯ DYNAMIC BLUEPRINT EVOLVER")
    print("=" * 80)
    
    print("\n[1/7] Перевірка косинусного розкладу gamma_t...")
    test_cosine_schedule_exact_values()
    print("  -> УСПІХ: Косинусний розклад відповідає специфікації 15!")
    
    print("\n[2/7] Перевірка збереження унітарної норми на 8 проходах...")
    test_vector_norm_preservation_8_passes()
    print("  -> УСПІХ: Норма векторів збережена строго на рівні 1.000000!")
    
    print("\n[3/7] Перевірка захисту від Zero-Vector Collapse та атрактора [COMPLETED]...")
    test_zero_vector_collapse_protection()
    print("  -> УСПІХ: Захист спрацьовує бездоганно, сингулярність усунено!")
    
    print("\n[4/7] Перевірка ортогоналізації Грама-Шмідта (Теорема 1)...")
    test_gram_schmidt_future_theses_protection()
    print("  -> УСПІХ: Майбутні тези надійно захищені від передчасного видалення!")
    
    print("\n[5/7] Перевірка градієнтного потоку BPTE через 8 кроків...")
    test_backpropagation_through_evolution()
    print("  -> УСПІХ: Градієнти стабільні, відсутній вибух або затухання!")
    
    print("\n[6/7] Перевірка субпросторової факторизації за 3 шкалами NCM...")
    test_multiscale_subspace_factorization()
    print("  -> УСПІХ: Спектральні ваги Scale 1, 2, 3 відповідають специфікації!")
    
    print("\n[7/7] Перевірка AdaLN-Zero модулятора...")
    test_adaln_zero_modulator()
    print("  -> УСПІХ: Нульова ініціалізація та розмірності підтверджені!")
    
    print("\n" + "=" * 80)
    print("УСІ ВЕРИФІКАЦІЙНІ ТЕСТИ ПРОЙДЕНО УСПІШНО! МОДУЛЬ ГОТОВИЙ ДО ПРОДАКШЕНУ.")
    print("=" * 80)


if __name__ == "__main__":
    run_all_tests()
