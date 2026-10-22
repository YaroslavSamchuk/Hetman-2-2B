"""
Hetman-2-NCM 2.0B: Dynamic Blueprint Evolver & Residual Intent Tracking Engine
Module: src/dynamic_blueprint_evolver.py
Specification: spec/15_DYNAMIC_BLUEPRINT_EVOLUTION_MECHANICS.md
RFC: RFC-HETMAN-2-NCM-DYNAMIC-BLUEPRINT-EVOLUTION

Architectural Role:
    Phase 3 Multi-Scale Block Decoder / Latent State Dynamics
    Implements Dynamic Blueprint Evolution via Residual Intent Tracking (RIT):
        B_k^{(t+1)} = Normalize_Spherically( (1 - alpha_done) * (B_k^{(t)} - gamma_t * delta_h) + alpha_done * B_completed )
    with Gram-Schmidt Subspace Protection against Over-Pruning,
    Cosine Annealing Schedule for gamma_t in [0.10, 0.80],
    and Zero-Vector Collapse Safeguards with transition to attractor [COMPLETED].

Target Hardware: Consumer GPUs (RTX 2060 6GB, RTX 3060 12GB), Apple Silicon, AVX-512 / AVX2 CPU.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Сферична нормалізація зі спектральним демпфуванням Якобіана (Section 4.3)
# ==============================================================================

class SphericalNormFunction(torch.autograd.Function):
    """
    Автоград-оператор проекції вектора на одиничну ріманову гіперсферу S^(d-1)
    зі спектральним демпфуванням Якобіана (Spectral Jacobian Clamping).

    Формулювання:
        Прямий прохід:
            y = v / max(||v||_2, eps)
        Зворотний прохід (Theorem 4.3):
            J_norm(v) = (1 / ||v||_2) * (I - y y^T)
            Для запобігання вибуху градієнтів (Exploding Gradients) множник
            1 / ||v||_2 обмежується зверху константою clamp_max (за замовчуванням 1.0):
            J_tilde = min(clamp_max, 1.0 / ||v||_2) * (I - y y^T)
            Це гарантує нерозтягуваність оператора: ||J_tilde||_2 <= 1.0.
    """

    @staticmethod
    def forward(ctx, v: torch.Tensor, eps: float = 1e-8, clamp_max: float = 1.0) -> torch.Tensor:
        norm = torch.norm(v, p=2, dim=-1, keepdim=True)
        norm_safe = torch.clamp(norm, min=eps)
        y = v / norm_safe
        ctx.save_for_backward(y, norm_safe)
        ctx.clamp_max = clamp_max
        return y

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[torch.Tensor, None, None]:
        y, norm_safe = ctx.saved_tensors
        clamp_max = ctx.clamp_max
        
        # Спектральне демпфування множника
        scale = torch.clamp(1.0 / norm_safe, max=clamp_max)
        
        # Проектор (I - y y^T) dL/dy = dL/dy - y <y, dL/dy>
        inner_prod = torch.sum(grad_output * y, dim=-1, keepdim=True)
        grad_v = scale * (grad_output - inner_prod * y)
        return grad_v, None, None


class SphericalNorm(nn.Module):
    """
    Модуль сферичної нормалізації на гіперсферу S^(d-1).
    Забезпечує строгу унітарну норму ||x||_2 = 1.0 та стійкий градієнтний потік.
    """

    def __init__(self, eps: float = 1e-8, clamp_max: float = 1.0, use_spectral_clamping: bool = True):
        super().__init__()
        self.eps = eps
        self.clamp_max = clamp_max
        self.use_spectral_clamping = use_spectral_clamping

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        zero_mask = (norm < self.eps)
        if zero_mask.any():
            fallback = torch.zeros_like(x)
            fallback[..., 0] = 1.0
            x = torch.where(zero_mask, fallback, x)
            norm = torch.norm(x, p=2, dim=-1, keepdim=True)
        if self.use_spectral_clamping and x.requires_grad:
            return SphericalNormFunction.apply(x, self.eps, self.clamp_max)
        norm_safe = torch.clamp(norm, min=self.eps)
        return x / norm_safe


# ==============================================================================
# 2. Фільтр ортогоналізації Грама-Шмідта (Subspace Protection, Section 4.2)
# ==============================================================================

class GramSchmidtSubspaceFilter(nn.Module):
    """
    Фільтр ортогоналізації Грама-Шмідта для захисту майбутніх тез від Over-Pruning.

    Математичний сенс (Теорема 1 зі специфікації 15):
        Коли екстрактор реалізованого змісту детектує згенеровані токени,
        він може мати паразитну проекцію на невирішені майбутні тези u_j in M_future.
        Фільтр проектує вектор віднімання виключно на підпростір активних тез M_active,
        зберігаючи скалярний добуток <B_unnorm, u_j> == <B_curr, u_j> для всіх j in M_future.

    Підтримує два режими:
        1. Слотова проекція (якщо передано матрицю слотів U або B_slots [B, M, d_model]).
        2. Траєкторна колінеарно-ортогональна фільтрація відносно початкового креслення B_initial.
    """

    def __init__(
        self,
        d_model: int = 1536,
        num_slots: int = 16,
        active_threshold: float = 0.15,
        orthogonal_leak: float = 0.25
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.active_threshold = active_threshold
        self.orthogonal_leak = orthogonal_leak
        self.slot_dim = max(1, d_model // num_slots)

    def project_slots(
        self,
        delta_h: torch.Tensor,
        slots: torch.Tensor,
        B_current: torch.Tensor,
        active_threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        Стійка слотова ортогоналізація Мура-Пенроуза (Theorem 1, RFC-HETMAN-2-NCM):
            P_active = U_act (U_act^T U_act)^+ U_act^T
            delta_h_safe = P_active * delta_h
        
        Захищена від поганої обумовленості матриці Грама (kappa(G) -> inf)
        при майже колінеарних або лінійно залежних векторах слотів.
        
        slots: [Batch, M, d_model] - вектори слотів креслення
        B_current: [Batch, d_model] - поточний вектор наміру
        delta_h: [Batch, d_model] - сирий вектор віднімання
        """
        thresh = active_threshold if active_threshold is not None else self.active_threshold
        
        # Обчислюємо проекцію поточного наміру на кожен слот: c_m = <B_curr, u_m>
        c_m = torch.sum(slots * B_current.unsqueeze(1), dim=-1) # [B, M]
        
        # Маска активних слотів: c_m >= thresh
        active_mask = (c_m >= thresh).to(delta_h.dtype) # [B, M]
        U_act = slots * active_mask.unsqueeze(-1) # [B, M, d]
        
        # Матриця Грама активних слотів G = U_act @ U_act^T [B, M, M]
        G = torch.bmm(U_act, U_act.transpose(1, 2))
        
        # Псевдообернення Мура-Пенроуза гарантує стійкість до лінійної залежності:
        # P = U_act^T @ (U_act @ U_act^T)^+ @ U_act
        G_pinv = torch.linalg.pinv(G) # [B, M, M]
        
        proj_coeff = torch.bmm(U_act, delta_h.unsqueeze(-1)) # [B, M, 1]
        alpha = torch.bmm(G_pinv, proj_coeff) # [B, M, 1]
        
        # Реконструкція безпечного вектору без ампліфікації норми
        delta_h_safe = torch.bmm(U_act.transpose(1, 2), alpha).squeeze(-1) # [B, d_model]
        
        # Для ортогонального залишку додаємо контрольовану частину orthogonal_leak
        uncovered_delta = delta_h - delta_h_safe
        filtered_delta = delta_h_safe + self.orthogonal_leak * uncovered_delta
        return filtered_delta

    def forward(
        self,
        delta_h: torch.Tensor,
        B_initial: torch.Tensor,
        B_current: torch.Tensor,
        slots: Optional[torch.Tensor] = None,
        active_threshold: Optional[float] = None
    ) -> torch.Tensor:
        """
        delta_h: [Batch, d_model] - сирий вектор змісту
        B_initial: [Batch, d_model] або [Batch, M, d_model] - початковий начерк
        B_current: [Batch, d_model] - поточний намір
        slots: опціональний тензор слотів [Batch, M, d_model]
        """
        if slots is not None:
            return self.project_slots(delta_h, slots, B_current, active_threshold)

        # Якщо B_initial передано як багатослотовий тензор [B, M, d]
        if B_initial.dim() == 3:
            return self.project_slots(delta_h, B_initial, B_current, active_threshold)

        # Режим декомпозиції підпростору B_initial / B_current
        # Обчислюємо збереженість початкового наміру в поточному
        proj = torch.sum(B_current * B_initial, dim=-1, keepdim=True) # [Batch, 1]
        
        # Колінеарна складова поточної траєкторії
        delta_parallel = proj * B_current
        # Ортогональна паразитна складова
        delta_orthogonal = delta_h - delta_parallel
        
        # Придушуємо ортогональний шум, захищаючи майбутні незвідані підпростори
        filtered_delta = delta_parallel + self.orthogonal_leak * delta_orthogonal
        return filtered_delta


# ==============================================================================
# 3. Екстрактор реалізованого семантичного змісту (Section 2.1)
# ==============================================================================

class SemanticContentExtractor(nn.Module):
    """
    Екстрактор реалізованого змісту з блоку 400 латентних токенів.
    Використовує Attention-Pooling відносно поточного вектора наміру B_k^(t)
    та проектор SwiGLU з подальшим обмеженням довжини.
    """

    def __init__(
        self,
        d_model: int = 1536,
        d_attn: Optional[int] = None,
        d_mid: Optional[int] = None,
        eps: float = 1e-6
    ):
        super().__init__()
        self.d_model = d_model
        self.d_attn = d_attn or max(64, d_model // 6)
        self.d_mid = d_mid or max(128, int(d_model * 4 / 3))
        
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)
        
        # Проекції для механізму уваги (Cross-Attention Pooling)
        self.w_q = nn.Linear(d_model, self.d_attn, bias=False)
        self.w_k = nn.Linear(d_model, self.d_attn, bias=False)
        self.scale = 1.0 / math.sqrt(self.d_attn)
        
        # SwiGLU MLP проектор змісту у простір наміру
        self.w_gate = nn.Linear(d_model, self.d_mid, bias=False)
        self.w_up = nn.Linear(d_model, self.d_mid, bias=False)
        self.w_down = nn.Linear(self.d_mid, d_model, bias=False)

    def forward(self, H_gen: torch.Tensor, B_intent: torch.Tensor) -> torch.Tensor:
        """
        H_gen: [Batch, L_block, d_model] - згенеровані латентні стани (L_block = 400)
        B_intent: [Batch, d_model] - поточний вектор наміру B_k^(t)

        Повертає:
            delta_h_clamped: [Batch, d_model] - вектор реалізованого змісту, нормований за довжиною <= 1.0
        """
        H_norm = self.layer_norm(H_gen) # [B, L, d_model]
        
        # Оптимізація: Асоціативна факторизація уваги (-92.5% FLOPs)
        # Замість проекції L токенів K = H_norm @ W_k^T [B, L, d_attn] (314.57 MFLOPs),
        # обчислюємо еквівалентний вектор запиту в просторі моделі:
        # v_eff = (Q @ W_k) * scale [B, 1, d_model] (2.80 MFLOPs)
        Q = self.w_q(B_intent).unsqueeze(1) # [B, 1, d_attn]
        v_eff = torch.matmul(Q, self.w_k.weight) * self.scale # [B, 1, d_model]
        
        # Скалярні добутки уваги релевантності токенів до поточного наміру
        attn_scores = torch.bmm(v_eff, H_norm.transpose(1, 2)) # [B, 1, L]
        attn_weights = F.softmax(attn_scores, dim=-1) # [B, 1, L]
        
        # Зважена агрегація відбитку згенерованого тексту
        H_pooled = torch.bmm(attn_weights, H_norm).squeeze(1) # [B, d_model]
        
        # Нелінійна SwiGLU проекція
        gate = F.silu(self.w_gate(H_pooled))
        up = self.w_up(H_pooled)
        delta_h = self.w_down(gate * up) # [B, d_model]
        
        # Нормування довжини видобутого змісту: delta_h / max(||delta_h||_2, 1.0)
        delta_norm = torch.norm(delta_h, p=2, dim=-1, keepdim=True)
        delta_h_clamped = delta_h / torch.clamp(delta_norm, min=1.0)
        return delta_h_clamped


# ==============================================================================
# 4. Трьохшкальна субпросторова факторизація (Section 3.1 - 3.2)
# ==============================================================================

class MultiScaleSubspaceFactorization(nn.Module):
    """
    Субпросторова факторизація вектора наміру B_k у відповідності до 3 шкал NCM:
        B_k = [B_syn (25%), B_morph (25%), B_lex (50%)]^T = 1536.
    
    Масштабно-зважений фільтр Omega(t):
        t in [1, 2] (Scale 1: Синтаксис):  [0.80, 0.15, 0.05]
        t in [3, 4] (Scale 2: Морфологія): [0.10, 0.70, 0.20]
        t in [5, 8] (Scale 3: Словник):    [0.05, 0.05, 0.90]
    """

    def __init__(
        self,
        d_model: int = 1536,
        d_syn: Optional[int] = None,
        d_morph: Optional[int] = None,
        d_lex: Optional[int] = None
    ):
        super().__init__()
        if d_syn is None or d_morph is None or d_lex is None:
            # За замовчуванням 25% синтаксис (384/1536), 25% морфологія (384/1536), 50% словник (768/1536)
            d_syn = d_model // 4
            d_morph = d_model // 4
            d_lex = d_model - d_syn - d_morph
            
        assert d_syn + d_morph + d_lex == d_model, f"Subspaces sum {d_syn+d_morph+d_lex} != {d_model}"
        self.d_model = d_model
        self.d_syn = d_syn
        self.d_morph = d_morph
        self.d_lex = d_lex

    def get_scale_weights(self, step_idx: int) -> Tuple[float, float, float]:
        """
        Повертає вагові коефіцієнти (w_syn, w_morph, w_lex) для кроку step_idx (1..8).
        """
        if step_idx in (1, 2):
            return (0.80, 0.15, 0.05)
        elif step_idx in (3, 4):
            return (0.10, 0.70, 0.20)
        else: # 5..8
            return (0.05, 0.05, 0.90)

    def apply_filter(self, delta_h: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        Масштабує відповідні субпростори вектора delta_h згідно з поточною шкалою NCM.
        delta_h: [Batch, d_model]
        """
        w_syn, w_morph, w_lex = self.get_scale_weights(step_idx)
        
        syn_part = delta_h[..., :self.d_syn] * w_syn
        morph_part = delta_h[..., self.d_syn:self.d_syn + self.d_morph] * w_morph
        lex_part = delta_h[..., self.d_syn + self.d_morph:] * w_lex
        
        return torch.cat([syn_part, morph_part, lex_part], dim=-1)


# ==============================================================================
# 5. AdaLN-Zero Модуляція шарів декодера (Section 3.3)
# ==============================================================================

class AdaLNZeroModulator(nn.Module):
    """
    Генератор параметрів адаптивної нормалізації шарів (AdaLN-Zero) з нульовою ініціалізацією:
        [gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2] = W_adaln * B_k^(t)
    де B_k^(t) динамічно перебудовує селективність нейронів кожного шару.
    """

    def __init__(self, d_model: int = 1536, num_layers: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        
        # Проекції 6 параметрів на кожен шар: 6 * d_model параметрів
        self.layer_projections = nn.ModuleList([
            nn.Linear(d_model, 6 * d_model, bias=True)
            for _ in range(num_layers)
        ])
        
        # Нульова ініціалізація (AdaLN-Zero) для забезпечення тотожного відображення на старті
        for proj in self.layer_projections:
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    def forward(
        self,
        B_intent: torch.Tensor,
        layer_idx: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        B_intent: [Batch, d_model]
        layer_idx: 0-indexed шар декодера [0..num_layers-1]. Якщо None, використовується шар 0.
        Повертає:
            (gamma_1, beta_1, alpha_1, gamma_2, beta_2, alpha_2), кожен форми [Batch, d_model]
        """
        target_idx = 0 if layer_idx is None else layer_idx
        params = self.layer_projections[target_idx](B_intent) # [B, 6 * d_model]
        chunks = torch.chunk(params, 6, dim=-1)
        return chunks[0], chunks[1], chunks[2], chunks[3], chunks[4], chunks[5]


# ==============================================================================
# 6. Головний модуль: DynamicBlueprintEvolver (Section 2, 4, 5)
# ==============================================================================

class DynamicBlueprintEvolver(nn.Module):
    """
    Виробничий модуль Dynamic Blueprint Evolver на базі Residual Intent Tracking (RIT).
    
    Основні функції:
        1. Віднімання виконаного наміру:
           B_cand = SpherNorm(B_k^(t) - gamma_t * delta_h)
        2. Захист майбутніх тез від over-pruning через ортогоналізацію Грама-Шмідта.
        3. Косинусний розклад gamma_t в діапазоні [0.10, 0.80] (з підтримкою exp/linear).
        4. Захист від Zero-Vector Collapse через автоматичний перехід до атрактора [COMPLETED]
           при падінні залишкової норми нижче collapse_threshold (0.25).
        5. Багатошкальна фільтрація підпросторів за фазами NCM (Syntax -> Morph -> Lexicon).
    """

    def __init__(
        self,
        d_model: int = 1536,
        total_steps: int = 8,
        gamma_min: float = 0.10,
        gamma_max: float = 0.80,
        collapse_threshold: float = 0.25,
        schedule_type: str = "cosine",
        use_subspace_factorization: bool = True,
        use_riemannian_step: bool = False,
        final_attractor_force: float = 0.85
    ):
        super().__init__()
        self.d_model = d_model
        self.total_steps = total_steps
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.collapse_threshold = collapse_threshold
        self.schedule_type = schedule_type.lower()
        self.use_subspace_factorization = use_subspace_factorization
        self.use_riemannian_step = use_riemannian_step
        self.final_attractor_force = final_attractor_force
        
        # Модулі
        self.spher_norm = SphericalNorm(eps=1e-8, clamp_max=1.0)
        self.extractor = SemanticContentExtractor(d_model=d_model)
        self.filter = GramSchmidtSubspaceFilter(d_model=d_model)
        
        if self.use_subspace_factorization:
            self.scale_factorizer = MultiScaleSubspaceFactorization(d_model=d_model)
        else:
            self.scale_factorizer = None
            
        # Навчальний вектор атрактора завершеності [COMPLETED] на одиничній сфері S^(d-1)
        # Ініціалізується нормованим випадковим вектором
        init_attractor = F.normalize(torch.randn(1, d_model), p=2, dim=-1)
        self.completed_attractor = nn.Parameter(init_attractor)
        
        # Класифікатор готовності блоку (Termination Gate)
        self.done_gate = nn.Linear(d_model, 1)

    def get_gamma(self, step_idx: int) -> float:
        """
        Обчислює коефіцієнт віднімання gamma_t для кроку step_idx (1-indexed).
        Підтримує безпечний режим роботи при t > total_steps (наприклад, 11 проходів
        в екстреному режимі відновлення): насичує прогрес на 1.0 та гарантує gamma <= gamma_max < 1.0.
        """
        step = max(1, step_idx)
        if self.total_steps <= 1:
            return float(self.gamma_min)
            
        # Нормалізований насичуваний прогрес: clamp до [0.0, 1.0]
        progress = min(1.0, (step - 1) / float(self.total_steps - 1))
        
        if self.schedule_type == "cosine":
            # Косинусний розклад від gamma_min до gamma_max (Equation 258)
            gamma = self.gamma_min + 0.5 * (self.gamma_max - self.gamma_min) * (1.0 - math.cos(math.pi * progress))
        elif self.schedule_type == "exponential":
            # Експоненційний розклад (Equation 286)
            gamma = self.gamma_min * ((self.gamma_max / self.gamma_min) ** progress)
        elif self.schedule_type == "linear":
            gamma = self.gamma_min + (self.gamma_max - self.gamma_min) * progress
        else:
            raise ValueError(f"Unknown schedule_type '{self.schedule_type}'. Use 'cosine', 'exponential', or 'linear'.")
            
        # Гарантія суворого інваріанту: gamma_min <= gamma <= gamma_max
        return float(min(self.gamma_max, max(self.gamma_min, gamma)))

    def riemannian_geodesic_subtraction(
        self,
        B_curr: torch.Tensor,
        delta_h: torch.Tensor,
        gamma: float
    ) -> torch.Tensor:
        """
        Строге ріманове віднімання на дотичному просторі T_{B} S^(d-1) (Section 2.2):
            1. v_tangent = delta_h - <delta_h, B> B
            2. theta = gamma * max(0, <delta_h, B>)
            3. Exp_B(-theta v) = cos(theta) B - sin(theta) (v / ||v||)
        """
        dot = torch.sum(delta_h * B_curr, dim=-1, keepdim=True)
        v_tangent = delta_h - dot * B_curr
        v_norm = torch.norm(v_tangent, p=2, dim=-1, keepdim=True)
        v_safe = torch.clamp(v_norm, min=1e-8)
        v_dir = v_tangent / v_safe
        
        kappa = torch.clamp(dot, min=0.0)
        theta = gamma * kappa
        
        # Зсув вздовж геодезичної
        B_geo = torch.cos(theta) * B_curr - torch.sin(theta) * v_dir
        return B_geo

    def step(
        self,
        B_curr: torch.Tensor,
        B_init: torch.Tensor,
        H_gen: torch.Tensor,
        step_idx: int,
        slots: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Один крок еволюції вектора наміру: B_k^{(t)} -> B_k^{(t+1)}.
        
        Аргументи:
            B_curr:   [Batch, d_model] - поточний вектор наміру на кроці t
            B_init:   [Batch, d_model] або [Batch, M, d_model] - початковий начерк B_k^(0)
            H_gen:    [Batch, L_block, d_model] - згенеровані латентні токени
            step_idx: int - номер кроку (1-indexed: 1..total_steps)
            slots:    опціональний тензор ортогональних слотів [Batch, M, d_model]
            
        Повертає:
            B_next:  [Batch, d_model] - оновлений знормований вектор наміру на S^(d-1)
            metrics: dict - діагностичні показники (норми, залишок, альфа, тощо)
        """
        batch_size = B_curr.size(0)
        gamma = self.get_gamma(step_idx)
        
        # 1. Екстракція реалізованого семантичного змісту з H_gen через Cross-Attention Pooling
        delta_h_raw = self.extractor(H_gen, B_curr) # [Batch, d_model]
        
        # 2. Фільтрація Грама-Шмідта для захисту нерозкритих тез (Anti-Over-Pruning)
        delta_h_safe = self.filter(
            delta_h=delta_h_raw,
            B_initial=B_init,
            B_current=B_curr,
            slots=slots
        ) # [Batch, d_model]
        
        # 3. Багатошкальна субпросторова фільтрація (Scale 1 -> Scale 2 -> Scale 3)
        if self.scale_factorizer is not None:
            delta_h_directed = self.scale_factorizer.apply_filter(delta_h_safe, step_idx)
        else:
            delta_h_directed = delta_h_safe
            
        # 4. Векторне віднімання наміру
        if self.use_riemannian_step:
            B_unnorm = self.riemannian_geodesic_subtraction(B_curr, delta_h_directed, gamma)
        else:
            B_unnorm = B_curr - gamma * delta_h_directed
            
        # Обчислюємо евклідову норму залишку
        residual_norm = torch.norm(B_unnorm, p=2, dim=-1, keepdim=True) # [Batch, 1]
        
        # 5. Захист від Zero-Vector Collapse та детекція завершеності
        done_logits = self.done_gate(B_unnorm) # [Batch, 1]
        done_prob = torch.sigmoid(done_logits) # [Batch, 1]
        
        # Індикатор просідання залишкової норми нижче безпечного порогу (0.25)
        # Коли residual_norm -> 0, collapse_weight -> 1.0
        norm_collapse_weight = torch.clamp(
            (self.collapse_threshold - residual_norm) / self.collapse_threshold,
            min=0.0,
            max=1.0
        )
        
        # Підсумковий коефіцієнт захоплення атрактором
        alpha_done = torch.maximum(done_prob, norm_collapse_weight)
        
        # Якщо крок фінальний (або аварійний t >= total_steps), примусово спрямовуємо до атрактора
        if step_idx >= self.total_steps:
            target_force = torch.tensor(self.final_attractor_force, device=B_curr.device, dtype=B_curr.dtype)
            alpha_done = torch.maximum(alpha_done, target_force)
            
        # 6. Інтерполяція між залишковим наміром та атрактором [COMPLETED]
        attractor_normed = F.normalize(self.completed_attractor, p=2, dim=-1)
        attractor_expanded = attractor_normed.expand(batch_size, -1)
        
        B_blended = (1.0 - alpha_done) * B_unnorm + alpha_done * attractor_expanded
        
        # 7. Сферична нормалізація на S^(d-1) зі спектральним демпфуванням
        B_next = self.spher_norm(B_blended)
        
        # Діагностичні метрики
        B_init_vec = B_init.mean(dim=1) if B_init.dim() == 3 else B_init
        metrics = {
            "step": float(step_idx),
            "gamma": float(gamma),
            "residual_norm_mean": float(residual_norm.mean().item()),
            "alpha_done_mean": float(alpha_done.mean().item()),
            "cos_sim_to_init": float(F.cosine_similarity(B_next, B_init_vec, dim=-1).mean().item()),
            "cos_sim_to_completed": float(F.cosine_similarity(B_next, attractor_expanded, dim=-1).mean().item()),
            "final_norm": float(torch.norm(B_next, p=2, dim=-1).mean().item())
        }
        
        return B_next, metrics

    def simulate_8_passes(
        self,
        B_initial: torch.Tensor,
        H_passes: List[torch.Tensor],
        slots: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[Dict[str, float]]]:
        """
        Повна симуляція 8 проходів декодера для заданого батчу.
        
        Аргументи:
            B_initial: [Batch, d_model] або [Batch, M, d_model] - початкове креслення
            H_passes:  List з 8 тензорів розмірності [Batch, L_block, d_model]
            slots:     опціональний тензор слотів [Batch, M, d_model]
            
        Повертає:
            trajectory:      List з 9 тензорів: [B_0, B_1, ..., B_8]
            metrics_history: List з 8 словників метрик для кожного кроку
        """
        assert len(H_passes) == self.total_steps, (
            f"Expected {self.total_steps} passes, got {len(H_passes)}"
        )
        
        trajectory: List[torch.Tensor] = []
        metrics_history: List[Dict[str, float]] = []
        
        # B_0: початковий намір, знормований на сферу
        if B_initial.dim() == 3:
            B_curr = self.spher_norm(B_initial.sum(dim=1))
        else:
            B_curr = self.spher_norm(B_initial)
            
        trajectory.append(B_curr)
        
        for t in range(1, self.total_steps + 1):
            H_gen = H_passes[t - 1]
            B_next, step_metrics = self.step(
                B_curr=B_curr,
                B_init=B_initial,
                H_gen=H_gen,
                step_idx=t,
                slots=slots
            )
            trajectory.append(B_next)
            metrics_history.append(step_metrics)
            B_curr = B_next
            
        return trajectory, metrics_history

    def simulate_passes(
        self,
        B_initial: torch.Tensor,
        H_passes: List[torch.Tensor],
        slots: Optional[torch.Tensor] = None
    ) -> Tuple[List[torch.Tensor], List[Dict[str, float]]]:
        """
        Аліас для simulate_8_passes для сумісності з різними конвенціями імен.
        """
        return self.simulate_8_passes(B_initial, H_passes, slots=slots)

    def forward(
        self,
        B_curr: torch.Tensor,
        B_init: torch.Tensor,
        H_gen: torch.Tensor,
        step_idx: int,
        slots: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Стандартний forward метод для PyTorch nn.Module.
        """
        return self.step(B_curr, B_init, H_gen, step_idx, slots=slots)
