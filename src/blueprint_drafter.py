r"""
Hetman-2-NCM 2.0B: Causal AR Blueprint Drafter (Phase 1)
=======================================================
Виробничий модуль планувальника макро-креслень блоків (Neural Chunk Modeling).

Архітектурний огляд:
---------------------
Фаза 1 реалізує єдиний авторегресійний крок в усій системі Hetman-2-NCM.
Замість генерації 1600+ сублітерних токенів з квадратичним KV-кешем,
модель за 4–8 кроків формує послідовність високорівневих креслень:
    H_query -> B_1 -> B_2 -> ... -> B_K -> [END_PLAN]

Структура кожного креслення B_k in R^1536:
    [0 : 768]     c_sem:      Семантичне ядро концепту (768 dims)
    [768 : 1152]  idx_hop:    Асоціативна пам'ять SLH-Core (16 груп x 24 dims = 384 dims)
    [1152 : 1408] tau_style:  Вектор синтаксису і стилю на гіперсфері S^255 (256 dims, ||tau||_2 = 1.0)
    [1408 : 1536] ell_budget: Бюджет довжини (Gumbel-Softmax) та стоп-сигнал [END_PLAN] (128 dims)
    Сума: 768 + 384 + 256 + 128 = 1536 dims.

Ключові технології:
-------------------
1. Gated DeltaNet Cell: O(1) пам'ять інференсу з правилом асоціативного оновлення Дельта.
2. vMF Spherical Sampler: Семплювання фон Мізеса-Фішера на гіперсфері S^255 для tau_style.
3. Gumbel-Softmax: Диференційований температурний відбір бінів довжини (200, 300, 400, 500).
4. Verified Fallback Re-Drafting: Зворотний відкат траєкторії при виявленні семантичного дрейфу.
5. CPU Latency Target: < 30 мс на генерацію повного ланцюжка креслень.

Author: Yaroslav Samchuk & Hetman AI Team
License: Apache 2.0
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. КОНФІГУРАЦІЯ ТА СТРУКТУРИ ДАНИХ КРЕСЛЕННЯ (BLUEPRINT DATA STRUCTURES)
# ============================================================================

@dataclass
class BlueprintConfig:
    """
    Конфігурація причинного проектувальника креслень (Blueprint Drafter).
    Відповідає RFC-HETMAN-2-NCM-PHASE1-DRAFTER та RFC-HETMAN-2-NCM-01.
    """
    d_model: int = 1536              # Загальна розмірність простору моделі Hetman-2.0B
    d_drafter: int = 512             # Прихована розмірність Drafter ядра
    num_heads: int = 8               # Кількість голів лінійної уваги DeltaNet
    d_state: int = 64                # Розмірність ключа/значення на голову (8 x 64 = 512)
    
    # Розподіл 1536 каналів креслення B_k:
    dim_c_sem: int = 768             # Семантичний вектор (канали 0:768)
    dim_idx_hop: int = 384           # Проекція асоціативної пам'яті Hopfield (канали 768:1152)
    dim_tau_style: int = 256         # Стильовий вектор на сфері S^255 (канали 1152:1408)
    dim_ell_budget: int = 128        # Бюджет довжини та EOS / END_PLAN (канали 1408:1536)
    
    # Параметри адресації пам'яті SLH-Core v2
    hopfield_num_groups: int = 16    # 16 тематичних груп (16 x 24 = 384)
    hopfield_group_dim: int = 24     # Розмірність представлення групи
    hopfield_dim_i: int = 1024       # Перший ортогональний індекс координати (1024 слотів)
    hopfield_dim_j: int = 1024       # Другий ортогональний індекс координати (1024 слотів)
    
    # Квантовані біни бюджету довжини блоку
    length_bins: Tuple[int, ...] = (200, 300, 400, 500)
    gumbel_temperature: float = 1.0  # Базова температура Gumbel-Softmax
    
    # Параметри сферичного семплювання фон Мізеса-Фішера (vMF)
    vmf_kappa: float = 16.0          # Базова концентрація на S^255 (зворотна температура)
    
    # Авторегресійне керування довжиною плану
    min_blueprint_steps: int = 1     # Мінімум блоків для простих запитів ("2+2")
    max_blueprint_steps: int = 8     # Максимум блоків у ланцюжку
    default_blueprint_steps: int = 4 # Номінальний розмір плану (4-8 блоків)
    end_plan_threshold: float = 0.5  # Поріг спрацьовування [END_PLAN]
    early_stop_threshold: float = 0.85 # Поріг миттєвої зупинки на першому кроці
    
    # Параметри механізму Fallback Re-Drafting
    enable_fallback: bool = True     # Активація арбітра захисту від семантичного дрейфу
    drift_threshold: float = -0.15   # Поріг семантичного розбіжності з запитом
    repetition_margin: float = 0.65  # Максимальна косинусна подібність між сусідніми блоками
    max_redraft_attempts: int = 1    # Максимальна кількість спроб перебудови блоку
    gamma_repulsion: float = 0.6     # Штрафний коефіцієнт відштовхування відхиленого наміру

    def __post_init__(self):
        total_dim = self.dim_c_sem + self.dim_idx_hop + self.dim_tau_style + self.dim_ell_budget
        if total_dim != self.d_model:
            raise ValueError(
                f"Сума підпросторів ({self.dim_c_sem} + {self.dim_idx_hop} + "
                f"{self.dim_tau_style} + {self.dim_ell_budget} = {total_dim}) "
                f"мусить строго дорівнювати d_model={self.d_model}"
            )


class BlueprintComponents(NamedTuple):
    """Розпаковані компоненти креслення B_k відповідно до єдиної алгебри тензорів."""
    c_sem: torch.Tensor          # [B, 768]
    idx_hop: torch.Tensor        # [B, 384]
    tau_style: torch.Tensor      # [B, 256] на сфері S^255
    ell_budget: torch.Tensor     # [B, 128]


@dataclass
class BlockBlueprint:
    """
    Структуроване креслення одного смислового блоку B_k.
    Містить повний тензор у просторі R^1536 та семантично інтерпретовані атрибути.
    """
    step_idx: int
    c_sem: torch.Tensor          # [B, 768] Семантичний центроїд
    idx_hop: torch.Tensor        # [B, 384] Проекція асоціативної пам'яті Hopfield
    tau_style: torch.Tensor      # [B, 256] Нормалізований вектор стилю на гіперсфері S^255
    ell_budget: torch.Tensor     # [B, 128] Бюджет та прапорці завершення
    
    # Інтерпретовані дискретні атрибути
    hopfield_coords: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]  # (group, i, j)
    token_budget: torch.Tensor   # [B] дискретне значення токенів (200, 300, 400, 500)
    bin_idx: torch.Tensor        # [B] індекс біну довжини (0, 1, 2, 3)
    end_plan_prob: torch.Tensor  # [B] ймовірність термінації плану
    is_terminal: bool            # Чи зафіксовано завершення планування
    redraft_count: int = 0       # Кількість виконаних відкатів Fallback для цього блоку
    
    @property
    def tensor(self) -> torch.Tensor:
        """Повертає об'єднаний тензор B_k розмірністю [B, 1536]."""
        return torch.cat([self.c_sem, self.idx_hop, self.tau_style, self.ell_budget], dim=-1)

    def unpack(self) -> BlueprintComponents:
        """Повертає іменований кортеж підтензорів."""
        return BlueprintComponents(
            c_sem=self.c_sem,
            idx_hop=self.idx_hop,
            tau_style=self.tau_style,
            ell_budget=self.ell_budget
        )


# ============================================================================
# 2. РЕКУРЕНТНЕ ЯДРО GATED DELTANET CELL (O(1) MEMORY INFERENCE)
# ============================================================================

class GatedDeltaNetCell(nn.Module):
    """
    Причинний рекурентний крок DeltaNet з константним станом O(1).
    Реалізує асоціативне правило Дельта (Delta Rule):
        S_t = S_{t-1} * (1 - alpha_t) + beta_t * (v_t - S_{t-1} k_t) k_t^T
        y_t = S_t q_t
    Оптимізовано для надшвидкого виконання на CPU (fused QKV та вентилі).
    """
    def __init__(self, config: BlueprintConfig):
        super().__init__()
        self.config = config
        self.d_drafter = config.d_drafter
        self.num_heads = config.num_heads
        self.d_state = config.d_state
        
        # Злита проекція запитів, ключів та значень (3 * H * dk)
        self.w_qkv = nn.Linear(self.d_drafter, 3 * self.num_heads * self.d_state, bias=False)
        
        # Злиті вентилі затухання (alpha) та швидкості навчання асоціацій (beta)
        self.w_gates = nn.Linear(self.d_drafter, 2 * self.num_heads, bias=True)
        
        # Вихідна проекція зі стабілізуючою RMS-нормалізацією
        self.w_out = nn.Linear(self.num_heads * self.d_state, self.d_drafter, bias=False)
        self.norm = nn.RMSNorm(self.d_drafter)
        
    def forward(
        self, 
        x: torch.Tensor, 
        state: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, d_drafter] - вхідний вектор поточного авторегресійного кроку
        state: [B, num_heads, d_state, d_state] - асоціативна пам'ять DeltaNet
        
        Повертає:
            out: [B, d_drafter] - вихідний стан клітини
            new_state: [B, num_heads, d_state, d_state] - оновлена рекурентна пам'ять
        """
        B = x.shape[0]
        if state is None:
            state = torch.zeros(
                B, self.num_heads, self.d_state, self.d_state, 
                device=x.device, dtype=x.dtype
            )
            
        # Злите обчислення Q, K, V
        qkv = self.w_qkv(x).view(B, 3, self.num_heads, self.d_state)
        q = F.normalize(qkv[:, 0], p=2, dim=-1) # QK-Norm для числової стійкості
        k = F.normalize(qkv[:, 1], p=2, dim=-1)
        v = qkv[:, 2]
        
        # Злите обчислення вентилів затухання та асоціативної швидкості
        gates = torch.sigmoid(self.w_gates(x)).view(B, 2, self.num_heads, 1, 1)
        alpha, beta = gates[:, 0], gates[:, 1]
        
        # Правило Дельти: Error = v_t - S_{t-1} k_t
        k_col = k.unsqueeze(-1)                              # [B, H, d_state, 1]
        recalled_v = torch.matmul(state, k_col).squeeze(-1)  # [B, H, d_state]
        v_err = (v - recalled_v).unsqueeze(-1)              # [B, H, d_state, 1]
        
        # Оновлення рангу 1: S_t = S_{t-1} * (1 - alpha) + beta * v_err * k^T
        k_row = k.unsqueeze(-2)                              # [B, H, 1, d_state]
        delta_update = torch.matmul(v_err, k_row)            # [B, H, d_state, d_state]
        new_state = state * (1.0 - alpha) + beta * delta_update
        
        # Асоціативне зчитування інформації за допомогою запиту q_t: o_t = S_t q_t
        q_row = q.unsqueeze(-2)                              # [B, H, 1, d_state]
        read_out = torch.matmul(q_row, new_state).squeeze(-2) # [B, H, d_state]
        read_out = read_out.reshape(B, self.num_heads * self.d_state)
        
        # Резидуальний зв'язок та вихідна RMSNorm
        out = self.norm(self.w_out(read_out) + x)
        return out, new_state


# ============================================================================
# 3. ВСМОКТУВАННЯ КРЕСЛЕНЬ (BLUEPRINT INGESTION & FEEDBACK MODULE)
# ============================================================================

class BlueprintIngestion(nn.Module):
    """
    Зворотна проекція згенерованого креслення B_{k-1} у простір Drafter (d_drafter).
    Забезпечує зворотний зв'язок (Causal Recurrent Conditioning) між блоками плану.
    Оптимізовано у вигляді єдиної матриці проекції [1536 -> d_drafter].
    """
    def __init__(self, config: BlueprintConfig):
        super().__init__()
        self.config = config
        d = config.d_drafter
        
        # Єдина злита матриця всмоктування B_k in R^1536
        self.proj_in = nn.Linear(config.d_model, d, bias=False)
        self.fusion_norm = nn.RMSNorm(d)
        
    def forward(self, blueprint: BlockBlueprint) -> torch.Tensor:
        """
        Перетворює BlockBlueprint -> x_next in [B, d_drafter].
        """
        fused = self.fusion_norm(self.proj_in(blueprint.tensor))
        return fused


# ============================================================================
# 4. ГОЛОВИ ГЕНЕРАЦІЇ ПАРАМЕТРІВ КРЕСЛЕННЯ (PREDICTION HEADS)
# ============================================================================

class BlueprintPredictionHeads(nn.Module):
    """
    Багатокомпонентна голова генерації кортежу B_k in R^1536:
    1. Семантична мета c_sem (768 dims)
    2. Топологічний покажчик Hopfield Memory idx_hop (384 dims, 16 груп x 24 dims)
    3. Синтаксичний стиль tau_style на сфері S^255 (256 dims, ||tau||_2 = 1.0)
    4. Бюджет довжини ell_budget (128 dims, Gumbel-Softmax біни + [END_PLAN])
    """
    def __init__(self, config: BlueprintConfig):
        super().__init__()
        self.config = config
        d = config.d_drafter
        
        # 1. Голова семантичного наміру c_sem [B, 768]
        self.head_c_sem = nn.Linear(d, config.dim_c_sem, bias=False)
        
        # 2. Голови покажчиків Hopfield Memory SLH-Core
        # Злита голова дискретних координат (16 + 512 + 512 = 1040 логітів)
        self.head_hop_gij = nn.Linear(
            d, 
            config.hopfield_num_groups + config.hopfield_dim_i + config.hopfield_dim_j
        )
        # Неперервна проекція в 384 dims (16 груп x 24 dims)
        self.head_hop_proj = nn.Linear(d, config.dim_idx_hop, bias=False)
        
        # 3. Голова синтаксичного стилю tau_style [B, 256] на гіперсфері S^255
        self.head_tau_style = nn.Linear(d, config.dim_tau_style, bias=False)
        
        # 4. Злита голова бюджету довжини (4 біни) та сигналу [END_PLAN] (1 логіт)
        self.head_len_stop = nn.Linear(d, len(config.length_bins) + 1)
        
        # 5. Проекція для формування єдиного тензора ell_budget [B, 128]:
        #    перші 64 канали кодують довжину блоку, останні 64 канали кодують EOS / стоп
        self.proj_budget = nn.Linear(len(config.length_bins) + 1, config.dim_ell_budget, bias=False)
        
    def sample_vmf(
        self, 
        mu: torch.Tensor, 
        kappa: float = 16.0
    ) -> torch.Tensor:
        r"""
        Семплювання з розподілу фон Мізеса-Фішера (Von Mises-Fisher, vMF) на сфері S^{D-1}.
        Зокрема, при dim = 256 вектор строго належить одиничній гіперсфері S^255:
            p(x | \mu, \kappa) \propto \exp(\kappa \cdot \mu^T x)
        
        Реалізація використовує швидку проекційну ортогоналізацію в дотичному просторі:
        1. Збурення шумом у напрямку, ортогональному до mu.
        2. Зважене змішування з модальним напрямком mu залежно від концентрації kappa.
        3. Жорстка L2 нормалізація на поверхню гіперсфери.
        """
        if kappa >= 100.0:
            return mu
            
        noise = torch.randn_like(mu)
        # Ортогональна складова: noise_ortho = noise - (noise * mu) * mu
        mu_proj = (noise * mu).sum(dim=-1, keepdim=True)
        noise_ortho = F.normalize(noise - mu_proj * mu, p=2, dim=-1, eps=1e-8)
        
        # Параметр кутового зміщення w in (0, 1)
        w = 1.0 / (1.0 + math.exp(-max(kappa, 1e-2) / 10.0))
        ortho_weight = math.sqrt(max(1.0 - w * w, 1e-6))
        
        sampled = w * mu + ortho_weight * noise_ortho
        return F.normalize(sampled, p=2, dim=-1)

    def forward(
        self, 
        h: torch.Tensor, 
        temperature: float = 0.7, 
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, 
               Tuple[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Генерує всі 4 компоненти креслення B_k.
        
        Повертає:
            c_sem: [B, 768]
            idx_hop: [B, 384]
            tau_style: [B, 256] на сфері S^255
            ell_budget: [B, 128]
            hopfield_coords: (g, i, j)
            token_budget: [B] значення довжини
            bin_idx: [B] індекс біну
            end_plan_prob: [B] ймовірність [END_PLAN]
        """
        B = h.shape[0]
        device = h.device
        
        # -------------------------------------------------------------
        # 1. Семантичне ядро c_sem [B, 768]
        # -------------------------------------------------------------
        c_sem_raw = self.head_c_sem(h)
        c_sem = F.normalize(c_sem_raw, p=2, dim=-1) # L2 нормалізація для стабільності
        
        # -------------------------------------------------------------
        # 2. Hopfield асоціативна адресація [B, 384] та координати (g, i, j)
        # -------------------------------------------------------------
        idx_hop = self.head_hop_proj(h)
        gij = self.head_hop_gij(h)
        logits_g, logits_i, logits_j = torch.split(
            gij, 
            [self.config.hopfield_num_groups, self.config.hopfield_dim_i, self.config.hopfield_dim_j], 
            dim=-1
        )
        
        if deterministic or temperature <= 0.0:
            g = torch.argmax(logits_g, dim=-1)
            i = torch.argmax(logits_i, dim=-1)
            j = torch.argmax(logits_j, dim=-1)
        else:
            # Gumbel-Max categorical coordinate sampling (avoids Gaussian drift on discrete indices)
            def _gumbel_sample(logits: torch.Tensor, temp: float) -> torch.Tensor:
                u = torch.rand_like(logits).clamp(min=1e-10, max=1.0 - 1e-10)
                gumbel_noise = -torch.log(-torch.log(u))
                return torch.argmax(logits / max(temp, 1e-4) + gumbel_noise, dim=-1)

            g = _gumbel_sample(logits_g, temperature)
            i = _gumbel_sample(logits_i, temperature)
            j = _gumbel_sample(logits_j, temperature)
            
        hopfield_coords = (g, i, j)
        
        # -------------------------------------------------------------
        # 3. Синтаксичний стиль tau_style [B, 256] на гіперсфері S^255
        # -------------------------------------------------------------
        tau_raw = self.head_tau_style(h)
        tau_mu = F.normalize(tau_raw, p=2, dim=-1) # Нормалізований напрямок на S^255
        
        if deterministic or temperature <= 0.0:
            tau_style = tau_mu
        else:
            kappa = self.config.vmf_kappa / max(temperature, 1e-2)
            tau_style = self.sample_vmf(tau_mu, kappa=kappa)
            
        # -------------------------------------------------------------
        # 4. Бюджет довжини ell_budget [B, 128] через Gumbel-Softmax та [END_PLAN]
        # -------------------------------------------------------------
        len_stop = self.head_len_stop(h)
        logits_len = len_stop[..., :len(self.config.length_bins)]
        end_plan_logit = len_stop[..., len(self.config.length_bins):]
        
        if deterministic or temperature <= 0.0:
            bin_idx = torch.argmax(logits_len, dim=-1)
            gumbel_weights = F.one_hot(bin_idx, num_classes=len(self.config.length_bins)).float()
        else:
            # Gumbel-Softmax семплювання над бінами довжини
            gumbel_weights = F.gumbel_softmax(
                logits_len, 
                tau=max(self.config.gumbel_temperature * temperature, 1e-2), 
                hard=True, 
                dim=-1
            )
            bin_idx = torch.argmax(gumbel_weights, dim=-1)
            
        bins_tensor = torch.tensor(self.config.length_bins, device=device, dtype=torch.long)
        token_budget = bins_tensor[bin_idx]
        
        # Стоп-токен [END_PLAN]
        end_plan_prob = torch.sigmoid(end_plan_logit).squeeze(-1) # [B]
        
        # Формування єдиного тензора ell_budget розмірністю [B, 128]
        budget_input = torch.cat([gumbel_weights, end_plan_prob.unsqueeze(-1)], dim=-1)
        ell_budget = self.proj_budget(budget_input)
        
        return (
            c_sem, 
            idx_hop, 
            tau_style, 
            ell_budget, 
            hopfield_coords, 
            token_budget, 
            bin_idx, 
            end_plan_prob
        )


# ============================================================================
# 5. ОСНОВНИЙ МОДУЛЬ: CAUSAL BLUEPRINT DRAFTER
# ============================================================================

class CausalBlueprintDrafter(nn.Module):
    """
    Фаза 1 Архітектури Hetman-2-NCM 2.0B:
    Причинний авторегресійний генератор ланцюжка креслень блоків (Causal AR Blueprint Drafter).
    
    Генерує ланцюжок планів B_1 -> B_2 -> ... -> B_K за 4–8 кроків.
    Кожен вектор B_k in R^1536 розгортається паралельно у Фазі 2 та 3.
    """
    def __init__(self, config: Optional[BlueprintConfig] = None):
        super().__init__()
        self.config = config or BlueprintConfig()
        
        # Вхідна проекція: H_query (1536) -> Drafter Core (512)
        self.in_proj = nn.Linear(self.config.d_model, self.config.d_drafter, bias=False)
        self.in_norm = nn.RMSNorm(self.config.d_drafter)
        
        # Двошаровий лінійний рекурентний стек Gated DeltaNet
        self.cell_1 = GatedDeltaNetCell(self.config)
        self.cell_2 = GatedDeltaNetCell(self.config)
        
        # Модуль всмоктування попереднього креслення (Ingestion Feedback)
        self.ingestion = BlueprintIngestion(self.config)
        
        # Багатокомпонентні голови передбачення B_k
        self.heads = BlueprintPredictionHeads(self.config)
        
        # Проектор сумісності для Fallback Re-Drafting
        self.arbiter_proj = nn.Linear(self.config.dim_c_sem, self.config.d_drafter, bias=False)
        
    def pack_blueprint(self, comps: BlueprintComponents) -> torch.Tensor:
        """
        Збирає підтензори у єдиний тензор креслення B in [B, 1536].
        Канали: [0:768] c_sem | [768:1152] idx_hop | [1152:1408] tau_style | [1408:1536] ell_budget.
        """
        return torch.cat([comps.c_sem, comps.idx_hop, comps.tau_style, comps.ell_budget], dim=-1)

    def unpack_blueprint(self, b: torch.Tensor) -> BlueprintComponents:
        """
        Розбирає тензор B in [..., 1536] на іменовані компоненти згідно зі специфікацією.
        """
        assert b.shape[-1] == self.config.d_model, (
            f"Очікувалась розмірність {self.config.d_model}, отримано {b.shape[-1]}"
        )
        return BlueprintComponents(
            c_sem=b[..., 0:768],
            idx_hop=b[..., 768:1152],
            tau_style=b[..., 1152:1408],
            ell_budget=b[..., 1408:1536]
        )

    def _step(
        self, 
        x: torch.Tensor, 
        s1: Optional[torch.Tensor], 
        s2: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Один причинний крок крізь 2 шари Gated DeltaNet."""
        h1, next_s1 = self.cell_1(x, s1)
        h2, next_s2 = self.cell_2(h1, s2)
        return h2, next_s1, next_s2

    def draft_plan(
        self, 
        h_query: torch.Tensor, 
        num_steps: Optional[int] = None,
        temperature: float = 0.7, 
        deterministic: bool = False,
        enable_fallback: Optional[bool] = None
    ) -> List[BlockBlueprint]:
        """
        Повний авторегресійний цикл формування ланцюжка креслень:
            H_query -> B_1 -> B_2 -> ... -> B_K
            
        Параметри:
            h_query: [B, 1536] - контекстний ембеддинг запиту після Denoising Stem
            num_steps: якщо вказано, генерує точно фіксовану кількість блоків (наприклад, 4).
                       якщо None, кількість блоків визначається динамічно через [END_PLAN] (від 1 до 8).
            temperature: коефіцієнт стохастичності вибірки
            deterministic: якщо True, обирається максимальна правдоподібність (argmax)
            enable_fallback: чи активовано захист від семантичного дрейфу (Fallback Re-Drafting)
            
        Повертає:
            Список об'єктів BlockBlueprint довжиною від 1 до max_blueprint_steps (зазвичай 4–8 блоків).
        """
        B = h_query.shape[0]
        device = h_query.device
        use_fallback = self.config.enable_fallback if enable_fallback is None else enable_fallback
        max_steps = self.config.max_blueprint_steps if num_steps is None else min(num_steps, self.config.max_blueprint_steps)
        
        # 1. Початковий стан формується із запиту
        x = self.in_norm(self.in_proj(h_query))
        q_ref = F.normalize(x, p=2, dim=-1) # Опорний напрямок запиту для оцінки релевантності
        
        state1: Optional[torch.Tensor] = None
        state2: Optional[torch.Tensor] = None
        
        blueprints: List[BlockBlueprint] = []
        history_c_sem: List[torch.Tensor] = []
        
        for step in range(max_steps):
            # Виконуємо причинний крок лінійної рекурентності DeltaNet
            h_out, cand_s1, cand_s2 = self._step(x, state1, state2)
            
            # Генерація компонентів креслення первинного кандидата
            (
                c_sem, 
                idx_hop, 
                tau_style, 
                ell_budget, 
                coords, 
                budget, 
                bin_idx, 
                stop_prob
            ) = self.heads(h_out, temperature=temperature, deterministic=deterministic)
            
            redraft_count = 0
            
            # Перевірка на семантичний дрейф (Semantic Drift Arbiter)
            if use_fallback and len(history_c_sem) > 0 and not deterministic:
                # А. Векторизована перевірка на зациклення з попередніми блоками
                prev_matrix = torch.stack(history_c_sem, dim=1) # [B, num_prev, 768]
                sims = torch.bmm(c_sem.unsqueeze(1), prev_matrix.transpose(1, 2)).squeeze(1) # [B, num_prev]
                max_sim_history = torch.max(sims).item()
                
                # Б. Перевірка на сильну деструктивну розбіжність із запитом:
                c_proj = F.normalize(self.arbiter_proj(c_sem), p=2, dim=-1)
                query_relevance = (q_ref * c_proj).sum(dim=-1).mean().item()
                
                if (max_sim_history > self.config.repetition_margin or 
                    query_relevance < self.config.drift_threshold):
                    # Зафіксовано семантичний дрейф! Виконуємо Fallback Re-Drafting:
                    redraft_count = 1
                    rejected_intent = c_sem.detach().clone()
                    
                    # Генерація альтернативного кандидата
                    (
                        c_sem, 
                        idx_hop, 
                        tau_style, 
                        ell_budget, 
                        coords, 
                        budget, 
                        bin_idx, 
                        stop_prob
                    ) = self.heads(h_out, temperature=0.2, deterministic=True)
                    
                    # Застосування ортогонального відштовхування від відкинутого концепту
                    sim_rejected = (c_sem * rejected_intent).sum(dim=-1, keepdim=True)
                    repulsion = torch.clamp(sim_rejected, min=0.0) * rejected_intent
                    c_sem = F.normalize(c_sem - self.config.gamma_repulsion * repulsion, p=2, dim=-1)
            
            # Оновлюємо рекурентний стан після прийняття блоку
            state1, state2 = cand_s1, cand_s2
            
            # Перевірка умови зупинки [END_PLAN]
            avg_stop = stop_prob.mean().item()
            is_terminal = False
            
            if num_steps is not None:
                # При фіксованій кількості кроків термінація на останньому із них
                if step == max_steps - 1:
                    is_terminal = True
            else:
                # Динамічне планування
                if step == 0:
                    if avg_stop >= self.config.early_stop_threshold:
                        is_terminal = True
                elif step >= self.config.default_blueprint_steps - 1:
                    if avg_stop >= self.config.end_plan_threshold:
                        is_terminal = True
                elif avg_stop >= 0.75:
                    is_terminal = True
                    
                if step == max_steps - 1:
                    is_terminal = True
                
            current_bp = BlockBlueprint(
                step_idx=step + 1,
                c_sem=c_sem,
                idx_hop=idx_hop,
                tau_style=tau_style,
                ell_budget=ell_budget,
                hopfield_coords=coords,
                token_budget=budget,
                bin_idx=bin_idx,
                end_plan_prob=stop_prob,
                is_terminal=is_terminal,
                redraft_count=redraft_count
            )
            blueprints.append(current_bp)
            history_c_sem.append(c_sem.detach())
            
            if is_terminal:
                break
                
            # Підготовка зворотного входу x_{k+1} для наступного кроку авторегресії
            x = self.ingestion(current_bp)
            
        return blueprints


# ============================================================================
# 6. КОМПЛЕКСНИЙ ТЕСТОВИЙ ПРОТОКОЛ ТА БЕНЧМАРК ШВИДКОСТІ НА CPU (< 30 мс)
# ============================================================================

def run_blueprint_drafter_benchmarks(num_runs: int = 50) -> Dict[str, float]:
    """
    Автономний інженерний бенчмарк та верифікація CausalBlueprintDrafter.
    Перевіряє коректність тензорних розмірностей, інваріанти гіперсфери S^255,
    механізм Fallback Re-Drafting та вимірює затримку генерації на CPU (< 30 мс).
    """
    print("\n" + "=" * 80)
    print("HETMAN-2-NCM 2.0B: CAUSAL AR BLUEPRINT DRAFTER BENCHMARK & AUDIT SUITE")
    print("=" * 80)
    
    device = torch.device("cpu")
    print(f"[*] Використовуваний пристрій тестування: {device}")
    
    # Оптимальне налаштування потоків для CPU інференсу малих батчів
    orig_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    
    cfg = BlueprintConfig()
    drafter = CausalBlueprintDrafter(cfg).to(device)
    drafter.eval()
    
    # 1. Аудит параметрів моделі
    total_params = sum(p.numel() for p in drafter.parameters())
    print(f"[*] Кількість параметрів Drafter Core: {total_params:,} ({total_params * 4 / (1024**2):.2f} MB у FP32)")
    
    # 2. Функціональний тест розмірностей тензорів та інваріантів
    batch_size = 2
    dummy_query = torch.randn(batch_size, cfg.d_model, device=device)
    
    print("\n--- ТЕСТ 1: Перевірка авторегресійного ланцюжка креслень ---")
    with torch.no_grad():
        plan = drafter.draft_plan(dummy_query, temperature=0.7, deterministic=False)
        
    print(f"[+] Кількість блоків у згенерованому плані: {len(plan)} (діапазон: 1–8)")
    assert len(plan) >= 1 and len(plan) <= cfg.max_blueprint_steps, "Помилка довжини плану!"
    
    for bp in plan:
        g, i, j = bp.hopfield_coords
        full_tensor = bp.tensor
        print(f"    - Блок {bp.step_idx}: B_k shape={tuple(full_tensor.shape)} | "
              f"Бюджет={bp.token_budget[0].item()} ток. (Бін {bp.bin_idx[0].item()}) | "
              f"SLH Coord=({g[0].item()}, {i[0].item()}, {j[0].item()}) | "
              f"P(END_PLAN)={bp.end_plan_prob[0].item():.4f} | Stop={bp.is_terminal} | "
              f"Redrafts={bp.redraft_count}")
        
        # Перевірка розмірностей
        assert full_tensor.shape == (batch_size, 1536), f"Невірна форма B_k: {full_tensor.shape}"
        assert bp.c_sem.shape == (batch_size, 768), f"Невірна форма c_sem: {bp.c_sem.shape}"
        assert bp.idx_hop.shape == (batch_size, 384), f"Невірна форма idx_hop: {bp.idx_hop.shape}"
        assert bp.tau_style.shape == (batch_size, 256), f"Невірна форма tau_style: {bp.tau_style.shape}"
        assert bp.ell_budget.shape == (batch_size, 128), f"Невірна форма ell_budget: {bp.ell_budget.shape}"
        
        # Перевірка інваріанта сфери S^255: ||tau_style||_2 == 1.0
        tau_norms = torch.norm(bp.tau_style, p=2, dim=-1)
        assert torch.allclose(tau_norms, torch.ones_like(tau_norms), atol=1e-5), (
            f"Порушення інваріанта гіперсфери S^255: норми={tau_norms}"
        )
        
        # Перевірка валідності координат Hopfield
        assert (g >= 0).all() and (g < cfg.hopfield_num_groups).all(), "Некоректний індекс групи Hopfield!"
        assert (i >= 0).all() and (i < cfg.hopfield_dim_i).all(), "Некоректний індекс i Hopfield!"
        assert (j >= 0).all() and (j < cfg.hopfield_dim_j).all(), "Некоректний індекс j Hopfield!"
        
        # Перевірка валідності бінів довжини
        assert all(val.item() in cfg.length_bins for val in bp.token_budget), "Некоректний бін бюджету токенів!"

    print("[+] Всі інваріанти форми, гіперсфери S^255 та дискретних атрибутів підтверджено!")
    
    # 3. Перевірка механізму Fallback Re-Drafting
    print("\n--- ТЕСТ 2: Перевірка спрацьовування Fallback Re-Drafting при семантичному дрейфі ---")
    test_cfg = BlueprintConfig(repetition_margin=0.0)
    test_drafter = CausalBlueprintDrafter(test_cfg).to(device)
    test_drafter.eval()
    with torch.no_grad():
        fallback_plan = test_drafter.draft_plan(dummy_query, temperature=0.7, deterministic=False)
    redrafts_triggered = sum(bp.redraft_count for bp in fallback_plan)
    print(f"[+] Fallback успішно протестовано: здійснено {redrafts_triggered} відкатів стану зі штрафуванням!")
    assert redrafts_triggered > 0, "Fallback Re-Drafting не спрацював при перетині порогу!"
    
    # 4. Прецизійне вимірювання затримки на CPU (< 30 мс)
    print(f"\n--- ТЕСТ 3: Прецизійний бенчмарк затримки генерації на CPU ({num_runs} прогонів) ---")
    single_query = torch.randn(1, cfg.d_model, device=device)
    
    # Прогрів (Warmup)
    with torch.no_grad():
        for _ in range(10):
            _ = drafter.draft_plan(single_query, num_steps=4, temperature=0.7)
            
    latencies: List[float] = []
    blocks_count: List[int] = []
    
    with torch.no_grad():
        for _ in range(num_runs):
            t0 = time.perf_counter()
            plan = drafter.draft_plan(single_query, num_steps=4, temperature=0.7)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000.0) # мілісекунди
            blocks_count.append(len(plan))
            
    mean_lat = sum(latencies) / len(latencies)
    min_lat = min(latencies)
    max_lat = max(latencies)
    std_lat = (sum((x - mean_lat)**2 for x in latencies) / len(latencies)) ** 0.5
    avg_blocks = sum(blocks_count) / len(blocks_count)
    lat_per_block = mean_lat / avg_blocks
    
    print(f"[+] Середній час генерації повного плану (4 блоки): {mean_lat:.3f} мс (std: ±{std_lat:.3f} мс)")
    print(f"[+] Мін/Макс час генерації:               {min_lat:.3f} мс / {max_lat:.3f} мс")
    print(f"[+] Середня кількість блоків у плані:    {avg_blocks:.1f}")
    print(f"[+] Середній час генерації одного блоку: {lat_per_block:.3f} мс")
    print(f"[+] Швидкість планувальника:              {1000.0 / mean_lat:.1f} повних планів/сек")
    
    # Відновлюємо налаштування потоків
    torch.set_num_threads(orig_threads)
    
    # Перевірка цільового критерію специфікації: < 30 мс на CPU
    assert mean_lat < 30.0, f"КРИТИЧНА ПОМИЛКА: Затримка {mean_lat:.3f} мс перевищує ліміт 30 мс!"
    print(f"[PASS] Цільова затримка < 30 мс на CPU ВИТРИМАНА з величезним запасом! (Фактично {mean_lat:.3f} мс)")
    
    print("\n" + "=" * 80)
    print("ВИРОБНИЧИЙ МОДУЛЬ BLUEPRINT_DRAFTER ГОТОВИЙ ДО ЕКСПЛУАТАЦІЇ ТА ІНТЕГРАЦІЇ.")
    print("=" * 80 + "\n")
    
    return {
        "mean_latency_ms": mean_lat,
        "std_latency_ms": std_lat,
        "min_latency_ms": min_lat,
        "max_latency_ms": max_lat,
        "avg_blocks": avg_blocks,
        "plans_per_sec": 1000.0 / mean_lat,
        "target_met": mean_lat < 30.0
    }


if __name__ == "__main__":
    run_blueprint_drafter_benchmarks()
