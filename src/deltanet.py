"""
Gated DeltaNet Linear Recurrent Attention Layer (Hetman-2-NCM Frontier Edition)
================================================================================
Виробничий модуль лінійної рекурентності Hetman-2-NCM 2.0B
Специфікація: spec/03_GATED_DELTANET_RECURRENT_STATE_DYNAMICS.md
Сумісність: TH1/hetman-core/src/deltanet.py + Hetman-2-NCM Block-Causal NCM

Ключові архітектурні інваріанти:
1. Просторова компресія пляшкового горла: 1536 -> 1024 (16 голів x 64 head_dim) -> 1536.
2. Рекурентна матриця асоціативної пам'яті S_t in R^{16 x 64 x 64} СТРОГО у FP32:
   - Розмір пам'яті одного шару (B=1): 256.000 KiB.
   - Сумарний стан 28 шарів DeltaNet (B=1): РІВНО 7.000000 MiB (7,340,032 байти).
3. Каузальна 1D Depthwise-згортка (k=4, groups=1536) для фіксації морфології.
4. Gated Delta Rule з мультимасштабним канальним затуханням:
   - Fast Heads (0..3): alpha_bias = +1.0 (tau ≈ 2.21 токенів)
   - Episodic Heads (4..11): alpha_bias = +3.5 (tau ≈ 23.31 токенів)
   - Anchor Heads (12..15): alpha_bias = +6.9 (tau ≈ 692.80 токенів)
   - Beta Write Gate: beta_bias = -1.0 (beta_0 ≈ 0.2689)
5. Механізм State Snapshot & Restore:
   - Миттєве збереження імутабельних знімків рекурентного стану.
   - Повний відкат повідомлень чату (Chat Rollback) та спекулятивне розгалуження
     без залишкового дрейфу чи забруднення пам'яті.
6. Конвенція Post-Update: y_t = S_t @ q_t (ортогональне самозвернення).
================================================================================
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union, Any
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. КОНФІГУРАЦІЯ ТА АУДИТ ПАМ'ЯТІ (CONFIGURATION & AUDIT)
# ============================================================================

@dataclass
class DeltaNetConfig:
    """
    Конфігураційні параметри рекурентного стека Gated DeltaNet для Hetman-2-NCM 2.0B.
    """
    hidden_size: int = 1536
    num_attention_heads: int = 16
    deltanet_head_dim: int = 64
    deltanet_conv_kernel: int = 4
    num_deltanet_layers: int = 28
    rms_norm_eps: float = 1e-6
    state_dtype: torch.dtype = torch.float32  # Суворий інваріант: ТІЛЬКИ FP32!
    io_dtype: torch.dtype = torch.float32     # Вхідний/вихідний тип (float32 або bfloat16)
    use_bitlinear: bool = False               # Чи використовувати тернарний BitLinear b1.58
    use_hadamard_rotation: bool = False       # Чи використовувати FWHT обертання для викидів
    hadamard_block_size: int = 512

    @property
    def total_dim(self) -> int:
        """Розмірність пляшкового горла: 16 * 64 = 1024."""
        return self.num_attention_heads * self.deltanet_head_dim

    @property
    def single_layer_s_elements(self) -> int:
        """Кількість елементів матриці S в одному шарі: 16 * 64 * 64 = 65,536."""
        return self.num_attention_heads * self.deltanet_head_dim * self.deltanet_head_dim

    @property
    def single_layer_s_bytes(self) -> int:
        """Обсяг пам'яті матриці S одного шару у FP32: 65,536 * 4 = 262,144 байти (256 KiB)."""
        return self.single_layer_s_elements * 4

    @property
    def total_s_elements(self) -> int:
        """Сумарна кількість елементів матриць S для 28 шарів: 28 * 65,536 = 1,835,008."""
        return self.num_deltanet_layers * self.single_layer_s_elements

    @property
    def total_s_bytes(self) -> int:
        """
        Сумарний обсяг пам'яті матриць S 28 шарів у FP32:
        28 * 262,144 = 7,340,032 байти (РІВНО 7.000000 MiB).
        """
        return self.total_s_elements * 4

    @property
    def total_s_mib(self) -> float:
        """Сумарний обсяг у двійкових MiB: 7,340,032 / 1,048,576 = 7.000000 MiB."""
        return self.total_s_bytes / (1024.0 * 1024.0)

    @property
    def conv_buffer_elements_per_layer(self) -> int:
        """Кількість елементів локального буфера згортки на шар: 1536 * (4 - 1) = 4608."""
        return self.hidden_size * (self.deltanet_conv_kernel - 1)

# Аліас конфігурації для зворотної сумісності зі специфікацією
DeltaNetBridgeConfig = DeltaNetConfig


# ============================================================================
# 2. СТРУКТУРИ ДАНИХ СТАНУ (RECURRENT STATE STRUCTURES)
# ============================================================================

@dataclass
class LayerRecurrentState:
    """
    Ізольований рекурентний стан для одного шару Gated DeltaNet.
    
    Компоненти:
    - S: Рекурентна матриця асоціативної пам'яті [B, num_heads, head_dim, head_dim]
         СТРОГО у форматі FP32 (256.000 KiB при B=1).
    - conv_state: Буфер контексту каузальної 1D-згортки [B, hidden_size, kernel_size - 1]
         (9 KiB при BF16/FP16 або 18 KiB при FP32).
    """
    S: torch.Tensor
    conv_state: torch.Tensor

    def __post_init__(self):
        # Перевірка суворого інваріанта FP32 для матриці стану S
        if self.S.dtype != torch.float32:
            raise TypeError(
                f"Критичне порушення інваріанта Hetman-2-NCM! "
                f"Матриця стану S повинна мати dtype=torch.float32, отримано: {self.S.dtype}. "
                f"Низька розрядність (FP16/BF16) викликає катастрофічний дрейф пам'яті."
            )

    def detach_clone(self) -> "LayerRecurrentState":
        """Створює глибоку копію стану без прив'язки до графу обчислень (Zero-Drift)."""
        return LayerRecurrentState(
            S=self.S.detach().clone(),
            conv_state=self.conv_state.detach().clone()
        )

    def s_memory_bytes(self) -> int:
        """Обсяг пам'яті матриці S у байтах."""
        return self.S.numel() * self.S.element_size()

    def s_memory_mib(self) -> float:
        """Обсяг пам'яті матриці S у MiB."""
        return self.s_memory_bytes() / (1024.0 * 1024.0)

    def total_memory_bytes(self) -> int:
        """Сумарний обсяг пам'яті шару (S + conv_state)."""
        conv_bytes = self.conv_state.numel() * self.conv_state.element_size()
        return self.s_memory_bytes() + conv_bytes

    def frobenius_norm(self) -> float:
        """Норма Фробеніуса матриці S для перевірки спектральної стійкості."""
        return torch.linalg.norm(self.S).item()

    def zero_(self) -> "LayerRecurrentState":
        """Обнулення стану на місці."""
        self.S.zero_()
        self.conv_state.zero_()
        return self

    # Дозволяє розпаковувати об'єкт як кортеж (conv_state, S) для зворотної сумісності з TH1
    def __iter__(self):
        return iter((self.conv_state, self.S))

    def __getitem__(self, idx: int):
        if idx == 0:
            return self.conv_state
        elif idx == 1:
            return self.S
        raise IndexError("LayerRecurrentState підтримує лише індекси 0 (conv_state) та 1 (S).")


@dataclass
class DeltaNetStateBundle:
    """
    Глобальний пакет рекурентного стану для всіх 28 шарів Gated DeltaNet
    між послідовними блоками NCM або кроками діалогової сесії.
    """
    block_index: int = 0
    global_token_offset: int = 0
    layer_states: Dict[int, LayerRecurrentState] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def detach_clone(self) -> "DeltaNetStateBundle":
        """Створює повністю ізольований імутабельний знімок усього стану моделі."""
        cloned_layers = {
            idx: st.detach_clone() for idx, st in self.layer_states.items()
        }
        return DeltaNetStateBundle(
            block_index=self.block_index,
            global_token_offset=self.global_token_offset,
            layer_states=cloned_layers,
            metadata=copy.deepcopy(self.metadata)
        )

    def total_s_memory_bytes(self) -> int:
        """Сумарні байти виключно матриць S по всіх 28 шарах."""
        return sum(st.s_memory_bytes() for st in self.layer_states.values())

    def total_s_memory_mib(self) -> float:
        """Сумарний обсяг матриць S у MiB (Рівно 7.000000 MiB при 28 шарах, B=1)."""
        return self.total_s_memory_bytes() / (1024.0 * 1024.0)

    def total_memory_bytes(self) -> int:
        """Повний обсяг пам'яті пакета стану (S + буфери згортки)."""
        return sum(st.total_memory_bytes() for st in self.layer_states.values())

    def total_memory_mib(self) -> float:
        """Повний обсяг пам'яті пакета стану в MiB."""
        return self.total_memory_bytes() / (1024.0 * 1024.0)

    def to(self, device: torch.device) -> "DeltaNetStateBundle":
        """Переміщує всі тензори стану на цільовий пристрій."""
        for st in self.layer_states.values():
            st.S = st.S.to(device=device)
            st.conv_state = st.conv_state.to(device=device)
        return self

# Аліас для зворотної сумісності зі специфікацією
BlockCausalStateBundle = DeltaNetStateBundle


# ============================================================================
# 3. МЕНЕДЖЕР ЗНІМКІВ СТАНУ ТА ВІДКАТУ ЧАТУ (STATE SNAPSHOT & RESTORE)
# ============================================================================

class StateSnapshotManager:
    """
    Менеджер миттєвих знімків рекурентного стану моделі (State Snapshots & Restore).
    
    Призначення:
    1. Відкат повідомлень чату (Chat Turn Rollback): коли користувач редагує або видаляє
       попереднє повідомлення, стан рекурентної пам'яті миттєво повертається до точки
       перед цим повідомленням, гарантуючи нульовий дрейф і відсутність "галюцинацій з минулого".
    2. Спекулятивне декодування та розгалуження діалогу (Branching / Tree-of-Thought):
       дослідження різних варіантів відповідей з одного кореневого стану.
    3. Контрольні точки NCM (NCM Block Checkpoints) для надійного відновлення після збоїв.
    """
    def __init__(self):
        self._snapshots: Dict[str, DeltaNetStateBundle] = {}
        self._chat_history: List[Dict[str, Any]] = []

    def save_snapshot(
        self,
        name: str,
        bundle: DeltaNetStateBundle,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Зберігає імутабельний знімок стану за унікальним іменем."""
        cloned = bundle.detach_clone()
        if metadata:
            cloned.metadata.update(metadata)
        self._snapshots[name] = cloned

    def restore_snapshot(self, name: str) -> DeltaNetStateBundle:
        """
        Відновлює збережений знімок стану. Повертає ізольовану копію,
        щоб подальші обчислення не пошкодили збережений знімок.
        """
        if name not in self._snapshots:
            available = list(self._snapshots.keys())
            raise KeyError(
                f"Знімок стану з іменем '{name}' не знайдено! "
                f"Наявні знімки: {available}"
            )
        return self._snapshots[name].detach_clone()

    def has_snapshot(self, name: str) -> bool:
        """Перевіряє наявність знімка."""
        return name in self._snapshots

    def drop_snapshot(self, name: str) -> bool:
        """Видаляє знімок для негайного звільнення пам'яті."""
        if name in self._snapshots:
            del self._snapshots[name]
            return True
        return False

    def list_snapshots(self) -> List[str]:
        """Повертає список імен усіх активних знімків."""
        return list(self._snapshots.keys())

    def clear(self) -> None:
        """Очищує всі збережені знімки."""
        self._snapshots.clear()
        self._chat_history.clear()

    # ------------------------------------------------------------------------
    # Спеціалізований інтерфейс для діалогового чату (Chat History Rollback)
    # ------------------------------------------------------------------------
    def save_chat_turn(
        self,
        turn_id: int,
        bundle: DeltaNetStateBundle,
        user_prompt: Optional[str] = None,
        assistant_response: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Фіксує стан після завершення чергового раунду (turn) діалогу.
        Створює знімок `turn_{turn_id}` та записує метадані.
        """
        snapshot_name = f"turn_{turn_id}"
        meta = {
            "turn_id": turn_id,
            "user_prompt": user_prompt,
            "assistant_response": assistant_response,
            "global_token_offset": bundle.global_token_offset,
            "block_index": bundle.block_index
        }
        if extra_info:
            meta.update(extra_info)

        self.save_snapshot(snapshot_name, bundle, metadata=meta)
        self._chat_history.append(meta)

    def rollback_to_chat_turn(self, turn_id: int) -> DeltaNetStateBundle:
        """
        Виконує відкат сесії до вказаного раунду діалогу turn_id.
        Всі знімки раундів > turn_id автоматично видаляються,
        забезпечуючи ідеальну чистоту пам'яті для нової гілки розмови.
        """
        snapshot_name = f"turn_{turn_id}"
        restored = self.restore_snapshot(snapshot_name)

        # Видаляємо всі пізніші знімки та записи історії (Pruning subsequent branches)
        to_delete = [
            name for name in self._snapshots.keys()
            if name.startswith("turn_") and int(name.split("_")[1]) > turn_id
        ]
        for name in to_delete:
            del self._snapshots[name]

        self._chat_history = [
            rec for rec in self._chat_history if rec.get("turn_id", 0) <= turn_id
        ]

        return restored

    def get_chat_history(self) -> List[Dict[str, Any]]:
        """Повертає метадані історії збережених раундів чату."""
        return list(self._chat_history)

    # ------------------------------------------------------------------------
    # Серіалізація стану для диска або мережевого протоколу Rada.cpp
    # ------------------------------------------------------------------------
    def export_state_dict(self, bundle: DeltaNetStateBundle) -> Dict[str, Any]:
        """Експортує пакет стану в серіалізований CPU-словник."""
        return {
            "block_index": bundle.block_index,
            "global_token_offset": bundle.global_token_offset,
            "metadata": bundle.metadata,
            "layers": {
                idx: {
                    "S": st.S.detach().cpu(),
                    "conv_state": st.conv_state.detach().cpu()
                }
                for idx, st in bundle.layer_states.items()
            }
        }

    def load_state_dict(
        self,
        data: Dict[str, Any],
        device: torch.device
    ) -> DeltaNetStateBundle:
        """Відновлює пакет стану із серіалізованого словника."""
        bundle = DeltaNetStateBundle(
            block_index=data.get("block_index", 0),
            global_token_offset=data.get("global_token_offset", 0),
            metadata=data.get("metadata", {})
        )
        for idx_key, layer_dict in data.get("layers", {}).items():
            idx = int(idx_key)
            bundle.layer_states[idx] = LayerRecurrentState(
                S=layer_dict["S"].to(device=device, dtype=torch.float32),
                conv_state=layer_dict["conv_state"].to(device=device)
            )
        return bundle


# ============================================================================
# 4. КВАНТОВАНІ ТЕРНАРНІ ШАРИ BITLINEAR (BITNET B1.58 INTEGRATION)
# ============================================================================

class BitLinear(nn.Module):
    """
    BitNet b1.58 Linear Layer з прямим оцінювачем Straight-Through Estimator (STE).
    Забезпечує тернарне зважування {-1, 0, +1} для проєкцій пляшкового горла.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        self.rms_norm = nn.RMSNorm(in_features, eps=1e-6)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def quantize_weights(self, w: torch.Tensor) -> torch.Tensor:
        gamma = torch.mean(torch.abs(w)).clamp(min=1e-6)
        w_scaled = w / gamma
        w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        return w + (w_quant * gamma - w).detach()

    def quantize_activations(self, x: torch.Tensor) -> torch.Tensor:
        eta = torch.max(torch.abs(x), dim=-1, keepdim=True).values.clamp(min=1e-6)
        scale = 127.0 / eta
        x_quant = torch.clamp(torch.round(x * scale), -128.0, 127.0)
        return x + (x_quant / scale - x).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.rms_norm(x)
        x_q = self.quantize_activations(x_norm)
        w_q = self.quantize_weights(self.weight)
        return F.linear(x_q, w_q, self.bias)


# ============================================================================
# 5. ШАР GATED DELTANET З ПЛЯШКОВИМ ГОРЛОМ (CORE RECURRENT LAYER)
# ============================================================================

class GatedDeltaNetLayer(nn.Module):
    """
    Виробничий шар лінійної рекурентної уваги Gated DeltaNet (Frontier Edition).
    
    Архітектура:
    - Локальна каузальна згортка 1D (k=4, groups=1536) з неперервним збереженням стану.
    - Просторове пляшкове горло: 1536 -> 1024 -> 1536.
    - 16 голів розмірністю d_k = d_v = 64.
    - Мультимасштабне затухання каналів: Fast (0..3), Episodic (4..11), Anchor (12..15).
    - Матриця асоціативного стану S_t in R^{16 x 64 x 64} СТРОГО у FP32.
    - Повна неперервність міжблокової передачі стану без витоків.
    """
    def __init__(self, config: Optional[Union[DeltaNetConfig, Any]] = None, layer_idx: int = 0):
        super().__init__()
        if config is None:
            config = DeltaNetConfig()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = getattr(config, "hidden_size", 1536)
        self.num_heads = getattr(config, "num_attention_heads", 16)
        self.head_dim = getattr(config, "deltanet_head_dim", 64)
        self.total_dim = self.num_heads * self.head_dim  # 16 * 64 = 1024
        self.conv_kernel = getattr(config, "deltanet_conv_kernel", 4)
        self.rms_norm_eps = getattr(config, "rms_norm_eps", 1e-6)
        self.use_bitlinear = getattr(config, "use_bitlinear", False)

        # --------------------------------------------------------------------
        # 1. Локальна каузальна 1D згортка (Depthwise Conv1D)
        # --------------------------------------------------------------------
        self.conv1d = nn.Conv1d(
            in_channels=self.hidden_size,
            out_channels=self.hidden_size,
            kernel_size=self.conv_kernel,
            groups=self.hidden_size,
            bias=True,
            padding=0
        )

        # --------------------------------------------------------------------
        # 2. Лінійні проєкції пляшкового горла: 1536 -> 1024
        # --------------------------------------------------------------------
        LinearClass = BitLinear if self.use_bitlinear else nn.Linear
        self.q_proj = LinearClass(self.hidden_size, self.total_dim, bias=False)
        self.k_proj = LinearClass(self.hidden_size, self.total_dim, bias=False)
        self.v_proj = LinearClass(self.hidden_size, self.total_dim, bias=False)

        # --------------------------------------------------------------------
        # 3. Динамічні гейти затухання (alpha) та навчання (beta): 1536 -> 16
        # --------------------------------------------------------------------
        self.alpha_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)
        self.beta_proj = nn.Linear(self.hidden_size, self.num_heads, bias=True)

        # --------------------------------------------------------------------
        # 4. Вихідне горло та нормалізація: 1024 -> 1536 + Swish Gate
        # --------------------------------------------------------------------
        self.out_proj = LinearClass(self.total_dim, self.hidden_size, bias=False)
        self.gate_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.norm = nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)
        self.v_norm = nn.RMSNorm(self.head_dim, eps=self.rms_norm_eps)

        # --------------------------------------------------------------------
        # 5. Мультимасштабна ініціалізація канального затухання
        # --------------------------------------------------------------------
        self._init_multiscale_decay_channels()

    def _init_multiscale_decay_channels(self):
        """
        Ініціалізує спектр затухання альфа-голів відповідно до специфікації Hetman-2-NCM:
        - Fast Heads (голова 0..3): bias = +1.0 (sigma ≈ 0.731, tau ≈ 2.21 токенів)
        - Episodic Heads (голова 4..11): bias = +3.5 (sigma ≈ 0.971, tau ≈ 23.31 токенів)
        - Anchor Heads (голова 12..15): bias = +6.9 (sigma ≈ 0.999, tau ≈ 692.80 токенів)
        - Beta Bias: -1.0 (sigma ≈ 0.269)
        """
        with torch.no_grad():
            n_fast = max(1, self.num_heads // 4)
            n_anchor = max(1, self.num_heads // 4)
            n_episodic = max(0, self.num_heads - n_fast - n_anchor)

            alpha_init = torch.tensor(
                [1.0] * n_fast + [3.5] * n_episodic + [6.9] * n_anchor,
                dtype=torch.float32
            )
            self.alpha_proj.bias.copy_(alpha_init)
            nn.init.constant_(self.beta_proj.bias, -1.0)

    def init_zero_state(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> LayerRecurrentState:
        """Створює чистий нульовий стан для початку нової сесії."""
        if device is None:
            device = next(self.parameters()).device

        S = torch.zeros(
            batch_size, self.num_heads, self.head_dim, self.head_dim,
            dtype=torch.float32, device=device
        )
        conv_state = torch.zeros(
            batch_size, self.hidden_size, self.conv_kernel - 1,
            dtype=next(self.conv1d.parameters()).dtype, device=device
        )
        return LayerRecurrentState(S=S, conv_state=conv_state)

    def forward(
        self,
        x: torch.Tensor,
        prev_state: Optional[Union[LayerRecurrentState, Tuple[Optional[torch.Tensor], torch.Tensor], torch.Tensor]] = None,
        state_reset_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, LayerRecurrentState]:
        """
        Прямий прохід токенів крізь шар Gated DeltaNet.
        
        Вхід:
            x: [B, L, hidden_size] (hidden_size = 1536)
            prev_state: LayerRecurrentState або (conv_state, S) або S або None
            state_reset_mask: [B, L] або None (прапорець скидання стану для нового документа)
            
        Повертає:
            y: [B, L, hidden_size]
            next_state: LayerRecurrentState з оновленими матрицею S та буфером згортки
        """
        B, L, D = x.shape
        device = x.device
        dtype = x.dtype

        # --------------------------------------------------------------------
        # 0. Розпаковка та приведення вхідного стану
        # --------------------------------------------------------------------
        conv_state = None
        S = None
        if prev_state is not None:
            if isinstance(prev_state, LayerRecurrentState):
                conv_state = prev_state.conv_state
                S = prev_state.S
            elif isinstance(prev_state, tuple):
                conv_state, S = prev_state
            elif isinstance(prev_state, torch.Tensor):
                S = prev_state

        if conv_state is None:
            conv_state = torch.zeros(
                B, self.hidden_size, self.conv_kernel - 1,
                dtype=dtype, device=device
            )
        else:
            conv_state = conv_state.to(device=device, dtype=dtype)

        if S is None:
            S = torch.zeros(
                B, self.num_heads, self.head_dim, self.head_dim,
                dtype=torch.float32, device=device
            )
        else:
            # Суворий інваріант: переведення в FP32 для усунення числового дрейфу
            S = S.to(device=device, dtype=torch.float32)

        # Обробити випадок нульової довжини
        if L == 0:
            empty_out = torch.empty(B, 0, D, device=device, dtype=dtype)
            return empty_out, LayerRecurrentState(S=S, conv_state=conv_state)

        k_sz = self.conv_kernel

        # --------------------------------------------------------------------
        # 1. Локальна каузальна 1D згортка з неперервним буфером (k=4)
        # --------------------------------------------------------------------
        x_trans = x.transpose(1, 2)  # [B, D, L]
        # Безшовне доповнення історії: [B, D, (k-1) + L]
        x_padded = torch.cat([conv_state, x_trans], dim=-1)
        next_conv_state = x_padded[:, :, -(k_sz - 1):].detach()

        x_conv = F.silu(self.conv1d(x_padded)).transpose(1, 2)  # [B, L, D]
        x_eff = x + x_conv  # Residual зв'язок локальної згортки

        # --------------------------------------------------------------------
        # 2. Проєкції пляшкового горла: 1536 -> 1024 -> [B, L, 16, 64]
        # --------------------------------------------------------------------
        q = self.q_proj(x_eff).view(B, L, self.num_heads, self.head_dim)
        k = self.k_proj(x_eff).view(B, L, self.num_heads, self.head_dim)
        v = self.v_norm(self.v_proj(x_eff).view(B, L, self.num_heads, self.head_dim))

        # Сферична L2-нормалізація Q та K у FP32 для гарантії ||k_t|| = 1 (спектральна стійкість)
        q_fp32 = F.normalize(q.float(), p=2, dim=-1, eps=1e-5)
        k_fp32 = F.normalize(k.float(), p=2, dim=-1, eps=1e-5)
        v_fp32 = v.float()

        # Розрахунок динамічних гейтів затухання alpha та запису beta
        alpha_fp32 = torch.sigmoid(self.alpha_proj(x_eff)).float()  # [B, L, 16]
        beta_fp32 = torch.sigmoid(self.beta_proj(x_eff)).float()    # [B, L, 16]

        # --------------------------------------------------------------------
        # 3. Рекурентна динаміка Дельта-правила (Strict FP32 Accumulation)
        # --------------------------------------------------------------------
        outputs = []
        for t in range(L):
            if state_reset_mask is not None:
                reset_t = state_reset_mask[:, t, None, None, None].float()
                S = S * reset_t

            q_t = q_fp32[:, t]                     # [B, H, 64]
            k_t = k_fp32[:, t]                     # [B, H, 64]
            v_t = v_fp32[:, t]                     # [B, H, 64]
            a_t = alpha_fp32[:, t, :, None, None]  # [B, H, 1, 1]
            b_t = beta_fp32[:, t, :, None, None]   # [B, H, 1, 1]

            # 1. Передбачення поточного збереженого образу: v_pred = S_{t-1} @ k_t
            # S: [B, H, 64 (v), 64 (k)], k_t: [B, H, 64 (k)] -> v_pred: [B, H, 64 (v)]
            v_pred = torch.einsum('bhvk,bhk->bhv', S, k_t)

            # 2. Помилка дельта-правила: v_err = v_t - v_pred
            v_err = v_t - v_pred

            # 3. Ранг-1 ортогональне оновлення пам'яті: update = v_err (x) k_t^T
            update = torch.einsum('bhv,bhk->bhvk', v_err, k_t)

            # 4. Оновлення рекурентного стану: S_t = S_{t-1} * alpha_t + beta_t * update
            S = S * a_t + b_t * update

            # 5. Зчитування результату за конвенцією Post-Update: y_t = S_t @ q_t
            y_t = torch.einsum('bhvk,bhk->bhv', S, q_t)
            outputs.append(y_t.to(dtype))

        # --------------------------------------------------------------------
        # 4. Вихідне пляшкове горло: 1024 -> 1536 + Swish Channel Gate
        # --------------------------------------------------------------------
        out_stacked = torch.stack(outputs, dim=1)  # [B, L, 16, 64]
        out_normed = self.norm(out_stacked).view(B, L, self.total_dim)  # [B, L, 1024]

        gate = F.silu(self.gate_proj(x_eff))  # [B, L, 1536]
        y = self.out_proj(out_normed) * gate  # [B, L, 1536]

        next_state = LayerRecurrentState(S=S, conv_state=next_conv_state)
        return y, next_state

# Аліас для зворотної сумісності з TH1
GatedDeltaNet = GatedDeltaNetLayer


# ============================================================================
# 6. МІСТ БЛОКОВО-КАУЗАЛЬНОЇ НЕПЕРЕРВНОСТІ (BLOCK-CAUSAL BRIDGE)
# ============================================================================

class GatedDeltaNetBlockCausalBridge(nn.Module):
    """
    Промисловий міст блоково-каузальної пам'яті для Hetman-2-NCM 2.0B.
    
    Керує повним рекурентним стеком з 28 шарів Gated DeltaNet:
    1. O(1) Memory Footprint: рівно 7.000000 MiB матричного стану S (FP32).
    2. Повна підтримка механізму State Snapshot & Restore для відкату чату.
    3. Відсікання градієнтів Truncated BPTT між блоками NCM.
    4. Аудит пам'яті та верифікація чисельної стійкості без витоків.
    """
    def __init__(self, config: Optional[Union[DeltaNetConfig, Any]] = None):
        super().__init__()
        if config is None:
            config = DeltaNetConfig()
        self.config = config
        self.num_layers = getattr(config, "num_deltanet_layers", 28)

        # 28 лінійних рекурентних шарів DeltaNet
        self.layers = nn.ModuleList([
            GatedDeltaNetLayer(self.config, layer_idx=i) for i in range(self.num_layers)
        ])

        # Репозиторій іменованих знімків стану
        self.snapshot_manager = StateSnapshotManager()

    def init_clean_bundle(
        self,
        batch_size: int = 1,
        device: Optional[torch.device] = None
    ) -> DeltaNetStateBundle:
        """Створює чистий пакет стану для початку сесії генерації."""
        if device is None:
            device = next(self.parameters()).device

        bundle = DeltaNetStateBundle(block_index=0, global_token_offset=0)
        for idx, layer in enumerate(self.layers):
            bundle.layer_states[idx] = layer.init_zero_state(batch_size, device)
        return bundle

    # ------------------------------------------------------------------------
    # Делегування управління знімками стану
    # ------------------------------------------------------------------------
    def create_snapshot(
        self,
        name: str,
        bundle: Optional[DeltaNetStateBundle] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Зберігає знімок стану моделі."""
        if bundle is None:
            raise ValueError("Необхідно надати state_bundle для створення знімка.")
        self.snapshot_manager.save_snapshot(name, bundle, metadata)

    def restore_snapshot(self, name: str) -> DeltaNetStateBundle:
        """Відновлює збережений знімок стану моделі."""
        return self.snapshot_manager.restore_snapshot(name)

    def drop_snapshot(self, name: str) -> bool:
        """Видаляє знімок."""
        return self.snapshot_manager.drop_snapshot(name)

    def list_snapshots(self) -> List[str]:
        """Повертає список наявних знімків."""
        return self.snapshot_manager.list_snapshots()

    def save_chat_turn(
        self,
        turn_id: int,
        bundle: DeltaNetStateBundle,
        user_prompt: Optional[str] = None,
        assistant_response: Optional[str] = None,
        extra_info: Optional[Dict[str, Any]] = None
    ) -> None:
        """Фіксує раунд чату для подальшого швидкого відкату."""
        self.snapshot_manager.save_chat_turn(
            turn_id=turn_id,
            bundle=bundle,
            user_prompt=user_prompt,
            assistant_response=assistant_response,
            extra_info=extra_info
        )

    def rollback_to_chat_turn(self, turn_id: int) -> DeltaNetStateBundle:
        """Здійснює безпечний відкат до вказаного раунду діалогу."""
        return self.snapshot_manager.rollback_to_chat_turn(turn_id)

    def export_state_dict(self, bundle: DeltaNetStateBundle) -> Dict[str, Any]:
        """Експортує стан у словник."""
        return self.snapshot_manager.export_state_dict(bundle)

    def load_state_dict(
        self,
        data: Dict[str, Any],
        device: Optional[torch.device] = None
    ) -> DeltaNetStateBundle:
        """Завантажує стан зі словника."""
        if device is None:
            device = next(self.parameters()).device
        return self.snapshot_manager.load_state_dict(data, device)

    # ------------------------------------------------------------------------
    # Прямий прохід через міст
    # ------------------------------------------------------------------------
    def forward_block(
        self,
        hidden_states: torch.Tensor,
        state_bundle: Optional[DeltaNetStateBundle] = None,
        detach_state_transfer: bool = False,
        state_reset_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, DeltaNetStateBundle]:
        """
        Послідовний прохід блоку токенів NCM крізь усі 28 шарів Gated DeltaNet.
        
        Вхід:
            hidden_states: [B, L_block, hidden_size]
            state_bundle: Поточний стан моста (якщо None — створюється чистий)
            detach_state_transfer: Якщо True — градієнти BPTT між блоками відсікаються.
            state_reset_mask: [B, L_block] або None
            
        Повертає:
            hidden_states: [B, L_block, hidden_size] з урахуванням residual skip
            next_state_bundle: Оновлений пакет стану для наступного блоку NCM.
        """
        B, L, D = hidden_states.shape
        device = hidden_states.device

        if state_bundle is None:
            state_bundle = self.init_clean_bundle(B, device)

        new_bundle = DeltaNetStateBundle(
            block_index=state_bundle.block_index + 1,
            global_token_offset=state_bundle.global_token_offset + L,
            metadata=copy.deepcopy(state_bundle.metadata)
        )

        h = hidden_states
        for idx, layer in enumerate(self.layers):
            layer_state = state_bundle.layer_states.get(idx, None)
            h_out, next_l_state = layer(
                h,
                prev_state=layer_state,
                state_reset_mask=state_reset_mask
            )

            # Residual зв'язок між шарами трансформера
            h = h + h_out

            if detach_state_transfer:
                next_l_state = next_l_state.detach_clone()

            new_bundle.layer_states[idx] = next_l_state

        return h, new_bundle

    # ------------------------------------------------------------------------
    # Аудит пам'яті (Memory Audit API)
    # ------------------------------------------------------------------------
    def audit_memory(self, batch_size: int = 1) -> Dict[str, Any]:
        """
        Виконує строгий побайтний аудит рекурентного стану.
        Підтверджує дотримання інваріанта рівно 7.000000 MiB при batch_size = 1.
        """
        device = next(self.parameters()).device
        bundle = self.init_clean_bundle(batch_size=batch_size, device=device)

        s_bytes = bundle.total_s_memory_bytes()
        s_mib = bundle.total_s_memory_mib()
        total_bytes = bundle.total_memory_bytes()
        total_mib = bundle.total_memory_mib()

        # Розрахунок теоретичного значення
        expected_s_bytes = batch_size * self.num_layers * 16 * 64 * 64 * 4
        expected_s_mib = expected_s_bytes / (1024.0 * 1024.0)

        is_exact_7mib = (batch_size == 1 and s_bytes == 7340032 and math.isclose(s_mib, 7.0, abs_tol=1e-6))

        return {
            "num_layers": self.num_layers,
            "batch_size": batch_size,
            "s_matrix_bytes": s_bytes,
            "s_matrix_mib": s_mib,
            "expected_s_bytes": expected_s_bytes,
            "expected_s_mib": expected_s_mib,
            "total_state_bytes": total_bytes,
            "total_state_mib": total_mib,
            "is_exact_7mib": is_exact_7mib,
            "per_layer_s_kib": 256.0 * batch_size
        }

# Аліаси для гнучкості імпорту
DeltaNetBlockCausalBridge = GatedDeltaNetBlockCausalBridge
DeltaNetStack = GatedDeltaNetBlockCausalBridge


# ============================================================================
# 7. ДОПОМІЖНІ ФУНКЦІЇ АУДИТУ (STANDALONE HELPER FUNCTIONS)
# ============================================================================

def compute_deltanet_memory_footprint(
    num_layers: int = 28,
    batch_size: int = 1,
    num_heads: int = 16,
    head_dim: int = 64,
    hidden_size: int = 1536,
    conv_kernel: int = 4,
    conv_dtype_bytes: int = 4
) -> Dict[str, Union[int, float]]:
    """
    Аналітичний розрахунок пам'яті рекурентного стану DeltaNet.
    """
    s_elements_per_layer = num_heads * head_dim * head_dim
    s_bytes_per_layer = s_elements_per_layer * 4  # FP32
    total_s_bytes = batch_size * num_layers * s_bytes_per_layer
    total_s_mib = total_s_bytes / (1024.0 * 1024.0)

    conv_elements_per_layer = hidden_size * (conv_kernel - 1)
    conv_bytes_per_layer = conv_elements_per_layer * conv_dtype_bytes
    total_conv_bytes = batch_size * num_layers * conv_bytes_per_layer
    total_conv_mib = total_conv_bytes / (1024.0 * 1024.0)

    grand_total_bytes = total_s_bytes + total_conv_bytes
    grand_total_mib = grand_total_bytes / (1024.0 * 1024.0)

    return {
        "s_bytes_per_layer": s_bytes_per_layer,
        "total_s_bytes": total_s_bytes,
        "total_s_mib": total_s_mib,
        "total_conv_bytes": total_conv_bytes,
        "total_conv_mib": total_conv_mib,
        "grand_total_bytes": grand_total_bytes,
        "grand_total_mib": grand_total_mib
    }
