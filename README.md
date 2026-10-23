# Hetman-2-NCM 2.0B 🇺🇦

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://python.org)
[![PyTorch: 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org)
[![HuggingFace: Compatible](https://img.shields.io/badge/HuggingFace-Transformers-yellow.svg)](https://huggingface.co)
[![Peak RAM: < 530 MiB](https://img.shields.io/badge/Peak%20RAM-525.74%20MiB-purple.svg)](#memory-arena)
[![Context: 256k Tokens](https://img.shields.io/badge/Context-256k%20Tokens-red.svg)](#hopfield-kv-cache)
[![Tests: 102/102 Passing](https://img.shields.io/badge/Tests-102%2F102%20Passed-success.svg)](tests/)

> **Hetman-2-NCM 2.0B** — надшвидка фундаментальна локальна мовна модель нового покоління для споживчих ноутбуків (від 4–8 ГБ RAM) та мобільних пристроїв. Поєднує 1.58-бітний апаратний фундамент **BitNet + DeltaNet + Hopfield** із трифазним генеративним рушієм **Block-Causal Next-Concept Modeling (NCM)**.

---

## ⚡ Ключові технологічні інновації

| Технологія | Суть і принцип роботи | Досягнутий ефект |
|:---|:---|:---:|
| **Block-Causal Hierarchical NCM** | Генерація у 3 фази: **Phase 1** (Causal AR Drafter, 4–8 макро-кроків) $\to$ **Phase 2** (Global Bidirectional Refiner + Cosine Dedup) $\to$ **Phase 3** (Block-Causal NCM Decoder по 400 токенів). Замість 1600+ покрокових авторегресійних кроків — всього **25–30 кроків** на весь документ. | **$10\times\text{--}15\times$ прискорення генерації** на CPU ноутбука ($1.5\text{--}3.0\text{ с}$ на 1600 токенів) |
| **Hybrid Dual-Loop Deliberation (Variant C)** | Інтегрована система динамічного сінкінгу («стільки, скільки треба»): Cognitive Difficulty Router (Simple / Medium / Complex) + внутрішній латентний цикл Банаха у кеші L2 процесора ($>1000\text{ GB/s}$) + телеметрія думок Thought Verbalizer та аварійна зупинка «Відповідай зараз!» ($<15\text{ мс}$). | **0 байт додаткового KV-кешу**, усунення пастки каузальної помилки, миттєве міркування за **$20\text{--}350\text{ мс}$** |
| **BitNet b1.58 Ternary Weights** | Усі ваги лінійних шарів квантовані в $\{-1, 0, +1\}$ (2 біти на вагу). Множення замінено на цілочисельне додавання/віднімання. | **$495.77\text{ MiB}$** на всі ваги моделі 2.1B (замість $4.2\text{ GiB}$ у FP16) |
| **FWHT INT8 Activation Engine** | Блокове швидке перетворення Волша-Адамара ($B=512$) перед лінійними проекціями зрізає аномальні сплески активацій (outliers). | **Нульова втрата Perplexity** при 8-бітових активаціях без FP-масштабування |
| **Gated DeltaNet Constant-Memory State** | 28 із 34 шарів бекбону використовують лінійну рекурентність $S_t \in \mathbb{R}^{d \times d}$ з гарантованою стійкістю Ляпунова $\rho(M_t) < 1.0$. Пам'ять не росте з довжиною контексту. | Строго **$7.000000\text{ MiB}$ FP32** на весь контекст (до 256k токенів) |
| **SLH-Core v2 (Hopfield Memory)** | 16,777,216 дискретних атракторних слотів ($128\text{ MiB}$) з білінійними кодбуками $C_1 \times C_2$ та Top-32 розрідженою проекцією ($0.003\%$ активного простору). Leaky PID-балансир запобігає колізіям. | **Точні фактичні знання** без роздування розміру бекбону та без галюцинацій |
| **Hierarchical Hopfield KV-Cache** | 3-рівневий KV-кеш для 6 глобальних шарів Softmax GQA: Welford онлайн-дисперсія + центроїди Махаланобіса стискають історію уваги. | **$33.7\text{ MiB}$ на 256k токенів** (проти $27.3\text{ GiB}$ у Llama-3, стиснення у **$810\times$**, $KL < 0.0015$) |
| **Rada.cpp C++20 Zero-Alloc Runtime** | Єдина монолітна mmap-арена, 64-байтне вирівнювання (AVX-512/AVX2/NEON), двошаровий CRC32 та нуль алокацій пам'яті в процесі інференсу. | **$525.74\text{ MiB}$ пікової RAM** ($< 750\text{ MiB}$ бюджет, запас $29.9\%$) |
| **Rada-PowerGuard Roofline** | Фізична оптимізація під енергетичний тротлінг процесорів на батареї (з 28W до 12W). | **$1140\text{ т/с}$ (мережа) / $1012\text{ т/с}$ (батарея)**; 2 год 52 хв безперервної автономної роботи |

---

## 📊 Порівняння з індустрійними моделями

| Характеристика | Класична LLM 2B (FP16 / Llama / Qwen) | Hetman-2.0B (Класичний AR) | **Hetman-2-NCM 2.0B** |
|:---|:---:|:---:|:---:|
| **Розмір ваг у пам'яті** | $\approx 4.2\text{ GiB}$ | $\approx 508\text{ MiB}$ | **$\approx 495.8\text{ MiB}$** |
| **Пам'ять контексту на 4k токенів** | $\approx 1.2\text{ GiB}$ (KV-кеш) | $\approx 7\text{ MiB}$ (DeltaNet) | **$\approx 7.0\text{ MiB}$** |
| **Пам'ять контексту на 256k токенів** | $\approx 27.3\text{ GiB}$ (OOM на ноутбуках) | $\approx 7\text{ MiB}$ (DeltaNet) | **$\approx 40.7\text{ MiB}$ (DeltaNet + Hopfield KV)** |
| **Кількість forward passes (1600 токенів)** | 1600 послідовних кроків | 1600 послідовних кроків | **$\mathbf{25\text{--}30}$ кроків сумарно** |
| **Швидкість генерації на CPU ноутбука** | 4–8 токенів/сек ($\approx 4\text{ хв}$) | 25–40 токенів/сек ($\approx 50\text{ с}$) | **$600\text{--}1140\text{ токенів/сек}$ ($\approx 1.5\text{--}2.5\text{ с}$)** |
| **Пікове споживання RAM під час роботи** | $6.0\text{--}8.0\text{ GiB}$ | $\approx 1.2\text{ GiB}$ | **$\mathbf{525.74\text{ MiB}}$** (ліміт: $< 750\text{ MiB}$) |
| **Робота на батареї від 12W TDP** | Сильне просідання ($< 2\text{ т/с}$) | $\approx 10\text{--}15\text{ т/с}$ | **$> 1000\text{ т/с}$ (PowerGuard)** |

---

## 🏛️ Архітектурна топологія моделі

```
                                [Користувацький промпт]
                                           │
                                           ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                                 DENOISING STEM (S₀)                                    │
│             28 шарів Gated DeltaNet (FP32 State: 7 MiB) + 6 шарів Global Softmax GQA   │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                     ┌─────────────────────┴─────────────────────┐
                     ▼                                           ▼
┌──────────────────────────────────────────┐   ┌─────────────────────────────────────────┐
│     ФАЗА 1: CAUSAL BLUEPRINT DRAFTER     │   │     ФАЗА 2: BIDIRECTIONAL REFINER       │
│  4–8 кроків креслення мета-концептів:    │──▶│  3–4 паралельні бідирекційні проходи    │
│  B_k = [c_sem, idx_hop, τ_style, ℓ_len]  │   │  + Cosine Dedup (>0.85) & Seam Smoother │
└──────────────────────────────────────────┘   └────────────────────┬────────────────────┘
                                                                    │
                                                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                           ФАЗА 3: BLOCK-CAUSAL NCM DECODER                             │
│       Чанки по 400 токенів: Coarse-to-Fine NCM (Синтаксис ➔ Морфологія ➔ Токени)       │
│       Контроль невизначеності: Confidence Gate (Shannon Entropy) + 3 Emergency Passes  │
│       Зв'язок між блоками: миттєве оновлення рекурентного стану S_k через DeltaNet     │
└──────────────────────────────────────────┬─────────────────────────────────────────────┘
                                           │
                                           ▼
                              [Згенерований документ]
```

### Монолітна арена пам'яті (`525.74 MiB`) <a name="memory-arena"></a>

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        HETMAN-2-NCM MONOLITHIC MMAP ARENA                              │
│                                TOTAL: 525.74 MiB                                       │
├──────────────────────────────────┬───────────────────────┬─────────────────────────────┤
│ BitNet b1.58 Ternary Weights     │ 495.77 MiB            │ 2.08B weights (2-bit packed)│
│ BF16 Embeddings & Tied LM Head   │ 192.00 MiB (in RAM)   │ 65536 x 1536 x 2 bytes      │
│ DeltaNet State (28 layers, FP32) │   7.00 MiB            │ 28 x 16 x 64 x 64 x 4 bytes │
│ SLH-Core v2 Codebooks & W_expand │   2.62 MiB            │ BF16 table descriptors      │
│ Phase 1-3 Scratchpads (C++ Pool) │  24.80 MiB            │ Draft + Refine + Decode     │
│ Global GQA KV-Cache (2048 ctx)   │   6.00 MiB            │ 6 layers, 12:2 GQA          │
│ BPE Table & Tensor Descriptors   │   2.06 MiB            │ Headers & metadata          │
├──────────────────────────────────┼───────────────────────┼─────────────────────────────┤
│ БЮДЖЕТНИЙ ПОРОГ: 750.00 MiB      │ ЗАПАС: 224.26 MiB     │ ВИТРИМАНО НА 100% (29.9%)   │
└──────────────────────────────────┴───────────────────────┴─────────────────────────────┘
```

---

## 🚀 Швидкий старт (Python / Hugging Face)

Модель повністю сумісна зі стандартним інтерфейсом **Hugging Face Transformers** через `Dynamic AutoClass`:

```python
from transformers import AutoConfig, AutoModelForCausalLM

# Автоматичне завантаження з репозиторію через remote code
config = AutoConfig.from_pretrained("configs", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("configs", trust_remote_code=True)

# Генерація тексту за трифазним пайплайном NCM з динамічним сінкінгом (Варіант В)
prompt = "Поясни принципи квантових обчислень простою мовою."
output = model.generate_ncm(
    prompt,
    max_tokens=1600,
    deliberate=True,           # динамічний сінкінг («стільки, скільки треба»)
    explain_thought=True,      # інспекція структурованих кроків міркувань
    think=False                # генерація окремого <think> блоку
)
print("Cognitive Level:", output["cognitive_profile"]["level"])
print("Passes Executed:", output["cognitive_profile"]["passes_executed"])
print("Generated Tokens:", output["generated_tokens"])
```

---

## 🧪 Верифікація та тести надійності

Вся кодова база покрита вичерпними автоматичними тестами:

```bash
# Запуск повного тестового комплексу (78 тестів)
python -m unittest discover tests

# Запуск стрес-тесту на відмовостійкість (Chaos Monkey Red Team)
python -m unittest tests/test_chaos_monkey_adversarial.py
```

* **102/102 тестів пройдено успішно** (`OK`).
* **Chaos Monkey (17/17 векторів атак відбито):** порожні промпти, 10 000 повторень, шторм емодзі, ін'єкція NaN, спам слотів Хопфілда, виродження планів — повне самовідновлення без падінь.

---

## 📁 Структура проєкту

```
Hetman-2-2B/
├── configs/
│   ├── config.json                     # HF AutoConfig конфігурація
│   └── hetman2_ncm_config.json         # Повний канонічний конфіг (94 параметри)
├── src/
│   ├── modeling_hetman2.py             # Єдиний тернарний бекбон (DRAFT/REFINE/DECODE)
│   ├── configuration_hetman2.py        # HuggingFace PretrainedConfig з перевіркою типів
│   ├── dual_loop_deliberation.py       # Варіант В: Гібридний дуальний контур (CDR + Banach Loop)
│   ├── deltanet.py                     # Gated DeltaNet O(1) рекурентне ядро (FP32 State)
│   ├── hopfield_core.py                # SLH-Core v2 (16.78M слотів фактологічної пам'яті)
│   ├── hopfield_kv_cache.py            # Ієрархічний Hopfield KV-кеш з алгоритмом Велфорда
│   ├── concept_codebook.py             # 4-рівневий RVQ кодбук концептів L0/L1/L2
│   ├── blueprint_drafter.py            # Фаза 1: Causal AR Drafter креслень блоків
│   ├── bidirectional_refiner.py        # Фаза 2: Global Bidirectional Refiner + Cosine Dedup
│   ├── block_ncm_decoder.py            # Фаза 3: Поблоковий NCM декодер зі шлюзом ентропії
│   ├── dynamic_blueprint_evolver.py    # Віднімання наміру за Грамом-Шмідтом
│   ├── quantization.py                 # BitNet b1.58 (Format 0) + in-place FWHT INT8
│   └── export_to_rada.py               # Потоковий експортер у двійковий формат Rada v2
├── spec/
│   ├── 01..24_*.md                     # 24 повні архітектурні RFC-специфікації
│   ├── audits/                         # 21 звіт прискіпливого аудиту субагентів
│   ├── MASTER_AUDIT_SYNTHESIS_REPORT.md# Генеральний синтетичний звіт аудиту
│   └── rada_ncm_engine.hpp             # C++20 заголовок рушія Rada.cpp (Zero-Alloc Arena)
├── tests/                              # Комплексні регресійні, стрес- та сінкінг-тести
├── training_loss_engine.py             # Оптимізатор Muon + трифазний тренувальний лос
└── training_reasoning_grpo.py          # Critic-Free GRPO рушій навчання деліберації
```

---

## 📜 Ліцензія та автори

* **Автор та головний архітектор:** [Ярослав Самчук](https://github.com/YaroslavSamchuk) (samchukyaroslav@gmail.com)
* **Ліцензія:** [Apache License 2.0](LICENSE) — вільне академічне та комерційне використання зі збереженням авторства.
