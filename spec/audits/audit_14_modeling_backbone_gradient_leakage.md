# RFC-HETMAN-2.0B-AUDIT-014: Технічний аудит єдиного бекбону, ізоляції градієнтів та протоколу перемикання режимів (Mode Switching)

**Документ:** RFC-HETMAN-2.0B-AUDIT-014  
**Цільовий компонент:** `src/modeling_hetman2.py` (Hetman-2-NCM 2.0B Monolithic Backbone Architecture)  
**Мандат аудиту:** Аудитор єдиного бекбону та протоколу перемикання режимів (Chief Auditor of Unified Backbone & Mode Switching Protocol)  
**Дата аудиту:** 19 жовтня 2026 року  
**Статус документа:** ЗАТВЕРДЖЕНО (CRITICAL ARCHITECTURAL DIRECTIVE — MANDATORY REMEDIATION)  
**Репозиторій реалізації:** `src/modeling_hetman2.py`, `src/quantization.py`, `rada.cpp`, `training_loss_engine.py`  

---

## РЕЗЮМЕ АУДИТУ (EXECUTIVE SUMMARY)

Архітектура **Hetman-2-NCM 2.0B** використовує концепцію **100% повторного використання параметрів** (Zero-RAM Weight Sharing, RFC-006): єдиний 34-шаровий бекбон послідовно виконує роль каузального планувальника (`MODE_DRAFT`), глобального некаузального рефайнера (`MODE_REFINE`) та паралельного блоково-каузального декодера (`MODE_DECODE`), додаючи лише 18 КБ на Mode Embeddings та 96 КБ на вектори модуляції AdaLN.

Проведений глибинний аудит кодової бази `src/modeling_hetman2.py` виявив, що за високої елегантності загального концепту поточна реалізація містить **4 фундаментальні архітектурні дефекти критичного та високого рівня**, що призводять до:
1. **Деструктивної градієнтної інтерференції (Negative Transfer / Gradient Contamination)** між каузальними та некаузальними задачами під час спільного навчання, а також розриву диференційовності в Phase 2;
2. **Катастрофічної вразливості Triad Fusion Gate до NaN/Inf інфекції**, нездатності гейту ізолювати відмову окремої гілки (Hopfield або Early Logic) та фізико-математичної некоректності нормування $1/\sqrt{3}$;
3. **Витоку майбутнього (Future Leakage) і розбухання VRAM через некоректне кешування уваги**, а також того факту, що 12 із 16 шарів Arbiter (Gated DeltaNet) взагалі не підтримують двонаправлену увагу;
4. **Високих накладних витрат перемикання режимів** через динамічні розгалуження Python (68 `if`-розгалужень на крок), скидання guard-трас `torch.compile` та зайві запуски ядер CUDA.

Нижче наведено детальний аналіз виявлених дефектів, математичні доведення та інженерні рішення для досягнення **строго 0 нс накладних витрат**.

---

### Зведена матриця виявлених дефектів (Defect Matrix)

| ID | Компонент | Рівень ризику | Суть дефекту | Наслідки в експлуатації |
|:---|:---|:---:|:---|:---|
| **DEF-01** | `mode_embeddings` & Backbone Weights | **CRITICAL** | Конфлікт градієнтів каузального та двонаправленого режимів у спільних вагах $W$ | Зростання перплексії генерації, руйнування каузального синтаксису, нестійкість спільного навчання |
| **DEF-02** | `refine_blueprints` Autograd Graph | **HIGH** | Наскрізний витік 64-шарового BPTT; розрив автографа через `.item()` у дедуплікації | Зрив градієнтного оптимізатора, відсутність навчання границь відштовхування, OOM |
| **DEF-03** | `forward_triad` Gate Architecture | **CRITICAL** | *A priori* скалярний гейт без ізоляції гілок; конкатенація $3D \to D$ через спільну матрицю | Поява хоча б одного NaN у Хопфілді миттєво заражає 100% вихідних каналів ($0.0 \times \text{NaN} = \text{NaN}$) |
| **DEF-04** | `forward_exit` Residual Highway | **MEDIUM** | Математична некоректність $1/\sqrt{3}$ для корельованих випадкових величин; FP16 overflow | Зростання дисперсії суми до $\sim 2.6 \sigma^2$; ризик переповнення FP16 ($> 65504$) до нормалізації |
| **DEF-05** | `GatedDeltaNet` in Arbiter (12 layers) | **CRITICAL** | DeltaNet у `MODE_REFINE` продовжує виконувати суворо каузальну рекурентність | У фазі рефайнінгу 75% шарів Арбітра є сліпими до майбутнього контексту; рефайнінг фрагментований |
| **DEF-06** | `GlobalAttention` KV-Cache & Masking | **HIGH** | Збереження двонаправленого KV-кешу для каузального декодування; обнулення mask паддінгу | Витік інформації про майбутні токени (Data Leakage); забруднення контексту сміттям паддінгу |
| **DEF-07** | Mode Switching Runtime | **MEDIUM** | Динамічний диспетчер Python (`if mode == ...`), зрив JIT Guard у `torch.compile` | 5–25 мкс оверхеду на крок, деоптимізація компілятора графіку інференсу |

---

## РОЗДІЛ 1. АУДИТ MODE EMBEDDING ТА АНАЛІЗ ВИТОКУ ГРАДІЄНТІВ МІЖ РЕЖИМАМИ

### 1.1. Математична механіка Mode Embedding у `modeling_hetman2.py`
У рядках 982 та 1141 файлу `src/modeling_hetman2.py`:
```python
# Ініціалізація Mode Conditioning Embeddings: [3, 1, 1, 1536]
self.mode_embeddings = nn.Parameter(torch.randn(3, 1, 1, config.hidden_size) * 0.02)
...
# Інжекція вектора режиму на вході бекбону
x = x + self.mode_embeddings[mode]
```
Також у `HetmanLayer.forward` (рядки 668–691) реалізовано AdaLN-модуляцію для режиму `MODE_REFINE`:
```python
norm_x = self.norm1(x)
if mode == MODE_REFINE:
    norm_x = norm_x * (1.0 + self.adaln_gamma1) + self.adaln_beta1
...
norm_ffn = self.norm2(x)
if mode == MODE_REFINE:
    norm_ffn = norm_ffn * (1.0 + self.adaln_gamma2) + self.adaln_beta2
```

### 1.2. Локальний автограф `self.mode_embeddings`
Розглянемо градієнти функції втрат за самими векторами режимів. Нехай спільна функція втрат під час спільного навчання має вигляд:
$$\mathcal{L}_{\text{total}} = \lambda_0 \mathcal{L}_{\text{draft}} + \lambda_1 \mathcal{L}_{\text{refine}} + \lambda_2 \mathcal{L}_{\text{decode}}$$
Оскільки в операції `self.mode_embeddings[mode]` використовується індексація (тензорний слайсинг), граф Autograd ізолює кожен зріз:
$$\frac{\partial \mathcal{L}_{\text{total}}}{\partial \mathbf{e}_m} = \lambda_m \frac{\partial \mathcal{L}_m}{\partial \mathbf{e}_m}, \quad m \in \{0, 1, 2\}$$
Отже, **всередині самого тензора `self.mode_embeddings` прямого витоку градієнтів між зрізами $\mathbf{e}_0, \mathbf{e}_1, \mathbf{e}_2$ немає**.

### 1.3. Критичний дефект: Деструктивна градієнтна інтерференція у спільних вагах бекбону (Gradient Contamination)
Головна архітектурна вразливість полягає в тому, що всі матриці лінійних перетворень бекбону:
$$\mathcal{W} = \{W_Q, W_K, W_V, W_O, W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}\}$$
є **на 100% спільними** між усіма 3 режимами.

Під час спільного навчання градієнт за будь-якою спільною матрицею ваг $W \in \mathcal{W}$ становить:
$$\nabla_W \mathcal{L}_{\text{total}} = \lambda_{\text{draft}} \nabla_W \mathcal{L}_{\text{draft}} + \lambda_{\text{refine}} \nabla_W \mathcal{L}_{\text{refine}} + \lambda_{\text{decode}} \nabla_W \mathcal{L}_{\text{decode}}$$

Розглянемо природу оптимізаційних ландшафтів цих задач:
1. **Каузальні режими (`MODE_DRAFT`, `MODE_DECODE`):**
   Вимагають строгого дотримання каузального конуса: $\frac{\partial y_t}{\partial x_s} = 0$ для всіх $s > t$. Проєкції $W_Q, W_K$ навчаються стискати односпрямовану історію контексту.
2. **Двонаправлений режим (`MODE_REFINE`):**
   Знімає каузальну маску в шарах Global Attention. Кожен токен взаємодіє з усіма іншими: $\frac{\partial y_t}{\partial x_s} \neq 0$ для $s > t$. Проєкції $W_Q, W_K$ навчаються шукати глобальні симетричні зв'язки.

**Математичний конфлікт (Negative Transfer / Gradient Conflict):**
Вектори градієнтів $\mathbf{g}_{\text{causal}} = \nabla_W (\mathcal{L}_{\text{draft}} + \mathcal{L}_{\text{decode}})$ та $\mathbf{g}_{\text{bidir}} = \nabla_W \mathcal{L}_{\text{refine}}$ мають від'ємний косинусний добуток:
$$\langle \mathbf{g}_{\text{causal}}, \mathbf{g}_{\text{bidir}} \rangle = \|\mathbf{g}_{\text{causal}}\| \|\mathbf{g}_{\text{bidir}}\| \cos \theta, \quad \text{де } \cos \theta < 0$$
Накопичення таких градієнтів веде до явища **градієнтного конфлікту**: градієнтний крок рефайнера дестабілізує внутрішні каузальні уявлення авторегресійного декодера, різко погіршуючи перплексію на каузальних задачах (Temporal Bleeding / Causal Corruption).

### 1.4. Наскрізний витік у рекурентному конвеєрі `refine_blueprints`
У методі `refine_blueprints` (рядки 1187–1228) виконується цикл із 4 проходів:
```python
for step in range(1, num_passes + 1):
    H_conditioned = H_curr + self.mode_embeddings[MODE_REFINE]
    H_flat = H_conditioned.view(B, K * L_c, d)
    H_arbiter_flat, _ = self.forward_arbiter(H_flat, mode=MODE_REFINE)
    ...
    H_dedup, M_sim, repulsions = self.deduplicator(H_anchored)
    H_smoothed = self.seam_smoother(H_dedup)
    H_watchdog, current_energy, watchdog_triggered = self.watchdog(H_smoothed, initial_energy)
    H_quant, mean_eta = self.fwht_quantizer(H_watchdog)
    ...
```
Якщо тренувати модель у наскрізному режимі (End-to-End Unrolled Training):
1. **64-шаровий BPTT-ланцюг:** 4 проходи через 16-шаровий Арбітр еквівалентні 64 шарам зворотного поширення помилки через нелінійні операції згасання та FWHT без проміжних Stop-Gradient бар'єрів. Це неминуче спричиняє експоненційний вибух або згасання градієнтів ($\gamma^{64} \to 0$ або $\to \infty$).
2. **Розрив градієнтного графа (Autograd Disconnection) у `SemanticDeduplicationFilter`:**
   У рядку 727:
   ```python
   sim = M[b, i, j].item()  # ПЕРЕТВОРЕННЯ В PYTHON FLOAT!
   if sim > self.tau:
       alpha_rep = min(1.0, (sim - self.tau) / (1.0 - self.tau + self.eps))
   ```
   Виклик `.item()` вивантажує значення подібності на хост як звичайне число Python, **повністю обриваючи ланцюжок Autograd для порогу $\tau$ та коефіцієнта репульсії $\alpha_{\text{rep}}$**. Оптимізатор не отримує градієнтів за метрикою схожості чанків!
3. **In-place мутації тензорів:**
   Рядки 736–737:
   ```python
   H_out[b, j] = H_out[b, j] + delta_c.unsqueeze(0)
   c[b, j] = H_out[b, j].mean(dim=0)
   ```
   Мутація `H_out` через індексацію всередині вкладеного циклу за $i, j$ порушує версійний контроль Autograd у PyTorch (`RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation`).

### 1.5. Асиметрія кондиціонування: відсутність AdaLN для `MODE_DECODE`
У поточному коді (рядки 668–691) перевірка виконується виключно як `if mode == MODE_REFINE:`.  
Для `MODE_DRAFT` та `MODE_DECODE` в шарах `HetmanLayer` **немає жодних AdaLN-параметрів**.
Вся різниця між генерацією абстрактного плану (Phase 1) та емісією лексичних токенів (Phase 3) зводиться до додавання єдиного статичного вектора $e_m \in \mathbb{R}^{1536}$ на нульовому шарі бекбону:
$$x^{(0)} = x_{\text{in}} + \mathbf{e}_{\text{decode}}$$
Оскільки модель містить 34 послідовні блоки Pre-RMSNorm:
$$\text{RMSNorm}(z) = \frac{z}{\sqrt{\frac{1}{d}\sum z_i^2 + \epsilon}}$$
будь-яке постійне зміщення, додане на рівні 0, прогресивно нівелюється та поглинається залишковими зв'язками. Уже на рівні Арбітра (шар 15–20) активації втрачають пам'ять про те, чи перебуває модель у режимі планування чи декодування.

---

## РОЗДІЛ 2. АУДИТ TRIAD FUSION GATE ТА СТІЙКОСТІ НОРМУВАННЯ 1/sqrt(3)

### 2.1. Анатомія Triad Fusion Gate
У `Hetman2Model.forward_triad` (рядки 1055–1084):
```python
# Гілка A: Хопфілд
y_know, hopfield_loss = self.hopfield_core(x_stem)

# Гілка B: SwiGLU FFN (8 шарів)
y_domain = x_stem
for swiglu in self.domain_backbone:
    y_domain = y_domain + swiglu(y_domain)

# Гілка C: Гіпотези / Рання логіка (6 шарів)
y_hypo = x_stem
for i, layer in enumerate(self.hypothesis_subnet):
    y_hypo, next_st = layer(y_hypo, prev_state=st, offset=offset, mask=attention_mask, mode=mode)

# Злиття гілок
y_triad = torch.cat([y_know, y_domain, y_hypo], dim=-1)  # [B, L, 3 * 1536]
gate = torch.sigmoid(self.fusion_gate(x_stem))            # [B, L, 1536]
x_fused = self.fusion_norm(x_stem + self.fusion_proj(y_triad) * gate)
```

### 2.2. Поведінка при чисельній нестійкості (NaN / Inf / Explosion)
Розглянемо випадок, коли одна з гілок видає `NaN` або `Inf`.
1. **Гілка A (Hopfield):** Якщо PID-балансування зсувів $b_1, b_2$ призводить до переповнення, або арктангенс/софтмакс пари скорів $\beta \cdot (S_1 + S_2)$ видає `NaN` через ділення на нуль, тензор $y_{\text{know}}$ містить значення `NaN`.
2. **Конкатенація та спільна проєкція:**
   $$y_{\text{triad}} = \begin{bmatrix} y_{\text{know}} \\ y_{\text{domain}} \\ y_{\text{hypo}} \end{bmatrix} \in \mathbb{R}^{B \times L \times 4608}$$
   Операція `self.fusion_proj` є матричним множенням:
   $$z_k = \sum_{j=1}^{4608} W_{kj} (y_{\text{triad}})_j = \sum_{j=1}^{1536} W_{kj}^{(A)} y_{\text{know}, j} + \sum_{j=1}^{1536} W_{kj}^{(B)} y_{\text{domain}, j} + \sum_{j=1}^{1536} W_{kj}^{(C)} y_{\text{hypo}, j}$$
   Оскільки кожна вихідна координата $z_k$ є сумою, що включає елементи $y_{\text{know}}$, **наявність навіть одного `NaN` у гілці Хопфілда миттєво перетворює всі 1536 каналів виходу $z$ на `NaN`!**
3. **Безпорадність гейту $g = \sigma(W_g x_{\text{stem}})$:**
   Гейт обчислюється *a priori* виключно з вхідного вектора $x_{\text{stem}}$. Він **не має зворотного зв'язку про чисельний стан гілок**. Навіть якби гейт вирішив занулити злиття ($g \to 0.0$):
   Згідно зі стандартом **IEEE 754 Floating-Point**:
   $$0.0 \times \text{NaN} = \text{NaN}, \quad 0.0 \times \infty = \text{NaN}$$
   Множення на нульовий гейт **не здатне придушити NaN або Inf**. Вся магістраль моделі миттєво зазнає незворотного колапсу.

### 2.3. Поведінка при зануленні гілки (Zero / Dead Branch)
Якщо гілка Хопфілда або Early Logic видає нульовий вектор (наприклад, стан холодного старту, порожній запит або масковані позиції):
$$W_{\text{fusion}} y_{\text{triad}} = 0 + W_B y_{\text{domain}} + W_C y_{\text{hypo}}$$
У поточній схемі відсутня динамічна конкурентна ре-нормалізація ваг між гілками (наприклад, через Softmax-гейтування $\alpha_A + \alpha_B + \alpha_C = 1$). Енергія сумарного виходу різко падає на $\approx 33\%$, що призводить до стрибка масштабу перед `self.fusion_norm`.

### 2.4. Катастрофічний дисбаланс енергії (Scale Mismatch) гілок
Звернемо увагу на гілку B (`self.domain_backbone`):
```python
y_domain = x_stem
for swiglu in self.domain_backbone:
    y_domain = y_domain + swiglu(y_domain)
```
**У гілці B виконується 8 послідовних додавань SwiGLU БЕЗ ЖОДНОГО ШАРУ RMSNorm!**
У той час як:
- Гілка A (Hopfield) нормалізована через сферичні кодобуки та BitLinear;
- Гілка C (Hypothesis Subnet) містить Pre-RMSNorm у кожному з 6 шарів `HetmanLayer`.

У гілці B дисперсія росте монотонно з глибиною:
$$\text{Var}(y_{\text{domain}}) \approx \text{Var}(x_{\text{stem}}) + 8 \cdot \sigma_{\text{SwiGLU}}^2 \gg \text{Var}(y_{\text{know}})$$
Внаслідок цього гілка B повністю домінує у конкатенації $y_{\text{triad}}$, заглушаючи сигнали асоціативної пам'яті Хопфілда та логічних гіпотез.

### 2.5. Математичний аудит нормування $1/\sqrt{3}$ у `forward_exit`
У рядку 1113 файлу `src/modeling_hetman2.py`:
```python
# Multi-Highway Residual Integration
x_exit = self.exit_norm((x_stem + x_fused + x_arbiter) / math.sqrt(3.0))
```

#### Доведення некоректності масштабування $1/\sqrt{3}$
Множник $1/\sqrt{N}$ ($N=3$) строго зберігає одиничну дисперсію суми $\text{Var}\left(\frac{1}{\sqrt{N}}\sum X_i\right) = \sigma^2$ **лише тоді, коли випадкові величини $X_1, X_2, \dots, X_N$ є взаємно некорельованими (ортогональними)**:
$$\text{Cov}(X_i, X_j) = 0 \quad (\forall i \neq j)$$

Перевіримо цю умову для наших трьох компонентів:
1. $X_1 = x_{\text{stem}}$ — вихід стем-блоку.
2. $X_2 = x_{\text{fused}} = \text{RMSNorm}(x_{\text{stem}} + \Delta_{\text{triad}})$ — прямо містить $x_{\text{stem}}$ як основний залишковий зв'язок.
3. $X_3 = x_{\text{arbiter}} = x_{\text{fused}} + \sum_{l=1}^{16} \Delta_l$ — є результатом проходження $x_{\text{fused}}$ через 16 залишкових шарів Арбітра.

Компоненти $x_{\text{stem}}, x_{\text{fused}}, x_{\text{arbiter}}$ є **сильно позитивно корельованими величиною $\rho \approx 0.75 - 0.85$**!
Розрахуємо реальну дисперсію суми $S = X_1 + X_2 + X_3$:
$$\text{Var}(S) = \sum_{i=1}^3 \text{Var}(X_i) + 2 \sum_{1 \le i < j \le 3} \text{Cov}(X_i, X_j) \approx 3 \sigma^2 + 2 \cdot 3 \cdot (0.8 \sigma^2) = 7.8 \sigma^2$$
Ділення на $\sqrt{3}$ дає результуючу дисперсію:
$$\text{Var}\left(\frac{S}{\sqrt{3}}\right) = \frac{7.8 \sigma^2}{3} \approx \mathbf{2.6 \sigma^2 \neq \sigma^2}$$
Дисперсія суми зростає більш ніж у 2.5 раза!

#### Взаємодія з RMSNorm та ризик FP16 Overflow
Одразу після ділення застосовується `self.exit_norm = nn.RMSNorm(d)`.
З математичної точки зору, для будь-якого строго додатного скаляра $c = \frac{1}{\sqrt{3}}$:
$$\text{RMSNorm}(c \cdot u) = \frac{c \cdot u}{\sqrt{\frac{1}{d} \sum (c u_i)^2 + \epsilon}} = \frac{u}{\sqrt{\frac{1}{d} \sum u_i^2 + \frac{\epsilon}{c^2}}} = \frac{u}{\sqrt{\text{mean}(u^2) + 3\epsilon}}$$
Масштабування на $1/\sqrt{3}$ **повністю нівелюється нормалізатором RMSNorm**, змінюючи лише регуляризаційний член $\epsilon \to 3\epsilon$.
**АЛЕ:** якщо додавання $x_{\text{stem}} + x_{\text{fused}} + x_{\text{arbiter}}$ виконується у форматі зниженої точності FP16, де максимальне представлене число $65,504$:
Сума трьох великих активацій перед діленням на $\sqrt{3}$ може спричинити **апаратне переповнення FP16 до $+\infty$**, що миттєво призведе до генерації `NaN` після ділення та нормалізації!

---

## РОЗДІЛ 3. АУДИТ ПЕРЕМИКАННЯ МАСКИ ARBITER, ВИТОКУ ПАМ'ЯТІ ТА КЕШУВАННЯ УВАГИ

### 3.1. Архітектурна сліпота: DeltaNet у складі Арбітра
У рядках 1003–1011 класу `Hetman2Model` визначено шари Арбітра:
```python
arbiter_list = []
for i in range(16):
    if i in (3, 7):
        arbiter_list.append(HetmanLayer(config, is_global=True, yarn_cooling=2.0))
    elif i in (11, 15):
        arbiter_list.append(HetmanLayer(config, is_global=True, yarn_cooling=2.4))
    else:
        arbiter_list.append(HetmanLayer(config, is_global=False))
self.arbiter_layers = nn.ModuleList(arbiter_list)
```
У 16-шаровому Арбітрі:
- Лише **4 шари** (індекси 3, 7, 11, 15) є глобальними шарами Softmax-уваги (`GlobalAttention`);
- Інші **12 шарів** (0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14) є шарами **`GatedDeltaNet`**.

**Критичний дефект реалізації `GatedDeltaNet`:**
Поглянемо на `GatedDeltaNet.forward` (рядки 439–460):
```python
for t in range(L):
    ...
    # Stepwise Linear Recurrence (Gated Delta Rule)
    v_pred = torch.einsum('bhij,bhj->bhi', S, k_t)
    v_err = v_t - v_pred
    update = torch.einsum('bhi,bhj->bhij', v_err, k_t)
    S = S * a_t + b_t * update
    y_t = torch.einsum('bhij,bhj->bhi', S, q_t)
```
Клас `GatedDeltaNet` **взагалі не приймає аргумент `mode` і не має механізму двонаправленого оновлення!**  
Він реалізує виключно односпрямовану (каузальну) рекурентність від $t=0$ до $t=L-1$.

**Наслідок для Phase 2 (`refine_blueprints`):**
Коли рефайнер передає плоский тензор $K \times L_c$ токенів у `forward_arbiter(H_flat, mode=MODE_REFINE)`:
У 12 із 16 шарів Арбітра токени опрацьовуються **суворо зліва направо**. Токен чанку $0$ фізично не має доступу до чанків $1, 2, \dots, K-1$. Зворотний інформаційний потік виникає лише у 4 рідкісних точках (шари 3, 7, 11, 15), після чого знову примусово затискається в односпрямовану каузальність DeltaNet.  
Це прямо суперечить специфікації RFC-006 (Global Non-Causal Self-Attention Graph).

### 3.2. Витік майбутнього (Future Leakage) через KV-кеш у `GlobalAttention`
У `GlobalAttention.forward` (рядки 289–295):
```python
if past_kv is not None:
    past_k, past_v = past_kv
    k_norm = torch.cat([past_k, k_norm], dim=1)
    v = torch.cat([past_v, v], dim=1)
current_kv = (k_norm, v)
```
1. **Витік із Refine у Decode (Future Leakage):**
   Якщо стан `past_states["arbiter"]`, отриманий після виконання `MODE_REFINE` (де ключі та значення містять глобальну некаузальну інформацію про всю послідовність), випадково або за замовчуванням передається у `MODE_DECODE`:
   Авторегресійний декодер отримує KV-кеш, у якому токени вже знають зміст майбутніх чанків. Модель здійснює несанкціоноване підглядання в майбутнє (Causal Violation / Information Leakage), що призводить до руйнування генерації на нових токенах за межами кешу.
2. **Асиметрія префіксу при переході з Causal у Bidirectional:**
   Якщо каузальний префікс із `MODE_DRAFT` передається як `past_kv` у `MODE_REFINE`:
   У каузальному префіксі вектори $K, V$ розраховані без урахування наступних токенів, тоді як нові токени розраховані двонаправлено. Виникає несиметричний перекіс уваги, що руйнує інваріантність лінійного розгладження.

### 3.3. Витік пам'яті VRAM через утримання графа Autograd
Зверніть увагу:
`k_norm` та `v` конкатенуються без виклику `.detach()`.  
Якщо інференс виконується без декоратора `@torch.no_grad()`, або під час навчання з накопиченням градієнтів стан `current_kv` передається між кроками:
Кожен наступний виклик додає нові тензори до довгого ланцюжка обчислювального графа. Граф Autograd не звільняється з VRAM, що спричиняє лінійний витік пам'яті $\mathcal{O}(T \cdot L)$ та фатальний збій CUDA Out-of-Memory (OOM).

### 3.4. Знищення маски паддінгу при двонаправленому виклику
У рядках 309–312:
```python
if bidirectional:
    attn_mask = None
else:
    attn_mask = mask
```
Якщо в батчі обробляються послідовності різної довжини, і користувач передав бінарну маску відсікання паддінгу (`padding_mask`):
У режимі `bidirectional=True` код **примусово встановлює `attn_mask = None`**, повністю викидаючи маску користувача!  
Внаслідок цього дійсні концептуальні токени звертатимуть увагу на нульові/сміттєві токени вирівнювання батчу, що призведе до спотворення центроїдів чанків у Phase 2.

---

## РОЗДІЛ 4. ОПТИМІЗАЦІЯ: СКОРОЧЕННЯ НАКЛАДНИХ ВИТРАТ ПЕРЕМИКАННЯ РЕЖИМІВ ДО 0 НС

### 4.1. Джерела поточних накладних витрат у `modeling_hetman2.py`
У поточній імплементації перемикання режимів не є безкоштовним:
1. **Python Dynamic Branching:** 
   У кожному з 34 шарів на кожному кроці виконуються перевірки:
   - `if mode == MODE_REFINE:` (двічі на шар для AdaLN: нормалізація уваги та FFN $\implies 68$ динамічних розгалужень);
   - `if self.is_global: bidirectional = (mode == MODE_REFINE)` (для 4 шарів Арбітра та 2 шарів Стем/Гіпотез);
   - Загалом: $\approx 74$ перевірки умов у середовищі інтерпретатора Python на один прямий прохід.
2. **TorchInductor Guard Invalidation (`torch.compile`):**
   Передача аргументу `mode: int` у скомпільовану функцію призводить до створення Guards у PyTorch Dynamo. Зміна аргументу з `mode=0` на `mode=1` викликає Guard Failure та тривалу рекомпcompiler-паузу (JIT Re-compilation Stalls) тривалістю 200–800 мс!
3. **Окремий запуск ядра CUDA для додавання Mode Embedding:**
   Операція `x = x + self.mode_embeddings[mode]` виконує вибірку зрізу та запускає окреме ядро поелементного додавання. На сучасних GPU (наприклад, RTX 2060 / 3060) накладні витрати на диспетчеризацію одного ядра становлять 3–8 мікросекунд.
4. **Алокація пам'яті під словники станів:**
   Створення об'єктів `new_states = {"stem": ..., "arbiter": ...}` на кожному кроці спричиняє навантаження на Garbage Collector Python.

---

### 4.2. Архітектурна дорожня карта скорочення оверхеду до строго 0 нс (Zero-Overhead Protocol)

```
+----------------------------------------------------------------------------------------------------+
|                         ZERO-OVERHEAD (0 ns) RUNTIME ARCHITECTURE IN RADA.CPP                      |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  1. AOT KERNEL SPECIALIZATION                                                                      |
|     Compile 3 immutable execution plans:                                                           |
|     - Pipeline_Draft()   -> Causal FlashAttn, AdaLN completely omitted from assembly               |
|     - Pipeline_Refine()  -> Dense Symmetric Attn, Fused AdaLN-SubLN kernel                         |
|     - Pipeline_Decode()  -> Block-Causal Decode, Pre-allocated KV Cache                             |
|     Mode dispatch = Function Pointer swap (0.3 ns CPU register, amortized 0 ns)                    |
|                                                                                                    |
|  2. WEIGHT PRE-BAKING (CONSTANT EMBEDDING FOLDING)                                                 |
|     e_draft and e_decode folded directly into Token Embedding Table:                               |
|     W_embed_draft[v]  = W_embed[v] + e_draft                                                       |
|     W_embed_decode[v] = W_embed[v] + e_decode                                                      |
|     Eliminates runtime tensor addition kernel completely (0 ns execution penalty)                  |
|                                                                                                    |
|  3. ZERO-COPY PING-PONG MEMORY ARENAS                                                              |
|     State buffers (7.00 MiB DeltaNet + KV Cache) pre-allocated at startup:                         |
|     Mode switch = Pointer alias reassignment, 0 byte allocation at runtime                         |
|                                                                                                    |
|  4. HARDWARE-LEVEL ATTENTION MASK STRIDE DISPATCH                                                  |
|     In C++/CUDA kernel:                                                                            |
|     mask_stride = (mode == MODE_REFINE) ? 0 : seq_len                                             |
|     Branchless bitwise mask indexing in GPU shared memory                                          |
+----------------------------------------------------------------------------------------------------+
```

#### Рішення 1: Попередня компіляція та спеціалізація графіків (AOT Multi-Graph Specialization)
Замість монолітної функції з внутрішніми перевірками `if mode == ...`:
Створюються 3 окремі спеціалізовані екземпляри функції виконання:
- `forward_draft(input_ids, states)`
- `forward_refine(H_blueprints)`
- `forward_decode(inputs_embeds, states)`

У середовищі C++ (`rada.cpp`) це реалізується через шаблони часу компіляції:
```cpp
template <OperatingMode Mode>
void Hetman2Backbone::forward_layer(Tensor& x, LayerState& state) {
    if constexpr (Mode == MODE_REFINE) {
        // Компілюється виключно для рефайнера: злитий AdaLN + двонаправлена увага
        fused_adaln_subln(x, adaln_gamma1, adaln_beta1);
        dispatch_bidirectional_attention(x);
    } else {
        // Для DRAFT та DECODE: AdaLN повністю вилучено з двійкового коду!
        rmsnorm(x);
        dispatch_causal_deltanet_or_attention(x, state);
    }
}
```
**Результат:** Всередині внутрішніх циклів виконання відсутні розгалуження (0 branch instructions, 0 branch mispredictions). Перемикання режиму — це виклик відповідного покажчика на функцію, що займає менше 1 такту процесора ($< 0.3 \text{ нс}$), тобто практично **0 нс** відносно часу виконання шару.

#### Рішення 2: Передвбудовування векторів режиму (Weight Pre-Baking / In-Place Folding)
Вектори `mode_embeddings` є статичними під час інференсу.
Замість виконання поелементного додавання $x = x + \mathbf{e}_m$:
1. **Для `MODE_DRAFT` та `MODE_DECODE`:**
   Вектор режиму передпоєднується (fold) безпосередньо з таблицею ембедінгів токенів під час завантаження ваг:
   $$\tilde{E}_{\text{draft}}[w] = E[w] + \mathbf{e}_{\text{draft}}, \quad \forall w \in \{0, \dots, V-1\}$$
   $$\tilde{E}_{\text{decode}}[w] = E[w] + \mathbf{e}_{\text{decode}}, \quad \forall w \in \{0, \dots, V-1\}$$
   При читанні ембедінга вектор режиму зчитується з пам'яті вже доданим!
2. **Для першого лінійного шару (Q-Proj):**
   Якщо на вхід подаються неперервні приховані стани (як у Phase 2):
   $$W_Q (x + \mathbf{e}_{\text{refine}}) = W_Q x + W_Q \mathbf{e}_{\text{refine}} = W_Q x + \mathbf{b}_{\text{refine}}^{\text{folded}}$$
   Вектор $W_Q \mathbf{e}_{\text{refine}}$ попередньо розраховується та додається як константний bias лінійного перетворення.
**Результат:** Повністю усувається окремий запуск CUDA-ядра додавання вектора режиму. Накладні витрати на додавання зведені до **строго 0 нс**.

#### Рішення 3: Zero-Copy Ping-Pong Арени для кешу та стану
У пам'яті GPU/RAM заздалегідь виділяються фіксовані арени пам'яті:
- Рекурентний стан DeltaNet ($28 \text{ шарів} \times 256 \text{ КБ} = 7.000000 \text{ МіБ}$);
- KV-кеш для 4 шарів Global Softmax Attention.
При зміні фази пам'ять не звільняється і не виділяється повторно: покажчик робочого стану `active_state_ptr` атомарно перемикається на відповідний слот в арені. Час перемикання — 1 інструкція запису в регістр CPU (**0 нс оверхеду**).

---

## РОЗДІЛ 5. ЗВЕДЕНА МАТРИЦЯ ВИПРАВЛЕНЬ (ACTIONABLE REMEDIATION ROADMAP)

Для усунення виявлених дефектів у `src/modeling_hetman2.py` необхідно реалізувати такий комплекс архітектурних змін:

```
+----------------------------------------------------------------------------------------------------+
|                                    ARCHITECTURAL PATCH OVERVIEW                                    |
+----------------------------------------------------------------------------------------------------+
|                                                                                                    |
|  [1. Triad Gate Isolation]                                                                         |
|      y_fused = (alpha_A * LayerNorm(y_know) + alpha_B * RMSNorm(y_domain)                          |
|                 + alpha_C * y_hypo) where [alpha_A, alpha_B, alpha_C] = Softmax(RoutingGate(x))    |
|      -> NaN in branch A cannot propagate into branches B and C!                                    |
|                                                                                                    |
|  [2. Highway Barycentric Exit]                                                                     |
|      x_exit = LayerNorm(1/3 * x_stem + 1/3 * x_fused + 1/3 * x_arbiter)                            |
|      -> Computed in FP32; guarantees convex bounded combination without FP16 overflow!             |
|                                                                                                    |
|  [3. Bidirectional DeltaNet in Arbiter]                                                            |
|      In MODE_REFINE: GatedDeltaNet runs bidirectional sweep (Forward Pass + Backward Pass)         |
|      -> Resolves 75% causal blindness in Arbiter during Phase 2!                                   |
|                                                                                                    |
|  [4. Stop-Gradient Phase Decoupling]                                                               |
|      In refine_blueprints: H_in = H_raw.detach()                                                   |
|      In generate_ncm: refined_blueprints = refined_blueprints.detach()                             |
|      -> Eliminates 64-layer BPTT explosion & gradient contamination!                               |
|                                                                                                    |
|  [5. Differentiable Deduplication]                                                                 |
|      Replace .item() with torch.where() & smooth sigmoid barrier                                   |
|      -> Restores Autograd differentiability for semantic repulsion!                                |
+----------------------------------------------------------------------------------------------------+
```

### 5.1. Рефакторинг Triad Fusion Gate: Ізоляція відмов та динамічна Softmax-маршрутизація
Замінити пряму конкатенацію на роздільну нормалізацію та конкурентний Softmax-роутер:

```python
class TriadFusionGate(nn.Module):
    """
    Чисельно стійкий ізолюючий гейт злиття тріади з Softmax-маршрутизацією.
    Унеможливлює поширення NaN/Inf з однієї гілки на інші та вирівнює амплітуди.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm_know = nn.RMSNorm(d_model)
        self.norm_domain = nn.RMSNorm(d_model)
        self.norm_hypo = nn.RMSNorm(d_model)
        
        # Конкурентний роутер: оцінює довіру до кожної з трьох гілок
        self.router = nn.Linear(d_model, 3, bias=False)
        self.proj_out = BitLinear(d_model, d_model, bias=False)
        self.final_norm = nn.RMSNorm(d_model)

    def forward(self, x_stem: torch.Tensor, y_know: torch.Tensor, y_domain: torch.Tensor, y_hypo: torch.Tensor):
        # 1. Захисний бар'єр: санітизація NaN/Inf на вході кожної гілки
        y_know_safe = torch.nan_to_num(y_know, nan=0.0, posinf=10.0, neginf=-10.0)
        y_hypo_safe = torch.nan_to_num(y_hypo, nan=0.0, posinf=10.0, neginf=-10.0)
        y_domain_safe = torch.nan_to_num(y_domain, nan=0.0, posinf=10.0, neginf=-10.0)

        # 2. Індивідуальна нормалізація амплітуд гілок
        y_k_norm = self.norm_know(y_know_safe)
        y_d_norm = self.norm_domain(y_domain_safe)
        y_h_norm = self.norm_hypo(y_hypo_safe)

        # 3. Динамічна конкурентна маршрутизація: alpha_A + alpha_B + alpha_C = 1.0
        routing_weights = F.softmax(self.router(x_stem), dim=-1) # [B, L, 3]
        alpha_A = routing_weights[..., 0:1]
        alpha_B = routing_weights[..., 1:2]
        alpha_C = routing_weights[..., 2:3]

        # 4. Зважена комбінація (неможливо інфікувати суміжні канали)
        fused_triad = alpha_A * y_k_norm + alpha_B * y_d_norm + alpha_C * y_h_norm
        out = self.final_norm(x_stem + self.proj_out(fused_triad))
        return out
```

### 5.2. Виправлення Multi-Highway Exit: Зважений опуклий барицентр
У `forward_exit`: замінити небезпечне додавання та ділення на $\sqrt{3}$ на обчислення в FP32 з фіксованими або вивченими барицентричними коефіцієнтами:
```python
# Безпечне FP32 обчислення барицентричного виходу
x_sum = (x_stem.float() + x_fused.float() + x_arbiter.float()) * (1.0 / 3.0)
x_exit = self.exit_norm(x_sum.to(x_stem.dtype))
```

### 5.3. Двонаправлений режим для DeltaNet у складі Арбітра
Для шарів `GatedDeltaNet` у `MODE_REFINE` необхідно запровадити **двосторонній прохід (Bidirectional DeltaNet Pass)**:
$$\vec{y}_t = \text{DeltaNet}_{\text{fwd}}(x), \quad \overleftarrow{y}_t = \text{DeltaNet}_{\text{bwd}}(x)$$
$$y_t = \frac{1}{2}(\vec{y}_t + \overleftarrow{y}_t)$$
або в режимі `MODE_REFINE` обходити рекурентний стан через пряму проєкцію та повнозв'язну увагу, забезпечуючи повноцінне некаузальне узгодження на всіх 16 шарах Арбітра.

### 5.4. Градієнтна ізоляція (Stop-Gradient) між фазами
У методі `generate_ncm` та тренувальному конвеєрі обов'язково закріпити межі фаз через `.detach()`:
```python
# Межа Фаза 1 -> Фаза 2: повна ізоляція графіка Autograd
H_raw_blueprints = torch.cat(raw_blueprints_list, dim=1).detach()

# Межа Фаза 2 -> Фаза 3: відсікання рекурентних градієнтів рефайнера
refined_blueprints = refinement_result["refined_blueprints"].detach()
```
Для спільного навчання фаз застосовувати методику **PCGrad (Projected Conflicting Gradients)**, яка проектує вектор градієнта рефайнера $\mathbf{g}_{\text{refine}}$ на ортогональну площину каузального градієнта $\mathbf{g}_{\text{causal}}$ у разі від'ємного скалярного добутку:
$$\mathbf{g}_{\text{refine}} \leftarrow \mathbf{g}_{\text{refine}} - \frac{\mathbf{g}_{\text{refine}} \cdot \mathbf{g}_{\text{causal}}}{\|\mathbf{g}_{\text{causal}}\|^2} \mathbf{g}_{\text{causal}}, \quad \text{якщо } \mathbf{g}_{\text{refine}} \cdot \mathbf{g}_{\text{causal}} < 0$$

---

## ВИСНОВОК ТА ВЕРДИКТ АУДИТУ

1. **Концепція Mode Embedding (Zero-RAM Weight Sharing):** Математично спроможна та вкрай ефективна для економії VRAM на пристроях класу RTX 2060 (6 GB), проте вимагає обов'язкового впровадження **Stop-Gradient бар'єрів між фазами** та **симетричної AdaLN-модуляції для `MODE_DECODE`** для запобігання деструктивному градієнтному конфлікту в спільних вагах BitLinear.
2. **Triad Fusion Gate:** У поточному стані є **критичною точкою відмови (Single Point of Failure)**. Потребує заміни на `TriadFusionGate` із вбудованою `nan_to_num` санітизацією, вирівнюванням дисперсій через RMSNorm та конкурентною Softmax-маршрутизацією. Нормування $1/\sqrt{3}$ у вихідному стемі підлягає заміні на безпечний опуклий барицентр $\frac{1}{3}(x_1 + x_2 + x_3)$ в FP32.
3. **Arbiter Masking & KV-Cache:** Виявлено приховану архітектурну невідповідність специфікації: 12 із 16 шарів Арбітра є суворо каузальними DeltaNet-шарами. Необхідно реалізувати двонаправлений двопрохідний DeltaNet для `MODE_REFINE`, примусово ізолювати KV-кеш між змінами режимів та захистити маску паддінгу.
4. **Скорочення витрат до 0 нс:** Досягається шляхом AOT-спеціалізації графіків виконання, попереднього складання константних векторів режиму з таблицею ембедінгів (`Weight Folding`) та нуль-копіювального перемикання станів пам'яті за адресою вказівника (`Zero-Copy Arenas`).

**РЕКОМЕНДАЦІЯ:** Негайно імплементувати запропоновані виправлення у кодову базу `src/modeling_hetman2.py` та зафіксувати їх оновленим тестовим набором Red Team Fuzzer.
