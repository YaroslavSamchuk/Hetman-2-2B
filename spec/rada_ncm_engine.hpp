// =================================================================================================
// Файл: rada_ncm_engine.hpp
// Проєкт: Rada.cpp Edge Inference Engine (Hetman-2-NCM 2.0B Architecture)
// Ліцензія: Apache License 2.0
// Стандарт мови: ISO/IEC C++20
// Опис: Високоефективний інференс-рушій без динамічних алокацій (Zero-Allocation Runtime)
// =================================================================================================

#pragma once

#include <cstdint>
#include <cstddef>
#include <cmath>
#include <bit>
#include <span>
#include <array>
#include <string_view>
#include <functional>
#include <concepts>
#include <type_traits>

namespace rada::ncm {

// =================================================================================================
// АРХІТЕКТУРНІ КОНСТАНТИ МОДЕЛІ HETMAN-2-NCM 2.0B
// =================================================================================================

inline constexpr uint32_t RADA_NCM_MAGIC               = 0x4D434E52; // "RNCM" у little-endian
inline constexpr uint32_t RADA_NCM_VERSION             = 2;
inline constexpr size_t   RADA_NCM_CACHE_LINE          = 64;         // 64-байтне вирівнювання

inline constexpr uint32_t VOCAB_SIZE                   = 65536;      // Кириличний BPE словник
inline constexpr uint32_t HIDDEN_SIZE                  = 1536;       // Розмірність прихованого вектора d
inline constexpr uint32_t INTERMEDIATE_SIZE            = 6144;       // SwiGLU FFN d_ffn
inline constexpr uint32_t NUM_TOTAL_LAYERS             = 34;         // 8 Stem + 6 Early + 16 Arbiter + 4 Exit
inline constexpr uint32_t NUM_DELTANET_LAYERS          = 28;         // Рекурентні лінійні шари DeltaNet
inline constexpr uint32_t NUM_GLOBAL_ATTN_LAYERS       = 6;          // Глобальні шари Softmax Attention
inline constexpr uint32_t DELTANET_HEADS               = 16;         // Кількість голів у DeltaNet
inline constexpr uint32_t DELTANET_HEAD_DIM            = 64;         // Розмірність голови DeltaNet (d_k = d_v = 64)
inline constexpr uint32_t GLOBAL_HEADS_Q               = 12;         // Кількість голів запиту QK-Norm Softmax
inline constexpr uint32_t GLOBAL_HEADS_KV              = 2;          // GQA голови ключів та значень
inline constexpr uint32_t GLOBAL_HEAD_DIM              = 128;        // Розмірність глобальної голови
inline constexpr uint32_t HOPFIELD_GROUPS              = 16;         // SLH-Core v2 групи
inline constexpr uint32_t HOPFIELD_CODEBOOK_SIZE       = 1024;       // Розмірність кодбуків C1, C2
inline constexpr uint32_t HOPFIELD_VAL_DIM             = 32;         // Довжина тернарного вектора пам'яті
inline constexpr uint32_t HOPFIELD_EXPAND_DIM          = 128;        // Лінійне розширення

inline constexpr uint32_t NCM_TOKENS_PER_BLOCK         = 400;        // Кількість токенів в 1 макро-блоці
inline constexpr uint32_t NCM_MAX_BLOCKS               = 8;          // Максимальна кількість блоків у кресленні
inline constexpr uint32_t NCM_LATENTS_PER_BLOCK        = 16;         // Концептуальних токенів на 1 блок
inline constexpr uint32_t NCM_TOTAL_BLUEPRINT_LATENTS  = NCM_MAX_BLOCKS * NCM_LATENTS_PER_BLOCK; // 128

// =================================================================================================
// ТОЧНІ БЮДЖЕТИ ПАМ'ЯТІ (У БАЙТАХ)
// =================================================================================================

inline constexpr size_t BYTES_STATIC_TERNARY_WEIGHTS   = 508559360ULL; // 485.00 MiB (BitNet b1.58 2-bit)
inline constexpr size_t BYTES_STATIC_EMBEDDINGS        = 201326592ULL; // 192.00 MiB (BF16 Tied Head)
inline constexpr size_t BYTES_STATIC_HOPFIELD_AUX      = 2228288ULL;   // 2.125 MiB (Codebooks + Exp)
inline constexpr size_t BYTES_DELTANET_STATE           = 7340032ULL;   // 7.00 MiB (28 шарів x 16 x 64 x 64 x 4B)
inline constexpr size_t BYTES_PHASE1_DRAFT_SCRATCH     = 4194304ULL;   // 4.00 MiB (Буфер Фази 1)
inline constexpr size_t BYTES_PHASE2_REFINE_SCRATCH    = 8388608ULL;   // 8.00 MiB (Буфер Фази 2)
inline constexpr size_t BYTES_PHASE3_DECODE_WORKSPACE  = 13417216ULL;  // 12.80 MiB (Буфер Фази 3 на 400 токенів)
inline constexpr size_t BYTES_GLOBAL_KV_CACHE          = 4915200ULL;   // 4.69 MiB (FP8 кеш на 1600 токенів)
inline constexpr size_t BYTES_METADATA_AND_BPE         = 2162688ULL;   // 2.06 MiB (Словник та дескриптори)

// Повний розмір монолітної пре-алокованої арени
inline constexpr size_t BYTES_TOTAL_MONOLITHIC_ARENA   = 752532288ULL; // 717.67 MiB (< 750 MiB!)

// =================================================================================================
// ПЕРЕЛІКИ ТА КОДИ СТАТУСІВ
// =================================================================================================

enum class RadaStatus : int32_t {
    SUCCESS                 = 0,
    ERROR_INVALID_ARGUMENT  = -1,
    ERROR_OUT_OF_MEMORY     = -2,
    ERROR_CORRUPTED_MODEL   = -3,
    ERROR_CHECKSUM_MISMATCH = -4,
    ERROR_UNSUPPORTED_ISA   = -5,
    ERROR_EXECUTION_ABORTED = -6
};

enum class BackendISA : uint32_t {
    AUTO_DETECT = 0,
    X86_AVX2_VNNI,
    X86_AVX512_VNNI,
    ARM_NEON_DOTPROD,
    CUDA_DP4A
};

// =================================================================================================
// СТРУКТУРИ ДАНИХ КОНВЕЄРА NCM (ВИРІВНЮВАННЯ НА 64 БАЙТИ)
// =================================================================================================

#pragma pack(push, 1)

// Заголовок бінарного формату файлу моделі .rncm
struct alignas(RADA_NCM_CACHE_LINE) RadaNcmFileHeader {
    uint32_t magic;                     // RADA_NCM_MAGIC ("RNCM")
    uint32_t version;                   // RADA_NCM_VERSION (2)
    uint32_t vocab_size;                // 65536
    uint32_t hidden_size;               // 1536
    uint32_t intermediate_size;         // 6144
    uint32_t num_layers;                // 34
    uint32_t num_deltanet_layers;       // 28
    uint32_t num_global_layers;         // 6
    uint32_t deltanet_heads;            // 16
    uint32_t deltanet_head_dim;         // 64
    uint32_t global_heads_q;            // 12
    uint32_t global_heads_kv;           // 2
    uint32_t global_head_dim;           // 128
    uint32_t hopfield_groups;           // 16
    uint32_t hopfield_codebook_size;    // 1024
    uint32_t hopfield_val_dim;          // 32
    uint32_t tokens_per_block;          // 400
    uint32_t max_blocks;                // 8
    uint64_t tensor_table_offset;       // Зсув до таблиці дескрипторів тензорів
    uint64_t tensor_data_offset;        // Зсув до вирівняних ваг моделі
    uint64_t bpe_vocab_offset;          // Зсув до BPE словника
    uint64_t bpe_vocab_size_bytes;      // Розмір BPE словника
    uint32_t header_crc32;              // Контрольна сума CRC32 заголовка (байти 0..103)
    uint32_t table_crc32;               // Контрольна сума CRC32 таблиці дескрипторів
    uint32_t flags;                     // Прапорці бінарного контейнера (0x01 = CRC32)
    uint8_t  padding[12];               // Доповнення до 128 байтів (104 + 12 + 12 = 128)
};

static_assert(sizeof(RadaNcmFileHeader) == 128, "RadaNcmFileHeader must be exactly 128 bytes");

// Дескриптор одного тензора моделі (Рівно 128 байтів = 2 кеш-лінії)
struct alignas(RADA_NCM_CACHE_LINE) RadaTensorDescriptor {
    char     name[64];                  // Унікальний ідентифікатор тензора у state_dict
    uint32_t dtype;                     // 1=BitNet 2-bit, 2=BF16, 3=FP16, 4=FP32
    uint32_t n_dims;                    // Ранг тензора (кількість вимірів)
    uint64_t shape[4];                  // Геометрія тензора [d0, d1, d2, d3]
    uint64_t offset;                    // Зсув початку даних у файлі (кратно 64)
    uint64_t size_bytes;                // Фізичний розмір вирівняного блоку (кратно 64)
    float    scale_gamma;               // Absmean масштабуючий коефіцієнт BitNet
    uint32_t crc32;                     // Індивідуальний CRC32 блоку даних (раніше reserved)
};

static_assert(sizeof(RadaTensorDescriptor) == 128, "RadaTensorDescriptor must be exactly 128 bytes");

#pragma pack(pop)

// Структура одного макро-креслення (Blueprint Block)
struct alignas(RADA_NCM_CACHE_LINE) BlueprintBlock {
    alignas(RADA_NCM_CACHE_LINE) float latents[NCM_LATENTS_PER_BLOCK * HIDDEN_SIZE];
    uint32_t block_id;
    float    semantic_coherence_score;
    uint8_t  reserved[56]; // Вирівнювання структури до кратності 64 байтам
};

// Стан пам'яті Gated DeltaNet (рівно 7,340,032 байти)
struct alignas(RADA_NCM_CACHE_LINE) DeltaNetFullState {
    alignas(RADA_NCM_CACHE_LINE) float matrices[NUM_DELTANET_LAYERS][DELTANET_HEADS][DELTANET_HEAD_DIM][DELTANET_HEAD_DIM];

    void reset() noexcept {
        for (auto& layer : matrices) {
            for (auto& head : layer) {
                for (auto& row : head) {
                    for (auto& val : row) {
                        val = 0.0f;
                    }
                }
            }
        }
    }
};

static_assert(sizeof(DeltaNetFullState) == BYTES_DELTANET_STATE, 
              "DeltaNetFullState must match exactly 7,340,032 bytes!");

// Робочий буфер Фази 1: Креслення (Draft Blueprints)
struct alignas(RADA_NCM_CACHE_LINE) RadaDraftScratchpad {
    alignas(RADA_NCM_CACHE_LINE) uint8_t memory[BYTES_PHASE1_DRAFT_SCRATCH];
};

// Робочий буфер Фази 2: Рефайнер (Refine Blueprints)
struct alignas(RADA_NCM_CACHE_LINE) RadaRefineScratchpad {
    alignas(RADA_NCM_CACHE_LINE) uint8_t memory[BYTES_PHASE2_REFINE_SCRATCH];
};

// Робочий буфер Фази 3: Активний блок на 400 токенів INT8
struct alignas(RADA_NCM_CACHE_LINE) RadaBlockDecodeWorkspace {
    alignas(RADA_NCM_CACHE_LINE) uint8_t memory[BYTES_PHASE3_DECODE_WORKSPACE];
};

// Глобальний KV-кеш для 6 шарів Softmax (FP8 E4M3)
struct alignas(RADA_NCM_CACHE_LINE) RadaGlobalKVCache {
    uint32_t current_seq_len;
    uint32_t max_seq_len;
    uint8_t  reserved[56]; // 64-байтне вирівнювання службового заголовка
    alignas(RADA_NCM_CACHE_LINE) uint8_t buffer[BYTES_GLOBAL_KV_CACHE - 64];

    void reset() noexcept {
        current_seq_len = 0;
    }
};

static_assert(sizeof(RadaGlobalKVCache) == BYTES_GLOBAL_KV_CACHE,
              "RadaGlobalKVCache must exactly match BYTES_GLOBAL_KV_CACHE!");

// =================================================================================================
// ВАРІАНТ В: ГІБРИДНИЙ ДУАЛЬНИЙ КОНТУР МИСЛЕННЯ (HYBRID DUAL-LOOP DELIBERATION)
// =================================================================================================

enum class CognitiveLevel : uint32_t {
    AUTO    = 0,
    SIMPLE  = 1, // 2 проходи Фази 2, 0 аварійних проходів (латентність ~20 мс)
    MEDIUM  = 2, // 4 проходи Фази 2, 1 аварійний прохід (латентність ~50 мс)
    COMPLEX = 3  // 8..32 проходів з адаптивною зупинкою Банаха (латентність ~200-350 мс)
};

struct alignas(RADA_NCM_CACHE_LINE) DeliberationConfig {
    bool           enable_deliberation  = true;
    bool           explain_thought      = false;
    bool           think                = false;
    CognitiveLevel forced_level         = CognitiveLevel::AUTO;
    uint32_t       max_refine_passes    = 32;
    float          banach_epsilon       = 1e-4f;
    const bool*    early_stop_signal    = nullptr; // Атомарний прапорець раннього переривання (<15 мс)
};

struct DeliberationTelemetry {
    uint32_t cognitive_level_assigned = 0;
    float    complexity_score         = 0.0f;
    uint32_t passes_executed          = 0;
    float    final_cauchy_ratio       = 0.0f;
    bool     banach_converged         = false;
    bool     early_stopped            = false;
};

// Параметри запуску генерації
struct GenerationConfig {
    float              temperature   = 0.7f;
    float              top_p         = 0.9f;
    uint32_t           target_tokens = 1600; // Типовий запит: 4 блоки по 400 токенів
    bool               stream        = true;
    DeliberationConfig deliberation{};       // Конфігурація мислення (Варіант В)
};

// Телеметрія виконання
struct ExecutionTelemetry {
    double                time_phase1_ms       = 0.0;
    double                time_phase2_ms       = 0.0;
    double                time_phase3_ms       = 0.0;
    double                total_time_sec       = 0.0;
    double                tokens_per_second    = 0.0;
    size_t                peak_ram_bytes       = 0;
    bool                  all_allocations_zero = true;
    DeliberationTelemetry deliberation_telem{}; // Звіт про збіжність Банаха та рівень складності
};

// =================================================================================================
// СИГНАТУРИ ОПЕРАТОРІВ NCM-КОНВЕЄРА
// =================================================================================================

struct RadaExecutionContext {
    const uint8_t* weights_ternary_base;
    const uint8_t* embeddings_bf16_base;
    const uint8_t* hopfield_aux_base;
    BackendISA     isa;
};

[[nodiscard]] RadaStatus rada_draft_blueprints(
    const RadaExecutionContext& ctx,
    std::span<const uint32_t>   prompt_token_ids,
    std::span<BlueprintBlock, 8> out_blueprints,
    RadaDraftScratchpad&        scratchpad
) noexcept;

[[nodiscard]] RadaStatus rada_refine_blueprints(
    const RadaExecutionContext&     ctx,
    std::span<BlueprintBlock, 8>    io_blueprints,
    std::span<const float, 7340032> deltanet_state,
    RadaRefineScratchpad&           scratchpad
) noexcept;

[[nodiscard]] RadaStatus rada_decode_block_ncm(
    const RadaExecutionContext& ctx,
    const BlueprintBlock&       blueprint,
    size_t                      block_index,
    std::span<uint32_t, 400>    out_tokens,
    RadaBlockDecodeWorkspace&   workspace,
    std::function<void(uint32_t token_id)> token_stream_callback = nullptr
) noexcept;

[[gnu::always_inline]] inline void rada_deltanet_step(
    float* __restrict__       s_matrix,      // [16, 64, 64] -> стан шару (256 KiB)
    const int8_t* __restrict__ q_vec,        // [16, 64]     -> квантований запит
    const int8_t* __restrict__ k_vec,        // [16, 64]     -> квантований ключ
    const int8_t* __restrict__ v_vec,        // [16, 64]     -> квантоване значення
    const float* __restrict__  alpha_gate,   // [16]         -> гейт затухання
    const float* __restrict__  beta_rate,    // [16]         -> темп асоціативного навчання
    float* __restrict__        y_out,        // [16, 64]     -> вихідний вектор уваги
    float                      scale_q,
    float                      scale_k,
    float                      scale_v
) noexcept;

// =================================================================================================
// ШВИДКА ЕКСПОНЕНТА ДЛЯ SOFTMAX: ПОЛІНОМ РЕМЕЗА 5-ГО ПОРЯДКУ (ЧИСЕЛЬНО БЕЗПЕЧНА)
// =================================================================================================

struct RemezExp5 {
    // Коефіцієнти мінімаксного полінома Ремеза 5-го порядку для 2^f на [0, 1)
    static constexpr float C0 = 1.0000000000f;
    static constexpr float C1 = 0.6931471806f;
    static constexpr float C2 = 0.2402265070f;
    static constexpr float C3 = 0.0555041087f;
    static constexpr float C4 = 0.0096181291f;
    static constexpr float C5 = 0.0013333558f;

    [[nodiscard]] static inline float eval(float x) noexcept {
        // Чисельний захист від NaN та запобігання підпороговому переповненню в -inf / від'ємні числа
        // При x <= -87.33f значення менше за FLT_MIN (~1.17e-38); у Softmax logit diff <-50.0f
        // робить внесок < 1.93e-22, що є нехтовним порівняно з машинною точністю float32 (1.19e-7).
        if (std::isnan(x) || x <= -87.3365f) return 0.0f;
        if (x >= 87.3365f) return 3.40282347e+38f; // FLT_MAX

        float t = x * 1.4426950408889634f; // x * log2(e)
        float fi = std::floor(t);
        float f = t - fi;
        int i = static_cast<int>(fi);

        // Схема Горнера для 5-го порядку (максимальна абсолютна похибка eps < 1.2e-7)
        float p = C0 + f * (C1 + f * (C2 + f * (C3 + f * (C4 + f * C5))));

        // Безпечний бітовий зсув експоненти IEEE-754 (без UB та без strict aliasing порушень)
        uint32_t exp_bits = static_cast<uint32_t>(i + 127) << 23;
        #if defined(__cpp_lib_bit_cast) && __cpp_lib_bit_cast >= 201806L
        float scale = std::bit_cast<float>(exp_bits);
        #else
        union { uint32_t u; float f; } cvt;
        cvt.u = exp_bits;
        float scale = cvt.f;
        #endif

        return p * scale;
    }
};

// =================================================================================================
// ГОЛОВНИЙ КЛАС ІНФЕРЕНС-РУШІЯ RADA.CPP NCM
// =================================================================================================

class alignas(RADA_NCM_CACHE_LINE) RadaNcmEngine {
public:
    RadaNcmEngine() noexcept;
    ~RadaNcmEngine() noexcept;

    RadaNcmEngine(RadaNcmEngine&& other) noexcept;
    RadaNcmEngine& operator=(RadaNcmEngine&& other) noexcept;
    RadaNcmEngine(const RadaNcmEngine&) = delete;
    RadaNcmEngine& operator=(const RadaNcmEngine&) = delete;

    [[nodiscard]] RadaStatus init(std::string_view model_path, BackendISA preferred_isa = BackendISA::AUTO_DETECT) noexcept;
    void reset_state() noexcept;

    [[nodiscard]] RadaStatus generate(
        std::span<const uint32_t> prompt_token_ids,
        const GenerationConfig&   config,
        std::span<uint32_t>       out_generated_tokens,
        size_t&                   out_generated_count,
        std::function<void(uint32_t token_id)> stream_callback = nullptr
    ) noexcept;

    [[nodiscard]] ExecutionTelemetry get_telemetry() const noexcept;
    [[nodiscard]] bool is_initialized() const noexcept;

private:
    uint8_t* monolithic_arena_base_ = nullptr;
    size_t   monolithic_arena_size_ = 0;

    uint8_t*                  weights_ternary_ptr_   = nullptr;
    uint8_t*                  embeddings_bf16_ptr_   = nullptr;
    uint8_t*                  hopfield_aux_ptr_      = nullptr;
    DeltaNetFullState*        deltanet_state_        = nullptr;
    RadaDraftScratchpad*      draft_scratchpad_      = nullptr;
    RadaRefineScratchpad*     refine_scratchpad_     = nullptr;
    RadaBlockDecodeWorkspace* decode_workspace_      = nullptr;
    RadaGlobalKVCache*        global_kv_cache_       = nullptr;

    RadaNcmFileHeader         header_{};
    BackendISA                active_isa_            = BackendISA::AUTO_DETECT;
    ExecutionTelemetry        last_telemetry_{};
    bool                      initialized_           = false;

    [[nodiscard]] RadaStatus execute_phase1_draft(
        std::span<const uint32_t>    prompt_tokens,
        std::span<BlueprintBlock, 8> out_blueprints
    ) noexcept;

    [[nodiscard]] RadaStatus execute_phase2_refine(
        std::span<BlueprintBlock, 8> io_blueprints
    ) noexcept;

    [[nodiscard]] RadaStatus execute_phase3_decode_block(
        const BlueprintBlock&    blueprint,
        size_t                   block_idx,
        std::span<uint32_t, 400> out_block_tokens,
        std::function<void(uint32_t)> stream_cb
    ) noexcept;
};

// =================================================================================================
// СИГНАТУРИ ЕКСПОРТОВАНИХ С-СУМІСНИХ ФУНКЦІЙ ДЛЯ FFI (PYTHON / RUST / C#)
// =================================================================================================

extern "C" {

[[nodiscard]] RadaStatus rada_ncm_create_engine(
    const char* model_path,
    RadaNcmEngine** out_engine_handle
);

void rada_ncm_destroy_engine(
    RadaNcmEngine* engine_handle
);

[[nodiscard]] RadaStatus rada_ncm_generate_stream(
    RadaNcmEngine* engine_handle,
    const uint32_t* prompt_tokens,
    size_t prompt_tokens_len,
    const GenerationConfig* config,
    void (*token_callback)(uint32_t token_id, void* user_data),
    void* user_data
);

[[nodiscard]] size_t rada_ncm_get_peak_ram_usage(
    const RadaNcmEngine* engine_handle
);

} // extern "C"

} // namespace rada::ncm
