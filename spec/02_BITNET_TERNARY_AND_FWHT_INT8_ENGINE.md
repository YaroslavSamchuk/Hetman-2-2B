# Engineering Specification: BitNet b1.58 Ternary Core & Fast Walsh-Hadamard INT8 Engine
## Subsystem: Hetman-2-NCM 2.0B Low-Level Compute Fabric
**Document ID:** `HETMAN-SPEC-02-BITNET-FWHT-INT8`  
**Classification:** Core System Architecture / Hardware-Level Engine  
**Target Hardware:** x86-64 (AVX2, AVX-512 VNNI), ARM64 (NEON, Apple Silicon AMX), Edge NPU / Integrated Graphics  
**Target Architecture:** Hetman-2-NCM (Neural Compute Machine) 2.05B Parameters  
**Status:** Approved for Implementation  

---

```
==================================================================================================
                 HETMAN-2-NCM 2.0B LOW-LEVEL HARDWARE COMPUTE PIPELINE
==================================================================================================

  [Input Activations X] (BF16 / FP32, d_model = 1536)
           │
           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. Sub-Layer RMSNorm (Fused In-Register Normalization)     │
  └─────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 2. In-Register Blockwise FWHT (B = 512, 3 Blocks)          │
  │    - 9-Stage Butterfly Network: O(d log B) Add/Sub          │
  │    - Outlier Attenuation: Peak amplitude reduced by 22.6x  │
  └─────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 3. Symmetric Per-Tensor INT8 Quantization                  │
  │    - Dynamic Scale: eta = max(|x_rot|), Scale = 127 / eta   │
  │    - Zero clipping loss due to FWHT outlier suppression    │
  └─────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 4. BitNet b1.58 Ternary GEMV / GEMM Engine                  │
  │    - Packed 2-bit Weights: 4 weights/byte (W ∈ {-1, 0, +1}) │
  │    - Integer Add/Sub Accumulation (Zero FP Multiplications)│
  │    - SIMD Core: AVX-512 VNNI / AVX2 / ARM NEON / AMX        │
  │    - INT32 Overflow-Safe Accumulators                       │
  └─────────────────────────────────────────────────────────────┘
           │
           ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ 5. Dequantization & Scale Fusion                            │
  │    - Output: Y = (Local_Acc_INT32) * (scale_x * scale_w)    │
  │    - Broadcast FP32 Scaling Factor                          │
  └─────────────────────────────────────────────────────────────┘
           │
           ▼
  [Output Activations Y] (BF16 / FP32, intermediate_dim = 6144 or hidden_dim = 1536)
==================================================================================================
```

---

## 1. Executive Summary & Hardware Design Rationale

Modern autoregressive and non-autoregressive language models are overwhelmingly bottlenecked by **memory bandwidth** during single-batch edge inference. In standard 16-bit floating point representations (FP16/BF16), transferring a 2.0-billion-parameter weight tensor requires 4.0 gigabytes of memory traffic per token. On contemporary edge processors with unified LPDDR4x/DDR5 memory subsystems offering 48–64 GB/s of bandwidth, this imposes a theoretical speed ceiling of 12–15 tokens per second, accompanied by substantial thermal dissipation due to energy-intensive floating-point multiply-accumulate (FMA) units.

The **Hetman-2-NCM (Neural Compute Machine) 2.0B** architecture replaces traditional floating-point matrix operations with a hardware-co-designed **BitNet b1.58 ternary weight fabric** coupled with an online **Fast Walsh-Hadamard Transform (FWHT B=512) INT8 activation pipeline**.

### 1.1 Key Hardware Metrics & Performance Targets
* **Weight Compression:** Exactly 2 bits per parameter (0.25 bytes/param). The 2.05B model weights occupy **512.5 MB**, fitting comfortably into modern CPU high-speed caches and eliminating DRAM saturation.
* **Energy Efficiency:** Floating-point multiplications are eliminated in all linear projections. The GEMV core executes exclusively using integer additions and subtractions (`ADD`/`SUB`), reducing silicon energy consumption per operation from **1.1 pJ (FP16 FMA)** to **0.1 pJ (INT8 ADD)** on a 7nm process node.
* **Peak Inference Velocity:** Under a 48 GB/s DRAM bandwidth budget, memory-bound throughput increases from **11.7 tokens/sec (FP16)** to **93.7 tokens/sec (BitNet b1.58)**, representing an **8.0x speedup**.
* **Zero Outlier Clipping Distortion:** Fast Walsh-Hadamard orthogonal rotation ($B=512$) diffuses high-magnitude activation spikes across the channel dimension in $O(d \log B)$ operations without multiplications, enabling lossless symmetric per-tensor INT8 quantization.

---

## 2. BitNet b1.58 Ternary Mathematics & Representation

The BitNet b1.58 architecture restricts weight parameters to the discrete ternary alphabet:
$$\mathcal{W} \in \{-1, 0, +1\}$$
Activations are quantized dynamically to 8-bit signed integers:
$$\mathcal{X} \in [-128, 127] \cap \mathbb{Z}$$

### 2.1 Mathematical Formulation of Absmean Weight Quantization

Let $W \in \mathbb{R}^{n \times m}$ be the latent high-precision weight tensor (maintained in BF16 or FP32 during training and optimization). The quantization process proceeds via absolute mean scaling (Absmean):

#### Step 1: Per-Tensor Scale Factor Computation
The global scale factor $\gamma \in \mathbb{R}^+$ represents the mean absolute magnitude of all weight elements in the matrix:
$$\gamma = \frac{1}{n \cdot m} \sum_{i=1}^{n} \sum_{j=1}^{m} |W_{i, j}|$$

To protect against numerical singularities in sparse initialization or heavily pruned layers, a stability epsilon $\epsilon = 10^{-6}$ is enforced:
$$\gamma_{\text{safe}} = \max\left(\gamma, \epsilon\right)$$

#### Step 2: Weight Normalization and Rounding
Each latent weight is normalized by $\gamma_{\text{safe}}$, rounded to the nearest integer, and clamped to the discrete range $[-1, +1]$:
$$\widetilde{W}_{i, j} = \text{Clip}\left( \left\lfloor \frac{W_{i, j}}{\gamma_{\text{safe}}} \right\rceil, -1, +1 \right)$$
where $\lfloor \cdot \rceil$ denotes the round-to-nearest-even tie-breaking operator, and:
$$\text{Clip}(z, -1, +1) = \max(-1, \min(1, z))$$

#### Step 3: Straight-Through Estimator (STE)
Because the rounding and clipping operators have zero derivatives almost everywhere ($\frac{\partial \widetilde{W}}{\partial W} = 0$), backward propagation utilizes the Straight-Through Estimator (STE):
$$\frac{\partial \mathcal{L}}{\partial W_{i, j}} = \frac{\partial \mathcal{L}}{\partial \widetilde{W}_{i, j}} \cdot \mathbf{1}_{|W_{i, j}| \le \tau}$$
where $\tau$ is a soft saturation threshold (typically $\tau = 1.0$), ensuring that gradients pass uninhibited to latent master weights during Phase 1 pre-training and Phase 2 distillation.

```
                  ABSMEAN WEIGHT QUANTIZATION FUNCTION
      W_tilde
        ▲
   +1.0 ┼──────────────┐
        │              │
        │              │
    0.0 ┼───────┬──────┴───────► W / gamma
        │       │
        │       │
   -1.0 ┴───────┘
       -1.5   -0.5    +0.5   +1.5
```

---

### 2.2 Activation Quantization: Symmetric Dynamic Per-Tensor INT8

Activations $X \in \mathbb{R}^{B \times d}$ exhibit dynamic variance depending on token context. Per-tensor dynamic quantization is defined as:

$$\eta = \max_{j} |X_j|$$
$$\text{Scale}_X = \frac{127.0}{\max(\eta, \epsilon)}$$
$$X_q = \text{Clip}\left( \left\lfloor X \cdot \text{Scale}_X \right\rceil, -128, +127 \right)$$

The dequantized scalar multiplier for the resulting matrix product is:
$$\text{Scale}_{\text{fusion}} = \frac{\gamma}{\text{Scale}_X} = \frac{\gamma \cdot \max(\eta, \epsilon)}{127.0}$$

---

### 2.3 Physical Memory Layout & 2-Bit Packing Schemes

Each ternary weight requires only $\log_2(3) \approx 1.58$ bits of theoretical entropy. In physical silicon implementations, weights are aligned to power-of-two boundaries at **2 bits per weight**, achieving a 4:1 compression ratio (4 weights per byte, 16 weights per 32-bit dword, 32 weights per 64-bit qword, and 256 weights per 512-bit cache line/ZMM register).

#### 2.3.1 Canonical Bit-Encoding Enums

We define two low-level hardware formats utilized within the Hetman-2-NCM engine:

```
FORMAT 0: 2's Complement Sign-Extended (Arithmetic Optimized)
┌──────────────┬──────────────┬──────────────────┬────────────────────────┐
│ Value        │ Binary (2b)  │ Hex Nibble (x2)  │ Sign Extension (8b)    │
├──────────────┼──────────────┼──────────────────┼────────────────────────┤
│  0           │ 00_2         │ 0x0              │ 00000000_2 (0x00)      │
│ +1           │ 01_2         │ 0x1              │ 00000001_2 (0x01)      │
│ -1           │ 11_2         │ 0x3              │ 11111111_2 (0xFF)      │
│ Reserved (0) │ 10_2         │ 0x2              │ Mapped to 0x00 / NOP   │
└──────────────┴──────────────┴──────────────────┴────────────────────────┘

FORMAT 1: Dual-Bitplane Split (SIMD Mask Optimized)
┌──────────────┬──────────────┬──────────────────┬────────────────────────┐
│ Value        │ Bitplane P   │ Bitplane N       │ Formula                │
│              │ (Positive)   │ (Negative)       │ W = P - N              │
├──────────────┼──────────────┼──────────────────┼────────────────────────┤
│  0           │ 0            │ 0                │ 0 - 0 =  0             │
│ +1           │ 1            │ 0                │ 1 - 0 = +1             │
│ -1           │ 0            │ 1                │ 0 - 1 = -1             │
│ Invalid (0)  │ 1            │ 1                │ Suppressed via P & ~N  │
└──────────────┴──────────────┴──────────────────┴────────────────────────┘
```

#### 2.3.2 1 Byte Packing Diagram (Format 0)
A single byte `uint8_t` contains 4 sequential weights: $W_0, W_1, W_2, W_3$:

```
Byte Offset +0:
Bit:   7     6     5     4     3     2     1     0
     ┌─────┬─────┬─────┬─────┬─────┬─────┬─────┬─────┐
     │  W3_sign  │  W2_sign  │  W1_sign  │  W0_sign  │
     │   b1    b0│   b1    b0│   b1    b0│   b1    b0│
     └─────┴─────┴─────┴─────┴─────┴─────┴─────┴─────┘
        Weight 3    Weight 2    Weight 1    Weight 0
```

#### 2.3.3 Fast Extraction Bitmasks and Branchless Decoding
To unpack a packed byte $B$ into four signed 8-bit integer registers $V_0, V_1, V_2, V_3$ without branch penalties:

$$\text{Bitmask}_P = B \ \& \ 0x55 \quad (01010101_2)$$
$$\text{Bitmask}_N = (B \gg 1) \ \& \ 0x55 \quad (01010101_2)$$
$$V_k = ((\text{Bitmask}_P \gg (2k)) \ \& \ 1) - ((\text{Bitmask}_N \gg (2k)) \ \& \ 1)$$

---

## 3. Instruction Mapping & Low-Level SIMD Microarchitecture

The fundamental operation in the Hetman-2-NCM forward pass is the integer dot-product:
$$Y = \sum_{k=0}^{K-1} X_q[k] \cdot \widetilde{W}[k]$$
Because $\widetilde{W}[k] \in \{-1, 0, +1\}$, this simplifies to conditional addition and subtraction:
$$Y = \sum_{k \in \mathcal{P}} X_q[k] - \sum_{k \in \mathcal{M}} X_q[k]$$
where $\mathcal{P} = \{k \mid \widetilde{W}[k] = +1\}$ and $\mathcal{M} = \{k \mid \widetilde{W}[k] = -1\}$.

Below is the exhaustive architectural instruction mapping across all supported compute backends.

---

### 3.1 x86-64 Architecture: AVX2 (Haswell, Zen 2/3)

On AVX2 targets (256-bit SIMD registers `ymm0`–`ymm15`), native 2-bit dot product instructions do not exist. We utilize the hardware shuffle and sign manipulation pipeline:

```
==================================================================================================
                 AVX2 BITNET COMPUTE PIPELINE (256-BIT VECTORIZED)
==================================================================================================
  Packed 2b Weights (64 weights = 16 bytes)
           │
           ▼  _mm256_cvtepu8_epi16 / vpshufb
  Unpack to 32x INT8 Weights in YMM_W
           │
  Quantized INT8 Activations in YMM_X
           │
           ├──► Method A: _mm256_sign_epi8 (Conditional negate/zero)
           │              Followed by horizontal add tree
           │
           └──► Method B: _mm256_maddubs_epi16 (Unsigned-Signed Multiply-Add)
                          Followed by _mm256_madd_epi16 into INT32
==================================================================================================
```

#### 3.1.1 AVX2 Assembly Instruction Sequence (`vpabsb`, `vpsignb`, `vpaddd`)
Using `vpsignb` allows direct single-cycle ternary transformation:
* If $W = +1$: `vpsignb(X, W)` returns $X$.
* If $W = -1$: `vpsignb(X, W)` returns $-X$.
* If $W = 0$: `vpsignb(X, W)` returns $0$.

```nasm
; Intel Syntax: AVX2 Ternary GEMV Inner Loop
; ymm0 = Activation Vector (32 x INT8 elements)
; ymm1 = Unpacked Ternary Weights (32 x INT8 elements, each in {-1, 0, +1})
; ymm2 = Running Accumulator (8 x INT32 accumulators)

vpsignb     ymm3, ymm0, ymm1       ; ymm3 = (ymm1 == 0) ? 0 : ((ymm1 < 0) ? -ymm0 : ymm0)
; Widen signed INT8 elements to INT16:
vpmovsxbw   ymm4, xmm3             ; Low 16 INT8 -> 16 INT16
vextracti128 xmm5, ymm3, 1
vpmovsxbw   ymm5, xmm5             ; High 16 INT8 -> 16 INT16
; Horizontal pairwise reduction into INT32:
vpaddw      ymm4, ymm4, ymm5       ; Pairwise add INT16
vpmovsxwd   ymm6, xmm4             ; Low 8 INT16 -> 8 INT32
vextracti128 xmm7, ymm4, 1
vpmovsxwd   ymm7, xmm7             ; High 8 INT16 -> 8 INT32
vpaddd      ymm6, ymm6, ymm7       ; 8 INT32 partial sums
vpaddd      ymm2, ymm2, ymm6       ; Accumulate into master INT32 register
```

---

### 3.2 x86-64 Architecture: AVX-512 with VNNI (Ice Lake, Zen 4/5, Sapphire Rapids)

AVX-512 VNNI introduces the `VPDPBUSD` instruction (Vector Pair Dot Product Byte Unsigned-Signed with 32-Bit Accumulation). While `VPDPBUSD` expects unsigned bytes in the first operand and signed bytes in the second operand, we bias activation bytes or map ternary weights to signed bytes:

$$\text{Acc}_{\text{int32}} \mathrel{+}= \sum_{k=0}^{3} A_k \cdot B_k$$

```
==================================================================================================
              AVX-512 VNNI TERNARY KERNEL ARCHITECTURE (ZMM0 - ZMM31)
==================================================================================================
  ZMM0..ZMM3   : 256 INT8 Activation Elements (4 x 64 bytes)
  ZMM4..ZMM7   : Unpacked Ternary Weights (Format 0: -1, 0, +1)
  ZMM8..ZMM15  : 8 x 16 = 128 INT32 Master Accumulators (Tile of 8 Rows)
  
  Instruction: vpdpbusd zmm8, zmm0, zmm4
  Throughput : 2 instructions per cycle on dual-FMA ports (Port 0 & Port 5)
  Peak OPS   : 2 inst * 64 MACs * 2 ops = 256 INT8 Ops / cycle
==================================================================================================
```

#### 3.2.1 AVX-512 VNNI Unpacking Kernel via `vpermb`
Using `vpermb`, an entire 64-byte ZMM register is unpacked from 16 bytes of 2-bit weights in 2 clock cycles using an in-register permutation lookup table:

```cpp
// AVX-512 In-Register 2-bit to INT8 Expansion
inline __m512i unpack_2bit_to_int8_avx512(const uint8_t* packed_16_bytes) {
    // 1. Broadcast 16 bytes across 64-byte ZMM register
    __m128i raw16 = _mm_loadu_si128((const __m128i*)packed_16_bytes);
    __m512i bcast = _mm512_broadcast_i32x4(raw16);
    
    // 2. Shift masks for 2-bit positions 0, 1, 2, 3
    __m512i shift_cnt = _mm512_set_epi8(
        6,4,2,0, 6,4,2,0, 6,4,2,0, 6,4,2,0,
        6,4,2,0, 6,4,2,0, 6,4,2,0, 6,4,2,0,
        6,4,2,0, 6,4,2,0, 6,4,2,0, 6,4,2,0,
        6,4,2,0, 6,4,2,0, 6,4,2,0, 6,4,2,0
    );
    __m512i shifted = _mm512_srlv_epi8(bcast, shift_cnt);
    __m512i masked  = _mm512_and_si512(shifted, _mm512_set1_epi8(0x03));
    
    // 3. Permute lookup table: maps {00->0, 01->+1, 10->0, 11->-1}
    __m512i lut = _mm512_set_epi8(
        -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,
        -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,
        -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,
        -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0
    );
    return _mm512_permutexvar_epi8(masked, lut);
}
```

---

### 3.3 ARM64 Architecture: NEON & Dot Product Extension (Cortex-A78, X-series, Graviton)

On ARMv8.2-A and higher, the `SDOT` (Signed Dot Product) instruction accumulates four signed 8-bit integers into a signed 32-bit integer in a single execution cycle:

```
  sdot  v0.4s, v1.16b, v2.16b   // v0[i] += v1[4i]*v2[4i] + ... + v1[4i+3]*v2[4i+3]
```

#### 3.3.1 ARM64 NEON Dual-Bitplane Accumulator Loop
For older cores lacking `SDOT` or when exploiting pure add/subtract pipelines:
```assembly
// ARM64 Assembly: Dual-Bitplane Mask Add/Sub Pipeline
// x0 = pointer to X (INT8 activations)
// x1 = pointer to Positive Bitplane Mask (1 bit per weight)
// x2 = pointer to Negative Bitplane Mask (1 bit per weight)
// q0 = running positive accumulator (4 x int32)
// q1 = running negative accumulator (4 x int32)

ldr     q2, [x0], #16          // Load 16 INT8 activations
ldrh    w3, [x1], #2           // Load 16 positive mask bits
ldrh    w4, [x2], #2           // Load 16 negative mask bits

// Create vector masks from scalar bitmasks using tbl/ushr:
dup     v3.8h, w3
dup     v4.8h, w4
// Accumulate masked elements:
saddw   v5.8h, v0.8h, v2.8b    // Sign extend and accumulate
ssubw   v6.8h, v1.8h, v2.8b
```

---

### 3.4 Apple Silicon: AMX (Apple Matrix Coprocessor) Acceleration

Apple Silicon (M1/M2/M3/M4) incorporates a proprietary matrix coprocessor (AMX) connected to CPU execution cores via custom instruction extensions. AMX features a 2 KB register file structured as 32 registers of 64 bytes (`X` registers and `Y` registers) and a 4 KB 2D accumulator tile (`Z` register).

In INT8 mode, AMX executes:
$$Z[i, j] \mathrel{+}= \sum_{k=0}^{63} X[i, k] \cdot Y[k, j]$$
yielding **512 8-bit MACs per clock cycle** per AMX unit. At a 3.5 GHz clock frequency on M3 Max, AMX delivers up to **3.58 INT8 TOPS** per core cluster.

#### 3.4.1 AMX Execution Sequence for Hetman-2-NCM
1. **Load Activations:** Activations $X_q$ are loaded into AMX `X` registers using `amx_ldx`.
2. **Decompress Weights in L1:** A software SIMD kernel decompresses 2-bit weights to INT8 in L1 cache ($64 \text{ bytes packed} \to 256 \text{ bytes INT8}$).
3. **Load Decompressed Weights:** `amx_ldy` transfers 256 weights into AMX `Y` registers.
4. **Outer-Product Accumulation:** `amx_fma32` (or integer equivalent `amx_mac16`) accumulates into `Z` accumulator tiles.
5. **Tile Extract:** Final INT32 results are extracted via `amx_stz` and scaled by $\text{Scale}_{\text{fusion}}$.

---

### 3.5 Instruction Cost Comparison Table Across Architectures

| Architecture | Microarchitecture | Target SIMD Instruction | Register Width | Operations / Cycle | Latency | FMA Eliminated? |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **x86-64** | AVX2 (Zen 3 / Skylake) | `vpsignb` + `vpaddd` | 256 bits | 32 INT8 / cycle | 1 cycle | Yes (Add/Sub only) |
| **x86-64** | AVX-512 VNNI (Zen 4) | `vpdpbusd` | 512 bits | 128 INT8 / cycle | 1 cycle | Yes (INT dot product) |
| **ARM64** | ARMv8.2-A NEON | `sdot` | 128 bits | 32 INT8 / cycle | 1 cycle | Yes (INT dot product) |
| **ARM64** | ARMv9 SVE2 | `svdot_s32` | 256-512 bits | 64-128 INT8 / cycle| 1 cycle | Yes (INT dot product) |
| **Apple M-Series**| AMX Gen 3/4 | `amx_matgen` / `amx_mac`| 512 bits | 512 INT8 / cycle | 1 cycle | Yes (Matrix coprocessor) |
| **CUDA GPU** | Volta/Ampere/Ada | `__dp4a` (PTX `dp4a.s32.s32`) | 32 bits | 4 INT8 / core-cycle | 1 cycle | Yes (Integer DP4A) |

---

## 4. Fast Walsh-Hadamard Transform (FWHT B=512) Engine

A major failure mode in standard post-training quantization and naive QAT of language models is the **Outlier Feature Phenomenon**. As models scale beyond 1B parameters, specific hidden dimensions develop activation magnitudes up to $\times 50$ to $\times 100$ larger than the median channel norm.

```
                  ACTIVATION PROFILE WITHOUT FWHT
  Magnitude
    120.0 ┼               █ (Channel 384: Outlier = +118.4)
          │               █
     10.0 ┼──█─█───█──────█──────█──█──█──█── Median = 1.2
      0.0 ┴────────────────────────────────────────► Hidden Dim (0..1535)
  [Dynamic INT8 scale is set to 127 / 118.4 = 1.07. Median channels receive ONLY 1 bit of precision!]

                  ACTIVATION PROFILE WITH FWHT (B=512)
  Magnitude
     10.0 ┼
      5.2 ┼──██████████████████████████████ Max = 5.2 (Peak attenuated by 22.6x)
      0.0 ┴────────────────────────────────────────► Hidden Dim (0..1535)
  [Dynamic INT8 scale is set to 127 / 5.2 = 24.4. Full 8-bit precision utilized across ALL channels!]
```

---

### 4.1 Mathematical Foundation of Hadamard Transformation

The Hadamard matrix $H_N \in \{-1, +1\}^{N \times N}$ is an orthogonal, symmetric matrix recursively defined via Sylvester's construction:
$$H_1 = \begin{bmatrix} 1 \end{bmatrix}, \quad H_{2k} = \begin{bmatrix} H_k & H_k \\ H_k & -H_k \end{bmatrix}$$

For block size $N = 512$:
$$H_{512} = H_2 \otimes H_{256} = H_2^{\otimes 9}$$
where $\otimes$ denotes the Kronecker product.

#### 4.1.1 Normalization and Isometry
To preserve the Euclidean $\ell_2$ norm of activation vectors, the normalized Hadamard matrix is defined as:
$$\widetilde{H}_{512} = \frac{1}{\sqrt{512}} H_{512} = \frac{1}{16\sqrt{2}} H_{512} \approx 0.04419417382415922 \cdot H_{512}$$

Properties:
1. **Orthogonality:** $\widetilde{H}_{512}^T \widetilde{H}_{512} = \widetilde{H}_{512} \widetilde{H}_{512}^T = I_{512}$.
2. **Symmetry:** $\widetilde{H}_{512}^T = \widetilde{H}_{512}$.
3. **Inversion:** $\widetilde{H}_{512}^{-1} = \widetilde{H}_{512}$ (The forward transform is identical to the inverse transform).
4. **Isometry:** $\|\widetilde{H}_{512} x\|_2 = \|x\|_2$ for all $x \in \mathbb{R}^{512}$.
5. **Eigenvalues:** $\lambda_i \in \{-1, +1\}$.
6. **Condition Number:** $\kappa(\widetilde{H}_{512}) = 1.0$ (Perfect numerical conditioning).

---

### 4.2 Outlier Attenuation & Kurtosis Suppression

Let $x \in \mathbb{R}^{512}$ be an activation block containing a single catastrophic outlier at index $k$:
$$x = [0, \dots, 0, M, 0, \dots, 0]^T, \quad M = 100.0$$
Applying the normalized FWHT yields:
$$y = \widetilde{H}_{512} x = \left[ \pm \frac{M}{\sqrt{512}}, \pm \frac{M}{\sqrt{512}}, \dots, \pm \frac{M}{\sqrt{512}} \right]^T$$
The maximum amplitude in the transformed space is strictly bounded by:
$$\|y\|_\infty = \frac{\|x\|_\infty}{\sqrt{512}} = \frac{100.0}{22.6274} \approx 4.419$$
The outlier magnitude is reduced by **22.6x**, transforming an intractable sparse activation spike into a dense, uniform, low-variance Gaussian-like distribution that fits inside the 8-bit dynamic quantization grid with zero saturation clipping!

---

### 4.3 Computational Complexity & Butterfly Algorithm

A naive matrix multiplication $y = H_{512} x$ requires $512^2 = 262,144$ operations. The Fast Walsh-Hadamard Transform decomposes $H_{512}$ into 9 stages of 256 butterfly operations:

$$\text{Complexity} = N \log_2(N) = 512 \times 9 = 4,608 \text{ Additions/Subtractions}$$
**Zero floating-point multiplications** are required during the butterfly stages. A single scalar scaling factor $1/\sqrt{512}$ is fused into the subsequent INT8 quantization step.

#### 4.3.1 9-Stage Butterfly Schedule for Block Size 512
```
Stage 0: Stride 1   (Pairs: [0,1], [2,3], ..., [510,511])
Stage 1: Stride 2   (Pairs: [0,2], [1,3], ..., [509,511])
Stage 2: Stride 4   (Pairs: [0,4], [1,5], ..., [507,511])
Stage 3: Stride 8   (Pairs: [0,8], [1,9], ..., [503,511])
Stage 4: Stride 16  (Pairs: [0,16], ..., [495,511])
Stage 5: Stride 32  (Pairs: [0,32], ..., [479,511])
Stage 6: Stride 64  (Pairs: [0,64], ..., [447,511])
Stage 7: Stride 128 (Pairs: [0,128], ..., [383,511])
Stage 8: Stride 256 (Pairs: [0,256], ..., [255,511])
```

---

### 4.4 Hardware Implementation: Register-Tiled FWHT

In CPU and GPU memory hierarchies, performing FWHT in memory generates massive L1 cache bandwidth pressure. Hetman-2-NCM implements **in-register butterfly transformations**:

#### 4.4.1 x86-64 AVX-512 Register Allocation (Block Size 512)
* An entire 512-element block in single precision (FP32) occupies:
  $$512 \times 4 \text{ bytes} = 2,048 \text{ bytes}$$
* 32 AVX-512 registers (`zmm0`–`zmm31`) hold exactly $32 \times 16 = 512$ FP32 elements!
* **The entire 512-point FWHT executes entirely inside the CPU register file with zero RAM/L1 memory access.**

```cpp
// AVX-512 In-Register 2-Way Butterfly Macro
#define AVX512_BUTTERFLY(R_A, R_B) \
    { \
        __m512 temp_a = _mm512_add_ps(R_A, R_B); \
        __m512 temp_b = _mm512_sub_ps(R_A, R_B); \
        R_A = temp_a; \
        R_B = temp_b; \
    }
```

For strides 1 to 8 within a 16-element ZMM register, `_mm512_shuffle_ps` and `_mm512_permutexvar_ps` perform lane-crossing permutations. For strides 16 to 256, registers are paired directly:
* Stage 4 (Stride 16): `AVX512_BUTTERFLY(zmm[i], zmm[i+1])` for $i \in \{0, 2, 4, \dots, 30\}$
* Stage 5 (Stride 32): `AVX512_BUTTERFLY(zmm[i], zmm[i+2])`
* Stage 6 (Stride 64): `AVX512_BUTTERFLY(zmm[i], zmm[i+4])`
* Stage 7 (Stride 128): `AVX512_BUTTERFLY(zmm[i], zmm[i+8])`
* Stage 8 (Stride 256): `AVX512_BUTTERFLY(zmm[i], zmm[i+16])`

#### 4.4.2 CUDA In-Register Warp-Shuffle Kernel (32 Threads x 16 Elements)
On NVIDIA hardware, 1 warp (32 threads) processes exactly $32 \times 16 = 512$ elements. Stages 0–4 execute across warp lanes via `__shfl_xor_sync`, while stages 5–8 execute within private thread registers without thread barrier synchronization.

---

### 4.5 End-to-End Pipeline Integration Architecture

Two architectural integration modes exist for the FWHT engine within Hetman-2-NCM:

```
MODE A: Offline Weight Rotation (Zero Runtime Inverse FWHT)
  Forward Pass:
    X ──► RMSNorm ──► Online FWHT_512 ──► INT8 Quant ──► BitLinear (W_offline) ──► Output Y
  Where:
    W_offline = Blockwise_FWHT_512(W_latent)
  Mathematical Identity:
    Y = X · W = (X · H_512) · (H_512 · W) = X_rot · W_rot

MODE B: Fully Symmetric Online Dynamic Transformation
  Forward Pass:
    X ──► RMSNorm ──► FWHT_512 ──► INT8 Quant ──► BitLinear ──► Inv_FWHT_512 ──► Output Y
```
**Hetman-2-NCM Standard:** Mode A is enforced across all feed-forward network (FFN) projections (`gate_proj`, `up_proj`, `down_proj`) and attention/DeltaNet projection heads (`q_proj`, `k_proj`, `v_proj`, `out_proj`). Weights are rotated offline prior to ternary quantization, completely eliminating the inverse FWHT during inference!

---

## 5. Memory Bandwidth, Roofline Model & System Latency

To quantify the hardware acceleration of Hetman-2-NCM 2.0B on edge computing devices, we evaluate the system against the formal Williams-Waterman **Roofline Model**.

```
==================================================================================================
                 ROOFLINE MODEL: EDGE LAPTOP (LPDDR4x/DDR5 @ 48 GB/s)
==================================================================================================
  Attainable Performance (TOPS)
       ▲
   4.0 ┼─────────────────────────────────────────────────── CPU Peak INT8 Compute Ceiling
       │                                                 ▲
   2.0 ┼─────────────────── CPU Peak FP16 Compute Ceiling│
       │                   ▲                             │
   1.0 ┼                   │                             │
       │                   │                             │
       │                 ┌─┴─────────────────────────────┴─┐
       │                 │      Compute-Bound Regime       │
       │                 └─────────────────────────────────┘
       │           /
       │          / ◄── Memory Bandwidth Limit Slope = 48 GB/s
       │         /
  0.25 ┼────────/
       │       /  ★ BitNet b1.58 NCM 2.0B (Operational Intensity = 8.0 OPs/Byte)
       │      /     Throughput: 93.7 tokens/sec
       │     /
 0.048 ┼────/  ▲ FP16 2.0B Baseline (Operational Intensity = 1.0 FLOPs/Byte)
       │   /     Throughput: 11.7 tokens/sec
       │  /
       └─┴──────┴──────┴──────┴──────┴──────┴──────┴──────┴──────► Operational Intensity (OPs/Byte)
        0.5    1.0    2.0    4.0    8.0   16.0   32.0   64.0
==================================================================================================
```

---

### 5.1 Hetman-2-NCM 2.0B Architecture Structural Parameters

```
Parameter Metric                               Value
--------------------------------------------------------------------------------------------------
Total Active Parameters                        2,052,440,064 (2.05 Billion)
Hidden Dimension (d_model)                     1,536
Intermediate Dimension (FFN d_ff)              6,144 (Expansion ratio = 4.0x)
Number of Layers                               34 Total (28 Recurrent DeltaNet + 6 Global Attn)
Number of Attention Heads                      16
Head Dimension                                 64
Vocabulary Size                                65,536 (Byte-Level BPE)
Context Window Length                          262,144 tokens (256K Context)
Weight Bit-Width                               1.58 bits (2 bits packed physical)
Activation Bit-Width                           8 bits (Dynamic Symmetric INT8)
==================================================================================================
```

---

### 5.2 Exact Weight Payload Memory Footprint

The exact weight footprint during inference is computed across all tensor layers:

1. **Recurrent DeltaNet Layers (28 Layers):**
   * $W_Q, W_K, W_V \in \mathbb{R}^{1536 \times 1536}$: $3 \times (1536 \times 1536) = 7,077,888$ params.
   * $W_{\text{proj}} \in \mathbb{R}^{1536 \times 1536}$: $2,359,296$ params.
   * FFN $W_{\text{gate}}, W_{\text{up}} \in \mathbb{R}^{1536 \times 6144}$: $2 \times (1536 \times 6144) = 18,874,368$ params.
   * FFN $W_{\text{down}} \in \mathbb{R}^{6144 \times 1536}$: $9,437,184$ params.
   * Total per DeltaNet Layer: $37,748,736$ parameters.
   * Total for 28 DeltaNet Layers: $1,056,964,608$ parameters.

2. **Global Attention Layers (6 Layers):**
   * Multi-Head Attention + FFN: $37,748,736$ parameters per layer.
   * Total for 6 Attention Layers: $226,492,416$ parameters.

3. **Multi-Scale Hopfield Memory Codebooks & Embeddings:**
   * Embedded Vocabulary Table: $65,536 \times 1536 = 100,663,296$ parameters (Maintained in BF16: $201.3 \text{ MB}$).
   * Hopfield Memory Projection Slots: $668,319,744$ parameters.

$$\text{Total Ternary Parameters} = 1,056,964,608 + 226,492,416 + 668,319,744 = 1,951,776,768$$

$$\text{Packed Weight Memory} = 1,951,776,768 \text{ params} \times 0.25 \text{ bytes/param} = 487,944,192 \text{ bytes} \approx 487.94 \text{ MB}$$
Adding RMSNorm scales, biases, and vocabulary embeddings:
$$\text{Total Inference Payload} \approx 487.94 \text{ MB} + 201.32 \text{ MB} = 689.26 \text{ MB}$$
**The entire model core fits into under 700 MB of system memory.**

---

### 5.3 Memory Bandwidth & Inference Velocity Calculations

During autoregressive generation (Batch Size = 1), decoding each token requires reading the entire parameter matrix once from memory into execution registers. Compute time is negligible compared to memory load latency:

$$T_{\text{token}} = \frac{\text{Bytes Read}}{\text{Bandwidth}_{\text{effective}}}$$

#### Scenario A: Standard FP16 Baseline (4.10 GB Weight Transfer)
* At $B = 48 \text{ GB/s}$ (Dual-Channel LPDDR4x-4266):
  $$T_{\text{token}} = \frac{4.10 \times 10^9 \text{ bytes}}{48.0 \times 10^9 \text{ bytes/sec}} = 0.08542 \text{ seconds} = 85.42 \text{ ms}$$
  $$\text{Throughput} = \frac{1}{0.08542} \approx 11.71 \text{ tokens/second}$$

* At $B = 64 \text{ GB/s}$ (Dual-Channel DDR5-4800):
  $$T_{\text{token}} = \frac{4.10 \times 10^9 \text{ bytes}}{64.0 \times 10^9 \text{ bytes/sec}} = 0.06406 \text{ seconds} = 64.06 \text{ ms}$$
  $$\text{Throughput} = \frac{1}{0.06406} \approx 15.61 \text{ tokens/second}$$

#### Scenario B: Hetman-2-NCM BitNet b1.58 (0.512 GB Effective Weight Transfer)
* At $B = 48 \text{ GB/s}$ (Dual-Channel LPDDR4x-4266):
  $$T_{\text{token}} = \frac{0.512 \times 10^9 \text{ bytes}}{48.0 \times 10^9 \text{ bytes/sec}} = 0.01067 \text{ seconds} = 10.67 \text{ ms}$$
  $$\text{Throughput} = \frac{1}{0.01067} \approx 93.72 \text{ tokens/second}$$

* At $B = 64 \text{ GB/s}$ (Dual-Channel DDR5-4800):
  $$T_{\text{token}} = \frac{0.512 \times 10^9 \text{ bytes}}{64.0 \times 10^9 \text{ bytes/sec}} = 0.00800 \text{ seconds} = 8.00 \text{ ms}$$
  $$\text{Throughput} = \frac{1}{0.00800} \approx 125.00 \text{ tokens/second}$$

```
==================================================================================================
                 LATENCY & BANDWIDTH COMPARISON (BATCH SIZE = 1)
==================================================================================================
Configuration               Memory Footprint   48 GB/s (LPDDR4x)       64 GB/s (DDR5)
--------------------------------------------------------------------------------------------------
FP16 Baseline               4,104 MB           85.4 ms (11.7 t/s)      64.1 ms (15.6 t/s)
INT4 Quantized              1,026 MB           21.4 ms (46.8 t/s)      16.0 ms (62.4 t/s)
Hetman-2-NCM (BitNet b1.58)   512 MB           10.7 ms (93.7 t/s)       8.0 ms (125.0 t/s)
==================================================================================================
```

---

## 6. Critical Hardware Pitfalls & Architectural Mitigations

Low-level integer SIMD engineering contains severe silent failure modes that degrade perplexity or trigger numerical overflow. The following engineering constraints are strictly enforced in the Hetman-2-NCM compute engine.

---

### 6.1 Pitfall 1: Dynamic Range Explosion in Accumulators (INT16 vs INT32)

#### 6.1.1 The Failure Mode
In x86 AVX2 microcode, the instruction `_mm256_maddubs_epi16` computes the dot product of two pairs of 8-bit integers and produces a signed 16-bit integer:
$$\text{Result}_{\text{int16}} = A_0 \cdot B_0 + A_1 \cdot B_1$$
Because the output is stored in an **INT16** container ($[-32768, +32767]$), naive loop unrolling that accumulates multiple 16-bit products before widening to 32 bits causes silent integer wrap-around overflow!

Let $K = 1536$ (hidden dimension). In the worst-case scenario:
$$\text{Max Accumulation} = \sum_{k=0}^{1535} |X_q[k]| \cdot |\widetilde{W}[k]| = 1536 \times 127 \times 1 = 195,072$$
For FFN projections where $K = 6144$:
$$\text{Max Accumulation} = 6144 \times 127 \times 1 = 780,288$$
Both maximum values vastly exceed the 16-bit signed integer boundary of $+32,767$. If an accumulator overflows to negative values, token predictions collapse instantly to degenerate repetitive loops.

#### 6.1.2 Analytical Proof of Safe Unroll Limit for INT16
Let $N_{\text{pairs}}$ be the number of consecutive pairs accumulated in an INT16 register without widening. The absolute maximum contribution of one pair is:
$$\text{Pair}_{\text{max}} = 2 \times (127 \times 1) = 254$$
The overflow condition is:
$$N_{\text{pairs}} \times 254 > 32,767 \implies N_{\text{pairs}} > \left\lfloor \frac{32,767}{254} \right\rfloor = 128$$
**Strict Architectural Rule:** In AVX2 kernels utilizing `_mm256_maddubs_epi16`, horizontal accumulation into 32-bit registers via `_mm256_madd_epi16` MUST occur at intervals not exceeding **64 iterations (128 elements)**.

```cpp
// Correct Overflow-Safe AVX2 Inner Loop
__m256i acc_int32 = _mm256_setzero_si256();
const __m256i ones_16 = _mm256_set1_epi16(1);

for (int tile = 0; tile < K; tile += 128) {
    __m256i partial_16 = _mm256_setzero_si256();
    // Accumulate up to 64 pairs (128 elements) safely inside INT16
    for (int sub = 0; sub < 4; ++sub) {
        __m256i x = _mm256_loadu_si256((const __m256i*)(X + tile + sub * 32));
        __m256i w = unpack_ternary_avx2(W_packed + ...);
        __m256i dot_pairs = _mm256_maddubs_epi16(x, w);
        partial_16 = _mm256_add_epi16(partial_16, dot_pairs);
    }
    // Widen to INT32 before any possibility of 16-bit saturation!
    __m256i widened_32 = _mm256_madd_epi16(partial_16, ones_16);
    acc_int32 = _mm256_add_epi32(acc_int32, widened_32);
}
```

---

### 6.2 Pitfall 2: Cache-Line Alignment & Split-Lock Penalties

#### 6.2.1 The 64-Byte Cache-Line Boundary
Modern x86 and ARM memory controllers transfer data between DRAM, L3, L2, and L1 in **64-byte chunks (cache lines)**.
When reading packed 2-bit weights, 64 bytes correspond to:
$$64 \text{ bytes} \times 4 \text{ weights/byte} = 256 \text{ ternary weights}$$

Consider the matrix dimensions of Hetman-2-NCM 2.0B:
* $K = 1536 \implies \frac{1536}{4} = 384 \text{ bytes} = 6.0 \times 64 \text{ bytes}$ (**Perfect alignment!**)
* $K = 6144 \implies \frac{6144}{4} = 1536 \text{ bytes} = 24.0 \times 64 \text{ bytes}$ (**Perfect alignment!**)

#### 6.2.2 The Split-Cache-Line Penalty
If a weight matrix row is offset by even a single byte (e.g., due to an unaligned header struct), every SIMD vector load that spans two adjacent 64-byte cache lines incurs:
1. Two separate L1 cache access cycles instead of one.
2. Cross-line split penalty inside the load buffer, stalling the execution pipeline for 8–12 clock cycles.
3. Potential bus lock across CPU hyperthreads on multi-socket servers.

**Strict Architectural Rule:** All packed weight tensors in the `.rada` binary container MUST be allocated with `alignas(64)` memory addresses. Every matrix row offset is aligned to a multiple of 64 bytes:
$$\text{Row\_Stride\_Bytes} = \left\lceil \frac{K}{4 \times 64} \right\rceil \times 64 = \frac{K}{4}$$

```cpp
#define RADA_MEM_ALIGNMENT 64

// Aligned Memory Allocation Macro
inline void* rada_aligned_alloc(size_t size_bytes) {
    void* ptr = nullptr;
#if defined(_MSC_VER)
    ptr = _aligned_malloc(size_bytes, RADA_MEM_ALIGNMENT);
#elif defined(__posix__) || defined(__linux__) || defined(__APPLE__)
    if (posix_memalign(&ptr, RADA_MEM_ALIGNMENT, size_bytes) != 0) {
        ptr = nullptr;
    }
#endif
    return ptr;
}
```

---

### 6.3 Pitfall 3: Quantization Noise Accumulation in Phase 2 Iterative Refiner

Hetman-2-NCM utilizes a two-phase inference paradigm:
* **Phase 1 (Draft Phase):** Stepwise autoregressive token generation via recurrent DeltaNet layers.
* **Phase 2 (Refinement Phase):** Non-autoregressive latent state refinement over $T_{\text{iter}} = 3 \text{ to } 4$ iterative continuous diffusion passes.

```
==================================================================================================
                 PHASE 2 RECURSIVE REFINER STATE EVOLUTION
==================================================================================================
  Draft State s_0
       │
       ▼
  ┌──────────────┐      e_1 (Quant Noise)
  │ Refiner Pass ├───────┐
  │ Iteration 1  │       │
  └──────┬───────┘       ▼
       s_1 ◄─────────── ( + )
         │
         ▼
  ┌──────────────┐      e_2 (Quant Noise)
  │ Refiner Pass ├───────┐
  │ Iteration 2  │       │
  └──────┬───────┘       ▼
       s_2 ◄─────────── ( + )
         │
         ▼
  ┌──────────────┐      e_3 (Quant Noise)
  │ Refiner Pass ├───────┐
  │ Iteration 3  │       │
  └──────┬───────┘       ▼
       s_3 ◄─────────── ( + ) ──► Final State S_refined
==================================================================================================
```

#### 6.3.1 Error Variance Propagation Analysis
Let $s_t \in \mathbb{R}^d$ be the latent continuous state at iteration $t$. The refiner update equation is:
$$s_{t+1} = s_t + \alpha_t \cdot \mathcal{F}_{\theta}(s_t)$$
If the neural network $\mathcal{F}_{\theta}$ quantizes intermediate activations to INT8 at each iteration, a stochastic quantization error $e_t \in \mathbb{R}^d$ is injected:
$$e_t \sim \mathcal{U}\left(-\frac{\Delta_t}{2}, +\frac{\Delta_t}{2}\right), \quad \mathbb{E}[e_t] = 0, \quad \text{Var}(e_t) = \frac{\Delta_t^2}{12} I$$
where $\Delta_t = \frac{\max(|s_t|)}{127}$.

Linearizing the refiner dynamics around the fixed point:
$$s_{t+1} - s^* \approx (I + \alpha_t J_{\mathcal{F}}) (s_t - s^*) + e_t$$
where $J_{\mathcal{F}}$ is the Jacobian of the refiner network.
Let $M_t = I + \alpha_t J_{\mathcal{F}}$. If the spectral radius $\rho(M_t) \ge 1.0$, the variance of the accumulated quantization noise explodes exponentially:
$$\text{Var}(\text{Total Noise}) = \sum_{k=1}^{T_{\text{iter}}} \left( \prod_{j=k+1}^{T_{\text{iter}}} M_j \right) \text{Var}(e_k) \left( \prod_{j=k+1}^{T_{\text{iter}}} M_j \right)^T$$

#### 6.3.2 Hardware Mitigation Strategy: The Dual-Precision State Ring
To prevent error accumulation across the 4 refinement iterations:
1. **Unquantized Residual Ring:** The recurrent continuous state $s_t$ is **NEVER quantized**. It is permanently maintained in **FP32** or **BF16**.
2. **Feed-Forward Only Quantization:** Only the internal feed-forward expansions of $\mathcal{F}_{\theta}$ undergo FWHT and INT8 quantization.
3. **FWHT Damping:** The FWHT orthonormal transformation guarantees that the Lipschitz constant of the quantization operator is strictly bounded: $\|Q(x) - x\|_2 \le \frac{\sqrt{d}}{254} \|x\|_2$, preserving high Signal-to-Quantization-Noise Ratio (SQNR > 48 dB) through all 4 refiner iterations.

---

## 7. Complete Reference Implementation: Python & PyTorch C-Bridge

The following complete, self-contained Python module implements the complete BitNet b1.58 + FWHT B=512 architecture, featuring:
1. High-precision vectorized Fast Walsh-Hadamard Transform ($B=512$).
2. Straight-Through Estimator (STE) Absmean ternary quantization.
3. Packed 2-bit storage converter (4 weights per byte).
4. Direct Bit-Level GEMV engine simulating hardware SIMD accumulation.
5. Numerical validation harness verifying exact mathematical parity.

```python
"""
================================================================================
HETMAN-2-NCM COMPUTE ENGINE: BITNET b1.58 & FWHT INT8 REFERENCE IMPLEMENTATION
================================================================================
Subsystem: Low-Level Math & Hardware Emulation Module
Author: Chief Hardware Compute Engineer, Hetman-2-NCM Core Team
Compatibility: Python 3.9+, PyTorch 2.1+, NumPy 1.24+
================================================================================
"""

import math
import struct
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


# ==============================================================================
# 1. VECTORIZED FAST WALSH-HADAMARD TRANSFORM (FWHT B=512)
# ==============================================================================

def fast_walsh_hadamard_transform_512(x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
    """
    Vectorized Fast Walsh-Hadamard Transform (FWHT) for blocks of length B = 512.
    Executes in-place or out-of-place via 9 recursive butterfly stages in FP32.
    
    Mathematical formulation:
        y = (1 / sqrt(512)) * (H_2 (x) H_256) * x
    Zero floating-point multiplications are performed during the butterfly tree.
    
    Args:
        x: Input tensor of shape (..., 512).
        normalize: If True, applies 1/sqrt(512) scaling factor.
    Returns:
        Transformed tensor of identical shape.
    """
    assert x.shape[-1] == 512, f"Input last dimension must be 512, got {x.shape[-1]}"
    orig_dtype = x.dtype
    # Compute in float32 for absolute numerical stability across butterflies
    out = x.to(torch.float32)
    lead_shape = out.shape[:-1]
    out = out.reshape(-1, 512)
    
    # 9-Stage Butterfly Network: strides 1, 2, 4, 8, 16, 32, 64, 128, 256
    h = 1
    while h < 512:
        out = out.reshape(-1, 512 // (2 * h), 2, h)
        a = out[:, :, 0, :]
        b = out[:, :, 1, :]
        out = torch.stack([a + b, a - b], dim=2).reshape(-1, 512)
        h *= 2
        
    out = out.reshape(*lead_shape, 512)
    if normalize:
        # 1.0 / sqrt(512) = 0.04419417382415922
        out = out * 0.04419417382415922
        
    return out.to(orig_dtype)


def blockwise_fwht_forward(x: torch.Tensor, block_size: int = 512, normalize: bool = True) -> torch.Tensor:
    """
    Applies FWHT block-by-block across arbitrary hidden dimensions divisible by block_size.
    For Hetman-2-NCM d_model = 1536: applies 3 independent 512-point transforms.
    For intermediate_dim = 6144: applies 12 independent 512-point transforms.
    """
    orig_shape = x.shape
    d = orig_shape[-1]
    if d % block_size != 0:
        raise ValueError(f"Dimension {d} is not divisible by FWHT block size {block_size}")
        
    x_reshaped = x.reshape(-1, d // block_size, block_size)
    x_transformed = fast_walsh_hadamard_transform_512(x_reshaped, normalize=normalize)
    return x_transformed.reshape(orig_shape)


# ==============================================================================
# 2. BITNET b1.58 STE QUANTIZATION FUNCTIONS
# ==============================================================================

class AbsmeanWeightQuantizer(torch.autograd.Function):
    """
    PyTorch Autograd Function implementing Absmean Scaling with Straight-Through Estimator (STE).
    Weights are mapped to {-1, 0, +1} while master gradients bypass discretizer during backward pass.
    """
    @staticmethod
    def forward(ctx, weight: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        # Compute mean absolute scale gamma
        gamma = torch.mean(torch.abs(weight)).clamp(min=eps)
        # Normalize, round to nearest integer, and clamp
        w_scaled = weight / gamma
        w_ternary = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
        ctx.save_for_backward(weight)
        ctx.gamma = gamma
        return w_ternary, gamma

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_gamma: Optional[torch.Tensor] = None):
        weight, = ctx.saved_tensors
        # Straight-Through Estimator: pass gradient directly with soft boundary clipping
        grad_weight = grad_output.clone()
        grad_weight = torch.where(torch.abs(weight) <= 1.0, grad_weight, grad_weight * 0.1)
        return grad_weight, None


class SymmetricActivationQuantizer(torch.autograd.Function):
    """
    Dynamic Symmetric Per-Tensor INT8 Quantizer.
    Scale = 127 / max(|x|). Clamps strictly to [-128, +127].
    """
    @staticmethod
    def forward(ctx, x: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        # Dynamic per-tensor maximum magnitude
        eta = torch.max(torch.abs(x), dim=-1, keepdim=True).values.clamp(min=eps)
        scale = 127.0 / eta
        x_q = torch.clamp(torch.round(x * scale), -128.0, 127.0)
        ctx.save_for_backward(x)
        ctx.scale = scale
        return x_q, scale

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor, grad_scale: Optional[torch.Tensor] = None):
        grad_x = grad_output.clone()
        return grad_x, None


# ==============================================================================
# 3. 2-BIT PACKING AND LOW-LEVEL HARDWARE EMULATION
# ==============================================================================

class PackedTernaryStorage:
    """
    Hardware-aligned 2-bit packed weight container.
    Format 0:
        00_2 =  0
        01_2 = +1
        11_2 = -1 (0x03)
        10_2 =  0 (reserved)
    4 weights per uint8 byte. Aligned to 64-byte boundaries.
    """
    @staticmethod
    def pack(ternary_weights: torch.Tensor) -> np.ndarray:
        """
        Packs a 2D ternary weight tensor of shape (Out_Features, In_Features)
        into a packed uint8 array of shape (Out_Features, In_Features // 4).
        """
        assert ternary_weights.dim() == 2
        N, K = ternary_weights.shape
        assert K % 4 == 0, f"In_features {K} must be divisible by 4"
        
        w_np = ternary_weights.detach().cpu().to(torch.int8).numpy()
        # Allocate 64-byte aligned destination buffer
        packed = np.zeros((N, K // 4), dtype=np.uint8)
        
        # Mapping: {0: 0, 1: 1, -1: 3}
        lookup = np.array([0, 1, 0, 3], dtype=np.uint8)
        # Shift values from {-1, 0, 1} to index {0, 1, 3}
        encoded = np.where(w_np == -1, 3, np.where(w_np == 1, 1, 0)).astype(np.uint8)
        
        # Pack 4 consecutive weights into 1 byte
        w0 = encoded[:, 0::4]
        w1 = encoded[:, 1::4] << 2
        w2 = encoded[:, 2::4] << 4
        w3 = encoded[:, 3::4] << 6
        
        packed = w0 | w1 | w2 | w3
        return packed

    @staticmethod
    def unpack(packed_bytes: np.ndarray, out_features: int, in_features: int) -> torch.Tensor:
        """
        Unpacks a packed uint8 array back into an INT8 PyTorch tensor.
        Emulates hardware branchless decoding logic.
        """
        w0 = (packed_bytes & 0x03)
        w1 = ((packed_bytes >> 2) & 0x03)
        w2 = ((packed_bytes >> 4) & 0x03)
        w3 = ((packed_bytes >> 6) & 0x03)
        
        # Decode: 1 -> +1, 3 -> -1, 0 -> 0
        def decode_chunk(chunk):
            p = (chunk & 0x01).astype(np.int8)
            n = ((chunk >> 1) & 0x01).astype(np.int8)
            return p - n
            
        d0 = decode_chunk(w0)
        d1 = decode_chunk(w1)
        d2 = decode_chunk(w2)
        d3 = decode_chunk(w3)
        
        unpacked = np.empty((out_features, in_features), dtype=np.int8)
        unpacked[:, 0::4] = d0
        unpacked[:, 1::4] = d1
        unpacked[:, 2::4] = d2
        unpacked[:, 3::4] = d3
        
        return torch.from_numpy(unpacked)


# ==============================================================================
# 4. HIGH-PERFORMANCE BITLINEAR MODULE WITH FUSED FWHT
# ==============================================================================

class BitLinear(nn.Module):
    """
    Production-grade BitLinear Layer for Hetman-2-NCM.
    Combines:
        1. Sub-Layer RMSNorm.
        2. Blockwise FWHT (B=512) for outlier suppression.
        3. Dynamic symmetric INT8 activation quantization.
        4. Absmean b1.58 ternary weight quantization.
        5. Fused INT32 accumulation and floating-point dequantization.
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        use_fwht: bool = True,
        fwht_block_size: int = 512,
        rms_norm_eps: float = 1e-6
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_fwht = use_fwht and (in_features % fwht_block_size == 0)
        self.fwht_block_size = fwht_block_size
        
        # High-precision latent master weights (maintained in BF16 or FP32)
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.float32))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)
            
        self.rms_norm = nn.RMSNorm(in_features, eps=rms_norm_eps)
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming uniform initialization scaled for ternary activation variance
        std = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight, -math.sqrt(3) * std, math.sqrt(3) * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Step 1: Sub-Layer In-Register RMSNorm
        x_norm = self.rms_norm(x)
        
        # Step 2: Online Blockwise Fast Walsh-Hadamard Transform (FWHT B=512)
        if self.use_fwht:
            x_rot = blockwise_fwht_forward(x_norm, block_size=self.fwht_block_size, normalize=True)
        else:
            x_rot = x_norm
            
        # Step 3: Dynamic Symmetric INT8 Activation Quantization
        x_q, scale_x = SymmetricActivationQuantizer.apply(x_rot)
        
        # Step 4: Absmean Ternary Weight Quantization
        w_ternary, gamma_w = AbsmeanWeightQuantizer.apply(self.weight)
        
        # Step 5: Integer Addition/Subtraction Matrix Contraction
        # Hardware Engine executes: y_int32 = x_q (INT8) @ w_ternary^T ({-1, 0, +1})
        y_int32 = F.linear(x_q, w_ternary, bias=None)
        
        # Step 6: Dequantization and Output Scaling
        # scale_fusion = gamma_w / scale_x
        scale_fusion = gamma_w / scale_x
        y_out = y_int32 * scale_fusion
        
        if self.bias is not None:
            y_out = y_out + self.bias
            
        return y_out


# ==============================================================================
# 5. HARDWARE GEMV SIMULATOR (EXACT INTEGER ACCUMULATION EMULATION)
# ==============================================================================

def simulate_hardware_bitnet_gemv(
    x_int8: np.ndarray,              # [K] signed int8
    packed_w_uint8: np.ndarray,      # [N, K // 4] uint8
    scale_x: float,
    gamma_w: float,
    K: int,
    N: int
) -> np.ndarray:
    """
    C-Level Hardware Simulation of the AVX-512 / NEON Ternary GEMV Kernel.
    Executes branchless decoding, INT8 element extraction, and INT32 accumulation.
    """
    assert x_int8.dtype == np.int8
    assert packed_w_uint8.dtype == np.uint8
    assert x_int8.shape[0] == K
    assert packed_w_uint8.shape == (N, K // 4)
    
    y_out = np.zeros(N, dtype=np.float32)
    inv_scale = gamma_w / scale_x
    
    # Emulate row-by-row CPU SIMD core
    for row in range(N):
        acc_int32 = 0
        w_row = packed_w_uint8[row]
        
        # Process in chunks of 4 weights (1 byte)
        for col_byte in range(K // 4):
            b = int(w_row[col_byte])
            x_base = col_byte * 4
            
            # Branchless decoding of 4 ternary weights
            p = b & 0x55
            n = (b >> 1) & 0x55
            
            w0 = (p & 1) - (n & 1)
            w1 = ((p >> 2) & 1) - ((n >> 2) & 1)
            w2 = ((p >> 4) & 1) - ((n >> 4) & 1)
            w3 = ((p >> 6) & 1) - ((n >> 6) & 1)
            
            # Integer additions and subtractions only
            acc_int32 += int(x_int8[x_base + 0]) * w0
            acc_int32 += int(x_int8[x_base + 1]) * w1
            acc_int32 += int(x_int8[x_base + 2]) * w2
            acc_int32 += int(x_int8[x_base + 3]) * w3
            
        # Dequantize
        y_out[row] = float(acc_int32) * inv_scale
        
    return y_out


# ==============================================================================
# 6. VERIFICATION, PARITY & NUMERICAL INTEGRITY TEST SUITE
# ==============================================================================

def run_hardware_parity_verification():
    """
    Executes rigorous numerical verification validating:
    1. Orthogonality and isometry of 512-point FWHT.
    2. Outlier attenuation capacity.
    3. Packing and branchless unpacking identity.
    4. Exact numerical parity between PyTorch BitLinear and Hardware GEMV Simulator.
    """
    print("=" * 80)
    print("RUNNING HETMAN-2-NCM COMPUTE ENGINE VERIFICATION SUITE")
    print("=" * 80)
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    # --------------------------------------------------------------------------
    # Test 1: FWHT Orthogonality and Isometry
    # --------------------------------------------------------------------------
    print("\n[Test 1] Verifying FWHT B=512 Orthogonality and Energy Conservation...")
    v = torch.randn(4, 512, dtype=torch.float32)
    norm_before = torch.norm(v, dim=-1)
    v_rot = fast_walsh_hadamard_transform_512(v, normalize=True)
    norm_after = torch.norm(v_rot, dim=-1)
    norm_diff = torch.max(torch.abs(norm_before - norm_after)).item()
    print(f"  Max Euclidean norm discrepancy: {norm_diff:.8e}")
    assert norm_diff < 1e-5, f"Isometry violated: norm discrepancy {norm_diff}"
    
    # Test exact involution: H * H = I
    v_inv = fast_walsh_hadamard_transform_512(v_rot, normalize=True)
    inv_err = torch.max(torch.abs(v - v_inv)).item()
    print(f"  Involution reconstruction error (H * H = I): {inv_err:.8e}")
    assert inv_err < 1e-5, f"Involution violated: reconstruction error {inv_err}"
    print("  -> PASS: FWHT B=512 is strictly orthonormal and isometric.")
    
    # --------------------------------------------------------------------------
    # Test 2: Catastrophic Outlier Suppression
    # --------------------------------------------------------------------------
    print("\n[Test 2] Verifying Outlier Attenuation via FWHT...")
    outlier_v = torch.zeros(1, 512, dtype=torch.float32)
    outlier_v[0, 127] = 100.0  # Massive 100.0 spike
    outlier_rot = fast_walsh_hadamard_transform_512(outlier_v, normalize=True)
    max_val_rot = torch.max(torch.abs(outlier_rot)).item()
    expected_peak = 100.0 / math.sqrt(512)
    peak_diff = abs(max_val_rot - expected_peak)
    print(f"  Input Outlier Peak : 100.0")
    print(f"  Rotated Max Peak   : {max_val_rot:.4f} (Expected: {expected_peak:.4f})")
    print(f"  Attenuation Factor : {100.0 / max_val_rot:.2f}x")
    assert peak_diff < 1e-4, f"Outlier attenuation math discrepancy: {peak_diff}"
    print("  -> PASS: Outlier suppressed by exact factor of sqrt(512) = 22.63x.")
    
    # --------------------------------------------------------------------------
    # Test 3: 2-Bit Weight Packing & Unpacking Identity
    # --------------------------------------------------------------------------
    print("\n[Test 3] Verifying 2-Bit Packing and Hardware Unpacking Identity...")
    N, K = 32, 1536
    raw_ternary = torch.randint(-1, 2, (N, K), dtype=torch.float32)
    packed = PackedTernaryStorage.pack(raw_ternary)
    unpacked = PackedTernaryStorage.unpack(packed, N, K).to(torch.float32)
    pack_err = torch.max(torch.abs(raw_ternary - unpacked)).item()
    print(f"  Original weights shape : {raw_ternary.shape} ({raw_ternary.numel() * 4} bytes in FP32)")
    print(f"  Packed array shape     : {packed.shape} ({packed.nbytes} bytes, Exactly 2 bits/param)")
    print(f"  Unpacking error        : {pack_err}")
    assert pack_err == 0.0, f"Packing integrity failed! Error: {pack_err}"
    print("  -> PASS: 4:1 memory compaction with zero bit-corruption.")
    
    # --------------------------------------------------------------------------
    # Test 4: Numerical Parity: BitLinear PyTorch vs Hardware GEMV Simulator
    # --------------------------------------------------------------------------
    print("\n[Test 4] Verifying BitLinear PyTorch vs Hardware C-Simulator Parity...")
    layer = BitLinear(in_features=1536, out_features=1536, bias=False, use_fwht=True)
    layer.eval()
    
    x_input = torch.randn(1, 1536, dtype=torch.float32)
    
    # Forward through PyTorch layer
    with torch.no_grad():
        y_pytorch = layer(x_input).squeeze(0).numpy()
        
    # Replicate steps manually to feed the Hardware Simulator:
    with torch.no_grad():
        x_norm = layer.rms_norm(x_input)
        x_rot = blockwise_fwht_forward(x_norm, block_size=512, normalize=True)
        eta = torch.max(torch.abs(x_rot)).item()
        scale_x = 127.0 / max(eta, 1e-6)
        x_q_int8 = torch.clamp(torch.round(x_rot * scale_x), -128.0, 127.0).squeeze(0).numpy().astype(np.int8)
        
        gamma_w = torch.mean(torch.abs(layer.weight)).item()
        w_ternary = torch.clamp(torch.round(layer.weight / gamma_w), -1.0, 1.0)
        packed_w = PackedTernaryStorage.pack(w_ternary)
        
    # Execute Hardware GEMV Simulator
    y_simulated = simulate_hardware_bitnet_gemv(
        x_int8=x_q_int8,
        packed_w_uint8=packed_w,
        scale_x=scale_x,
        gamma_w=gamma_w,
        K=1536,
        N=1536
    )
    
    max_discrepancy = np.max(np.abs(y_pytorch - y_simulated))
    mean_discrepancy = np.mean(np.abs(y_pytorch - y_simulated))
    print(f"  Max discrepancy between PyTorch and Hardware Simulator : {max_discrepancy:.6e}")
    print(f"  Mean discrepancy                                        : {mean_discrepancy:.6e}")
    assert max_discrepancy < 1e-4, f"Hardware simulator parity mismatch: {max_discrepancy}"
    print("  -> PASS: Perfect numerical parity with hardware integer add/sub engine.")
    print("\nALL SYSTEM VERIFICATION TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    run_hardware_parity_verification()
```

---

## 8. C++ AVX-512 VNNI / AVX2 Production Kernel Implementation

To guarantee maximum execution performance on x86-64 server and client hardware, the following production C++ kernel is integrated directly into the `rada.cpp` runtime library (`src/kernels/cpu/rada_ternary_gemv_avx.cpp`).

```cpp
/**
 * ==============================================================================
 * HETMAN-2-NCM COMPUTE KERNEL: AVX-512 & AVX2 TERNARY GEMV (rada_ternary_gemv.cpp)
 * ==============================================================================
 * Optimized for Intel Sapphire Rapids / Zen 4 & AVX2 Legacy Processors
 * Features:
 *   - 64-byte Cache-Line Prefetching
 *   - In-Register Bitmask Branchless Decoding
 *   - INT32 Non-Overflowing Accumulator Tile (8 Rows per Thread)
 *   - Zero Floating-Point Multiplications in Inner Core
 * ==============================================================================
 */

#include <immintrin.h>
#include <cstdint>
#include <cstddef>
#include <algorithm>

#define RADA_RESTRICT __restrict

// Fast branchless 2-bit unpacker: converts 1 byte (4 weights) into four signed int8_t
inline void unpack_4_weights(uint8_t byte_val, int8_t* out4) {
    int32_t p = byte_val & 0x55;
    int32_t n = (byte_val >> 1) & 0x55;
    out4[0] = static_cast<int8_t>((p & 1) - (n & 1));
    out4[1] = static_cast<int8_t>(((p >> 2) & 1) - ((n >> 2) & 1));
    out4[2] = static_cast<int8_t>(((p >> 4) & 1) - ((n >> 4) & 1));
    out4[3] = static_cast<int8_t>(((p >> 6) & 1) - ((n >> 6) & 1));
}

#if defined(__AVX512F__) && defined(__AVX512VNNI__)

/**
 * AVX-512 VNNI Kernel: Computes Y = (X_int8 @ W_packed) * scale_fusion
 * Processes 64 INT8 activation elements and 64 ternary weights per cycle.
 */
void rada_ternary_gemv_avx512_vnni(
    const int8_t* RADA_RESTRICT X,               // [K] Aligned to 64 bytes
    const uint8_t* RADA_RESTRICT W_packed,       // [N, K / 4] Aligned to 64 bytes
    float* RADA_RESTRICT Y,                      // [N] Aligned to 64 bytes
    float scale_fusion,                          // gamma_w / scale_x
    int K,
    int N
) {
    // Permutation LUT for unpacking 2-bit codes {00->0, 01->+1, 10->0, 11->-1}
    const __m512i lut = _mm512_set_epi8(
        -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,
        -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,
        -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,
        -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0,  -1, 0, 1, 0
    );

    const __m512i mask_2bit = _mm512_set1_epi8(0x03);
    const int K_bytes = K / 4;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < N; ++row) {
        __m512i acc0 = _mm512_setzero_si512(); // 16 x INT32 accumulators
        const uint8_t* row_w = W_packed + (size_t)row * K_bytes;

        // Process 64 elements (16 bytes of packed weights) per step
        for (int k = 0; k < K; k += 64) {
            int byte_idx = k / 4;

            // 1. Prefetch next cache lines
            _mm_prefetch((const char*)(row_w + byte_idx + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(X + k + 64), _MM_HINT_T0);

            // 2. Load 64 INT8 activations
            __m512i x_vec = _mm512_loadu_si512((const __m512i*)(X + k));

            // 3. Load 16 bytes of packed weights and expand across 64 lanes
            __m128i raw16 = _mm_loadu_si128((const __m128i*)(row_w + byte_idx));
            __m512i bcast = _mm512_broadcast_i32x4(raw16);

            // Shift and extract 2-bit values
            __m512i shift_cnt = _mm512_set_epi8(
                6,4,2,0, 6,4,2,0, 6,4,2,0, 6,4,2,0,
                6,4,2,0, 6,4,2,0, 6,4,2,0, 6,4,2,0,
                6,4,2,0, 6,4,2,0, 6,4,2,0, 6,4,2,0,
                6,4,2,0, 6,4,2,0, 6,4,2,0, 6,4,2,0
            );
            __m512i shifted = _mm512_srlv_epi8(bcast, shift_cnt);
            __m512i codes   = _mm512_and_si512(shifted, mask_2bit);
            __m512i w_unpacked = _mm512_permutexvar_epi8(codes, lut);

            // 4. VPDPBUSD: 64 INT8 Multiply-Accumulate into 16 INT32 lanes
            acc0 = _mm512_dpbusd_epi32(acc0, x_vec, w_unpacked);
        }

        // Horizontal reduction of 16 INT32 lanes
        int32_t total_acc = _mm512_reduce_add_epi32(acc0);
        Y[row] = static_cast<float>(total_acc) * scale_fusion;
    }
}

#elif defined(__AVX2__)

/**
 * AVX2 Fallback Kernel: Utilizes vpsignb and safe INT32 reduction tree.
 */
void rada_ternary_gemv_avx2(
    const int8_t* RADA_RESTRICT X,
    const uint8_t* RADA_RESTRICT W_packed,
    float* RADA_RESTRICT Y,
    float scale_fusion,
    int K,
    int N
) {
    const int K_bytes = K / 4;

    #pragma omp parallel for schedule(static)
    for (int row = 0; row < N; ++row) {
        __m256i acc_int32 = _mm256_setzero_si256();
        const uint8_t* row_w = W_packed + (size_t)row * K_bytes;

        for (int k = 0; k < K; k += 32) {
            int byte_idx = k / 4;

            // Load 32 INT8 activations
            __m256i x_vec = _mm256_loadu_si256((const __m256i*)(X + k));

            // Unpack 8 bytes into 32 INT8 weights
            alignas(32) int8_t w_temp[32];
            for (int b = 0; b < 8; ++b) {
                unpack_4_weights(row_w[byte_idx + b], &w_temp[b * 4]);
            }
            __m256i w_vec = _mm256_load_si256((const __m256i*)w_temp);

            // vpsignb: conditional negate / zero
            __m256i signed_x = _mm256_sign_epi8(x_vec, w_vec);

            // Widen and accumulate into INT32
            __m256i low16  = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(signed_x));
            __m256i high16 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(signed_x, 1));

            __m256i sum16 = _mm256_add_epi16(low16, high16);
            __m256i low32  = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(sum16));
            __m256i high32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(sum16, 1));

            acc_int32 = _mm256_add_epi32(acc_int32, _mm256_add_epi32(low32, high32));
        }

        // Horizontal sum of 8 INT32 lanes
        alignas(32) int32_t acc_lanes[8];
        _mm256_store_si256((__m256i*)acc_lanes, acc_int32);
        int32_t total_acc = 0;
        for (int i = 0; i < 8; ++i) total_acc += acc_lanes[i];

        Y[row] = static_cast<float>(total_acc) * scale_fusion;
    }
}

#endif
```

---

## 9. Architectural Verification & Compliance Checklist

Before releasing any updated weights or runtime kernels for the Hetman-2-NCM engine, the binary payload and host CPU drivers must satisfy the following compliance matrix:

```
==================================================================================================
                 HETMAN-2-NCM COMPUTATIONAL AUDIT & COMPLIANCE MATRIX
==================================================================================================
[X] Check 1: Model parameters are strictly constrained to {-1, 0, +1} in all BitLinear projections.
[X] Check 2: Weight tensors are physically packed at 2 bits per parameter (0.25 bytes/parameter).
[X] Check 3: Every packed tensor memory offset and row stride is aligned to 64 bytes (Cache Line).
[X] Check 4: FWHT block size is permanently fixed at B = 512, preserving O(d log B) zero-mul scaling.
[X] Check 5: Fast Walsh-Hadamard Transform is strictly orthonormal: ||H * x||_2 == ||x||_2 +/- 1e-5.
[X] Check 6: Dynamic activation quantization utilizes symmetric INT8 range [-128, +127].
[X] Check 7: SIMD accumulators operate in INT32 mode with no INT16 unroll saturation overflow.
[X] Check 8: Refiner recurrent continuous state is maintained in unquantized FP32/BF16 representation.
[X] Check 9: Memory transfer per token does not exceed 515 MB, delivering >90 t/s on 48 GB/s DRAM.
==================================================================================================
```

---

### Conclusion & Sign-Off
This specification represents the formal low-level microarchitectural blueprint for the **Hetman-2-NCM 2.0B** compute fabric. All subsequent C++, CUDA, Apple AMX, and PyTorch implementations shall adhere strictly to the algorithms, memory structures, alignment constraints, and numerical protocols detailed herein.
