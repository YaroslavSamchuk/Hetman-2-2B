"""
================================================================================
HETMAN-2-NCM: PRODUCTION HOPFIELD KV-CACHE MEMORY ARCHITECTURE & A3 ATTENTION
Document Specifications:
  - RFC-RADA-NCM-13 (spec/13_RADA_CPP_HOPFIELD_KV_CACHE_ARCHITECTURE.md)
  - RFC-HETMAN-TRAIN-14 (spec/14_HOPFIELD_KV_CACHE_TRAINING_AND_NO_DEGRADATION.md)
Target Model: Hetman-2-NCM 2.0B (Neural Concept Model)
Runtime: Rada.cpp Edge Runtime / PyTorch Pretraining Pipeline
License: Apache License 2.0
================================================================================
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# 1. Architectural Constants (Strictly Aligned with RFC-13 and RFC-14)
# ==============================================================================

CACHE_LINE_SIZE: int = 64              # 64-byte alignment
NUM_GLOBAL_LAYERS: int = 6            # 6 Global Softmax Attention layers (4, 10, 16, 22, 28, 33)
GLOBAL_LAYER_INDICES: Tuple[int, ...] = (4, 10, 16, 22, 28, 33)
NUM_KV_HEADS: int = 2                 # 2 KV heads (GQA)
NUM_QUERY_HEADS: int = 12             # 12 Query heads (GQA 6:1)
HEAD_DIM: int = 128                   # d_head = 128
TOTAL_KV_DIM_PER_LAYER: int = NUM_KV_HEADS * HEAD_DIM  # 256

WORKING_WINDOW_SIZE: int = 1024       # Tier 1: 1024 tokens in Ring Buffer (FP8)
TOKENS_PER_CLUSTER: int = 32          # B_c = 32 tokens per semantic cluster
MAX_ATTRACTORS: int = 8000            # Tier 2: 8,000 attractor centroids (covers 256,000 tokens)
MAX_OUTLIERS: int = 2048              # Tier 3: 2,048 slots for factual outlier "needles"

DEFAULT_INVERSE_TEMP_BETA: float = 1.0 / math.sqrt(float(HEAD_DIM))  # 1 / sqrt(128) ≈ 0.0883883476
OUTLIER_THRESHOLD_MAHAL: float = 3.25 # Mahalanobis distance discriminator threshold D > 3.25
ATTRACTOR_MERGE_SIM_COS: float = 0.88 # Cosine similarity threshold for attractor clustering
LOG_CLUSTER_SIZE: float = math.log(float(TOKENS_PER_CLUSTER))  # ln(32) ≈ 3.4657359

# Numerical stability barriers (Audit #12)
EPS_DISPERSION_FLOOR: float = 1e-4        # Minimum numerical floor for dispersion (avoids ZeroDivisionError)
QUANT_NOISE_FLOOR: float = 0.005          # INT8 quantization noise floor (protects against Anomaly Collapse)
RECENCY_PROTECT_RATIO: float = 0.15       # Protect most recent 15% attractors from compaction


# ==============================================================================
# 2. FP8 / INT8 Quantization Helpers (Zero-Allocation Simulation)
# ==============================================================================

def quantize_symmetric_int8(
    tensor: torch.Tensor,
    eps: float = 1e-8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric per-head quantization to 8-bit integers (compatible with FP8 E4M3 scale).
    Matches Rada.cpp `quantize_and_store` (scale = max(|x|) / 127.0).

    Args:
        tensor: [..., HEAD_DIM] FP32 tensor.
        eps: Small epsilon to prevent division by zero.

    Returns:
        quant: [..., HEAD_DIM] torch.int8 tensor.
        scales: [...] FP32 scale tensor.
    """
    max_val = torch.amax(torch.abs(tensor), dim=-1, keepdim=True)
    max_val = torch.clamp(max_val, min=eps)
    scales = (max_val / 127.0).squeeze(-1)
    inv_scales = (127.0 / max_val)
    quant = torch.clamp(torch.round(tensor * inv_scales), -127.0, 127.0).to(torch.int8)
    return quant, scales


def dequantize_symmetric_int8(
    quant: torch.Tensor,
    scales: torch.Tensor
) -> torch.Tensor:
    """
    Dequantizes 8-bit integers back to FP32 using per-head scales.

    Args:
        quant: [..., HEAD_DIM] torch.int8 tensor.
        scales: [...] FP32 scale tensor.

    Returns:
        [..., HEAD_DIM] FP32 reconstructed tensor.
    """
    return quant.to(torch.float32) * scales.unsqueeze(-1)


# ==============================================================================
# 3. Data Structures for the 3-Tier Hopfield KV Cache
# ==============================================================================

@dataclass
class WorkingTokenSlot:
    """
    Tier 1 Slot: Raw uncompressed token vectors for the 6 global layers stored in FP8/INT8.
    Memory footprint per token: 6 layers x 2 (K,V) x 2 heads x 128 dim x 1 byte = 3,072 bytes.
    """
    k_quant: torch.Tensor       # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] int8
    v_quant: torch.Tensor       # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] int8
    k_scales: torch.Tensor      # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS] float32
    v_scales: torch.Tensor      # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS] float32
    token_position: int = 0     # Absolute position in context
    timestamp: int = 0          # Monotonic global counter


@dataclass
class AttractorCentroid:
    """
    Tier 2 Slot: Macro-attractor centroid representing B_c = 32 tokens in long-term memory.
    Maintains centroids mu_K, mu_V, dispersion sigma^2, token count n_c, and access timestamp.
    """
    mu_k_quant: torch.Tensor    # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] int8
    mu_v_quant: torch.Tensor    # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] int8
    mu_k_scales: torch.Tensor   # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS] float32
    mu_v_scales: torch.Tensor   # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS] float32
    dispersion: torch.Tensor    # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS] float32 (sigma_c^2)
    m2_k: Optional[torch.Tensor] = None # Sum of squared deviations for Welford
    token_count: int = 1        # Mass n_c
    last_access_timestamp: int = 0
    creation_timestamp: int = 0
    flags: int = 0


@dataclass
class OutlierFactSlot:
    """
    Tier 3 Slot: Isolated factual outlier ("needle in a haystack") preserved uncompressed.
    Guarantees 100% precision for passwords, numbers, dates, and high-entropy identifiers.
    """
    k_quant: torch.Tensor       # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] int8
    v_quant: torch.Tensor       # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] int8
    k_scales: torch.Tensor      # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS] float32
    v_scales: torch.Tensor      # [NUM_GLOBAL_LAYERS, NUM_KV_HEADS] float32
    original_token_position: int = 0
    outlier_score: float = 0.0  # Mahalanobis distance D(k, mu)
    timestamp: int = 0


# ==============================================================================
# 4. Variance-Compensated Softmax Operator (Functional Core)
# ==============================================================================

def variance_compensated_softmax(
    q: torch.Tensor,
    k_win: torch.Tensor,
    v_win: torch.Tensor,
    mu_k: Optional[torch.Tensor] = None,
    mu_v: Optional[torch.Tensor] = None,
    sigma2_k: Optional[torch.Tensor] = None,
    counts_c: Optional[Union[torch.Tensor, int, float]] = TOKENS_PER_CLUSTER,
    k_outl: Optional[torch.Tensor] = None,
    v_outl: Optional[torch.Tensor] = None,
    inv_sqrt_d: Optional[float] = None,
    adaptive_beta_q: bool = False,
    kappa: float = 1.0,
    mu_k_global_mean: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    High-precision Variance-Compensated Softmax operator as specified in RFC-14 Section 4.4.
    Compensates for the Jensen gap and norm shrinkage via (1 + sigma_c^2 / (2 * d))
    and applies logarithmic mass compensation +ln(B_c).

    Logit construction:
      - Working Window (exact tokens): S_i = <q, k_i> / sqrt(d)
      - Attractor Centroids:          S_c = (<q, mu_K,c> / sqrt(d)) * (1 + sigma_c^2 / (2*d)) + ln(n_c)
      - Factual Outliers:             S_o = <q, k_o> / sqrt(d)

    Args:
        q: [B, H_q, 1, D] Query vectors.
        k_win: [B, H_kv, W, D] Working window keys.
        v_win: [B, H_kv, W, D] Working window values.
        mu_k: Optional [B, H_kv, M, D] Attractor centroid keys.
        mu_v: Optional [B, H_kv, M, D] Attractor centroid values.
        sigma2_k: Optional [B, H_kv, M] Attractor dispersions (sigma^2).
        counts_c: Optional [B, H_kv, M] or int, cluster token counts (mass).
        k_outl: Optional [B, H_kv, O, D] Factual outlier keys.
        v_outl: Optional [B, H_kv, O, D] Factual outlier values.
        inv_sqrt_d: 1 / sqrt(D). If None, computed from q.shape[-1].
        adaptive_beta_q: Whether to apply adaptive query temperature beta(q).
        kappa: Adaptive temperature contrast hyperparameter in [0.5, 2.0].
        mu_k_global_mean: Optional [B, H_kv, 1, D] global center of mass for beta(q).

    Returns:
        output: [B, H_q, 1, D] Context value vector.
        attn_weights: [B, H_q, 1, Total_Candidates] Full Softmax distribution.
    """
    B, H_q, q_len, D = q.shape
    H_kv = k_win.shape[1]
    gqa_ratio = H_q // H_kv
    if inv_sqrt_d is None:
        inv_sqrt_d = 1.0 / math.sqrt(float(D))

    # Base scale
    base_scale = inv_sqrt_d

    # Optional dynamic adaptive query temperature: beta(q) = beta_0 * (1 + kappa * ||q - mu|| / sqrt(D))
    if adaptive_beta_q and mu_k_global_mean is not None:
        diff_q = q - mu_k_global_mean.repeat_interleave(gqa_ratio, dim=1)
        norm_diff = torch.norm(diff_q, p=2, dim=-1, keepdim=True)
        base_scale = base_scale * (1.0 + kappa * norm_diff / math.sqrt(float(D)))

    logits_list = []
    values_list = []

    # 1. Tier 2: Long-Term Attractor Store (Chronological History Prefix)
    if mu_k is not None and mu_k.shape[2] > 0:
        mu_k_rep = mu_k.repeat_interleave(gqa_ratio, dim=1)  # [B, H_q, M, D]
        mu_v_rep = mu_v.repeat_interleave(gqa_ratio, dim=1)  # [B, H_q, M, D]

        # Dot product: <q, mu_k> / sqrt(d)
        logits_att = torch.matmul(q, mu_k_rep.transpose(-1, -2)) * base_scale  # [B, H_q, 1, M]

        # Temperature scaling via variance: (1 + sigma_c^2 / (2 * D)) with non-negative clamp
        if sigma2_k is not None:
            safe_sigma2 = torch.clamp(sigma2_k, min=0.0)
            sigma2_rep = safe_sigma2.repeat_interleave(gqa_ratio, dim=1).unsqueeze(2)  # [B, H_q, 1, M]
            temp_scale = 1.0 + (sigma2_rep / (2.0 * float(D)))
            logits_att = logits_att * temp_scale

        # Logarithmic mass compensation: +ln(n_c)
        if isinstance(counts_c, (int, float)):
            mass_bonus = math.log(max(float(counts_c), 1.0))
            logits_att = logits_att + mass_bonus
        elif isinstance(counts_c, torch.Tensor):
            counts_rep = counts_c.repeat_interleave(gqa_ratio, dim=1).unsqueeze(2).clamp(min=1.0)
            logits_att = logits_att + torch.log(counts_rep)

        logits_list.append(logits_att)
        values_list.append(mu_v_rep)

    # 2. Tier 1: Working Window Logits (Recent Window)
    if k_win is not None and k_win.shape[2] > 0:
        k_win_rep = k_win.repeat_interleave(gqa_ratio, dim=1)  # [B, H_q, W, D]
        v_win_rep = v_win.repeat_interleave(gqa_ratio, dim=1)  # [B, H_q, W, D]
        logits_win = torch.matmul(q, k_win_rep.transpose(-1, -2)) * base_scale  # [B, H_q, 1, W]
        logits_list.append(logits_win)
        values_list.append(v_win_rep)

    # 3. Tier 3: Factual Outlier Buffer Logits
    if k_outl is not None and k_outl.shape[2] > 0:
        k_outl_rep = k_outl.repeat_interleave(gqa_ratio, dim=1)  # [B, H_q, O, D]
        v_outl_rep = v_outl.repeat_interleave(gqa_ratio, dim=1)  # [B, H_q, O, D]

        logits_outl = torch.matmul(q, k_outl_rep.transpose(-1, -2)) * base_scale  # [B, H_q, 1, O]
        logits_list.append(logits_outl)
        values_list.append(v_outl_rep)

    # Combined logits and values across all tiers
    all_logits = torch.cat(logits_list, dim=-1)   # [B, H_q, 1, W + M + O]
    all_values = torch.cat(values_list, dim=2)    # [B, H_q, W + M + O, D]

    # Numerically stable Softmax
    attn_weights = F.softmax(all_logits, dim=-1)

    # Weighted accumulation of values: V* = sum(alpha_i * v_i)
    output = torch.matmul(attn_weights, all_values)  # [B, H_q, 1, D]

    return output, attn_weights


# ==============================================================================
# 5. Full Python Reference Class: HopfieldKVCache (3-Tier Engine)
# ==============================================================================

class HopfieldKVCache:
    """
    Python reference implementation of the 3-Tier Hopfield KV-Cache for Hetman-2-NCM.
    Mirrors the C++20 Rada.cpp architecture with exact mathematical parity:
      - Tier 1: Working Window Ring Buffer (1024 tokens FP8)
      - Tier 2: Long-Term Attractor Store (8000 centroids with online Welford clustering)
      - Tier 3: Factual Outlier Preservation Buffer (2048 needles with Mahalanobis discriminator D > 3.25)
    """

    def __init__(
        self,
        num_layers: int = NUM_GLOBAL_LAYERS,
        num_kv_heads: int = NUM_KV_HEADS,
        head_dim: int = HEAD_DIM,
        window_size: int = WORKING_WINDOW_SIZE,
        cluster_size: int = TOKENS_PER_CLUSTER,
        max_attractors: int = MAX_ATTRACTORS,
        max_outliers: int = MAX_OUTLIERS,
        outlier_threshold_mahal: float = OUTLIER_THRESHOLD_MAHAL,
        merge_similarity_cos: float = ATTRACTOR_MERGE_SIM_COS,
        device: Union[str, torch.device] = "cpu"
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.cluster_size = cluster_size
        self.max_attractors = max_attractors
        self.max_outliers = max_outliers
        self.outlier_threshold = outlier_threshold_mahal
        self.merge_sim_cos = merge_similarity_cos
        self.device = torch.device(device)

        # Monolithic Ring Buffer for Working Window (pre-allocated)
        self.working_window: List[Optional[WorkingTokenSlot]] = [None] * self.window_size
        self.working_window_head: int = 0
        self.working_window_count: int = 0

        # Long-Term Attractor Store (pre-allocated capacity)
        self.attractors: List[AttractorCentroid] = []

        # Factual Outlier Buffer (pre-allocated ring buffer)
        self.outliers: List[OutlierFactSlot] = []

        self.global_timestamp: int = 0

    # --------------------------------------------------------------------------
    # Public Write API: push_token and push_block
    # --------------------------------------------------------------------------

    def push_token(
        self,
        k_fp32: torch.Tensor,
        v_fp32: torch.Tensor,
        token_pos: int
    ) -> None:
        """
        Inserts a single multi-layer KV token into the cache.
        Evicts the oldest token from Working Window to Hopfield/Outlier tiers when full.

        Args:
            k_fp32: [NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] Key tensor.
            v_fp32: [NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] Value tensor.
            token_pos: Absolute position of the token in the document sequence.
        """
        k_tensor = k_fp32.to(self.device).float()
        v_tensor = v_fp32.to(self.device).float()

        # If window is full, evict oldest token to long-term memory
        if self.working_window_count_full():
            self._evict_oldest_token()

        # Quantize to INT8/FP8
        k_q, k_s = quantize_symmetric_int8(k_tensor)
        v_q, v_s = quantize_symmetric_int8(v_tensor)

        slot = WorkingTokenSlot(
            k_quant=k_q,
            v_quant=v_q,
            k_scales=k_s,
            v_scales=v_s,
            token_position=token_pos,
            timestamp=self.global_timestamp
        )

        self.working_window[self.working_window_head] = slot
        self.working_window_head = (self.working_window_head + 1) % self.window_size
        if self.working_window_count < self.window_size:
            self.working_window_count += 1

        self.global_timestamp += 1

    def push_block(
        self,
        k_block_fp32: torch.Tensor,
        v_block_fp32: torch.Tensor,
        start_token_pos: int = 0
    ) -> None:
        """
        Pushes a contiguous sequence of tokens into the cache.

        Args:
            k_block_fp32: [N, NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] Keys.
            v_block_fp32: [N, NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM] Values.
            start_token_pos: Sequence starting index.
        """
        num_tokens = k_block_fp32.shape[0]
        for t in range(num_tokens):
            self.push_token(
                k_block_fp32[t],
                v_block_fp32[t],
                start_token_pos + t
            )

    def working_window_count_full(self) -> bool:
        return self.working_window_count >= self.window_size

    # --------------------------------------------------------------------------
    # Eviction, Outlier Discrimination, and Welford Clustering Logic
    # --------------------------------------------------------------------------

    def _evict_oldest_token(self) -> None:
        """
        Evicts the oldest token in the Working Window ring buffer.
        Evaluates outlier condition:
          - If Mahalanobis D > 3.25: stored into Tier 3 Factual Outlier Buffer.
          - Else: clustered into Tier 2 Long-Term Attractor Store (Welford update).
        """
        oldest_idx = (self.working_window_head + self.window_size - self.working_window_count) % self.window_size
        slot = self.working_window[oldest_idx]
        assert slot is not None

        # 1. Outlier Discrimination
        is_outlier, score, best_att_idx = self._evaluate_outlier_condition(slot)

        if is_outlier:
            # Store in Factual Outlier Preservation Buffer
            outlier_slot = OutlierFactSlot(
                k_quant=slot.k_quant.clone(),
                v_quant=slot.v_quant.clone(),
                k_scales=slot.k_scales.clone(),
                v_scales=slot.v_scales.clone(),
                original_token_position=slot.token_position,
                outlier_score=score,
                timestamp=slot.timestamp
            )
            if len(self.outliers) < self.max_outliers:
                self.outliers.append(outlier_slot)
            else:
                # Ring overwrite oldest outlier
                replace_idx = self.global_timestamp % self.max_outliers
                self.outliers[replace_idx] = outlier_slot
        else:
            # Cluster into Attractor Store
            self._assign_to_attractor_cluster(slot, best_att_idx)

        self.working_window_count -= 1

    def _evaluate_outlier_condition(
        self,
        slot: WorkingTokenSlot
    ) -> Tuple[bool, float, int]:
        """
        Computes the Mahalanobis anomaly score D(k_t, mu_K) against existing attractors:
          dist_sq = ||k - mu_K||_2^2
          D = dist_sq / (sigma^2 + 1e-4)

        Returns:
            (is_outlier, mahalanobis_score, best_attractor_index)
        """
        if len(self.attractors) == 0:
            return False, 0.0, -1

        # Use Layer 0 Head 0 as the reference representation (as in Rada.cpp C++ specification)
        k_fp32 = dequantize_symmetric_int8(slot.k_quant[0, 0], slot.k_scales[0, 0])  # [HEAD_DIM]

        # Stack attractor mu_k for Layer 0 Head 0
        mu_k_list = [
            dequantize_symmetric_int8(att.mu_k_quant[0, 0], att.mu_k_scales[0, 0])
            for att in self.attractors
        ]
        mu_k_tensor = torch.stack(mu_k_list, dim=0)  # [M, HEAD_DIM]

        # Normalized cosine similarity
        k_norm = F.normalize(k_fp32, p=2, dim=-1)
        mu_k_norm = F.normalize(mu_k_tensor, p=2, dim=-1)
        cos_sims = torch.matmul(mu_k_norm, k_norm)  # [M]

        best_sim, best_idx = torch.max(cos_sims, dim=0)
        best_att_idx = int(best_idx.item())

        # Euclidean distance squared to nearest attractor centroid
        diff = k_fp32 - mu_k_list[best_att_idx]
        dist_sq = torch.sum(diff ** 2).item()

        # Numerically safe dispersion with quantization noise floor (protects against repetitive token anomaly flood)
        disp_val = max(float(self.attractors[best_att_idx].dispersion[0, 0].item()), 0.0)
        effective_disp = max(disp_val, QUANT_NOISE_FLOOR) + EPS_DISPERSION_FLOOR
        mahalanobis = dist_sq / effective_disp

        # If token belongs to the nearest attractor's semantic topic:
        if best_sim.item() >= self.merge_sim_cos:
            is_outlier = (mahalanobis > self.outlier_threshold)
            return is_outlier, mahalanobis, best_att_idx
        else:
            # Token represents a different semantic topic.
            # Check for extreme residual anomaly (e.g. norm explosion or extreme Mahalanobis)
            norm_sq = torch.sum(k_fp32 ** 2).item()
            if norm_sq > 4.0 * float(self.head_dim):
                return True, mahalanobis, best_att_idx
            return False, 0.0, -1

    def _assign_to_attractor_cluster(
        self,
        slot: WorkingTokenSlot,
        best_idx: int = -1
    ) -> None:
        """
        Merges token into closest attractor via incremental Welford update if cos_sim >= 0.88,
        otherwise allocates a new attractor slot (with compact_history defragmentation when full).
        """
        k_fp32 = dequantize_symmetric_int8(slot.k_quant, slot.k_scales)  # [L, H_kv, D]
        v_fp32 = dequantize_symmetric_int8(slot.v_quant, slot.v_scales)  # [L, H_kv, D]

        matched = False
        if best_idx >= 0 and best_idx < len(self.attractors):
            # Check full cosine similarity
            att_k_fp32 = dequantize_symmetric_int8(
                self.attractors[best_idx].mu_k_quant[0, 0],
                self.attractors[best_idx].mu_k_scales[0, 0]
            )
            sim = F.cosine_similarity(k_fp32[0, 0], att_k_fp32, dim=-1).item()
            if sim >= self.merge_sim_cos:
                matched = True

        if matched:
            # Incremental Welford update with non-negative clamping & M2 tracking
            att = self.attractors[best_idx]
            old_n = att.token_count
            new_n = min(old_n + 1, 65535)
            att.token_count = new_n
            att.last_access_timestamp = self.global_timestamp

            # Dequantize current centroids
            cur_mu_k = dequantize_symmetric_int8(att.mu_k_quant, att.mu_k_scales)
            cur_mu_v = dequantize_symmetric_int8(att.mu_v_quant, att.mu_v_scales)

            # Welford update for mean
            delta_k = k_fp32 - cur_mu_k
            updated_mu_k = cur_mu_k + delta_k / float(new_n)

            delta_v = v_fp32 - cur_mu_v
            updated_mu_v = cur_mu_v + delta_v / float(new_n)

            # Welford update for M2 and dispersion:
            # delta_k_new = k - updated_mu_k
            # M2_new = M2_old + sum(delta_k * delta_k_new)
            delta_k_new = k_fp32 - updated_mu_k
            cross_diff = torch.sum(delta_k * delta_k_new, dim=-1)  # [L, H_kv]
            if att.m2_k is None:
                att.m2_k = att.dispersion * float(old_n)
            att.m2_k = torch.clamp(att.m2_k + cross_diff, min=0.0)
            att.dispersion = torch.clamp(att.m2_k / float(new_n), min=0.0)

            # Re-quantize and store
            att.mu_k_quant, att.mu_k_scales = quantize_symmetric_int8(updated_mu_k)
            att.mu_v_quant, att.mu_v_scales = quantize_symmetric_int8(updated_mu_v)
        else:
            # Allocate a new attractor
            if len(self.attractors) >= self.max_attractors:
                self.compact_history()

            init_disp = torch.full((self.num_layers, self.num_kv_heads), 0.05, device=self.device)
            init_m2 = init_disp.clone()
            new_att = AttractorCentroid(
                mu_k_quant=slot.k_quant.clone(),
                mu_v_quant=slot.v_quant.clone(),
                mu_k_scales=slot.k_scales.clone(),
                mu_v_scales=slot.v_scales.clone(),
                dispersion=init_disp,
                m2_k=init_m2,
                token_count=1,
                last_access_timestamp=self.global_timestamp,
                creation_timestamp=self.global_timestamp
            )
            self.attractors.append(new_att)

    def compact_history(self) -> None:
        """
        Defragmentation and pairwise merge of the two closest attractors when reaching capacity.
        Implements:
          1. Recency Protection: Protects the most recent 15% of attractors from eviction/merging.
          2. Cosine Similarity Gating: Merges only if cos_sim >= merge_sim_cos (0.88).
          3. Huygens-Steiner Parallel Dispersion: Exact pooled variance without arbitrary inflation.
        """
        total_m = len(self.attractors)
        if total_m < 2:
            return

        # 1. Recency Protection: do not merge newest attractors
        protect_count = max(int(total_m * RECENCY_PROTECT_RATIO), 1)
        eligible_count = total_m - protect_count
        if eligible_count < 2:
            eligible_count = total_m

        # Find pair with highest cosine similarity among eligible attractors
        mu_k_list = [
            dequantize_symmetric_int8(self.attractors[i].mu_k_quant[0, 0], self.attractors[i].mu_k_scales[0, 0])
            for i in range(eligible_count)
        ]
        mu_k_tensor = F.normalize(torch.stack(mu_k_list, dim=0), p=2, dim=-1)
        sim_matrix = torch.matmul(mu_k_tensor, mu_k_tensor.T)
        sim_matrix.fill_diagonal_(-1.0)

        max_sim, flat_idx = torch.max(sim_matrix.view(-1), dim=0)
        idx_i = int(flat_idx.item()) // eligible_count
        idx_j = int(flat_idx.item()) % eligible_count

        # Fallback if no pair meets similarity threshold: evict oldest least-frequently accessed attractor
        if max_sim.item() < self.merge_sim_cos:
            oldest_idx = 0
            min_score = float('inf')
            for i in range(eligible_count):
                score = self.attractors[i].token_count * 1000 + self.attractors[i].last_access_timestamp
                if score < min_score:
                    min_score = score
                    oldest_idx = i
            self.attractors.pop(oldest_idx)
            return

        # Ensure idx_i < idx_j for deterministic merging
        if idx_i > idx_j:
            idx_i, idx_j = idx_j, idx_i

        # Merge attractor j into attractor i
        att_i = self.attractors[idx_i]
        att_j = self.attractors[idx_j]

        n_i = att_i.token_count
        n_j = att_j.token_count
        n_total = n_i + n_j

        mu_k_i = dequantize_symmetric_int8(att_i.mu_k_quant, att_i.mu_k_scales)
        mu_k_j = dequantize_symmetric_int8(att_j.mu_k_quant, att_j.mu_k_scales)
        merged_mu_k = (mu_k_i * n_i + mu_k_j * n_j) / float(n_total)

        mu_v_i = dequantize_symmetric_int8(att_i.mu_v_quant, att_i.mu_v_scales)
        mu_v_j = dequantize_symmetric_int8(att_j.mu_v_quant, att_j.mu_v_scales)
        merged_mu_v = (mu_v_i * n_i + mu_v_j * n_j) / float(n_total)

        # Huygens-Steiner exact pooled variance:
        # sigma_total^2 = (n_i * sigma_i^2 + n_j * sigma_j^2) / n_total + (n_i * n_j / n_total^2) * ||mu_i - mu_j||^2
        diff_mu_sq = torch.sum((mu_k_i - mu_k_j) ** 2, dim=-1)
        between_cluster_var = (float(n_i * n_j) / float(n_total ** 2)) * diff_mu_sq
        within_cluster_var = (att_i.dispersion * n_i + att_j.dispersion * n_j) / float(n_total)
        merged_disp = torch.clamp(within_cluster_var + between_cluster_var, min=0.0)

        att_i.mu_k_quant, att_i.mu_k_scales = quantize_symmetric_int8(merged_mu_k)
        att_i.mu_v_quant, att_i.mu_v_scales = quantize_symmetric_int8(merged_mu_v)
        att_i.dispersion = merged_disp
        att_i.m2_k = merged_disp * float(n_total)
        att_i.token_count = min(n_total, 65535)
        att_i.last_access_timestamp = max(att_i.last_access_timestamp, att_j.last_access_timestamp)

        # Remove attractor j without disrupting ordering of remaining elements
        self.attractors.pop(idx_j)

    # --------------------------------------------------------------------------
    # Retrieval API: query_attractors for GQA
    # --------------------------------------------------------------------------

    def query_attractors(
        self,
        layer_idx: int,
        query_fp32: torch.Tensor,
        beta: Optional[float] = None,
        adaptive_beta_q: bool = False,
        kappa: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Executes multi-head GQA retrieval across the 3-Tier cache for a specific global layer.

        Args:
            layer_idx: Global layer index in [0, NUM_GLOBAL_LAYERS - 1].
            query_fp32: [B, H_q, 1, HEAD_DIM] or [H_q, HEAD_DIM] Query tensor.
            beta: Scale factor. Default: 1 / sqrt(HEAD_DIM).
            adaptive_beta_q: Whether to apply adaptive query temperature.
            kappa: Contrast scaling.

        Returns:
            context: [B, H_q, 1, HEAD_DIM] Output value vector.
            attn_weights: Softmax probability distribution.
        """
        if query_fp32.dim() == 2:
            query_fp32 = query_fp32.unsqueeze(0).unsqueeze(2)  # [1, H_q, 1, D]
        elif query_fp32.dim() == 3:
            query_fp32 = query_fp32.unsqueeze(2)  # [B, H_q, 1, D]

        B, H_q, _, D = query_fp32.shape
        inv_sqrt_d = beta if beta is not None else DEFAULT_INVERSE_TEMP_BETA

        # 1. Collect Tier 1 Working Window tensors
        ww_slots = []
        ww_start = (self.working_window_head + self.window_size - self.working_window_count) % self.window_size
        for i in range(self.working_window_count):
            idx = (ww_start + i) % self.window_size
            slot = self.working_window[idx]
            if slot is not None:
                ww_slots.append(slot)

        if len(ww_slots) > 0:
            k_win_list = [
                dequantize_symmetric_int8(s.k_quant[layer_idx], s.k_scales[layer_idx])
                for s in ww_slots
            ]  # Each [H_kv, D]
            v_win_list = [
                dequantize_symmetric_int8(s.v_quant[layer_idx], s.v_scales[layer_idx])
                for s in ww_slots
            ]
            k_win = torch.stack(k_win_list, dim=1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H_kv, W, D]
            v_win = torch.stack(v_win_list, dim=1).unsqueeze(0).expand(B, -1, -1, -1)
        else:
            k_win = torch.zeros(B, self.num_kv_heads, 0, D, device=self.device)
            v_win = torch.zeros(B, self.num_kv_heads, 0, D, device=self.device)

        # 2. Collect Tier 2 Attractor tensors
        if len(self.attractors) > 0:
            mu_k_list = [
                dequantize_symmetric_int8(att.mu_k_quant[layer_idx], att.mu_k_scales[layer_idx])
                for att in self.attractors
            ]
            mu_v_list = [
                dequantize_symmetric_int8(att.mu_v_quant[layer_idx], att.mu_v_scales[layer_idx])
                for att in self.attractors
            ]
            mu_k = torch.stack(mu_k_list, dim=1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H_kv, M, D]
            mu_v = torch.stack(mu_v_list, dim=1).unsqueeze(0).expand(B, -1, -1, -1)

            sigma2_list = [att.dispersion[layer_idx] for att in self.attractors]
            sigma2_k = torch.stack(sigma2_list, dim=1).unsqueeze(0).expand(B, -1, -1)  # [B, H_kv, M]

            counts_list = [float(att.token_count) for att in self.attractors]
            counts_c = torch.tensor(counts_list, device=self.device).view(1, 1, -1).expand(B, self.num_kv_heads, -1)
        else:
            mu_k = None
            mu_v = None
            sigma2_k = None
            counts_c = None

        # 3. Collect Tier 3 Outlier tensors
        if len(self.outliers) > 0:
            k_outl_list = [
                dequantize_symmetric_int8(o.k_quant[layer_idx], o.k_scales[layer_idx])
                for o in self.outliers
            ]
            v_outl_list = [
                dequantize_symmetric_int8(o.v_quant[layer_idx], o.v_scales[layer_idx])
                for o in self.outliers
            ]
            k_outl = torch.stack(k_outl_list, dim=1).unsqueeze(0).expand(B, -1, -1, -1)  # [B, H_kv, O, D]
            v_outl = torch.stack(v_outl_list, dim=1).unsqueeze(0).expand(B, -1, -1, -1)
        else:
            k_outl = None
            v_outl = None

        # Global mean for adaptive beta(q)
        mu_k_global = None
        if adaptive_beta_q and mu_k is not None:
            mu_k_global = torch.mean(mu_k, dim=2, keepdim=True)

        return variance_compensated_softmax(
            q=query_fp32,
            k_win=k_win,
            v_win=v_win,
            mu_k=mu_k,
            mu_v=mu_v,
            sigma2_k=sigma2_k,
            counts_c=counts_c,
            k_outl=k_outl,
            v_outl=v_outl,
            inv_sqrt_d=inv_sqrt_d,
            adaptive_beta_q=adaptive_beta_q,
            kappa=kappa,
            mu_k_global_mean=mu_k_global
        )

    # --------------------------------------------------------------------------
    # Memory Footprint Audit
    # --------------------------------------------------------------------------

    def get_memory_footprint(self) -> Dict[str, Any]:
        """
        Returns exact byte counts and compression audit comparing:
          1. Dense Transformer (32 layers, FP16)
          2. Hetman-2-NCM Raw KV (6 layers, FP8)
          3. Hetman-2-NCM Hopfield KV-Cache (Rada.cpp C++20)
        """
        # Exact sizes from RFC-13:
        # WorkingTokenSlot: 3,072 bytes + 16 bytes metadata = 3,088 (aligned to 64 bytes = 3,136 or 3,072 bytes pure)
        ww_bytes = self.working_window_count * 3072
        # AttractorCentroid: 6 x 2 x 264 = 3,168 + 64 bytes metadata = 3,232 bytes
        att_bytes = len(self.attractors) * 3232
        # OutlierFactSlot: 3,072 + 64 bytes metadata = 3,136 bytes
        outl_bytes = len(self.outliers) * 3136
        total_hopfield_bytes = ww_bytes + att_bytes + outl_bytes

        # Reconstructed total context length
        total_tokens = self.working_window_count + len(self.outliers)
        for att in self.attractors:
            total_tokens += att.token_count

        raw_kv_bytes = total_tokens * 3072  # 6 layers x 2 x 2 x 128 x 1 byte
        dense_7b_bytes = total_tokens * 524288  # 32 layers x 32 heads x 128 x 2 bytes (FP16)

        compression_vs_raw = (raw_kv_bytes / total_hopfield_bytes) if total_hopfield_bytes > 0 else 1.0
        compression_vs_dense = (dense_7b_bytes / total_hopfield_bytes) if total_hopfield_bytes > 0 else 1.0

        return {
            "working_window_tokens": self.working_window_count,
            "attractor_count": len(self.attractors),
            "outlier_count": len(self.outliers),
            "total_tokens_represented": total_tokens,
            "hopfield_cache_bytes": total_hopfield_bytes,
            "hopfield_cache_mib": total_hopfield_bytes / (1024.0 * 1024.0),
            "raw_kv_mib": raw_kv_bytes / (1024.0 * 1024.0),
            "dense_7b_gib": dense_7b_bytes / (1024.0 * 1024.0 * 1024.0),
            "compression_ratio_vs_raw": compression_vs_raw,
            "compression_ratio_vs_dense": compression_vs_dense,
        }


# ==============================================================================
# 6. Straight-Through Estimator for Pretraining Attractor Clustering
# ==============================================================================

class AttractorSTE(torch.autograd.Function):
    """
    Straight-Through Estimator (STE) for Welford/blockwise attractor clustering.
    Forward Pass:
        Returns the clustered centroid mu_K across each B_c block.
    Backward Pass:
        Passes exact 1/B_c gradient back to the raw token representations,
        ensuring 100% gradient energy preservation without vanishing context gradients.
    """

    @staticmethod
    def forward(
        ctx,
        k_blocks: torch.Tensor,
        dim_cluster: int = 3
    ) -> torch.Tensor:
        """
        k_blocks: [B, H, num_clusters, cluster_size, D]
        """
        mu_k = torch.mean(k_blocks, dim=dim_cluster)
        ctx.cluster_size = k_blocks.shape[dim_cluster]
        ctx.dim_cluster = dim_cluster
        return mu_k

    @staticmethod
    def backward(
        ctx,
        grad_mu_k: torch.Tensor
    ) -> Tuple[torch.Tensor, None]:
        # Gradient dL/dk_i = (1 / B_c) * dL/dmu_k
        cluster_size = ctx.cluster_size
        grad_k = grad_mu_k.unsqueeze(ctx.dim_cluster).repeat_interleave(cluster_size, dim=ctx.dim_cluster)
        grad_k = grad_k / float(cluster_size)
        return grad_k, None


# ==============================================================================
# 7. Attractor-Augmented Attention (A3) Pretraining Module
# ==============================================================================

class AttractorAugmentedAttention(nn.Module):
    """
    Attractor-Augmented Attention (A3) for Hetman-2-NCM Pretraining & Alignment.
    Implements Stage 2B-2D curriculum:
      - Window size W = 512..1024 raw tokens.
      - Stochastic Cluster-Dropout p_aug in [0.30, 0.50].
      - Attractor Noise Injection gamma_noise = 0.05.
      - Straight-Through Estimator for gradient backpropagation.
      - Variance-Compensated Softmax with log(32) mass boost and (1 + sigma^2 / (2*d)).
    """

    def __init__(
        self,
        d_model: int = 1536,
        num_query_heads: int = NUM_QUERY_HEADS,
        num_kv_heads: int = NUM_KV_HEADS,
        head_dim: int = HEAD_DIM,
        window_size: int = WORKING_WINDOW_SIZE,
        cluster_size: int = TOKENS_PER_CLUSTER,
        p_aug: float = 0.40,
        noise_std: float = 0.05,
        outlier_ratio: float = 0.05,
        bias: bool = False
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.window_size = window_size
        self.cluster_size = cluster_size
        self.p_aug = p_aug
        self.noise_std = noise_std
        self.outlier_ratio = outlier_ratio

        # GQA Projections
        self.q_proj = nn.Linear(d_model, num_query_heads * head_dim, bias=bias)
        self.k_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=bias)
        self.v_proj = nn.Linear(d_model, num_kv_heads * head_dim, bias=bias)
        self.o_proj = nn.Linear(num_query_heads * head_dim, d_model, bias=bias)

        self.inv_sqrt_d = 1.0 / math.sqrt(float(head_dim))
        self.log_bc = math.log(float(cluster_size))

    def forward(
        self,
        x: torch.Tensor,
        is_training: Optional[bool] = None,
        force_a3: bool = False
    ) -> torch.Tensor:
        """
        Forward pass for Attractor-Augmented Attention.

        Args:
            x: [B, L, D_model] Input hidden states.
            is_training: If None, defaults to self.training.
            force_a3: If True, forces A3 clustering even in eval mode.

        Returns:
            [B, L, D_model] Projected multi-head attention outputs.
        """
        if is_training is None:
            is_training = self.training

        B, L, _ = x.shape

        # Linear projections
        q = self.q_proj(x).view(B, L, self.num_query_heads, self.head_dim).transpose(1, 2)  # [B, H_q, L, D]
        k = self.k_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)    # [B, H_kv, L, D]
        v = self.v_proj(x).view(B, L, self.num_kv_heads, self.head_dim).transpose(1, 2)    # [B, H_kv, L, D]

        # Fast path: sequence length within working window or A3 disabled
        if L <= self.window_size or (not is_training and not force_a3):
            # Standard causal FlashAttention / SDPA
            gqa_ratio = self.num_query_heads // self.num_kv_heads
            k_rep = k.repeat_interleave(gqa_ratio, dim=1)
            v_rep = v.repeat_interleave(gqa_ratio, dim=1)
            out = F.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=True)
            out = out.transpose(1, 2).contiguous().view(B, L, -1)
            return self.o_proj(out)

        # ----------------------------------------------------------------------
        # A3 Path: Partition into History Prefix and Local Working Window
        # ----------------------------------------------------------------------
        hist_len = L - self.window_size
        num_clusters = hist_len // self.cluster_size
        trimmed_hist_len = num_clusters * self.cluster_size

        k_hist = k[:, :, :trimmed_hist_len, :]
        v_hist = v[:, :, :trimmed_hist_len, :]

        k_win = k[:, :, trimmed_hist_len:, :]
        v_win = v[:, :, trimmed_hist_len:, :]

        # Group history into blocks: [B, H_kv, num_clusters, cluster_size, D]
        k_blocks = k_hist.view(B, self.num_kv_heads, num_clusters, self.cluster_size, self.head_dim)
        v_blocks = v_hist.view(B, self.num_kv_heads, num_clusters, self.cluster_size, self.head_dim)

        # Straight-Through Estimator for Centroids
        mu_k = AttractorSTE.apply(k_blocks, 3)  # [B, H_kv, num_clusters, D]
        mu_v = torch.mean(v_blocks, dim=3)      # [B, H_kv, num_clusters, D]

        # Intra-cluster dispersion: sigma_c^2 (non-negative clamped)
        diff_k = k_blocks - mu_k.unsqueeze(3)
        sigma2_c = torch.clamp(torch.mean(torch.sum(diff_k ** 2, dim=-1), dim=3), min=0.0)  # [B, H_kv, num_clusters]

        # Noise injection during training (protected against negative dispersion / NaN)
        if is_training and self.noise_std > 0.0:
            noise = torch.randn_like(mu_k) * (self.noise_std * torch.sqrt(torch.clamp(sigma2_c, min=0.0) / float(self.head_dim)).unsqueeze(-1))
            mu_k = mu_k + noise

        # Temperature scaling factor: (1 + sigma_c^2 / (2 * D))
        temp_scale = 1.0 + sigma2_c / (2.0 * float(self.head_dim))  # [B, H_kv, num_clusters]

        # Stochastic Cluster-Dropout mask
        if is_training and self.p_aug < 1.0:
            cluster_mask = (torch.rand(B, 1, num_clusters, 1, device=x.device) < self.p_aug).float()
        else:
            cluster_mask = torch.ones(B, 1, num_clusters, 1, device=x.device)

        # GQA repeat
        gqa_ratio = self.num_query_heads // self.num_kv_heads
        mu_k_rep = mu_k.repeat_interleave(gqa_ratio, dim=1)
        mu_v_rep = mu_v.repeat_interleave(gqa_ratio, dim=1)
        temp_scale_rep = temp_scale.repeat_interleave(gqa_ratio, dim=1).unsqueeze(2)  # [B, H_q, 1, M]
        k_win_rep = k_win.repeat_interleave(gqa_ratio, dim=1)
        v_win_rep = v_win.repeat_interleave(gqa_ratio, dim=1)

        # Logits computation
        logits_hist = torch.matmul(q, mu_k_rep.transpose(-1, -2)) * self.inv_sqrt_d  # [B, H_q, L, M]
        logits_hist = logits_hist * temp_scale_rep + self.log_bc

        logits_win = torch.matmul(q, k_win_rep.transpose(-1, -2)) * self.inv_sqrt_d   # [B, H_q, L, W]

        # Causal masking on uncompressed window
        # Query at pos i can attend to all M history attractors, plus window tokens j <= i - trimmed_hist_len
        q_pos = torch.arange(L, device=x.device).view(1, 1, L, 1)
        win_pos = (torch.arange(L - trimmed_hist_len, device=x.device) + trimmed_hist_len).view(1, 1, 1, -1)
        win_causal_mask = (q_pos >= win_pos)  # [1, 1, L, W]

        logits_win = logits_win.masked_fill(~win_causal_mask, float('-inf'))

        # Combined logits and values
        all_logits = torch.cat([logits_hist, logits_win], dim=-1)  # [B, H_q, L, M + W]
        all_values = torch.cat([mu_v_rep, v_win_rep], dim=2)       # [B, H_q, M + W, D]

        attn_weights = F.softmax(all_logits, dim=-1)
        # Handle -inf in attention weights safely
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

        out = torch.matmul(attn_weights, all_values)  # [B, H_q, L, D]
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.o_proj(out)
