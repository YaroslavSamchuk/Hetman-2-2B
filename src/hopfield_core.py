"""
Speculative Latent-Hopfield Core v2 (SLH-Core v2) for Hetman-2-NCM 2.0B
RFC Specification: RFC-HETMAN-2-NCM-04 (spec/04_SLH_CORE_V2_HOPFIELD_FACTUAL_MEMORY.md)
Target Model: Hetman-2-NCM 2.0B (Neural Concept Model)

Architecture Summary:
  - 16 Independent Functional Groups (Isolating semantic domain basins)
  - Factorized Dual Spherical Codebooks: C1, C2 on S^63 (1024 x 64 each per group)
  - 16,777,216 Virtual Attractor Slots (16 groups x 1024 x 1024)
  - Ternary Attractor Tensor V in {-1, 0, +1}^{16 x 1024 x 1024 x 32}
  - Hardware 2-bit Packed Storage (~128.0 MiB VRAM footprint, 0.25 bytes/weight)
  - TPU MXU Systolic Linear Expansion: 32 -> 128 via semi-orthogonal W_expand
  - Sparse Bilinear Retrieval: Top-8 (C1) x Top-4 (C2) = Top-32 slots per group (0.00305% active space)
  - Modern Continuous Hopfield Associative Energy with beta = 1/sqrt(32) ≈ 0.1767767
  - Non-differentiable Dynamic PID Load Balancer (Aux-Loss-Free, K_i=0.01, K_d=0.005)
  - Online Hot-Patching Engine: hot_patch_fact() for zero-shot factual updates without fine-tuning
  - Factual Anchoring Bridge for Phase 3 Word Decoder (Anti-Hallucination Gate)
"""

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Configuration Dataclass
# ==============================================================================

@dataclass
class SLHCoreConfig:
    """Configuration parameters for SLH-Core v2 Associative Factual Memory."""
    hidden_size: int = 1536
    hopfield_num_groups: int = 16
    hopfield_codebook_size: int = 1024
    hopfield_code_dim: int = 64
    hopfield_val_dim: int = 32
    hopfield_expanded_dim: int = 128
    hopfield_topk_1: int = 8
    hopfield_topk_2: int = 4
    hopfield_beta: float = 0.1767766952966369  # 1.0 / math.sqrt(32)
    hopfield_pid_ki: float = 0.01
    hopfield_pid_kd: float = 0.005
    hopfield_pack_2bit: bool = True
    rms_norm_eps: float = 1e-6


# ==============================================================================
# BitNet b1.58 Ternary Linear Layer
# ==============================================================================

class BitLinear(nn.Linear):
    """
    Ternary 1.58-bit Linear Layer (BitNet b1.58) with symmetric weight quantization
    to {-1, 0, +1} and dynamic 8-bit activation quantization with STE.
    Includes eval weight caching for microsecond edge inference.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__(in_features, out_features, bias=bias)
        self._cached_w_eff: Optional[torch.Tensor] = None
        self._cached_gamma: Optional[torch.Tensor] = None

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cached_w_eff = None
            self._cached_gamma = None
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training and self._cached_w_eff is not None:
            w_eff = self._cached_w_eff
            gamma = self._cached_gamma
        else:
            w = self.weight
            gamma = torch.mean(torch.abs(w)).clamp(min=1e-5)
            w_scaled = w / gamma
            w_quant = torch.clamp(torch.round(w_scaled), -1.0, 1.0)
            # Straight-Through Estimator (STE) for weights
            w_eff = w + (w_quant - w).detach()
            if not self.training:
                self._cached_w_eff = w_eff
                self._cached_gamma = gamma

        # Dynamic INT8 activation quantization
        eta = torch.max(torch.abs(x), dim=-1, keepdim=True).values.clamp(min=1e-5)
        x_scaled = x * (127.0 / eta)
        x_quant = torch.clamp(torch.round(x_scaled), -128.0, 127.0)
        # Straight-Through Estimator (STE) for activations
        x_eff = x + (x_quant - x).detach()

        # Linear projection with scale restoration
        y = F.linear(x_eff, w_eff, self.bias)
        y = y * (eta / 127.0) * gamma
        return y


# ==============================================================================
# SLH-Core v2 Production NCM Memory Adapter
# ==============================================================================

class SLHCoreNCMAdapter(nn.Module):
    """
    Speculative Latent-Hopfield Core v2 (SLH-Core v2) Associative Knowledge Memory.
    Production module for Hetman-2-NCM 2.0B architecture.
    """
    def __init__(
        self,
        config: Optional[Any] = None,
        hidden_size: int = 1536,
        num_groups: int = 16,
        codebook_size: int = 1024,
        code_dim: int = 64,
        val_dim: int = 32,
        expanded_dim: int = 128,
        topk_1: int = 8,
        topk_2: int = 4,
        beta: float = 0.1767766952966369,  # 1 / sqrt(32)
        pid_ki: float = 0.01,
        pid_kd: float = 0.005,
        pack_2bit: bool = True,
        rms_norm_eps: float = 1e-6,
    ):
        super().__init__()
        # Ingest configuration if supplied
        if config is not None:
            hidden_size = getattr(config, "hidden_size", hidden_size)
            num_groups = getattr(config, "hopfield_num_groups", getattr(config, "num_groups", num_groups))
            codebook_size = getattr(config, "hopfield_codebook_size", getattr(config, "codebook_size", codebook_size))
            code_dim = getattr(config, "hopfield_code_dim", getattr(config, "code_dim", code_dim))
            val_dim = getattr(config, "hopfield_val_dim", getattr(config, "val_dim", val_dim))
            expanded_dim = getattr(config, "hopfield_expanded_dim", getattr(config, "expanded_dim", expanded_dim))
            topk_1 = getattr(config, "hopfield_topk_1", getattr(config, "topk_1", topk_1))
            topk_2 = getattr(config, "hopfield_topk_2", getattr(config, "topk_2", topk_2))
            beta = getattr(config, "hopfield_beta", getattr(config, "beta", beta))
            pid_ki = getattr(config, "hopfield_pid_ki", getattr(config, "pid_ki", pid_ki))
            pid_kd = getattr(config, "hopfield_pid_kd", getattr(config, "pid_kd", pid_kd))
            pack_2bit = getattr(config, "hopfield_pack_2bit", getattr(config, "pack_2bit", pack_2bit))
            rms_norm_eps = getattr(config, "rms_norm_eps", rms_norm_eps)

        self.hidden_size = hidden_size
        self.num_groups = num_groups
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.val_dim = val_dim
        self.expanded_dim = expanded_dim
        self.topk_1 = topk_1
        self.topk_2 = topk_2
        self.beta = beta
        self.pid_ki = pid_ki
        self.pid_kd = pid_kd
        self.pack_2bit = pack_2bit
        self.rms_norm_eps = rms_norm_eps

        # Subquery dimension: 16 groups * 2 codebooks * 64 dimensions = 2048
        self.subquery_dim = self.num_groups * 2 * self.code_dim

        # 1. Subquery Projector and Normalizer
        self.q_proj = BitLinear(self.hidden_size, self.subquery_dim, bias=False)
        self.q_norm = nn.RMSNorm(self.subquery_dim, eps=self.rms_norm_eps)

        # 2. Dual Spherical Codebooks C1 & C2: [16, 1024, 64]
        self.C1 = nn.Parameter(torch.empty(self.num_groups, self.codebook_size, self.code_dim))
        self.C2 = nn.Parameter(torch.empty(self.num_groups, self.codebook_size, self.code_dim))

        # 3. Dynamic PID Load Balancer Buffers (Non-differentiable state)
        self.register_buffer("b1", torch.zeros(self.num_groups, self.codebook_size))
        self.register_buffer("b2", torch.zeros(self.num_groups, self.codebook_size))
        self.register_buffer("prev_err1", torch.zeros(self.num_groups, self.codebook_size))
        self.register_buffer("prev_err2", torch.zeros(self.num_groups, self.codebook_size))
        target_f1_init = self.topk_1 / float(self.codebook_size)
        target_f2_init = self.topk_2 / float(self.codebook_size)
        self.register_buffer("f1_ema", torch.full((self.num_groups, self.codebook_size), target_f1_init))
        self.register_buffer("f2_ema", torch.full((self.num_groups, self.codebook_size), target_f2_init))

        # Static 256-element LUT table for sub-millisecond CPU dequantization
        decode_map = torch.tensor([0.0, 1.0, -1.0, 0.0], dtype=torch.float32)
        lut = torch.zeros(256, 4, dtype=torch.float32)
        for b in range(256):
            lut[b, 0] = decode_map[b & 0x03]
            lut[b, 1] = decode_map[(b >> 2) & 0x03]
            lut[b, 2] = decode_map[(b >> 4) & 0x03]
            lut[b, 3] = decode_map[(b >> 6) & 0x03]
        self.register_buffer("lut_2bit", lut, persistent=False)

        # Bit shift constant for 2-bit vectorized decoding
        self.register_buffer(
            "bit_shifts",
            torch.tensor([0, 2, 4, 6], dtype=torch.uint8).view(1, 1, 1, 1, 1, 4),
            persistent=False
        )

        # Codebook norm cache for inference
        self._cached_c1_norm: Optional[torch.Tensor] = None
        self._cached_c2_norm: Optional[torch.Tensor] = None

        # 4. Ternary Knowledge Attractor Tensor V: 16.78M slots (val_dim = 32)
        # When pack_2bit=True: 2 bits per value = 8 bytes per slot = 134,217,728 bytes (128.0 MiB)
        # When pack_2bit=False: full FP32 tensor for end-to-end gradient pre-training (2.0 GiB)
        if self.pack_2bit:
            self.register_buffer(
                "V_packed",
                torch.zeros((self.num_groups, self.codebook_size, self.codebook_size, self.val_dim // 4), dtype=torch.uint8)
            )
            self.V = None
        else:
            self.V = nn.Parameter(
                torch.empty(self.num_groups, self.codebook_size, self.codebook_size, self.val_dim)
            )
            self.register_buffer("V_packed", None)

        # 5. TPU MXU Systolic Linear Expansion: [16, 32, 128]
        self.W_expand = nn.Parameter(torch.empty(self.num_groups, self.val_dim, self.expanded_dim))

        # 6. Shared Anchor Attractors: [16, 128] (Representation Baseline Stability)
        self.V_shared = nn.Parameter(torch.empty(self.num_groups, self.expanded_dim))

        # 7. Knowledge Output Projection: 16 * 128 = 2048 -> 1536
        self.out_proj = BitLinear(self.num_groups * self.expanded_dim, self.hidden_size, bias=False)

        # 8. Phase 3 Decoder Factual Anti-Hallucination Gate
        self.anchor_gate = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)

        self.reset_parameters()

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._cached_c1_norm = None
            self._cached_c2_norm = None
        return self

    def reset_parameters(self):
        """Initializes parameters on the unit hypersphere and semi-orthogonal matrices."""
        self._cached_c1_norm = None
        self._cached_c2_norm = None
        with torch.no_grad():
            # Spherical codebooks C1, C2 normalized on S^63
            for g in range(self.num_groups):
                nn.init.orthogonal_(self.C1[g])
                nn.init.orthogonal_(self.C2[g])
            self.C1.copy_(F.normalize(self.C1, p=2, dim=-1, eps=1e-5))
            self.C2.copy_(F.normalize(self.C2, p=2, dim=-1, eps=1e-5))

            # Initialize ternary attractor values
            if not self.pack_2bit and self.V is not None:
                self.V.uniform_(-1.0, 1.0)
                self.V.copy_(torch.clamp(torch.round(self.V), -1.0, 1.0))
            elif self.pack_2bit and hasattr(self, "V_packed") and self.V_packed is not None:
                self.V_packed.zero_()

            # Semi-orthogonal linear expansion 32 -> 128 (preserves Euclidean norm)
            for g in range(self.num_groups):
                nn.init.orthogonal_(self.W_expand[g])

            # Shared anchor attractors
            self.V_shared.normal_(mean=0.0, std=1.0 / math.sqrt(self.expanded_dim))

    @torch.no_grad()
    def pack_ternary_tensor(self):
        """
        Compresses unpacked FP32 tensor V into the 2-bit packed hardware format (128.0 MiB).
        Frees the 2.0 GiB FP32 tensor immediately.
        """
        if hasattr(self, "V") and self.V is not None:
            v_discrete = torch.clamp(torch.round(self.V.data), -1.0, 1.0).to(torch.int8)
            # Code mapping: 0 -> 0, +1 -> 1, -1 -> 2
            v_encoded = torch.where(v_discrete == -1, torch.tensor(2, dtype=torch.int8, device=self.V.device), v_discrete)
            v_reshaped = v_encoded.view(self.num_groups, self.codebook_size, self.codebook_size, self.val_dim // 4, 4)
            # 4 ternary values packed into 1 byte (little-endian bit order)
            packed = (
                (v_reshaped[..., 0] & 0x03) |
                ((v_reshaped[..., 1] & 0x03) << 2) |
                ((v_reshaped[..., 2] & 0x03) << 4) |
                ((v_reshaped[..., 3] & 0x03) << 6)
            ).to(torch.uint8)

            self.register_buffer("V_packed", packed)
            del self.V
            self._parameters.pop("V", None)
            self.V = None
            self.pack_2bit = True

    @torch.no_grad()
    def unpack_ternary_tensor(self):
        """
        Unpacks 2-bit packed buffer V_packed back into an FP32 Parameter V.
        """
        if hasattr(self, "V_packed") and self.V_packed is not None:
            raw_bytes = self.V_packed
            shifts = torch.tensor([0, 2, 4, 6], dtype=torch.uint8, device=raw_bytes.device).view(1, 1, 1, 1, 4)
            codes = (raw_bytes.unsqueeze(-1) >> shifts) & 0x03
            v_unpacked = torch.where(codes == 2, -1.0, codes.to(torch.float32)).view(
                self.num_groups, self.codebook_size, self.codebook_size, self.val_dim
            )
            self.V = nn.Parameter(v_unpacked)
            del self.V_packed
            self._buffers.pop("V_packed", None)
            self.V_packed = None
            self.pack_2bit = False

    def _unpack_subtensor(self, g: int, idx1: torch.Tensor, idx2: torch.Tensor) -> torch.Tensor:
        """
        Fast selective gathering and hardware dequantization of active pairs.
        Reads only 8 bytes per active pair per group.
        
        Args:
            g: Group index (0..15)
            idx1: [B, K, topk_1] indices from C1
            idx2: [B, K, topk_2] indices from C2
            
        Returns:
            v_unpacked: [B, K, topk_1, topk_2, val_dim] ternary micro-facts in {-1.0, 0.0, +1.0}
        """
        B, K, dim1 = idx1.shape
        dim2 = idx2.shape[-1]
        if not self.pack_2bit and self.V is not None:
            Vg = self.V[g]  # [1024, 1024, 32]
            return Vg[idx1.unsqueeze(-1), idx2.unsqueeze(-2)]

        # Packed 2-bit path: read 8 bytes per active (i, j) pair
        V_packed_g = self.V_packed[g]  # [1024, 1024, 8]
        raw_bytes = V_packed_g[idx1.unsqueeze(-1), idx2.unsqueeze(-2)]  # [B, K, dim1, dim2, 8]

        # Sub-microsecond LUT path if available
        if hasattr(self, "lut_2bit") and self.lut_2bit is not None:
            return self.lut_2bit[raw_bytes.long()].view(B, K, dim1, dim2, self.val_dim)

        # Vectorized bit-shift dequantization fallback
        shifts = self.bit_shifts.to(raw_bytes.device)
        codes = (raw_bytes.unsqueeze(-1) >> shifts) & 0x03
        # Decode: 0 -> 0.0, 1 -> +1.0, 2 -> -1.0
        v_unpacked = torch.where(codes == 2, -1.0, codes.to(torch.float32)).view(
            B, K, dim1, dim2, self.val_dim
        )
        return v_unpacked

    def compute_hopfield_energy(
        self,
        pair_scores_flat: torch.Tensor,
        q1: Optional[torch.Tensor] = None,
        q2: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Modern Continuous Hopfield Energy Function (Ramsauer et al., 2020):
            E(q) = - (1 / beta) * LogSumExp(beta * pair_scores) + 0.5 * (||q1||^2 + ||q2||^2)
        
        Args:
            pair_scores_flat: [B, K, num_groups, 32] raw bilinear similarity scores
            q1: Optional [B, K, num_groups, 64] query on S^63
            q2: Optional [B, K, num_groups, 64] query on S^63
            
        Returns:
            energy: [B, K, num_groups] Continuous Hopfield Lyapunov energy per group
        """
        # Log-Sum-Exp contraction over the 32 active attractor states
        lse = torch.logsumexp(self.beta * pair_scores_flat, dim=-1)  # [B, K, num_groups]
        energy = - (1.0 / self.beta) * lse

        # Add quadratic query regularization if provided (for unit vectors on S^63 this is exactly 1.0)
        if q1 is not None and q2 is not None:
            norm_q = 0.5 * (q1.pow(2).sum(dim=-1) + q2.pow(2).sum(dim=-1))
            energy = energy + norm_q
        else:
            # Theoretical constant for unit hypersphere S^63: 0.5 * (1.0 + 1.0) = 1.0
            energy = energy + 1.0

        return energy

    def forward(
        self,
        blueprint_B_k: torch.Tensor,
        phase3_decoder_state: Optional[torch.Tensor] = None,
        return_diagnostics: bool = True
    ) -> Union[Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]], Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward associative retrieval pass.
        
        Args:
            blueprint_B_k: [B, K, hidden_size] or [B, hidden_size] NCM concept blueprint blocks
            phase3_decoder_state: Optional [B, L_dec, hidden_size] token decoder states
            return_diagnostics: Whether to return diagnostic metadata dictionary
            
        Returns:
            If return_diagnostics=True:
                (y_factual_context, commit_loss, diagnostics)
            If return_diagnostics=False:
                (y_factual_context, commit_loss)
        """
        # Handle 2D inputs [B, hidden_size] -> [B, 1, hidden_size]
        is_2d = (blueprint_B_k.dim() == 2)
        if is_2d:
            blueprint_B_k = blueprint_B_k.unsqueeze(1)

        B, K, _ = blueprint_B_k.shape

        # 1. Project and normalize subqueries
        z = self.q_norm(self.q_proj(blueprint_B_k))  # [B, K, 2048]
        z_grouped = z.view(B, K, self.num_groups, 2, self.code_dim)
        q1 = F.normalize(z_grouped[..., 0, :], p=2, dim=-1, eps=1e-5)  # [B, K, 16, 64]
        q2 = F.normalize(z_grouped[..., 1, :], p=2, dim=-1, eps=1e-5)  # [B, K, 16, 64]

        # 2. Dual Codebook Cosine Retrieval (with eval cache)
        if not self.training and self._cached_c1_norm is not None:
            C1_norm = self._cached_c1_norm
            C2_norm = self._cached_c2_norm
        else:
            C1_norm = F.normalize(self.C1, p=2, dim=-1, eps=1e-5)  # [16, 1024, 64]
            C2_norm = F.normalize(self.C2, p=2, dim=-1, eps=1e-5)  # [16, 1024, 64]
            if not self.training:
                self._cached_c1_norm = C1_norm
                self._cached_c2_norm = C2_norm

        S1_raw = torch.einsum('bkgd,gmd->bkgm', q1, C1_norm)  # [B, K, 16, 1024]
        S2_raw = torch.einsum('bkgd,gnd->bkgn', q2, C2_norm)  # [B, K, 16, 1024]

        # 3. Dynamic PID Bias for Aux-Loss-Free Routing (prevents slot starvation)
        S1_routed = S1_raw + self.b1.unsqueeze(0).unsqueeze(0)
        S2_routed = S2_raw + self.b2.unsqueeze(0).unsqueeze(0)

        # 4. Sparse Top-K Selection: Top-8 from C1, Top-4 from C2
        top1_vals, top1_idx = torch.topk(S1_routed, self.topk_1, dim=-1)  # [B, K, 16, 8]
        top2_vals, top2_idx = torch.topk(S2_routed, self.topk_2, dim=-1)  # [B, K, 16, 4]

        # Update non-differentiable PID router biases in training mode
        if self.training:
            with torch.no_grad():
                total_events = max(B * K, 1)
                f1 = torch.zeros(self.num_groups, self.codebook_size, device=blueprint_B_k.device)
                f2 = torch.zeros(self.num_groups, self.codebook_size, device=blueprint_B_k.device)

                ones1 = torch.ones_like(top1_idx, dtype=torch.float32)
                idx1_flat = top1_idx.view(-1, self.num_groups, self.topk_1).permute(1, 0, 2).reshape(self.num_groups, -1)
                val1_flat = ones1.view(-1, self.num_groups, self.topk_1).permute(1, 0, 2).reshape(self.num_groups, -1)
                f1.scatter_add_(1, idx1_flat, val1_flat)
                f1 = f1 / float(total_events)

                ones2 = torch.ones_like(top2_idx, dtype=torch.float32)
                idx2_flat = top2_idx.view(-1, self.num_groups, self.topk_2).permute(1, 0, 2).reshape(self.num_groups, -1)
                val2_flat = ones2.view(-1, self.num_groups, self.topk_2).permute(1, 0, 2).reshape(self.num_groups, -1)
                f2.scatter_add_(1, idx2_flat, val2_flat)
                f2 = f2 / float(total_events)

                target_f1 = self.topk_1 / float(self.codebook_size)
                target_f2 = self.topk_2 / float(self.codebook_size)

                # Smooth EMA update (Low-Pass Filter) to eliminate high-frequency noise & derivative kick
                ema_alpha = 0.05
                self.f1_ema.mul_(1.0 - ema_alpha).add_(f1, alpha=ema_alpha)
                self.f2_ema.mul_(1.0 - ema_alpha).add_(f2, alpha=ema_alpha)

                # Normalized relative error (bounded to [-5.0, 5.0] to prevent bang-bang oscillations)
                err1 = torch.clamp((self.f1_ema - target_f1) / target_f1, -5.0, 5.0)
                err2 = torch.clamp((self.f2_ema - target_f2) / target_f2, -5.0, 5.0)

                # Low-pass filtered derivative term (prevents phase lag and limit cycles)
                d_err1 = 0.2 * (err1 - self.prev_err1)
                d_err2 = 0.2 * (err2 - self.prev_err2)

                # Leaky Integrator decay (prevents integrator windup during long dialogues)
                leak_rate = 0.001
                self.b1.mul_(1.0 - leak_rate)
                self.b2.mul_(1.0 - leak_rate)

                # PID update
                self.b1.sub_(self.pid_ki * err1 + self.pid_kd * d_err1)
                self.b2.sub_(self.pid_ki * err2 + self.pid_kd * d_err2)
                self.b1.clamp_(-0.5, 0.5)
                self.b2.clamp_(-0.5, 0.5)

                self.prev_err1.copy_(err1)
                self.prev_err2.copy_(err2)

        # 5. Joint Bilinear Hopfield Associative Weighting over Top-32 Pairs
        # Use uncorrupted raw similarity scores to preserve authentic associative energies
        raw_top1_vals = torch.gather(S1_raw, dim=-1, index=top1_idx)  # [B, K, 16, 8]
        raw_top2_vals = torch.gather(S2_raw, dim=-1, index=top2_idx)  # [B, K, 16, 4]

        # Cartesian sum: Score(u, v) = S1[i_u] + S2[j_v]
        pair_scores = raw_top1_vals.unsqueeze(-1) + raw_top2_vals.unsqueeze(-2)  # [B, K, 16, 8, 4]
        pair_scores_flat = pair_scores.view(B, K, self.num_groups, self.topk_1 * self.topk_2)  # [B, K, 16, 32]
        
        # Continuous Hopfield Softmax Attention (inverse temperature beta = 1 / sqrt(32))
        attn_weights = F.softmax(self.beta * pair_scores_flat, dim=-1)  # [B, K, 16, 32]

        # Compute Continuous Hopfield Network Energy
        hopfield_energy = self.compute_hopfield_energy(pair_scores_flat, q1, q2)  # [B, K, 16]

        # 6. Anti-Spurious Attractor Dynamics & Vectorized Attractor Retrieval
        # Calculate per-group normalized Shannon entropy H in [0, 1]
        log_attn = torch.log(attn_weights.clamp(min=1e-12))
        entropy_g = - (attn_weights * log_attn).sum(dim=-1) / math.log(32.0)  # [B, K, 16]
        attn_var_g = torch.var(attn_weights, dim=-1)  # [B, K, 16]

        # Group-level jamming gate: suppresses dynamic micro-facts when trapped in uniform spurious superposition
        # Smooth sigmoid gate: 0.0 when selective, 1.0 when jammed
        spurious_jammed_g = (entropy_g > 0.95) | (attn_var_g < 1e-5)
        jam_gate_g = torch.sigmoid((entropy_g - 0.94) * 40.0)  # [B, K, 16]

        if self.pack_2bit and hasattr(self, "lut_2bit") and self.lut_2bit is not None and self.V_packed is not None:
            # High-performance Vectorized LUT path across all 16 groups (Sub-millisecond)
            g_idx_all = torch.arange(self.num_groups, device=blueprint_B_k.device).view(1, 1, self.num_groups, 1, 1)
            raw_bytes = self.V_packed[g_idx_all, top1_idx.unsqueeze(-1), top2_idx.unsqueeze(-2)]  # [B, K, 16, 8, 4, 8]
            v_all = self.lut_2bit[raw_bytes.long()].view(B, K, self.num_groups, 32, self.val_dim)  # [B, K, 16, 32, 32]
            
            # Weighted attractor retrieval (32-dim micro-facts)
            m_dyn = torch.einsum('bkgs,bkgsv->bkgv', attn_weights, v_all)  # [B, K, 16, 32]
            
            # Isolate spurious attractors per-group: attenuate dynamic vector towards zero, falling back to anchor V_shared
            m_dyn_safe = m_dyn * (1.0 - jam_gate_g.unsqueeze(-1))
            
            # Linear systolic expansion 32 -> 128 (TPU MXU alignment)
            m_expanded = torch.einsum('bkgv,gvd->bkgd', m_dyn_safe, self.W_expand)  # [B, K, 16, 128]
            
            # Add shared anchor baseline
            m_g = m_expanded + self.V_shared.view(1, 1, self.num_groups, self.expanded_dim)  # [B, K, 16, 128]
            m_concat = m_g.reshape(B, K, self.num_groups * self.expanded_dim)  # [B, K, 2048]
        else:
            # Fallback path (FP32 training parameter V or sequential)
            group_outputs = []
            for g in range(self.num_groups):
                idx1_g = top1_idx[..., g, :]        # [B, K, 8]
                idx2_g = top2_idx[..., g, :]        # [B, K, 4]
                weights_g = attn_weights[..., g, :]  # [B, K, 32]

                # Selective subtensor extraction (2-bit or FP32)
                V_sub = self._unpack_subtensor(g, idx1_g, idx2_g)  # [B, K, 8, 4, 32]
                V_sub_flat = V_sub.view(B, K, self.topk_1 * self.topk_2, self.val_dim)  # [B, K, 32, 32]

                # Weighted attractor retrieval (32-dim micro-fact)
                m_dyn = (weights_g.unsqueeze(-2) @ V_sub_flat).squeeze(-2)  # [B, K, 32]
                m_dyn_safe = m_dyn * (1.0 - jam_gate_g[..., g].unsqueeze(-1))

                # Linear systolic expansion 32 -> 128 (TPU MXU alignment)
                m_expanded = m_dyn_safe @ self.W_expand[g]  # [B, K, 128]

                # Add shared anchor attractor baseline
                m_g = m_expanded + self.V_shared[g].unsqueeze(0).unsqueeze(0)  # [B, K, 128]
                group_outputs.append(m_g)

            # Concatenate across all 16 groups: [B, K, 16 * 128 = 2048]
            m_concat = torch.cat(group_outputs, dim=-1)

        # Project back to model hidden dimension: 2048 -> 1536
        y_factual_context = self.out_proj(m_concat)  # [B, K, 1536]

        # 7. Symmetric Commitment Loss (Binds queries to code centroids)
        weights_c1 = F.softmax(top1_vals, dim=-1).unsqueeze(-1)
        g_idx = torch.arange(self.num_groups, device=blueprint_B_k.device).view(1, 1, self.num_groups, 1)
        c1_active = C1_norm[g_idx, top1_idx]
        c1_bar = (weights_c1 * c1_active).sum(dim=-2)

        weights_c2 = F.softmax(top2_vals, dim=-1).unsqueeze(-1)
        c2_active = C2_norm[g_idx, top2_idx]
        c2_bar = (weights_c2 * c2_active).sum(dim=-2)

        commit_loss = (
            F.mse_loss(q1, c1_bar.detach(), reduction='none').sum(dim=-1).mean() +
            0.25 * F.mse_loss(q1.detach(), c1_bar, reduction='none').sum(dim=-1).mean() +
            F.mse_loss(q2, c2_bar.detach(), reduction='none').sum(dim=-1).mean() +
            0.25 * F.mse_loss(q2.detach(), c2_bar, reduction='none').sum(dim=-1).mean()
        )

        # 8. Optional Phase 3 Word Decoder Gating Bridge with Anti-Jamming Isolation
        final_output = y_factual_context
        # Detect spurious attractor jamming (global indicator when > 50% groups jammed)
        is_jammed_global = (spurious_jammed_g.float().mean().item() > 0.5)

        if phase3_decoder_state is not None:
            # Align context with decoder length via pooling or identity
            if y_factual_context.shape[1] != phase3_decoder_state.shape[1]:
                context_pool = y_factual_context.mean(dim=1, keepdim=True).expand(-1, phase3_decoder_state.shape[1], -1)
            else:
                context_pool = y_factual_context
            gate_input = torch.cat([phase3_decoder_state, context_pool], dim=-1)
            g_fact = torch.sigmoid(self.anchor_gate(gate_input))
            # If memory is jammed with spurious attractors, attenuate factual gate to isolate decoder
            if is_jammed_global:
                g_fact = g_fact * 0.0  # Decouple poisoned memory
            final_output = (1.0 - g_fact) * phase3_decoder_state + g_fact * context_pool
            # If memory is jammed with spurious attractors, attenuate factual gate to isolate decoder
            if is_jammed_global:
                g_fact = g_fact * 0.0  # Decouple poisoned memory
            final_output = (1.0 - g_fact) * phase3_decoder_state + g_fact * context_pool

        # Squeeze 2D input back if needed
        if is_2d:
            final_output = final_output.squeeze(1)

        if not return_diagnostics:
            return final_output, commit_loss

        diagnostics = {
            "top1_indices": top1_idx,
            "top2_indices": top2_idx,
            "attn_weights": attn_weights,
            "hopfield_energy": hopfield_energy,
            "active_sparsity": float(self.topk_1 * self.topk_2) / float(self.codebook_size * self.codebook_size),
            "active_slots_per_group": self.topk_1 * self.topk_2,
            "total_slots": self.num_groups * self.codebook_size * self.codebook_size,
            "router_b1_mean": self.b1.mean().item(),
            "router_b2_mean": self.b2.mean().item(),
            "spurious_jamming_detected": is_jammed_global,
            "attn_variance_mean": attn_var_g.mean().item(),
            "spurious_jammed_groups": spurious_jammed_g,
            "entropy_mean": entropy_g.mean().item()
        }

        return final_output, commit_loss, diagnostics

    @torch.no_grad()
    def hot_patch_fact(
        self,
        group_id: int,
        query_subject: torch.Tensor,
        query_predicate: torch.Tensor,
        factual_vector_128: torch.Tensor,
        direct_indices: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        """
        Direct Hot-Patching of an associative fact in long-term Hopfield memory
        without gradient backpropagation or catastrophic forgetting.
        
        Args:
            group_id: Group index in [0, num_groups - 1] (0..15).
            query_subject: [64] or [1, 64] subject query vector.
            query_predicate: [64] or [1, 64] predicate query vector.
            factual_vector_128: [128] target factual feature vector.
            direct_indices: Optional (i_star, j_star) tuple to bypass cosine retrieval.
            
        Returns:
            (i_star, j_star): The updated attractor slot coordinates.
        """
        if not (0 <= group_id < self.num_groups):
            raise ValueError(f"group_id {group_id} out of valid range [0, {self.num_groups - 1}]")

        # 1. Identify slot coordinates (i*, j*)
        if direct_indices is not None:
            i_star, j_star = direct_indices
        else:
            c1_norm = F.normalize(self.C1[group_id], p=2, dim=-1)
            c2_norm = F.normalize(self.C2[group_id], p=2, dim=-1)
            q1_norm = F.normalize(query_subject.reshape(-1), p=2, dim=-1)
            q2_norm = F.normalize(query_predicate.reshape(-1), p=2, dim=-1)

            s1 = torch.mv(c1_norm, q1_norm)
            s2 = torch.mv(c2_norm, q2_norm)

            i_star = int(torch.argmax(s1).item())
            j_star = int(torch.argmax(s2).item())

        # 2. Centering against shared anchor attractor baseline
        fact_128 = factual_vector_128.reshape(-1)
        target_centered = fact_128 - self.V_shared[group_id]

        # 3. Tikhonov Regularized Pseudoinversion (Damped Least Squares / Ridge)
        #    W_exp has shape [32, 128]. Solves for v_target_32 such that:
        #    v_target_32 @ W_exp ≈ target_centered
        #    v = target_centered @ W^+ = (W W^T + lambda * I)^(-1) @ W @ target_centered
        #    Guarantees numerical stability even when condition number kappa(W W^T) -> infinity!
        W_exp = self.W_expand[group_id]  # [32, 128]
        G = torch.mm(W_exp, W_exp.t())   # Gramian matrix [32, 32]
        trace_val = torch.trace(G).item()
        lambda_reg = max(1e-4 * (trace_val / 32.0), 1e-5)
        damped_G = G + lambda_reg * torch.eye(32, device=W_exp.device, dtype=W_exp.dtype)
        rhs = torch.mv(W_exp, target_centered)
        v_target_32 = torch.linalg.solve(damped_G, rhs)  # [32]

        # 4. Absmean 1.58-bit ternary quantization: {-1.0, 0.0, +1.0}
        gamma = torch.mean(torch.abs(v_target_32)).clamp(min=1e-5)
        v_ternary = torch.clamp(torch.round(v_target_32 / gamma), -1.0, 1.0)

        # 5. In-place memory write
        if hasattr(self, "V") and self.V is not None:
            self.V[group_id, i_star, j_star, :].copy_(v_ternary)

        if hasattr(self, "V_packed") and self.V_packed is not None:
            v_discrete = v_ternary.to(torch.int8)
            v_encoded = torch.where(v_discrete == -1, torch.tensor(2, dtype=torch.int8, device=v_discrete.device), v_discrete)
            v_reshaped = v_encoded.view(self.val_dim // 4, 4)
            packed_slot = (
                (v_reshaped[..., 0] & 0x03) |
                ((v_reshaped[..., 1] & 0x03) << 2) |
                ((v_reshaped[..., 2] & 0x03) << 4) |
                ((v_reshaped[..., 3] & 0x03) << 6)
            ).to(torch.uint8)
            self.V_packed[group_id, i_star, j_star, :].copy_(packed_slot)

        return i_star, j_star

    def get_memory_footprint(self) -> Dict[str, Any]:
        """
        Computes the exact memory footprint of all parameters and buffers in bytes and MiB.
        """
        param_bytes = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_bytes = sum(b.numel() * b.element_size() for b in self.buffers())
        total_bytes = param_bytes + buffer_bytes

        v_bytes = 0
        if hasattr(self, "V_packed") and self.V_packed is not None:
            v_bytes = self.V_packed.numel() * self.V_packed.element_size()
        elif hasattr(self, "V") and self.V is not None:
            v_bytes = self.V.numel() * self.V.element_size()

        return {
            "total_bytes": total_bytes,
            "total_mib": total_bytes / (1024 * 1024),
            "param_bytes": param_bytes,
            "param_mib": param_bytes / (1024 * 1024),
            "buffer_bytes": buffer_bytes,
            "buffer_mib": buffer_bytes / (1024 * 1024),
            "v_tensor_bytes": v_bytes,
            "v_tensor_mib": v_bytes / (1024 * 1024),
            "pack_2bit": self.pack_2bit,
            "total_attractor_slots": self.num_groups * self.codebook_size * self.codebook_size,
        }

    def verify_invariants(self) -> Dict[str, bool]:
        """
        Audits mathematical and architectural invariants specified in RFC-04.
        """
        with torch.no_grad():
            c1_norm = torch.norm(self.C1, p=2, dim=-1)
            c2_norm = torch.norm(self.C2, p=2, dim=-1)
            c1_unit = torch.allclose(c1_norm, torch.ones_like(c1_norm), atol=1e-4)
            c2_unit = torch.allclose(c2_norm, torch.ones_like(c2_norm), atol=1e-4)

            total_slots = self.num_groups * self.codebook_size * self.codebook_size
            expected_slots = 16 * 1024 * 1024  # 16,777,216

            w_expand_shape = (self.W_expand.shape == (self.num_groups, self.val_dim, self.expanded_dim))
            
            sparsity = (self.topk_1 * self.topk_2) / float(self.codebook_size * self.codebook_size)
            expected_sparsity = 32.0 / (1024 * 1024)
            sparsity_match = abs(sparsity - expected_sparsity) < 1e-9

            v_packed_size_ok = True
            if self.pack_2bit and self.V_packed is not None:
                v_packed_size_ok = (self.V_packed.numel() == 134217728)

            return {
                "c1_on_unit_sphere": bool(c1_unit),
                "c2_on_unit_sphere": bool(c2_unit),
                "slots_count_16_78M": bool(total_slots == expected_slots),
                "w_expand_systolic_128": bool(w_expand_shape),
                "sparsity_top32_exact": bool(sparsity_match),
                "v_packed_exact_128mib": bool(v_packed_size_ok),
            }


# Backward-compatibility alias for TH1 codebase imports
SpeculativeLatentHopfieldCore = SLHCoreNCMAdapter

__all__ = [
    "SLHCoreConfig",
    "BitLinear",
    "SLHCoreNCMAdapter",
    "SpeculativeLatentHopfieldCore",
]
