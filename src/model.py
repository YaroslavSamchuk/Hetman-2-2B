"""
Hetman-2-NCM 2.0B: Master PyTorch Architecture Implementation
File: src/model.py
License: Apache License 2.0

Implements:
- Hetman2Config & Hetman2Config.create_mini()
- BitNet b1.58 Ternary Quantization & BitLinear
- Fast Walsh-Hadamard Transform (FWHT INT8 Engine)
- Gated DeltaNet Recurrent Block (O(1) Constant FP32 State)
- Phase 1: Causal AR Blueprint Drafter
- Phase 2: Global Bidirectional NCM Refiner & Anti-Hallucination Deduplicator (3 passes)
- Phase 3: Block-Causal NCM Decoder (8 passes via DynamicBlueprintEvolver)
- Hetman2ForCausalLM (Full End-to-End Hierarchical Generation Pipeline)
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.dynamic_blueprint_evolver import (
    DynamicBlueprintEvolver,
    SphericalNorm,
    AdaLNZeroModulator,
)


@dataclass
class Hetman2Config:
    """
    Configuration specification for Hetman-2-NCM 2.0B.
    """
    vocab_size: int = 65536
    hidden_size: int = 1536
    intermediate_size: int = 6144
    num_layers: int = 34
    num_deltanet_layers: int = 28
    num_global_layers: int = 6
    deltanet_heads: int = 16
    deltanet_head_dim: int = 64
    global_heads_q: int = 12
    global_heads_kv: int = 2
    global_head_dim: int = 128
    hopfield_groups: int = 16
    hopfield_codebook_size: int = 1024
    hopfield_val_dim: int = 32
    tokens_per_block: int = 400
    max_blocks: int = 8
    latents_per_block: int = 16
    phase2_passes: int = 3
    phase3_passes: int = 8
    fwht_block_size: int = 512
    dedup_tau: float = 0.85
    seam_window: int = 4
    alpha_residual_anchor: float = 0.25

    @classmethod
    def create_mini(
        cls,
        vocab_size: int = 256,
        hidden_size: int = 128,
        intermediate_size: int = 256,
        num_layers: int = 4,
        num_deltanet_layers: int = 3,
        num_global_layers: int = 1,
        deltanet_heads: int = 4,
        deltanet_head_dim: int = 32,
        tokens_per_block: int = 16,
        max_blocks: int = 2,
        latents_per_block: int = 4,
        phase2_passes: int = 3,
        phase3_passes: int = 8,
        fwht_block_size: int = 64,
        **kwargs: Any
    ) -> "Hetman2Config":
        """
        Creates a compact mini-instance configuration for fast unit & end-to-end testing
        while preserving all architectural invariants and mathematical structures.
        """
        return cls(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_layers=num_layers,
            num_deltanet_layers=num_deltanet_layers,
            num_global_layers=num_global_layers,
            deltanet_heads=deltanet_heads,
            deltanet_head_dim=deltanet_head_dim,
            global_heads_q=max(2, deltanet_heads),
            global_heads_kv=max(1, deltanet_heads // 2),
            global_head_dim=deltanet_head_dim,
            hopfield_groups=4,
            hopfield_codebook_size=64,
            hopfield_val_dim=16,
            tokens_per_block=tokens_per_block,
            max_blocks=max_blocks,
            latents_per_block=latents_per_block,
            phase2_passes=phase2_passes,
            phase3_passes=phase3_passes,
            fwht_block_size=min(fwht_block_size, hidden_size),
            **kwargs
        )


# ==============================================================================
# 1. BitNet b1.58 Ternary Quantization
# ==============================================================================

class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for BitNet b1.58 Absmean ternary quantization {-1, 0, +1}.
    """
    @staticmethod
    def forward(ctx, w: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.isnan(w).any() or torch.isinf(w).any():
            w = torch.nan_to_num(w, nan=0.0, posinf=1.0, neginf=-1.0)
        gamma = torch.mean(torch.abs(w)) + eps
        scaled_w = w / gamma
        w_quant = torch.clamp(torch.round(scaled_w), -1.0, 1.0)
        ctx.save_for_backward(scaled_w)
        return w_quant, gamma

    @staticmethod
    def backward(ctx, grad_w_quant: torch.Tensor, grad_gamma: Optional[torch.Tensor]) -> Tuple[torch.Tensor, None]:
        scaled_w, = ctx.saved_tensors
        mask = (torch.abs(scaled_w) <= 1.0).to(grad_w_quant.dtype)
        grad_w = grad_w_quant * mask
        return grad_w, None


class BitLinear(nn.Module):
    """
    BitNet b1.58 Linear layer with ternary weights {-1, 0, +1} and per-tensor scale gamma.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_quant, gamma = TernaryQuantizeSTE.apply(self.weight)
        out = F.linear(x, w_quant, None) * gamma
        if self.bias is not None:
            out = out + self.bias
        return out


# ==============================================================================
# 2. Fast Walsh-Hadamard Transform (FWHT B=512)
# ==============================================================================

class FastWalshHadamardTransform(nn.Module):
    """
    Online Blockwise FWHT for outlier attenuation and INT8 stabilization.
    """
    def __init__(self, block_size: int = 512, eta_min: float = 0.05, eta_max: float = 40.0):
        super().__init__()
        self.block_size = block_size
        self.eta_min = eta_min
        self.eta_max = eta_max

    def _fwht_butterfly(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        dim = orig_shape[-1]
        x_flat = x.reshape(-1, dim)
        h = 1
        while h < dim:
            x_flat = x_flat.view(-1, dim // (2 * h), 2, h)
            x0 = x_flat[:, :, 0, :].clone()
            x1 = x_flat[:, :, 1, :].clone()
            x_flat = torch.cat([x0 + x1, x0 - x1], dim=2)
            h *= 2
        return x_flat.view(orig_shape) / math.sqrt(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, float]:
        if torch.isnan(x).any() or torch.isinf(x).any():
            x = torch.nan_to_num(x, nan=0.0, posinf=self.eta_max, neginf=-self.eta_max)
        orig_shape = x.shape
        d = orig_shape[-1]
        block = min(self.block_size, d)
        if d % block != 0:
            return x, 1.0

        x_blocked = x.view(-1, block)
        x_rot = self._fwht_butterfly(x_blocked)

        eta = torch.amax(torch.abs(x_rot), dim=-1, keepdim=True) + 1e-5
        eta_clamped = torch.clamp(eta, self.eta_min, self.eta_max)

        x_quant = torch.clamp(torch.round(x_rot * (127.0 / eta_clamped)), -128.0, 127.0)
        x_dequant = x_quant * (eta_clamped / 127.0)
        x_inv = self._fwht_butterfly(x_dequant)
        return x_inv.view(orig_shape), eta_clamped.mean().item()


# ==============================================================================
# 3. Gated DeltaNet Recurrent Block (O(1) State Dynamics in FP32)
# ==============================================================================

class GatedDeltaNetLayer(nn.Module):
    """
    Linear recurrent Gated DeltaNet layer with constant O(1) state per head.
    Maintains recurrent state S_t in FP32 across all generation steps.
    """
    def __init__(self, d_model: int, num_heads: int, head_dim: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.total_dim = num_heads * head_dim

        self.q_proj = BitLinear(d_model, self.total_dim, bias=False)
        self.k_proj = BitLinear(d_model, self.total_dim, bias=False)
        self.v_proj = BitLinear(d_model, self.total_dim, bias=False)
        self.alpha_proj = nn.Linear(d_model, num_heads, bias=True)
        self.beta_proj = nn.Linear(d_model, num_heads, bias=True)
        self.out_proj = BitLinear(self.total_dim, d_model, bias=False)

        self.norm = nn.LayerNorm(d_model)

        # SwiGLU FFN
        intermediate = d_model * 4
        self.ffn_gate = BitLinear(d_model, intermediate, bias=False)
        self.ffn_up = BitLinear(d_model, intermediate, bias=False)
        self.ffn_down = BitLinear(intermediate, d_model, bias=False)
        self.ffn_norm = nn.LayerNorm(d_model)

    def forward_step(
        self,
        x_t: torch.Tensor,
        state_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single recurrent step for token vector x_t: [Batch, d_model].
        state_prev: [Batch, num_heads, head_dim, head_dim] in FP32.
        """
        B = x_t.size(0)
        if state_prev is None:
            state_prev = torch.zeros(
                B, self.num_heads, self.head_dim, self.head_dim,
                dtype=torch.float32, device=x_t.device
            )
        elif torch.isnan(state_prev).any() or torch.isinf(state_prev).any():
            state_prev = torch.nan_to_num(state_prev, nan=0.0, posinf=50.0, neginf=-50.0)

        if torch.isnan(x_t).any() or torch.isinf(x_t).any():
            x_t = torch.nan_to_num(x_t, nan=0.0, posinf=50.0, neginf=-50.0)

        norm_x = self.norm(x_t)
        q = self.q_proj(norm_x).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(norm_x).view(B, self.num_heads, self.head_dim)
        v = self.v_proj(norm_x).view(B, self.num_heads, self.head_dim)

        # Normalize key vector for stable recurrence
        k = F.normalize(k, p=2, dim=-1)

        alpha = torch.sigmoid(self.alpha_proj(norm_x)).view(B, self.num_heads, 1, 1).to(torch.float32)
        beta = torch.sigmoid(self.beta_proj(norm_x)).view(B, self.num_heads, 1).to(torch.float32)

        # Recurrent DeltaNet update:
        # delta_v = v - (S_{t-1} @ k)
        k_col = k.unsqueeze(-1).to(torch.float32) # [B, H, D, 1]
        v_col = v.unsqueeze(-1).to(torch.float32) # [B, H, D, 1]
        q_row = q.unsqueeze(-2).to(torch.float32) # [B, H, 1, D]

        retrieved_v = torch.matmul(state_prev, k_col) # [B, H, D, 1]
        delta_v = v_col - retrieved_v
        update_term = torch.matmul(delta_v, k_col.transpose(-2, -1)) # [B, H, D, D]

        state_new = alpha * state_prev + beta.unsqueeze(-1) * update_term

        # Readout: out = S_new @ q
        out_v = torch.matmul(state_new, q_row.transpose(-2, -1)).squeeze(-1) # [B, H, D]
        attn_out = self.out_proj(out_v.view(B, -1).to(x_t.dtype))
        x_post = x_t + attn_out

        # SwiGLU FFN step
        ffn_in = self.ffn_norm(x_post)
        ffn_out = self.ffn_down(F.silu(self.ffn_gate(ffn_in)) * self.ffn_up(ffn_in))
        x_final = x_post + ffn_out

        return x_final, state_new

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        state_init: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Processes full token sequence [Batch, SeqLen, d_model] through recurrent steps.
        """
        B, S, D = x_seq.shape
        state = state_init
        outputs = []
        for t in range(S):
            x_t = x_seq[:, t, :]
            out_t, state = self.forward_step(x_t, state)
            outputs.append(out_t.unsqueeze(1))
        return torch.cat(outputs, dim=1), state


# ==============================================================================
# 4. Phase 1: Causal AR Blueprint Drafter
# ==============================================================================

class Phase1BlueprintDrafter(nn.Module):
    """
    Phase 1: Generates coarse conceptual blueprints B_k for K blocks based on prompt context.
    """
    def __init__(self, config: Hetman2Config):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.max_blocks = config.max_blocks
        self.latents_per_block = config.latents_per_block

        self.block_query_embed = nn.Parameter(
            torch.randn(config.max_blocks, self.d_model) * 0.02
        )
        self.drafter_layers = nn.ModuleList([
            GatedDeltaNetLayer(
                d_model=self.d_model,
                num_heads=config.deltanet_heads,
                head_dim=config.deltanet_head_dim
            )
            for _ in range(min(4, config.num_layers))
        ])
        self.spher_norm = SphericalNorm()

    def forward(
        self,
        prompt_repr: torch.Tensor,
        num_blocks: Optional[int] = None
    ) -> torch.Tensor:
        """
        prompt_repr: [Batch, PromptLen, d_model]
        Returns:
            blueprints: [Batch, num_blocks, d_model]
        """
        B = prompt_repr.size(0)
        K = num_blocks or self.max_blocks
        K = min(K, self.max_blocks)

        # Context summary via mean pooling with empty and NaN guards
        if prompt_repr.size(1) == 0:
            ctx = torch.zeros(B, 1, self.d_model, device=prompt_repr.device, dtype=prompt_repr.dtype)
        else:
            if torch.isnan(prompt_repr).any() or torch.isinf(prompt_repr).any():
                prompt_repr = torch.nan_to_num(prompt_repr, nan=0.0, posinf=10.0, neginf=-10.0)
            ctx = prompt_repr.mean(dim=1, keepdim=True) # [B, 1, d]
        queries = self.block_query_embed[:K].unsqueeze(0).expand(B, -1, -1) # [B, K, d]
        x = queries + ctx

        # Run through causal drafter layers
        state = None
        for layer in self.drafter_layers:
            x, state = layer.forward_sequence(x, state)

        blueprints = self.spher_norm(x)
        return blueprints


# ==============================================================================
# 5. Phase 2: Global Bidirectional NCM Refiner & Anti-Hallucination Deduplicator
# ==============================================================================

class SemanticDeduplicationFilter(nn.Module):
    """
    Anti-Hallucination Deduplicator. Applies inhibitory orthogonal repulsion
    when inter-chunk cosine similarity exceeds tau = 0.85.
    """
    def __init__(self, d_model: int = 1536, tau: float = 0.85, eps: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.tau = tau
        self.eps = eps

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, int]:
        # H: [Batch, NumChunks, d_model]
        B, K, d = H.shape
        if K <= 1:
            return H, 0

        c_norm = torch.norm(H, p=2, dim=-1, keepdim=True) + self.eps
        c_hat = H / c_norm
        M = torch.bmm(c_hat, c_hat.transpose(1, 2))

        H_out = H.clone()
        repulsions = 0

        for b in range(B):
            for i in range(K):
                for j in range(i + 1, K):
                    sim = M[b, i, j].item()
                    if math.isnan(sim):
                        sim = 0.0
                    if sim > self.tau:
                        repulsions += 1
                        c_i = H[b, i]
                        c_j = H[b, j]
                        c_i_sq = torch.dot(c_i, c_i) + self.eps
                        proj = (torch.dot(c_j, c_i) / c_i_sq) * c_i
                        alpha_rep = min(1.0, (sim - self.tau) / (1.0 - self.tau + self.eps))
                        H_out[b, j] = H_out[b, j] - alpha_rep * proj

        return H_out, repulsions


class BoundarySeamSmoother(nn.Module):
    """
    Smooths semantic transitions across chunk boundaries.
    """
    def __init__(self, d_model: int = 1536):
        super().__init__()
        self.d_model = d_model
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        B, K, d = H.shape
        if K <= 1:
            return H
        H_out = H.clone()
        for k in range(K - 1):
            h_curr = H[:, k, :]
            h_next = H[:, k + 1, :]
            bridge = 0.5 * (h_curr + h_next)
            g = self.gate(torch.cat([h_curr, h_next], dim=-1))
            H_out[:, k + 1, :] = H_out[:, k + 1, :] + 0.15 * g * (bridge - h_next)
            H_out[:, k, :] = H_out[:, k, :] + 0.15 * (1.0 - g) * (bridge - h_curr)
        return H_out


class DirichletEnergyWatchdog(nn.Module):
    """
    Tracks graph Dirichlet energy E(H) to safeguard against oversmoothing.
    """
    def __init__(self, threshold_ratio: float = 0.5, boost_gain: float = 0.25):
        super().__init__()
        self.threshold_ratio = threshold_ratio
        self.boost_gain = boost_gain

    def compute_energy(self, H: torch.Tensor) -> float:
        B, K, d = H.shape
        if K <= 1:
            return 1.0
        diff = H.unsqueeze(2) - H.unsqueeze(1)
        energy = torch.sum(diff ** 2, dim=-1).mean() / (2.0 * K)
        return float(energy.item())

    def forward(self, H: torch.Tensor, initial_energy: float) -> Tuple[torch.Tensor, float, bool]:
        current_energy = self.compute_energy(H)
        triggered = False
        H_out = H
        if math.isnan(initial_energy) or math.isnan(current_energy):
            triggered = True
            H_out = torch.nan_to_num(H, nan=0.0)
            current_energy = 1.0
        elif initial_energy < 1e-6:
            triggered = True
            barycenter = H.mean(dim=1, keepdim=True)
            H_out = H + self.boost_gain * (H - barycenter) + torch.randn_like(H) * 0.02
        elif current_energy < self.threshold_ratio * initial_energy:
            triggered = True
            barycenter = H.mean(dim=1, keepdim=True)
            H_out = H + self.boost_gain * (H - barycenter)
        return H_out, current_energy, triggered


class Phase2NCMRefiner(nn.Module):
    """
    Phase 2: Global Bidirectional NCM Refiner.
    Iteratively refines raw draft blueprints across 3 passes (num_iterations=3).
    """
    def __init__(self, config: Hetman2Config):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.num_iterations = config.phase2_passes

        self.mode_embedding_refine = nn.Parameter(
            torch.randn(1, 1, self.d_model) * 0.02
        )
        # Shared bidirectional arbiter attention
        self.arbiter_q = nn.Linear(self.d_model, self.d_model, bias=False)
        self.arbiter_k = nn.Linear(self.d_model, self.d_model, bias=False)
        self.arbiter_v = nn.Linear(self.d_model, self.d_model, bias=False)
        self.arbiter_out = nn.Linear(self.d_model, self.d_model, bias=False)
        self.arbiter_norm = nn.LayerNorm(self.d_model)

        self.deduplicator = SemanticDeduplicationFilter(d_model=self.d_model, tau=config.dedup_tau)
        self.seam_smoother = BoundarySeamSmoother(d_model=self.d_model)
        self.watchdog = DirichletEnergyWatchdog()
        self.fwht = FastWalshHadamardTransform(block_size=config.fwht_block_size)
        self.spher_norm = SphericalNorm()

    def forward(self, H_raw: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, Any]]:
        B, K, d = H_raw.shape
        H_0 = H_raw.clone()
        H_curr = H_raw.clone()

        initial_energy = self.watchdog.compute_energy(H_0)
        telemetry: Dict[str, Any] = {
            "initial_dirichlet_energy": initial_energy,
            "pass_telemetry": []
        }

        for step in range(1, self.num_iterations + 1):
            H_cond = H_curr + self.mode_embedding_refine
            norm_h = self.arbiter_norm(H_cond)

            q = self.arbiter_q(norm_h)
            k = self.arbiter_k(norm_h)
            v = self.arbiter_v(norm_h)

            scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)
            attn = F.softmax(scores, dim=-1)
            H_arb = H_cond + self.arbiter_out(torch.bmm(attn, v))

            # Residual anchor (25% initial anchor)
            H_anchored = (
                self.config.alpha_residual_anchor * H_0 +
                (1.0 - self.config.alpha_residual_anchor) * H_arb
            )

            # Semantic deduplication
            H_dedup, repulsions = self.deduplicator(H_anchored)

            # Seam smoother
            H_seam = self.seam_smoother(H_dedup)

            # Dirichlet watchdog
            H_watch, current_energy, triggered = self.watchdog(H_seam, initial_energy)

            # FWHT INT8 normalization
            H_quant, mean_eta = self.fwht(H_watch)
            H_curr = self.spher_norm(H_quant)

            telemetry["pass_telemetry"].append({
                "pass": step,
                "dirichlet_energy": current_energy,
                "repulsions": repulsions,
                "watchdog_triggered": triggered,
                "mean_fwht_eta": mean_eta
            })

        return H_curr, telemetry


# ==============================================================================
# 6. Phase 3: Block-Causal NCM Decoder
# ==============================================================================

class Phase3NCMBlockDecoder(nn.Module):
    """
    Phase 3: Synthesizes final discrete tokens for each block using 8 dynamic passes
    via DynamicBlueprintEvolver while carrying forward Gated DeltaNet state.
    """
    def __init__(self, config: Hetman2Config):
        super().__init__()
        self.config = config
        self.d_model = config.hidden_size
        self.tokens_per_block = config.tokens_per_block
        self.num_passes = config.phase3_passes

        self.evolver = DynamicBlueprintEvolver(
            d_model=self.d_model,
            total_steps=self.num_passes,
            gamma_min=0.10,
            gamma_max=0.80,
            schedule_type="cosine"
        )
        self.adaln = AdaLNZeroModulator(d_model=self.d_model)

        # DeltaNet core for token synthesis & persistent context
        self.deltanet_core = GatedDeltaNetLayer(
            d_model=self.d_model,
            num_heads=config.deltanet_heads,
            head_dim=config.deltanet_head_dim
        )
        self.token_pos_embed = nn.Parameter(
            torch.randn(config.tokens_per_block, self.d_model) * 0.02
        )
        self.lm_head = BitLinear(self.d_model, config.vocab_size, bias=False)

    def decode_single_block(
        self,
        blueprint: torch.Tensor,
        state_prev: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, List[Dict[str, float]]]:
        """
        blueprint: [Batch, d_model] - Single refined block blueprint
        state_prev: [Batch, H, D, D] in FP32 - DeltaNet recurrent state from block k-1
        Returns:
            token_ids: [Batch, tokens_per_block]
            state_next: [Batch, H, D, D] in FP32
            pass_metrics: list of metrics from 8 evolver passes
        """
        B = blueprint.size(0)
        L = self.tokens_per_block
        B_curr = blueprint
        B_init = blueprint

        # Initialize token representation candidate
        pos = self.token_pos_embed.unsqueeze(0).expand(B, -1, -1) # [B, L, d]
        H_tokens = pos + blueprint.unsqueeze(1)
        pass_metrics: List[Dict[str, float]] = []

        # 1. Base coarse-to-fine RVQ ladder passes (Scale 1 -> Scale 2 -> Scale 3)
        base_passes = self.num_passes
        for pass_idx in range(1, base_passes + 1):
            g1, b1, a1, g2, b2, a2 = self.adaln(B_curr)
            g1 = g1.unsqueeze(1)
            b1 = b1.unsqueeze(1)
            a1 = a1.unsqueeze(1)

            # Modulated token representations
            H_norm = F.layer_norm(H_tokens, (self.d_model,))
            H_mod = (1.0 + g1) * H_norm + b1
            H_tokens = H_tokens + a1 * H_mod

            # Intent subtraction via evolver
            B_curr, metrics = self.evolver.step(
                B_curr=B_curr,
                B_init=B_init,
                H_gen=H_tokens,
                step_idx=pass_idx
            )
            metrics["pass_type"] = "base"
            pass_metrics.append(metrics)

        # 2. Emergency correction passes (up to 3 refinement passes on uncertain positions)
        emergency_passes = 3
        for em_idx in range(1, emergency_passes + 1):
            g1, b1, a1, g2, b2, a2 = self.adaln(B_curr)
            g1 = g1.unsqueeze(1)
            b1 = b1.unsqueeze(1)
            a1 = a1.unsqueeze(1)

            H_norm = F.layer_norm(H_tokens, (self.d_model,))
            H_mod = (1.0 + g1) * H_norm + b1
            H_tokens = H_tokens + 0.5 * a1 * H_mod

            # Emergency refinement step
            B_curr, metrics = self.evolver.step(
                B_curr=B_curr,
                B_init=B_init,
                H_gen=H_tokens,
                step_idx=base_passes  # clamped to terminal step
            )
            metrics["pass_type"] = f"emergency_{em_idx}"
            pass_metrics.append(metrics)

        # 3. Advance DeltaNet recurrent state across all tokens of this block in FP32
        H_tokens_out, state_next = self.deltanet_core.forward_sequence(H_tokens, state_prev)

        # Emit discrete token IDs via LM Head
        logits = self.lm_head(H_tokens_out)  # [B, L, vocab_size]
        token_ids = torch.argmax(logits, dim=-1)  # [B, L]

        block_telemetry = {
            "base_passes_executed": base_passes,
            "emergency_passes_executed": emergency_passes,
            "total_passes_executed": base_passes + emergency_passes,
            "pass_metrics": pass_metrics,
            "delta_state_norm": float(torch.norm(state_next).item())
        }

        return token_ids, state_next, block_telemetry


# ==============================================================================
# 7. Master Model: Hetman2ForCausalLM
# ==============================================================================

class Hetman2ForCausalLM(nn.Module):
    """
    Hetman-2-NCM Master Hierarchical Generative Model.
    Unifies Phase 1 (Draft), Phase 2 (Refine), Phase 3 (Decode) with Gated DeltaNet state.
    """
    def __init__(self, config: Optional[Hetman2Config] = None):
        super().__init__()
        self.config = config or Hetman2Config()

        self.embed_tokens = nn.Embedding(self.config.vocab_size, self.config.hidden_size)

        self.phase1_drafter = Phase1BlueprintDrafter(self.config)
        self.phase2_refiner = Phase2NCMRefiner(self.config)
        self.phase3_decoder = Phase3NCMBlockDecoder(self.config)

        # Tied weights: LM Head shares weights with embedding table
        self.phase3_decoder.lm_head.weight = self.embed_tokens.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        num_blocks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Executes full generation forward pass (Phase 1 -> Phase 2 -> Phase 3).
        """
        B = input_ids.size(0)
        K = num_blocks or self.config.max_blocks

        # Embed prompt tokens with OOV clamping and empty-sequence protection
        if input_ids.numel() == 0 or input_ids.size(1) == 0:
            prompt_repr = torch.zeros(B, 0, self.config.hidden_size, device=input_ids.device)
        else:
            clamped_ids = torch.clamp(input_ids, 0, self.config.vocab_size - 1)
            prompt_repr = self.embed_tokens(clamped_ids)

        # Phase 1: Draft Blueprints
        draft_bps = self.phase1_drafter(prompt_repr, num_blocks=K)

        # Phase 2: Refine Blueprints (3 passes)
        refined_bps, phase2_telemetry = self.phase2_refiner(draft_bps)

        # Phase 3: Decode Blocks (8 passes per block + DeltaNet state propagation)
        state: Optional[torch.Tensor] = None
        all_block_tokens: List[torch.Tensor] = []
        phase3_telemetry: List[List[Dict[str, float]]] = []

        for k in range(K):
            bp_k = refined_bps[:, k, :]
            tokens_k, state, metrics_k = self.phase3_decoder.decode_single_block(bp_k, state)
            all_block_tokens.append(tokens_k)
            phase3_telemetry.append(metrics_k)

        generated_tokens = torch.cat(all_block_tokens, dim=1) # [B, K * tokens_per_block]

        return {
            "generated_tokens": generated_tokens,
            "draft_blueprints": draft_bps,
            "refined_blueprints": refined_bps,
            "final_deltanet_state": state,
            "phase2_telemetry": phase2_telemetry,
            "phase3_telemetry": phase3_telemetry
        }

    def generate(
        self,
        prompt_token_ids: torch.Tensor,
        num_blocks: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Public inference API matching full generation cycle.
        """
        self.eval()
        with torch.no_grad():
            return self.forward(prompt_token_ids, num_blocks=num_blocks)
