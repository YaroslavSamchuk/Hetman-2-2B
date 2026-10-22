"""
Hetman-2-NCM 2.0B: Training Loss Engine & Optimization Framework
File: training_loss_engine.py
Version: 2.0.0-PROD

This module provides the complete PyTorch implementation of the composite loss engine,
quantization estimators, Hopfield attractor energy regularization, numerical stabilizers,
and the hybrid Muon + AdamW optimization pipeline for Hetman-2-NCM 2.0B.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer


# ==============================================================================
# 1. Straight-Through Estimators (STE) for Ternary BitNet & RVQ
# ==============================================================================

class TernaryQuantizeSTE(torch.autograd.Function):
    """
    Straight-Through Estimator for BitNet b1.58 ternary quantization {-1, 0, +1}.
    Forward:
        gamma = mean(|W|) + eps
        W_tilde = clamp(round(W / gamma), -1, +1)
    Backward:
        dL/dW = (dL/dW_tilde) * 1(|W / gamma| <= 1.0)
    """

    @staticmethod
    def forward(ctx, w: torch.Tensor, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
        gamma = torch.mean(torch.abs(w)) + eps
        scaled_w = w / gamma
        w_quant = torch.clamp(torch.round(scaled_w), -1.0, 1.0)
        ctx.save_for_backward(scaled_w)
        return w_quant, gamma

    @staticmethod
    def backward(ctx, grad_w_quant: torch.Tensor, grad_gamma: Optional[torch.Tensor]) -> Tuple[torch.Tensor, None]:
        scaled_w, = ctx.saved_tensors
        # Straight-Through Estimator with gradient clipping outside [-1, 1]
        mask = (torch.abs(scaled_w) <= 1.0).to(grad_w_quant.dtype)
        grad_w = grad_w_quant * mask
        return grad_w, None


class BitLinear(nn.Module):
    """
    BitNet b1.58 Linear Layer using ternary weights and FP/BF16 activations.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * (1.0 / math.sqrt(in_features)))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_quant, gamma = TernaryQuantizeSTE.apply(self.weight)
        # Y = X (W_quant * gamma)^T = (X W_quant^T) * gamma
        out = F.linear(x, w_quant, None) * gamma
        if self.bias is not None:
            out = out + self.bias
        return out


# ==============================================================================
# 2. Numerical Stabilization Safeguards
# ==============================================================================

class BoundedQKNorm(nn.Module):
    """
    RMSNorm on Queries and Keys with hard-clamped scale factor <= tau_qk.
    Prevents attention entropy collapse and numerical overflow.
    """

    def __init__(self, dim: int, tau_qk: float = 8.0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.tau_qk = tau_qk
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RMS normalization
        rms = torch.rsqrt(torch.mean(x.pow(2), dim=-1, keepdim=True) + self.eps)
        x_norm = x * rms
        # Clamped scale
        clamped_weight = torch.clamp(self.weight, -self.tau_qk, self.tau_qk)
        return x_norm * clamped_weight


class SwiGLUClamped(nn.Module):
    """
    SwiGLU feed-forward layer with activation output clamping <= 10.0.
    Prevents activation explosion in BF16/FP16 training.
    """

    def __init__(self, d_model: int, d_ffn: int, clamp_val: float = 10.0):
        super().__init__()
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.clamp_val = clamp_val

        self.w_gate = BitLinear(d_model, d_ffn, bias=False)
        self.w_up = BitLinear(d_model, d_ffn, bias=False)
        self.w_down = BitLinear(d_ffn, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        activated = gate * up
        # Hard clamp before down-projection to bound activations
        clamped = torch.clamp(activated, -self.clamp_val, self.clamp_val)
        return self.w_down(clamped)


class LogitSoftCapping(nn.Module):
    """
    Logit Soft-Capping via hyperbolic tangent:
        logits_capped = C * tanh(logits / C)
    Prevents cross-entropy loss saturation and exploding classifier gradients.
    """

    def __init__(self, cap_val: float = 30.0):
        super().__init__()
        self.cap_val = cap_val

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.cap_val * torch.tanh(logits / self.cap_val)


# ==============================================================================
# 3. RVQ (Residual Vector Quantizer) Loss Module
# ==============================================================================

class HierarchicalRVQLoss(nn.Module):
    """
    2-Level Hierarchical RVQ (Residual Vector Quantization) Module & Loss.
    Level 0 (L_0): Macro-semantic conceptual codes.
    Level 1 (L_1): Micro-syntactic refinement codes.
    """

    def __init__(self, d_model: int, d_code: int, num_codes_l0: int = 2048, num_codes_l1: int = 4096):
        super().__init__()
        self.d_model = d_model
        self.d_code = d_code
        self.num_codes_l0 = num_codes_l0
        self.num_codes_l1 = num_codes_l1

        self.proj_in = nn.Linear(d_model, d_code, bias=False)
        self.proj_out = nn.Linear(d_code, d_model, bias=False)

        # Codebooks L_0 and L_1
        self.codebook_l0 = nn.Embedding(num_codes_l0, d_code)
        self.codebook_l1 = nn.Embedding(num_codes_l1, d_code)

        # Initialize codebooks with unit variance
        nn.init.normal_(self.codebook_l0.weight, std=1.0 / math.sqrt(d_code))
        nn.init.normal_(self.codebook_l1.weight, std=1.0 / math.sqrt(d_code))

    def quantize_level(self, z: torch.Tensor, codebook: nn.Embedding) -> Tuple[torch.Tensor, torch.Tensor]:
        # z: [Batch, Dim]
        # Distance ||z - e_k||^2 = ||z||^2 + ||e_k||^2 - 2 z . e_k
        dists = (
            torch.sum(z.pow(2), dim=-1, keepdim=True)
            + torch.sum(codebook.weight.pow(2), dim=-1)
            - 2.0 * torch.matmul(z, codebook.weight.t())
        )
        indices = torch.argmin(dists, dim=-1)
        z_q = codebook(indices)
        return z_q, indices

    def forward(self, h_continuous: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Project into latent conceptual space
        z0 = self.proj_in(h_continuous)

        # Level 0 quantization
        q0, indices_l0 = self.quantize_level(z0, self.codebook_l0)

        # Level 1 residual quantization
        residual = z0 - q0
        q1, indices_l1 = self.quantize_level(residual, self.codebook_l1)

        # Composite quantized vector
        z_q = q0 + q1

        # Commitment loss: encoder learns to commit to centroids
        commit_loss = F.mse_loss(z0, q0.detach()) + F.mse_loss(residual, q1.detach())

        # Codebook update loss (if codebook is trainable)
        codebook_loss = F.mse_loss(z0.detach(), q0) + F.mse_loss(residual.detach(), q1)

        # Orthogonality loss for codebooks (encourage codebook diversity)
        c0_norm = F.normalize(self.codebook_l0.weight, p=2, dim=-1)
        sim_c0 = torch.matmul(c0_norm, c0_norm.t())
        diag_c0 = torch.eye(self.num_codes_l0, device=z0.device)
        ortho_loss = torch.mean((sim_c0 - diag_c0).pow(2))

        # Straight-Through Estimator for continuous flow
        z_q_ste = z0 + (z_q - z0).detach()
        h_quantized = self.proj_out(z_q_ste)

        # Bottleneck adapter reconstruction loss (trains proj_out)
        recon_loss = F.mse_loss(h_quantized, h_continuous.detach())

        return {
            "h_quantized": h_quantized,
            "z_q": z_q,
            "indices_l0": indices_l0,
            "indices_l1": indices_l1,
            "commit_loss": commit_loss,
            "codebook_loss": codebook_loss,
            "ortho_loss": ortho_loss,
            "recon_loss": recon_loss,
        }


# ==============================================================================
# 4. Hopfield Attractor Loss (Modern Continuous Hopfield Network)
# ==============================================================================

class HopfieldAttractorLoss(nn.Module):
    """
    Modern Continuous Hopfield Network Energy Regularizer.
    Energy function:
        E(z) = -1/beta * log(sum_k exp(beta <z, M_k>)) + 1/2 ||z||^2
    """

    def __init__(self, num_patterns: int = 4096, dim: int = 2048, beta: float = 8.0):
        super().__init__()
        self.num_patterns = num_patterns
        self.dim = dim
        self.beta = beta
        # Prototype associative patterns M in memory
        self.patterns = nn.Parameter(torch.randn(num_patterns, dim) * (1.0 / math.sqrt(dim)))

    def energy(self, z: torch.Tensor) -> torch.Tensor:
        # z: [Batch, Dim]
        # beta <z, M_k>
        logits = self.beta * F.linear(z, self.patterns)  # [Batch, NumPatterns]
        logsumexp_val = torch.logsumexp(logits, dim=-1)
        quad_term = 0.5 * torch.sum(z.pow(2), dim=-1)
        return - (1.0 / self.beta) * logsumexp_val + quad_term

    def forward(self, z_target: torch.Tensor, z_neg: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Minimize energy for target representation
        energy_pos = self.energy(z_target)
        loss = torch.mean(energy_pos)

        # Contrastive energy if negative samples are provided
        if z_neg is not None:
            energy_neg = self.energy(z_neg)
            contrastive = -torch.log(torch.sigmoid(energy_neg - energy_pos) + 1e-7)
            loss = loss + torch.mean(contrastive)

        return loss


# ==============================================================================
# 5. Teacher-Forcing with Latent Noise (TFLN)
# ==============================================================================

class TeacherForcingLatentNoise(nn.Module):
    """
    Applies Teacher-Forcing with scheduled latent Gaussian noise to solve
    the Chicken-and-Egg Cold Start dilemma between Backbone and Drafter.
    """

    def __init__(self, initial_sigma: float = 0.35, decay_steps: int = 150000):
        super().__init__()
        self.initial_sigma = initial_sigma
        self.decay_steps = decay_steps

    def get_current_sigma(self, step: int) -> float:
        return self.initial_sigma * math.exp(-float(step) / float(self.decay_steps))

    def forward(self, gt_latent: torch.Tensor, pred_latent: torch.Tensor, step: int) -> torch.Tensor:
        if not self.training:
            return pred_latent

        sigma = self.get_current_sigma(step)
        # Scheduled sampling probability
        p_free = min(1.0, max(0.0, (step - 50000.0) / 200000.0))

        if torch.rand(1).item() > p_free:
            # Teacher-Forcing mode with injected latent noise
            noise = torch.randn_like(gt_latent) * sigma
            # Stop gradient to decouple phases during early training
            return (gt_latent + noise).detach()
        else:
            # Transition to self-generated latents with scaled gradient bridge
            # Forward: pred_latent, Backward: scaled gradient 0.05
            return pred_latent.detach() + 0.05 * (pred_latent - pred_latent.detach())


# ==============================================================================
# 6. Muon Optimizer (Newton-Schulz Orthogonalized Momentum)
# ==============================================================================

class MuonOptimizer(Optimizer):
    """
    Muon (Momentum Orthogonalized by Newton-Schulz) for 2D Weight Matrices.
    Computes 5th-order Newton-Schulz polar decomposition of the momentum matrix.
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.01,
        ns_steps: int = 5,
        eps: float = 1e-7,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps, eps=eps)
        super().__init__(params, defaults)

    @staticmethod
    def _newton_schulz_5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
        """
        Computes polar decomposition via 5th-order Newton-Schulz iteration:
            X_0 = G / (||G||_F + eps)
            X_{k+1} = a X_k + b (X_k X_k^T) X_k + c (X_k X_k^T)^2 X_k
        """
        d_out, d_in = G.shape
        transposed = d_out > d_in
        X = G.t() if transposed else G

        # Scale spectral radius <= 1
        norm = torch.linalg.matrix_norm(X, ord="fro") + eps
        X = X / norm

        # Optimal coefficients
        a = 3.4445
        b = -4.7750
        c = 2.0315

        for _ in range(steps):
            A = torch.mm(X, X.t())
            B = b * A + c * torch.mm(A, A)
            X = a * X + torch.mm(B, X)

        if transposed:
            X = X.t()
        return X

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            wd = group["weight_decay"]
            steps = group["ns_steps"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.ndim != 2:
                    raise ValueError(f"Muon only supports 2D matrices, got shape {grad.shape}")

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(grad)

                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(grad, alpha=1.0 - momentum)

                # Newton-Schulz polar decomposition
                ortho_update = self._newton_schulz_5(buf, steps=steps, eps=eps)

                # Aspect ratio scaling
                scale = max(1.0, math.sqrt(p.shape[0] / p.shape[1]))

                # Weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)

                p.add_(ortho_update, alpha=-lr * scale)

        return loss


def create_hybrid_optimizers(
    model: nn.Module,
    muon_lr: float = 0.02,
    adamw_lr: float = 3e-4,
    weight_decay: float = 0.01,
) -> Tuple[MuonOptimizer, AdamW]:
    """
    Automatically groups parameters:
    - 2D parameters (BitLinear, Linear weights) -> Muon
    - 1D parameters (Norms, Biases, Embeddings, Codebooks) -> AdamW
    """
    muon_params: List[torch.Tensor] = []
    adamw_params: List[torch.Tensor] = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 2 and "embed" not in name.lower() and "codebook" not in name.lower():
            muon_params.append(p)
        else:
            adamw_params.append(p)

    opt_muon = MuonOptimizer(muon_params, lr=muon_lr, weight_decay=weight_decay)
    opt_adamw = AdamW(adamw_params, lr=adamw_lr, betas=(0.9, 0.98), weight_decay=weight_decay)
    return opt_muon, opt_adamw


# ==============================================================================
# 7. Composite Loss Engine Orchestrator
# ==============================================================================

class HetmanCompositeLossEngine(nn.Module):
    """
    Composite Objective Function for Hetman-2-NCM 2.0B:
    L_total = lambda_ar * L_blueprint_ar
            + lambda_refine * L_bidirectional
            + lambda_ncm * L_block_decode
            + lambda_hopfield * L_attractor
            + lambda_commit * L_rvq_commit
    """

    def __init__(self, vocab_size: int = 32768, code_l0_size: int = 2048, code_l1_size: int = 4096):
        super().__init__()
        self.vocab_size = vocab_size
        self.code_l0_size = code_l0_size
        self.code_l1_size = code_l1_size
        self.soft_capper = LogitSoftCapping(cap_val=30.0)

    def get_lambdas(self, step: int, max_steps: int = 375000) -> Dict[str, float]:
        """
        Dynamic cosine curriculum schedule for loss coefficients.
        """
        progress = min(1.0, max(0.0, float(step) / float(max_steps)))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))

        return {
            "ar": 0.50 + 0.50 * cosine,          # 1.00 -> 0.50
            "ncm": 0.80 - 0.70 * cosine,         # 0.10 -> 0.80
            "refine": 0.70 - 0.65 * cosine,      # 0.05 -> 0.70
            "hopfield": 0.05 + 0.15 * cosine,    # 0.20 -> 0.05
            "commit": 0.10 + 0.40 * cosine,      # 0.50 -> 0.10
        }

    def forward(
        self,
        # Blueprint AR logits
        logits_k0: torch.Tensor,
        logits_k1: torch.Tensor,
        target_k0: torch.Tensor,
        target_k1: torch.Tensor,
        # Block Drafter logits
        draft_logits: torch.Tensor,
        target_tokens: torch.Tensor,
        # Refiner logits & mask
        refine_logits: torch.Tensor,
        refine_mask: torch.Tensor,
        # Hopfield energy
        hopfield_loss: torch.Tensor,
        # RVQ losses
        rvq_commit_loss: torch.Tensor,
        step: int,
        max_steps: int = 375000,
    ) -> Dict[str, torch.Tensor]:
        # 1. Blueprint AR Loss with Soft-Capping & Label Smoothing
        capped_k0 = self.soft_capper(logits_k0)
        capped_k1 = self.soft_capper(logits_k1)
        loss_ar_l0 = F.cross_entropy(capped_k0, target_k0, label_smoothing=0.05)
        loss_ar_l1 = F.cross_entropy(capped_k1, target_k1, label_smoothing=0.05)
        loss_blueprint_ar = 0.5 * (loss_ar_l0 + loss_ar_l1)

        # 2. Block Drafter Loss
        capped_draft = self.soft_capper(draft_logits)
        loss_block_decode = F.cross_entropy(
            capped_draft.view(-1, self.vocab_size),
            target_tokens.view(-1),
            label_smoothing=0.02,
        )

        # 3. Bidirectional Refinement Loss (Inpainting over masked tokens)
        capped_refine = self.soft_capper(refine_logits)
        # Flatten masked tokens
        mask_flat = refine_mask.view(-1).bool()
        refine_flat = capped_refine.view(-1, self.vocab_size)[mask_flat]
        target_flat = target_tokens.view(-1)[mask_flat]
        if target_flat.numel() > 0:
            loss_bidirectional = F.cross_entropy(refine_flat, target_flat)
        else:
            loss_bidirectional = torch.tensor(0.0, device=logits_k0.device, requires_grad=True)

        # 4. Get dynamic lambdas
        lambdas = self.get_lambdas(step, max_steps)

        # 5. Composite Total Loss
        loss_total = (
            lambdas["ar"] * loss_blueprint_ar
            + lambdas["ncm"] * loss_block_decode
            + lambdas["refine"] * loss_bidirectional
            + lambdas["hopfield"] * hopfield_loss
            + lambdas["commit"] * rvq_commit_loss
        )

        return {
            "loss_total": loss_total,
            "loss_blueprint_ar": loss_blueprint_ar,
            "loss_block_decode": loss_block_decode,
            "loss_bidirectional": loss_bidirectional,
            "loss_hopfield": hopfield_loss,
            "loss_rvq_commit": rvq_commit_loss,
            "lambdas": lambdas,
        }


# ==============================================================================
# 8. Mini Hetman-2-NCM Integration Model & Verification Test
# ==============================================================================

class MiniHetmanNCM(nn.Module):
    """
    Compact integrated architecture for testing all losses, straight-through estimators,
    and hybrid optimization on CPU/GPU.
    """

    def __init__(self, vocab_size: int = 1000, d_model: int = 128, block_size: int = 16):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

        # Embeddings
        self.embed = nn.Embedding(vocab_size, d_model)

        # Stabilization components
        self.qk_norm = BoundedQKNorm(d_model, tau_qk=8.0)
        self.ffn = SwiGLUClamped(d_model, d_ffn=d_model * 2, clamp_val=10.0)

        # RVQ Conceptual Tokenizer
        self.rvq = HierarchicalRVQLoss(d_model=d_model, d_code=64, num_codes_l0=256, num_codes_l1=256)

        # Hopfield Memory Attractor
        self.hopfield = HopfieldAttractorLoss(num_patterns=128, dim=d_model, beta=8.0)

        # Phase 1: Blueprint AR Heads (Predicting L0 and L1 codes)
        self.head_k0 = BitLinear(d_model, 256)
        self.head_k1 = BitLinear(d_model, 256)

        # Phase 2: Block Drafter (Predicting block_size tokens simultaneously)
        self.drafter = BitLinear(d_model, block_size * vocab_size)

        # Phase 3: Bidirectional Refiner
        self.refiner = BitLinear(d_model, vocab_size)

        # Teacher-Forcing Noise Scheduler
        self.tfln = TeacherForcingLatentNoise(initial_sigma=0.35, decay_steps=1000)

    def forward(
        self,
        token_chunks: torch.Tensor,
        step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # token_chunks: [Batch, BlockSize]
        b, s = token_chunks.shape
        x_embed = self.embed(token_chunks).mean(dim=1)  # [Batch, d_model]

        # Backbone block with BitLinear & SwiGLU
        h = self.qk_norm(x_embed)
        h = h + self.ffn(h)

        # RVQ quantization
        rvq_out = self.rvq(h)
        h_q = rvq_out["h_quantized"]
        rvq_commit_loss = (
            rvq_out["commit_loss"]
            + rvq_out["codebook_loss"]
            + 0.1 * rvq_out["ortho_loss"]
            + rvq_out["recon_loss"]
        )

        # Hopfield attractor loss
        hopfield_loss = self.hopfield(h)

        # Phase 1: Blueprint logits
        logits_k0 = self.head_k0(h)
        logits_k1 = self.head_k1(h)

        # Latent concept conditioning with TFLN
        conditioned_latent = self.tfln(h_q, h, step=step)

        # Phase 2: Drafter logits [Batch, BlockSize, VocabSize]
        draft_flat = self.drafter(conditioned_latent)
        draft_logits = draft_flat.view(b, self.block_size, self.vocab_size)

        # Phase 3: Refiner logits [Batch, BlockSize, VocabSize]
        # Simulate draft token embedding + refiner pass
        refine_in = self.embed(token_chunks) + conditioned_latent.unsqueeze(1)
        refine_logits = self.refiner(refine_in)

        return (
            logits_k0,
            logits_k1,
            draft_logits,
            refine_logits,
            hopfield_loss,
            rvq_commit_loss,
            rvq_out["indices_l0"],
            rvq_out["indices_l1"],
        )


def run_verification_test():
    """
    Full unit test verifying:
    1. Forward pass stability without NaN/Inf.
    2. Straight-Through Estimator gradient propagation.
    3. Composite loss decomposition.
    4. Backward pass gradient check for all parameters.
    5. Muon Newton-Schulz and AdamW optimization steps.
    """
    print("=" * 80)
    print(" HETMAN-2-NCM 2.0B: TRAINING LOSS ENGINE & OPTIMIZATION VERIFICATION")
    print("=" * 80)

    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Execution device: {device}")

    # Hyperparameters for test
    vocab_size = 500
    d_model = 128
    block_size = 16
    batch_size = 4

    # Instantiate model and loss engine
    model = MiniHetmanNCM(vocab_size=vocab_size, d_model=d_model, block_size=block_size).to(device)
    loss_engine = HetmanCompositeLossEngine(vocab_size=vocab_size, code_l0_size=256, code_l1_size=256).to(device)

    # Setup hybrid optimizers
    opt_muon, opt_adamw = create_hybrid_optimizers(model, muon_lr=0.02, adamw_lr=3e-4)
    print(f"[*] Muon 2D parameter groups: {len(opt_muon.param_groups[0]['params'])} tensors")
    print(f"[*] AdamW 1D parameter groups: {len(opt_adamw.param_groups[0]['params'])} tensors")

    # Generate synthetic batch of token blocks
    tokens = torch.randint(0, vocab_size, (batch_size, block_size), device=device)
    # Generate random boolean mask for 25% refinement tokens
    refine_mask = (torch.rand(batch_size, block_size, device=device) < 0.25).long()

    # Step counter
    step = 1000

    print("\n--- 1. Forward Pass Execution ---")
    (
        logits_k0,
        logits_k1,
        draft_logits,
        refine_logits,
        hopfield_loss,
        rvq_commit_loss,
        target_k0,
        target_k1,
    ) = model(tokens, step=step)

    print(f"[*] logits_k0 shape: {logits_k0.shape}")
    print(f"[*] logits_k1 shape: {logits_k1.shape}")
    print(f"[*] draft_logits shape: {draft_logits.shape}")
    print(f"[*] refine_logits shape: {refine_logits.shape}")
    print(f"[*] hopfield_loss: {hopfield_loss.item():.4f}")
    print(f"[*] rvq_commit_loss: {rvq_commit_loss.item():.4f}")

    print("\n--- 2. Composite Loss Computation ---")
    loss_dict = loss_engine(
        logits_k0=logits_k0,
        logits_k1=logits_k1,
        target_k0=target_k0,
        target_k1=target_k1,
        draft_logits=draft_logits,
        target_tokens=tokens,
        refine_logits=refine_logits,
        refine_mask=refine_mask,
        hopfield_loss=hopfield_loss,
        rvq_commit_loss=rvq_commit_loss,
        step=step,
    )

    for k, v in loss_dict.items():
        if k != "lambdas":
            print(f"    - {k:22s}: {v.item():.6f}")

    total_loss = loss_dict["loss_total"]
    assert not torch.isnan(total_loss), "FAILED: Total loss is NaN!"
    assert not torch.isinf(total_loss), "FAILED: Total loss is Inf!"
    print("[PASS] Forward pass verified: all losses finite and valid.")

    print("\n--- 3. Backward Pass & Gradient Flow ---")
    opt_muon.zero_grad()
    opt_adamw.zero_grad()
    total_loss.backward()

    # Verify gradients for all parameters
    nan_count = 0
    zero_grad_count = 0
    total_param_count = 0

    for name, param in model.named_parameters():
        if param.requires_grad:
            total_param_count += 1
            if param.grad is None:
                zero_grad_count += 1
                print(f"[!] Warning: No gradient for {name}")
            else:
                if torch.isnan(param.grad).any():
                    nan_count += 1
                    print(f"[X] NaN gradient in {name}")

    print(f"[*] Total trainable parameters: {total_param_count}")
    print(f"[*] Parameters with valid gradients: {total_param_count - zero_grad_count}")
    print(f"[*] NaN gradient count: {nan_count}")
    assert nan_count == 0, "FAILED: Detected NaN gradients!"
    assert zero_grad_count == 0, f"FAILED: {zero_grad_count} parameters received no gradients!"
    print("[PASS] Backward pass verified: gradients propagated smoothly through all components.")

    print("\n--- 4. Optimizer Step Verification (Muon + AdamW) ---")
    # Global gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # Perform optimizer steps
    opt_muon.step()
    opt_adamw.step()
    print("[PASS] Both Muon (Newton-Schulz orthogonalization) and AdamW completed parameter updates.")

    print("\n" + "=" * 80)
    print(" ALL VERIFICATION CHECKS PASSED SUCCESSFULLY (100% OPERATIONAL)")
    print("=" * 80)


if __name__ == "__main__":
    run_verification_test()
