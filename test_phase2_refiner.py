import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Any, List

class FastWalshHadamardTransform(nn.Module):
    """
    Online Blockwise Fast Walsh-Hadamard Transform (FWHT, B_had = 512).
    Disperses activation energy across 512-dim blocks, reducing peak outlier magnitude
    by sqrt(512) ≈ 22.63x and enabling stable INT8 quantization without clipping artifacts.
    Includes Gain Clamping to prevent recurrent amplification across passes.
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
        orig_shape = x.shape
        d = orig_shape[-1]
        assert d % self.block_size == 0, f"Last dimension {d} must be a multiple of {self.block_size}"
        
        x_blocked = x.view(-1, self.block_size)
        x_rot = self._fwht_butterfly(x_blocked)
        
        # Outlier scale estimation with gain clamping
        eta = torch.amax(torch.abs(x_rot), dim=-1, keepdim=True) + 1e-5
        eta_clamped = torch.clamp(eta, self.eta_min, self.eta_max)
        
        # Simulated INT8 quantization & dequantization
        x_quant = torch.clamp(torch.round(x_rot * (127.0 / eta_clamped)), -128.0, 127.0)
        x_dequant = x_quant * (eta_clamped / 127.0)
        
        # Inverse FWHT (FWHT is self-inverse due to orthogonal symmetry)
        x_inv = self._fwht_butterfly(x_dequant)
        return x_inv.view(orig_shape), eta_clamped.mean().item()


class SemanticDeduplicationFilter(nn.Module):
    """
    Anti-Hallucination Semantic Deduplication Filter.
    Computes pairwise cosine similarity matrix M_ij between chunk concept centroids.
    When M_ij > tau (tau = 0.85), an inhibitory repulsion operator projects chunk j
    onto the orthogonal subspace of chunk i, eliminating redundant thought loops.
    """
    def __init__(self, d_model: int = 1536, tau: float = 0.85, eps: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.tau = tau
        self.eps = eps

    def forward(self, H: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
        # H shape: [Batch, NumChunks, ChunkLen, HiddenDim]
        B, K, L_c, d = H.shape
        c = H.mean(dim=2)  # [B, K, d] Concept centroids
        c_norm = torch.norm(c, p=2, dim=-1, keepdim=True) + self.eps
        c_hat = c / c_norm  # [B, K, d] Unit normalized
        
        # Pairwise Cosine Similarity: M[b, i, j] = c_hat_i · c_hat_j
        M = torch.bmm(c_hat, c_hat.transpose(1, 2))  # [B, K, K]
        
        H_out = H.clone()
        repulsion_events = 0
        
        for b in range(B):
            for i in range(K):
                for j in range(i + 1, K):
                    sim = M[b, i, j].item()
                    if sim > self.tau:
                        repulsion_events += 1
                        c_i = c[b, i]
                        c_j = c[b, j]
                        
                        # Orthogonal projection: proj_{c_i}(c_j) = ((c_j · c_i) / ||c_i||^2) * c_i
                        c_i_norm_sq = torch.dot(c_i, c_i) + self.eps
                        proj_coeff = torch.dot(c_j, c_i) / c_i_norm_sq
                        proj_vec = proj_coeff * c_i
                        
                        # Inhibitory Repulsion Dynamic: alpha in [0, 1]
                        alpha_rep = min(1.0, (sim - self.tau) / (1.0 - self.tau + self.eps))
                        delta_c = - alpha_rep * proj_vec
                        
                        # Broadcast semantic shift to all token vectors within chunk j
                        H_out[b, j] = H_out[b, j] + delta_c.unsqueeze(0)
                        
                        # Update running centroid for subsequent pairwise comparisons in the batch
                        c[b, j] = H_out[b, j].mean(dim=0)
                        
        return H_out, M, repulsion_events


class BoundarySeamSmoother(nn.Module):
    """
    Boundary Seam Smoothing Module.
    Resolves representation jumps across discrete chunk interfaces by establishing
    a logical bridge: 'conclusion(chunk_k) is a prerequisite for premise(chunk_{k+1})'.
    Uses bilateral gated exponential decay over a seam window W_seam.
    """
    def __init__(self, d_model: int = 1536, window_size: int = 4, lambda_decay: float = 2.0):
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size
        self.lambda_decay = lambda_decay
        self.w_exit = nn.Linear(d_model, d_model, bias=False)
        self.w_entry = nn.Linear(d_model, d_model, bias=False)
        self.bridge_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: [B, K, L_c, d]
        B, K, L_c, d = H.shape
        H_out = H.clone()
        
        for k in range(K - 1):
            h_exit = H[:, k, -1, :]      # [B, d] Last token of chunk k
            h_entry = H[:, k + 1, 0, :]   # [B, d] First token of chunk k+1
            
            u_k = self.w_exit(h_exit)
            v_next = self.w_entry(h_entry)
            
            bridge = 0.5 * (u_k + v_next)
            diff_fwd = bridge - h_entry
            diff_bwd = bridge - h_exit
            
            gate = self.bridge_gate(torch.cat([u_k, v_next], dim=-1))
            
            for m in range(min(self.window_size, L_c)):
                decay = math.exp(-m / self.lambda_decay)
                # Forward bridge: influence early tokens of chunk k+1
                H_out[:, k + 1, m, :] = H_out[:, k + 1, m, :] + decay * gate * diff_fwd
                # Backward bridge: influence tail tokens of chunk k
                if L_c - 1 - m >= 0:
                    H_out[:, k, L_c - 1 - m, :] = H_out[:, k, L_c - 1 - m, :] + decay * (1.0 - gate) * diff_bwd
                    
        return H_out


class DirichletEnergyWatchdog(nn.Module):
    """
    Oversmoothing Watchdog & Dirichlet Energy Preserver.
    Tracks the graph Dirichlet energy E(H) = (1 / 2K^2) sum_{i,j} ||c_i - c_j||^2.
    If energy drops below theta * E_0 after recurrent bidirectional passes,
    applies contrastive dispersion from the sequence barycenter to restore individuality.
    """
    def __init__(self, energy_threshold_ratio: float = 0.5, boost_gain: float = 0.25):
        super().__init__()
        self.threshold_ratio = energy_threshold_ratio
        self.boost_gain = boost_gain

    def compute_energy(self, H: torch.Tensor) -> torch.Tensor:
        c = H.mean(dim=2)  # [B, K, d]
        B, K, d = c.shape
        diff = c.unsqueeze(2) - c.unsqueeze(1)  # [B, K, K, d]
        energy = torch.sum(diff ** 2, dim=-1).mean() / (2.0 * K)
        return energy

    def forward(self, H: torch.Tensor, initial_energy: float) -> Tuple[torch.Tensor, float, bool]:
        current_energy = self.compute_energy(H).item()
        triggered = False
        H_out = H
        if current_energy < self.threshold_ratio * initial_energy:
            triggered = True
            c = H.mean(dim=2, keepdim=True)         # [B, K, 1, d]
            c_barycenter = c.mean(dim=1, keepdim=True) # [B, 1, 1, d]
            contrastive_dispersion = c - c_barycenter
            H_out = H + self.boost_gain * contrastive_dispersion
        return H_out, current_energy, triggered


class MockSharedArbiter(nn.Module):
    """
    Mock representation of the 16-layer Verified Logic Arbiter from Hetman-2.0B.
    Demonstrates shared-weight execution via MODE_REFINE mode conditioning without adding RAM.
    """
    def __init__(self, d_model: int = 1536, n_heads: int = 12):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, bidirectional: bool = True) -> torch.Tensor:
        # x: [B, SeqLen, d_model]
        B, S, D = x.shape
        norm_x = self.norm(x)
        q = self.q_proj(norm_x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(norm_x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(norm_x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Bounded QK-Norm
        q = F.normalize(q, p=2, dim=-1) * math.sqrt(self.head_dim)
        k = F.normalize(k, p=2, dim=-1) * math.sqrt(self.head_dim)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if not bidirectional:
            mask = torch.triu(torch.full((S, S), float('-inf'), device=x.device), diagonal=1)
            scores = scores + mask
            
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, D)
        return x + self.out_proj(out)


class GlobalBidirectionalRefiner(nn.Module):
    """
    Phase 2: Global Bidirectional NCM Refiner & Anti-Hallucination Deduplicator.
    Executes a fixed budget of 3 to 4 recurrent refinement passes over raw chunk drafts [B_1, ..., B_K].
    Integrates Shared Arbiter Weights, Semantic Deduplication, Seam Smoothing, Dirichlet Watchdog,
    and FWHT INT8 Normalization.
    """
    def __init__(
        self,
        d_model: int = 1536,
        num_iterations: int = 4,
        dedup_tau: float = 0.85,
        seam_window: int = 4,
        fwht_block_size: int = 512,
        alpha_residual_anchor: float = 0.25
    ):
        super().__init__()
        self.d_model = d_model
        self.num_iterations = num_iterations
        self.alpha_residual_anchor = alpha_residual_anchor
        
        # Mode conditioning vector: MODE_REFINE (Occupies only 1536 * 4 bytes = 6 KB RAM)
        self.mode_embedding_refine = nn.Parameter(torch.randn(1, 1, 1, d_model) * 0.02)
        
        # Shared Arbiter (reused from Phase 1/3 backbone)
        self.shared_arbiter = MockSharedArbiter(d_model=d_model)
        
        # Phase 2 Specialized Sub-Modules
        self.deduplicator = SemanticDeduplicationFilter(d_model=d_model, tau=dedup_tau)
        self.seam_smoother = BoundarySeamSmoother(d_model=d_model, window_size=seam_window)
        self.watchdog = DirichletEnergyWatchdog()
        self.fwht_quantizer = FastWalshHadamardTransform(block_size=fwht_block_size)
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, H_raw: torch.Tensor) -> Dict[str, Any]:
        """
        Input:
            H_raw: [Batch, NumChunks, ChunkLen, HiddenDim] - Raw drafts from Phase 1
        Output:
            Dictionary containing refined representations and convergence telemetry.
        """
        B, K, L_c, d = H_raw.shape
        H_0 = H_raw.clone()
        H_curr = H_raw.clone()
        
        initial_energy = self.watchdog.compute_energy(H_0).item()
        telemetry = {
            "initial_dirichlet_energy": initial_energy,
            "pass_telemetry": []
        }
        
        for step in range(1, self.num_iterations + 1):
            # Step 1: Mode Conditioning injection
            H_conditioned = H_curr + self.mode_embedding_refine
            
            # Step 2: Flatten across sequence length for Global Non-Causal Attention
            # Sequence Length S = K * L_c
            H_flat = H_conditioned.view(B, K * L_c, d)
            H_arbiter_flat = self.shared_arbiter(H_flat, bidirectional=True)
            H_arbiter = H_arbiter_flat.view(B, K, L_c, d)
            
            # Step 3: Residual Anchoring to prevent representation drift and oversmoothing
            # H^{(t)} = alpha_0 * H^{(0)} + (1 - alpha_0) * H_arbiter
            H_anchored = self.alpha_residual_anchor * H_0 + (1.0 - self.alpha_residual_anchor) * H_arbiter
            
            # Step 4: Semantic Deduplication Filter (Inhibitory Repulsion)
            H_dedup, M_sim, repulsions = self.deduplicator(H_anchored)
            
            # Step 5: Boundary Seam Smoothing (Inter-chunk premise-conclusion bridging)
            H_smoothed = self.seam_smoother(H_dedup)
            
            # Step 6: Dirichlet Energy Watchdog (Anti-oversmoothing check)
            H_watchdog, current_energy, watchdog_triggered = self.watchdog(H_smoothed, initial_energy)
            
            # Step 7: FWHT INT8 Normalization & Gain Clamping
            H_quant, mean_eta = self.fwht_quantizer(H_watchdog)
            
            # Step 8: SubLN Equivariant Stabilization
            rms = torch.sqrt(torch.mean(H_quant ** 2, dim=-1, keepdim=True) + 1e-5)
            H_curr = H_quant / rms
            
            telemetry["pass_telemetry"].append({
                "pass": step,
                "repulsion_events": repulsions,
                "dirichlet_energy": current_energy,
                "watchdog_triggered": watchdog_triggered,
                "fwht_eta": mean_eta,
                "mean_rms": rms.mean().item()
            })
            
        H_final = self.final_norm(H_curr)
        telemetry["final_dirichlet_energy"] = self.watchdog.compute_energy(H_final).item()
        
        return {
            "refined_chunks": H_final,
            "telemetry": telemetry
        }


def run_unit_tests():
    print("=" * 80)
    print("RUNNING GLOBAL BIDIRECTIONAL NCM REFINER VERIFICATION SUITE")
    print("=" * 80)
    
    torch.manual_seed(42)
    B, K, L_c, d = 1, 4, 16, 1536
    
    # -------------------------------------------------------------
    # TEST 1: SEMANTIC DEDUPLICATION & INHIBITORY REPULSION TEST
    # -------------------------------------------------------------
    print("\n--- TEST 1: Semantic Deduplication & Anti-Hallucination ---")
    H_test = torch.randn(B, K, L_c, d)
    # Intentionally duplicate chunk 1 into chunk 2 with high similarity > 0.95
    H_test[0, 2] = H_test[0, 1] * 0.98 + torch.randn(L_c, d) * 0.02
    
    dedup = SemanticDeduplicationFilter(d_model=d, tau=0.85)
    H_deduped, M_orig, rep_count = dedup(H_test)
    
    c_new = H_deduped.mean(dim=2)
    c_new_hat = c_new / (torch.norm(c_new, p=2, dim=-1, keepdim=True) + 1e-8)
    M_after = torch.bmm(c_new_hat, c_new_hat.transpose(1, 2))
    
    sim_before = M_orig[0, 1, 2].item()
    sim_after = M_after[0, 1, 2].item()
    print(f"Pre-repulsion Cosine Similarity (Chunk 1 vs Chunk 2): {sim_before:.4f}")
    print(f"Post-repulsion Cosine Similarity (Chunk 1 vs Chunk 2): {sim_after:.4f}")
    print(f"Repulsion triggers fired: {rep_count}")
    assert sim_before > 0.85, "Test setup failure: similarity should exceed tau=0.85"
    assert sim_after < 0.85, f"Deduplication failure: similarity {sim_after} not below tau"
    print(">>> TEST 1 PASSED: Deduplication successfully projected onto orthogonal subspace.")
    
    # -------------------------------------------------------------
    # TEST 2: BOUNDARY SEAM SMOOTHING TEST
    # -------------------------------------------------------------
    print("\n--- TEST 2: Boundary Seam Smoothing (Inter-Chunk Coherence) ---")
    smoother = BoundarySeamSmoother(d_model=d, window_size=4)
    jump_before = torch.norm(H_test[0, 0, -1] - H_test[0, 1, 0], p=2).item()
    H_smooth = smoother(H_test)
    jump_after = torch.norm(H_smooth[0, 0, -1] - H_smooth[0, 1, 0], p=2).item()
    print(f"Boundary Discontinuity Jump Before Smoothing: {jump_before:.4f}")
    print(f"Boundary Discontinuity Jump After Smoothing:  {jump_after:.4f}")
    assert jump_after < jump_before, "Seam smoothing failed to reduce boundary step jump"
    print(">>> TEST 2 PASSED: Boundary discontinuity step successfully softened.")
    
    # -------------------------------------------------------------
    # TEST 3: OVERSMOOTHING & DIRICHLET ENERGY PRESERVATION TEST
    # -------------------------------------------------------------
    print("\n--- TEST 3: Dirichlet Energy & Anti-Oversmoothing ---")
    refiner = GlobalBidirectionalRefiner(d_model=d, num_iterations=4, alpha_residual_anchor=0.25)
    res = refiner(H_test)
    telem = res["telemetry"]
    
    e0 = telem["initial_dirichlet_energy"]
    e_final = telem["final_dirichlet_energy"]
    ratio = e_final / e0
    print(f"Initial Dirichlet Energy: {e0:.4f}")
    print(f"Final Dirichlet Energy:   {e_final:.4f} (Retention: {ratio * 100:.2f}%)")
    for pass_info in telem["pass_telemetry"]:
        print(f"  Pass {pass_info['pass']}: Energy={pass_info['dirichlet_energy']:.4f}, "
              f"Repulsions={pass_info['repulsion_events']}, Watchdog={pass_info['watchdog_triggered']}")
    assert ratio > 0.35, f"Oversmoothing detected! Dirichlet energy retention {ratio:.2f} < 0.35"
    print(">>> TEST 3 PASSED: Chunk representations preserve distinct identities across 4 passes.")
    
    # -------------------------------------------------------------
    # TEST 4: FWHT INT8 NORMALIZATION STABILITY TEST
    # -------------------------------------------------------------
    print("\n--- TEST 4: FWHT INT8 Normalization & Dynamic Scale Stability ---")
    rms_values = [p["mean_rms"] for p in telem["pass_telemetry"]]
    eta_values = [p["fwht_eta"] for p in telem["pass_telemetry"]]
    print(f"RMS values across 4 iterations: {[round(r, 4) for r in rms_values]}")
    print(f"FWHT scale eta values across 4 iterations: {[round(e, 4) for e in eta_values]}")
    for r in rms_values:
        assert 0.5 <= r <= 2.0, f"RMS divergence: {r} outside stable bounds [0.5, 2.0]"
    for e in eta_values:
        assert not math.isnan(e) and not math.isinf(e), "FWHT scale exploded to NaN/Inf"
    print(">>> TEST 4 PASSED: Recurrent passes exhibit unconditional numerical stability.")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 80)

if __name__ == "__main__":
    run_unit_tests()
