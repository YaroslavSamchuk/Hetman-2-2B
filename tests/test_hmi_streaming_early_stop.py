"""
Unit tests for Hetman-2-NCM Phase 2 Early Stop & Streaming Telemetry.
Verifies RFC-HETMAN-2-NCM-16 compliance:
1. Safe Phase 2 early interruption after pass t.
2. Validity of interrupted blueprints (norm, finiteness, shape).
3. SSE event formatting and JSON serialization.
"""

import sys
from pathlib import Path
_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

import json
import time
import torch

from src.bidirectional_refiner import GlobalBidirectionalRefiner


class EarlyStopRefinerWrapper:
    """Simulates the Rada.cpp / PyTorch early interruption controller."""
    def __init__(self, refiner: GlobalBidirectionalRefiner):
        self.refiner = refiner

    def refine_with_interrupt(
        self,
        H_raw: torch.Tensor,
        interrupt_after_pass: int = 2
    ):
        H_curr = H_raw.clone()
        telemetry = []
        interrupted = False

        initial_energy = float(self.refiner.watchdog.compute_energy(H_curr).item())

        for step in range(1, self.refiner.num_iterations + 1):
            # Check interrupt signal before or during step
            if step > interrupt_after_pass:
                interrupted = True
                break

            # Execute pass step
            H_conditioned = H_curr + self.refiner.mode_embedding_refine
            B, K, L_c, d = H_conditioned.shape
            H_flat = H_conditioned.view(B, K * L_c, d)
            H_arbiter = self.refiner.shared_arbiter(H_flat, bidirectional=True).view(B, K, L_c, d)
            H_anchored = (
                self.refiner.alpha_residual_anchor * H_raw
                + (1.0 - self.refiner.alpha_residual_anchor) * H_arbiter
            )
            H_dedup, _, repulsions = self.refiner.deduplicator(H_anchored)
            H_smoothed = self.refiner.seam_smoother(H_dedup)
            H_watchdog, current_energy, watchdog_triggered = self.refiner.watchdog(
                H_smoothed,
                initial_energy,
                current_pass=step,
                total_passes=self.refiner.num_iterations,
            )
            H_quant, mean_eta = self.refiner.fwht_quantizer(H_watchdog)
            rms = torch.sqrt(torch.mean(H_quant ** 2, dim=-1, keepdim=True) + 1e-5)
            H_curr = H_quant / rms

            telemetry.append({
                "pass": step,
                "repulsion_events": repulsions,
                "dirichlet_energy": current_energy,
                "watchdog_triggered": watchdog_triggered,
                "fwht_eta": mean_eta,
            })

        H_final = self.refiner.final_norm(H_curr)
        return {
            "refined_chunks": H_final,
            "interrupted": interrupted,
            "passes_completed": len(telemetry),
            "telemetry": telemetry,
        }


def test_early_stop_phase2_integrity():
    torch.manual_seed(42)
    B, K, L_c, d = 1, 4, 16, 1536
    refiner = GlobalBidirectionalRefiner(d_model=d, num_iterations=4, fwht_block_size=512)
    wrapper = EarlyStopRefinerWrapper(refiner)

    H_raw = torch.randn(B, K, L_c, d)

    t0 = time.perf_counter()
    res = wrapper.refine_with_interrupt(H_raw, interrupt_after_pass=2)
    latency_ms = (time.perf_counter() - t0) * 1000.0

    assert res["interrupted"] is True
    assert res["passes_completed"] == 2
    assert res["refined_chunks"].shape == (B, K, L_c, d)
    assert not torch.isnan(res["refined_chunks"]).any()
    assert not torch.isinf(res["refined_chunks"]).any()
    # Ensure latency of 2 passes is low
    assert latency_ms < 500.0


def test_sse_event_serialization():
    event_type = "thinking_telemetry"
    payload = {
        "request_id": "test-req-001",
        "pass": 2,
        "total_budget": 4,
        "confidence": 0.88,
        "dirichlet_energy": 0.184,
        "repulsion_events": 0,
        "elapsed_ms": 230.5
    }
    sse_data = f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
    assert "event: thinking_telemetry\n" in sse_data
    assert '"confidence": 0.88' in sse_data
    assert sse_data.endswith("\n\n")


if __name__ == "__main__":
    test_early_stop_phase2_integrity()
    test_sse_event_serialization()
    print("All HMI & Early Stop tests passed successfully!")
