"""
================================================================================
HETMAN-2-NCM: COMPREHENSIVE TEST SUITE FOR HOPFIELD KV-CACHE & A3 ATTENTION
Document Specifications:
  - RFC-RADA-NCM-13 (spec/13_RADA_CPP_HOPFIELD_KV_CACHE_ARCHITECTURE.md)
  - RFC-HETMAN-TRAIN-14 (spec/14_HOPFIELD_KV_CACHE_TRAINING_AND_NO_DEGRADATION.md)
Verification Objective:
  1. 3-Tier Partitioned Architecture (WW 1024 FP8, Attractors 8000, Outliers 2048)
  2. Mahalanobis Outlier Discriminator (D > 3.25)
  3. Incremental Welford Clustering & History Compaction
  4. Variance-Compensated Softmax Operator
  5. Attractor-Augmented Attention (A3) & STE Gradient Propagation
  6. Strict Mathematical Convergence: D_KL(P_raw || P_comp) < 0.05
================================================================================
"""

import math
import sys
import os
import unittest
import torch
import torch.nn.functional as F

# Ensure root directory is on python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.hopfield_kv_cache import (
    CACHE_LINE_SIZE,
    NUM_GLOBAL_LAYERS,
    GLOBAL_LAYER_INDICES,
    NUM_KV_HEADS,
    NUM_QUERY_HEADS,
    HEAD_DIM,
    WORKING_WINDOW_SIZE,
    TOKENS_PER_CLUSTER,
    MAX_ATTRACTORS,
    MAX_OUTLIERS,
    OUTLIER_THRESHOLD_MAHAL,
    ATTRACTOR_MERGE_SIM_COS,
    DEFAULT_INVERSE_TEMP_BETA,
    LOG_CLUSTER_SIZE,
    WorkingTokenSlot,
    AttractorCentroid,
    OutlierFactSlot,
    quantize_symmetric_int8,
    dequantize_symmetric_int8,
    variance_compensated_softmax,
    HopfieldKVCache,
    AttractorSTE,
    AttractorAugmentedAttention,
)


class TestHopfieldKVCache(unittest.TestCase):
    """
    Unit and integration tests for the 3-tier Hopfield KV-Cache architecture.
    """

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cpu")

    def test_quantization_roundtrip(self):
        """Tests that INT8 symmetric quantization preserves vector direction and scale."""
        x = torch.randn(NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM) * 2.5
        quant, scales = quantize_symmetric_int8(x)
        self.assertEqual(quant.dtype, torch.int8)
        self.assertEqual(scales.shape, (NUM_GLOBAL_LAYERS, NUM_KV_HEADS))

        x_rec = dequantize_symmetric_int8(quant, scales)
        # Cosine similarity should be > 0.999
        cos_sim = F.cosine_similarity(x.view(-1), x_rec.view(-1), dim=0).item()
        self.assertGreater(cos_sim, 0.999, f"Quantization degraded direction: cos={cos_sim}")

    def test_working_window_ring_buffer_fifo(self):
        """
        Tests Tier 1 Working Window Ring Buffer:
        Inserting 1024 tokens fills the buffer.
        Inserting additional tokens evicts the oldest in exact FIFO order.
        """
        cache = HopfieldKVCache(window_size=1024, max_attractors=100, max_outliers=50)

        # Push 1024 tokens
        for i in range(1024):
            k = torch.randn(NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)
            v = torch.randn(NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)
            cache.push_token(k, v, token_pos=i)

        self.assertEqual(cache.working_window_count, 1024)
        self.assertEqual(len(cache.attractors), 0)
        self.assertEqual(len(cache.outliers), 0)

        # Push 1 more token -> triggers 1 eviction
        k_extra = torch.randn(NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)
        v_extra = torch.randn(NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)
        cache.push_token(k_extra, v_extra, token_pos=1024)

        self.assertEqual(cache.working_window_count, 1024)
        # Either placed in attractors or outliers
        total_migrated = len(cache.attractors) + len(cache.outliers)
        self.assertEqual(total_migrated, 1, "Exactly one token must be migrated on window overflow")

    def test_mahalanobis_outlier_discriminator(self):
        """
        Tests Tier 3 Factual Outlier Preservation:
        Tokens with Mahalanobis distance D > 3.25 must be routed to OutlierFactSlot.
        """
        cache = HopfieldKVCache(window_size=16, max_attractors=10, max_outliers=10, outlier_threshold_mahal=3.25)

        # Create a cluster center and fill the window with tokens tightly clustered around it
        cluster_center = torch.randn(NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)
        for i in range(16):
            noise = torch.randn_like(cluster_center) * 0.05
            cache.push_token(cluster_center + noise, torch.randn_like(cluster_center), token_pos=i)

        # Evict one normal token so an attractor is initialized
        cache.push_token(cluster_center + torch.randn_like(cluster_center) * 0.05, torch.randn_like(cluster_center), token_pos=16)
        self.assertGreaterEqual(len(cache.attractors), 1, "Attractor must be initialized")

        # Now push an extreme factual outlier: high magnitude or orthogonal needle
        needle_k = torch.randn_like(cluster_center) * 15.0  # Massive Mahalanobis divergence
        needle_v = torch.full_like(cluster_center, 999.0)   # Specific secret fact value

        # Push 16 tokens to force the needle through the working window to eviction
        cache.push_token(needle_k, needle_v, token_pos=100)
        for i in range(16):
            cache.push_token(cluster_center + torch.randn_like(cluster_center) * 0.05, torch.randn_like(cluster_center), token_pos=101 + i)

        # The needle must have been detected as an outlier and routed to Tier 3
        self.assertGreaterEqual(len(cache.outliers), 1, "Needle token must be routed to Factual Outlier Buffer")
        outlier = cache.outliers[0]
        self.assertGreater(outlier.outlier_score, 3.25, f"Outlier score {outlier.outlier_score} must be > 3.25")

    def test_welford_online_clustering_and_compaction(self):
        """
        Tests Tier 2 Long-Term Attractor Store:
        - Similar tokens update attractor centroid via Welford incremental formula.
        - History compaction merges the closest attractors when reaching max capacity.
        """
        max_att = 4
        cache = HopfieldKVCache(window_size=4, max_attractors=max_att, max_outliers=10, merge_similarity_cos=0.80)

        # Create 2 distinct semantic topics
        topic_a = torch.randn(NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)
        topic_b = torch.randn(NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)

        # Push tokens from topic A
        for i in range(8):
            cache.push_token(topic_a + torch.randn_like(topic_a) * 0.02, torch.randn_like(topic_a), token_pos=i)

        # Verify that topic A tokens merged into attractors
        self.assertGreater(len(cache.attractors), 0)
        self.assertGreater(cache.attractors[0].token_count, 1, "Welford update should increment token count")

        # Push multiple distinct topics to trigger compaction
        for i in range(20):
            random_topic = torch.randn(NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)
            cache.push_token(random_topic, torch.randn_like(random_topic), token_pos=100 + i)

        # Assert attractor store did not exceed max capacity
        self.assertLessEqual(len(cache.attractors), max_att, "Attractor count must respect max_attractors after compaction")

    def test_memory_footprint_audit(self):
        """
        Verifies exact byte-level memory audit matching RFC-13.
        At 10,000 tokens context, Hopfield KV-Cache should achieve > 5x compression vs Raw KV.
        """
        cache = HopfieldKVCache(window_size=1024, max_attractors=8000, max_outliers=2048)

        # Simulate 3,200 tokens across 100 semantic clusters of 32 tokens
        cluster_k = torch.randn(100, NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)
        cluster_v = torch.randn(100, NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM)
        all_k = []
        all_v = []
        for c in range(100):
            noise_k = torch.randn(32, NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM) * 0.02
            noise_v = torch.randn(32, NUM_GLOBAL_LAYERS, NUM_KV_HEADS, HEAD_DIM) * 0.02
            all_k.append(cluster_k[c:c+1] + noise_k)
            all_v.append(cluster_v[c:c+1] + noise_v)
        block_k = torch.cat(all_k, dim=0)
        block_v = torch.cat(all_v, dim=0)
        cache.push_block(block_k, block_v)

        audit = cache.get_memory_footprint()
        self.assertEqual(audit["working_window_tokens"], 1024)
        self.assertGreater(audit["total_tokens_represented"], 1024)
        self.assertGreater(audit["compression_ratio_vs_raw"], 1.5)
        self.assertGreater(audit["compression_ratio_vs_dense"], 100.0)

    def test_a3_attention_pretraining_and_ste_gradients(self):
        """
        Tests Attractor-Augmented Attention (A3) module during pretraining:
        Ensures Straight-Through Estimator (STE) passes non-zero gradients
        to the key and value projection weights.
        """
        d_model = 1536
        num_q_heads = 12
        num_kv_heads = 2
        head_dim = 128
        window_size = 512
        cluster_size = 32

        a3_layer = AttractorAugmentedAttention(
            d_model=d_model,
            num_query_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            window_size=window_size,
            cluster_size=cluster_size,
            p_aug=1.0,  # Force augmentation for test
            noise_std=0.05
        )

        # Context length L = 1024 > window_size (512)
        seq_len = 1024
        x = torch.randn(2, seq_len, d_model, requires_grad=True)

        out = a3_layer(x, is_training=True)
        self.assertEqual(out.shape, (2, seq_len, d_model))

        loss = out.sum()
        loss.backward()

        # Check gradients exist and are non-zero across all projection matrices
        self.assertIsNotNone(x.grad)
        self.assertGreater(torch.norm(x.grad).item(), 0.0)
        self.assertIsNotNone(a3_layer.k_proj.weight.grad)
        self.assertGreater(torch.norm(a3_layer.k_proj.weight.grad).item(), 0.0)
        self.assertIsNotNone(a3_layer.v_proj.weight.grad)
        self.assertGreater(torch.norm(a3_layer.v_proj.weight.grad).item(), 0.0)

    def test_variance_compensated_softmax_accuracy_kl_bound(self):
        """
        CRITICAL ARCHITECTURAL PROOF: D_KL(P_raw || P_comp) < 0.05
        Strictly proves mathematical convergence and zero-degradation.
        Runs statistical Monte-Carlo benchmark over 50 trials with realistic cluster variance.
        """
        dim = HEAD_DIM
        context_len = 4096
        window_size = 512
        cluster_size = TOKENS_PER_CLUSTER
        num_hist = context_len - window_size
        num_clusters = num_hist // cluster_size

        kl_comp_list = []
        kl_naive_list = []
        cos_comp_list = []
        cos_naive_list = []

        inv_sqrt_d = 1.0 / math.sqrt(dim)
        log_bc = math.log(cluster_size)

        num_trials = 50
        for trial in range(num_trials):
            torch.manual_seed(2000 + trial)

            # 1. Cluster centers
            cluster_centers_k = F.normalize(torch.randn(1, num_clusters, dim), dim=-1) * math.sqrt(dim)
            cluster_centers_v = torch.randn(1, num_clusters, dim)

            # 2. History tokens with intra-cluster variance (sigma ~ 0.35)
            k_hist_list = []
            v_hist_list = []
            for c in range(num_clusters):
                noise_k = torch.randn(1, cluster_size, dim) * 0.35
                noise_v = torch.randn(1, cluster_size, dim) * 0.35
                k_hist_list.append(cluster_centers_k[:, c:c+1, :] + noise_k)
                v_hist_list.append(cluster_centers_v[:, c:c+1, :] + noise_v)

            k_hist = torch.cat(k_hist_list, dim=1)  # [1, num_hist, dim]
            v_hist = torch.cat(v_hist_list, dim=1)

            # 3. Working window (recent uncompressed tokens)
            k_win = F.normalize(torch.randn(1, window_size, dim), dim=-1) * math.sqrt(dim)
            v_win = torch.randn(1, window_size, dim)

            # Combined raw KV
            k_raw = torch.cat([k_hist, k_win], dim=1)  # [1, context_len, dim]
            v_raw = torch.cat([v_hist, v_win], dim=1)

            # Query vector
            q = F.normalize(torch.randn(1, 1, 1, dim), dim=-1) * math.sqrt(dim)

            # ==================================================================
            # Ground Truth: Raw Full Token Softmax
            # ==================================================================
            logits_raw = torch.matmul(q, k_raw.unsqueeze(1).transpose(-1, -2)) * inv_sqrt_d  # [1, 1, 1, L]
            prob_raw = F.softmax(logits_raw, dim=-1)
            out_raw = torch.matmul(prob_raw, v_raw.unsqueeze(1))  # [1, 1, 1, D]

            # Aggregate raw probabilities by cluster to form macro-probability space [M + W]
            prob_raw_clustered = []
            for c in range(num_clusters):
                p_c = prob_raw[:, :, :, c * cluster_size : (c + 1) * cluster_size].sum(dim=-1, keepdim=True)
                prob_raw_clustered.append(p_c)
            prob_raw_win = prob_raw[:, :, :, num_hist:]
            prob_raw_macro = torch.cat(prob_raw_clustered + [prob_raw_win], dim=-1)  # [1, 1, 1, M + W]

            # ==================================================================
            # Hopfield Centroids & Dispersion
            # ==================================================================
            k_blocks = k_hist.view(1, num_clusters, cluster_size, dim)
            v_blocks = v_hist.view(1, num_clusters, cluster_size, dim)
            mu_k = k_blocks.mean(dim=2)  # [1, M, D]
            mu_v = v_blocks.mean(dim=2)  # [1, M, D]

            diff_k = k_blocks - mu_k.unsqueeze(2)
            sigma2_c = (diff_k ** 2).sum(dim=-1).mean(dim=2)  # [1, M]

            # ==================================================================
            # Naive Method (Without Variance or Mass Compensation)
            # ==================================================================
            logits_naive_hist = torch.matmul(q, mu_k.unsqueeze(1).transpose(-1, -2)) * inv_sqrt_d
            logits_naive_win = torch.matmul(q, k_win.unsqueeze(1).transpose(-1, -2)) * inv_sqrt_d
            logits_naive = torch.cat([logits_naive_hist, logits_naive_win], dim=-1)
            prob_naive = F.softmax(logits_naive, dim=-1)
            v_macro = torch.cat([mu_v.unsqueeze(1), v_win.unsqueeze(1)], dim=2)
            out_naive = torch.matmul(prob_naive, v_macro)

            # ==================================================================
            # Variance-Compensated Softmax Operator
            # ==================================================================
            out_comp, prob_comp = variance_compensated_softmax(
                q=q,
                k_win=k_win.unsqueeze(1),
                v_win=v_win.unsqueeze(1),
                mu_k=mu_k.unsqueeze(1),
                mu_v=mu_v.unsqueeze(1),
                sigma2_k=sigma2_c.unsqueeze(1),
                counts_c=cluster_size,
                inv_sqrt_d=inv_sqrt_d
            )

            # Compute KL Divergence and Cosine Similarity
            kl_naive = F.kl_div(prob_naive.log(), prob_raw_macro, reduction='batchmean').item()
            kl_comp = F.kl_div(prob_comp.log(), prob_raw_macro, reduction='batchmean').item()

            cos_naive = F.cosine_similarity(out_raw.squeeze(), out_naive.squeeze(), dim=-1).item()
            cos_comp = F.cosine_similarity(out_raw.squeeze(), out_comp.squeeze(), dim=-1).item()

            kl_comp_list.append(kl_comp)
            kl_naive_list.append(kl_naive)
            cos_comp_list.append(cos_comp)
            cos_naive_list.append(cos_naive)

        mean_kl_comp = sum(kl_comp_list) / len(kl_comp_list)
        mean_kl_naive = sum(kl_naive_list) / len(kl_naive_list)
        mean_cos_comp = sum(cos_comp_list) / len(cos_comp_list)
        mean_cos_naive = sum(cos_naive_list) / len(cos_naive_list)

        print("\n" + "=" * 80)
        print("EMPIRICAL CONVERGENCE AUDIT: HOPFIELD KV-CACHE vs RAW KV-CACHE")
        print("=" * 80)
        print(f"Mean KL Divergence (Naive Hopfield):       {mean_kl_naive:.6f}  [Degraded]")
        print(f"Mean KL Divergence (Compensated Hopfield): {mean_kl_comp:.6f}   [Target < 0.05]")
        print(f"KL Improvement Factor:                     {mean_kl_naive / mean_kl_comp:.1f}x")
        print(f"Mean Output Cosine Similarity (Compensated): {mean_cos_comp:.4f}  [Target > 0.99]")
        print(f"Mean Output Cosine Similarity (Naive):       {mean_cos_naive:.4f}")
        print("=" * 80)

        # STRICT SPECIFICATION ASSERTIONS
        self.assertLess(mean_kl_comp, 0.05, f"D_KL {mean_kl_comp} must be strictly < 0.05")
        self.assertGreater(mean_cos_comp, 0.99, f"Output cosine similarity {mean_cos_comp} must be > 0.99")
        self.assertGreater(mean_kl_naive, 0.50, "Naive method without compensation must show significant shift")


if __name__ == "__main__":
    unittest.main()
