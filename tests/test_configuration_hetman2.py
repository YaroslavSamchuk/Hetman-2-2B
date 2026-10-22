"""
================================================================================
HETMAN-2-NCM: PRODUCTION CONFIGURATION VERIFICATION SUITE
================================================================================
Target: src/configuration_hetman2.py (Hetman2Config)
        configs/hetman2_ncm_config.json
Standards: RFC-01 through RFC-15 specifications
           PyTorch / HuggingFace Transformers / Rada.cpp compatibility
================================================================================
"""

import json
import os
import struct
import tempfile
import unittest

from transformers import AutoConfig

from src.configuration_hetman2 import Hetman2Config


class TestHetman2Config(unittest.TestCase):
    """
    Test suite verifying the architectural invariants, Hugging Face compatibility,
    and Rada.cpp exporter compliance for Hetman-2-NCM 2.0B configuration.
    """

    def setUp(self):
        self.config = Hetman2Config()

    def test_01_mandatory_parameters(self):
        """Verify all 10 mandatory architectural parameters specified by the architect."""
        self.assertEqual(self.config.d_model, 1536)
        self.assertEqual(self.config.layers, 34)
        self.assertEqual(self.config.intermediate_dim, 6144)
        self.assertEqual(self.config.vocab_size, 65536)
        self.assertEqual(self.config.hopfield_slots, 16777216)
        self.assertEqual(self.config.block_size, 400)
        self.assertEqual(self.config.base_passes, 8)
        self.assertEqual(self.config.emergency_passes, 3)
        self.assertEqual(self.config.rvq_codebooks, 4)
        self.assertEqual(self.config.version, 2)

    def test_02_hybrid_topology_split(self):
        """Verify the 28 DeltaNet + 6 Global Softmax GQA (12:2) topology."""
        self.assertEqual(self.config.num_deltanet_layers, 28)
        self.assertEqual(self.config.num_global_layers, 6)
        self.assertEqual(self.config.layers, 28 + 6)
        self.assertEqual(len(self.config.global_layer_indices), 6)
        self.assertEqual(self.config.global_layer_indices, [7, 13, 17, 21, 25, 29])

        # GQA 12:2 verification
        self.assertEqual(self.config.global_attention_heads, 12)
        self.assertEqual(self.config.global_kv_heads, 2)
        self.assertEqual(self.config.global_head_dim, 128)
        self.assertEqual(self.config.global_attention_heads * self.config.global_head_dim, 1536)

        # DeltaNet recurrence verification
        self.assertEqual(self.config.deltanet_num_heads, 16)
        self.assertEqual(self.config.deltanet_head_dim, 64)
        self.assertEqual(self.config.deltanet_inner_dim, 1024)
        self.assertEqual(self.config.deltanet_conv_kernel, 4)
        self.assertEqual(self.config.deltanet_state_dtype, "float32")

    def test_03_hopfield_and_blueprint_channels(self):
        """Verify SLH-Core v2 and Blueprint channel partitioning (sum = 1536)."""
        # Hopfield slots: 16 groups * 1024 C1 * 1024 C2 = 16,777,216
        self.assertEqual(self.config.hopfield_num_groups, 16)
        self.assertEqual(self.config.hopfield_codebook_size, 1024)
        self.assertEqual(
            self.config.hopfield_slots,
            self.config.hopfield_num_groups * (self.config.hopfield_codebook_size ** 2),
        )
        self.assertEqual(self.config.hopfield_val_dim, 32)
        self.assertEqual(self.config.hopfield_topk_1 * self.config.hopfield_topk_2, 32)

        # Blueprint partition: 768 + 384 + 256 + 128 = 1536
        self.assertEqual(self.config.dim_c_sem, 768)
        self.assertEqual(self.config.dim_idx_hop, 384)
        self.assertEqual(self.config.dim_tau_style, 256)
        self.assertEqual(self.config.dim_ell_budget, 128)
        self.assertEqual(
            self.config.dim_c_sem
            + self.config.dim_idx_hop
            + self.config.dim_tau_style
            + self.config.dim_ell_budget,
            1536,
        )

    def test_04_huggingface_alias_synchrony(self):
        """Verify standard Hugging Face attribute aliases are synchronized."""
        self.assertEqual(self.config.hidden_size, self.config.d_model)
        self.assertEqual(self.config.num_hidden_layers, self.config.layers)
        self.assertEqual(self.config.num_layers, self.config.layers)
        self.assertEqual(self.config.intermediate_size, self.config.intermediate_dim)
        self.assertEqual(self.config.tokens_per_block, self.config.block_size)
        self.assertEqual(self.config.num_rvq_codebooks, self.config.rvq_codebooks)
        self.assertEqual(self.config.rvq_levels, self.config.rvq_codebooks)

    def test_05_huggingface_serialization(self):
        """Verify save_pretrained and from_pretrained roundtrip with Hugging Face."""
        with tempfile.TemporaryDirectory() as tmpdir:
            self.config.save_pretrained(tmpdir)
            loaded_cfg = Hetman2Config.from_pretrained(tmpdir)

            self.assertEqual(loaded_cfg.model_type, "hetman2_ncm")
            self.assertEqual(loaded_cfg.d_model, 1536)
            self.assertEqual(loaded_cfg.layers, 34)
            self.assertEqual(loaded_cfg.intermediate_dim, 6144)
            self.assertEqual(loaded_cfg.vocab_size, 65536)
            self.assertEqual(loaded_cfg.hopfield_slots, 16777216)
            self.assertEqual(loaded_cfg.block_size, 400)
            self.assertEqual(loaded_cfg.base_passes, 8)
            self.assertEqual(loaded_cfg.emergency_passes, 3)
            self.assertEqual(loaded_cfg.rvq_codebooks, 4)
            self.assertEqual(loaded_cfg.version, 2)

    def test_06_autoconfig_registration(self):
        """Verify AutoConfig can resolve Hetman2Config by model_type."""
        auto_cfg = AutoConfig.for_model("hetman2_ncm")
        self.assertIsInstance(auto_cfg, Hetman2Config)
        self.assertEqual(auto_cfg.d_model, 1536)
        self.assertEqual(auto_cfg.layers, 34)

    def test_07_memory_budget_audit(self):
        """Verify strict memory budget: DeltaNet = 7.00 MiB, Peak Arena < 750 MiB."""
        budget = self.config.calculate_memory_budget()

        # DeltaNet state memory check: exactly 7,340,032 bytes (7.00 MiB)
        self.assertEqual(budget["bytes_deltanet_state"], 7340032)
        self.assertAlmostEqual(budget["mib_deltanet_state"], 7.0, places=5)

        # Monolithic arena check: < 750 MiB (strictly 525.74 MiB)
        self.assertTrue(budget["budget_passed"])
        self.assertLess(budget["mib_total_monolithic_arena"], 750.0)
        self.assertAlmostEqual(budget["mib_total_monolithic_arena"], 525.74, delta=1.0)

    def test_08_rada_binary_header_packing(self):
        """Verify exact 128-byte binary container header matching rada_ncm_engine.hpp."""
        hdr_bytes = self.config.pack_rada_header_bytes(
            tensor_table_offset=128,
            tensor_data_offset=2048,
            bpe_vocab_offset=4096,
            bpe_vocab_size_bytes=65536,
        )

        self.assertEqual(len(hdr_bytes), 128)

        # Unpack according to struct layout: <18I4Q24s
        fields = struct.unpack("<18I4Q24s", hdr_bytes)
        magic = fields[0]
        version = fields[1]
        vocab_size = fields[2]
        hidden_size = fields[3]
        intermediate_size = fields[4]
        num_layers = fields[5]
        num_deltanet_layers = fields[6]
        num_global_layers = fields[7]
        deltanet_heads = fields[8]
        deltanet_head_dim = fields[9]
        global_heads_q = fields[10]
        global_heads_kv = fields[11]
        global_head_dim = fields[12]
        hopfield_groups = fields[13]
        hopfield_codebook_size = fields[14]
        hopfield_val_dim = fields[15]
        tokens_per_block = fields[16]
        max_blocks = fields[17]
        tensor_table_offset = fields[18]
        tensor_data_offset = fields[19]
        bpe_vocab_offset = fields[20]
        bpe_vocab_size_bytes = fields[21]

        self.assertEqual(magic, 0x4D434E52)  # "RNCM"
        self.assertEqual(version, 2)
        self.assertEqual(vocab_size, 65536)
        self.assertEqual(hidden_size, 1536)
        self.assertEqual(intermediate_size, 6144)
        self.assertEqual(num_layers, 34)
        self.assertEqual(num_deltanet_layers, 28)
        self.assertEqual(num_global_layers, 6)
        self.assertEqual(deltanet_heads, 16)
        self.assertEqual(deltanet_head_dim, 64)
        self.assertEqual(global_heads_q, 12)
        self.assertEqual(global_heads_kv, 2)
        self.assertEqual(global_head_dim, 128)
        self.assertEqual(hopfield_groups, 16)
        self.assertEqual(hopfield_codebook_size, 1024)
        self.assertEqual(hopfield_val_dim, 32)
        self.assertEqual(tokens_per_block, 400)
        self.assertEqual(max_blocks, 8)
        self.assertEqual(tensor_table_offset, 128)
        self.assertEqual(tensor_data_offset, 2048)
        self.assertEqual(bpe_vocab_offset, 4096)
        self.assertEqual(bpe_vocab_size_bytes, 65536)

    def test_09_production_json_config_file(self):
        """Verify that configs/hetman2_ncm_config.json exists and loads with 100% fidelity."""
        cfg_path = os.path.join(
            os.path.dirname(__file__), "..", "configs", "hetman2_ncm_config.json"
        )
        self.assertTrue(os.path.exists(cfg_path), f"Config file not found at {cfg_path}")

        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.assertEqual(data["d_model"], 1536)
        self.assertEqual(data["layers"], 34)
        self.assertEqual(data["num_deltanet_layers"], 28)
        self.assertEqual(data["num_global_layers"], 6)
        self.assertEqual(data["intermediate_dim"], 6144)
        self.assertEqual(data["vocab_size"], 65536)
        self.assertEqual(data["hopfield_slots"], 16777216)
        self.assertEqual(data["block_size"], 400)
        self.assertEqual(data["base_passes"], 8)
        self.assertEqual(data["emergency_passes"], 3)
        self.assertEqual(data["rvq_codebooks"], 4)
        self.assertEqual(data["version"], 2)

        # Ensure Hetman2Config can load directly from this json file
        loaded_from_json = Hetman2Config.from_json_file(cfg_path)
        self.assertEqual(loaded_from_json.d_model, 1536)
        self.assertEqual(loaded_from_json.layers, 34)
        self.assertEqual(loaded_from_json.hopfield_slots, 16777216)

    def test_10_invariant_validation_failure_modes(self):
        """Verify that invalid configurations raise assertion errors."""
        # Invalid layer sum (e.g., 35 layers != 28 + 6)
        with self.assertRaises(AssertionError):
            Hetman2Config(layers=35, num_deltanet_layers=28, num_global_layers=6)

        # Invalid FWHT block size (not divisible by 512)
        with self.assertRaises(AssertionError):
            Hetman2Config(d_model=1500, hadamard_block_size=512)

        # Invalid DeltaNet state dtype (must be FP32)
        with self.assertRaises(AssertionError):
            Hetman2Config(deltanet_state_dtype="float16")


if __name__ == "__main__":
    unittest.main()
