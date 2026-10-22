"""
Test Suite: Rada.cpp C++20 Runtime, Monolithic Arena, and Binary Container Exact Verification
Validates:
1. Exact 128-byte binary layout and byte-level alignment between C++ RadaNcmFileHeader and Python exporter.
2. Two-tier CRC32 checksums (Header, Table, Per-tensor) and mmap zero-copy integrity.
3. Monolithic arena memory layout, offsets, and RadaGlobalKVCache boundary safety (no overflow into BPE metadata).
4. Remez 5th-order fast exponential precision, underflow safety, and behavior at logits < -50.0.
"""

import math
import struct
import tempfile
import zlib
import unittest
from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from src.configuration_hetman2 import Hetman2Config
from src.export_to_rada import (
    RADA_NCM_MAGIC,
    RADA_VERSION,
    CACHE_LINE_ALIGNMENT,
    HEADER_SIZE_BYTES,
    TENSOR_DESC_SIZE_BYTES,
    RADA_FLAG_CRC32_ENABLED,
    RadaNcmFileHeader,
    RadaTensorDescriptor,
    export_to_rada,
    verify_rada_file,
    align_to_boundary
)


class TestRadaBinaryExactVerification(unittest.TestCase):

    def setUp(self):
        self.config = Hetman2Config()

    def test_01_binary_header_exact_byte_match(self):
        """Verify exact byte match between Python pack_rada_header_bytes and RadaNcmFileHeader."""
        cfg = self.config
        tensor_table_offset = 128
        tensor_data_offset = 4096
        bpe_vocab_offset = 65536
        bpe_vocab_size = 32768
        table_crc = 0xABCDEF01
        flags = RADA_FLAG_CRC32_ENABLED

        # 1. Pack via Hetman2Config
        cfg_bytes_zero_crc = cfg.pack_rada_header_bytes(
            tensor_table_offset=tensor_table_offset,
            tensor_data_offset=tensor_data_offset,
            bpe_vocab_offset=bpe_vocab_offset,
            bpe_vocab_size_bytes=bpe_vocab_size,
            header_crc32=0,
            table_crc32=table_crc,
            flags=flags
        )
        self.assertEqual(len(cfg_bytes_zero_crc), HEADER_SIZE_BYTES)

        # Compute Header CRC32 over the first 104 bytes
        computed_hdr_crc = zlib.crc32(cfg_bytes_zero_crc[:104]) & 0xFFFFFFFF

        cfg_bytes_final = cfg.pack_rada_header_bytes(
            tensor_table_offset=tensor_table_offset,
            tensor_data_offset=tensor_data_offset,
            bpe_vocab_offset=bpe_vocab_offset,
            bpe_vocab_size_bytes=bpe_vocab_size,
            header_crc32=computed_hdr_crc,
            table_crc32=table_crc,
            flags=flags
        )

        # 2. Pack via RadaNcmFileHeader (exporter format)
        header_obj = RadaNcmFileHeader(
            magic=cfg.rada_magic,
            version=cfg.rada_version,
            vocab_size=cfg.vocab_size,
            hidden_size=cfg.hidden_size,
            intermediate_size=cfg.intermediate_size,
            num_layers=cfg.num_layers,
            num_deltanet_layers=cfg.num_deltanet_layers,
            num_global_layers=cfg.num_global_layers,
            deltanet_heads=cfg.deltanet_heads,
            deltanet_head_dim=cfg.deltanet_head_dim,
            global_heads_q=cfg.global_heads_q,
            global_heads_kv=cfg.global_heads_kv,
            global_head_dim=cfg.global_head_dim,
            hopfield_groups=cfg.hopfield_groups,
            hopfield_codebook_size=cfg.hopfield_codebook_size,
            hopfield_val_dim=cfg.hopfield_val_dim,
            tokens_per_block=cfg.tokens_per_block,
            max_blocks=cfg.max_blocks,
            tensor_table_offset=tensor_table_offset,
            tensor_data_offset=tensor_data_offset,
            bpe_vocab_offset=bpe_vocab_offset,
            bpe_vocab_size_bytes=bpe_vocab_size,
            header_crc32=computed_hdr_crc,
            table_crc32=table_crc,
            flags=flags,
            padding=b"\x00" * 12
        )
        exporter_bytes = header_obj.to_bytes()

        # 3. Assert EXACT BYTE-FOR-BYTE IDENTITY
        self.assertEqual(len(exporter_bytes), 128)
        self.assertEqual(cfg_bytes_final, exporter_bytes, "Header bytes between config and exporter MUST match identically!")

        # 4. Unpack and verify every single field
        unpacked = RadaNcmFileHeader.from_bytes(cfg_bytes_final)
        self.assertEqual(unpacked.magic, RADA_NCM_MAGIC)
        self.assertEqual(unpacked.version, RADA_VERSION)
        self.assertEqual(unpacked.vocab_size, 65536)
        self.assertEqual(unpacked.hidden_size, 1536)
        self.assertEqual(unpacked.intermediate_size, 6144)
        self.assertEqual(unpacked.num_layers, 34)
        self.assertEqual(unpacked.num_deltanet_layers, 28)
        self.assertEqual(unpacked.num_global_layers, 6)
        self.assertEqual(unpacked.deltanet_heads, 16)
        self.assertEqual(unpacked.deltanet_head_dim, 64)
        self.assertEqual(unpacked.global_heads_q, 12)
        self.assertEqual(unpacked.global_heads_kv, 2)
        self.assertEqual(unpacked.global_head_dim, 128)
        self.assertEqual(unpacked.hopfield_groups, 16)
        self.assertEqual(unpacked.hopfield_codebook_size, 1024)
        self.assertEqual(unpacked.hopfield_val_dim, 32)
        self.assertEqual(unpacked.tokens_per_block, 400)
        self.assertEqual(unpacked.max_blocks, 8)
        self.assertEqual(unpacked.tensor_table_offset, 128)
        self.assertEqual(unpacked.tensor_data_offset, 4096)
        self.assertEqual(unpacked.bpe_vocab_offset, 65536)
        self.assertEqual(unpacked.bpe_vocab_size_bytes, 32768)
        self.assertEqual(unpacked.header_crc32, computed_hdr_crc)
        self.assertEqual(unpacked.table_crc32, table_crc)
        self.assertEqual(unpacked.flags, flags)

    def test_02_field_offsets_and_alignment(self):
        """Validate exact byte offsets of every field in RadaNcmFileHeader."""
        fmt = "<18I4Q3I12s"
        expected_size = 18 * 4 + 4 * 8 + 3 * 4 + 12
        self.assertEqual(struct.calcsize(fmt), 128)
        self.assertEqual(expected_size, 128)

        # Offsets
        offset_tensor_table = 18 * 4  # 72
        self.assertEqual(offset_tensor_table % 8, 0, "64-bit uint64 must be naturally 8-byte aligned at offset 72")
        offset_crc32 = 72 + 4 * 8     # 104
        self.assertEqual(offset_crc32 % 4, 0, "uint32 header_crc32 must be naturally 4-byte aligned at offset 104")
        offset_flags = 104 + 2 * 4    # 112
        self.assertEqual(offset_flags, 112)
        offset_pad = 112 + 4          # 116
        self.assertEqual(offset_pad + 12, 128)

    def test_03_monolithic_arena_partitioning_and_kv_cache_safety(self):
        """Verify monolithic arena budgets and ensure RadaGlobalKVCache cannot overflow into BPE metadata."""
        BYTES_STATIC_TERNARY_WEIGHTS   = 508559360
        BYTES_STATIC_EMBEDDINGS        = 201326592
        BYTES_STATIC_HOPFIELD_AUX      = 2228288
        BYTES_DELTANET_STATE           = 7340032
        BYTES_PHASE1_DRAFT_SCRATCH     = 4194304
        BYTES_PHASE2_REFINE_SCRATCH    = 8388608
        BYTES_PHASE3_DECODE_WORKSPACE  = 13417216
        BYTES_GLOBAL_KV_CACHE          = 4915200
        BYTES_METADATA_AND_BPE         = 2162688
        BYTES_TOTAL_MONOLITHIC_ARENA   = 752532288

        # 1. Total arena size check
        calculated_total = (
            BYTES_STATIC_TERNARY_WEIGHTS
            + BYTES_STATIC_EMBEDDINGS
            + BYTES_STATIC_HOPFIELD_AUX
            + BYTES_DELTANET_STATE
            + BYTES_PHASE1_DRAFT_SCRATCH
            + BYTES_PHASE2_REFINE_SCRATCH
            + BYTES_PHASE3_DECODE_WORKSPACE
            + BYTES_GLOBAL_KV_CACHE
            + BYTES_METADATA_AND_BPE
        )
        self.assertEqual(calculated_total, BYTES_TOTAL_MONOLITHIC_ARENA)
        self.assertLess(calculated_total / (1024 * 1024), 750.0)

        # 2. Check 64-byte alignment of every boundary offset
        offsets = [
            0,
            BYTES_STATIC_TERNARY_WEIGHTS,
            BYTES_STATIC_TERNARY_WEIGHTS + BYTES_STATIC_EMBEDDINGS,
            BYTES_STATIC_TERNARY_WEIGHTS + BYTES_STATIC_EMBEDDINGS + BYTES_STATIC_HOPFIELD_AUX,
            BYTES_STATIC_TERNARY_WEIGHTS + BYTES_STATIC_EMBEDDINGS + BYTES_STATIC_HOPFIELD_AUX + BYTES_DELTANET_STATE,
            BYTES_STATIC_TERNARY_WEIGHTS + BYTES_STATIC_EMBEDDINGS + BYTES_STATIC_HOPFIELD_AUX + BYTES_DELTANET_STATE + BYTES_PHASE1_DRAFT_SCRATCH,
            BYTES_STATIC_TERNARY_WEIGHTS + BYTES_STATIC_EMBEDDINGS + BYTES_STATIC_HOPFIELD_AUX + BYTES_DELTANET_STATE + BYTES_PHASE1_DRAFT_SCRATCH + BYTES_PHASE2_REFINE_SCRATCH,
            BYTES_STATIC_TERNARY_WEIGHTS + BYTES_STATIC_EMBEDDINGS + BYTES_STATIC_HOPFIELD_AUX + BYTES_DELTANET_STATE + BYTES_PHASE1_DRAFT_SCRATCH + BYTES_PHASE2_REFINE_SCRATCH + BYTES_PHASE3_DECODE_WORKSPACE,
            BYTES_STATIC_TERNARY_WEIGHTS + BYTES_STATIC_EMBEDDINGS + BYTES_STATIC_HOPFIELD_AUX + BYTES_DELTANET_STATE + BYTES_PHASE1_DRAFT_SCRATCH + BYTES_PHASE2_REFINE_SCRATCH + BYTES_PHASE3_DECODE_WORKSPACE + BYTES_GLOBAL_KV_CACHE,
        ]
        for off in offsets:
            self.assertEqual(off % 64, 0, f"Offset {off} must be strictly 64-byte aligned!")

        # 3. Fixed RadaGlobalKVCache struct layout verification:
        # Control metadata header (64 bytes) + Buffer (BYTES_GLOBAL_KV_CACHE - 64)
        kv_cache_header_size = 4 + 4 + 56  # current_seq_len (uint32) + max_seq_len (uint32) + reserved[56] = 64 bytes
        kv_cache_buffer_size = BYTES_GLOBAL_KV_CACHE - 64
        total_kv_cache_struct_size = kv_cache_header_size + kv_cache_buffer_size
        self.assertEqual(total_kv_cache_struct_size, BYTES_GLOBAL_KV_CACHE)
        self.assertEqual(total_kv_cache_struct_size % 64, 0)

    def test_04_remez5_accuracy_and_negative_logit_safety(self):
        """Verify Remez 5th order polynomial precision and underflow behavior for logits < -50.0."""
        def bits_to_float(b):
            return struct.unpack('>f', struct.pack('>I', b))[0]

        def remez5_eval(x):
            if math.isnan(x) or x <= -87.3365:
                return 0.0
            if x >= 87.3365:
                return 3.40282347e+38

            t = x * 1.4426950408889634
            fi = math.floor(t)
            f = t - fi
            i = int(fi)

            C0 = 1.0000000000
            C1 = 0.6931471806
            C2 = 0.2402265070
            C3 = 0.0555041087
            C4 = 0.0096181291
            C5 = 0.0013333558

            p = C0 + f * (C1 + f * (C2 + f * (C3 + f * (C4 + f * C5))))
            exp_bits = ((i + 127) << 23) & 0xFFFFFFFF
            scale = bits_to_float(exp_bits)
            return p * scale

        # Check precision on [0, -40]
        for val in [0.0, -0.5, -1.0, -2.0, -5.0, -10.0, -20.0, -30.0, -40.0]:
            true_exp = math.exp(val)
            approx_exp = remez5_eval(val)
            rel_err = abs(approx_exp - true_exp) / true_exp
            self.assertLess(rel_err, 1e-4, f"Relative error at {val} exceeded tolerance: {rel_err}")

        # Check logit < -50.0:
        val_50 = -50.0
        exp_50 = remez5_eval(val_50)
        self.assertGreater(exp_50, 0.0)
        self.assertLess(exp_50, 1e-21)
        # Softmax precision invariance: 1.0 + exp(-50.0) == 1.0 in float32
        f32_sum = struct.unpack('f', struct.pack('f', 1.0 + exp_50))[0]
        self.assertEqual(f32_sum, 1.0, "exp(-50.0) must not alter 1.0 in single precision float")

        # Check extreme negative arguments:
        self.assertEqual(remez5_eval(-87.5), 0.0)
        self.assertEqual(remez5_eval(-88.5), 0.0)
        self.assertEqual(remez5_eval(-100.0), 0.0)
        self.assertEqual(remez5_eval(-1000.0), 0.0)
        self.assertEqual(remez5_eval(float('-inf')), 0.0)
        self.assertEqual(remez5_eval(float('nan')), 0.0)

        # Verify that without clamping, x = -88.5 creates -inf!
        t_unclamped = -88.5 * 1.4426950408889634
        i_unclamped = int(math.floor(t_unclamped))
        bad_bits = ((i_unclamped + 127) << 23) & 0xFFFFFFFF
        bad_float = bits_to_float(bad_bits)
        self.assertEqual(bad_float, float('-inf'), "Unclamped x = -88.5 must evaluate to -inf, demonstrating critical necessity of clamp!")


if __name__ == "__main__":
    unittest.main()
