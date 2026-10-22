"""
Test suite for AUDIT-16: 64-byte alignment, hierarchical CRC32 checksums,
backward-compatibility safe probing, and streaming export.
"""

import os
import struct
import tempfile
import zlib
import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from src.export_to_rada import (
    RADA_NCM_MAGIC,
    RADA_LEGACY_MAGIC,
    RADA_VERSION,
    CACHE_LINE_ALIGNMENT,
    HEADER_SIZE_BYTES,
    TENSOR_DESC_SIZE_BYTES,
    DTYPE_TERNARY_2BIT,
    align_to_boundary,
    pack_ternary_weights,
    unpack_ternary_weights,
    export_to_rada,
    verify_rada_file,
    RadaNcmFileHeader,
    RadaTensorDescriptor,
)


def test_64byte_alignment_with_odd_dimensions():
    """
    Check 1: Verify that for ANY matrix with odd dimensions,
    every tensor offset and the next tensor offset is strictly a multiple of 64 bytes.
    """
    odd_state_dict = {
        "model.layers.0.weight_odd_2d": torch.randn(3, 7),        # Odd rows (3) and cols (7)
        "model.layers.0.weight_large_odd": torch.randn(15, 127),   # Odd rows (15) and cols (127)
        "model.layers.0.weight_single": torch.randn(1, 1),         # Odd 1x1
        "model.layers.0.norm.weight": torch.randn(1535),           # Odd 1D FP32 norm
        "model.layers.0.embed_tokens": torch.randn(31, 33),        # Odd 2D non-ternary
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_odd.rada")
        res = export_to_rada(odd_state_dict, out_path)

        audit = verify_rada_file(out_path, verify_all_crcs=True)
        assert audit["valid"] is True

        prev_end = audit["tensor_data_offset"]
        for desc in audit["descriptors"]:
            # Check 64-byte alignment of this tensor's offset
            assert desc.offset % CACHE_LINE_ALIGNMENT == 0, f"Offset {desc.offset} not aligned for {desc.name}"
            assert desc.size_bytes % CACHE_LINE_ALIGNMENT == 0, f"Size {desc.size_bytes} not aligned for {desc.name}"
            # Check that the previous tensor's end strictly equals or aligns with next offset
            assert desc.offset == prev_end, f"Offset gap: {desc.offset} != {prev_end}"
            prev_end = desc.offset + desc.size_bytes

        print("Test 1 (64-byte alignment with odd dimensions): PASSED")


def test_hierarchical_crc32_and_corruption_detection():
    """
    Check 2: Verify that CRC32 checksums are written, verified,
    and accurately catch bit-flips in payload, header, or table.
    """
    state_dict = {
        "weight_a": torch.randn(32, 64),
        "weight_b": torch.randn(16, 32),
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = os.path.join(tmpdir, "test_crc.rada")
        export_to_rada(state_dict, out_path)

        # Baseline verification should pass
        audit = verify_rada_file(out_path, verify_all_crcs=True)
        assert audit["valid"] is True
        assert int(audit["header_crc32"], 16) != 0
        assert int(audit["table_crc32"], 16) != 0

        # Simulate bit-flip in payload of weight_a
        with open(out_path, "r+b") as f:
            f.seek(audit["descriptors"][0].offset + 10)
            orig_byte = f.read(1)[0]
            f.seek(-1, os.SEEK_CUR)
            f.write(bytes([(orig_byte ^ 0xFF)]))

        # Verification must detect corruption and fail
        detected = False
        try:
            verify_rada_file(out_path, verify_all_crcs=True)
        except ValueError as e:
            if "CRC32 mismatch" in str(e):
                detected = True
        assert detected, "Failed to detect CRC32 corruption!"

        print("Test 2 (CRC32 and bit-flip corruption detection): PASSED")


def test_legacy_v1_detection_and_safe_probing():
    """
    Check 3: Verify that legacy Hetman-1 (magic=RADA, version=1)
    and truncated files are safely rejected without crashing / Segfault.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # 1. Truncated file (< 8 bytes)
        small_path = os.path.join(tmpdir, "small.rada")
        with open(small_path, "wb") as f:
            f.write(b"RADA")
        detected = False
        try:
            verify_rada_file(small_path)
        except ValueError as e:
            if "File too small" in str(e):
                detected = True
        assert detected, "Failed to reject truncated file!"

        # 2. Legacy v1 container
        legacy_path = os.path.join(tmpdir, "legacy_v1.rada")
        with open(legacy_path, "wb") as f:
            f.write(struct.pack("<II", RADA_LEGACY_MAGIC, 1))
            f.write(b"\x00" * 120)  # 128 bytes total
        detected = False
        try:
            verify_rada_file(legacy_path)
        except ValueError as e:
            if "Legacy Hetman-1 container detected" in str(e):
                detected = True
        assert detected, "Failed to reject legacy v1 container!"

        # 3. Invalid magic
        bad_magic_path = os.path.join(tmpdir, "bad_magic.rada")
        with open(bad_magic_path, "wb") as f:
            f.write(struct.pack("<II", 0xDEADBEEF, 2))
            f.write(b"\x00" * 120)
        detected = False
        try:
            verify_rada_file(bad_magic_path)
        except ValueError as e:
            if "Invalid magic number" in str(e):
                detected = True
        assert detected, "Failed to reject bad magic container!"

        print("Test 3 (Legacy v1 detection and safe 8-byte probe): PASSED")


def test_bitwise_alignment_fast_properties():
    """
    Verify bitwise align_to_boundary properties for arbitrary numbers.
    """
    for val in [0, 1, 63, 64, 65, 127, 128, 1000, 4194303]:
        aligned = align_to_boundary(val, 64)
        assert aligned % 64 == 0
        assert aligned >= val
        if val % 64 == 0:
            assert aligned == val
        else:
            assert aligned == ((val // 64) + 1) * 64

    print("Test 4 (Bitwise alignment properties): PASSED")


if __name__ == "__main__":
    test_64byte_alignment_with_odd_dimensions()
    test_hierarchical_crc32_and_corruption_detection()
    test_legacy_v1_detection_and_safe_probing()
    test_bitwise_alignment_fast_properties()
    print("ALL AUDIT-16 TESTS PASSED!")
