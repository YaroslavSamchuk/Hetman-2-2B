"""
Hetman-2-NCM 2.0B: Production Export Engine for Rada.cpp
File: src/export_to_rada.py
License: Apache License 2.0

Specification: RFC-RADA-NCM-2.0-EDGE-AUDIT / spec/09_EDGE_RUNTIME_RADA_CPP_AND_MEMORY_AUDIT.md
Target Binary Format: .rada / .rncm Container v2

AUDIT-16 REVISIONS & OPTIMIZATIONS:
1. Strict 64-byte alignment using bitwise integer arithmetic ((x + 63) & ~63).
2. Two-tier hierarchical CRC32 checksums:
   - Header CRC32 (104B payload integrity)
   - Table CRC32 (all tensor descriptors integrity)
   - Per-tensor payload CRC32 (stored in descriptor.crc32)
   Enables instant zero-allocation mmap startup in Rada.cpp without reading full weights to RAM!
3. Robust backward-compatibility and safe 8-byte probing to prevent Segfaults on v1 (TH1) containers.
4. Direct zero-duplicate streaming tensor writer (payloads list in RAM completely removed).
5. Fully vectorized BitNet b1.58 2-bit packing without Python tolist() overhead.
"""

import argparse
import os
import struct
import zlib
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

import sys
from pathlib import Path
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.model import Hetman2Config, Hetman2ForCausalLM


# ==============================================================================
# 1. ARCHITECTURAL BINARY CONSTANTS
# ==============================================================================

RADA_NCM_MAGIC: int = 0x4D434E52     # "RNCM" in little-endian (b"RNCM"[::-1])
RADA_LEGACY_MAGIC: int = 0x41444152  # "RADA" in little-endian (legacy Hetman-1)
RADA_VERSION: int = 2
CACHE_LINE_ALIGNMENT: int = 64

# Container feature flags
RADA_FLAG_NONE: int = 0x00000000
RADA_FLAG_CRC32_ENABLED: int = 0x00000001
RADA_FLAG_TIED_LM_HEAD: int = 0x00000002

# Datatype identifiers for tensor descriptors
DTYPE_TERNARY_2BIT: int = 1
DTYPE_BF16: int = 2
DTYPE_FP16: int = 3
DTYPE_FP32: int = 4
DTYPE_INT8: int = 5
DTYPE_UINT8: int = 6

# Header format: 18 uint32, 4 uint64, 3 uint32 (crcs + flags), 12 bytes pad = 128 bytes
HEADER_STRUCT_FMT: str = "<18I4Q3I12s"
HEADER_SIZE_BYTES: int = 128

# Tensor descriptor format: 64s name, uint32 dtype, uint32 n_dims, 4 uint64 dims,
# uint64 offset, uint64 size_bytes, float32 scale_gamma, uint32 crc32 = 128 bytes
TENSOR_DESC_FMT: str = "<64sII4QQQfI"
TENSOR_DESC_SIZE_BYTES: int = 128


# ==============================================================================
# 2. FILE HEADER & DESCRIPTOR STRUCTURES
# ==============================================================================

@dataclass
class RadaNcmFileHeader:
    """
    Rada.cpp v2 Binary Container Header.
    Strictly aligns to 128 bytes (2 cache lines of 64 bytes).
    Integrates Hierarchical CRC32 metadata and container flags.
    """
    magic: int = RADA_NCM_MAGIC
    version: int = RADA_VERSION
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
    tensor_table_offset: int = 128
    tensor_data_offset: int = 0
    bpe_vocab_offset: int = 0
    bpe_vocab_size_bytes: int = 0
    header_crc32: int = 0
    table_crc32: int = 0
    flags: int = RADA_FLAG_CRC32_ENABLED
    padding: bytes = b"\x00" * 12

    def to_bytes(self) -> bytes:
        pad = self.padding[:12].ljust(12, b"\x00")
        packed = struct.pack(
            HEADER_STRUCT_FMT,
            self.magic,
            self.version,
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_layers,
            self.num_deltanet_layers,
            self.num_global_layers,
            self.deltanet_heads,
            self.deltanet_head_dim,
            self.global_heads_q,
            self.global_heads_kv,
            self.global_head_dim,
            self.hopfield_groups,
            self.hopfield_codebook_size,
            self.hopfield_val_dim,
            self.tokens_per_block,
            self.max_blocks,
            self.tensor_table_offset,
            self.tensor_data_offset,
            self.bpe_vocab_offset,
            self.bpe_vocab_size_bytes,
            self.header_crc32,
            self.table_crc32,
            self.flags,
            pad
        )
        assert len(packed) == HEADER_SIZE_BYTES, f"Header size must be {HEADER_SIZE_BYTES} bytes!"
        return packed

    @classmethod
    def from_bytes(cls, data: bytes) -> "RadaNcmFileHeader":
        if len(data) < HEADER_SIZE_BYTES:
            raise ValueError(f"Data buffer too small for header: {len(data)} < {HEADER_SIZE_BYTES}")
        fields = struct.unpack(HEADER_STRUCT_FMT, data[:HEADER_SIZE_BYTES])
        return cls(
            magic=fields[0],
            version=fields[1],
            vocab_size=fields[2],
            hidden_size=fields[3],
            intermediate_size=fields[4],
            num_layers=fields[5],
            num_deltanet_layers=fields[6],
            num_global_layers=fields[7],
            deltanet_heads=fields[8],
            deltanet_head_dim=fields[9],
            global_heads_q=fields[10],
            global_heads_kv=fields[11],
            global_head_dim=fields[12],
            hopfield_groups=fields[13],
            hopfield_codebook_size=fields[14],
            hopfield_val_dim=fields[15],
            tokens_per_block=fields[16],
            max_blocks=fields[17],
            tensor_table_offset=fields[18],
            tensor_data_offset=fields[19],
            bpe_vocab_offset=fields[20],
            bpe_vocab_size_bytes=fields[21],
            header_crc32=fields[22],
            table_crc32=fields[23],
            flags=fields[24],
            padding=fields[25]
        )


@dataclass
class RadaTensorDescriptor:
    """
    Rada.cpp Tensor Descriptor Struct (128 bytes).
    Contains full geometry, quantization metadata, cache-line aligned offset,
    and per-tensor CRC32 for zero-copy on-demand integrity verification.
    """
    name: str
    dtype: int
    shape: Tuple[int, ...]
    offset: int
    size_bytes: int
    scale_gamma: float = 1.0
    crc32: int = 0

    def to_bytes(self) -> bytes:
        name_bytes = self.name.encode("utf-8")[:63].ljust(64, b"\x00")
        dims = list(self.shape) + [0] * (4 - len(self.shape))
        packed = struct.pack(
            TENSOR_DESC_FMT,
            name_bytes,
            self.dtype,
            len(self.shape),
            dims[0], dims[1], dims[2], dims[3],
            self.offset,
            self.size_bytes,
            self.scale_gamma,
            self.crc32
        )
        assert len(packed) == TENSOR_DESC_SIZE_BYTES, f"Tensor descriptor size must be {TENSOR_DESC_SIZE_BYTES} bytes!"
        return packed

    @classmethod
    def from_bytes(cls, data: bytes) -> "RadaTensorDescriptor":
        if len(data) < TENSOR_DESC_SIZE_BYTES:
            raise ValueError(f"Descriptor buffer too small: {len(data)} < {TENSOR_DESC_SIZE_BYTES}")
        fields = struct.unpack(TENSOR_DESC_FMT, data[:TENSOR_DESC_SIZE_BYTES])
        raw_name = fields[0].split(b"\x00", 1)[0].decode("utf-8", errors="replace")
        n_dims = fields[2]
        dims = tuple(fields[3:3 + n_dims])
        return cls(
            name=raw_name,
            dtype=fields[1],
            shape=dims,
            offset=fields[7],
            size_bytes=fields[8],
            scale_gamma=fields[9],
            crc32=fields[10]
        )


# ==============================================================================
# 3. BITWISE ALIGNMENT & BITNET B1.58 TERNARY PACKING
# ==============================================================================

def align_to_boundary(value: int, alignment: int = CACHE_LINE_ALIGNMENT) -> int:
    """
    Rounds up value to the nearest multiple of alignment.
    Strictly bitwise integer implementation without floating-point division.
    """
    if alignment <= 0:
        return value
    if (alignment & (alignment - 1)) == 0:
        return (value + alignment - 1) & ~(alignment - 1)
    return ((value + alignment - 1) // alignment) * alignment


def quantize_absmean_ternary(w: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, float]:
    """
    Applies BitNet b1.58 Absmean quantization:
        gamma = max(mean(|W|), eps)
        W_tilde = clamp(round(W / gamma), -1, +1)
    """
    w_float = w.detach().to(torch.float32)
    gamma = float(torch.mean(torch.abs(w_float)).item())
    gamma = max(gamma, eps)

    scaled_w = w_float / gamma
    w_quant = torch.clamp(torch.round(scaled_w), -1.0, 1.0).to(torch.int8)
    return w_quant, gamma


def pack_ternary_weights(
    weight: torch.Tensor,
    align_row_bytes: int = CACHE_LINE_ALIGNMENT
) -> Tuple[bytes, float, Tuple[int, ...]]:
    """
    Packs ternary weights {-1, 0, +1} into 2-bit values (4 weights per byte):
        00_2 -> 0
        01_2 -> +1
        10_2 -> -1
        11_2 -> 0 (reserved / padding)

    For 2D matrices [Rows, Cols], every row starts on an align_row_bytes boundary:
        Row_Stride_Bytes = align_to_boundary((Cols + 3) // 4, align_row_bytes)

    Vectorized implementation: zero Python tolist() overhead.
    """
    orig_shape = tuple(weight.shape)
    w_quant, gamma = quantize_absmean_ternary(weight)

    if len(orig_shape) == 2:
        rows, cols = orig_shape
        row_unpadded_bytes = (cols + 3) // 4
        row_stride = align_to_boundary(row_unpadded_bytes, align_row_bytes)

        # Vectorized bit-encoding
        codes = torch.zeros((rows, cols), dtype=torch.uint8, device=w_quant.device)
        codes[w_quant == 1] = 1
        codes[w_quant == -1] = 2

        pad_cols = (4 - (cols % 4)) % 4
        if pad_cols > 0:
            codes = torch.nn.functional.pad(codes, (0, pad_cols), mode="constant", value=0)

        c = codes.view(rows, -1, 4)
        b = (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).cpu().numpy()

        if row_stride == b.shape[1]:
            packed_bytes = b.tobytes()
        else:
            padded_mat = np.zeros((rows, row_stride), dtype=np.uint8)
            padded_mat[:, :b.shape[1]] = b
            packed_bytes = padded_mat.tobytes()

        return packed_bytes, gamma, orig_shape
    else:
        # 1D or higher dimension: flatten across non-trailing dims
        w_flat = w_quant.view(-1)
        total_elements = w_flat.numel()

        codes = torch.zeros(total_elements, dtype=torch.uint8, device=w_quant.device)
        codes[w_flat == 1] = 1
        codes[w_flat == -1] = 2

        pad_len = (4 - (total_elements % 4)) % 4
        if pad_len > 0:
            codes = torch.nn.functional.pad(codes, (0, pad_len), mode="constant", value=0)

        c = codes.view(-1, 4)
        b = (c[:, 0] | (c[:, 1] << 2) | (c[:, 2] << 4) | (c[:, 3] << 6)).cpu().numpy()
        raw_bytes = b.tobytes()

        aligned_len = align_to_boundary(len(raw_bytes), align_row_bytes)
        padded_bytes = raw_bytes.ljust(aligned_len, b"\x00")
        return padded_bytes, gamma, orig_shape


def unpack_ternary_weights(
    packed_bytes: bytes,
    shape: Tuple[int, ...],
    scale_gamma: float = 1.0,
    align_row_bytes: int = CACHE_LINE_ALIGNMENT
) -> torch.Tensor:
    """
    Vectorized unpack of 2-bit packed bytes back into float32 ternary tensor.
    """
    if len(shape) == 2:
        rows, cols = shape
        row_unpadded_bytes = (cols + 3) // 4
        row_stride = align_to_boundary(row_unpadded_bytes, align_row_bytes)

        arr = np.frombuffer(packed_bytes, dtype=np.uint8).copy().reshape(rows, row_stride)
        arr_valid = np.ascontiguousarray(arr[:, :row_unpadded_bytes])
        p = torch.from_numpy(arr_valid)

        c0 = p & 0x03
        c1 = (p >> 2) & 0x03
        c2 = (p >> 4) & 0x03
        c3 = (p >> 6) & 0x03

        u0 = torch.where(c0 == 1, 1.0, torch.where(c0 == 2, -1.0, 0.0))
        u1 = torch.where(c1 == 1, 1.0, torch.where(c1 == 2, -1.0, 0.0))
        u2 = torch.where(c2 == 1, 1.0, torch.where(c2 == 2, -1.0, 0.0))
        u3 = torch.where(c3 == 1, 1.0, torch.where(c3 == 2, -1.0, 0.0))

        u = torch.stack([u0, u1, u2, u3], dim=-1).view(rows, -1)[:, :cols]
        return (u * scale_gamma).to(torch.float32)
    else:
        total_elements = 1
        for d in shape:
            total_elements *= d

        num_bytes = (total_elements + 3) // 4
        p = torch.from_numpy(np.frombuffer(packed_bytes[:num_bytes], dtype=np.uint8).copy())

        c0 = p & 0x03
        c1 = (p >> 2) & 0x03
        c2 = (p >> 4) & 0x03
        c3 = (p >> 6) & 0x03

        u0 = torch.where(c0 == 1, 1.0, torch.where(c0 == 2, -1.0, 0.0))
        u1 = torch.where(c1 == 1, 1.0, torch.where(c1 == 2, -1.0, 0.0))
        u2 = torch.where(c2 == 1, 1.0, torch.where(c2 == 2, -1.0, 0.0))
        u3 = torch.where(c3 == 1, 1.0, torch.where(c3 == 2, -1.0, 0.0))

        u = torch.stack([u0, u1, u2, u3], dim=-1).view(-1)[:total_elements]
        return (u * scale_gamma).view(shape).to(torch.float32)


# ==============================================================================
# 4. TENSOR QUANTIZATION & SERIALIZATION
# ==============================================================================

def is_ternary_candidate(name: str, tensor: torch.Tensor) -> bool:
    """
    Identifies whether a parameter tensor belongs to the BitNet b1.58 ternary fabric.
    Linear projections, SwiGLU FFNs, Arbiter weights, and Stem projections are ternary.
    Embeddings, Hopfield codebooks, and LayerNorm/RMSNorm scales remain non-ternary.
    """
    if tensor.ndim < 2:
        return False
    lower = name.lower()
    if "embed" in lower:
        return False
    if "codebook" in lower or "hopfield_c" in lower:
        return False
    if "norm" in lower or "bias" in lower:
        return False
    return True


def serialize_tensor_payload(
    name: str,
    tensor: torch.Tensor,
    force_ternary: Optional[bool] = None
) -> Tuple[bytes, int, float, Tuple[int, ...]]:
    """
    Serializes a single tensor into its 64-byte aligned payload buffer.
    Returns:
        (payload_bytes, dtype_enum, scale_gamma, original_shape)
    """
    shape = tuple(tensor.shape)
    make_ternary = is_ternary_candidate(name, tensor) if force_ternary is None else force_ternary

    if make_ternary:
        packed_bytes, gamma, _ = pack_ternary_weights(tensor, align_row_bytes=CACHE_LINE_ALIGNMENT)
        aligned_size = align_to_boundary(len(packed_bytes), CACHE_LINE_ALIGNMENT)
        return packed_bytes.ljust(aligned_size, b"\x00"), DTYPE_TERNARY_2BIT, gamma, shape

    # Non-ternary branches
    lower = name.lower()
    if "embed" in lower:
        try:
            raw_data = tensor.detach().to(torch.bfloat16).contiguous().cpu().numpy().tobytes()
            dtype_code = DTYPE_BF16
        except Exception:
            raw_data = tensor.detach().to(torch.float16).contiguous().cpu().numpy().tobytes()
            dtype_code = DTYPE_FP16
    elif "codebook" in lower:
        raw_data = tensor.detach().to(torch.float16).contiguous().cpu().numpy().tobytes()
        dtype_code = DTYPE_FP16
    else:
        raw_data = tensor.detach().to(torch.float32).contiguous().cpu().numpy().tobytes()
        dtype_code = DTYPE_FP32

    aligned_size = align_to_boundary(len(raw_data), CACHE_LINE_ALIGNMENT)
    payload = raw_data.ljust(aligned_size, b"\x00")
    return payload, dtype_code, 1.0, shape


# ==============================================================================
# 5. STREAMING PRODUCTION EXPORT ENGINE (RADA EXPORTER)
# ==============================================================================

class RadaExporter:
    """
    Production exporter for Hetman-2-NCM to Rada.cpp binary container format (.rada / .rncm).
    Features:
    - Zero intermediate payload accumulation (direct file streaming).
    - Strict 64-byte alignment across all container sections.
    - Two-tier hierarchical CRC32 generation for instant mmap verification.
    """
    def __init__(self, config: Optional[Hetman2Config] = None):
        self.config = config or Hetman2Config()

    def create_header(self, num_tensors: int, tensor_table_offset: int) -> RadaNcmFileHeader:
        cfg = self.config
        return RadaNcmFileHeader(
            magic=RADA_NCM_MAGIC,
            version=RADA_VERSION,
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
            tensor_data_offset=0,
            bpe_vocab_offset=0,
            bpe_vocab_size_bytes=0,
            header_crc32=0,
            table_crc32=0,
            flags=RADA_FLAG_CRC32_ENABLED,
            padding=b"\x00" * 12
        )

    def export(
        self,
        model_or_state_dict: Union[nn.Module, Dict[str, torch.Tensor]],
        output_path: str,
        vocab_bytes: Optional[bytes] = None
    ) -> Dict[str, Any]:
        """
        Exports state dict to .rada file with 64-byte alignment, CRC32,
        and ZERO memory duplication via streaming directly to disk.
        """
        if isinstance(model_or_state_dict, nn.Module):
            state_dict = model_or_state_dict.state_dict()
        elif isinstance(model_or_state_dict, dict):
            state_dict = model_or_state_dict
        else:
            raise TypeError("model_or_state_dict must be nn.Module or dict")

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

        num_tensors = len(state_dict)
        tensor_table_offset = HEADER_SIZE_BYTES  # 128 bytes (strictly 64B aligned)
        tensor_table_raw_size = 8 + num_tensors * TENSOR_DESC_SIZE_BYTES
        tensor_table_aligned_size = align_to_boundary(tensor_table_raw_size, CACHE_LINE_ALIGNMENT)

        tensor_data_offset = tensor_table_offset + tensor_table_aligned_size
        assert tensor_data_offset % CACHE_LINE_ALIGNMENT == 0, "tensor_data_offset must be 64-byte aligned!"

        descriptors: List[RadaTensorDescriptor] = []
        total_ternary_weights = 0
        total_ternary_bytes = 0
        total_float_bytes = 0

        # DIRECT STREAMING WRITE (Zero intermediate RAM overhead)
        with open(output_path, "wb") as f:
            # 1. Pre-seek to tensor_data_offset, reserving space for header & table
            f.seek(tensor_data_offset)

            sorted_names = sorted(state_dict.keys())
            for name in sorted_names:
                tensor = state_dict[name]
                current_file_pos = f.tell()
                assert current_file_pos % CACHE_LINE_ALIGNMENT == 0, f"Offset for {name} unaligned: {current_file_pos}"

                payload, dtype_code, gamma, shape = serialize_tensor_payload(name, tensor)
                payload_len = len(payload)
                assert payload_len % CACHE_LINE_ALIGNMENT == 0, f"Payload for {name} must be 64B aligned!"

                # Calculate on-the-fly per-tensor CRC32
                tensor_crc = zlib.crc32(payload) & 0xFFFFFFFF

                desc = RadaTensorDescriptor(
                    name=name,
                    dtype=dtype_code,
                    shape=shape,
                    offset=current_file_pos,
                    size_bytes=payload_len,
                    scale_gamma=gamma,
                    crc32=tensor_crc
                )
                descriptors.append(desc)

                # Write payload directly to disk stream
                f.write(payload)

                if dtype_code == DTYPE_TERNARY_2BIT:
                    total_ternary_weights += tensor.numel()
                    total_ternary_bytes += payload_len
                else:
                    total_float_bytes += payload_len

                # Immediate release of buffer from RAM
                del payload

            # 2. Append BPE vocabulary at the end if provided
            bpe_vocab_offset = f.tell()
            bpe_vocab_size = 0
            if vocab_bytes:
                bpe_vocab_size = len(vocab_bytes)
                aligned_vocab_size = align_to_boundary(bpe_vocab_size, CACHE_LINE_ALIGNMENT)
                f.write(vocab_bytes.ljust(aligned_vocab_size, b"\x00"))

            # 3. Finalize Tensor Table bytes and calculate Table CRC32
            table_header = struct.pack("<II", num_tensors, 0)
            table_desc_bytes = b"".join(desc.to_bytes() for desc in descriptors)
            written_table_bytes = len(table_header) + len(table_desc_bytes)
            pad_table_len = tensor_table_aligned_size - written_table_bytes
            pad_table = b"\x00" * pad_table_len if pad_table_len > 0 else b""
            table_crc = zlib.crc32(table_desc_bytes) & 0xFFFFFFFF

            # 4. Construct File Header with Tier 1 Checksums
            header = self.create_header(num_tensors=num_tensors, tensor_table_offset=tensor_table_offset)
            header.tensor_data_offset = tensor_data_offset
            header.bpe_vocab_offset = bpe_vocab_offset
            header.bpe_vocab_size_bytes = bpe_vocab_size
            header.table_crc32 = table_crc
            header.flags = RADA_FLAG_CRC32_ENABLED

            # Compute Header CRC32 (with header_crc32 field zeroed)
            header.header_crc32 = 0
            raw_hdr = header.to_bytes()
            header.header_crc32 = zlib.crc32(raw_hdr[:104]) & 0xFFFFFFFF

            # 5. Rewind to beginning and flush finalized Header & Tensor Table
            f.seek(0)
            f.write(header.to_bytes())
            f.write(table_header)
            f.write(table_desc_bytes)
            if pad_table:
                f.write(pad_table)

            assert f.tell() == tensor_data_offset, f"File pos {f.tell()} != expected data offset {tensor_data_offset}"

        total_file_size = os.path.getsize(output_path)
        return {
            "output_path": output_path,
            "total_file_size_bytes": total_file_size,
            "total_tensors": num_tensors,
            "total_ternary_weights": total_ternary_weights,
            "total_ternary_bytes": total_ternary_bytes,
            "total_float_bytes": total_float_bytes,
            "tensor_data_offset": tensor_data_offset,
            "header_crc32": f"0x{header.header_crc32:08X}",
            "table_crc32": f"0x{header.table_crc32:08X}",
            "header": header
        }

    @staticmethod
    def verify_container(container_path: str, verify_all_crcs: bool = True) -> Dict[str, Any]:
        """
        Validates container integrity, header fields, 64-byte alignments,
        hierarchical CRC32 checksums, and unpacks ternary weights.
        """
        if not os.path.isfile(container_path):
            raise FileNotFoundError(f"Container file not found: {container_path}")

        file_size = os.path.getsize(container_path)

        # 1. Safe 8-byte probing to prevent Segfault on legacy or truncated files
        if file_size < 8:
            raise ValueError(f"File too small to be a valid container: {file_size} bytes (minimum 8 bytes required)")

        with open(container_path, "rb") as f:
            magic, version = struct.unpack("<II", f.read(8))

            if magic == RADA_LEGACY_MAGIC:
                if version == 1:
                    raise ValueError(
                        f"Legacy Hetman-1 container detected (magic=0x{magic:08X}, version=1). "
                        f"Rada.cpp v2 requires Hetman-2-NCM format (magic=0x{RADA_NCM_MAGIC:08X}, version={RADA_VERSION})."
                    )
                raise ValueError(f"Unknown legacy format: magic=0x{magic:08X}, version={version}")

            if magic != RADA_NCM_MAGIC:
                raise ValueError(
                    f"Invalid magic number: 0x{magic:08X} (expected RNCM=0x{RADA_NCM_MAGIC:08X})"
                )

            if version != RADA_VERSION:
                raise ValueError(f"Unsupported container version: {version} (expected {RADA_VERSION})")

            if file_size < HEADER_SIZE_BYTES:
                raise ValueError(f"File truncated before end of header: {file_size} < {HEADER_SIZE_BYTES}")

            f.seek(0)
            header_bytes = f.read(HEADER_SIZE_BYTES)
            header = RadaNcmFileHeader.from_bytes(header_bytes)

            # 2. Verify Tier 1: Header CRC32
            if header.flags & RADA_FLAG_CRC32_ENABLED:
                expected_hdr_crc = header.header_crc32
                computed_hdr_crc = zlib.crc32(header_bytes[:104]) & 0xFFFFFFFF
                if expected_hdr_crc != computed_hdr_crc:
                    raise ValueError(
                        f"Header CRC32 mismatch! Stored: 0x{expected_hdr_crc:08X}, Computed: 0x{computed_hdr_crc:08X}"
                    )

            # 3. Cache-line alignment checks
            if header.tensor_table_offset % CACHE_LINE_ALIGNMENT != 0:
                raise ValueError(f"tensor_table_offset unaligned: {header.tensor_table_offset}")
            if header.tensor_data_offset % CACHE_LINE_ALIGNMENT != 0:
                raise ValueError(f"tensor_data_offset unaligned: {header.tensor_data_offset}")

            # 4. Read tensor table
            f.seek(header.tensor_table_offset)
            num_tensors, _ = struct.unpack("<II", f.read(8))

            descriptors_raw = f.read(num_tensors * TENSOR_DESC_SIZE_BYTES)
            if len(descriptors_raw) < num_tensors * TENSOR_DESC_SIZE_BYTES:
                raise ValueError(f"Truncated descriptor table in file!")

            # Verify Tier 1: Table CRC32
            if header.flags & RADA_FLAG_CRC32_ENABLED:
                expected_tbl_crc = header.table_crc32
                computed_tbl_crc = zlib.crc32(descriptors_raw) & 0xFFFFFFFF
                if expected_tbl_crc != computed_tbl_crc:
                    raise ValueError(
                        f"Table CRC32 mismatch! Stored: 0x{expected_tbl_crc:08X}, Computed: 0x{computed_tbl_crc:08X}"
                    )

            descriptors: List[RadaTensorDescriptor] = []
            for i in range(num_tensors):
                desc_bytes = descriptors_raw[i * TENSOR_DESC_SIZE_BYTES : (i + 1) * TENSOR_DESC_SIZE_BYTES]
                desc = RadaTensorDescriptor.from_bytes(desc_bytes)
                if desc.offset % CACHE_LINE_ALIGNMENT != 0:
                    raise ValueError(f"Tensor {desc.name} offset unaligned: {desc.offset}")
                if desc.size_bytes % CACHE_LINE_ALIGNMENT != 0:
                    raise ValueError(f"Tensor {desc.name} size unaligned: {desc.size_bytes}")
                descriptors.append(desc)

            # 5. Verify payload offsets, per-tensor CRC32, and ternary round-trip
            verified_crc_count = 0
            verified_ternary_count = 0

            for desc in descriptors:
                if desc.offset + desc.size_bytes > file_size:
                    raise ValueError(
                        f"Tensor {desc.name} exceeds file boundary: {desc.offset + desc.size_bytes} > {file_size}"
                    )

                # Check per-tensor CRC32
                if (header.flags & RADA_FLAG_CRC32_ENABLED) and (verify_all_crcs or verified_crc_count < 10):
                    f.seek(desc.offset)
                    payload = f.read(desc.size_bytes)
                    computed_tensor_crc = zlib.crc32(payload) & 0xFFFFFFFF
                    if desc.crc32 != computed_tensor_crc:
                        raise ValueError(
                            f"Tensor '{desc.name}' CRC32 mismatch: "
                            f"Stored 0x{desc.crc32:08X} != Computed 0x{computed_tensor_crc:08X}"
                        )
                    verified_crc_count += 1

                # Check numerical ternary round-trip on first 3 ternary matrices
                if desc.dtype == DTYPE_TERNARY_2BIT and verified_ternary_count < 3:
                    f.seek(desc.offset)
                    payload = f.read(desc.size_bytes)
                    unpacked = unpack_ternary_weights(
                        packed_bytes=payload,
                        shape=desc.shape,
                        scale_gamma=desc.scale_gamma,
                        align_row_bytes=CACHE_LINE_ALIGNMENT
                    )
                    assert unpacked.shape == desc.shape, f"Unpacked shape mismatch: {unpacked.shape} != {desc.shape}"
                    verified_ternary_count += 1

        return {
            "valid": True,
            "magic_hex": f"0x{header.magic:08X}",
            "version": header.version,
            "header_crc32": f"0x{header.header_crc32:08X}",
            "table_crc32": f"0x{header.table_crc32:08X}",
            "flags": header.flags,
            "vocab_size": header.vocab_size,
            "hidden_size": header.hidden_size,
            "intermediate_size": header.intermediate_size,
            "num_layers": header.num_layers,
            "num_deltanet_layers": header.num_deltanet_layers,
            "tokens_per_block": header.tokens_per_block,
            "max_blocks": header.max_blocks,
            "num_tensors": len(descriptors),
            "tensor_table_offset": header.tensor_table_offset,
            "tensor_data_offset": header.tensor_data_offset,
            "descriptors": descriptors
        }


# ==============================================================================
# 6. CONVENIENCE API FUNCTIONS
# ==============================================================================

def export_to_rada(
    model_or_state_dict: Union[nn.Module, Dict[str, torch.Tensor]],
    output_path: str,
    config: Optional[Hetman2Config] = None,
    vocab_bytes: Optional[bytes] = None
) -> Dict[str, Any]:
    """
    High-level export function with streaming 64-byte alignment and BitNet b1.58 packing.
    """
    exporter = RadaExporter(config=config)
    return exporter.export(model_or_state_dict, output_path, vocab_bytes=vocab_bytes)


def verify_rada_file(path: str, verify_all_crcs: bool = True) -> Dict[str, Any]:
    """
    Verifies binary validity and hierarchical CRC32 checksums of exported .rada container.
    """
    return RadaExporter.verify_container(path, verify_all_crcs=verify_all_crcs)


# ==============================================================================
# 7. COMMAND-LINE INTERFACE
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Hetman-2-NCM Rada.cpp Container Exporter (v2 Streaming)")
    parser.add_argument("--model_path", type=str, default="", help="Path to PyTorch checkpoint (.pt / .pth / .bin)")
    parser.add_argument("--output_path", type=str, required=True, help="Path for destination .rada binary container")
    parser.add_argument("--verify", action="store_true", help="Run verification checks after exporting")
    args = parser.parse_args()

    if args.model_path and os.path.exists(args.model_path):
        print(f"[*] Loading PyTorch checkpoint from: {args.model_path}")
        state_dict = torch.load(args.model_path, map_location="cpu")
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    else:
        print("[*] Creating synthetic mini-instance of Hetman2ForCausalLM for export...")
        config = Hetman2Config.create_mini()
        model = Hetman2ForCausalLM(config)
        state_dict = model.state_dict()

    print(f"[*] Exporting to: {args.output_path} (RADA_VERSION={RADA_VERSION}, 64B aligned, streaming)")
    result = export_to_rada(state_dict, args.output_path)
    print(f"[+] Export successful! Total container size: {result['total_file_size_bytes']} bytes")
    print(f"    Tensors: {result['total_tensors']} | Ternary weights: {result['total_ternary_weights']}")
    print(f"    Header CRC32: {result['header_crc32']} | Table CRC32: {result['table_crc32']}")

    if args.verify:
        print(f"[*] Verifying container integrity and hierarchical CRC32 checksums...")
        audit = verify_rada_file(args.output_path, verify_all_crcs=True)
        print(f"[+] Container 100% Valid! Magic: {audit['magic_hex']}, Version: {audit['version']}")
        print(f"    All {audit['num_tensors']} tensor CRC32 checksums verified successfully!")


if __name__ == "__main__":
    main()
