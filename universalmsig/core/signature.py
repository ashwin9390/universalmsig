"""
universalmsig/core/signature.py

The ModelSignature is the single source of truth — a vendor-neutral
description of an AI model's layer topology, memory layout, and
execution profile. Every backend translates FROM this, never to it.

This is the gap that ONNX doesn't fill: ONNX describes the compute graph.
ModelSignature describes WHERE each layer should run and HOW it should
be mapped to hardware memory.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Optional


# ── Execution tier ────────────────────────────────────────────────────────────
class ExecutionTier(str, Enum):
    """Which hardware tier a layer is assigned to."""
    NPU_FAST    = "npu_fast"     # On-chip SRAM / NPU compute cores
    GPU_FAST    = "gpu_fast"     # GPU HBM + compute (NVIDIA, AMD)
    CPU_FALLBACK = "cpu_fallback" # System RAM + CPU cores
    ACCELERATOR = "accelerator"  # Apple ANE, Qualcomm HTP, etc.


# ── Precision ─────────────────────────────────────────────────────────────────
class Precision(str, Enum):
    FP32  = "fp32"
    FP16  = "fp16"
    BF16  = "bf16"
    INT8  = "int8"
    INT4  = "int4"
    FP4   = "fp4"


# ── Per-layer descriptor ──────────────────────────────────────────────────────
@dataclass
class LayerSignature:
    index:          int
    name:           str
    tier:           ExecutionTier
    precision:      Precision
    weight_bytes:   int
    hidden_size:    int
    num_heads:      int
    num_kv_heads:   int
    is_attention:   bool = False
    is_mlp:         bool = False
    is_embedding:   bool = False
    kv_cache_bytes: int  = 0       # estimated KV-cache bytes at max_seq_len
    notes:          str  = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["tier"]      = self.tier.value
        d["precision"] = self.precision.value
        return d


# ── Top-level signature ───────────────────────────────────────────────────────
@dataclass
class ModelSignature:
    """
    Universal model execution signature (.msig v2).

    Vendor-neutral. Every backend (TensorRT, CoreML, QNN, ONNX, llama.cpp)
    reads this and produces its own optimised execution plan.
    """
    msig_version:    str   = "2.0.0"
    model_id:        str   = ""          # HF repo ID e.g. "Qwen/Qwen2.5-0.5B"
    model_family:    str   = ""          # "qwen2", "llama", "deepseek", ...
    architecture:    str   = ""          # "transformer-decoder"
    total_layers:    int   = 0
    hidden_size:     int   = 0
    num_heads:       int   = 0
    num_kv_heads:    int   = 0
    vocab_size:      int   = 0
    max_seq_len:     int   = 4096
    total_params:    int   = 0           # approx parameter count
    default_precision: Precision = Precision.FP16
    npu_split_ratio: float = 0.70        # fraction of layers on fast tier
    layers:          list[LayerSignature] = field(default_factory=list)
    content_hash:    str   = ""          # sha256 of canonical JSON
    source_url:      str   = ""
    notes:           str   = ""

    # ── Computed properties ───────────────────────────────────────────────────
    @property
    def npu_layers(self) -> list[LayerSignature]:
        """
        Transformer layers on the fast execution tier.
        Excludes embed_tokens and lm_head — these are infrastructure
        layers, not part of the 70/30 transformer split.
        """
        return [
            l for l in self.layers
            if l.tier in (ExecutionTier.NPU_FAST, ExecutionTier.GPU_FAST,
                          ExecutionTier.ACCELERATOR)
            and not l.is_embedding
            and l.name != "lm_head"
        ]

    @property
    def cpu_layers(self) -> list[LayerSignature]:
        """
        Transformer layers on the CPU fallback tier.
        Excludes embed_tokens and lm_head.
        """
        return [
            l for l in self.layers
            if l.tier == ExecutionTier.CPU_FALLBACK
            and not l.is_embedding
            and l.name != "lm_head"
        ]

    @property
    def transformer_layers(self) -> list[LayerSignature]:
        """All transformer sub-layers (attention + MLP), excluding embed and lm_head."""
        return [
            l for l in self.layers
            if not l.is_embedding and l.name != "lm_head"
        ]

    @property
    def total_weight_bytes(self) -> int:
        return sum(l.weight_bytes for l in self.layers)

    @property
    def total_kv_cache_bytes(self) -> int:
        return sum(l.kv_cache_bytes for l in self.layers)

    # ── Serialisation ─────────────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {
            "msig_version":      self.msig_version,
            "model_id":          self.model_id,
            "model_family":      self.model_family,
            "architecture":      self.architecture,
            "total_layers":      self.total_layers,
            "hidden_size":       self.hidden_size,
            "num_heads":         self.num_heads,
            "num_kv_heads":      self.num_kv_heads,
            "vocab_size":        self.vocab_size,
            "max_seq_len":       self.max_seq_len,
            "total_params":      self.total_params,
            "default_precision": self.default_precision.value,
            "npu_split_ratio":   self.npu_split_ratio,
            "content_hash":      self.content_hash,
            "source_url":        self.source_url,
            "notes":             self.notes,
            "layers":            [l.to_dict() for l in self.layers],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json())

    # 4-char binary tag for the 2.x header layout (msig_version "2.0.0")
    _BINARY_VERSION = b"2.00"
    _BINARY_STRUCT  = "=4sIIIIQ"

    def to_binary(self) -> bytes:
        """
        Compact binary header, 28 bytes:
          version(4s) layers(I) hidden(I) heads(I) kv_heads(I) block_bytes(Q)

        block_bytes is the weight size of ONE transformer block (first
        attention + first MLP sub-layer) — NOT layers[0], which is the token
        embedding table and was previously (and wrongly) written here.
        """
        attn = next((l.weight_bytes for l in self.layers if l.is_attention), 0)
        mlp  = next((l.weight_bytes for l in self.layers if l.is_mlp), 0)
        return struct.pack(
            self._BINARY_STRUCT,
            self._BINARY_VERSION,
            self.total_layers,
            self.hidden_size,
            self.num_heads,
            self.num_kv_heads,
            attn + mlp,
        )

    def save_binary(self, path: str | Path) -> None:
        Path(path).write_bytes(self.to_binary())

    @classmethod
    def from_binary(cls, data: bytes) -> "ModelSignature":
        """
        Parse a 28-byte binary header back into a skeleton signature.
        Only header-level fields are recoverable (no per-layer detail).
        """
        version, layers, hidden, heads, kv_heads, block_bytes = struct.unpack(
            cls._BINARY_STRUCT, data[:28]
        )
        if version != cls._BINARY_VERSION:
            raise ValueError(
                f"Unsupported .msig binary version {version!r} "
                f"(expected {cls._BINARY_VERSION!r})"
            )
        return cls(
            total_layers = layers,
            hidden_size  = hidden,
            num_heads    = heads,
            num_kv_heads = kv_heads,
            notes        = (f"loaded from binary header; "
                            f"block_bytes={block_bytes}, no per-layer detail"),
        )

    @classmethod
    def load_binary(cls, path: str | Path) -> "ModelSignature":
        return cls.from_binary(Path(path).read_bytes())

    @classmethod
    def from_dict(cls, d: dict) -> "ModelSignature":
        layers = [
            LayerSignature(
                index        = l["index"],
                name         = l["name"],
                tier         = ExecutionTier(l["tier"]),
                precision    = Precision(l["precision"]),
                weight_bytes = l["weight_bytes"],
                hidden_size  = l["hidden_size"],
                num_heads    = l["num_heads"],
                num_kv_heads = l["num_kv_heads"],
                is_attention = l.get("is_attention", False),
                is_mlp       = l.get("is_mlp", False),
                is_embedding = l.get("is_embedding", False),
                kv_cache_bytes = l.get("kv_cache_bytes", 0),
                notes        = l.get("notes", ""),
            )
            for l in d.get("layers", [])
        ]
        return cls(
            msig_version     = d.get("msig_version", "2.0.0"),
            model_id         = d.get("model_id", ""),
            model_family     = d.get("model_family", ""),
            architecture     = d.get("architecture", ""),
            total_layers     = d.get("total_layers", 0),
            hidden_size      = d.get("hidden_size", 0),
            num_heads        = d.get("num_heads", 0),
            num_kv_heads     = d.get("num_kv_heads", 0),
            vocab_size       = d.get("vocab_size", 0),
            max_seq_len      = d.get("max_seq_len", 4096),
            total_params     = d.get("total_params", 0),
            default_precision = Precision(d.get("default_precision", "fp16")),
            npu_split_ratio  = d.get("npu_split_ratio", 0.70),
            content_hash     = d.get("content_hash", ""),
            source_url       = d.get("source_url", ""),
            notes            = d.get("notes", ""),
            layers           = layers,
        )

    @classmethod
    def from_json(cls, text: str) -> "ModelSignature":
        return cls.from_dict(json.loads(text))

    @classmethod
    def load_json(cls, path: str | Path) -> "ModelSignature":
        return cls.from_json(Path(path).read_text())

    def compute_hash(self) -> str:
        """Stable content hash — excludes the hash field itself."""
        d = self.to_dict()
        d.pop("content_hash", None)
        canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
        self.content_hash = hashlib.sha256(canonical.encode()).hexdigest()
        return self.content_hash

    def summary(self) -> str:
        # Count transformer BLOCKS (not sub-layers) for user-facing display
        # Each block = 1 attention + 1 MLP sub-layer
        npu_blocks  = len([l for l in self.npu_layers if l.is_attention])
        cpu_blocks  = len([l for l in self.cpu_layers if l.is_attention])
        total_blocks = npu_blocks + cpu_blocks
        npu_pct     = round(npu_blocks / total_blocks * 100) if total_blocks else 0
        cpu_pct     = 100 - npu_pct

        gqa_ratio = self.num_heads // self.num_kv_heads if self.num_kv_heads > 0 else 1
        gqa_note  = (
            f"GQA — each KV head serves {gqa_ratio} Q heads "
            f"({self.num_kv_heads} KV → {self.num_heads} Q, "
            f"broadcast required in CoreML/QNN)"
            if self.num_kv_heads < self.num_heads else "MHA (no GQA)"
        )

        lines = [
            f"ModelSignature v{self.msig_version}",
            f"  Model         : {self.model_id or '(unknown)'}",
            f"  Family        : {self.model_family}",
            f"  Architecture  : {self.architecture}",
            f"  Transformer   : {self.total_layers} blocks "
            f"({self.total_layers} attn + {self.total_layers} mlp sub-layers)",
            f"  Total objects : {len(self.layers)} "
            f"(embed_tokens + {self.total_layers*2} sub-layers + lm_head)",
            f"  Hidden size   : {self.hidden_size}",
            f"  Attention     : {self.num_heads} heads  KV: {self.num_kv_heads} — {gqa_note}",
            f"  Vocab size    : {self.vocab_size:,}",
        ]
        if self.total_params:
            lines.append(f"  Parameters    : {self.total_params/1e9:.2f}B")
        lines += [
            f"  Precision     : {self.default_precision.value}",
            f"  Fast tier     : {npu_blocks} transformer blocks ({npu_pct}%)",
            f"  CPU tier      : {cpu_blocks} transformer blocks ({cpu_pct}%)",
            f"  Weight size   : {self.total_weight_bytes/1e9:.2f} GB",
            f"  KV-cache      : {self.total_kv_cache_bytes/1e6:.1f} MB (at {self.max_seq_len} tokens)",
            f"  Content hash  : {self.content_hash[:16]}..." if self.content_hash else "",
        ]
        return "\n".join(l for l in lines if l)
