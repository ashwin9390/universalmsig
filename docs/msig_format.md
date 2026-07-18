# .msig Format Specification — v2.0.0

The `.msig` file is a JSON document that encodes a model's layer topology,
execution tier routing, memory layout, and quantization profile in a
vendor-neutral way. Every backend reads from this single source of truth.

---

## Why .msig and not ONNX?

ONNX describes the **compute graph** (operators, weights, connections).
`.msig` describes **where each layer should run and how it maps to hardware memory**:
- Which layers go to the GPU/NPU fast path vs CPU fallback
- How much SRAM/HBM each layer needs
- What quantization scheme each layer uses
- How the KV-cache is sized and placed

These are the decisions ONNX leaves to the vendor toolchain — and why
re-compiling the same model for NVIDIA, Apple, and Qualcomm today requires
three completely separate pipelines.

---

## Top-Level Fields

```json
{
  "msig_version":      "2.0.0",
  "model_id":          "Qwen/Qwen2.5-0.5B",
  "model_family":      "qwen2",
  "architecture":      "transformer-decoder",
  "total_layers":      24,
  "hidden_size":       896,
  "num_heads":         14,
  "num_kv_heads":      2,
  "vocab_size":        151936,
  "max_seq_len":       4096,
  "total_params":      494032896,
  "default_precision": "fp16",
  "npu_split_ratio":   0.70,
  "content_hash":      "sha256...",
  "source_url":        "https://huggingface.co/Qwen/Qwen2.5-0.5B",
  "notes":             "",
  "layers":            [ ... ]
}
```

| Field | Type | Description |
|---|---|---|
| `msig_version` | string | Format version (`"2.0.0"`) |
| `model_id` | string | HuggingFace repo ID or custom identifier |
| `model_family` | string | Architecture family: `qwen2`, `llama`, `phi`, `gemma` |
| `architecture` | string | High-level type: `transformer-decoder` |
| `total_layers` | int | Number of transformer block layers (excluding embed/lm_head) |
| `hidden_size` | int | Model hidden dimension |
| `num_heads` | int | Number of attention heads |
| `num_kv_heads` | int | Number of KV heads (< num_heads for GQA) |
| `vocab_size` | int | Vocabulary size |
| `max_seq_len` | int | Maximum sequence length for KV-cache sizing |
| `total_params` | int | Approximate parameter count |
| `default_precision` | string | `fp32`, `fp16`, `bf16`, `int8`, `int4`, `fp4` |
| `npu_split_ratio` | float | Fraction of layers on fast execution tier (0.0–1.0) |
| `content_hash` | string | SHA-256 of canonical JSON (excludes hash field itself) |
| `source_url` | string | Where the model config came from |
| `layers` | array | Per-layer descriptors (see below) |

---

## Layer Descriptor

Each element of `"layers"` describes one sub-layer:

```json
{
  "index":          1,
  "name":           "model.layers.0.self_attn",
  "tier":           "gpu_fast",
  "precision":      "fp16",
  "weight_bytes":   3211264,
  "hidden_size":    896,
  "num_heads":      14,
  "num_kv_heads":   2,
  "is_attention":   true,
  "is_mlp":         false,
  "is_embedding":   false,
  "kv_cache_bytes": 229376,
  "notes":          "GQA heads=14 kv=2"
}
```

| Field | Type | Description |
|---|---|---|
| `index` | int | Position in layer list (0-based) |
| `name` | string | Fully qualified layer name (matches HF model state dict) |
| `tier` | string | `gpu_fast`, `npu_fast`, `cpu_fallback`, `accelerator` |
| `precision` | string | Per-layer precision (inherits `default_precision` if not overridden) |
| `weight_bytes` | int | Estimated weight tensor size in bytes |
| `hidden_size` | int | Hidden dimension at this layer |
| `num_heads` | int | Attention heads at this layer |
| `num_kv_heads` | int | KV heads at this layer |
| `is_attention` | bool | True for self-attention sub-layers |
| `is_mlp` | bool | True for MLP/FFN sub-layers |
| `is_embedding` | bool | True for token embedding layer |
| `kv_cache_bytes` | int | KV-cache bytes at `max_seq_len` (0 for non-attention layers) |
| `notes` | string | Human-readable notes (GQA config, special handling, etc.) |

---

## Execution Tiers

| Tier value | Meaning | Typical hardware |
|---|---|---|
| `gpu_fast` | GPU HBM + compute cores | NVIDIA H100/A100, AMD MI300 |
| `npu_fast` | On-chip SRAM + NPU | Qualcomm Hexagon HTP, Apple ANE |
| `accelerator` | External accelerator | Apple Neural Engine, Edge TPU |
| `cpu_fallback` | System RAM + CPU | Any host CPU |

The `npu_split_ratio` field (default `0.70`) controls the boundary:
- Layers 0 → ceil(N × 0.70) → `gpu_fast` or `npu_fast`
- Layers ceil(N × 0.70) → N → `cpu_fallback`

---

## Precision Values

| Value | Bytes/param | Supported by |
|---|---|---|
| `fp32` | 4 | All backends |
| `fp16` | 2 | All backends |
| `bf16` | 2 | TensorRT (promoted to fp16 on CoreML/QNN) |
| `int8` | 1 | TensorRT (INT8 SQ), CoreML (int8), QNN (SFIXED_POINT_8) |
| `int4` | 0.5 | TensorRT (INT4 AWQ), CoreML (uint4 palettization), QNN (SFIXED_POINT_4) |
| `fp4` | 0.5 | TensorRT only (promoted to fp16 on CoreML/QNN) |

---

## Binary Format (.bin)

The compact 28-byte binary format is compatible with the C firmware emulator:

```
Offset  Size  Type     Field
0       4     char[4]  version ("2.00")
4       4     uint32   total_layers
8       4     uint32   hidden_size
12      4     uint32   num_heads
16      4     uint32   num_kv_heads
20      8     uint64   bytes_per_layer (first layer's weight_bytes)
```

Python pack: `struct.pack("=4sIIIIQ", b"2.00", layers, hidden, heads, kv, bpl)`

---

## Content Hash

The `content_hash` field is a SHA-256 of the canonical JSON with
the `content_hash` field itself excluded. This enables:

- Detecting if a signature file was tampered with
- Deduplicating identical model configs across vendors
- Pinning a specific compilation to a specific model version

```python
import hashlib, json
d = json.loads(msig_text)
d.pop("content_hash", None)
canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
hash_val = hashlib.sha256(canonical.encode()).hexdigest()
```

---

## Example — Minimal Valid .msig

```json
{
  "msig_version": "2.0.0",
  "model_id": "my-org/my-model",
  "model_family": "llama",
  "architecture": "transformer-decoder",
  "total_layers": 16,
  "hidden_size": 2048,
  "num_heads": 16,
  "num_kv_heads": 4,
  "vocab_size": 32000,
  "max_seq_len": 4096,
  "total_params": 1000000000,
  "default_precision": "fp16",
  "npu_split_ratio": 0.70,
  "content_hash": "",
  "source_url": "",
  "notes": "",
  "layers": [
    {
      "index": 0,
      "name": "model.embed_tokens",
      "tier": "gpu_fast",
      "precision": "fp16",
      "weight_bytes": 131072000,
      "hidden_size": 2048,
      "num_heads": 16,
      "num_kv_heads": 4,
      "is_attention": false,
      "is_mlp": false,
      "is_embedding": true,
      "kv_cache_bytes": 0,
      "notes": "Token embedding"
    }
  ]
}
```
