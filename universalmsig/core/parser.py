"""
universalmsig/core/parser.py

Reads a HuggingFace config.json (live or offline) and constructs
a full ModelSignature with per-layer descriptors, tier assignments,
precision, weight estimates, and KV-cache sizing.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Optional

from .signature import (
    ModelSignature, LayerSignature,
    ExecutionTier, Precision,
)

# ── Built-in offline specs (no HF download needed) ───────────────────────────
OFFLINE_SPECS: dict[str, dict] = {
    "Qwen/Qwen2.5-0.5B": {
        "num_hidden_layers": 24, "hidden_size": 896,
        "num_attention_heads": 14, "num_key_value_heads": 2,
        "vocab_size": 151936, "max_position_embeddings": 131072,
        "intermediate_size": 4864,
        "model_type": "qwen2", "total_params": 494_032_896,
    },
    "Qwen/Qwen2.5-1.5B": {
        "num_hidden_layers": 28, "hidden_size": 1536,
        "num_attention_heads": 12, "num_key_value_heads": 2,
        "vocab_size": 151936, "max_position_embeddings": 131072,
        "intermediate_size": 8960,
        "model_type": "qwen2", "total_params": 1_543_714_816,
    },
    "meta-llama/Llama-3.2-1B": {
        "num_hidden_layers": 16, "hidden_size": 2048,
        "num_attention_heads": 32, "num_key_value_heads": 8,
        "vocab_size": 128256, "max_position_embeddings": 131072,
        "intermediate_size": 8192,
        "model_type": "llama", "total_params": 1_235_814_400,
    },
    "meta-llama/Llama-3.2-3B": {
        "num_hidden_layers": 28, "hidden_size": 3072,
        "num_attention_heads": 24, "num_key_value_heads": 8,
        "vocab_size": 128256, "max_position_embeddings": 131072,
        "intermediate_size": 8192,
        "model_type": "llama", "total_params": 3_212_749_824,
    },
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": {
        "num_hidden_layers": 28, "hidden_size": 1536,
        "num_attention_heads": 12, "num_key_value_heads": 2,
        "vocab_size": 151936, "max_position_embeddings": 131072,
        "intermediate_size": 8960,
        "model_type": "qwen2", "total_params": 1_781_088_256,
    },
    "microsoft/phi-2": {
        "num_hidden_layers": 32, "hidden_size": 2560,
        "num_attention_heads": 32, "num_key_value_heads": 32,
        "vocab_size": 51200, "max_position_embeddings": 2048,
        "intermediate_size": 10240, "mlp_matrices": 2,
        "model_type": "phi", "total_params": 2_779_683_840,
    },
    "google/gemma-2b": {
        "num_hidden_layers": 18, "hidden_size": 2048,
        "num_attention_heads": 8, "num_key_value_heads": 1,
        "vocab_size": 256000, "max_position_embeddings": 8192,
        "intermediate_size": 16384,
        "model_type": "gemma", "total_params": 2_506_172_416,
    },
}


def _bytes_per_element(precision: Precision) -> float:
    """Storage bytes per weight element. 4-bit types pack two elements per byte."""
    return {
        Precision.FP32: 4.0, Precision.FP16: 2.0, Precision.BF16: 2.0,
        Precision.INT8: 1.0, Precision.INT4: 0.5, Precision.FP4: 0.5,
    }[precision]


def _estimate_layer_weight_bytes(
    hidden_size: int,
    num_heads: int,
    num_kv_heads: int,
    precision: Precision,
    is_attention: bool,
    is_mlp: bool,
    intermediate_size: int = 0,
    mlp_matrices: int = 3,
) -> int:
    bpe = _bytes_per_element(precision)
    if is_attention:
        head_dim = hidden_size // num_heads
        q_bytes  = hidden_size * num_heads * head_dim * bpe
        k_bytes  = hidden_size * num_kv_heads * head_dim * bpe
        v_bytes  = k_bytes
        o_bytes  = num_heads * head_dim * hidden_size * bpe
        return int(q_bytes + k_bytes + v_bytes + o_bytes)
    elif is_mlp:
        # Gated MLPs (SwiGLU: gate + up + down) have 3 hidden x intermediate
        # matrices; classic 2-layer MLPs (e.g. phi-2 fc1/fc2) have 2.
        intermediate = intermediate_size or hidden_size * 4
        return hidden_size * intermediate * mlp_matrices * bpe
    else:
        return int(hidden_size * hidden_size * bpe)


def _estimate_kv_cache_bytes(
    num_kv_heads: int,
    hidden_size: int,
    num_heads: int,
    max_seq_len: int,
    precision: Precision,
) -> int:
    head_dim = hidden_size // max(num_heads, 1)
    bpe = _bytes_per_element(precision)
    return int(2 * num_kv_heads * head_dim * max_seq_len * bpe)  # K + V


def _fetch_hf_config(model_id: str) -> Optional[dict]:
    try:
        from huggingface_hub import hf_hub_download  # type: ignore
        path = hf_hub_download(repo_id=model_id, filename="config.json")
        return json.loads(Path(path).read_text())
    except Exception:
        return None


def build_signature(
    model_id: str,
    precision: Precision = Precision.FP16,
    npu_split_ratio: float = 0.70,
    max_seq_len: int = 4096,
    offline: bool = False,
) -> ModelSignature:
    """
    Build a full ModelSignature for any supported model.

    Args:
        model_id:        HuggingFace repo ID or offline key
        precision:       weight/activation precision
        npu_split_ratio: fraction of layers on fast execution tier
        max_seq_len:     sequence length for KV-cache estimation
        offline:         skip HF download, use built-in specs only
    """
    cfg: dict = {}

    if not offline:
        cfg = _fetch_hf_config(model_id) or {}

    if not cfg:
        if model_id in OFFLINE_SPECS:
            cfg = OFFLINE_SPECS[model_id]
        else:
            # No config could be resolved for this model. Silently substituting a
            # template would produce a signature for the wrong architecture while
            # claiming this model_id, so fail loudly instead.
            hint = (
                "no config fetched from HuggingFace (is huggingface_hub installed? "
                "is the repo id correct?)"
                if not offline else
                "offline mode is on, so no download was attempted (CLI: pass --online)"
            )
            raise ValueError(
                f"Unknown model {model_id!r}: not in the built-in offline registry "
                f"and {hint}. Built-in models: {', '.join(OFFLINE_SPECS)}"
            )

    num_layers   = int(cfg.get("num_hidden_layers", 24))
    hidden_size  = int(cfg.get("hidden_size", 896))
    num_heads    = int(cfg.get("num_attention_heads", 14))
    num_kv_heads = int(cfg.get("num_key_value_heads", num_heads))
    vocab_size   = int(cfg.get("vocab_size", 32000))
    model_type   = cfg.get("model_type", "unknown")
    total_params = int(cfg.get("total_params", 0))
    intermediate = int(cfg.get("intermediate_size", hidden_size * 4))
    mlp_matrices = int(cfg.get("mlp_matrices", 3))
    cap_seq      = min(max_seq_len, int(cfg.get("max_position_embeddings", 131072)))

    npu_boundary = int(math.ceil(num_layers * npu_split_ratio))

    layers: list[LayerSignature] = []

    # Embedding layer
    emb_bytes = int(vocab_size * hidden_size * _bytes_per_element(precision))
    layers.append(LayerSignature(
        index        = 0,
        name         = "model.embed_tokens",
        tier         = ExecutionTier.GPU_FAST,
        precision    = precision,
        weight_bytes = emb_bytes,
        hidden_size  = hidden_size,
        num_heads    = num_heads,
        num_kv_heads = num_kv_heads,
        is_embedding = True,
        notes        = "Token embedding table",
    ))

    # Transformer layers (attention + MLP pairs)
    for i in range(num_layers):
        tier = (ExecutionTier.GPU_FAST
                if i < npu_boundary else ExecutionTier.CPU_FALLBACK)

        attn_bytes = _estimate_layer_weight_bytes(
            hidden_size, num_heads, num_kv_heads, precision,
            is_attention=True, is_mlp=False,
        )
        mlp_bytes = _estimate_layer_weight_bytes(
            hidden_size, num_heads, num_kv_heads, precision,
            is_attention=False, is_mlp=True,
            intermediate_size=intermediate, mlp_matrices=mlp_matrices,
        )
        kv_bytes = _estimate_kv_cache_bytes(
            num_kv_heads, hidden_size, num_heads, cap_seq, precision,
        )

        # Attention sub-layer
        layers.append(LayerSignature(
            index          = len(layers),
            name           = f"model.layers.{i}.self_attn",
            tier           = tier,
            precision      = precision,
            weight_bytes   = attn_bytes,
            hidden_size    = hidden_size,
            num_heads      = num_heads,
            num_kv_heads   = num_kv_heads,
            is_attention   = True,
            kv_cache_bytes = kv_bytes,
            notes          = f"GQA heads={num_heads} kv={num_kv_heads}" if num_kv_heads < num_heads else "",
        ))

        # MLP sub-layer
        layers.append(LayerSignature(
            index        = len(layers),
            name         = f"model.layers.{i}.mlp",
            tier         = tier,
            precision    = precision,
            weight_bytes = mlp_bytes,
            hidden_size  = hidden_size,
            num_heads    = num_heads,
            num_kv_heads = num_kv_heads,
            is_mlp       = True,
        ))

    # LM head
    lm_bytes = int(hidden_size * vocab_size * _bytes_per_element(precision))
    layers.append(LayerSignature(
        index        = len(layers),
        name         = "lm_head",
        tier         = ExecutionTier.GPU_FAST,
        precision    = precision,
        weight_bytes = lm_bytes,
        hidden_size  = hidden_size,
        num_heads    = num_heads,
        num_kv_heads = num_kv_heads,
        notes        = "Language model head",
    ))

    sig = ModelSignature(
        model_id         = model_id,
        model_family     = model_type,
        architecture     = "transformer-decoder",
        total_layers     = num_layers,
        hidden_size      = hidden_size,
        intermediate_size = intermediate,
        num_heads        = num_heads,
        num_kv_heads     = num_kv_heads,
        vocab_size       = vocab_size,
        max_seq_len      = cap_seq,
        total_params     = total_params,
        default_precision = precision,
        npu_split_ratio  = npu_split_ratio,
        layers           = layers,
        source_url       = f"https://huggingface.co/{model_id}",
    )
    sig.compute_hash()
    return sig
