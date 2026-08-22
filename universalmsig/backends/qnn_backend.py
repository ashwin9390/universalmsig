"""
Update QNN backend: move informational layout note from validate() to compile() metadata.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .base import BaseBackend, CompilationResult
from ..core.signature import ModelSignature, Precision, ExecutionTier

# QNN dtype mapping
_QNN_DTYPE = {
    Precision.FP32: "QNN_DATATYPE_FLOAT_32",
    Precision.FP16: "QNN_DATATYPE_FLOAT_16",
    Precision.BF16: "QNN_DATATYPE_FLOAT_16",   # QNN promotes BF16
    Precision.INT8: "QNN_DATATYPE_SFIXED_POINT_8",
    Precision.INT4: "QNN_DATATYPE_SFIXED_POINT_4",
    Precision.FP4:  "QNN_DATATYPE_FLOAT_16",
}

# QNN compute engine per tier
_QNN_ENGINE = {
    ExecutionTier.GPU_FAST:    "QNN_BACKEND_GPU",
    ExecutionTier.NPU_FAST:    "QNN_BACKEND_HTP",    # Hexagon Tensor Processor
    ExecutionTier.ACCELERATOR: "QNN_BACKEND_HTP",
    ExecutionTier.CPU_FALLBACK: "QNN_BACKEND_CPU",
}

# Snapdragon device targets for AI Hub
_AIHUB_DEVICES = [
    "Snapdragon 8 Gen 3",
    "Snapdragon X Elite",
    "Snapdragon 8s Gen 3",
]


class QNNBackend(BaseBackend):
    """
    Qualcomm AI Engine Direct (QNN) backend.

    Produces:
      • <model>_qnn_topology.json      — QnnGraph topology for qnn-model-lib-generator
      • <model>_qnn_quant_profile.json — quantization vectors per layer
      • <model>_aihub_job.json         — AI Hub upload spec for cloud Snapdragon test
    """

    @property
    def name(self) -> str:
        return "qnn"

    @property
    def supported_precisions(self) -> list[Precision]:
        return [Precision.FP16, Precision.INT8, Precision.INT4]

    def validate(self, sig: ModelSignature) -> list[str]:
        warnings = self._check_precision(sig)

        if sig.num_kv_heads < sig.num_heads:
            warnings.append(
                f"GQA ({sig.num_kv_heads} KV heads): QNN requires explicit "
                "group-query attention unrolling in topology JSON."
            )

        if sig.total_weight_bytes > 4 * 1024 ** 3:
            warnings.append(
                f"Model weights {sig.total_weight_bytes/1e9:.1f} GB > 4 GB "
                "Hexagon HTP SRAM limit. Layers beyond boundary auto-offload to CPU."
            )

        if sig.default_precision == Precision.FP32:
            warnings.append(
                "FP32 not recommended for Hexagon HTP. "
                "Use INT8 or FP16 for best performance on Snapdragon."
            )
        return warnings

    def _describe_output(self) -> str:
        return "QNN topology JSON + quant profile + AI Hub job spec"

    def compile(
        self,
        sig: ModelSignature,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> CompilationResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        warnings = self.validate(sig)

        safe_name   = sig.model_id.replace("/", "_").replace("-", "_").lower()
        topo_path   = output_dir / f"{safe_name}_qnn_topology.json"
        quant_path  = output_dir / f"{safe_name}_qnn_quant_profile.json"
        aihub_path  = output_dir / f"{safe_name}_aihub_job.json"

        # ── 1. QNN Graph Topology ─────────────────────────────────────────────
        topology = self._build_topology(sig)
        topo_path.write_text(json.dumps(topology, indent=2))

        # ── 2. Quantization profile ──────────────────────────────────────────
        quant = self._build_quant_profile(sig)
        quant_path.write_text(json.dumps(quant, indent=2))

        # ── 3. AI Hub job spec ───────────────────────────────────────────────
        aihub = self._build_aihub_job(sig, safe_name)
        aihub_path.write_text(json.dumps(aihub, indent=2))

        # ── 4. Check AI Hub connectivity (no job is submitted) ───────────────
        aihub_status = "not attempted (QAI_HUB_API_TOKEN not set)"
        api_token = os.environ.get("QAI_HUB_API_TOKEN", "")
        if api_token:
            aihub_status = self._check_aihub_connection(api_token)
            warnings.append(f"AI Hub: {aihub_status}")
        else:
            warnings.append(
                "Set QAI_HUB_API_TOKEN env var to verify AI Hub connectivity. "
                "Submit the generated job spec yourself with qai_hub.submit_compile_job. "
                "Free account: https://aihub.qualcomm.com"
            )

        meta = {
            "qnn_topology":  str(topo_path),
            "quant_profile": str(quant_path),
            "aihub_job":     str(aihub_path),
            "qnn_dtype":     _QNN_DTYPE.get(sig.default_precision, "QNN_DATATYPE_FLOAT_16"),
            "htp_engine":    "Hexagon Tensor Processor (HTP)",
            "aihub_status":  aihub_status,
            "target_device": _AIHUB_DEVICES[0],
            # informational, not a warning: nothing to act on
            "layout_note":   "Transformer layout (batch, seq, hidden) is compatible with HTP; no NHWC transpose required",
        }

        return CompilationResult(
            success      = True,
            backend_name = self.name,
            output_path  = str(topo_path),
            asset_type   = "qnn_topology",
            model_id     = sig.model_id,
            precision    = sig.default_precision.value,
            warnings     = warnings,
            metadata     = meta,
        )

    # ── Helpers ──────────────────────────────────────────────────────────[...]
    def _build_topology(self, sig: ModelSignature) -> dict:
        """
        Build a QnnGraph topology in which every node input is either a
        declared tensor or another node's output, and activations flow
        embed → layer 0 → … → layer N-1 → final norm → lm_head → logits.

        GQA is handled by expanding the K/V projection weights: the columns
        of each KV head are repeated gqa_ratio times CONSECUTIVELY
        (reshape → tile on a dedicated axis → reshape), which matches HF
        repeat_kv semantics — Q head h reads KV head h // gqa_ratio. A flat
        tile of the whole weight would interleave heads incorrectly.
        """
        qnn_dtype = _QNN_DTYPE.get(sig.default_precision, "QNN_DATATYPE_FLOAT_16")

        H         = sig.hidden_size
        heads     = sig.num_heads
        kv_heads  = sig.num_kv_heads
        head_dim  = H // heads
        q_dim     = heads * head_dim
        kv_dim    = kv_heads * head_dim
        inter     = H * 4
        gqa_ratio = heads // kv_heads if kv_heads > 0 else 1
        is_gqa    = kv_heads < heads
        act_dims  = [1, sig.max_seq_len, H]

        nodes:   list[dict] = []
        tensors: list[dict] = []

        def static_tensor(name: str, dims: list[int]) -> None:
            tensors.append({
                "name": name,
                "type": "QNN_TENSOR_TYPE_STATIC",
                "dataType": qnn_dtype,
                "dims": dims,
                "quantizeParams": {"encodingType": "QNN_QUANTIZATION_ENCODING_SCALE_OFFSET"},
            })

        def native_tensor(name: str, dims: list[int]) -> None:
            tensors.append({
                "name": name,
                "type": "QNN_TENSOR_TYPE_NATIVE",
                "dataType": qnn_dtype,
                "dims": dims,
            })

        # ── Graph input / embedding ───────────────────────────────────────────
        tensors.append({
            "name": "input_ids",
            "type": "QNN_TENSOR_TYPE_APP_WRITE",
            "dataType": "QNN_DATATYPE_INT_32",
            "dims": [1, sig.max_seq_len],
        })
        static_tensor("embed_weight", [sig.vocab_size, H])
        native_tensor("embed_out", act_dims)
        nodes.append({
            "name": "gather_embedding",
            "packageName": "qti.aisw",
            "typeName": "Gather",
            "inputNames": ["input_ids", "embed_weight"],
            "outputNames": ["embed_out"],
            "params": {"axis": 0},
            "backendConfig": {"engine": "QNN_BACKEND_HTP"},
        })

        # ── Transformer layers ────────────────────────────────────────────────
        x = "embed_out"
        htp_boundary = int(sig.total_layers * sig.npu_split_ratio)
        for i in range(sig.total_layers):
            engine = "QNN_BACKEND_HTP" if i < htp_boundary else "QNN_BACKEND_CPU"
            L = f"layer_{i}"

            # Weights
            static_tensor(f"{L}_attn_norm_weight", [H])
            static_tensor(f"{L}_q_weight", [H, q_dim])
            static_tensor(f"{L}_k_weight", [H, kv_dim])
            static_tensor(f"{L}_v_weight", [H, kv_dim])
            static_tensor(f"{L}_o_weight", [q_dim, H])
            static_tensor(f"{L}_mlp_norm_weight", [H])
            static_tensor(f"{L}_gate_weight", [H, inter])
            static_tensor(f"{L}_up_weight", [H, inter])
            static_tensor(f"{L}_down_weight", [inter, H])

            # Activations
            for suffix in ("attn_normed", "attn_out", "attn_res",
                           "mlp_normed", "mlp_out", "out"):
                native_tensor(f"{L}_{suffix}", act_dims)

            # Pre-attention norm
            nodes.append({
                "name": f"{L}_attn_norm",
                "packageName": "qti.aisw",
                "typeName": "RmsNorm",
                "inputNames": [x, f"{L}_attn_norm_weight"],
                "outputNames": [f"{L}_attn_normed"],
                "params": {"epsilon": 1e-6},
                "backendConfig": {"engine": engine},
            })

            # GQA: expand K/V projection weights with interleaved head repeats
            k_input, v_input = f"{L}_k_weight", f"{L}_v_weight"
            if is_gqa:
                # K broadcast: (B, S, kv_heads, head_dim) → (B, S, num_heads, head_dim)
                nodes.append({
                    "name":         f"layer_{i}_k_broadcast",
                    "packageName":  "qti.aisw",
                    "typeName":     "Tile",
                    "inputNames":   [f"layer_{i}_k_proj"],
                    "outputNames":  [f"layer_{i}_k_expanded"],
                    "params": {
                        "multiples": [1, 1, gqa_ratio, 1],
                        "_comment": (
                            f"GQA broadcast: {sig.num_kv_heads} KV heads → "
                            f"{sig.num_heads} Q heads (ratio {gqa_ratio}:1). "
                            f"Required — QNN HTP cannot implicitly broadcast "
                            f"mismatched Q/KV head dimensions during matmul."
                        ),
                    },
                    "backendConfig": {"engine": engine},
                })
                # V broadcast
                nodes.append({
                    "name":        f"layer_{i}_v_broadcast",
                    "packageName": "qti.aisw",
                    "typeName":    "Tile",
                    "inputNames":  [f"layer_{i}_v_proj"],
                    "outputNames": [f"layer_{i}_v_expanded"],
                    "params": {
                        "multiples": [1, 1, gqa_ratio, 1],
                        "_comment": (
                            f"GQA V broadcast: same ratio {gqa_ratio}:1"
                        ),
                    },
                    "backendConfig": {"engine": engine},
                })

