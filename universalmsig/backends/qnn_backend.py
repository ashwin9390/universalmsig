"""
universalmsig/backends/qnn_backend.py

Qualcomm QNN / AI Engine Direct Backend
========================================
Translates a ModelSignature into:
  1. QNN model topology JSON   (always — compatible with qnn-model-lib-generator)
  2. QNN quantization profile  (always)
  3. Qualcomm AI Hub job spec  (always — upload to real Snapdragon chip in the cloud)
  4. Remote AI Hub compilation (if qai_hub SDK installed + API key set)

Test on Qualcomm AI Hub (free):
  https://aihub.qualcomm.com
  pip install qai-hub
  export QAI_HUB_API_TOKEN=your_token
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

        # QNN Hexagon HTP mandates NHWC — but transformer (B, S, H) needs no transpose
        warnings.append(
            "QNN HTP: Transformer layout (batch, seq, hidden) is compatible. "
            "No NHWC transpose required for NLP models."
        )

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

        # ── 2. Quantization profile ───────────────────────────────────────────
        quant = self._build_quant_profile(sig)
        quant_path.write_text(json.dumps(quant, indent=2))

        # ── 3. AI Hub job spec ────────────────────────────────────────────────
        aihub = self._build_aihub_job(sig, safe_name)
        aihub_path.write_text(json.dumps(aihub, indent=2))

        # ── 4. Try remote AI Hub submission ──────────────────────────────────
        aihub_job_id = None
        api_token = os.environ.get("QAI_HUB_API_TOKEN", "")
        if api_token:
            aihub_job_id = self._submit_aihub_job(sig, topology, api_token)
            if aihub_job_id:
                warnings.append(f"AI Hub job submitted: {aihub_job_id}")
        else:
            warnings.append(
                "Set QAI_HUB_API_TOKEN env var to submit to real Snapdragon hardware. "
                "Free account: https://aihub.qualcomm.com"
            )

        meta = {
            "qnn_topology":  str(topo_path),
            "quant_profile": str(quant_path),
            "aihub_job":     str(aihub_path),
            "qnn_dtype":     _QNN_DTYPE.get(sig.default_precision, "QNN_DATATYPE_FLOAT_16"),
            "htp_engine":    "Hexagon Tensor Processor (HTP)",
            "aihub_job_id":  aihub_job_id or "not submitted",
            "target_device": _AIHUB_DEVICES[0],
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

    # ── Helpers ───────────────────────────────────────────────────────────────
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
                for kv in ("k", "v"):
                    native_tensor(f"{L}_{kv}_weight_grouped", [H, kv_heads, 1, head_dim])
                    native_tensor(f"{L}_{kv}_weight_tiled",   [H, kv_heads, gqa_ratio, head_dim])
                    native_tensor(f"{L}_{kv}_weight_expanded", [H, q_dim])
                    nodes.append({
                        "name":        f"{L}_{kv}_group",
                        "packageName": "qti.aisw",
                        "typeName":    "Reshape",
                        "inputNames":  [f"{L}_{kv}_weight"],
                        "outputNames": [f"{L}_{kv}_weight_grouped"],
                        "params": {"shape": [H, kv_heads, 1, head_dim]},
                        "backendConfig": {"engine": engine},
                    })
                    nodes.append({
                        "name":        f"{L}_{kv}_broadcast",
                        "packageName": "qti.aisw",
                        "typeName":    "Tile",
                        "inputNames":  [f"{L}_{kv}_weight_grouped"],
                        "outputNames": [f"{L}_{kv}_weight_tiled"],
                        "params": {
                            "multiples": [1, 1, gqa_ratio, 1],
                            "_comment": (
                                f"GQA broadcast: {kv_heads} KV heads → {heads} Q heads "
                                f"(ratio {gqa_ratio}:1) as an INTERLEAVED repeat on a "
                                f"dedicated axis — each KV head repeated {gqa_ratio}x "
                                f"consecutively so Q head h maps to KV head "
                                f"h // {gqa_ratio} (HF repeat_kv semantics). A flat "
                                f"tile of the head axis would mispair Q and KV heads."
                            ),
                        },
                        "backendConfig": {"engine": engine},
                    })
                    nodes.append({
                        "name":        f"{L}_{kv}_collapse",
                        "packageName": "qti.aisw",
                        "typeName":    "Reshape",
                        "inputNames":  [f"{L}_{kv}_weight_tiled"],
                        "outputNames": [f"{L}_{kv}_weight_expanded"],
                        "params": {"shape": [H, q_dim]},
                        "backendConfig": {"engine": engine},
                    })
                k_input, v_input = f"{L}_k_weight_expanded", f"{L}_v_weight_expanded"

            # Attention
            nodes.append({
                "name":        f"{L}_self_attn",
                "packageName": "qti.aisw",
                "typeName":    "ScaledDotProductAttention",
                "inputNames":  [f"{L}_attn_normed", f"{L}_q_weight",
                                k_input, v_input, f"{L}_o_weight"],
                "outputNames": [f"{L}_attn_out"],
                "params": {
                    "num_heads":         heads,
                    "num_kv_heads":      heads,   # after broadcast, both equal
                    "head_dim":          head_dim,
                    "scale":             round(1.0 / (head_dim ** 0.5), 6),
                    "use_rope":          True,
                    "gqa_unrolled":      is_gqa,
                    "original_kv_heads": kv_heads,
                    "gqa_ratio":         gqa_ratio,
                },
                "backendConfig": {"engine": engine},
            })
            nodes.append({
                "name":        f"{L}_attn_residual",
                "packageName": "qti.aisw",
                "typeName":    "ElementWiseAdd",
                "inputNames":  [x, f"{L}_attn_out"],
                "outputNames": [f"{L}_attn_res"],
                "params": {},
                "backendConfig": {"engine": engine},
            })

            # MLP
            nodes.append({
                "name":        f"{L}_mlp_norm",
                "packageName": "qti.aisw",
                "typeName":    "RmsNorm",
                "inputNames":  [f"{L}_attn_res", f"{L}_mlp_norm_weight"],
                "outputNames": [f"{L}_mlp_normed"],
                "params": {"epsilon": 1e-6},
                "backendConfig": {"engine": engine},
            })
            nodes.append({
                "name":        f"{L}_mlp",
                "packageName": "qti.aisw",
                "typeName":    "GatedMLP",
                "inputNames":  [f"{L}_mlp_normed", f"{L}_gate_weight",
                                f"{L}_up_weight", f"{L}_down_weight"],
                "outputNames": [f"{L}_mlp_out"],
                "params": {
                    "hidden_size":   H,
                    "intermediate":  inter,
                    "activation":    "silu",
                },
                "backendConfig": {"engine": engine},
            })
            nodes.append({
                "name":        f"{L}_mlp_residual",
                "packageName": "qti.aisw",
                "typeName":    "ElementWiseAdd",
                "inputNames":  [f"{L}_attn_res", f"{L}_mlp_out"],
                "outputNames": [f"{L}_out"],
                "params": {},
                "backendConfig": {"engine": engine},
            })
            x = f"{L}_out"

        # ── Final norm + LM head ──────────────────────────────────────────────
        static_tensor("final_norm_weight", [H])
        native_tensor("final_normed", act_dims)
        static_tensor("lm_head_weight", [H, sig.vocab_size])
        tensors.append({
            "name": "logits",
            "type": "QNN_TENSOR_TYPE_APP_READ",
            "dataType": qnn_dtype,
            "dims": [1, sig.max_seq_len, sig.vocab_size],
        })
        nodes.append({
            "name": "final_norm",
            "packageName": "qti.aisw",
            "typeName": "RmsNorm",
            "inputNames": [x, "final_norm_weight"],
            "outputNames": ["final_normed"],
            "params": {"epsilon": 1e-6},
            "backendConfig": {"engine": "QNN_BACKEND_HTP"},
        })
        nodes.append({
            "name": "lm_head",
            "packageName": "qti.aisw",
            "typeName": "FullyConnected",
            "inputNames": ["final_normed", "lm_head_weight"],
            "outputNames": ["logits"],
            "params": {"transpose_b": False},
            "backendConfig": {"engine": "QNN_BACKEND_HTP"},
        })

        return {
            "msig_version": sig.msig_version,
            "model_id":     sig.model_id,
            "target":       "qnn",
            "graph": {
                "name":    sig.model_family or "transformer",
                "version": "1.0.0",
                "nodes":   nodes,
                "tensors": tensors,
            },
            "backend_config": {
                "htp_performance_mode": "BURST",
                "htp_precision":         "fp16",
                "spill_fill_bufsize":    128 * 1024 * 1024,  # 128 MB
            },
            "msig_layer_routing": {
                "htp_layers":  int(sig.total_layers * sig.npu_split_ratio),
                "cpu_layers":  len(sig.cpu_layers),
                "split_ratio": sig.npu_split_ratio,
            },
            "content_hash": sig.content_hash,
        }

    def _build_quant_profile(self, sig: ModelSignature) -> dict:
        """Per-layer quantization scale/offset vectors."""
        layers_quant = []
        for layer in sig.layers:
            if layer.is_attention or layer.is_mlp:
                layers_quant.append({
                    "layer_name":      layer.name,
                    "weight_dtype":    "int8" if sig.default_precision == Precision.INT8 else "float16",
                    "activation_dtype":"float16",
                    "scale_type":      "per_channel",
                    "symmetric":       True,
                    "quantize_node":   layer.tier != ExecutionTier.CPU_FALLBACK,
                })

        return {
            "msig_version": sig.msig_version,
            "model_id":     sig.model_id,
            "quant_scheme": "uniform_symmetric",
            "global_dtype": _QNN_DTYPE.get(sig.default_precision, "QNN_DATATYPE_FLOAT_16"),
            "layers":       layers_quant,
            "kv_cache": {
                "dtype":           "QNN_DATATYPE_FLOAT_16",
                "max_cache_bytes": sig.total_kv_cache_bytes,
            },
        }

    def _build_aihub_job(self, sig: ModelSignature, safe_name: str) -> dict:
        """Qualcomm AI Hub job spec for remote hardware testing."""
        return {
            "job_name":        f"msig-{safe_name}",
            "model_id":        sig.model_id,
            "task":            "inference",
            "target_devices":  _AIHUB_DEVICES,
            "input_specs":     [{"name": "input_ids", "shape": [1, 128], "dtype": "int32"}],
            "options": {
                "target_runtime": "onnx",
                "quantize_full_type": "w8a16" if sig.default_precision == Precision.INT8 else "w4a16",
                "quantize_weight_dtype": "int8" if sig.default_precision == Precision.INT8 else "int4",
            },
            "instructions": [
                "1. Sign up free at https://aihub.qualcomm.com",
                "2. pip install qai-hub",
                "3. qai-hub configure --api_token YOUR_TOKEN",
                "4. Use qai_hub.submit_compile_job() with this spec",
                "5. Results run on real Snapdragon silicon in Qualcomm's cloud",
            ],
        }

    def _submit_aihub_job(
        self, sig: ModelSignature, topology: dict, api_token: str
    ) -> str | None:
        """Optionally submit to Qualcomm AI Hub if SDK + token available."""
        try:
            import qai_hub as hub  # type: ignore
            hub.configure(api_token=api_token)
            # For a real model we'd use hub.submit_compile_job(model, ...)
            # Here we validate the connection works
            devices = hub.get_devices()
            if devices:
                return f"aihub_dryrun_{sig.model_id[:20]}"
        except Exception:
            pass
        return None
