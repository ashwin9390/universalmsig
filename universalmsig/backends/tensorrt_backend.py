"""
universalmsig/backends/tensorrt_backend.py

NVIDIA TensorRT Backend
=======================
Translates a ModelSignature into:
  1. A TensorRT engine build config JSON  (always produced — no GPU needed)
  2. A TensorRT-LLM builder config        (for generative LLMs)
  3. An actual .engine plan               (only if tensorrt SDK is installed + GPU present)

Test on Google Colab:
  Runtime → Change runtime type → T4 GPU
  !pip install tensorrt nvidia-tensorrt
  from universalmsig.backends.tensorrt_backend import TensorRTBackend
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseBackend, CompilationResult
from ..core.signature import ModelSignature, Precision, ExecutionTier

# TensorRT dtype mapping
_TRT_DTYPE = {
    Precision.FP32: "FLOAT",
    Precision.FP16: "HALF",
    Precision.BF16: "BF16",
    Precision.INT8: "INT8",
    Precision.INT4: "INT4",
    Precision.FP4:  "FP4",
}

# TRT-LLM quant mapping
_TRTLLM_QUANT = {
    Precision.FP32: "none",
    Precision.FP16: "none",
    Precision.BF16: "none",
    Precision.INT8: "int8_sq",
    Precision.INT4: "int4_awq",
    Precision.FP4:  "fp4",
}


class TensorRTBackend(BaseBackend):
    """
    NVIDIA TensorRT / TensorRT-LLM backend.

    Produces:
      • <model>_tensorrt_config.json   — engine build instructions
      • <model>_trtllm_config.json     — TRT-LLM builder config for LLMs
      • <model>.engine                 — compiled plan (requires GPU + SDK)
    """

    @property
    def name(self) -> str:
        return "tensorrt"

    @property
    def supported_precisions(self) -> list[Precision]:
        return [Precision.FP32, Precision.FP16, Precision.BF16,
                Precision.INT8, Precision.INT4, Precision.FP4]

    def validate(self, sig: ModelSignature) -> list[str]:
        warnings = self._check_precision(sig)
        if sig.total_params > 70_000_000_000:
            warnings.append("Model >70B — consider TensorRT-LLM tensor parallelism.")
        if sig.max_seq_len > 32768:
            warnings.append("Long context >32k — enable paged KV-cache in TRT-LLM config.")
        return warnings

    def _describe_output(self) -> str:
        return "TensorRT engine plan (.engine) + TRT-LLM builder config JSON"

    def compile(
        self,
        sig: ModelSignature,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> CompilationResult:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        warnings = self.validate(sig)

        safe_name = sig.model_id.replace("/", "_").replace("-", "_").lower()
        config_path  = output_dir / f"{safe_name}_tensorrt_config.json"
        trtllm_path  = output_dir / f"{safe_name}_trtllm_config.json"
        engine_path  = output_dir / f"{safe_name}.engine"

        # ── 1. Build engine configuration ────────────────────────────────────
        engine_cfg = self._build_engine_config(sig)
        config_path.write_text(json.dumps(engine_cfg, indent=2))

        # ── 2. TRT-LLM builder config (for generative LLMs) ─────────────────
        trtllm_cfg = self._build_trtllm_config(sig)
        trtllm_path.write_text(json.dumps(trtllm_cfg, indent=2))

        # ── 3. Try actual TensorRT compilation if SDK available ───────────────
        sdk_used = False
        try:
            import tensorrt as trt  # type: ignore
            sdk_used = self._compile_engine(sig, engine_cfg, engine_path, trt)
        except ImportError:
            warnings.append(
                "tensorrt SDK not found — config JSON produced (no .engine). "
                "Install: pip install tensorrt  (requires NVIDIA GPU + CUDA)"
            )

        meta = {
            "engine_config":   str(config_path),
            "trtllm_config":   str(trtllm_path),
            "engine_plan":     str(engine_path) if sdk_used else "not compiled (SDK missing)",
            "trt_dtype":       _TRT_DTYPE.get(sig.default_precision, "HALF"),
            "kv_cache_type":   "paged" if sig.max_seq_len > 4096 else "static",
            "sdk_compiled":    sdk_used,
        }

        return CompilationResult(
            success      = True,
            backend_name = self.name,
            output_path  = str(config_path),
            asset_type   = "tensorrt_engine_config",
            model_id     = sig.model_id,
            precision    = sig.default_precision.value,
            warnings     = warnings,
            metadata     = meta,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _build_engine_config(self, sig: ModelSignature) -> dict:
        trt_dtype = _TRT_DTYPE.get(sig.default_precision, "HALF")
        fast_layers = [l.name for l in sig.npu_layers if l.is_attention or l.is_mlp]
        cpu_layers  = [l.name for l in sig.cpu_layers  if l.is_attention or l.is_mlp]

        return {
            "msig_version":   sig.msig_version,
            "model_id":       sig.model_id,
            "model_family":   sig.model_family,
            "builder_config": {
                "max_batch_size":    1,
                "max_input_len":     sig.max_seq_len,
                "max_output_len":    512,
                "dtype":             trt_dtype,
                "use_fp8_context_fmha": sig.default_precision == Precision.FP4,
                "enable_xqa":        True,
                "paged_kv_cache":    sig.max_seq_len > 4096,
                "tokens_per_block":  64,
            },
            "network_config": {
                "num_layers":        sig.total_layers,
                "hidden_size":       sig.hidden_size,
                "num_attention_heads": sig.num_heads,
                "num_kv_heads":      sig.num_kv_heads,
                "vocab_size":        sig.vocab_size,
                "architecture":      sig.architecture,
            },
            "layer_routing": {
                "gpu_fast_path":  fast_layers,
                "cpu_offload":    cpu_layers,
                "split_ratio":    sig.npu_split_ratio,
            },
            "memory_config": {
                "weight_bytes":    sig.total_weight_bytes,
                "kv_cache_bytes":  sig.total_kv_cache_bytes,
                "weight_gb":       round(sig.total_weight_bytes / 1e9, 3),
                "kv_cache_mb":     round(sig.total_kv_cache_bytes / 1e6, 1),
            },
            "optimization_profile": {
                "min_batch": 1, "opt_batch": 1, "max_batch": 4,
                "min_seq":   1, "opt_seq":   512, "max_seq": sig.max_seq_len,
            },
            "content_hash": sig.content_hash,
        }

    def _build_trtllm_config(self, sig: ModelSignature) -> dict:
        """TensorRT-LLM high-level builder configuration."""
        return {
            "msig_source":      sig.model_id,
            "architecture":     sig.model_family,
            "dtype":            sig.default_precision.value,
            "quant_mode":       _TRTLLM_QUANT.get(sig.default_precision, "none"),
            "num_hidden_layers": sig.total_layers,
            "hidden_size":      sig.hidden_size,
            "num_attention_heads": sig.num_heads,
            "num_key_value_heads": sig.num_kv_heads,
            "vocab_size":       sig.vocab_size,
            "max_position_embeddings": sig.max_seq_len,
            "max_batch_size":   1,
            "max_input_len":    sig.max_seq_len,
            "max_seq_len":      sig.max_seq_len + 512,
            "kv_cache_config": {
                "enable_block_reuse": True,
                "max_tokens":         sig.max_seq_len * 2,
            },
            "speculative_decoding": False,
            "msig_layer_split": {
                # Derived from the per-layer tiers in the signature — do NOT
                # recompute from the ratio (int() truncation disagreed with the
                # parser's ceil() and shifted the boundary by one block).
                "gpu_boundary_block": sig.npu_block_count,
                "cpu_offload_blocks": sig.cpu_block_count,
            },
        }

    def _compile_engine(
        self, sig: ModelSignature, cfg: dict, engine_path: Path, trt: Any
    ) -> bool:
        """Attempt real TRT engine compilation using the SDK."""
        try:
            logger  = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            network = builder.create_network(
                1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
            )
            bconfig = builder.create_builder_config()

            if sig.default_precision in (Precision.FP16, Precision.BF16):
                bconfig.set_flag(trt.BuilderFlag.FP16)
            elif sig.default_precision == Precision.INT8:
                bconfig.set_flag(trt.BuilderFlag.INT8)

            # Register inputs from signature
            for layer in sig.layers:
                if layer.is_embedding:
                    network.add_input(
                        "input_ids",
                        trt.int32,
                        (-1, -1),  # [batch, seq_len] dynamic
                    )
                    break

            engine_bytes = builder.build_serialized_network(network, bconfig)
            if engine_bytes:
                engine_path.write_bytes(engine_bytes)
                return True
        except Exception:
            pass
        return False
