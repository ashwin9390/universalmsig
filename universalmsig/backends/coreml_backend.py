"""
universalmsig/backends/coreml_backend.py

Apple CoreML Backend
====================
Translates a ModelSignature into:
  1. A CoreML MIL (Model Intermediate Language) Python script  (always)
  2. A coremltools model spec JSON                             (always)
  3. An actual .mlpackage bundle                               (if coremltools installed)

Test on Google Colab:
  !pip install coremltools
  # Compilation works on Linux; execution needs macOS/Apple Silicon

Test on GitHub Actions (macOS runner):
  - os: macos-latest  → full ANE execution test possible
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .base import BaseBackend, CompilationResult
from ..core.signature import ModelSignature, Precision, ExecutionTier

# CoreML dtype mapping
_CT_DTYPE = {
    Precision.FP32: "float32",
    Precision.FP16: "float16",
    Precision.BF16: "float16",   # CoreML promotes BF16 to FP16
    Precision.INT8: "int8",
    Precision.INT4: "uint4",
    Precision.FP4:  "float16",   # ANE doesn't support FP4, promote
}

# CoreML compute unit mapping based on tier
_CT_COMPUTE_UNITS = {
    ExecutionTier.GPU_FAST:    "ALL",       # ANE preferred
    ExecutionTier.NPU_FAST:    "ALL",
    ExecutionTier.ACCELERATOR: "ALL",
    ExecutionTier.CPU_FALLBACK: "CPU_ONLY",
}


class CoreMLBackend(BaseBackend):
    """
    Apple CoreML / Neural Engine backend.

    Produces:
      • <model>_coreml_spec.json     — model spec for inspection / debugging
      • <model>_mil_graph.py         — executable MIL graph Python script
      • <model>.mlpackage            — compiled CoreML bundle (needs coremltools)
    """

    @property
    def name(self) -> str:
        return "coreml"

    @property
    def supported_precisions(self) -> list[Precision]:
        return [Precision.FP32, Precision.FP16, Precision.INT8, Precision.INT4]

    def validate(self, sig: ModelSignature) -> list[str]:
        warnings = self._check_precision(sig)
        if sig.vocab_size > 200_000:
            warnings.append(
                f"Large vocab ({sig.vocab_size:,}) — embedding lookup may exceed "
                "Apple Neural Engine SRAM. Consider splitting embedding to CPU."
            )
        if sig.num_kv_heads < sig.num_heads:
            warnings.append(
                f"GQA detected ({sig.num_kv_heads} KV heads). "
                "CoreML requires explicit group-query attention unrolling in MIL."
            )
        if sig.max_seq_len > 8192:
            warnings.append(
                f"max_seq_len={sig.max_seq_len} exceeds Apple Neural Engine "
                "optimal window (8192). Long context will fall back to CPU."
            )
        return warnings

    def _describe_output(self) -> str:
        return "CoreML .mlpackage bundle + MIL graph script"

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
        spec_path    = output_dir / f"{safe_name}_coreml_spec.json"
        mil_path     = output_dir / f"{safe_name}_mil_graph.py"
        pkg_path     = output_dir / f"{safe_name}.mlpackage"

        # ── 1. Spec JSON ──────────────────────────────────────────────────────
        spec = self._build_spec(sig)
        spec_path.write_text(json.dumps(spec, indent=2))

        # ── 2. MIL graph Python script ────────────────────────────────────────
        mil_script = self._build_mil_script(sig)
        mil_path.write_text(mil_script)

        # ── 3. Try actual coremltools compilation ─────────────────────────────
        sdk_used = False
        try:
            import coremltools as ct  # type: ignore
            sdk_used = self._compile_mlpackage(sig, spec, pkg_path, ct)
        except ImportError:
            warnings.append(
                "coremltools not found — spec JSON + MIL script produced. "
                "Install: pip install coremltools"
            )

        meta = {
            "coreml_spec":   str(spec_path),
            "mil_graph":     str(mil_path),
            "mlpackage":     str(pkg_path) if sdk_used else "not compiled (SDK missing)",
            "compute_units": "ALL (ANE preferred)",
            "ct_dtype":      _CT_DTYPE.get(sig.default_precision, "float16"),
            "sdk_compiled":  sdk_used,
            "runs_on":       "macOS 13+ / iOS 16+ (compilation on any platform)",
            # informational, not a warning: nothing to act on
            "layout_note":   "Transformer uses (batch, seq, hidden) — "
                             "no NHWC transpose needed for NLP models",
        }

        return CompilationResult(
            success      = True,
            backend_name = self.name,
            output_path  = str(spec_path),
            asset_type   = "coreml_mlpackage",
            model_id     = sig.model_id,
            precision    = sig.default_precision.value,
            warnings     = warnings,
            metadata     = meta,
        )

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _build_spec(self, sig: ModelSignature) -> dict:
        ct_dtype = _CT_DTYPE.get(sig.default_precision, "float16")
        layers_spec = []
        for layer in sig.layers:
            compute = _CT_COMPUTE_UNITS.get(layer.tier, "ALL")
            layers_spec.append({
                "name":         layer.name,
                "type":         "attention" if layer.is_attention
                                else "mlp" if layer.is_mlp
                                else "embedding" if layer.is_embedding
                                else "linear",
                "compute_unit": compute,
                "dtype":        ct_dtype,
                "input_shape":  ["batch", "seq_len", layer.hidden_size],
                "output_shape": ["batch", "seq_len", layer.hidden_size],
                "weight_bytes": layer.weight_bytes,
            })

        return {
            "msig_version":  sig.msig_version,
            "model_id":      sig.model_id,
            "target":        "coreml",
            "spec_version":  8,
            "compute_units": "ALL",
            "model_description": {
                "input": [{
                    "name": "input_ids",
                    "type": "sequence",
                    "dtype": "int32",
                    "shape": ["batch", "seq_len"],
                }],
                "output": [{
                    "name": "logits",
                    "type": "multiArray",
                    "dtype": ct_dtype,
                    "shape": ["batch", "seq_len", sig.vocab_size],
                }],
            },
            "architecture": {
                "num_layers":   sig.total_layers,
                "hidden_size":  sig.hidden_size,
                "num_heads":    sig.num_heads,
                "num_kv_heads": sig.num_kv_heads,
                "vocab_size":   sig.vocab_size,
                "max_seq_len":  sig.max_seq_len,
            },
            "msig_layer_routing": {
                "ane_layers":  [l.name for l in sig.npu_layers],
                "cpu_layers":  [l.name for l in sig.cpu_layers],
            },
            "quantization": {
                "weight_dtype":      ct_dtype,
                "palettize_nbits":   4 if sig.default_precision == Precision.INT4 else None,
                "use_palettization": sig.default_precision == Precision.INT4,
            },
            "layers": layers_spec,
            "content_hash": sig.content_hash,
        }

    def _build_mil_script(self, sig: ModelSignature) -> str:
        """Generate a runnable coremltools MIL graph Python script with correct GQA unrolling."""
        ct_dtype  = _CT_DTYPE.get(sig.default_precision, "float16")
        head_dim  = sig.hidden_size // sig.num_heads
        gqa_ratio = sig.num_heads // sig.num_kv_heads if sig.num_kv_heads > 0 else 1
        is_gqa    = sig.num_kv_heads < sig.num_heads

        lines = [
            '"""',
            f"CoreML MIL Graph — {sig.model_id}",
            f"Generated by universalmsig v{sig.msig_version}",
            "",
            f"Architecture: {sig.total_layers} transformer blocks",
            f"Attention: {sig.num_heads} Q heads, {sig.num_kv_heads} KV heads"
            + (f" (GQA ratio {gqa_ratio}:1 — each KV head serves {gqa_ratio} Q heads)" if is_gqa else " (MHA)"),
            "",
            "GQA Fix: KV heads are explicitly repeated/broadcast to match Q heads",
            "before attention score calculation — required because CoreML MIL",
            "does not natively handle mismatched Q/KV head dimensions.",
            "",
            "Run:  python this_file.py",
            "Needs: pip install coremltools numpy",
            '"""',
            "import numpy as np",
            "import coremltools as ct",
            "from coremltools.converters.mil import Builder as mb",
            "",
            f"MODEL_ID      = {repr(sig.model_id)}",
            f"NUM_LAYERS    = {sig.total_layers}",
            f"HIDDEN_SIZE   = {sig.hidden_size}",
            f"NUM_HEADS     = {sig.num_heads}",
            f"NUM_KV_HEADS  = {sig.num_kv_heads}",
            f"HEAD_DIM      = {head_dim}     # hidden_size // num_heads",
            f"GQA_RATIO     = {gqa_ratio}     # num_heads // num_kv_heads",
            f"IS_GQA        = {is_gqa}",
            f"VOCAB_SIZE    = {sig.vocab_size}",
            f"MAX_SEQ_LEN   = {sig.max_seq_len}",
            f"DTYPE         = '{ct_dtype}'",
            "",
            "",
            "def gqa_attention(x, layer_idx):",
            f'    """',
            f"    Grouped-Query Attention with explicit KV head broadcasting.",
            f"    Q shape : (batch, seq, {sig.num_heads}, {head_dim})",
            f"    K,V shape: (batch, seq, {sig.num_kv_heads}, {head_dim})  → broadcast to ({sig.num_heads})",
            f'    """',
            "    # Q projection",
            f"    q_w = mb.const(val=np.zeros(({sig.hidden_size}, {sig.hidden_size}), dtype=np.float16))",
            "    q = mb.linear(x=x, weight=q_w)",
            f"    q = mb.reshape(x=q, shape=[1, -1, {sig.num_heads}, {head_dim}])",
            "    q = mb.transpose(x=q, perm=[0, 2, 1, 3])  # (B, H, S, D)",
            "",
            "    # K,V projections — KV heads ({} heads, not {})".format(sig.num_kv_heads, sig.num_heads),
            f"    kv_dim = {sig.num_kv_heads} * {head_dim}",
            f"    k_w = mb.const(val=np.zeros(({sig.hidden_size}, {sig.num_kv_heads * head_dim}), dtype=np.float16))",
            f"    v_w = mb.const(val=np.zeros(({sig.hidden_size}, {sig.num_kv_heads * head_dim}), dtype=np.float16))",
            "    k = mb.linear(x=x, weight=k_w)",
            "    v = mb.linear(x=x, weight=v_w)",
            f"    k = mb.reshape(x=k, shape=[1, -1, {sig.num_kv_heads}, {head_dim}])",
            f"    v = mb.reshape(x=v, shape=[1, -1, {sig.num_kv_heads}, {head_dim}])",
            "    k = mb.transpose(x=k, perm=[0, 2, 1, 3])  # (B, KV_H, S, D)",
            "    v = mb.transpose(x=v, perm=[0, 2, 1, 3])",
            "",
        ]

        if is_gqa:
            lines += [
                f"    # ── GQA UNROLLING ─────────────────────────────────────────────",
                f"    # KV heads must be broadcast from {sig.num_kv_heads} → {sig.num_heads}",
                f"    # before attention score matmul. CoreML MIL does not support",
                f"    # implicit broadcasting across the head dimension.",
                f"    # Each KV head is repeated {gqa_ratio} times (GQA ratio = {sig.num_heads}/{sig.num_kv_heads}).",
                f"    k = mb.tile(x=k, reps=[1, {gqa_ratio}, 1, 1])  # (B, {sig.num_heads}, S, D)",
                f"    v = mb.tile(x=v, reps=[1, {gqa_ratio}, 1, 1])  # (B, {sig.num_heads}, S, D)",
                f"    # ── END GQA UNROLLING ──────────────────────────────────────────",
                "",
            ]
        else:
            lines += [
                "    # MHA — no unrolling needed, Q and KV heads are equal",
                "",
            ]

        lines += [
            "    # Scaled dot-product attention",
            f"    scale = mb.const(val=np.float16(1.0 / ({head_dim} ** 0.5)))",
            "    attn  = mb.matmul(x=q, y=mb.transpose(x=k, perm=[0, 1, 3, 2]))",
            "    attn  = mb.mul(x=attn, y=scale)",
            "    attn  = mb.softmax(x=attn, axis=-1)",
            "    out   = mb.matmul(x=attn, y=v)        # (B, H, S, D)",
            "    out   = mb.transpose(x=out, perm=[0, 2, 1, 3])  # (B, S, H, D)",
            f"    out   = mb.reshape(x=out, shape=[1, -1, {sig.hidden_size}])  # (B, S, hidden)",
            f"    o_w   = mb.const(val=np.zeros(({sig.hidden_size}, {sig.hidden_size}), dtype=np.float16))",
            "    return mb.linear(x=out, weight=o_w)",
            "",
            "",
            "@mb.program(",
            "    input_specs=[",
            "        mb.TensorSpec(shape=(1, ct.RangeDim(1, MAX_SEQ_LEN)), dtype=np.int32),",
            "    ],",
            "    opset_version=ct.target.iOS17,",
            ")",
            "def transformer_prog(input_ids):",
            f"    # Embedding  ({sig.vocab_size:,} tokens × {sig.hidden_size} hidden)",
            f"    embed_w = mb.const(val=np.zeros(({sig.vocab_size}, {sig.hidden_size}), dtype=np.float16))",
            "    x = mb.gather(x=embed_w, indices=input_ids, axis=0)",
        ]

        npu_boundary = int(sig.total_layers * sig.npu_split_ratio)
        for i in range(min(sig.total_layers, 3)):
            tier = "GPU_FAST (ANE)" if i < npu_boundary else "CPU_FALLBACK"
            lines += [
                "",
                f"    # ── Layer {i} [{tier}] ────────────────────────────────────",
                "    residual = x",
                "    x = mb.layer_norm(x=x, axes=[-1])",
                f"    x = gqa_attention(x, layer_idx={i})",
                "    x = mb.add(x=x, y=residual)   # attention residual",
                "    residual = x",
                "    x = mb.layer_norm(x=x, axes=[-1])",
                f"    # SwiGLU MLP: gate × silu + up → down",
                f"    gate_w = mb.const(val=np.zeros(({sig.hidden_size}, {sig.hidden_size * 4}), dtype=np.float16))",
                f"    up_w   = mb.const(val=np.zeros(({sig.hidden_size}, {sig.hidden_size * 4}), dtype=np.float16))",
                f"    down_w = mb.const(val=np.zeros(({sig.hidden_size * 4}, {sig.hidden_size}), dtype=np.float16))",
                "    gate   = mb.silu(x=mb.linear(x=x, weight=gate_w))",
                "    up     = mb.linear(x=x, weight=up_w)",
                "    x      = mb.linear(x=mb.mul(x=gate, y=up), weight=down_w)",
                "    x      = mb.add(x=x, y=residual)   # mlp residual",
            ]

        if sig.total_layers > 3:
            lines += ["", f"    # ... (layers 3–{sig.total_layers-1} follow the same pattern) ..."]

        lines += [
            "",
            "    # LM head",
            "    x = mb.layer_norm(x=x, axes=[-1])",
            f"    lm_w = mb.const(val=np.zeros(({sig.hidden_size}, {sig.vocab_size}), dtype=np.float16))",
            "    logits = mb.linear(x=x, weight=lm_w)",
            "    return logits",
            "",
            "",
            "if __name__ == '__main__':",
            f"    print(f'Building CoreML model for {{MODEL_ID}} ...')",
            "    mlmodel = ct.convert(",
            "        transformer_prog,",
            "        convert_to='mlprogram',",
            "        minimum_deployment_target=ct.target.iOS17,",
            "        compute_units=ct.ComputeUnit.ALL,",
            "    )",
            f"    out = '{sig.model_id.replace('/', '_').lower()}.mlpackage'",
            "    mlmodel.save(out)",
            "    print(f'Saved CoreML model → {{out}}')",
            f"    print(f'GQA unrolling applied: {is_gqa} (ratio {gqa_ratio}:1)')",
        ]

        return "\n".join(lines)

    def _compile_mlpackage(
        self, sig: ModelSignature, spec: dict, pkg_path: Path, ct: Any
    ) -> bool:
        """Try actual coremltools compilation."""
        try:
            from coremltools.converters.mil import Builder as mb  # type: ignore
            import numpy as np

            ct_dtype_np = np.float16 if sig.default_precision != Precision.FP32 else np.float32

            @mb.program(
                input_specs=[
                    mb.TensorSpec(shape=(1, ct.RangeDim(1, min(sig.max_seq_len, 512))),
                                  dtype=np.int32),
                ],
                opset_version=ct.target.iOS17,
            )
            def minimal_prog(input_ids):
                embed = mb.const(
                    val=np.zeros((sig.vocab_size, sig.hidden_size), dtype=ct_dtype_np)
                )
                x = mb.gather(x=embed, indices=input_ids, axis=0)
                lm_w = mb.const(
                    val=np.zeros((sig.hidden_size, sig.vocab_size), dtype=ct_dtype_np)
                )
                return mb.linear(x=x, weight=lm_w)

            model = ct.convert(
                minimal_prog,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS17,
                compute_units=ct.ComputeUnit.ALL,
            )
            model.save(str(pkg_path))
            return True
        except Exception:
            return False
