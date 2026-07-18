"""
universalmsig/translator.py

MSigTranslator — The core routing engine.
Reads a ModelSignature and dispatches to one or all backends.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .core.signature import ModelSignature, Precision
from .core.parser import build_signature, OFFLINE_SPECS
from .backends.base import BaseBackend, CompilationResult
from .backends.tensorrt_backend import TensorRTBackend
from .backends.coreml_backend import CoreMLBackend
from .backends.qnn_backend import QNNBackend


class MSigTranslator:
    """
    Universal .msig translator.

    Usage
    -----
    translator = MSigTranslator()

    # From HuggingFace model ID (offline mode — no download)
    results = translator.translate_model(
        "Qwen/Qwen2.5-0.5B",
        targets=["tensorrt", "coreml", "qnn"],
        output_dir="./output",
        offline=True,
    )

    # From saved .msig JSON file
    results = translator.translate_file("model.msig", targets=["coreml"])

    # Dry-run (no file I/O)
    plan = translator.dry_run("Qwen/Qwen2.5-0.5B", targets=["tensorrt"])
    """

    def __init__(self) -> None:
        self._backends: dict[str, BaseBackend] = {
            "tensorrt": TensorRTBackend(),
            "coreml":   CoreMLBackend(),
            "qnn":      QNNBackend(),
        }

    @property
    def available_backends(self) -> list[str]:
        return list(self._backends.keys())

    # ── Main entry points ─────────────────────────────────────────────────────
    def translate_model(
        self,
        model_id: str,
        targets: Optional[list[str]] = None,
        output_dir: str | Path = "./msig_output",
        precision: str = "fp16",
        npu_split_ratio: float = 0.70,
        max_seq_len: int = 4096,
        offline: bool = True,
        **kwargs,
    ) -> list[CompilationResult]:
        """Build signature from model_id, compile to all target backends."""
        prec = Precision(precision)
        sig  = build_signature(
            model_id        = model_id,
            precision       = prec,
            npu_split_ratio = npu_split_ratio,
            max_seq_len     = max_seq_len,
            offline         = offline,
        )
        return self._run_backends(sig, targets, output_dir, **kwargs)

    def translate_file(
        self,
        msig_path: str | Path,
        targets: Optional[list[str]] = None,
        output_dir: str | Path = "./msig_output",
        **kwargs,
    ) -> list[CompilationResult]:
        """Load .msig JSON and compile to target backends."""
        sig = ModelSignature.load_json(msig_path)
        return self._run_backends(sig, targets, output_dir, **kwargs)

    def translate_signature(
        self,
        sig: ModelSignature,
        targets: Optional[list[str]] = None,
        output_dir: str | Path = "./msig_output",
        **kwargs,
    ) -> list[CompilationResult]:
        """Compile a ModelSignature object directly."""
        return self._run_backends(sig, targets, output_dir, **kwargs)

    def dry_run(
        self,
        model_id: str,
        targets: Optional[list[str]] = None,
        precision: str = "fp16",
        offline: bool = True,
    ) -> dict:
        """Describe compilation plan with no file I/O."""
        prec = Precision(precision)
        sig  = build_signature(model_id=model_id, precision=prec, offline=offline)
        use  = self._resolve_targets(targets)
        return {
            "model_id":  model_id,
            "signature": sig.summary(),
            "backends":  {t: self._backends[t].dry_run(sig) for t in use},
        }

    # ── Internal ──────────────────────────────────────────────────────────────
    def _resolve_targets(self, targets: Optional[list[str]]) -> list[str]:
        if targets is None:
            return list(self._backends.keys())
        unknown = set(targets) - set(self._backends)
        if unknown:
            raise ValueError(
                f"Unknown backend(s): {unknown}. "
                f"Available: {self.available_backends}"
            )
        return targets

    def _run_backends(
        self,
        sig: ModelSignature,
        targets: Optional[list[str]],
        output_dir: str | Path,
        **kwargs,
    ) -> list[CompilationResult]:
        use     = self._resolve_targets(targets)
        results = []
        out     = Path(output_dir)
        for name in use:
            backend     = self._backends[name]
            backend_dir = out / name
            print(f"\n  [{name.upper()}] Compiling {sig.model_id} …")
            result = backend.compile(sig, backend_dir, **kwargs)
            results.append(result)
            print(result.summary())
        return results

    def save_signature(
        self,
        model_id: str,
        path: str | Path,
        precision: str = "fp16",
        offline: bool = True,
    ) -> ModelSignature:
        """Build and save a .msig JSON for later use."""
        sig = build_signature(
            model_id  = model_id,
            precision = Precision(precision),
            offline   = offline,
        )
        sig.save_json(path)
        return sig


def list_supported_models() -> list[str]:
    return list(OFFLINE_SPECS.keys())
