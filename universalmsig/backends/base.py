"""
universalmsig/backends/base.py

BaseBackend — the abstract contract every hardware backend must fulfil.
Each backend receives a ModelSignature and produces vendor-specific
compilation assets (TensorRT engine plan, CoreML .mlpackage, QNN JSON, etc.)
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..core.signature import ModelSignature, Precision


@dataclass
class CompilationResult:
    """Returned by every backend after compile()."""
    success:       bool
    backend_name:  str
    output_path:   str
    asset_type:    str           # "tensorrt_engine", "mlpackage", "qnn_json", etc.
    model_id:      str
    precision:     str
    warnings:      list[str] = field(default_factory=list)
    error:         Optional[str] = None
    metadata:      dict = field(default_factory=dict)

    def summary(self) -> str:
        status = "✅ SUCCESS" if self.success else "❌ FAILED"
        lines = [
            f"{status} [{self.backend_name}]",
            f"  Model    : {self.model_id}",
            f"  Asset    : {self.asset_type}",
            f"  Output   : {self.output_path}",
            f"  Precision: {self.precision}",
        ]
        if self.warnings:
            lines += [f"  ⚠  {w}" for w in self.warnings]
        if self.error:
            lines.append(f"  Error    : {self.error}")
        for k, v in self.metadata.items():
            lines.append(f"  {k:<10}: {v}")
        return "\n".join(lines)


class BaseBackend(ABC):
    """
    Every vendor backend implements this interface.

    Methods
    -------
    name            : short identifier used in CLI --target
    supported_precisions : list of Precision values this backend can handle
    validate        : check signature is compatible, return list of warnings
    compile         : translate signature → vendor asset, return CompilationResult
    dry_run         : validate + describe what would be compiled (no file I/O)
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def supported_precisions(self) -> list[Precision]: ...

    @abstractmethod
    def validate(self, sig: ModelSignature) -> list[str]:
        """
        Return a list of warning strings (empty = fully compatible).
        Raise ValueError for hard incompatibilities.
        """
        ...

    @abstractmethod
    def compile(
        self,
        sig: ModelSignature,
        output_dir: str | Path,
        **kwargs: Any,
    ) -> CompilationResult:
        """
        Translate sig → vendor-specific compilation asset.
        Must work even without the actual vendor SDK installed
        (produce a validated config/JSON scaffold in that case).
        """
        ...

    def dry_run(self, sig: ModelSignature) -> dict:
        """Describe compilation plan without writing any files."""
        warnings = self.validate(sig)
        return {
            "backend":    self.name,
            "model_id":   sig.model_id,
            "precision":  sig.default_precision.value,
            "total_layers": sig.total_layers,
            "fast_layers":  len(sig.npu_layers),
            "cpu_layers":   len(sig.cpu_layers),
            "weight_gb":    round(sig.total_weight_bytes / 1e9, 3),
            "kv_cache_mb":  round(sig.total_kv_cache_bytes / 1e6, 1),
            "warnings":   warnings,
            "would_produce": self._describe_output(),
        }

    def _describe_output(self) -> str:
        return f"{self.name} compilation asset"

    def _check_precision(self, sig: ModelSignature) -> list[str]:
        warnings = []
        if sig.default_precision not in self.supported_precisions:
            supported = [p.value for p in self.supported_precisions]
            warnings.append(
                f"{self.name} does not natively support {sig.default_precision.value}. "
                f"Supported: {supported}. Will auto-cast."
            )
        return warnings
