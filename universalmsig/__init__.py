"""
universalmsig — Universal Model Signature cross-vendor compiler.
Translates a unified .msig to TensorRT, CoreML, and QNN assets.
"""
from .translator import MSigTranslator, list_supported_models
from .core.signature import ModelSignature, LayerSignature, ExecutionTier, Precision
from .core.parser import build_signature

__version__ = "0.1.0"
__all__ = [
    "MSigTranslator",
    "ModelSignature",
    "LayerSignature",
    "ExecutionTier",
    "Precision",
    "build_signature",
    "list_supported_models",
]
