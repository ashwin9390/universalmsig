"""
examples/04_precision_comparison.py

Compile the same model at different precisions and show what
each backend produces — and which precisions each backend supports.

Run:
    python examples/04_precision_comparison.py
"""

from universalmsig import MSigTranslator, build_signature, Precision

PRECISIONS = [Precision.FP16, Precision.INT8, Precision.INT4]
MODEL_ID   = "Qwen/Qwen2.5-0.5B"

translator = MSigTranslator()

print("=" * 70)
print(f"  Precision Comparison — {MODEL_ID}")
print("=" * 70)
print()

for prec in PRECISIONS:
    sig = build_signature(MODEL_ID, precision=prec, offline=True)
    weight_gb = sig.total_weight_bytes / 1e9

    print(f"  [{prec.value.upper()}]  Weights ≈ {weight_gb:.2f} GB")

    results = translator.translate_model(
        MODEL_ID,
        precision  = prec.value,
        targets    = ["tensorrt", "coreml", "qnn"],
        output_dir = f"./output/precision/{prec.value}",
        offline    = True,
    )

    for r in results:
        status = "✅" if r.success else "❌"
        # Show precision-specific metadata
        meta_note = ""
        if r.backend_name == "tensorrt":
            meta_note = f"  trt_dtype={r.metadata.get('trt_dtype','?')}"
        elif r.backend_name == "coreml":
            meta_note = f"  ct_dtype={r.metadata.get('ct_dtype','?')}"
        elif r.backend_name == "qnn":
            meta_note = f"  qnn_dtype={r.metadata.get('qnn_dtype','?')}"
        print(f"    {status} {r.backend_name:<12}{meta_note}")

    print()

# ── Backend precision support matrix ─────────────────────────────────────────
print("=" * 70)
print("  Backend Precision Support Matrix")
print("=" * 70)

from universalmsig.backends.tensorrt_backend import TensorRTBackend
from universalmsig.backends.coreml_backend import CoreMLBackend
from universalmsig.backends.qnn_backend import QNNBackend

backends = [TensorRTBackend(), CoreMLBackend(), QNNBackend()]
all_precs = [Precision.FP32, Precision.FP16, Precision.BF16,
             Precision.INT8, Precision.INT4, Precision.FP4]

header = f"  {'Backend':<12}" + "".join(f"  {p.value:>6}" for p in all_precs)
print(header)
print("  " + "─" * (len(header) - 2))

for b in backends:
    row = f"  {b.name:<12}"
    for p in all_precs:
        supported = "✅" if p in b.supported_precisions else "❌"
        row += f"  {supported:>6}"
    print(row)
print()
