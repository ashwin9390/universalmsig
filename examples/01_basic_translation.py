"""
examples/01_basic_translation.py

Simplest possible usage — translate Qwen2.5-0.5B to all three backends.
No internet, no GPU, no vendor SDKs required.

Run:
    cd universalmsig
    python examples/01_basic_translation.py
"""

from universalmsig import MSigTranslator

translator = MSigTranslator()

results = translator.translate_model(
    model_id   = "Qwen/Qwen2.5-0.5B",
    targets    = ["tensorrt", "coreml", "qnn"],
    output_dir = "./output/basic",
    offline    = True,
)

print("\n=== Results ===")
for r in results:
    status = "✅" if r.success else "❌"
    print(f"  {status} [{r.backend_name}] → {r.output_path}")
