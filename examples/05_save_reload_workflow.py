"""
examples/05_save_reload_workflow.py

Real-world workflow:
  1. Engineer A builds and saves the .msig file
  2. Engineer B loads it and compiles for their target (no model access needed)

Run:
    python examples/05_save_reload_workflow.py
"""

import json
import os
from universalmsig import build_signature, ModelSignature, MSigTranslator, Precision

print("=" * 60)
print("  Step 1 — Build and save .msig (Engineer A)")
print("=" * 60)

sig = build_signature(
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    precision       = Precision.INT8,
    npu_split_ratio = 0.75,    # 75% fast tier
    max_seq_len     = 8192,
    offline         = True,
)
sig.save_json("deepseek_r1.msig")
sig.save_binary("deepseek_r1.bin")     # compact 28-byte binary (C firmware compatible)

print(f"  Saved:  deepseek_r1.msig  ({os.path.getsize('deepseek_r1.msig'):,} bytes JSON)")
print(f"  Saved:  deepseek_r1.bin   ({os.path.getsize('deepseek_r1.bin')} bytes binary)")
print(f"  Hash:   {sig.content_hash[:32]}...")
print()

print("=" * 60)
print("  Step 2 — Reload and verify (Engineer B)")
print("=" * 60)

loaded = ModelSignature.load_json("deepseek_r1.msig")
print(f"  Model   : {loaded.model_id}")
print(f"  Layers  : {loaded.total_layers}")
print(f"  Hash    : {loaded.content_hash[:32]}...")
print(f"  Match   : {loaded.content_hash == sig.content_hash} ✅")
print()

print("=" * 60)
print("  Step 3 — Compile from file (Engineer B, no model needed)")
print("=" * 60)

translator = MSigTranslator()
results = translator.translate_file(
    "deepseek_r1.msig",
    targets    = ["tensorrt", "qnn"],
    output_dir = "./output/deepseek_reload",
)
for r in results:
    status = "✅" if r.success else "❌"
    print(f"  {status} [{r.backend_name}] {r.asset_type}")

print()

# Show a snippet of the .msig JSON
print("=" * 60)
print("  .msig file contents (first 20 lines)")
print("=" * 60)
with open("deepseek_r1.msig") as f:
    for i, line in enumerate(f):
        if i >= 20:
            print("  ...")
            break
        print(" ", line, end="")

# Cleanup
for f in ["deepseek_r1.msig", "deepseek_r1.bin"]:
    if os.path.exists(f):
        os.remove(f)
