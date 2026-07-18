"""
examples/03_all_models_comparison.py

Translate all 7 supported offline models to all 3 backends.
Prints a comparison table of weight sizes, KV-cache, and layer splits.

Run:
    python examples/03_all_models_comparison.py
"""

from universalmsig import build_signature, MSigTranslator, list_supported_models, Precision

models = list_supported_models()
translator = MSigTranslator()

print("=" * 90)
print("  universalmsig — All Models × All Backends")
print("=" * 90)
print()

# ── Signature comparison table ────────────────────────────────────────────────
print(f"{'Model':<45} {'Layers':>6} {'Hidden':>7} {'Heads':>6} {'KV':>4} "
      f"{'Weights':>9} {'KV-cache':>9} {'Fast':>5} {'CPU':>4}")
print("─" * 90)

for model_id in models:
    sig = build_signature(model_id, offline=True)
    name = model_id.split("/")[-1][:44]
    print(
        f"  {name:<43} {sig.total_layers:>6} {sig.hidden_size:>7} "
        f"{sig.num_heads:>6} {sig.num_kv_heads:>4} "
        f"{sig.total_weight_bytes/1e9:>7.1f} GB "
        f"{sig.total_kv_cache_bytes/1e6:>7.0f} MB "
        f"{len(sig.npu_layers):>5} "
        f"{len(sig.cpu_layers):>4}"
    )

print()
print("  Fast = GPU/NPU fast-path layers  |  CPU = CPU fallback layers")
print()

# ── Translate all to all backends ─────────────────────────────────────────────
print("=" * 90)
print("  Translation Results")
print("=" * 90)

total_ok  = 0
total_all = 0

for model_id in models:
    results = translator.translate_model(
        model_id,
        output_dir = f"./output/comparison/{model_id.replace('/', '_')}",
        offline    = True,
    )
    ok  = sum(1 for r in results if r.success)
    total_ok  += ok
    total_all += len(results)
    backends = " ".join(
        f"{'✅' if r.success else '❌'}{r.backend_name}"
        for r in results
    )
    print(f"  {model_id.split('/')[-1]:<44} {backends}")

print()
print(f"  Total: {total_ok}/{total_all} backend compilations succeeded")
