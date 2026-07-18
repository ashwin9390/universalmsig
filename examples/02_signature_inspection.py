"""
examples/02_signature_inspection.py

Build a ModelSignature and inspect every layer's routing decision.
Shows exactly which layers go to GPU/NPU and which fall back to CPU.

Run:
    python examples/02_signature_inspection.py
"""

from universalmsig import build_signature, Precision

sig = build_signature(
    model_id        = "meta-llama/Llama-3.2-1B",
    precision       = Precision.FP16,
    npu_split_ratio = 0.70,
    max_seq_len     = 4096,
    offline         = True,
)

# High-level summary
print(sig.summary())
print()

# Per-layer routing table
print(f"{'Layer':<6} {'Tier':<16} {'Type':<12} {'Weight MB':>10}  Name")
print("─" * 70)
for layer in sig.layers:
    layer_type = (
        "attention" if layer.is_attention else
        "mlp"       if layer.is_mlp       else
        "embedding" if layer.is_embedding else
        "lm_head"
    )
    tier_label = layer.tier.value
    weight_mb  = layer.weight_bytes / 1e6
    print(f"  {layer.index:<4} {tier_label:<16} {layer_type:<12} {weight_mb:>8.1f} MB  {layer.name}")

print()
print(f"Fast tier total : {sig.total_weight_bytes / 1e9 * sig.npu_split_ratio:.2f} GB")
print(f"CPU tier total  : {sig.total_weight_bytes / 1e9 * (1 - sig.npu_split_ratio):.2f} GB")
print(f"KV-cache (4096) : {sig.total_kv_cache_bytes / 1e6:.1f} MB")

# Save to JSON for inspection
sig.save_json("llama_1b.msig")
print(f"\nSaved → llama_1b.msig")
