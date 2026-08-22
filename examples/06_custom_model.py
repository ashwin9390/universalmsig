"""
examples/06_custom_model.py

Add a custom model spec at runtime (without editing parser.py).
Useful for proprietary models or fine-tuned variants.

Run:
    python examples/06_custom_model.py
"""

from universalmsig.core.signature import (
    ModelSignature, LayerSignature, ExecutionTier, Precision
)
from universalmsig import MSigTranslator
import math

# ── Define a custom model spec manually ──────────────────────────────────────
MODEL_ID        = "my-org/my-custom-7b"
NUM_LAYERS      = 32
HIDDEN_SIZE     = 4096
NUM_HEADS       = 32
NUM_KV_HEADS    = 8
VOCAB_SIZE      = 32000
MAX_SEQ_LEN     = 8192
NPU_SPLIT       = 0.70
PRECISION       = Precision.INT8

# Revised: Use math.ceil to match the v2.0.0 boundary routing scheme
npu_boundary = int(math.ceil(NUM_LAYERS * NPU_SPLIT))
attn_bytes   = HIDDEN_SIZE * HIDDEN_SIZE * 4 * 1    # INT8 = 1 byte
mlp_bytes    = HIDDEN_SIZE * HIDDEN_SIZE * 4 * 2 * 1
kv_bytes     = 2 * NUM_KV_HEADS * (HIDDEN_SIZE // NUM_HEADS) * MAX_SEQ_LEN * 1

layers = []

# Embedding
layers.append(LayerSignature(
    index=0, name="model.embed_tokens",
    tier=ExecutionTier.GPU_FAST, precision=PRECISION,
    weight_bytes=VOCAB_SIZE * HIDDEN_SIZE, hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS, is_embedding=True,
))

# Transformer layers
for i in range(NUM_LAYERS):
    tier = ExecutionTier.GPU_FAST if i < npu_boundary else ExecutionTier.CPU_FALLBACK
    layers.append(LayerSignature(
        index=len(layers), name=f"model.layers.{i}.self_attn",
        tier=tier, precision=PRECISION, weight_bytes=attn_bytes,
        hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
        is_attention=True, kv_cache_bytes=kv_bytes,
    ))
    layers.append(LayerSignature(
        index=len(layers), name=f"model.layers.{i}.mlp",
        tier=tier, precision=PRECISION, weight_bytes=mlp_bytes,
        hidden_size=HIDDEN_SIZE, num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
        is_mlp=True,
    ))

# LM head
layers.append(LayerSignature(
    index=len(layers), name="lm_head",
    tier=ExecutionTier.GPU_FAST, precision=PRECISION,
    weight_bytes=HIDDEN_SIZE * VOCAB_SIZE, hidden_size=HIDDEN_SIZE,
    num_heads=NUM_HEADS, num_kv_heads=NUM_KV_HEADS,
))

# Build signature
sig = ModelSignature(
    model_id         = MODEL_ID,
    model_family     = "llama",
    architecture     = "transformer-decoder",
    total_layers     = NUM_LAYERS,
    hidden_size      = HIDDEN_SIZE,
    num_heads        = NUM_HEADS,
    num_kv_heads     = NUM_KV_HEADS,
    vocab_size       = VOCAB_SIZE,
    max_seq_len      = MAX_SEQ_LEN,
    total_params     = 7_000_000_000,
    default_precision = PRECISION,
    npu_split_ratio  = NPU_SPLIT,
    layers           = layers,
    notes            = "Custom 7B INT8 model for edge deployment",
)
sig.compute_hash()

print(sig.summary())
print()

# Translate
translator = MSigTranslator()
results = translator.translate_signature(
    sig,
    targets    = ["tensorrt", "coreml", "qnn"],
    output_dir = "./output/custom_7b",
)

print("\
=== Custom Model Translation Results ===")
for r in results:
    status = "✅" if r.success else "❌"
    print(f"  {status} [{r.backend_name}] → {r.output_path}")
