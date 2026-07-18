# Backend Reference

Each backend in `universalmsig/backends/` translates a `ModelSignature`
into vendor-specific compilation assets. All backends implement the
`BaseBackend` abstract interface and work fully offline — vendor SDKs
are optional enhancements.

---

## BaseBackend Interface

```python
class BaseBackend(ABC):
    @property
    def name(self) -> str: ...                    # "tensorrt" | "coreml" | "qnn"

    @property
    def supported_precisions(self) -> list[Precision]: ...

    def validate(self, sig: ModelSignature) -> list[str]:
        """Return warnings list. Raise ValueError for hard incompatibilities."""

    def compile(self, sig: ModelSignature, output_dir: Path, **kwargs) -> CompilationResult:
        """Translate sig → vendor assets. Works without SDK (produces config JSON)."""

    def dry_run(self, sig: ModelSignature) -> dict:
        """Describe what would be compiled, no file I/O."""
```

---

## TensorRT Backend

**File:** `universalmsig/backends/tensorrt_backend.py`
**Target:** NVIDIA GPUs (H100, A100, RTX series, T4 in Colab)

### What it produces

| File | Always? | Needs SDK? |
|---|---|---|
| `*_tensorrt_config.json` | ✅ Yes | No |
| `*_trtllm_config.json` | ✅ Yes | No |
| `*.engine` | ❌ Optional | `pip install tensorrt` + NVIDIA GPU |

### tensorrt_config.json structure

```json
{
  "builder_config": {
    "max_batch_size": 1,
    "max_input_len": 4096,
    "dtype": "HALF",
    "use_fp8_context_fmha": false,
    "enable_xqa": true,
    "paged_kv_cache": false
  },
  "network_config": {
    "num_layers": 24,
    "hidden_size": 896,
    "num_attention_heads": 14,
    "num_kv_heads": 2,
    "vocab_size": 151936
  },
  "layer_routing": {
    "gpu_fast_path": ["model.layers.0.self_attn", ...],
    "cpu_offload":   ["model.layers.17.self_attn", ...], // Verified 17 blocks Fast / 7 blocks CPU for Qwen-0.5B
    "split_ratio": 0.70
  },
  "memory_config": {
    "weight_bytes": 987654321,
    "kv_cache_bytes": 12345678,
    "weight_gb": 0.988,
    "kv_cache_mb": 12.3
  }
}
```

### How to get a real .engine

```bash
# 1. Google Colab with T4 GPU runtime
# 2. Install TensorRT
!pip install tensorrt

# 3. Run translation
from universalmsig import MSigTranslator
t = MSigTranslator()
results = t.translate_model("Qwen/Qwen2.5-0.5B", targets=["tensorrt"], output_dir="./out")
# → If GPU present, produces .engine file
```

### Supported precisions

`fp32`, `fp16`, `bf16`, `int8` (SmoothQuant), `int4` (AWQ), `fp4`

### Known warnings

- Models > 70B → recommend TensorRT-LLM tensor parallelism
- max_seq_len > 32k → enable paged KV-cache in TRT-LLM config

---

## CoreML Backend

**File:** `universalmsig/backends/coreml_backend.py`
**Target:** Apple Silicon (M1/M2/M3/M4) and iPhone/iPad (A-series)

### What it produces

| File | Always? | Needs SDK? |
|---|---|---|
| `*_coreml_spec.json` | ✅ Yes | No |
| `*_mil_graph.py` | ✅ Yes | No |
| `*.mlpackage` | ❌ Optional | `pip install coremltools` |

### coreml_spec.json structure

```json
{
  "model_description": {
    "input":  [{"name": "input_ids", "type": "sequence", "dtype": "int32"}],
    "output": [{"name": "logits",    "type": "multiArray", "dtype": "float16"}]
  },
  "msig_layer_routing": {
    "ane_layers": ["model.layers.0.self_attn", ...],
    "cpu_layers": ["model.layers.17.self_attn", ...]
  },
  "quantization": {
    "weight_dtype": "float16",
    "palettize_nbits": null,
    "use_palettization": false
  }
}
```

### mil_graph.py

The MIL (Model Intermediate Language) script is directly executable with
`coremltools` installed. It builds a real CoreML `mlprogram`:

```python
@mb.program(input_specs=[...], opset_version=ct.target.iOS17)
def transformer_prog(input_ids):
    x = mb.gather(x=embed_weight, indices=input_ids, axis=0)
    # ... attention + MLP blocks (includes unrolled GQA expansions) ...
    return mb.linear(x=x, weight=lm_head_weight)

mlmodel = ct.convert(transformer_prog, convert_to="mlprogram", ...)
mlmodel.save("model.mlpackage")
```

### How to compile and run on Apple Silicon

```bash
# Compilation (works on Linux too)
pip install coremltools
python msig_output/coreml/qwen_qwen2.5_0.5b_mil_graph.py
# → produces qwen_qwen2.5-0.5b.mlpackage

# Execution (macOS only, uses Apple Neural Engine)
import coremltools as ct
model = ct.models.MLModel("qwen_qwen2.5-0.5b.mlpackage")
result = model.predict({"input_ids": ...})
```

### Layout note

CoreML's Neural Engine (ANE) typically expects NHWC tensors for vision models.
Transformer NLP models use `(batch, seq_len, hidden_size)` layout which is
natively compatible — **no transpose needed**.

### Supported precisions

`fp32`, `fp16`, `int8`, `int4` (via uint4 palettization)
- `bf16` → promoted to `fp16`
- `fp4` → promoted to `fp16`

### GitHub Actions — free macOS test

The CI job `test-macos` runs on `macos-latest` which includes Apple Silicon.
This tests CoreML compilation + model execution automatically on every push.

---

## QNN Backend

**File:** `universalmsig/backends/qnn_backend.py`
**Target:** Qualcomm Snapdragon (Hexagon HTP — Tensor Processor)

### What it produces

| File | Always? | Needs SDK? |
|---|---|---|
| `*_qnn_topology.json` | ✅ Yes | No |
| `*_qnn_quant_profile.json` | ✅ Yes | No |
| `*_aihub_job.json` | ✅ Yes | No |
| Remote Snapdragon run | ❌ Optional | `pip install qai-hub` + free API token |

### qnn_topology.json structure

Compatible with Qualcomm's `qnn-model-lib-generator` tool:

```json
{
  "graph": {
    "name": "qwen2",
    "nodes": [
      {
        "name": "layer_0_self_attn",
        "typeName": "MultiHeadAttention",
        "params": {"num_heads": 14, "num_kv_heads": 2, "use_rope": true},
        "backendConfig": {"engine": "QNN_BACKEND_HTP"}
      },
      {
        "name": "layer_17_self_attn",
        "backendConfig": {"engine": "QNN_BACKEND_CPU"}
      }
    ]
  },
  "backend_config": {
    "htp_performance_mode": "BURST",
    "htp_precision": "fp16",
    "spill_fill_bufsize": 134217728
  },
  "msig_layer_routing": {
    "htp_layers": 17,
    "cpu_layers": 7
  }
}
```

### How to test on real Snapdragon hardware (free)

```bash
# 1. Sign up free at https://aihub.qualcomm.com
# 2. Install SDK
pip install qai-hub

# 3. Configure token
export QAI_HUB_API_TOKEN=your_token_here

# 4. Generate QNN assets
msig-translate --model Qwen/Qwen2.5-0.5B --target qnn

# 5. Submit to Snapdragon 8 Gen 3 in Qualcomm's cloud
#    (follow instructions inside *_aihub_job.json)
import qai_hub as hub
# See: https://app.aihub.qualcomm.com/docs/
```

### Supported devices (AI Hub free tier)

- Snapdragon 8 Gen 3
- Snapdragon X Elite
- Snapdragon 8s Gen 3

### Supported precisions

`fp16`, `int8` (uniform symmetric), `int4`
- `fp32` → works but not recommended (poor HTP performance)
- `bf16` / `fp4` → promoted to `fp16`

### Layout note

Qualcomm HTP (Hexagon) mandates NHWC for vision CNNs.
Transformer models use `(batch, seq, hidden)` which maps directly —
**no NHWC transpose required for NLP workloads**.

### Quantization profile

The `*_qnn_quant_profile.json` specifies per-layer scale/offset:

```json
{
  "quant_scheme": "uniform_symmetric",
  "global_dtype": "QNN_DATATYPE_FLOAT_16",
  "layers": [
    {
      "layer_name": "model.layers.0.self_attn",
      "weight_dtype": "float16",
      "activation_dtype": "float16",
      "scale_type": "per_channel",
      "symmetric": true,
      "quantize_node": true
    }
  ]
}
```

---

## Adding a New Backend

```python
# universalmsig/backends/my_backend.py
from .base import BaseBackend, CompilationResult
from ..core.signature import ModelSignature, Precision

class MyBackend(BaseBackend):

    @property
    def name(self) -> str:
        return "mybackend"

    @property
    def supported_precisions(self) -> list[Precision]:
        return [Precision.FP16, Precision.INT8]

    def validate(self, sig: ModelSignature) -> list[str]:
        warnings = self._check_precision(sig)
        # Add your validation logic
        return warnings

    def compile(self, sig, output_dir, **kwargs) -> CompilationResult:
        # Translate sig → your vendor assets
        # Always produce at least a config JSON (no SDK required)
        return CompilationResult(
            success=True, backend_name=self.name,
            output_path=str(output_dir / "config.json"),
            asset_type="my_asset", model_id=sig.model_id,
            precision=sig.default_precision.value,
        )
```

Register in `universalmsig/translator.py`:
```python
from .backends.my_backend import MyBackend

class MSigTranslator:
    def __init__(self):
        self._backends = {
            "tensorrt": TensorRTBackend(),
            "coreml":   CoreMLBackend(),
            "qnn":      QNNBackend(),
            "mybackend": MyBackend(),   # ← add here
        }
```
