# universalmsig

**Universal Model Signature (.msig) — Cross-vendor AI compiler.**

One unified model description → TensorRT (NVIDIA) + CoreML (Apple) + QNN (Qualcomm). No recompilation. No vendor lock-in

```
$ msig-translate --model Qwen/Qwen2.5-0.5B --target all

  [TENSORRT] Compiling Qwen/Qwen2.5-0.5B …
  ✅ SUCCESS [tensorrt]
    Model    : Qwen/Qwen2.5-0.5B
    Asset    : tensorrt_engine_config
    Output   : msig_output/tensorrt/qwen_qwen2.5_0.5b_tensorrt_config.json
    Precision: fp16
    fast_blocks: 17   cpu_fallback: 7

  [COREML] Compiling Qwen/Qwen2.5-0.5B …
  ✅ SUCCESS [coreml]
    Asset    : coreml_mlpackage
    Output   : msig_output/coreml/qwen_qwen2.5_0.5b_coreml_spec.json

  [QNN] Compiling Qwen/Qwen2.5-0.5B …
  ✅ SUCCESS [qnn]
    Asset    : qnn_topology
    Output   : msig_output/qnn/qwen_qwen2.5_0.5b_qnn_topology.json
    htp_engine: Hexagon Tensor Processor (HTP)
```

---

## 1. Project Overview & Capabilities

The `universalmsig` project enables cross-vendor AI model translation for LLMs. It converts a unified Model Signature (.msig) into optimized configurations for NVIDIA TensorRT, Apple CoreML, and Qualcomm QNN. It now supports dynamic 70/30 layer routing for any model size.

ONNX describes the raw compute graph; `.msig` maps **where each block should execute and how it balances across hardware memory domains**.

---

## 2. Verified Deployment Environment

- **Hardware**: NVIDIA Tesla T4 GPU (16GB GDDR6)
- **Compute**: CUDA Available (True)
- **Platform**: Google Colab / Linux 6.1 / Python 3.12

---

## 3. Folder Structure

```text
universalmsig/
├── universalmsig/
│   ├── __init__.py
│   ├── core/
│   │   ├── signature.py        ← ModelSignature format (.msig v2)
│   │   └── parser.py           ← HF config → ModelSignature builder
│   ├── backends/
│   │   ├── base.py             ← BaseBackend abstract class
│   │   ├── tensorrt_backend.py ← NVIDIA TensorRT / TRT-LLM backend
│   │   ├── coreml_backend.py   ← Apple CoreML / ANE backend
│   │   └── qnn_backend.py      ← Qualcomm QNN / AI Hub backend
│   ├── translator.py           ← MSigTranslator routing engine
│   └── cli.py                  ← msig-translate CLI entry point
├── tests/
│   └── test_universalmsig.py   ← 51 unit tests (fully offline)
├── universalmsig_demo.ipynb    ← Google Colab notebook (13 cells)
├── .github/workflows/ci.yml    ← GitHub Actions CI pipeline
├── TEST_RESULTS.txt            ← Detailed pre-run test output summary
├── INSTRUCTIONS.md             ← Step-by-step local & cloud configuration guide
├── pyproject.toml              ← Project metadata and zero-dependency build configuration
└── LICENSE                     ← Apache-2.0 License
```

---

## 4. Production & Development Requirements

### Production Requirements (`requirements.txt`)
Core functionalities feature **zero hard dependencies**; the system operates immediately on clean installations without local vendor framework footprints.
```text
# universalmsig — Production Requirements
huggingface_hub>=0.23      # Optional: Live HuggingFace model download (config fetch)
tensorrt>=10.0             # Optional: NVIDIA TensorRT (real .engine compilation)
coremltools>=7.0           # Optional: Apple CoreML (real .mlpackage compilation)
numpy>=1.24                # Optional: Support runtime for CoreML builders
qai-hub>=0.15              # Optional: Qualcomm AI Hub hardware submission
```

### Development Requirements (`requirements-dev.txt`)
```text
# universalmsig — Development & Test Suite Requirements
pytest>=8.2                # Framework test discovery and execution
pytest-asyncio>=0.23        # Async test mapping utilities
ruff>=0.4                  # Quality linting & formatting pipeline
mypy>=1.10                 # Strict typing matrix analysis
coremltools>=7.0           # Local validation runtimes
numpy>=1.24
huggingface_hub>=0.23
```

---

## 5. Quick Start — Zero Dependencies

```bash
git clone https://github.com/YOUR_USERNAME/universalmsig
cd universalmsig
pip install -e .

# List supported offline models (no download needed)
msig-translate --list-models

# Dry run — see what each backend would produce
msig-translate --dry-run --model Qwen/Qwen2.5-0.5B

# Translate to all three backends
msig-translate --model Qwen/Qwen2.5-0.5B --target all

# Translate to TensorRT only, INT8 precision
msig-translate --model meta-llama/Llama-3.2-1B --target tensorrt --precision int8

# Translate to QNN, 28-layer DeepSeek configuration
msig-translate --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --target qnn

# Save .msig JSON design file to local workspace
msig-translate --save-msig Qwen/Qwen2.5-0.5B qwen.msig

# Translate directly from a saved .msig file
msig-translate --file qwen.msig --target coreml

# Stream JSON output straight to command-line pipes
msig-translate --model Qwen/Qwen2.5-0.5B --target tensorrt --json-output | jq .
```

---

## 6. Verified Offline Model Suite (Fixed Scaling)

All backends report synchronized sub-layer layouts according to the structural limits:

### `meta-llama/Llama-3.2-3B` (28 Total Blocks)
```
ModelSignature v2.0.1
  Model         : Llama-3.2-3B
  Fast tier     : 20 blocks (40 sub-layers) [71%]
  CPU tier      : 8 blocks (16 sub-layers)  [29%]
  [TENSORRT]    : Fast=40, CPU=16 (Synchronized)
  [COREML]      : Fast=40, CPU=16 (Synchronized)
  [QNN]         : Fast=40, CPU=16 (Synchronized)
  ⚠ QNN Warning: 7.2GB exceeds 4GB HTP SRAM limit. Auto-offloading active.
```

### `Qwen/Qwen2.5-0.5B` (24 Total Blocks)
```
ModelSignature v2.0.1
  Model         : Qwen2.5-0.5B
  Fast tier     : 17 blocks (34 sub-layers) [71%]
  CPU tier      : 7 blocks (14 sub-layers)  [29%]
  [BACKENDS]    : All reporting 34/14 sub-layers (Synchronized)
```

### `meta-llama/Llama-3.2-1B` (16 Total Blocks)
```
ModelSignature v2.0.1
  Model         : Llama-3.2-1B
  Fast tier     : 12 blocks (24 sub-layers) [75%]
  CPU tier      : 4 blocks (8 sub-layers)   [25%]
  [BACKENDS]    : All reporting 24/8 sub-layers (Synchronized)
```

---

## 7. Target Output Artifacts

### Deployment Tracking Record (`llama_3b_deployment_config.json`)
```json
{
    "model": "Llama-3.2-3B",
    "total_blocks": 28,
    "allocation": {
        "fast_tier": 20,
        "cpu_tier": 8
    },
    "backends": [
        "tensorrt",
        "coreml",
        "qnn"
    ],
    "precision": "fp16",
    "status": "final_synchronized_deployment"
}
```

---

## 8. What Each Backend Produces

### TensorRT
- `*_tensorrt_config.json` — Engine build config (precision, KV-cache, paged attention).
- `*_trtllm_config.json` — TensorRT-LLM builder config for generative models.
- `*.engine` — Compiled binary plan (requires active local CUDA environments).

### CoreML
- `*_coreml_spec.json` — Model spec (input/output shapes, ANE routing matrices, quantization paths).
- `*_mil_graph.py` — Executable intermediate MIL graph Python scripting pipeline.
- `*.mlpackage` — Compiled Apple Silicon runtime bundle directory.

### QNN
- `*_qnn_topology.json` — Validated layout configuration target for `qnn-model-lib-generator`.
- `*_qnn_quant_profile.json` — Per-layer quantization tensors carrying execution range details.
- `*_aihub_job.json` — Ready-to-submit Qualcomm AI Hub job specifications.

---

## 9. Test Suite & Quality Assurance

To execute the verification suite completely offline, run:
```bash
python tests/test_universalmsig.py -v
```

### Performance Execution Summary
- **Framework**: pytest-8.4.2
- **Core Unit Tests**: 51 / 51 PASSED (100% Stability)
- **Execution Time**: 0.163 seconds

```text
SUITE BREAKDOWN:
  TestModelSignature   — 15 tests  ✅ (Verifies block arithmetic, version headers, and memory bounds)
  TestTensorRTBackend  — 10 tests  ✅ (Validates builder metadata configurations)
  TestCoreMLBackend    — 9  tests  ✅ (Verifies explicit mb.tile GQA broadcasting scripts)
  TestQNNBackend       — 10 tests  ✅ (Validates HTP-boundary layer offloading vectors)
  TestMSigTranslator   — 7  tests  ✅ (Validates target routing execution loops)
```

---

## 10. Core Regressions and Bugs Resolved

### BUG 1 — Layer Allocation Arithmetic Mismatch
- **Problem**: Layer boundaries reported sub-layers and helper segments as full blocks, polluting core calculations and failing total block checks.
- **Fix**: Replaced old hardcoded allocations with a clean ceiling strategy `int(math.ceil(total_layers * ratio))`. Embed tokens and language model heads are now successfully exempted from the execution balancing math.

### BUG 2 — GQA Shape Mismatches on CoreML ANE
- **Problem**: Passing mismatched head dimensions (e.g., Qwen's 14 Query vs. 2 Key/Value heads) into matrix multiplications caused hardware execution errors.
- **Fix**: The engine injects explicit `mb.tile` scripts to broadcast the KV dimensions by a proportional factor ($14 / 2 = 7$) ahead of attention matmuls.

### BUG 3 — Missing Head Broadcasters in Qualcomm QNN
- **Problem**: QNN frameworks could not natively compute mismatched multi-head dimensions inside the MultiHeadAttention nodes, resulting in HTP routing crashes.
- **Fix**: The architecture inserts explicit structural `Tile` nodes ($48$ broadcast objects across Qwen's 24-layer setup) to normalize data shapes.

### BUG 4 — Colab Container Building Crash
- **Problem**: Legacy build-backend declarations (`setuptools.backends.legacy:build`) triggered configuration failures when testing against modern Python runtimes.
- **Fix**: Migrated core setup metadata entirely to the modern `setuptools.build_meta` packaging pipeline.

---

## 11. Contributing

Register a new engine specification inside `universalmsig/core/parser.py` within `OFFLINE_SPECS`:

```python
"your-org/your-model": {
    "num_hidden_layers": 32,
    "hidden_size": 4096,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "vocab_size": 32000,
    "max_position_embeddings": 8192,
    "model_type": "llama",
    "total_params": 7_000_000_000,
}
```

New backend compilers can easily be implemented by deriving standard concrete layers off of `BaseBackend`.

---
## 👤 Author & License

* **Author**: Ashwin — [@ashwin9390](https://github.com/ashwin9390)

* **License**: Apache 2.0. See [LICENSE]


* **GitHub**: [github.com/ashwin9390/archgraph-ai](https://github.com/ashwin9390)