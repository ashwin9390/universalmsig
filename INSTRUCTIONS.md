# universalmsig

**Universal Model Signature (.msig) — Cross-Vendor AI Compiler**

*One unified model description → TensorRT (NVIDIA) + CoreML (Apple) + QNN (Qualcomm).*


## Overview
`universalmsig` enables developers to deploy AI models across heterogeneous hardware with a single, vendor-neutral signature. Eliminate recompilation, bypass vendor lock-in, and standardize your deployment pipeline.

---

## WHAT THIS PROJECT DOES

Takes one unified model description (.msig JSON) and translates it to:
  - NVIDIA TensorRT engine config + TRT-LLM builder config
  - Apple CoreML spec + MIL graph Python script
  - Qualcomm QNN topology JSON + quant profile + AI Hub job spec

No vendor lock-in. No recompilation from scratch for each target.
Works fully offline (no model downloads, no internet needed).


## FILE STRUCTURE AFTER UNZIP

├── universalmsig/
│   ├── core/
│   │   ├── signature.py        # ModelSignature format (.msig v2)
│   │   └── parser.py           # HF config → ModelSignature builder
│   ├── backends/
│   │   ├── base.py             # BaseBackend abstract class
│   │   ├── tensorrt_backend.py # NVIDIA TensorRT / TRT-LLM backend
│   │   ├── coreml_backend.py   # Apple CoreML / ANE backend
│   │   └── qnn_backend.py      # Qualcomm QNN / AI Hub backend
│   ├── translator.py           # MSigTranslator routing engine
│   └── cli.py                  # msig-translate CLI entry point
├── tests/
│   └── test_universalmsig.py   # 51 tests, all offline
├── universalmsig_demo.ipynb    # Google Colab notebook (13 cells)
├── .github/workflows/ci.yml    # CI (Linux + macOS runners)
├── TEST_RESULTS.txt            # Pre-run test results (51/51 PASS)
├── INSTRUCTIONS.md             # This file
├── README.md
├── pyproject.toml
└── LICENSE


----------------------------------------------------------
## OPTION A — LOCAL MACHINE (Linux / macOS / Windows WSL2)
----------------------------------------------------------

### Requirements
  - Python 3.11 or 3.12
  - git

### Step 1 — Unzip and enter
  unzip universalmsig.zip
  cd universalmsig

### Step 2 — Install (zero hard dependencies)
  pip install -e .

### Step 3 — Verify CLI works
  msig-translate --list-models

  Expected output:
    Supported offline models (no download needed):
      Qwen/Qwen2.5-0.5B        layers=24  hidden=896   heads=14 (kv=2)
      Qwen/Qwen2.5-1.5B        layers=28  hidden=1536  heads=12 (kv=2)
      meta-llama/Llama-3.2-1B  layers=16  hidden=2048  heads=32 (kv=8)
      ...

### Step 4 — Dry run (no files written)
  msig-translate --dry-run --model Qwen/Qwen2.5-0.5B

### Step 5 — Full translation (all 3 backends)
  msig-translate --model Qwen/Qwen2.5-0.5B --target all

  Output files in ./msig_output/:
    msig_output/tensorrt/qwen_qwen2.5_0.5b_tensorrt_config.json
    msig_output/tensorrt/qwen_qwen2.5_0.5b_trtllm_config.json
    msig_output/coreml/qwen_qwen2.5_0.5b_coreml_spec.json
    msig_output/coreml/qwen_qwen2.5_0.5b_mil_graph.py
    msig_output/qnn/qwen_qwen2.5_0.5b_qnn_topology.json
    msig_output/qnn/qwen_qwen2.5_0.5b_qnn_quant_profile.json
    msig_output/qnn/qwen_qwen2.5_0.5b_aihub_job.json

### Step 6 — Run tests
  python tests/test_universalmsig.py -v
  # Expected: Ran 51 tests in <1s — OK

### More CLI examples
  # Single backend
  msig-translate --model meta-llama/Llama-3.2-1B --target tensorrt

  # INT8 precision
  msig-translate --model Qwen/Qwen2.5-0.5B --target tensorrt --precision int8

  # Custom output directory
  msig-translate --model Qwen/Qwen2.5-0.5B --out ~/my_output

  # Save .msig file for later
  msig-translate --save-msig Qwen/Qwen2.5-0.5B qwen.msig

  # Translate from saved .msig
  msig-translate --file qwen.msig --target coreml

  # JSON output (pipe to jq)
  msig-translate --model Qwen/Qwen2.5-0.5B --json-output | jq .

  # DeepSeek to QNN
  msig-translate --model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --target qnn

  # All 7 models to all 3 backends
  for model in "Qwen/Qwen2.5-0.5B" "meta-llama/Llama-3.2-1B" "microsoft/phi-2"; do
    msig-translate --model "$model" --target all
  done


----------------------------------------------------------
## OPTION B — GOOGLE COLAB (recommended for TensorRT on T4 GPU)
----------------------------------------------------------

### Step 1 — Open the notebook
  Go to: https://colab.research.google.com
  File → Upload notebook → universalmsig_demo.ipynb

### Step 2 — Set T4 GPU runtime (for TensorRT)
  Runtime → Change runtime type → T4 GPU → Save

### Step 3 — Run Cell 1 (clone + install)
  !git clone https://github.com/YOUR_USERNAME/universalmsig.git
  %cd universalmsig
  !pip install -e .

### Step 4 — Run Cell 2 (optional SDK install)
  !pip install tensorrt    ← real .engine on T4
  !pip install coremltools ← CoreML compilation on Linux

### Step 5 — Run all cells in order (Ctrl+F9)
  Cell 3:  List models
  Cell 4:  Build Qwen2.5-0.5B signature
  Cell 5:  Inspect layer routing
  Cell 6:  Dry run all backends
  Cell 7:  Full translation → 3 backends
  Cell 8:  Inspect TensorRT config
  Cell 9:  Inspect CoreML spec + run MIL graph
  Cell 10: Inspect QNN topology + AI Hub spec
  Cell 11: DeepSeek + Llama translation
  Cell 12: Save/reload .msig file
  Cell 13: Run test suite (51 tests)

### What works on Colab T4:
  ✅ All 51 tests pass
  ✅ TensorRT config JSON (always)
  ✅ TensorRT .engine (if pip install tensorrt succeeds on T4)
  ✅ CoreML spec + MIL graph (always)
  ✅ CoreML .mlpackage compilation (pip install coremltools works on Linux)
  ✅ QNN topology JSON + AI Hub spec (always)
  ⚠  CoreML model execution needs macOS (compile works on Linux)
  ⚠  QNN real hardware needs AI Hub token (see Option D)


----------------------------------------------------------
## OPTION C — GITHUB (push and let CI test automatically)
----------------------------------------------------------

### Step 1 — Create GitHub repo
  Go to https://github.com/new
  Name: universalmsig
  Public, no README (we have our own)

### Step 2 — Push
  cd universalmsig
  git init
  git add .
  git commit -m "feat: universalmsig v0.1.0 — cross-vendor .msig compiler"
  git remote add origin https://github.com/YOUR_USERNAME/universalmsig.git
  git push -u origin main

### Step 3 — Watch CI
  Go to: https://github.com/YOUR_USERNAME/universalmsig/actions

  CI runs 4 jobs automatically:
    test-linux (Ubuntu, Python 3.11)    → 51 tests
    test-linux (Ubuntu, Python 3.12)    → 51 tests
    test-coreml-linux                   → CoreML backend with SDK
    test-macos (macOS-latest, Apple Si) → CoreML compilation + execution
    test-cli                            → All CLI commands

### Step 4 — Add CI badge to README
  Replace YOUR_USERNAME in this badge URL:
  ![CI](https://github.com/YOUR_USERNAME/universalmsig/actions/workflows/ci.yml/badge.svg)


══════════════════════════════════════════════════════════════════════
## OPTION D — TEST ON REAL HARDWARE (free)
══════════════════════════════════════════════════════════════════════

### NVIDIA TensorRT — Google Colab T4 (free)
  1. Open Colab with T4 GPU runtime
  2. !pip install tensorrt
  3. Run Cell 7 — will produce real .engine file
  4. Verify with: import tensorrt as trt; trt.__version__

### Apple CoreML — GitHub Actions macOS runner (free)
  The CI job test-macos runs on macos-latest automatically.
  It tests CoreML compilation + execution on Apple Silicon.
  No Mac needed on your end.

  To test locally on Mac:
    pip install coremltools
    msig-translate --model Qwen/Qwen2.5-0.5B --target coreml
    # Then run the generated *_mil_graph.py:
    python msig_output/coreml/qwen_qwen2.5_0.5b_mil_graph.py

### Qualcomm QNN — AI Hub cloud (free account)
  1. Sign up free at: https://aihub.qualcomm.com
  2. pip install qai-hub
  3. qai-hub configure --api_token YOUR_TOKEN
     OR: export QAI_HUB_API_TOKEN=YOUR_TOKEN
  4. msig-translate --model Qwen/Qwen2.5-0.5B --target qnn --out ./out
  5. The *_aihub_job.json spec is ready — submit it:

     import qai_hub as hub
     import json
     spec = json.load(open("msig_output/qnn/qwen_qwen2.5_0.5b_aihub_job.json"))
     # Upload ONNX export of model + submit to Snapdragon 8 Gen 3
     # See: https://app.aihub.qualcomm.com/docs/

  Supported devices (free tier):
    - Snapdragon 8 Gen 3
    - Snapdragon X Elite
    - Snapdragon 8s Gen 3


----------------------------------------------------------
## PYTHON API (use in your own code)
----------------------------------------------------------

  from universalmsig import MSigTranslator, build_signature, Precision

  # 1. Build a signature
  sig = build_signature(
      "Qwen/Qwen2.5-0.5B",
      precision       = Precision.FP16,
      npu_split_ratio = 0.70,   # 70% GPU, 30% CPU
      max_seq_len     = 4096,
      offline         = True,   # no download
  )
  print(sig.summary())

  # 2. Translate to all backends
  translator = MSigTranslator()
  results = translator.translate_model(
      "Qwen/Qwen2.5-0.5B",
      targets    = ["tensorrt", "coreml", "qnn"],
      output_dir = "./output",
      offline    = True,
  )

  # 3. Translate a single backend
  results = translator.translate_model(
      "meta-llama/Llama-3.2-1B",
      targets    = ["tensorrt"],
      precision  = "int8",
      output_dir = "./output",
  )

  # 4. Dry run (no files)
  plan = translator.dry_run("Qwen/Qwen2.5-0.5B")
  for backend, info in plan["backends"].items():
      print(f"{backend}: {info['fast_layers']} fast layers, {info['weight_gb']} GB")

  # 5. Load from .msig file
  from universalmsig import ModelSignature
  sig = ModelSignature.load_json("model.msig")
  results = translator.translate_signature(sig, targets=["coreml"])

  # 6. Use with live HuggingFace download
  pip install huggingface_hub
  sig = build_signature("Qwen/Qwen2.5-0.5B", offline=False)


----------------------------------------------------------
## SUPPORTED MODELS (offline, no download)
----------------------------------------------------------

  Model                                   Layers  Hidden  Heads  KV  Params
  ──────────────────────────────────────────────────────────────────────────
  Qwen/Qwen2.5-0.5B                         24     896     14    2    0.5B
  Qwen/Qwen2.5-1.5B                         28    1536     12    2    1.5B
  meta-llama/Llama-3.2-1B                   16    2048     32    8    1.2B
  meta-llama/Llama-3.2-3B                   28    3072     24    8    3.2B
  deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B 28    1536     12    2    1.8B
  microsoft/phi-2                            32    2560     32   32    2.8B
  google/gemma-2b                            18    2048      8    1    2.5B

  Any HuggingFace model ID also works:
    pip install huggingface_hub
    msig-translate --model YOUR_ORG/YOUR_MODEL --target all


----------------------------------------------------------
## INSTALL VENDOR SDKs (optional — for real hardware compilation)
----------------------------------------------------------

  # NVIDIA TensorRT (requires NVIDIA GPU + CUDA 12.x)
  pip install tensorrt

  # Apple CoreML (works on Linux for compilation; macOS for execution)
  pip install coremltools

  # Qualcomm AI Hub (requires free account)
  pip install qai-hub
  export QAI_HUB_API_TOKEN=your_token_here

  # HuggingFace (to download real model configs)
  pip install huggingface_hub

  # All at once:
  pip install -e ".[all]"           # coremltools + huggingface_hub
  pip install -e ".[tensorrt]"      # tensorrt
  pip install -e ".[qnn]"           # qai-hub


----------------------------------------------------------
## TROUBLESHOOTING
----------------------------------------------------------

  Q: msig-translate not found after pip install -e .
  A: Make sure you're in the right directory:
     cd universalmsig && pip install -e .

  Q: Tests fail with ModuleNotFoundError
  A: Run from the repo root:
     cd universalmsig
     python tests/test_universalmsig.py -v

  Q: tensorrt import fails
  A: TensorRT needs an NVIDIA GPU + CUDA. Use Google Colab T4 runtime.
     The library still produces config JSONs without TensorRT installed.

  Q: CoreML .mlpackage not produced
  A: pip install coremltools
     CoreML spec JSON + MIL graph script are always produced even without it.

  Q: QNN topology produced but how do I submit to AI Hub?
  A: Sign up at https://aihub.qualcomm.com (free)
     export QAI_HUB_API_TOKEN=your_token
     The *_aihub_job.json has step-by-step instructions inside it.

  Q: I want to add a new model
  A: Add an entry to universalmsig/core/parser.py OFFLINE_SPECS dict:
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
     Then open a PR.

  Q: I want to add a new backend (e.g. ONNX Runtime, llama.cpp)
  A: Subclass BaseBackend in universalmsig/backends/
     Implement: name, supported_precisions, validate(), compile()
     Register in universalmsig/translator.py MSigTranslator.__init__


══════════════════════════════════════════════════════════════════════
## TEST RESULTS SUMMARY
══════════════════════════════════════════════════════════════════════

  See TEST_RESULTS.txt for the full output.

  51 tests — ALL PASS
  Suite breakdown:
    ModelSignature (core format)    15 tests  PASS
    TensorRT backend                10 tests  PASS
    CoreML backend                   9 tests  PASS
    QNN backend                     10 tests  PASS
    MSigTranslator (routing engine)  7 tests  PASS

  7 models × 3 backends = 21 combinations tested — all SUCCESS
  Runtime: 0.163 seconds (fully offline)

## License
This project is licensed under the [Apache License 2.0](https://github.com/ashwin9390/universalmsig/blob/main/LICENSE).
