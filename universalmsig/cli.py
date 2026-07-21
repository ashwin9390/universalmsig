"""
universalmsig/cli.py

CLI entry point: msig-translate

Usage:
  msig-translate --model Qwen/Qwen2.5-0.5B --target all
  msig-translate --model meta-llama/Llama-3.2-1B --target tensorrt --precision fp16
  msig-translate --model Qwen/Qwen2.5-0.5B --target coreml --out ./output
  msig-translate --file model.msig --target qnn
  msig-translate --dry-run --model Qwen/Qwen2.5-0.5B
  msig-translate --list-models
  msig-translate --save-msig Qwen/Qwen2.5-0.5B model.msig
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .translator import MSigTranslator, list_supported_models
from .core.parser import OFFLINE_SPECS


def main():
    parser = argparse.ArgumentParser(
        prog="msig-translate",
        description=(
            "Universal Model Signature (.msig) cross-vendor compiler.\n"
            "Translates a unified model signature to TensorRT, CoreML, or QNN assets."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compile Qwen2.5-0.5B to all three backends (offline, no download)
  msig-translate --model Qwen/Qwen2.5-0.5B --target all

  # Compile to TensorRT only, FP16
  msig-translate --model meta-llama/Llama-3.2-1B --target tensorrt --precision fp16

  # Compile from a saved .msig file
  msig-translate --file model.msig --target coreml

  # Dry run — describe what would be compiled, no files written
  msig-translate --dry-run --model Qwen/Qwen2.5-0.5B

  # Save .msig JSON for a model
  msig-translate --save-msig Qwen/Qwen2.5-0.5B qwen.msig

  # List all supported offline models
  msig-translate --list-models
        """,
    )

    parser.add_argument("--model",     help="HuggingFace model ID (offline supported)")
    parser.add_argument("--file",      help="Path to saved .msig JSON file")
    parser.add_argument(
        "--target",
        default="all",
        choices=["all", "tensorrt", "coreml", "qnn"],
        help="Backend target (default: all)",
    )
    parser.add_argument(
        "--precision",
        default="fp16",
        choices=["fp32", "fp16", "bf16", "int8", "int4", "fp4"],
        help="Weight precision (default: fp16)",
    )
    parser.add_argument("--out",        default="./msig_output", help="Output directory")
    parser.add_argument("--split",      type=float, default=0.70,
                        help="NPU/GPU split ratio 0.0–1.0 (default: 0.70)")
    parser.add_argument("--max-seq",    type=int, default=4096,
                        help="Max sequence length for KV-cache sizing (default: 4096)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Describe compilation plan without writing files")
    parser.add_argument("--save-msig",  nargs=2, metavar=("MODEL_ID", "PATH"),
                        help="Save .msig JSON for MODEL_ID to PATH")
    parser.add_argument("--online",      action="store_true",
                        help="Fetch model config from HuggingFace Hub "
                             "(requires: pip install huggingface_hub). "
                             "Without this flag only the built-in offline models work.")
    parser.add_argument("--list-models", action="store_true",
                        help="List all supported offline models")
    parser.add_argument("--json-output", action="store_true",
                        help="Output results as JSON")

    args = parser.parse_args()
    translator = MSigTranslator()

    # ── list-models ───────────────────────────────────────────────────────────
    if args.list_models:
        models = list_supported_models()
        print("\nSupported offline models (no HF download needed):\n")
        for m in models:
            spec = OFFLINE_SPECS[m]
            print(f"  {m}")
            print(f"    layers={spec['num_hidden_layers']}  "
                  f"hidden={spec['hidden_size']}  "
                  f"heads={spec['num_attention_heads']} "
                  f"(kv={spec['num_key_value_heads']})")
        print("\nAny other HF model ID works with --online (requires: pip install huggingface_hub)")
        return

    # ── save-msig ─────────────────────────────────────────────────────────────
    if args.save_msig:
        model_id, path = args.save_msig
        print(f"\nBuilding signature for {model_id} …")
        sig = translator.save_signature(model_id, path, precision=args.precision,
                                        offline=not args.online)
        print(sig.summary())
        print(f"\nSaved → {path}")
        return

    # ── dry-run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        if not args.model:
            print("ERROR: --dry-run requires --model")
            sys.exit(1)
        targets = None if args.target == "all" else [args.target]
        try:
            plan = translator.dry_run(args.model, targets=targets, precision=args.precision,
                                      offline=not args.online)
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)
        if args.json_output:
            print(json.dumps(plan, indent=2))
        else:
            print("\n" + "=" * 66)
            print("  DRY RUN — No files written")
            print("=" * 66)
            print(plan["signature"])
            print()
            for backend, info in plan["backends"].items():
                print(f"  [{backend.upper()}]")
                print(f"    Would produce : {info['would_produce']}")
                print(f"    Fast layers   : {info['fast_layers']}")
                print(f"    CPU layers    : {info['cpu_layers']}")
                print(f"    Weight        : {info['weight_gb']} GB")
                print(f"    KV-cache      : {info['kv_cache_mb']} MB")
                for w in info.get("warnings", []):
                    print(f"    ⚠  {w}")
                print()
        return

    # ── translate ─────────────────────────────────────────────────────────────
    if not args.model and not args.file:
        parser.print_help()
        sys.exit(1)

    targets = None if args.target == "all" else [args.target]

    print("\n" + "=" * 66)
    print("  Universal .msig Translator")
    print("=" * 66)

    if args.file:
        results = translator.translate_file(
            args.file, targets=targets, output_dir=args.out
        )
    else:
        try:
            results = translator.translate_model(
                model_id        = args.model,
                targets         = targets,
                output_dir      = args.out,
                precision       = args.precision,
                npu_split_ratio = args.split,
                max_seq_len     = args.max_seq,
                offline         = not args.online,
            )
        except ValueError as e:
            print(f"ERROR: {e}")
            sys.exit(1)

    print("\n" + "=" * 66)
    print("  Translation Complete")
    print("=" * 66)
    ok  = sum(1 for r in results if r.success)
    err = sum(1 for r in results if not r.success)
    print(f"  ✅ {ok} backend(s) succeeded")
    if err:
        print(f"  ❌ {err} backend(s) failed")
    print(f"  Output directory: {args.out}")

    if args.json_output:
        out = [
            {
                "backend":    r.backend_name,
                "success":    r.success,
                "asset_type": r.asset_type,
                "output":     r.output_path,
                "warnings":   r.warnings,
                "metadata":   r.metadata,
            }
            for r in results
        ]
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
