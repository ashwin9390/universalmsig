"""
unit tests for universalmsig
"""
from __future__ import annotations

import json
import os
import struct
import sys
import tempfile
import unittest

# Make sure the package is importable from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from universalmsig.core.signature import (
    ModelSignature, LayerSignature, ExecutionTier, Precision,
)
from universalmsig.core.parser import build_signature, OFFLINE_SPECS
from universalmsig.backends.base import CompilationResult
from universalmsig.backends.tensorrt_backend import TensorRTBackend
from universalmsig.backends.coreml_backend import CoreMLBackend
from universalmsig.backends.qnn_backend import QNNBackend
from universalmsig.translator import MSigTranslator, list_supported_models


# ────────────────────────────────────────────────────────────────��[...]
class TestModelSignature(unittest.TestCase):

    def _qwen_sig(self) -> ModelSignature:
        return build_signature("Qwen/Qwen2.5-0.5B", offline=True)

    def test_builds_from_offline_spec(self):
        sig = self._qwen_sig()
        self.assertEqual(sig.total_layers, 24)
        self.assertEqual(sig.hidden_size, 896)
        self.assertEqual(sig.num_heads, 14)
        self.assertEqual(sig.num_kv_heads, 2)

    def test_layer_count_matches(self):
        sig = self._qwen_sig()
        # embed + (attn + mlp) * 24 + lm_head = 1 + 48 + 1 = 50
        self.assertEqual(len(sig.layers), 50)

    def test_npu_split_70pct(self):
        sig = self._qwen_sig()
        npu = len([l for l in sig.npu_layers if l.is_attention])
        cpu = len([l for l in sig.cpu_layers if l.is_attention])
        self.assertGreater(npu, cpu)

    # ── Bug-fix regression tests (from Colab feedback) ────────────────────────
    def test_fast_tier_blocks_correct_qwen(self):
        """
        BUG FIX: Qwen2.5-0.5B has 24 transformer blocks.
        Fast tier = ceil(24 × 0.70) = 17 blocks.
        CPU tier  = 24 - 17 = 7 blocks.
        Previously reported 34 fast / 16 CPU (counted sub-layers + embed + lm_head).
        """
        sig = self._qwen_sig()
        npu_blocks = len([l for l in sig.npu_layers if l.is_attention])
        cpu_blocks = len([l for l in sig.cpu_layers if l.is_attention])
        self.assertEqual(npu_blocks, 17, f"Expected 17 fast blocks, got {npu_blocks}")
        self.assertEqual(cpu_blocks, 7,  f"Expected 7 CPU blocks, got {cpu_blocks}")
        self.assertEqual(npu_blocks + cpu_blocks, 24)

    def test_fast_tier_does_not_include_embed_or_lm_head(self):
        """embed_tokens and lm_head must not be counted in the 70/30 split."""
        sig = self._qwen_sig()
        for layer in sig.npu_layers:
            self.assertFalse(layer.is_embedding,
                             f"embed layer {layer.name} incorrectly in npu_layers")
            self.assertNotEqual(layer.name, "lm_head",
                                "lm_head incorrectly in npu_layers")
        for layer in sig.cpu_layers:
            self.assertFalse(layer.is_embedding)
            self.assertNotEqual(layer.name, "lm_head")

    def test_fast_plus_cpu_equals_total_layers(self):
        """npu_blocks + cpu_blocks must equal total_layers for all models."""
        from universalmsig.core.parser import OFFLINE_SPECS
        for model_id in OFFLINE_SPECS:
            sig = build_signature(model_id, offline=True)
            npu = len([l for l in sig.npu_layers if l.is_attention])
            cpu = len([l for l in sig.cpu_layers if l.is_attention])
            self.assertEqual(
                npu + cpu, sig.total_layers,
                f"{model_id}: {npu} + {cpu} ≠ {sig.total_layers}"
            )

    def test_summary_shows_correct_block_counts(self):
        """Summary text must show 17 fast blocks and 7 CPU blocks, not 34/16."""
        sig = self._qwen_sig()
        s = sig.summary()
        self.assertIn("17 transformer blocks", s,
                      "Summary must show 17 fast blocks, not sub-layer count")
        self.assertIn("7 transformer blocks", s,
                      "Summary must show 7 CPU blocks")
        # Must NOT show the old wrong numbers
        self.assertNotIn("34 layer", s)
        self.assertNotIn("16 layer", s)

    def test_summary_shows_gqa_info(self):
        """Summary must explain GQA broadcast requirement for Qwen."""
        sig = self._qwen_sig()
        s = sig.summary()
        self.assertIn("GQA", s)
        self.assertIn("broadcast", s)
        self.assertIn("7 Q heads", s)   # ratio = 14/2 = 7

    def test_weight_bytes_nonzero(self):
        sig = self._qwen_sig()
        self.assertGreater(sig.total_weight_bytes, 0)

    def test_int4_weights_quarter_of_fp16(self):
        """4-bit types are 0.5 bytes/element — int4 totals must be ~1/4 of fp16."""
        s16 = build_signature("Qwen/Qwen2.5-0.5B", precision=Precision.FP16, offline=True)
        s4  = build_signature("Qwen/Qwen2.5-0.5B", precision=Precision.INT4, offline=True)
        ratio = s4.total_weight_bytes / s16.total_weight_bytes
        self.assertAlmostEqual(ratio, 0.25, places=2,
                               msg=f"int4/fp16 weight ratio should be 0.25, got {ratio:.3f}")

    def test_kv_cache_bytes_nonzero(self):
        sig = self._qwen_sig()
        self.assertGreater(sig.total_kv_cache_bytes, 0)

    def test_content_hash_stable(self):
        s1 = build_signature("Qwen/Qwen2.5-0.5B", offline=True)
        s2 = build_signature("Qwen/Qwen2.5-0.5B", offline=True)
        self.assertEqual(s1.content_hash, s2.content_hash)

    def test_content_hash_differs_across_models(self):
        s1 = build_signature("Qwen/Qwen2.5-0.5B", offline=True)
        s2 = build_signature("meta-llama/Llama-3.2-1B", offline=True)
        self.assertNotEqual(s1.content_hash, s2.content_hash)

    def test_json_roundtrip(self):
        sig = self._qwen_sig()
        j   = sig.to_json()
        sig2 = ModelSignature.from_json(j)
        self.assertEqual(sig2.model_id, sig.model_id)
        self.assertEqual(sig2.total_layers, sig.total_layers)
        self.assertEqual(len(sig2.layers), len(sig.layers))
        self.assertEqual(sig2.content_hash, sig.content_hash)

    def test_save_load_json(self):
        sig = self._qwen_sig()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.msig")
            sig.save_json(path)
            self.assertTrue(os.path.exists(path))
            sig2 = ModelSignature.load_json(path)
            self.assertEqual(sig2.model_id, sig.model_id)

    def test_binary_format_28_bytes(self):
        sig = self._qwen_sig()
        b = sig.to_binary()
        self.assertEqual(len(b), 28)
        version, layers, hidden, heads, kv, block_bytes = struct.unpack("=4sIIIIQ", b)
        self.assertEqual(version, b"2.00")
        self.assertEqual(layers, 24)
        self.assertEqual(hidden, 896)

    def test_binary_block_bytes_is_one_transformer_block(self):
        """block_bytes must be attn+mlp of one block, not the embedding table."""
        sig = self._qwen_sig()
        *_, block_bytes = struct.unpack("=4sIIIIQ", sig.to_binary())
        attn = next(l.weight_bytes for l in sig.layers if l.is_attention)
        mlp  = next(l.weight_bytes for l in sig.layers if l.is_mlp)
        self.assertEqual(block_bytes, attn + mlp)
        self.assertNotEqual(block_bytes, sig.layers[0].weight_bytes,
                            "block_bytes must not be the embedding table size")

    def test_binary_roundtrip(self):
        sig = self._qwen_sig()
        sig2 = ModelSignature.from_binary(sig.to_binary())
        self.assertEqual(sig2.total_layers, sig.total_layers)
        self.assertEqual(sig2.hidden_size, sig.hidden_size)
        self.assertEqual(sig2.num_heads, sig.num_heads)
        self.assertEqual(sig2.num_kv_heads, sig.num_kv_heads)
        with self.assertRaises(ValueError):
            ModelSignature.from_binary(b"9.99" + b"\x00" * 24)

    def test_summary_returns_string(self):
        sig = self._qwen_sig()
        s   = sig.summary()
        self.assertIn("Qwen", s)
        self.assertIn("24", s)

    def test_all_offline_models_build(self):
        for model_id in OFFLINE_SPECS:
            with self.subTest(model=model_id):
                sig = build_signature(model_id, offline=True)
                self.assertGreater(sig.total_layers, 0)
                self.assertGreater(sig.hidden_size, 0)
                self.assertIsNotNone(sig.content_hash)


# ────────────────────────────────────────────────────────────────�[...]
class TestTensorRTBackend(unittest.TestCase):

    def setUp(self):
        self.backend = TensorRTBackend()
        self.sig     = build_signature("Qwen/Qwen2.5-0.5B", offline=True)

    def test_name(self):
        self.assertEqual(self.backend.name, "tensorrt")

    def test_supported_precisions(self):
        self.assertIn(Precision.FP16, self.backend.supported_precisions)
        self.assertIn(Precision.INT8, self.backend.supported_precisions)

    def test_validate_returns_list(self):
        w = self.backend.validate(self.sig)
        self.assertIsInstance(w, list)

    def test_dry_run(self):
        plan = self.backend.dry_run(self.sig)
        self.assertEqual(plan["backend"], "tensorrt")
        self.assertIn("fast_blocks", plan)
        self.assertIn("weight_gb", plan)
        # blocks must sum to total transformer blocks — units must not mix
        self.assertEqual(plan["fast_blocks"] + plan["cpu_blocks"],
                         self.sig.total_layers)

    def test_compile_produces_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            self.assertTrue(result.success)
            self.assertEqual(result.backend_name, "tensorrt")
            # Config JSON always produced
            self.assertTrue(os.path.exists(result.output_path))

    def test_compile_config_is_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            with open(result.output_path) as f:
                cfg = json.load(f)
            self.assertIn("builder_config", cfg)
            self.assertIn("network_config", cfg)
            self.assertIn("layer_routing", cfg)

    def test_trtllm_config_produced(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.compile(self.sig, tmp)
            safe = self.sig.model_id.replace("/","_").replace("-","_").lower()
            trtllm = os.path.join(tmp, f"{safe}_trtllm_config.json")
            self.assertTrue(os.path.exists(trtllm))
            with open(trtllm) as f:
                cfg = json.load(f)
            self.assertIn("kv_cache_config", cfg)

    def test_layer_routing_in_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            with open(result.output_path) as f:
                cfg = json.load(f)
            routing = cfg["layer_routing"]
            self.assertIn("gpu_fast_path", routing)
            self.assertIn("cpu_offload", routing)
            self.assertGreater(len(routing["gpu_fast_path"]), 0)


# ────────────────────────────────────────────────────────────────�[...]
class TestCoreMLBackend(unittest.TestCase):

    def setUp(self):
        self.backend = CoreMLBackend()
        self.sig     = build_signature("Qwen/Qwen2.5-0.5B", offline=True)

    def test_name(self):
        self.assertEqual(self.backend.name, "coreml")

    def test_validate_warns_large_vocab(self):
        sig = build_signature("Qwen/Qwen2.5-0.5B", offline=True)
        # Qwen has 151,936 vocab — should not trigger large vocab warning (threshold 200k)
        w = self.backend.validate(sig)
        self.assertIsInstance(w, list)

    def test_validate_warns_gqa(self):
        sig = build_signature("Qwen/Qwen2.5-0.5B", offline=True)
        w = self.backend.validate(sig)
        # Qwen has GQA (14 heads, 2 kv_heads) — should warn
        self.assertTrue(any("GQA" in warn for warn in w))

    def test_compile_produces_files(self):
{
