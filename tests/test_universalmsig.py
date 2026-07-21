"""
tests/test_universalmsig.py

Full test suite — runs entirely offline, no GPU, no SDK needed.
All vendor SDKs (tensorrt, coremltools, qai_hub) are optional.

Run:  python tests/test_universalmsig.py -v
      python -m pytest tests/ -v
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


# ─────────────────────────────────────────────────────────────────────────────
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
        version, layers, hidden, heads, kv, bpl = struct.unpack("=4sIIIIQ", b)
        self.assertEqual(version, b"2.00")
        self.assertEqual(layers, 24)
        self.assertEqual(hidden, 896)

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


# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
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
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            self.assertTrue(result.success)
            self.assertTrue(os.path.exists(result.output_path))

    def test_coreml_spec_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            with open(result.output_path) as f:
                spec = json.load(f)
            self.assertIn("model_description", spec)
            self.assertIn("msig_layer_routing", spec)
            self.assertIn("quantization", spec)

    def test_mil_script_produced(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.compile(self.sig, tmp)
            safe = self.sig.model_id.replace("/","_").replace("-","_").lower()
            mil  = os.path.join(tmp, f"{safe}_mil_graph.py")
            self.assertTrue(os.path.exists(mil))
            content = open(mil).read()
            self.assertIn("coremltools", content)
            self.assertIn("mb.program", content)

    def test_gqa_unrolling_in_mil_script(self):
        """
        BUG FIX: CoreML MIL does not natively handle mismatched Q/KV head
        dimensions. For Qwen GQA (14 Q, 2 KV), the MIL script must explicitly
        broadcast KV heads 2→14 using mb.tile before attention matmul.
        """
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.compile(self.sig, tmp)
            safe = self.sig.model_id.replace("/","_").replace("-","_").lower()
            mil  = os.path.join(tmp, f"{safe}_mil_graph.py")
            content = open(mil).read()
            self.assertIn("mb.tile", content,
                          "GQA broadcast via mb.tile must be present in MIL script")
            self.assertIn("GQA UNROLLING", content,
                          "GQA unrolling comment must be present")
            self.assertIn("GQA_RATIO", content,
                          "GQA_RATIO constant must be defined")
            # Qwen ratio = 14/2 = 7 — the tile must use 7
            self.assertIn("reps=[1, 7, 1, 1]", content,
                          "mb.tile must broadcast by factor 7 for Qwen GQA")

    def test_ane_cpu_routing_in_spec(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            with open(result.output_path) as f:
                spec = json.load(f)
            routing = spec["msig_layer_routing"]
            self.assertIn("ane_layers", routing)
            self.assertIn("cpu_layers", routing)
            self.assertGreater(len(routing["ane_layers"]), 0)


# ─────────────────────────────────────────────────────────────────────────────
class TestQNNBackend(unittest.TestCase):

    def setUp(self):
        self.backend = QNNBackend()
        self.sig     = build_signature("Qwen/Qwen2.5-0.5B", offline=True)

    def test_name(self):
        self.assertEqual(self.backend.name, "qnn")

    def test_validate_warns_layout(self):
        w = self.backend.validate(self.sig)
        self.assertTrue(any("layout" in warn.lower() or "nhwc" in warn.lower()
                            for warn in w))

    def test_compile_produces_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            self.assertTrue(result.success)
            self.assertTrue(os.path.exists(result.output_path))

    def test_qnn_topology_valid_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            with open(result.output_path) as f:
                topo = json.load(f)
            self.assertIn("graph", topo)
            self.assertIn("nodes", topo["graph"])
            self.assertGreater(len(topo["graph"]["nodes"]), 0)

    def test_qnn_quant_profile_produced(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.compile(self.sig, tmp)
            safe  = self.sig.model_id.replace("/","_").replace("-","_").lower()
            quant = os.path.join(tmp, f"{safe}_qnn_quant_profile.json")
            self.assertTrue(os.path.exists(quant))
            with open(quant) as f:
                q = json.load(f)
            self.assertIn("quant_scheme", q)
            self.assertIn("layers", q)

    def test_aihub_job_spec_produced(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.backend.compile(self.sig, tmp)
            safe  = self.sig.model_id.replace("/","_").replace("-","_").lower()
            hub   = os.path.join(tmp, f"{safe}_aihub_job.json")
            self.assertTrue(os.path.exists(hub))
            with open(hub) as f:
                j = json.load(f)
            self.assertIn("target_devices", j)
            self.assertIn("instructions", j)

    def test_htp_cpu_split_in_topology(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            with open(result.output_path) as f:
                topo = json.load(f)
            routing = topo["msig_layer_routing"]
            self.assertIn("htp_blocks", routing)
            self.assertIn("cpu_blocks", routing)
            self.assertGreater(routing["htp_blocks"], 0)
            # 17 + 7 = 24 for Qwen 0.5B — counts must be blocks and add up
            self.assertEqual(routing["htp_blocks"] + routing["cpu_blocks"],
                             self.sig.total_layers)

    def test_gqa_broadcast_nodes_present_for_qwen(self):
        """
        BUG FIX: Qwen uses GQA (14 Q heads, 2 KV heads).
        QNN topology must contain explicit Tile broadcast nodes
        to expand KV heads 2→14 before attention matmul.
        QNN HTP cannot implicitly broadcast mismatched head dimensions.
        """
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            with open(result.output_path) as f:
                topo = json.load(f)
            nodes = topo["graph"]["nodes"]
            broadcast_nodes = [n for n in nodes if n.get("typeName") == "Tile"]
            # 24 layers × 2 (K + V) = 48 broadcast nodes
            self.assertEqual(
                len(broadcast_nodes), 48,
                f"Expected 48 GQA broadcast nodes (24 K + 24 V), got {len(broadcast_nodes)}"
            )

    def test_gqa_broadcast_ratio_correct(self):
        """Each broadcast node must tile by ratio = num_heads / num_kv_heads = 7."""
        with tempfile.TemporaryDirectory() as tmp:
            result = self.backend.compile(self.sig, tmp)
            with open(result.output_path) as f:
                topo = json.load(f)
            nodes = topo["graph"]["nodes"]
            broadcast_nodes = [n for n in nodes if n.get("typeName") == "Tile"]
            if broadcast_nodes:
                multiples = broadcast_nodes[0]["params"]["multiples"]
                # Should be [1, 1, 7, 1] for Qwen (ratio 14/2 = 7)
                self.assertEqual(multiples[2], 7,
                                 f"Expected GQA ratio 7, got {multiples[2]}")


# ─────────────────────────────────────────────────────────────────────────────
class TestMSigTranslator(unittest.TestCase):

    def setUp(self):
        self.translator = MSigTranslator()

    def test_available_backends(self):
        backs = self.translator.available_backends
        self.assertIn("tensorrt", backs)
        self.assertIn("coreml", backs)
        self.assertIn("qnn", backs)

    def test_translate_model_all_backends(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = self.translator.translate_model(
                "Qwen/Qwen2.5-0.5B",
                targets=None,  # all
                output_dir=tmp,
                offline=True,
            )
            self.assertEqual(len(results), 3)
            self.assertTrue(all(r.success for r in results))

    def test_translate_model_single_backend(self):
        with tempfile.TemporaryDirectory() as tmp:
            results = self.translator.translate_model(
                "meta-llama/Llama-3.2-1B",
                targets=["tensorrt"],
                output_dir=tmp,
                offline=True,
            )
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0].backend_name, "tensorrt")

    def test_translate_from_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            msig_path = os.path.join(tmp, "model.msig")
            sig = build_signature("Qwen/Qwen2.5-0.5B", offline=True)
            sig.save_json(msig_path)

            results = self.translator.translate_file(
                msig_path, targets=["coreml"], output_dir=tmp
            )
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0].success)

    def test_dry_run_returns_dict(self):
        plan = self.translator.dry_run("Qwen/Qwen2.5-0.5B", offline=True)
        self.assertIn("backends", plan)
        self.assertIn("signature", plan)
        for name in ["tensorrt", "coreml", "qnn"]:
            self.assertIn(name, plan["backends"])

    def test_unknown_backend_raises(self):
        with self.assertRaises(ValueError):
            self.translator.translate_model(
                "Qwen/Qwen2.5-0.5B",
                targets=["banana"],
                offline=True,
            )

    def test_save_signature(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.msig")
            sig  = self.translator.save_signature("Qwen/Qwen2.5-0.5B", path)
            self.assertTrue(os.path.exists(path))
            self.assertEqual(sig.total_layers, 24)

    def test_list_supported_models(self):
        models = list_supported_models()
        self.assertGreater(len(models), 3)
        self.assertIn("Qwen/Qwen2.5-0.5B", models)

    def test_split_boundary_consistent_across_backends(self):
        """Every artifact must agree with the signature: 17 fast blocks for
        Qwen 0.5B (ceil(24*0.7)), and QNN layer 16 on HTP like the signature
        says — previously int() truncation put it on CPU."""
        sig = build_signature("Qwen/Qwen2.5-0.5B", offline=True)
        with tempfile.TemporaryDirectory() as tmp:
            TensorRTBackend().compile(sig, tmp)
            QNNBackend().compile(sig, tmp)
            safe = "qwen_qwen2.5_0.5b"
            with open(os.path.join(tmp, f"{safe}_trtllm_config.json")) as f:
                trtllm = json.load(f)
            with open(os.path.join(tmp, f"{safe}_qnn_topology.json")) as f:
                topo = json.load(f)
        self.assertEqual(trtllm["msig_layer_split"]["gpu_boundary_block"], 17)
        self.assertEqual(trtllm["msig_layer_split"]["cpu_offload_blocks"], 7)
        self.assertEqual(topo["msig_layer_routing"]["htp_blocks"], 17)
        boundary_attn = [n for n in topo["graph"]["nodes"]
                         if n["name"] == "layer_16_self_attn"][0]
        self.assertEqual(boundary_attn["backendConfig"]["engine"],
                         "QNN_BACKEND_HTP",
                         "layer 16 is gpu_fast in the signature; QNN must agree")

    def test_all_models_all_backends(self):
        """Every offline model should compile to every backend without error."""
        models = list_supported_models()
        with tempfile.TemporaryDirectory() as tmp:
            for model_id in models:
                with self.subTest(model=model_id):
                    results = self.translator.translate_model(
                        model_id, output_dir=tmp, offline=True
                    )
                    self.assertEqual(len(results), 3)
                    failed = [r for r in results if not r.success]
                    self.assertEqual(failed, [],
                                     f"Failed backends for {model_id}: "
                                     f"{[r.backend_name for r in failed]}")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    verbosity = 2 if "-v" in sys.argv else 1
    unittest.main(verbosity=verbosity)
