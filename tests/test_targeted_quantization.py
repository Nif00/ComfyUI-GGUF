import unittest
from collections import OrderedDict
import importlib.util
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import gguf
import numpy as np
import torch
import comfy.sd
from safetensors.torch import save_file

from tools.convert import (
    MEBIBYTE,
    ModelMinimaxH3,
    ModelTemplate,
    convert_file,
    detect_arch,
    plan_target_size_quantization,
)


def load_gguf_loader():
    loader_path = Path(__file__).parents[1] / "loader.py"
    package_name = "comfyui_gguf_test"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.loader",
        loader_path,
        submodule_search_locations=[str(loader_path.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[package_name] = module
    sys.modules[f"{package_name}.loader"] = module
    spec.loader.exec_module(module)
    return module


class TargetSizeQuantizationTests(unittest.TestCase):
    def setUp(self):
        self.model_arch = ModelTemplate()
        self.state_dict = OrderedDict(
            (f"blocks.{index}.weight", torch.ones((4096, 32), dtype=torch.float32))
            for index in range(3)
        )
        self.state_dict["normalization.weight"] = torch.ones((4096,), dtype=torch.float32)

    def test_reduces_center_core_layers_before_outer_layers(self):
        plan, _, selected_size = plan_target_size_quantization(
            self.state_dict, self.model_arch, 0.38
        )

        self.assertEqual(plan["blocks.1.weight"], gguf.GGMLQuantizationType.Q4_0)
        self.assertEqual(plan["blocks.0.weight"], gguf.GGMLQuantizationType.I8)
        self.assertEqual(plan["blocks.2.weight"], gguf.GGMLQuantizationType.I8)
        self.assertLessEqual(selected_size, int(0.38 * MEBIBYTE))

    def test_reduces_one_dimensional_weights_only_after_all_core_layers(self):
        plan, _, selected_size = plan_target_size_quantization(
            self.state_dict, self.model_arch, 0.222
        )

        for index in range(3):
            self.assertEqual(plan[f"blocks.{index}.weight"], gguf.GGMLQuantizationType.Q4_0)
        self.assertEqual(plan["normalization.weight"], gguf.GGMLQuantizationType.BF16)
        self.assertLessEqual(selected_size, int(0.222 * MEBIBYTE))

    def test_reports_minimum_when_target_is_unsupported(self):
        with self.assertRaisesRegex(ValueError, "smallest supported TARGET_SIZE output"):
            plan_target_size_quantization(self.state_dict, self.model_arch, 0.1)


class Qwen3VLDetectionMarkerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    def test_uses_minimax_32b_detection_marker_for_5120_hidden_size(self):
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.zeros(5120),
            "model.layers.49.self_attn.q_proj.weight": torch.zeros(1),
        }

        self.loader.inject_qwen3vl_detection_markers(state_dict)

        self.assertEqual(
            comfy.sd.detect_te_model(state_dict),
            comfy.sd.TEModel.QWEN3VL_32B,
        )
        self.assertIn("visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.merger.linear_fc2.weight", state_dict)
        self.assertEqual(
            state_dict["visual.deepstack_merger_list.0.norm.weight"].shape,
            (4608,),
        )

    def test_uses_model_prefixed_detection_markers_for_8b(self):
        state_dict = {
            "model.layers.0.input_layernorm.weight": torch.zeros(4096),
        }

        self.loader.inject_qwen3vl_detection_markers(state_dict)

        self.assertIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertEqual(
            state_dict["model.visual.merger.linear_fc2.weight"].shape,
            (4096, 4608),
        )


class Qwen35LoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    def test_maps_hybrid_attention_keys_for_comfy_detection(self):
        state_dict = {
            "blk.0.attn_norm.weight": torch.zeros(4096),
            "blk.0.attn_qkv.weight": torch.zeros(8192, 4096),
            "blk.0.ssm_a": -torch.ones(32),
            "blk.0.post_attention_norm.weight": torch.zeros(4096),
        }

        state_dict = self.loader.sd_map_replace(state_dict, self.loader.QWEN35_SD_MAP)

        self.assertIn("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", state_dict)
        self.assertIn("model.language_model.layers.0.linear_attn.A_log", state_dict)
        self.assertEqual(comfy.sd.detect_te_model(state_dict), comfy.sd.TEModel.QWEN35_9B)

    def test_reverses_llamacpp_value_head_tiling(self):
        grouped = torch.arange(2 * 3 * 2).reshape(2, 3, 2)
        tiled = grouped.permute(1, 0, 2).contiguous().reshape(-1)
        transform = (("qwen35_inverse_v_heads", 0, 0, 2, 3, 2),)

        restored = self.loader.apply_tensor_postprocess(tiled, transform)

        self.assertTrue(torch.equal(restored, grouped.reshape(-1)))

    def test_folds_add_one_norms_once(self):
        # ComfyUI's checkpoint norms add one to their weight every forward; the
        # loader bakes that in so the per-step temporary and kernel disappear.
        import comfy.text_encoders.llama as llama_enc

        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.input_layernorm = llama_enc.RMSNorm(8, eps=1e-5, add=True)
                self.plain = llama_enc.RMSNorm(8, eps=1e-5, add=False)

        block = Block()
        with torch.no_grad():
            block.input_layernorm.weight.copy_(torch.linspace(0.5, 1.5, 8))
            block.plain.weight.copy_(torch.linspace(-0.5, 0.5, 8))
        x = torch.randn(2, 8)
        before = block.input_layernorm(x)
        original_weight = block.input_layernorm.weight
        original_values = original_weight.detach().clone()
        folded_weight = original_weight + 1.0

        self.loader.fold_add_one_norms(block)

        self.assertIs(block.input_layernorm.add, False)
        self.assertIs(block.plain.add, False)
        self.assertTrue(torch.equal(block.input_layernorm.weight, folded_weight))
        self.assertTrue(torch.equal(block.input_layernorm(x), before))
        self.assertTrue(torch.equal(block.plain(x), block.plain(x)))
        # the weight is replaced, never written through: GGUF weights can be
        # views of the read-only file mapping
        self.assertIsNot(block.input_layernorm.weight, original_weight)
        self.assertTrue(torch.equal(original_weight.detach(), original_values))

        # a repeat call must not add one twice
        self.loader.fold_add_one_norms(block)
        self.assertTrue(torch.equal(block.input_layernorm.weight, folded_weight))

    def test_compile_trunk_is_opt_in(self):
        # Off by default, and a compiler failure must leave the model eager.
        model = torch.nn.Linear(4, 4)
        eager_forward = model.forward

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COMFYUI_GGUF_COMPILE_TRUNK", None)
            self.assertIsNone(self.loader.compile_gguf_trunk(model))

        with mock.patch.dict(os.environ, {"COMFYUI_GGUF_COMPILE_TRUNK": "1"}), \
             mock.patch.object(self.loader.torch, "compile", side_effect=RuntimeError("no compiler")):
            self.assertIsNone(self.loader.compile_gguf_trunk(model))
        self.assertEqual(model.forward, eager_forward, "failed compile must restore the eager forward")

    def test_reverses_norm_and_a_log_conversion(self):
        state_dict = {
            "model.language_model.layers.0.input_layernorm.weight": torch.tensor([1.25]),
            "model.language_model.layers.0.linear_attn.norm.weight": torch.tensor([0.5]),
            "model.language_model.layers.0.linear_attn.A_log": torch.tensor([-torch.exp(torch.tensor(2.0))]),
            "model.language_model.layers.0.linear_attn.conv1d.weight": torch.zeros(8, 4),
        }

        corrected = self.loader.qwen35_corrections(state_dict)

        self.assertTrue(torch.allclose(corrected["model.language_model.layers.0.input_layernorm.weight"], torch.tensor([0.25])))
        self.assertTrue(torch.equal(corrected["model.language_model.layers.0.linear_attn.norm.weight"], torch.tensor([0.5])))
        self.assertTrue(torch.allclose(corrected["model.language_model.layers.0.linear_attn.A_log"], torch.tensor([2.0])))
        self.assertEqual(
            corrected["model.language_model.layers.0.linear_attn.conv1d.weight"].shape,
            (8, 1, 4),
        )

    def test_installs_quantized_qwen35_logits_path(self):
        from types import SimpleNamespace
        import comfy.text_encoders.qwen35
        from comfyui_gguf_test.loader.ops import GGMLOps

        self.assertTrue(comfy.text_encoders.qwen35.Qwen35._comfyui_gguf_logits)
        source = np.arange(128, dtype=np.float32).reshape(4, 32)
        packed = gguf.quants.quantize(source, gguf.GGMLQuantizationType.Q8_0)
        layer = GGMLOps.Linear(32, 4, bias=False)
        layer.load_state_dict(
            {"weight": self.loader.GGMLTensor(
                torch.from_numpy(packed),
                tensor_type=gguf.GGMLQuantizationType.Q8_0,
                tensor_shape=torch.Size(source.shape),
            )},
            strict=False,
        )
        model = SimpleNamespace(model=SimpleNamespace(lm_head=layer))
        model._gguf_logits_weight_cache = {}
        original_cast_bias_weight = layer.cast_bias_weight
        cast_calls = 0

        def counted_cast_bias_weight(input_tensor):
            nonlocal cast_calls
            cast_calls += 1
            return original_cast_bias_weight(input_tensor)

        layer.cast_bias_weight = counted_cast_bias_weight

        logits_first = comfy.text_encoders.qwen35.Qwen35.logits(
            model, torch.ones(1, 2, 32, dtype=torch.float16)
        )
        logits_second = comfy.text_encoders.qwen35.Qwen35.logits(
            model, torch.ones(1, 2, 32, dtype=torch.float16)
        )

        self.assertEqual(logits_first.shape, (1, 1, 4))
        self.assertEqual(logits_second.shape, (1, 1, 4))
        self.assertEqual(cast_calls, 1)
        self.assertEqual(len(model._gguf_logits_weight_cache), 1)
        self.assertIsInstance(layer.weight, self.loader.GGMLTensor)

    def test_gguf_logits_support_covers_llama_family_embed_tie(self):
        from types import SimpleNamespace
        import comfy.text_encoders.llama
        from comfyui_gguf_test.loader.ops import GGMLOps

        self.assertTrue(comfy.text_encoders.llama.BaseGenerate._comfyui_gguf_logits)
        source = np.arange(128, dtype=np.float32).reshape(4, 32)
        packed = gguf.quants.quantize(source, gguf.GGMLQuantizationType.Q8_0)
        embedding = GGMLOps.Embedding(4, 32)
        embedding.load_state_dict(
            {"weight": self.loader.GGMLTensor(
                torch.from_numpy(packed),
                tensor_type=gguf.GGMLQuantizationType.Q8_0,
                tensor_shape=torch.Size(source.shape),
            )},
            strict=False,
        )
        model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embedding))
        model._gguf_logits_weight_cache = {}
        x = torch.ones(1, 2, 32, dtype=torch.float16)
        reference = torch.nn.functional.linear(
            x[:, -1:].float(),
            self.loader.dequantize_tensor(embedding.weight, torch.float32),
            None,
        )

        for logits_fn in (comfy.text_encoders.llama.BaseGenerate.logits,
                          comfy.text_encoders.llama.BaseQwen3.logits):
            logits = logits_fn(model, x)
            self.assertEqual(tuple(logits.shape), (1, 1, 4))
            self.assertTrue(
                torch.allclose(logits.float(), reference, atol=5e-2, rtol=1e-2)
            )
        self.assertEqual(len(model._gguf_logits_weight_cache), 1)

    def test_materializes_quantized_decode_gate_weights(self):
        source = np.arange(128, dtype=np.float32).reshape(4, 32)
        packed = gguf.quants.quantize(source, gguf.GGMLQuantizationType.Q8_0)
        state_dict = {}
        for gate in ("in_proj_a", "in_proj_b"):
            state_dict[f"model.language_model.layers.0.linear_attn.{gate}.weight"] = (
                self.loader.GGMLTensor(
                    torch.from_numpy(packed.copy()),
                    tensor_type=gguf.GGMLQuantizationType.Q8_0,
                    tensor_shape=torch.Size(source.shape),
                )
            )

        corrected = self.loader.qwen35_corrections(state_dict)

        for value in corrected.values():
            self.assertNotIsInstance(value, self.loader.GGMLTensor)
            self.assertEqual(value.shape, source.shape)


class MinimaxH3DetectionTests(unittest.TestCase):
    def test_detects_native_minimax_h3_checkpoint_layout(self):
        checkpoint_keys = {
            "video_patch_proj.weight",
            "audio_patch_proj.weight",
            "blocks.0.attn.qkv_proj.weight",
            "final_layer.video_out.weight",
        }

        model_arch = detect_arch(checkpoint_keys)

        self.assertIsInstance(model_arch, ModelMinimaxH3)
        self.assertEqual(model_arch.arch, "minimax_h3")

    def test_keeps_adaln_curve_table_in_full_precision(self):
        model_arch = ModelMinimaxH3()

        self.assertIn("adaln_t_table", model_arch.keys_hiprec)

    def test_converts_to_minimax_h3_gguf_with_full_precision_adaln_table(self):
        state_dict = {
            "video_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "audio_patch_proj.weight": torch.ones((32, 32), dtype=torch.float32),
            "blocks.0.attn.qkv_proj.weight": torch.ones((96, 32), dtype=torch.float32),
            "final_layer.video_out.weight": torch.ones((96, 32), dtype=torch.float32),
            "adaln_t_table": torch.ones((32, 32), dtype=torch.float32),
        }

        with TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "minimax_h3.safetensors"
            output_path = Path(temp_dir) / "minimax_h3-Q8_0.gguf"
            save_file(state_dict, str(source_path))

            converted_path, model_arch = convert_file(
                str(source_path),
                str(output_path),
                interact=False,
                quant_type_name="Q8_0",
            )

            reader = gguf.GGUFReader(converted_path)
            tensor_types = {tensor.name: tensor.tensor_type for tensor in reader.tensors}
            architecture = reader.get_field("general.architecture")
            architecture_name = str(architecture.parts[architecture.data[-1]], "utf-8")
            del architecture
            reader.tensors.clear()
            reader.fields.clear()
            reader.data._mmap.close()
            del reader

        self.assertEqual(model_arch.arch, "minimax_h3")
        self.assertEqual(architecture_name, "minimax_h3")
        self.assertEqual(
            tensor_types["blocks.0.attn.qkv_proj.weight"],
            gguf.GGMLQuantizationType.Q8_0,
        )
        self.assertEqual(
            tensor_types["adaln_t_table"],
            gguf.GGMLQuantizationType.F32,
        )


if __name__ == "__main__":
    unittest.main()
