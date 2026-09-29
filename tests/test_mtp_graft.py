# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Tests for MTP draft-head extraction, naming, and loading."""
import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import gguf
import numpy as np
import torch


PACKAGE_NAME = "comfyui_gguf_mtp_test"

LLAMA_STYLE_KEYS = {
    "nextn.eh_proj.weight": "fc.weight",
    "nextn.enorm.weight": "pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight": "pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "norm.weight",
    "attn_q.weight": "layers.0.self_attn.q_proj.weight",
    "attn_k.weight": "layers.0.self_attn.k_proj.weight",
    "attn_v.weight": "layers.0.self_attn.v_proj.weight",
    "attn_output.weight": "layers.0.self_attn.o_proj.weight",
    "attn_q_norm.weight": "layers.0.self_attn.q_norm.weight",
    "attn_k_norm.weight": "layers.0.self_attn.k_norm.weight",
    "attn_norm.weight": "layers.0.input_layernorm.weight",
    "post_attention_norm.weight": "layers.0.post_attention_layernorm.weight",
    "ffn_gate.weight": "layers.0.mlp.gate_proj.weight",
    "ffn_up.weight": "layers.0.mlp.up_proj.weight",
    "ffn_down.weight": "layers.0.mlp.down_proj.weight",
}


def to_llama_style(canonical):
    for llama_name, canonical_tail in LLAMA_STYLE_KEYS.items():
        if canonical == "mtp." + canonical_tail:
            return llama_name
    raise KeyError(canonical)


class MtpGraftTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        repo = Path(__file__).parents[1]
        if PACKAGE_NAME in sys.modules:
            module = sys.modules[PACKAGE_NAME]
        else:
            spec = importlib.util.spec_from_file_location(
                f"{PACKAGE_NAME}.loader",
                repo / "loader.py",
                submodule_search_locations=[str(repo)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[PACKAGE_NAME] = module
            sys.modules[f"{PACKAGE_NAME}.loader"] = module
            spec.loader.exec_module(module)
        cls.loader = module

    def test_normalize_mtp_key_variants(self):
        cases = {
            "blk.32.nextn.eh_proj.weight": "mtp.fc.weight",
            "blk.32.nextn.enorm.weight": "mtp.pre_fc_norm_embedding.weight",
            "blk.32.nextn.hnorm.weight": "mtp.pre_fc_norm_hidden.weight",
            "blk.32.nextn.shared_head_norm.weight": "mtp.norm.weight",
            "blk.32.attn_q.weight": "mtp.layers.0.self_attn.q_proj.weight",
            "blk.32.attn_norm.weight": "mtp.layers.0.input_layernorm.weight",
            "blk.32.post_attention_norm.weight": "mtp.layers.0.post_attention_layernorm.weight",
            "blk.32.ffn_gate.weight": "mtp.layers.0.mlp.gate_proj.weight",
            "mtp.layers.0.attn_q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
            "model.mtp.fc.weight": "mtp.fc.weight",
            "layers.0.mlp.down_proj.weight": "mtp.layers.0.mlp.down_proj.weight",
            # canonical names must pass through untouched
            "mtp.fc.weight": "mtp.fc.weight",
            "mtp.layers.0.self_attn.q_norm.weight": "mtp.layers.0.self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.q_proj.weight": "mtp.layers.0.self_attn.q_proj.weight",
        }
        for source, expected in cases.items():
            got = self.loader.normalize_mtp_keys({source: torch.zeros(1)})
            self.assertEqual(list(got), [expected], source)

    def test_split_mtp_tensors_extracts_nextn_block(self):
        values = {
            "blk.0.attn_qkv.weight": torch.zeros(2, 2),
            "blk.32.attn_q.weight": torch.ones(2, 2),
            "blk.32.nextn.eh_proj.weight": torch.full((2, 2), 2.0),
            "token_embd.weight": torch.zeros(2, 2),
        }
        mtp_sd, rest_sd = self.loader.split_mtp_tensors(values)
        self.assertEqual(
            sorted(mtp_sd),
            ["mtp.fc.weight", "mtp.layers.0.self_attn.q_proj.weight"],
        )
        self.assertEqual(sorted(rest_sd), ["blk.0.attn_qkv.weight", "token_embd.weight"])
        self.assertIs(mtp_sd["mtp.fc.weight"], values["blk.32.nextn.eh_proj.weight"])

    def _small_mtp_head(self):
        import comfy.ops
        import comfy.text_encoders.qwen35 as qwen35

        config = qwen35.Qwen35Config(
            vocab_size=512,
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=128,
            linear_num_key_heads=2,
            linear_num_value_heads=2,
            linear_key_head_dim=64,
            linear_value_head_dim=64,
            layer_types=qwen35._qwen35_layer_types(4),
        )
        return qwen35.MTPHead(config, device="cpu", dtype=torch.float32, ops=comfy.ops.manual_cast)

    def _extract_tool(self):
        cls = type(self)
        tool = getattr(cls, "_extract_tool_mod", None)
        if tool is None:
            spec = importlib.util.spec_from_file_location(
                "comfyui_gguf_extract_tool",
                Path(__file__).resolve().parents[1] / "tools" / "extract_mtp.py",
            )
            tool = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(tool)
            cls._extract_tool_mod = tool
        return tool

    def test_extract_roundtrip_loads_into_comfy_mtp_head(self):
        from safetensors.torch import save_file

        head = self._small_mtp_head()
        source = {}
        for name, tensor in head.state_dict().items():
            value = torch.randn(tensor.shape) * 0.02
            source["mtp." + name] = value

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            tmp = Path(tmp)
            # HF-style safetensors source
            st_path = tmp / "mtp-src.safetensors"
            save_file({k: v.contiguous() for k, v in source.items()}, str(st_path))
            # llama.cpp-style GGUF source (blk.N / nextn naming)
            gguf_path = tmp / "mtp-src.gguf"
            writer = gguf.GGUFWriter(str(gguf_path), arch="qwen35")
            writer.add_name("mtp-src")
            for name, value in source.items():
                writer.add_tensor("blk.4." + to_llama_style(name), value.numpy().astype(np.float16))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()

            for src in (st_path, gguf_path):
                if src.suffix == ".gguf":
                    sd = self.loader.gguf_mtp_loader(str(src))
                else:
                    sd = self.loader.normalize_mtp_keys(dict(source))
                self.assertEqual(sorted(sd), sorted(source), src.name)

                target = self._small_mtp_head()
                target.load_state_dict(
                    {k.removeprefix("mtp."): v.to(torch.float32) for k, v in sd.items()},
                    strict=True,
                )
                for name in head.state_dict():
                    self.assertTrue(
                        torch.allclose(
                            target.state_dict()[name],
                            source["mtp." + name].to(torch.float32),
                            atol=2e-3,
                        ),
                        f"{src.name}:{name}",
                    )

    def test_packed_source_requantizes_with_logical_shapes(self):
        tool = self._extract_tool()
        value = torch.randn(256, 256)
        packed_array = gguf.quants.quantize(value.numpy(), gguf.GGMLQuantizationType.Q8_0)
        packed = self.loader.GGMLTensor(
            torch.from_numpy(np.asarray(packed_array)),
            tensor_type=gguf.GGMLQuantizationType.Q8_0,
            tensor_shape=torch.Size((256, 256)),
        )
        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            out = Path(tmp) / "mtp-out.gguf"
            tool.write_mtp_gguf({"mtp.fc.weight": packed}, out, gguf.GGMLQuantizationType.Q8_0)
            sd = self.loader.gguf_mtp_loader(str(out))
            reloaded = sd["mtp.fc.weight"]
            # logical shape must survive packed sources; values must reflect a
            # single re-quantization (a double pass eats block bytes as weights)
            self.assertEqual(tuple(reloaded.tensor_shape), (256, 256))
            restored = self.loader.dequantize_tensor(reloaded, torch.float32)
            self.assertTrue(torch.allclose(restored, value, atol=0.05))

    def test_gguf_mtp_loader_rejects_file_without_mtp(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "no-mtp.gguf"
            writer = gguf.GGUFWriter(str(path), arch="qwen35")
            writer.add_name("no-mtp")
            writer.add_tensor("blk.0.attn_q.weight", np.zeros((4, 4), dtype=np.float16))
            writer.write_header_to_file()
            writer.write_kv_data_to_file()
            writer.write_tensors_to_file()
            writer.close()
            with self.assertRaises(ValueError):
                self.loader.gguf_mtp_loader(str(path))


if __name__ == "__main__":
    unittest.main()
