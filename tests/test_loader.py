import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).parents[1]
COMFY_ROOT = REPO_ROOT.parents[1]
if str(COMFY_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFY_ROOT))


def load_gguf_loader():
    package_name = "comfyui_gguf_loader_test"
    loader_path = REPO_ROOT / "loader.py"
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.loader",
        loader_path,
        submodule_search_locations=[str(REPO_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    sys.modules[f"{package_name}.loader"] = module
    spec.loader.exec_module(module)
    return module


class Qwen3VLLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    @staticmethod
    def text_state_dict(hidden_size=5120):
        return {
            "model.layers.0.input_layernorm.weight": torch.zeros(hidden_size),
            "model.layers.49.self_attn.q_proj.weight": torch.zeros(1),
        }

    def test_iq2_xxs_without_mmproj_does_not_read_unbound_flag(self):
        self.loader.gguf_sd_loader = lambda *args, **kwargs: (
            self.text_state_dict(),
            {"arch_str": "qwen3vl"},
        )
        self.loader.gguf_mmproj_loader = lambda *args, **kwargs: {}

        state_dict = self.loader.gguf_clip_loader(
            "Qwen3-VL-32B-Instruct-IQ2_XXS.gguf"
        )

        self.assertIn("visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertEqual(
            state_dict["visual.deepstack_merger_list.0.norm.weight"].shape,
            (4608,),
        )

    def test_minimax_mmproj_is_remapped_to_unprefixed_visual_keys(self):
        self.loader.gguf_sd_loader = lambda *args, **kwargs: (
            self.text_state_dict(),
            {"arch_str": "qwen3vl"},
        )
        self.loader.gguf_mmproj_loader = lambda *args, **kwargs: {
            "model.visual.deepstack_merger_list.0.norm.weight": torch.zeros(4608),
        }

        state_dict = self.loader.gguf_clip_loader(
            "Qwen3-VL-32B-Instruct-IQ2_XXS.gguf"
        )

        self.assertIn("visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertNotIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)

    def test_standard_qwen3vl_fallback_preserves_detector_shapes(self):
        self.loader.gguf_sd_loader = lambda *args, **kwargs: (
            {
                "model.layers.0.input_layernorm.weight": torch.zeros(4096),
            },
            {"arch_str": "qwen3vl"},
        )
        self.loader.gguf_mmproj_loader = lambda *args, **kwargs: {}

        state_dict = self.loader.gguf_clip_loader(
            "Qwen3-VL-8B-Instruct-IQ2_XXS.gguf"
        )

        self.assertIn("model.visual.deepstack_merger_list.0.norm.weight", state_dict)
        self.assertEqual(
            state_dict["model.visual.merger.linear_fc2.weight"].shape,
            (4096, 4608),
        )


class PythonIntSliceArray(np.ndarray):
    def __getitem__(self, index):
        if isinstance(index, slice):
            bounds = (index.start, index.stop, index.step)
            if any(bound is not None and type(bound) is not int for bound in bounds):
                raise TypeError(
                    "slice indices must be integers or None or have an __index__ method"
                )
        return super().__getitem__(index)


class LazyGGUFReaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = load_gguf_loader()

    @staticmethod
    def make_reader(array_len, sub_type, payload):
        data = np.zeros(12 + len(payload), dtype=np.uint8).view(PythonIntSliceArray)
        data[12:] = np.frombuffer(payload, dtype=np.uint8)

        reader = object.__new__(LazyGGUFReaderTests.loader.LazyGGUFReader)
        reader.data = data
        reader.byte_order = "I"
        reader.gguf_scalar_to_np = LazyGGUFReaderTests.loader.GGUFReader.gguf_scalar_to_np

        def fake_get(offset, dtype, count=1, override_order=None):
            if np.dtype(dtype) == np.dtype(np.uint32):
                return np.array([sub_type], dtype=np.uint32)
            if np.dtype(dtype) == np.dtype(np.uint64):
                return np.array([array_len], dtype=np.uint64)
            raise AssertionError(f"Unexpected dtype: {dtype}")

        reader._get = fake_get
        return reader

    def test_large_scalar_array_uses_native_slice_bounds(self):
        array_len = 1001
        values = np.arange(array_len, dtype=np.int32)
        reader = self.make_reader(
            array_len,
            int(self.loader.GGUFValueType.INT32),
            values.tobytes(),
        )

        size, parts, data_idxs, types = reader._get_field_parts(
            np.uint64(0), self.loader.GGUFValueType.ARRAY
        )

        self.assertIs(type(size), int)
        self.assertEqual(size, 12 + values.nbytes)
        self.assertEqual(len(data_idxs), array_len)
        self.assertEqual(parts[data_idxs[0]][0], 0)
        self.assertEqual(parts[data_idxs[-1]][0], array_len - 1)
        self.assertEqual(
            types,
            [self.loader.GGUFValueType.ARRAY, self.loader.GGUFValueType.INT32],
        )

    def test_large_string_array_uses_native_slice_bounds(self):
        array_len = 1001
        payload = b"".join(
            (1).to_bytes(8, "little") + b"x" for _ in range(array_len)
        )
        reader = self.make_reader(
            array_len,
            int(self.loader.GGUFValueType.STRING),
            payload,
        )

        size, parts, data_idxs, types = reader._get_field_parts(
            np.uint64(0), self.loader.GGUFValueType.ARRAY
        )

        self.assertIs(type(size), int)
        self.assertEqual(size, 12 + len(payload))
        self.assertEqual(len(data_idxs), array_len)
        self.assertEqual(bytes(parts[data_idxs[0]]), b"x")
        self.assertEqual(bytes(parts[data_idxs[-1]]), b"x")
        self.assertEqual(
            types,
            [self.loader.GGUFValueType.ARRAY, self.loader.GGUFValueType.STRING],
        )


if __name__ == "__main__":
    unittest.main()
