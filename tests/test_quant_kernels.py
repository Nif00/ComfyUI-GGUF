# (c) City96 || Apache-2.0 (apache.org/licenses/LICENSE-2.0)
"""Numerical regression tests for the fused GGML quantization kernels."""
import importlib.util
import unittest
from pathlib import Path

import gguf
import numpy as np
import torch


# Byte offsets of IEEE fp16 fields (scales/minima) inside each block format.
# Random bytes there could decode to NaN/Inf, so synthetic tests pin them.
F16_FIELD_OFFSETS = {
    gguf.GGMLQuantizationType.Q2_K: (80, 82),
    gguf.GGMLQuantizationType.Q3_K: (108,),
    gguf.GGMLQuantizationType.Q4_K: (0, 2),
    gguf.GGMLQuantizationType.Q5_K: (0, 2),
    gguf.GGMLQuantizationType.Q6_K: (208,),
    gguf.GGMLQuantizationType.Q4_0: (0,),
    gguf.GGMLQuantizationType.Q4_1: (0, 2),
    gguf.GGMLQuantizationType.Q5_0: (0,),
    gguf.GGMLQuantizationType.Q5_1: (0, 2),
    gguf.GGMLQuantizationType.Q8_0: (0,),
    gguf.GGMLQuantizationType.IQ4_NL: (0,),
    gguf.GGMLQuantizationType.IQ4_XS: (0,),
}


def synthetic_blocks(qtype, n_blocks, seed):
    """Random valid packed blocks with finite fp16 scale fields."""
    rng = np.random.default_rng(seed)
    _, type_size = gguf.GGML_QUANT_SIZES[qtype]
    blocks = rng.integers(0, 256, size=(n_blocks, type_size)).astype(np.uint8)
    for offset in F16_FIELD_OFFSETS[qtype]:
        blocks[:, offset:offset + 2] = np.array([0x00, 0x1C], dtype=np.uint8)  # f16 2**-10
    return blocks


class QuantKernelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys

        repo = Path(__file__).parents[1]
        package_name = "comfyui_gguf_kernels_test"
        if package_name in sys.modules:
            module = sys.modules[package_name]
        else:
            # ops.py is loaded as the package root so dequant/quant_matmul come
            # along without loader.py's import-time monkeypatching.
            spec = importlib.util.spec_from_file_location(
                f"{package_name}.ops",
                repo / "ops.py",
                submodule_search_locations=[str(repo)],
            )
            module = importlib.util.module_from_spec(spec)
            sys.modules[package_name] = module
            sys.modules[f"{package_name}.ops"] = module
            spec.loader.exec_module(module)
        cls.ops = module
        cls.dequant = sys.modules[f"{package_name}.ops.dequant"]
        cls.quant_matmul = sys.modules[f"{package_name}.ops.quant_matmul"]
        # exercise the dp4a kernel outside graph capture too
        cls.quant_matmul._DP4A_EAGER_MIN_BYTES = 0

    def test_dequantize_matches_gguf_reference(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for qtype in sorted(self.dequant.CUDA_QTYPES, key=int):
                block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
                blocks_np = synthetic_blocks(qtype, 32, seed=int(qtype))
                reference = torch.from_numpy(
                    gguf.quants.dequantize(blocks_np, qtype).astype(np.float32)
                ).reshape(32, block_size)
                got = self.dequant.dequantize(
                    torch.from_numpy(blocks_np).to(device),
                    qtype, torch.Size((32, block_size)), dtype=torch.float32,
                )
                tolerance = 1e-4 * reference.abs().max().item() + 1e-5
                self.assertLessEqual(
                    (got.cpu() - reference).abs().max().item(), tolerance,
                    f"{qtype} on {device}",
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_k_quant_matmul_matches_dense_reference(self):
        # in_features 768 has an odd superblock count, which Q3_K/Q6_K leave to
        # the generic kernel. Single rows quantize the activation to int8.
        for qtype in [gguf.GGMLQuantizationType.Q3_K, gguf.GGMLQuantizationType.Q4_K,
                      gguf.GGMLQuantizationType.Q5_K, gguf.GGMLQuantizationType.Q6_K]:
            block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
            for out_features, in_features in [(128, 512), (72, 768)]:
                blocks = synthetic_blocks(qtype, out_features * in_features // block_size, seed=7)
                packed = torch.from_numpy(blocks).to("cuda")
                shape = (out_features, in_features)
                dense = self.dequant.dequantize(packed, qtype, torch.Size(shape), dtype=torch.float32)
                for rows, with_bias, dtype in [(1, False, torch.float16), (1, True, torch.bfloat16), (3, True, torch.float16),
                                               (8, False, torch.float16), (16, True, torch.float16)]:
                    x = torch.randn(rows, in_features, dtype=dtype, device="cuda")
                    bias = torch.randn(out_features, dtype=dtype, device="cuda") if with_bias else None
                    self.assertTrue(self.quant_matmul.can_use_k_quant_matmul(qtype, shape, x))
                    got = self.quant_matmul.k_quant_matmul(x, packed, qtype, shape, bias)
                    reference = torch.nn.functional.linear(
                        x.float(), dense, bias.float() if bias is not None else None,
                    )
                    tolerance = (2e-2 if rows == 1 else 2e-3) * reference.abs().max().item()
                    self.assertLessEqual(
                        (got.float() - reference).abs().max().item(), tolerance,
                        f"{qtype} {shape} rows={rows} bias={with_bias} {dtype}",
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_k_quant_matmul_applies_qwen35_v_head_maps(self):
        num_k, num_v_per_k, head_dim = 4, 2, 64
        value_size = num_k * num_v_per_k * head_dim
        cases = [
            # (transform dim, start, weight shape)
            (0, 128, (128 + value_size, 512)),  # row map, qk region in front
            (1, 0, (128, value_size)),          # col map over the full input
        ]
        for qtype in [gguf.GGMLQuantizationType.Q3_K, gguf.GGMLQuantizationType.Q5_K]:
            block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
            for dim, start, shape in cases:
                transform = ("qwen35_inverse_v_heads", dim, start, num_k, num_v_per_k, head_dim)
                out_features, in_features = shape
                blocks = synthetic_blocks(qtype, out_features * in_features // block_size, seed=11)
                packed = torch.from_numpy(blocks).to("cuda")
                dense = self.dequant.apply_tensor_postprocess(
                    self.dequant.dequantize(packed, qtype, torch.Size(shape), dtype=torch.float32),
                    (transform,),
                )
                for rows in (1, 3):
                    x = torch.randn(rows, in_features, dtype=torch.float16, device="cuda")
                    got = self.quant_matmul.k_quant_matmul(x, packed, qtype, shape, postprocess=(transform,))
                    reference = torch.nn.functional.linear(x.float(), dense)
                    tolerance = (2e-2 if rows == 1 else 2e-3) * reference.abs().max().item()
                    self.assertLessEqual(
                        (got.float() - reference).abs().max().item(), tolerance, f"{qtype} dim={dim} rows={rows}",
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_k_quant_matmul_replays_in_cuda_graph(self):
        qtype = gguf.GGMLQuantizationType.Q3_K
        shape = (64, 512)
        packed = torch.from_numpy(synthetic_blocks(qtype, shape[0] * shape[1] // 256, seed=23)).to("cuda")
        x = torch.randn(1, shape[1], dtype=torch.bfloat16, device="cuda")
        expected = self.quant_matmul.k_quant_matmul(x, packed, qtype, shape)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            got = self.quant_matmul.k_quant_matmul(x, packed, qtype, shape)
        x.copy_(torch.randn_like(x))
        expected = self.quant_matmul.k_quant_matmul(x, packed, qtype, shape)
        graph.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(got, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_quantized_tensor_dispatch_stays_packed(self):
        # DynamicVRAM hands GGML weights over as QuantizedTensor; linear (incl.
        # the expand/bmm route a sliced activation takes) and embedding must
        # not fall back to dequantizing the whole weight.
        import comfy_kitchen.tensor.base as kitchen
        quant_ops = importlib.import_module(self.ops.__name__ + ".quant_ops")
        qtype = gguf.GGMLQuantizationType.Q4_K
        shape = (96, 512)
        blocks = synthetic_blocks(qtype, shape[0] * shape[1] // 256, seed=17)
        packed = torch.from_numpy(blocks).to("cuda")
        weight = quant_ops.make_quantized(packed.reshape(-1), qtype, shape, orig_dtype=torch.bfloat16)
        dense = self.dequant.dequantize(packed, qtype, torch.Size(shape), dtype=torch.float32)

        fallbacks = []
        original = kitchen.QuantizedTensor._dequant_and_fallback.__func__
        kitchen.QuantizedTensor._dequant_and_fallback = classmethod(
            lambda cls, func, args, kwargs: fallbacks.append(func) or original(cls, func, args, kwargs)
        )
        try:
            x = torch.randn(1, 5, shape[1], dtype=torch.bfloat16, device="cuda")
            for name, inp in [("sliced row", x[:, -1:]), ("rows", x), ("2d", x[0])]:
                got = torch.nn.functional.linear(inp, weight)
                reference = torch.nn.functional.linear(inp.float(), dense)
                self.assertEqual(got.shape, reference.shape, name)
                self.assertLessEqual((got.float() - reference).abs().max().item(),
                                     2e-2 * reference.abs().max().item(), name)
            ids = torch.tensor([[3, 95, 0]], device="cuda")
            got = torch.nn.functional.embedding(ids, weight)
            reference = torch.nn.functional.embedding(ids, dense)
            self.assertLessEqual((got.float() - reference).abs().max().item(), 1e-2 * reference.abs().max().item())
        finally:
            kitchen.QuantizedTensor._dequant_and_fallback = classmethod(original)
        self.assertEqual(fallbacks, [])

    def test_embedding_row_select_matches_full_table(self):
        for device in ["cpu"] + (["cuda"] if torch.cuda.is_available() else []):
            for qtype in [gguf.GGMLQuantizationType.Q8_0, gguf.GGMLQuantizationType.Q4_0]:
                vocab, columns = 64, 32
                source = np.arange(vocab * columns, dtype=np.float32).reshape(vocab, columns) / 8.0
                packed = np.asarray(gguf.quants.quantize(source, qtype))
                layer = self.ops.GGMLOps.Embedding(vocab, columns)
                layer.load_state_dict(
                    {"weight": self.ops.GGMLTensor(
                        torch.from_numpy(packed),
                        tensor_type=qtype,
                        tensor_shape=torch.Size((vocab, columns)),
                    )},
                    strict=False,
                )
                ids = torch.tensor([[3, 3, 17], [63, 0, 8]], dtype=torch.int64, device=device)
                got = layer(ids)
                table = self.dequant.dequantize(
                    torch.from_numpy(packed).to(device), qtype,
                    torch.Size((vocab, columns)), dtype=got.dtype,
                )
                reference = torch.nn.functional.embedding(ids, table)
                self.assertLessEqual(
                    (got.cpu().float() - reference.cpu().float()).abs().max().item(), 1e-3,
                    f"{qtype} on {device}",
                )


    def packed_weight(self, qtype, shape, seed, device="cpu"):
        block_size, _ = gguf.GGML_QUANT_SIZES[qtype]
        blocks = synthetic_blocks(qtype, shape[0] * shape[1] // block_size, seed=seed)
        packed = torch.from_numpy(blocks)
        return self.ops.GGMLTensor(
            packed.to(device) if device != "cpu" else packed,
            tensor_type=qtype,
            tensor_shape=torch.Size(shape),
        )

    def test_dtype_cast_keeps_packed_storage(self):
        # MTP verify re-casts the output projection to the activation dtype on
        # every step; the packed payload must survive that without allocating
        # the dense weight (2 GiB for a 9B lm_head).
        qtype = gguf.GGMLQuantizationType.Q4_K
        shape = (64, 512)
        weight = self.packed_weight(qtype, shape, seed=3)
        for target in (weight.dtype, torch.bfloat16, torch.float32):
            cast = weight.to(target)
            self.assertEqual(cast.dtype, target, f"advertised compute dtype for {target}")
            self.assertEqual(torch.Tensor(cast).dtype, torch.uint8, f"storage dtype for {target}")
            self.assertEqual(torch.Tensor(cast).numel(), torch.Tensor(weight).numel())
            self.assertEqual(tuple(cast.tensor_shape), shape)
            self.assertEqual(cast.tensor_type, qtype)
        if torch.cuda.is_available():
            moved = weight.to("cuda", torch.bfloat16)
            self.assertEqual(moved.device.type, "cuda")
            self.assertEqual(moved.dtype, torch.bfloat16)
            self.assertEqual(torch.Tensor(moved).numel(), torch.Tensor(weight).numel())

    def test_dtype_cast_materializes_patched_weights(self):
        # Patched weights keep the dense path: GGUFModelPatcher applies LoRA
        # weight functions to real values, never to packed blocks.
        qtype = gguf.GGMLQuantizationType.Q4_K
        shape = (64, 512)
        weight = self.packed_weight(qtype, shape, seed=5)
        diff = torch.randn(*shape)
        weight.patches = [([(1.0, (diff,), 1.0, None, None)], "probe.weight")]
        cast = weight.to(torch.float32)
        reference = self.dequant.dequantize(
            weight.data, qtype, torch.Size(shape), dtype=torch.float32,
        ) + diff
        self.assertEqual(cast.shape, reference.shape)
        self.assertLessEqual((cast - reference).abs().max().item(), 1e-4)

    def test_linear_dispatch_on_packed_weight(self):
        # Packed weights reach F.linear from comfy's cast paths; the dispatch
        # must compute from the packed storage with logical shapes.
        qtype = gguf.GGMLQuantizationType.Q6_K
        shape = (64, 512)
        weight = self.packed_weight(qtype, shape, seed=9)
        dense = self.dequant.dequantize(
            weight.data, qtype, torch.Size(shape), dtype=torch.float32,
        )
        for rows in (1, 12):
            x = torch.randn(rows, shape[1])
            got = torch.nn.functional.linear(x, weight)
            reference = torch.nn.functional.linear(x, dense)
            tolerance = 2e-3 * reference.abs().max().item()
            self.assertLessEqual(
                (got - reference).abs().max().item(), tolerance, f"rows={rows}",
            )

    def test_embedding_dispatch_on_packed_weight(self):
        qtype = gguf.GGMLQuantizationType.Q8_0
        shape = (64, 32)
        table = self.packed_weight(qtype, shape, seed=13)
        ids = torch.tensor([[1, 5], [63, 0]], dtype=torch.int64)
        got = torch.nn.functional.embedding(ids, table)
        reference = torch.nn.functional.embedding(
            ids, self.dequant.dequantize(table.data, qtype, torch.Size(shape), dtype=got.dtype),
        )
        self.assertLessEqual((got - reference).abs().max().item(), 1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_linear_dispatch_uses_fused_kernel_for_small_batches(self):
        qtype = gguf.GGMLQuantizationType.Q4_K
        shape = (128, 512)
        weight = self.packed_weight(qtype, shape, seed=21, device="cuda")
        dense = self.dequant.dequantize(
            weight.data, qtype, torch.Size(shape), dtype=torch.float32,
        )
        for rows in (1, 4, 12):
            x = torch.randn(rows, shape[1], dtype=torch.float16, device="cuda")
            got = torch.nn.functional.linear(x, weight)
            reference = torch.nn.functional.linear(x.float(), dense)
            tolerance = (2e-2 if rows == 1 else 2e-3) * reference.abs().max().item() + 1e-3
            self.assertLessEqual(
                (got.float() - reference).abs().max().item(), tolerance, f"rows={rows}",
            )


if __name__ == "__main__":
    unittest.main()
