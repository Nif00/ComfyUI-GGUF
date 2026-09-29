"""Microbenchmark the GGUF linear paths used by MTP and ordinary inference."""
import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import gguf
import numpy as np
import torch


def load_ops():
    root = Path(__file__).resolve().parents[1]
    name = "comfyui_gguf_profile"
    spec = importlib.util.spec_from_file_location(
        name + ".ops", root / "ops.py", submodule_search_locations=[str(root)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    sys.modules[name + ".ops"] = module
    spec.loader.exec_module(module)
    return module


def measure(fn, rounds):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    events = []
    hosts = []
    for _ in range(rounds):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        host_start = time.perf_counter()
        fn()
        stop.record()
        stop.synchronize()
        hosts.append((time.perf_counter() - host_start) * 1000)
        events.append(start.elapsed_time(stop))
    return statistics.median(hosts), statistics.median(events)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=30)
    args = parser.parse_args()
    ops = load_ops()
    qmatmul = sys.modules[ops.__package__ + ".quant_matmul"]
    torch.manual_seed(123)
    print(f"torch={torch.__version__} gpu={torch.cuda.get_device_name()}", flush=True)
    for qtype in (gguf.GGMLQuantizationType.Q4_K, gguf.GGMLQuantizationType.Q6_K,
                  gguf.GGMLQuantizationType.Q4_0, gguf.GGMLQuantizationType.Q8_0):
        block_size, type_size = gguf.GGML_QUANT_SIZES[qtype]
        for rows, out_features, in_features in ((1, 2048, 2048), (4, 2048, 2048), (32, 2048, 2048)):
            nblocks = out_features * in_features // block_size
            rng = np.random.default_rng(123)
            data = rng.integers(0, 256, (nblocks, type_size), dtype=np.uint8)
            data[:, 0:4] = np.array([0, 28, 0, 28], dtype=np.uint8)
            if qtype == gguf.GGMLQuantizationType.Q6_K:
                data[:, 208:210] = np.array([0, 28], dtype=np.uint8)
            weight = torch.from_numpy(data).to("cuda")
            weight.tensor_type = qtype
            weight.tensor_shape = (out_features, in_features)
            weight.patches = []
            x = torch.randn(rows, in_features, device="cuda", dtype=torch.float16)
            if qmatmul.can_use_k_quant_matmul(qtype, weight.tensor_shape, x):
                native = lambda: qmatmul.k_quant_matmul(x, weight, qtype, weight.tensor_shape)
                host, elapsed = measure(native, args.rounds)
                print(f"{qtype.name} rows={rows} native host_ms={host:.3f} event_ms={elapsed:.3f}", flush=True)
            dense = lambda: torch.nn.functional.linear(x, ops.dequantize_weight(weight, x.dtype))
            host, elapsed = measure(dense, args.rounds)
            print(f"{qtype.name} rows={rows} dense host_ms={host:.3f} event_ms={elapsed:.3f}", flush=True)


if __name__ == "__main__":
    main()
